"""Hand edits of a stack: patches painted over the automatic labels, each showing the target or one frame."""
from dataclasses import dataclass, replace

import numpy as np

from nss.focus_stack.blend import composite
from nss.focus_stack.clean import CLEAN_MEAN, suggest_layer
from nss.focus_stack.labels import GROW_SIGMAS, label_weights
from nss.focus_stack.stack import StackResult

PATCH_FEATHER = 0.15  # softness of a painted patch's edge as a share of the plan's feather sigma, so a patch ends at the brush circle


@dataclass(frozen=True, eq=False)
class Patch:
    """A painted region and the label it shows (0 the target, k frame k).

    Never changed in place, so history can share it; two patches are equal only if they are the same object.
    """

    top: int
    left: int
    mask: np.ndarray  # bool, the region cropped to its bounding box whose corner is (top, left)
    label: int

    @property
    def box(self) -> tuple[slice, slice]:
        """The bounding box of the region as (row slice, column slice) of the full picture."""
        return slice(self.top, self.top + self.mask.shape[0]), slice(self.left, self.left + self.mask.shape[1])


def disk(x: float, y: float, radius: float, shape: tuple[int, int]) -> tuple[int, int, np.ndarray]:
    """The disk of `radius` pixels centred on column `x`, row `y`, clipped to a picture of `shape`: its top, left and cropped bool mask."""
    top, bottom = max(0, int(np.floor(y - radius))), min(shape[0], int(np.ceil(y + radius)) + 1)
    left, right = max(0, int(np.floor(x - radius))), min(shape[1], int(np.ceil(x + radius)) + 1)
    rows, cols = np.ogrid[top:bottom, left:right]
    return top, left, (cols - x) ** 2 + (rows - y) ** 2 <= radius * radius


class EditState:
    """The patches a user painted over the automatic labels of a `StackResult`, in working-resolution pixels.

    A patch is painted with `dab`, trimmed with `erase`, and shows one label that `cycle` or `set_label` changes. Newer
    patches are on top where they overlap. Pixels under a patch are `manual`: they may use a frame even where its agreement
    with the target is low, but never where the frame has no pixels. Every change can be undone.
    """

    def __init__(self, plan: StackResult) -> None:
        """Starts with no patches, so the labels are the plan's own."""
        self._plan = plan
        self._patches: tuple[Patch, ...] = ()
        self.active: int | None = None
        self._undo: list[tuple[tuple[Patch, ...], int | None]] = []
        self._redo: list[tuple[tuple[Patch, ...], int | None]] = []
        self._relabeling: int | None = None
        self._dirty: tuple[int, int, int, int] | None = None

    @property
    def target(self) -> np.ndarray:
        """The target at the working size, without any patches."""
        return self._plan.target

    @property
    def frame_count(self) -> int:
        """How many extra frames the patches can show."""
        return len(self._plan.frames)

    @property
    def layer_count(self) -> int:
        """How many layers a patch can show: the target, the frames, and the median when the plan has one."""
        return len(self._plan.frames) + 1 + (self._plan.median is not None)

    def layer_image(self, layer: int) -> np.ndarray:
        """The whole picture of a layer at the working size: the target, a frame as aligned and brightness matched, or the median."""
        if layer == 0:
            return self._plan.target
        if 1 <= layer <= len(self._plan.frames):
            return self._plan.frames[layer - 1].aligned
        if layer == len(self._plan.frames) + 1 and self._plan.median is not None:
            return self._plan.median
        raise IndexError(f"no layer {layer}: there are {self.layer_count}")

    @property
    def patch_count(self) -> int:
        """How many patches there are."""
        return len(self._patches)

    def patch_label(self, index: int) -> int:
        """The label patch `index` shows: 0 for the target, k for frame k."""
        return self._patches[index].label

    def patch_mask(self, index: int) -> np.ndarray:
        """A bool mask of the full picture for patch `index`."""
        patch = self._patches[index]
        full = np.zeros(self._plan.labels.shape, dtype=bool)
        full[patch.box] = patch.mask
        return full

    def active_mask(self) -> np.ndarray | None:
        """The full-picture mask of the active patch, or None when no patch is active."""
        return None if self.active is None else self.patch_mask(self.active)

    def centers(self) -> list[tuple[float, float]]:
        """The (column, row) middle of every patch, where its pin is drawn."""
        out = []
        for patch in self._patches:
            rows, cols = np.nonzero(patch.mask)
            out.append((float(cols.mean() + patch.left), float(rows.mean() + patch.top)))
        return out

    def labels(self) -> np.ndarray:
        """The label of every pixel: the plan's, with each patch's label laid over it in order."""
        return self._layers(*self._everything())[0]

    def manual(self) -> np.ndarray:
        """A bool map of the pixels under any patch."""
        return self._layers(*self._everything())[1]

    def weights(self) -> np.ndarray:
        """Blending weights for the current labels; index 0 is the target and k is frame k."""
        return self._weights(*self._everything())

    def result(self) -> np.ndarray:
        """The composite of the target and the frames for the current patches, at the working size."""
        return self.result_region(*self._everything())

    def result_region(self, rows: slice, cols: slice) -> np.ndarray:
        """The rows and columns of `result()` given by two slices with explicit bounds, computed without rendering the rest.

        The region is padded by the reach of the feathering, so it equals the same crop of the full result.
        """
        reach = self._reach()
        total_rows, total_cols = self._plan.labels.shape
        padded_rows = slice(max(0, rows.start - reach), min(total_rows, rows.stop + reach))
        padded_cols = slice(max(0, cols.start - reach), min(total_cols, cols.stop + reach))
        weights = self._weights(padded_rows, padded_cols)
        layers = [frame.aligned[padded_rows, padded_cols] for frame in self._plan.frames]
        if self._plan.median is not None:
            layers.append(self._plan.median[padded_rows, padded_cols])
        out = composite(self._plan.target[padded_rows, padded_cols], layers, weights)
        return out[rows.start - padded_rows.start:rows.stop - padded_rows.start, cols.start - padded_cols.start:cols.stop - padded_cols.start]

    def take_dirty(self) -> tuple[slice, slice] | None:
        """The (rows, cols) box of the picture that changed since the last call, with the feathering's reach, or None if nothing did."""
        if self._dirty is None:
            return None
        top, bottom, left, right = self._dirty
        self._dirty = None
        reach = self._reach()
        total_rows, total_cols = self._plan.labels.shape
        return slice(max(0, top - reach), min(total_rows, bottom + reach)), slice(max(0, left - reach), min(total_cols, right + reach))

    def begin_stroke(self) -> None:
        """Marks the start of one mouse stroke, so all its dabs or erases undo together."""
        self._remember()

    def end_stroke(self) -> None:
        """Ends a stroke, dropping its undo step when it changed nothing."""
        if self._undo and self._undo[-1] == (self._patches, self.active):
            self._undo.pop()

    def dab(self, x: float, y: float, radius: float) -> None:
        """Adds a disk to the active patch, or starts a patch.

        A new patch starts on the layer `suggest_layer` picks when the plan has a median, otherwise on the label after the one under (`x`, `y`).
        """
        top, left, mask = disk(x, y, radius, self._plan.labels.shape)
        self._mark(top, top + mask.shape[0], left, left + mask.shape[1])
        if self.active is None:
            if self._plan.median is not None:
                region = (slice(top, top + mask.shape[0]), slice(left, left + mask.shape[1]))
                label = suggest_layer(self._plan.frames, self._plan.median, region, mask, CLEAN_MEAN)
            else:
                under = int(self.labels()[min(int(round(y)), self._plan.labels.shape[0] - 1), min(int(round(x)), self._plan.labels.shape[1] - 1)])
                label = (under + 1) % self.layer_count
            self._patches += (Patch(top, left, mask, label),)
            self.active = len(self._patches) - 1
            return
        old = self._patches[self.active]
        new_top, new_left = min(old.top, top), min(old.left, left)
        new_bottom = max(old.top + old.mask.shape[0], top + mask.shape[0])
        new_right = max(old.left + old.mask.shape[1], left + mask.shape[1])
        grown = np.zeros((new_bottom - new_top, new_right - new_left), dtype=bool)
        grown[old.top - new_top:old.top - new_top + old.mask.shape[0], old.left - new_left:old.left - new_left + old.mask.shape[1]] = old.mask
        grown[top - new_top:top - new_top + mask.shape[0], left - new_left:left - new_left + mask.shape[1]] |= mask
        self._replace_active(Patch(new_top, new_left, grown, old.label))

    def erase(self, x: float, y: float, radius: float) -> None:
        """Removes a disk from the active patch; a patch with nothing left is deleted. Does nothing without an active patch."""
        if self.active is None:
            return
        old = self._patches[self.active]
        top, left, mask = disk(x, y, radius, self._plan.labels.shape)
        self._mark(top, top + mask.shape[0], left, left + mask.shape[1])
        kept = old.mask.copy()
        y0, y1 = max(top, old.top), min(top + mask.shape[0], old.top + kept.shape[0])
        x0, x1 = max(left, old.left), min(left + mask.shape[1], old.left + kept.shape[1])
        if y0 < y1 and x0 < x1:
            kept[y0 - old.top:y1 - old.top, x0 - old.left:x1 - old.left] &= ~mask[y0 - top:y1 - top, x0 - left:x1 - left]
        if not kept.any():
            self._delete(self.active)
            return
        rows, cols = np.nonzero(kept)
        trimmed = kept[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
        self._replace_active(Patch(old.top + int(rows.min()), old.left + int(cols.min()), trimmed, old.label))

    def select(self, index: int) -> None:
        """Makes patch `index` the active one."""
        self.active = index
        self._relabeling = None

    def select_at(self, x: float, y: float) -> bool:
        """Activates the topmost patch covering column `x`, row `y`. Returns False, changing nothing, when there is none."""
        column, row = int(round(x)), int(round(y))
        for index in reversed(range(len(self._patches))):
            patch = self._patches[index]
            r, c = row - patch.top, column - patch.left
            if 0 <= r < patch.mask.shape[0] and 0 <= c < patch.mask.shape[1] and patch.mask[r, c]:
                self.select(index)
                return True
        return False

    def deselect(self) -> None:
        """Makes no patch active, so the next dab starts a new one."""
        self.active = None
        self._relabeling = None

    def set_label(self, label: int) -> None:
        """Makes the active patch show `label`. Repeated relabelling of one patch is a single undo step."""
        if self.active is None:
            return
        if self._relabeling != self.active:
            self._remember()
            self._relabeling = self.active
        self._mark_patch(self._patches[self.active])
        self._replace_active(replace(self._patches[self.active], label=label))

    def cycle(self, step: int) -> None:
        """Makes the active patch show the label `step` places on, wrapping through the target and every frame."""
        if self.active is not None:
            self.set_label((self._patches[self.active].label + step) % self.layer_count)

    def delete_active(self) -> None:
        """Removes the active patch."""
        if self.active is not None:
            self._remember()
            self._delete(self.active)

    def undo(self) -> None:
        """Reverts the latest change, if any."""
        if self._undo:
            self._redo.append((self._patches, self.active))
            self._restore(*self._undo.pop())

    def redo(self) -> None:
        """Applies again the change `undo` last reverted, if any."""
        if self._redo:
            self._undo.append((self._patches, self.active))
            self._restore(*self._redo.pop())

    def _remember(self) -> None:
        """Saves the current patches as an undo step and forgets redo."""
        self._undo.append((self._patches, self.active))
        self._redo.clear()
        self._relabeling = None

    def _replace_active(self, patch: Patch) -> None:
        """Swaps the active patch for a changed copy."""
        self._patches = self._patches[:self.active] + (patch,) + self._patches[self.active + 1:]

    def _delete(self, index: int) -> None:
        """Drops patch `index`, leaving no patch active."""
        self._mark_patch(self._patches[index])
        self._patches = self._patches[:index] + self._patches[index + 1:]
        self.active = None

    def _restore(self, patches: tuple[Patch, ...], active: int | None) -> None:
        """Goes back to saved patches, marking every patch that differs from the current ones as changed."""
        for patch in self._patches:
            if not any(patch is other for other in patches):
                self._mark_patch(patch)
        for patch in patches:
            if not any(patch is other for other in self._patches):
                self._mark_patch(patch)
        self._patches, self.active = patches, active
        self._relabeling = None

    def _everything(self) -> tuple[slice, slice]:
        """Slices covering the whole picture."""
        return slice(0, self._plan.labels.shape[0]), slice(0, self._plan.labels.shape[1])

    def _reach(self) -> int:
        """How many pixels from a changed label the feathering still changes the blend: the growth, then the blur."""
        return int(np.ceil((GROW_SIGMAS + 4.0) * self._plan.feather_sigma)) + 2

    def _mark(self, top: int, bottom: int, left: int, right: int) -> None:
        """Adds a box of changed pixels to the dirty box."""
        if self._dirty is not None:
            top, bottom = min(top, self._dirty[0]), max(bottom, self._dirty[1])
            left, right = min(left, self._dirty[2]), max(right, self._dirty[3])
        self._dirty = (top, bottom, left, right)

    def _mark_patch(self, patch: Patch) -> None:
        """Adds a patch's bounding box to the dirty box."""
        self._mark(patch.top, patch.top + patch.mask.shape[0], patch.left, patch.left + patch.mask.shape[1])

    def _layers(self, rows: slice, cols: slice) -> tuple[np.ndarray, np.ndarray]:
        """The label and manual maps for a box with explicit bounds: the plan's labels with each patch laid over them in order."""
        labels = self._plan.labels[rows, cols].copy()
        manual = np.zeros(labels.shape, dtype=bool)
        for patch in self._patches:
            top, bottom = max(patch.top, rows.start), min(patch.top + patch.mask.shape[0], rows.stop)
            left, right = max(patch.left, cols.start), min(patch.left + patch.mask.shape[1], cols.stop)
            if top >= bottom or left >= right:
                continue
            inside = patch.mask[top - patch.top:bottom - patch.top, left - patch.left:right - patch.left]
            labels[top - rows.start:bottom - rows.start, left - cols.start:right - cols.start][inside] = patch.label
            manual[top - rows.start:bottom - rows.start, left - cols.start:right - cols.start] |= inside
        return labels, manual

    def _weights(self, rows: slice, cols: slice) -> np.ndarray:
        """Blending weights for a box with explicit bounds: the plan's labels with their wide feather, and the patches over them with a narrow one."""
        labels, manual = self._layers(rows, cols)
        eligible = [frame.eligible_in(rows, cols, manual) for frame in self._plan.frames]
        if self._plan.median is not None:
            eligible.append(np.ones(labels.shape, dtype=bool))
        automatic = label_weights(self._plan.labels[rows, cols], eligible, self._plan.feather_sigma)
        if not manual.any():
            return automatic
        patch_sigma = PATCH_FEATHER * self._plan.feather_sigma
        patched = label_weights(np.where(manual, labels, 0).astype(np.int16), eligible, patch_sigma)
        cover = label_weights(manual.astype(np.int16), [np.ones(labels.shape, dtype=bool)], patch_sigma)[1]
        return cover * patched + (1.0 - cover) * automatic
