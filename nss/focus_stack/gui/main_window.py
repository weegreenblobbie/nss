"""The focus stack window: the canvas, the menus, and the background threads that align the frames and save the full-size picture."""
import copy
from pathlib import Path

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

from nss.focus_stack.edit import EditState
from nss.focus_stack.clean import stack_clean, stack_clean2
from nss.focus_stack.gui.canvas import CanvasView
from nss.focus_stack.gui.layers_panel import LayersPanel
from nss.focus_stack.register import RegistrationError
from nss.focus_stack.stack import AlignedFrame, StackResult, render_strip, stack_frames
from nss.image_utils import downsample_image, to_uint8_display
from nss.tiff_file import TiffFile
from plugin import stack_bridge

TILE_ROWS = 1024  # rows of the full-size picture rendered at a time when saving
THUMBNAIL_DIM = 160  # long side in pixels of the thumbnails in the Layers panel
MEDIAN_NAME = "Clean plate (median)"  # how the median layer is listed in the Layers panel
PLATE_NAME = "Clean plate"  # how the layer is listed when it is `clean2`'s plate, which is not a median
STACKERS = {"focus": stack_frames, "clean": stack_clean, "clean2": stack_clean2}  # the stacking function for each mode


class AlignThread(QtCore.QThread):
    """Reads the pictures and aligns every frame to the target off the GUI thread."""

    progress = QtCore.pyqtSignal(str)  # what it is doing
    loaded = QtCore.pyqtSignal(object)  # (the target TiffFile, the frame arrays, the StackResult)
    failed = QtCore.pyqtSignal(str)  # one line: why it could not

    def __init__(self, target_path: str, frame_paths: list[str], work_dim: int, mode: str) -> None:
        """A thread that will stack `frame_paths` onto `target_path` at working size `work_dim` once started; `mode` is "focus", "clean" or "clean2"."""
        super().__init__()
        self.target_path = target_path
        self.frame_paths = frame_paths
        self.work_dim = work_dim
        self.mode = mode

    def run(self) -> None:
        """Emits loaded, or failed with the reason."""
        try:
            self.progress.emit("Reading the pictures")
            target_file = TiffFile().read(self.target_path)
            frames = [TiffFile().read(path).array for path in self.frame_paths]
            self.progress.emit(f"Aligning {len(frames)} frame{'s' if len(frames) != 1 else ''}")
            plan = STACKERS[self.mode](target_file.array, frames, self.work_dim)
        except RegistrationError as error:
            self.failed.emit(f"registration failed: {error}")
            return
        except Exception as error:
            self.failed.emit(f"{type(error).__name__}: {error}")
            return
        self.loaded.emit((target_file, frames, plan))


class RestackThread(QtCore.QThread):
    """Aligns frames to a new target, from pictures already in memory, off the GUI thread."""

    loaded = QtCore.pyqtSignal(object)  # the StackResult
    failed = QtCore.pyqtSignal(str)  # one line: why it could not

    def __init__(self, target: np.ndarray, frames: list[np.ndarray], work_dim: int, mode: str) -> None:
        """A thread that will stack `frames` onto `target` at working size `work_dim` once started; `mode` is "focus", "clean" or "clean2"."""
        super().__init__()
        self.target = target
        self.frames = frames
        self.work_dim = work_dim
        self.mode = mode

    def run(self) -> None:
        """Emits loaded, or failed with the reason."""
        try:
            plan = STACKERS[self.mode](self.target, self.frames, self.work_dim)
        except RegistrationError as error:
            self.failed.emit(f"registration failed: {error}")
            return
        except Exception as error:
            self.failed.emit(f"{type(error).__name__}: {error}")
            return
        self.loaded.emit(plan)


class SaveThread(QtCore.QThread):
    """Renders the full-size picture strip by strip and writes it with the target's bit depth and ICC profile."""

    progress = QtCore.pyqtSignal(int)  # rows done
    saved = QtCore.pyqtSignal(str)  # the path written
    failed = QtCore.pyqtSignal(str)  # one line: the exception

    def __init__(
        self,
        target_file: TiffFile,
        frame_arrays: list[np.ndarray],
        frames: list[AlignedFrame],
        weights: np.ndarray,
        path: str,
        plate: np.ndarray | None = None,
    ) -> None:
        """A thread that will render with `weights` (a snapshot, so later edits do not matter) and write `path` once started; `plate` is as for `render_strip`."""
        super().__init__()
        self.target_file = target_file
        self.frame_arrays = frame_arrays
        self.frames = frames
        self.weights = weights
        self.path = path
        self.plate = plate

    def run(self) -> None:
        """Emits saved, or failed on an exception."""
        try:
            target = self.target_file.array
            out = np.empty_like(target)
            for first in range(0, target.shape[0], TILE_ROWS):
                last = min(first + TILE_ROWS, target.shape[0])
                out[first:last] = render_strip(target, self.frame_arrays, self.frames, self.weights, first, last, self.plate)
                self.progress.emit(last)
            result = copy.copy(self.target_file)
            result.array = out
            result.saveas(self.path)
        except Exception as error:
            self.failed.emit(f"{type(error).__name__}: {error}")
            return
        self.saved.emit(self.path)


class MainWindow(QtWidgets.QMainWindow):
    """The focus stack window: paint patches over the automatic merge and choose which frame each shows, then save at full size."""

    def __init__(self) -> None:
        """An empty window; open() loads pictures and set_stack() shows a finished stack."""
        super().__init__()
        self.exit_code = 0
        self._request: stack_bridge.Request | None = None  # set by use_lightroom
        self.lightroom_status = stack_bridge.CANCELLED
        self.lightroom_message = ""
        self.canvas = CanvasView()
        self.setCentralWidget(self.canvas)
        self.setWindowTitle("Focus Stack")
        self.resize(1000, 800)
        self._target_path = ""
        self._paths: list[str] = []  # the pictures' paths in the order they were given: the target first, then the frames
        self._order: list[int] = []  # for each layer, the index in _paths of the picture it shows (0 is the target)
        self._restack_thread: RestackThread | None = None
        self._target_file: TiffFile | None = None
        self._frame_arrays: list[np.ndarray] = []
        self._plan: StackResult | None = None
        self._unsaved = False
        self._load_thread: AlignThread | None = None
        self._save_thread: SaveThread | None = None

        self.status_label = QtWidgets.QLabel()
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setMaximumWidth(220)
        self.progress_bar.hide()
        self.statusBar().addPermanentWidget(self.progress_bar)
        self.statusBar().addPermanentWidget(self.status_label)
        self.canvas.status_changed.connect(self.status_label.setText)
        self.canvas.edited.connect(self._on_edited)

        self.save_action = self._action("Save Full Resolution...", self.save_full_resolution, "Ctrl+S")
        self.save_action.setEnabled(False)
        self.undo_action = self._action("Undo", self.canvas.undo, "Ctrl+Z")
        self.redo_action = self._action("Redo", self.canvas.redo, ["Ctrl+Shift+Z", "Ctrl+Y"])
        file_menu = self.menuBar().addMenu("File")
        file_menu.addAction(self.save_action)
        file_menu.addAction(self._action("Quit", self.close, "Ctrl+Q"))
        edit_menu = self.menuBar().addMenu("Edit")
        edit_menu.addAction(self.undo_action)
        edit_menu.addAction(self.redo_action)
        edit_menu.addAction(self._action("Deselect\tEsc", self.canvas.deselect))
        edit_menu.addAction(self._action("Delete Patch\tDel", self.canvas.delete_patch))
        self.zoom_fit_action = self._action("Fit", self.canvas.zoom_fit, "Ctrl+0")
        self.zoom_100_action = self._action("100%", lambda: self.canvas.set_zoom(1.0), "Ctrl+1")
        self.zoom_200_action = self._action("200%", lambda: self.canvas.set_zoom(2.0), "Ctrl+2")
        self.zoom_400_action = self._action("400%", lambda: self.canvas.set_zoom(4.0), "Ctrl+3")
        self.zoom_out_action = self._action("Zoom Out", self.canvas.zoom_out, "Ctrl+-")
        self.zoom_in_action = self._action("Zoom In", self.canvas.zoom_in, ["Ctrl+=", "Ctrl++"])
        zoom_actions = [
            self.zoom_fit_action, self.zoom_100_action, self.zoom_200_action, self.zoom_400_action, self.zoom_out_action, self.zoom_in_action
        ]
        view_menu = self.menuBar().addMenu("View")
        for action in zoom_actions:
            view_menu.addAction(action)
        view_menu.addSeparator()
        view_menu.addAction(self._action("Toggle Layer Tint\tL", self.canvas.toggle_tint))
        self.pan_action = QtGui.QAction("Pan", self)
        self.pan_action.setCheckable(True)
        self.pan_action.setToolTip("Pan mode: drag to move the picture (Space or Esc to leave)")
        self.pan_action.triggered.connect(self.canvas.set_pan_mode)
        self.canvas.pan_mode_changed.connect(self.pan_action.setChecked)
        view_menu.addAction(self.pan_action)
        self.toolbar = self.addToolBar("Zoom")
        self.toolbar.setMovable(False)
        for action in zoom_actions:
            self.toolbar.addAction(action)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.pan_action)

        self.layers = LayersPanel()
        self.layers.layer_chosen.connect(self._choose_layer)
        self.layers.layer_viewed.connect(self._view_layer)
        self.layers.target_requested.connect(self._set_target)
        self.canvas.view_changed.connect(self._on_view_changed)
        dock = QtWidgets.QDockWidget("Layers", self)
        dock.setWidget(self.layers)
        dock.setFeatures(QtWidgets.QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        dock.setMinimumWidth(260)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, dock)

    def open(self, target_path: str, frame_paths: list[str], work_dim: int, mode: str) -> None:
        """Aligns `frame_paths` to `target_path` in a thread, as `mode` ("focus", "clean" or "clean2") says, and shows the stack when it is ready."""
        self.setWindowTitle(f"Focus Stack - {Path(target_path).name} (loading)")
        self.progress_bar.setRange(0, 0)
        self.progress_bar.show()
        thread = AlignThread(target_path, frame_paths, work_dim, mode)
        thread.progress.connect(self.statusBar().showMessage)
        thread.loaded.connect(lambda loaded: self._on_loaded(target_path, frame_paths, loaded))
        thread.failed.connect(self._on_load_failed)
        self._load_thread = thread
        thread.start()

    def set_stack(
        self,
        target_path: str,
        target_file: TiffFile,
        frame_paths: list[str],
        frame_arrays: list[np.ndarray],
        plan: StackResult,
        *,
        order: list[int] | None = None,
    ) -> None:
        """Shows a finished stack: the canvas gets an edit state with no patches, and saving becomes available.

        `order` says, for each layer, which of the pictures first given it shows; None for a fresh stack in the order given.
        """
        if order is None:
            self._paths = [target_path] + list(frame_paths)
            order = list(range(len(self._paths)))
        self._order = order
        self._target_path = self._paths[order[0]]
        self._target_file = target_file
        self._frame_arrays = frame_arrays
        self._plan = plan
        self._unsaved = False
        state = EditState(plan)
        frame_names = [Path(self._paths[i]).name for i in order[1:]]
        thumbnails = [self._thumbnail(state.target)] + [self._thumbnail(frame.aligned) for frame in plan.frames]
        if plan.median is not None:
            frame_names.append(MEDIAN_NAME if plan.plate_is_median else PLATE_NAME)
            thumbnails.append(self._thumbnail(plan.median))
        self.canvas.set_state(state, ["target"] + frame_names, full_shape=target_file.array.shape[:2])
        self.layers.set_layers(frame_names, thumbnails, len(plan.frames))
        self.layers.show_active(None)
        self.setWindowTitle(f"Focus Stack - {Path(self._target_path).name}")
        self.save_action.setEnabled(True)
        self.canvas.setFocus()

    def use_lightroom(self, request: stack_bridge.Request) -> None:
        """Switches to Lightroom mode: Save writes `request.output` and the saved result and closes, Cancel (or closing) leaves the photo alone.

        Until it saves, the result is cancelled; a failure makes it an error.
        """
        self._request = request
        self.lightroom_status, self.lightroom_message = stack_bridge.CANCELLED, ""
        self.save_action.setText("Save to Lightroom")
        self.cancel_action = self._action("Cancel", self.close)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.save_action)
        self.toolbar.addAction(self.cancel_action)

    def finish_lightroom(self) -> int:
        """Writes the result file unless the picture was saved (which wrote it), and returns the exit code for the result."""
        if self.lightroom_status != stack_bridge.SAVED:
            try:
                stack_bridge.write_result(self._request.result, self.lightroom_status, message=self.lightroom_message)
            except OSError as error:
                print(f"Cannot write the result file {self._request.result}: {error}")
        return stack_bridge.EXIT_CODES[self.lightroom_status]

    def save_full_resolution(self) -> None:
        """Renders and writes the full-size picture: to the request's output in Lightroom mode, otherwise to a file the user picks."""
        if self._request is not None:
            self.save_to(self._request.outputs[self._order[0]])
            return
        default = str(Path(self._target_path).with_name(Path(self._target_path).stem + "_stacked.tif"))
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Full Resolution", default, "TIFF (*.tif *.tiff)")
        if path:
            self.save_to(path)

    def save_to(self, path: str) -> None:
        """Renders the full-size picture with the patches as they are now in a thread, with a progress bar, and writes `path`."""
        rows = self._target_file.array.shape[0]
        self.progress_bar.setRange(0, rows)
        self.progress_bar.setValue(0)
        self.progress_bar.show()
        self.save_action.setEnabled(False)
        plate = None if self._plan.plate_is_median else self._plan.median
        thread = SaveThread(self._target_file, self._frame_arrays, self._plan.frames, self.canvas.state.weights(), path, plate)
        thread.progress.connect(self.progress_bar.setValue)
        thread.saved.connect(self._on_saved)
        thread.failed.connect(self._on_save_failed)
        self._save_thread = thread
        thread.start()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        """Asks before closing when there are patches that were not saved."""
        question = "Discard the patches? The photo in Lightroom stays unchanged." if self._request is not None else "The patches have not been saved. Quit anyway?"
        if self._unsaved and QtWidgets.QMessageBox.question(self, "Focus Stack", question) != QtWidgets.QMessageBox.StandardButton.Yes:
            event.ignore()
            return
        event.accept()

    def _action(self, text: str, slot, shortcut: str | list[str] | None = None) -> QtGui.QAction:
        """A menu action running `slot`, with an optional shortcut or list of shortcuts."""
        action = QtGui.QAction(text, self)
        action.triggered.connect(lambda checked=False: slot())
        if isinstance(shortcut, list):
            action.setShortcuts([QtGui.QKeySequence(s) for s in shortcut])
        elif shortcut is not None:
            action.setShortcut(QtGui.QKeySequence(shortcut))
        return action

    @staticmethod
    def _thumbnail(image: np.ndarray) -> np.ndarray:
        """A small 8-bit copy of a float picture for the Layers panel."""
        return to_uint8_display(downsample_image(image, THUMBNAIL_DIM))

    def _on_edited(self) -> None:
        """Remembers that there are patches that a save has not seen, and highlights the active patch's layer in the panel."""
        self._unsaved = self.canvas.state.patch_count > 0
        self._show_active_layer()

    def _show_active_layer(self) -> None:
        """Highlights the active patch's layer in the panel, or nothing without an active patch."""
        state = self.canvas.state
        self.layers.show_active(None if state.active is None else state.patch_label(state.active))

    def _view_layer(self, layer: int) -> None:
        """Ctrl+click in the panel: looks at that layer's picture, or goes back to the merged picture if it was the one being looked at."""
        self.canvas.view_layer(None if self.canvas.viewed_layer == layer else layer)
        self.canvas.setFocus()

    def _on_view_changed(self, layer: int | None) -> None:
        """Marks the viewed layer in the panel, and puts the active patch's highlight back after the click that changed the view."""
        self.layers.show_viewed(layer)
        self._show_active_layer()

    def _set_target(self, layer: int) -> None:
        """Makes frame `layer` the target: the old target takes its place among the frames and everything is aligned again, in a thread.

        Patches are discarded, so with any it asks first.
        """
        state = self.canvas.state
        busy = self._restack_thread is not None and self._restack_thread.isRunning()
        if state is None or busy or not 1 <= layer <= state.frame_count:
            return
        name = Path(self._paths[self._order[layer]]).name
        if state.patch_count and QtWidgets.QMessageBox.question(
            self, "Focus Stack", f"Making {name} the target discards the {state.patch_count} patch(es) painted so far. Continue?"
        ) != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        order = list(self._order)
        order[0], order[layer] = order[layer], order[0]
        new_file = copy.copy(self._target_file)
        new_file.array = self._frame_arrays[layer - 1]
        new_frames = list(self._frame_arrays)
        new_frames[layer - 1] = self._target_file.array
        self.layers.setEnabled(False)
        self.canvas.setEnabled(False)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.show()
        self.statusBar().showMessage(f"Aligning with {name} as the target")
        mode = "focus" if self._plan.median is None else ("clean" if self._plan.plate_is_median else "clean2")
        thread = RestackThread(new_file.array, new_frames, max(self._plan.target.shape[:2]), mode)
        thread.loaded.connect(lambda plan: self._on_restacked(order, new_file, new_frames, plan))
        thread.failed.connect(self._on_restack_failed)
        self._restack_thread = thread
        thread.start()

    def _on_restacked(self, order: list[int], target_file: TiffFile, frame_arrays: list[np.ndarray], plan: StackResult) -> None:
        """Shows the stack made with the new target."""
        self.progress_bar.hide()
        self.statusBar().clearMessage()
        self.layers.setEnabled(True)
        self.canvas.setEnabled(True)
        self.set_stack(self._paths[order[0]], target_file, [self._paths[i] for i in order[1:]], frame_arrays, plan, order=order)

    def _on_restack_failed(self, message: str) -> None:
        """Says why the new target could not be used; the old stack stays as it was."""
        self.progress_bar.hide()
        self.statusBar().clearMessage()
        self.layers.setEnabled(True)
        self.canvas.setEnabled(True)
        QtWidgets.QMessageBox.critical(self, "Focus Stack", message)

    def _choose_layer(self, layer: int) -> None:
        """Gives the active patch the layer clicked in the panel, or says to paint a patch first."""
        if self.canvas.state is None or self.canvas.state.active is None:
            self.statusBar().showMessage("Paint a patch first, then pick its layer", 4000)
            return
        self.canvas.set_layer(layer)
        self.canvas.setFocus()

    def _on_loaded(self, target_path: str, frame_paths: list[str], loaded: tuple) -> None:
        """Shows the stack the thread built."""
        self.progress_bar.hide()
        self.statusBar().clearMessage()
        self.set_stack(target_path, loaded[0], frame_paths, loaded[1], loaded[2])

    def _on_load_failed(self, message: str) -> None:
        """Reports why the frames could not be stacked and closes with exit code 1."""
        self.progress_bar.hide()
        self.exit_code = 1
        if self._request is not None:
            self.lightroom_status, self.lightroom_message = stack_bridge.ERROR, message
        QtWidgets.QMessageBox.critical(self, "Focus Stack", message)
        self.close()

    def _on_saved(self, path: str) -> None:
        """Hides the progress bar and says where the picture went."""
        self.progress_bar.hide()
        self.save_action.setEnabled(True)
        self._unsaved = False
        self.statusBar().showMessage(f"Saved {path}")
        if self._request is not None:
            try:
                stack_bridge.write_result(self._request.result, stack_bridge.SAVED, path, target=self._order[0])
            except OSError as error:
                self._on_save_failed(f"cannot write the result file {self._request.result}: {error}")
                return
            self.lightroom_status = stack_bridge.SAVED
            self.close()

    def _on_save_failed(self, message: str) -> None:
        """Hides the progress bar and reports the error."""
        self.progress_bar.hide()
        self.save_action.setEnabled(True)
        if self._request is not None:
            self.lightroom_status, self.lightroom_message = stack_bridge.ERROR, f"could not save: {message}"
        QtWidgets.QMessageBox.critical(self, "Focus Stack", f"Could not save: {message}")
