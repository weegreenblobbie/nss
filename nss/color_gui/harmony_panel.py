"""
The Color Harmony panel: numbered eyedropper samples of the preview, a wheel painted with the Color Mixer's output colours, harmony wedges warped
through the mixer's hue mapping, the best-fit wedge rotation and a per-sample info popup.

Coordinates: the wheel angle is the Color Grading wheel's angle (wheel_lut: 0 at 3 o'clock, counter-clockwise).  A mixer hue h (linear ProPhoto,
nss.color_mixer.hue_remap) sits at angle g(h), the angle whose Lightroom rim colour has the screen hue of h (MIXER_TO_WHEEL).  The mixer's hue
mapping f becomes F = g . f . g^-1 on wheel angles (wheel_remap_lut); angle a is painted with the grading wheel's colour at F(a), so with neutral
sliders the wheel is the Color Grading wheel.  A sample's marker sits at g(input hue) and never moves with the sliders.  The wedges are target
OUTPUT angles, drawn at the angles F maps into them (nss.color_explore.harmony.wedge_arcs), so a sample is in harmony when its marker lies inside a drawn wedge.
"""
import math
from typing import Callable, List, NamedTuple, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike
from PyQt6.QtCore import QPoint, QPointF, QRectF, QSize, Qt, pyqtSignal
from PyQt6.QtGui import (QBrush, QColor, QContextMenuEvent, QIcon, QImage, QKeySequence, QMouseEvent, QPainter, QPainterPath, QPaintEvent, QPen,
                         QShortcut)
from PyQt6.QtWidgets import (QComboBox, QDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QMenu, QPushButton, QSlider, QToolButton, QVBoxLayout,
                             QWidget)

from nss.color_explore.harmony import HARMONY_MODES, ROTATION_STEP, best_rotation, in_wedge, nearest_wedge, wedge_arcs
from nss.color_mixer.hue_remap import hue_remap_forward, hue_remap_lut
from nss.color_mixer.mixer import hsv_to_rgb
from nss.color_gui.mixer_panel import LINEAR_PROPHOTO, PROBE_SAT, PROBE_VALUE, RIM_HUES, screen_hue
from nss.color_gui.sample_view import eyedropper_pixmap
from nss.color_gui.wheel_lut import wheel_color
from nss.icc import icc_to_srgb

MAX_SAMPLES = 16
WHEEL_SIZE = 240            # harmony wheel widget side, pixels (markers and arrows sit in its margin)
RIM_MARKER_RADIUS = 7       # sample marker radius outside the rim, pixels
PATCH_SIZE = 26             # patch grid cell side, pixels
PATCH_GAP = 4
PATCH_COLUMNS = 8


class Sample(NamedTuple):
    """One eyedropper sample: slot 1..16, preview pixel (x column, y row), displayed colour at sampling time (encoded 0..1) and mixer-input hue."""
    slot: int
    x: int
    y: int
    rgb: Tuple[float, float, float]
    hue: float


def lowest_free_slot(samples: List[Sample]) -> Optional[int]:
    """The lowest slot 1..MAX_SAMPLES no sample holds, None when all are taken."""
    taken = {sample.slot for sample in samples}
    return next((slot for slot in range(1, MAX_SAMPLES + 1) if slot not in taken), None)


def qcolor(rgb: Tuple[float, float, float]) -> QColor:
    """The 8-bit screen colour of displayed values 0..1, rounded as the preview (nss.image_utils.to_uint8_display)."""
    r, g, b = (int(np.clip(v, 0.0, 1.0) * 255.0 + 0.5) for v in rgb)
    return QColor(r, g, b)


def color_fields(rgb: Tuple[float, float, float]) -> str:
    """'Hue 25°  Sat 60%  Lum 45%  #b4502d' (HSL of the screen colour; hue '-' for a grey)."""
    color = qcolor(rgb)
    h, s, l, _ = color.getHslF()
    hue = "-" if h < 0 else f"{h * 360.0:.0f}°"
    return f"Hue {hue}  Sat {s * 100.0:.0f}%  Lum {l * 100.0:.0f}%  {color.name()}"


def mixer_to_wheel_table() -> np.ndarray:
    """g at the mixer hues 0..359 in the table format of hue_remap (unwrapped): the Color Grading wheel angle whose rim colour has the screen hue of
    a ProPhoto colour of that mixer hue (the match mixer_panel.band_color makes), interpolated between rim degrees.  Both screen-hue curves rise
    strictly, so g is monotonic."""
    h = np.arange(360.0)
    probe = icc_to_srgb(hsv_to_rgb(h, np.full(360, PROBE_SAT), np.full(360, PROBE_VALUE)), LINEAR_PROPHOTO, clip=True)
    rim = np.degrees(np.unwrap(np.radians(RIM_HUES)))                   # screen hue of the rim at angles 0..359, rising
    rim_closed, angles_closed = np.append(rim, rim[0] + 360.0), np.arange(361.0)
    g = np.interp(rim[0] + (screen_hue(probe) - rim[0]) % 360.0, rim_closed, angles_closed)
    return h + (g - h + 180.0) % 360.0 - 180.0


MIXER_TO_WHEEL = mixer_to_wheel_table()


def wheel_angle(mixer_hue: ArrayLike) -> np.ndarray:
    """g: the Color Grading wheel angle (0..360) of mixer hues."""
    return hue_remap_forward(MIXER_TO_WHEEL, mixer_hue)


def wheel_remap_lut(lut: np.ndarray) -> np.ndarray:
    """F = g . f . g^-1 at the wheel angles 0..359 (table format of hue_remap) for the mixer hue table lut (hue_remap_lut, one entry per degree)."""
    angles = np.arange(360.0)
    y, x = np.append(MIXER_TO_WHEEL, MIXER_TO_WHEEL[0] + 360.0), np.arange(361.0)
    mixer_hue = np.interp(y[0] + (angles - y[0]) % 360.0, y, x)               # g^-1, g is monotonic
    out = wheel_angle(hue_remap_forward(lut, mixer_hue))
    return angles + (out - angles + 180.0) % 360.0 - 180.0


def wheel_image(lut: np.ndarray, size: int) -> QImage:
    """size x size ARGB image of the Color Grading wheel's painting with the colour of angle F(a) shown at angle a (lut = F, wheel_remap_lut);
    antialiased edge, transparent outside."""
    c = (size - 1) / 2.0
    yy, xx = np.mgrid[0:size, 0:size].astype(float)
    dx, dy = xx - c, c - yy
    d = np.hypot(dx, dy)
    radius = size / 2.0
    rgb = wheel_color(hue_remap_forward(lut, np.degrees(np.arctan2(dy, dx))), np.minimum(d / radius, 1.0))
    alpha = np.clip(radius - d + 0.5, 0.0, 1.0)
    argb = np.empty((size, size, 4), np.uint8)
    argb[..., :3] = np.rint(rgb * alpha[..., None] * 255.0)
    argb[..., 3] = np.rint(alpha * 255.0)
    bgra = np.ascontiguousarray(argb[..., [2, 1, 0, 3]])
    return QImage(bgra.data, size, size, 4 * size, QImage.Format.Format_ARGB32_Premultiplied).copy()


class HarmonyWheel(QWidget):
    """The harmony wheel: wheel angles painted with their output colours, the warped wedges, the sample markers outside the rim and, for samples
    outside their wedge, an arrow to the nearest edge of it."""
    sample_clicked = pyqtSignal(int, QPoint)            # slot, global position
    sample_menu_requested = pyqtSignal(int, QPoint)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Identity mapping, no samples, mode None."""
        super().__init__(parent)
        self.lut = wheel_remap_lut(hue_remap_lut(np.zeros(8)))      # F on wheel angles
        self.samples: List[Sample] = []
        self.mode = "None"
        self.rotation = 0.0
        self._image: Optional[QImage] = None
        self.setFixedSize(WHEEL_SIZE, WHEEL_SIZE)

    def set_harmony(self, lut: np.ndarray, samples: List[Sample], mode: str, rotation: float) -> None:
        """Sets what is drawn (lut = F on wheel angles, wheel_remap_lut); the wheel colours are repainted only when lut is a new table."""
        if lut is not self.lut:
            self.lut = lut
            self._image = None
        self.samples, self.mode, self.rotation = list(samples), mode, rotation
        self.update()

    def geometry_px(self) -> Tuple[float, float, float]:
        """(cx, cy, radius) of the painted disc, leaving a margin for the markers."""
        return self.width() / 2.0, self.height() / 2.0, min(self.width(), self.height()) / 2.0 - 2.0 * RIM_MARKER_RADIUS - 4.0

    def point_at(self, angle: float, r: float) -> QPointF:
        """Widget position at screen angle (degrees, counter-clockwise) and distance r from the centre."""
        cx, cy, _ = self.geometry_px()
        a = math.radians(angle)
        return QPointF(cx + r * math.cos(a), cy - r * math.sin(a))

    def marker_center(self, sample: Sample) -> QPointF:
        """Centre of a sample's marker, just outside the rim at the wheel angle of its input hue."""
        return self.point_at(float(wheel_angle(sample.hue)), self.geometry_px()[2] + RIM_MARKER_RADIUS + 2.0)

    def marker_at(self, pos: QPointF) -> Optional[int]:
        """Slot of the marker under pos, None if there is none."""
        for sample in reversed(self.samples):
            d = pos - self.marker_center(sample)
            if d.x() * d.x() + d.y() * d.y() <= RIM_MARKER_RADIUS * RIM_MARKER_RADIUS:
                return sample.slot
        return None

    def wedge_centres(self) -> np.ndarray:
        """Output-hue centres of the current mode's wedges."""
        return self.rotation + np.asarray(HARMONY_MODES[self.mode]["offsets"], np.float64)

    def paintEvent(self, event: QPaintEvent) -> None:
        """Wheel, wedges, arrows, then the markers on top."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        cx, cy, radius = self.geometry_px()
        size = int(round(2.0 * radius))
        if self._image is None or self._image.width() != size:
            self._image = wheel_image(self.lut, size)
        painter.drawImage(QPointF(cx - size / 2.0, cy - size / 2.0), self._image)

        width = HARMONY_MODES[self.mode]["width"]
        disc = QRectF(cx - radius, cy - radius, 2.0 * radius, 2.0 * radius)
        arcs = [wedge_arcs(self.lut, centre, width) for centre in self.wedge_centres()]
        painter.setPen(QPen(QColor(30, 30, 30), 1.5))
        painter.setBrush(QBrush(QColor(255, 255, 255, 80)))
        for wedge in arcs:
            for start, end in wedge:
                path = QPainterPath(QPointF(cx, cy))
                path.arcTo(disc, start, end - start)
                path.closeSubpath()
                painter.drawPath(path)

        if arcs:
            offsets = HARMONY_MODES[self.mode]["offsets"]
            angles = wheel_angle([sample.hue for sample in self.samples])
            angles_out = hue_remap_forward(self.lut, angles)
            for sample, angle, k, a_out in zip(self.samples, angles, nearest_wedge(angles_out, self.rotation, offsets), angles_out):
                if in_wedge(a_out, self.wedge_centres()[k], width):
                    continue
                edges = np.array([edge for arc in arcs[k] for edge in arc])
                edge = edges[np.argmin(np.abs((edges - angle + 180.0) % 360.0 - 180.0))]
                self.draw_arrow(painter, self.marker_center(sample), self.point_at(edge, radius - 2.0))

        font = painter.font()
        font.setBold(True)
        font.setPointSize(7)
        painter.setFont(font)
        for sample in self.samples:
            c = self.marker_center(sample)
            painter.setPen(QPen(QColor("black"), 1.2))
            painter.setBrush(QBrush(QColor("white")))
            painter.drawEllipse(c, RIM_MARKER_RADIUS, RIM_MARKER_RADIUS)
            painter.drawText(QRectF(c.x() - RIM_MARKER_RADIUS, c.y() - RIM_MARKER_RADIUS, 2 * RIM_MARKER_RADIUS, 2 * RIM_MARKER_RADIUS),
                             Qt.AlignmentFlag.AlignCenter, str(sample.slot))
        painter.end()

    @staticmethod
    def draw_arrow(painter: QPainter, start: QPointF, end: QPointF) -> None:
        """A black arrow from start to end with a small head at end."""
        painter.setPen(QPen(QColor("black"), 1.5))
        painter.drawLine(start, end)
        a = math.atan2(end.y() - start.y(), end.x() - start.x())
        for side in (-0.45, 0.45):
            painter.drawLine(end, QPointF(end.x() - 6.0 * math.cos(a + side), end.y() - 6.0 * math.sin(a + side)))

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Left click on a marker emits sample_clicked."""
        slot = self.marker_at(event.position())
        if event.button() == Qt.MouseButton.LeftButton and slot is not None:
            self.sample_clicked.emit(slot, event.globalPosition().toPoint())

    def contextMenuEvent(self, event: QContextMenuEvent) -> None:
        """Right click on a marker emits sample_menu_requested."""
        slot = self.marker_at(QPointF(event.pos()))
        if slot is not None:
            self.sample_menu_requested.emit(slot, event.globalPos())


class PatchGrid(QWidget):
    """2 x 8 colour patches, slot N in cell N (row by row); an empty slot is an outlined cell."""
    sample_clicked = pyqtSignal(int, QPoint)
    sample_menu_requested = pyqtSignal(int, QPoint)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """All slots empty."""
        super().__init__(parent)
        self.samples: List[Sample] = []
        rows = MAX_SAMPLES // PATCH_COLUMNS
        self.setFixedSize(PATCH_COLUMNS * (PATCH_SIZE + PATCH_GAP) - PATCH_GAP + 2, rows * (PATCH_SIZE + PATCH_GAP) - PATCH_GAP + 2)

    def set_samples(self, samples: List[Sample]) -> None:
        """Sets the patches."""
        self.samples = list(samples)
        self.update()

    def cell_rect(self, slot: int) -> QRectF:
        """Widget rectangle of slot's cell."""
        row, col = divmod(slot - 1, PATCH_COLUMNS)
        return QRectF(1 + col * (PATCH_SIZE + PATCH_GAP), 1 + row * (PATCH_SIZE + PATCH_GAP), PATCH_SIZE, PATCH_SIZE)

    def slot_at(self, pos: QPointF) -> Optional[int]:
        """Slot of the filled patch under pos, None if there is none."""
        return next((sample.slot for sample in self.samples if self.cell_rect(sample.slot).contains(pos)), None)

    def paintEvent(self, event: QPaintEvent) -> None:
        """Outlined empty cells, filled patches numbered in a contrasting colour."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        by_slot = {sample.slot: sample for sample in self.samples}
        for slot in range(1, MAX_SAMPLES + 1):
            rect = self.cell_rect(slot)
            sample = by_slot.get(slot)
            if sample is None:
                painter.setPen(QPen(QColor(170, 170, 170), 1.0, Qt.PenStyle.DashLine))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRoundedRect(rect, 3.0, 3.0)
                continue
            color = qcolor(sample.rgb)
            painter.setPen(QPen(QColor(90, 90, 90), 1.0))
            painter.setBrush(QBrush(color))
            painter.drawRoundedRect(rect, 3.0, 3.0)
            painter.setPen(QColor("black") if color.lightnessF() > 0.5 else QColor("white"))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, str(slot))
        painter.end()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Left click on a patch emits sample_clicked."""
        slot = self.slot_at(event.position())
        if event.button() == Qt.MouseButton.LeftButton and slot is not None:
            self.sample_clicked.emit(slot, event.globalPosition().toPoint())

    def contextMenuEvent(self, event: QContextMenuEvent) -> None:
        """Right click on a patch emits sample_menu_requested."""
        slot = self.slot_at(QPointF(event.pos()))
        if slot is not None:
            self.sample_menu_requested.emit(slot, event.globalPos())


class SampleInfoDialog(QDialog):
    """A sample's original colour (at sampling time) and current colour (the preview at that pixel now): swatch, hue, saturation, lightness, hex."""

    def __init__(self, slot: int, original: Tuple[float, float, float], current: Optional[Tuple[float, float, float]], parent: Optional[QWidget] = None) -> None:
        """Builds the two rows; current None (no preview shown) reads 'not available'."""
        super().__init__(parent)
        self.setWindowTitle(f"Sample {slot}")
        layout = QGridLayout(self)
        for row, (title, rgb) in enumerate((("Original", original), ("Current", current))):
            layout.addWidget(QLabel(title), row, 0)
            swatch = QLabel()
            swatch.setFixedSize(36, 24)
            if rgb is not None:
                swatch.setStyleSheet(f"background: {qcolor(rgb).name()}; border: 1px solid #5c5c5c;")
            layout.addWidget(swatch, row, 1)
            layout.addWidget(QLabel("not available" if rgb is None else color_fields(rgb)), row, 2)


class HarmonyPanel(QGroupBox):
    """The Color Harmony controls: mode and eyedropper, the harmony wheel, the 2 x 8 sample patches, the wedge rotation, the fit readout and
    Reset Harmony.  The snapshot (samples, mode, rotation) takes part in undo / redo; the eyedropper on / off does not."""
    STATE_KEY = "color_harmony"
    sampling_changed = pyqtSignal(bool)     # the eyedropper was turned on / off
    samples_changed = pyqtSignal()          # samples were added, removed or restored

    def __init__(self, on_released_slot: Callable[[], None], current_rgb: Callable[[int, int], Optional[np.ndarray]], parent: Optional[QWidget] = None) -> None:
        """on_released_slot records an undo step; current_rgb(x, y) is the preview's colour now at a sample's pixel (None without a preview)."""
        super().__init__("COLOR HARMONY", parent)
        self.on_released_slot = on_released_slot
        self.current_rgb = current_rgb
        self.samples: List[Sample] = []
        self.hue_levels: Tuple[float, ...] = (0.0,) * 8
        self.lut = wheel_remap_lut(hue_remap_lut(np.zeros(8)))       # F on wheel angles
        self.rotation = 0.0
        self.setStyleSheet(
            "QGroupBox { font-weight: bold; border: 1px solid #cccccc; border-radius: 4px; margin-top: 10px; padding: 10px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 3px; }"
        )
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        top_row = QHBoxLayout()
        top_row.addWidget(QLabel("Mode:"))
        self.mode_dropdown = QComboBox()
        self.mode_dropdown.addItems(list(HARMONY_MODES))
        self.mode_dropdown.currentTextChanged.connect(self.on_mode_changed)
        top_row.addWidget(self.mode_dropdown, stretch=1)
        self.eyedropper_btn = QToolButton()
        self.eyedropper_btn.setIcon(QIcon(eyedropper_pixmap()))
        self.eyedropper_btn.setIconSize(QSize(18, 18))
        self.eyedropper_btn.setCheckable(True)
        self.eyedropper_btn.setToolTip("Sample colours from the preview (Esc to stop)")
        self.eyedropper_btn.toggled.connect(self.sampling_changed.emit)
        top_row.addWidget(self.eyedropper_btn)
        layout.addLayout(top_row)

        self.wheel = HarmonyWheel()
        layout.addWidget(self.wheel, alignment=Qt.AlignmentFlag.AlignHCenter)
        self.patches = PatchGrid()
        layout.addWidget(self.patches, alignment=Qt.AlignmentFlag.AlignHCenter)
        for view in (self.wheel, self.patches):
            view.sample_clicked.connect(self.show_sample_info)
            view.sample_menu_requested.connect(self.show_sample_menu)

        self.rotation_lbl = QLabel("Wedge Rotation: 0.0°")
        layout.addWidget(self.rotation_lbl)
        self.rotation_slider = QSlider(Qt.Orientation.Horizontal)
        self.rotation_slider.setRange(0, int(round(360.0 / ROTATION_STEP)) - 1)
        self.rotation_slider.setToolTip("Rotates all wedges together")
        self.rotation_slider.valueChanged.connect(self.on_rotation_changed)
        self.rotation_slider.sliderReleased.connect(self.on_released_slot)
        layout.addWidget(self.rotation_slider)

        self.readout_lbl = QLabel("")
        layout.addWidget(self.readout_lbl)

        self.reset_btn = QPushButton("Reset Harmony")
        self.reset_btn.setToolTip("Remove all samples, set the mode to None and the rotation to 0 (the Color Mixer is not touched)")
        self.reset_btn.clicked.connect(self.reset)
        layout.addWidget(self.reset_btn)

        self.escape = QShortcut(QKeySequence(Qt.Key.Key_Escape), self)
        self.escape.setContext(Qt.ShortcutContext.WindowShortcut)
        self.escape.activated.connect(lambda: self.eyedropper_btn.setChecked(False))
        self.refresh()

    @property
    def mode(self) -> str:
        """The harmony mode shown in the dropdown."""
        return self.mode_dropdown.currentText()

    def output_angles(self) -> np.ndarray:
        """Each sample's current output wheel angle F(g(input hue))."""
        return hue_remap_forward(self.lut, wheel_angle([sample.hue for sample in self.samples]))

    def refresh(self) -> None:
        """Redraws the wheel, patches, rotation label and readout, enables the eyedropper while a slot is free, and emits samples_changed."""
        if len(self.samples) >= MAX_SAMPLES:
            self.eyedropper_btn.setChecked(False)
        self.eyedropper_btn.setEnabled(len(self.samples) < MAX_SAMPLES)
        self.wheel.set_harmony(self.lut, self.samples, self.mode, self.rotation)
        self.patches.set_samples(self.samples)
        self.rotation_lbl.setText(f"Wedge Rotation: {self.rotation:.1f}°")
        offsets, width = HARMONY_MODES[self.mode]["offsets"], HARMONY_MODES[self.mode]["width"]
        if offsets and self.samples:
            angles = self.output_angles()
            centres = self.rotation + np.asarray(offsets)[nearest_wedge(angles, self.rotation, offsets)]
            self.readout_lbl.setText(f"{int(in_wedge(angles, centres, width).sum())} / {len(self.samples)} samples in harmony")
        else:
            self.readout_lbl.setText("")
        self.samples_changed.emit()

    def set_rotation(self, rotation: float) -> None:
        """Sets the rotation and the slider without emitting its signals."""
        self.rotation = rotation
        self.rotation_slider.blockSignals(True)
        self.rotation_slider.setValue(int(round(rotation / ROTATION_STEP)))
        self.rotation_slider.blockSignals(False)

    def set_hue_levels(self, levels: Tuple[float, ...]) -> None:
        """The Color Mixer's eight Hue sliders changed: recomputes f and warps the wheel colours and wedges (the markers stay)."""
        levels = tuple(float(v) for v in levels)
        if levels == self.hue_levels:
            return
        self.hue_levels = levels
        self.lut = wheel_remap_lut(hue_remap_lut(np.array(levels)))
        self.refresh()

    def add_sample(self, x: int, y: int, rgb: np.ndarray, hue: float) -> None:
        """Adds a sample in the lowest free slot and records an undo step; filling the last slot stops sampling."""
        slot = lowest_free_slot(self.samples)
        if slot is None:
            return
        self.samples.append(Sample(slot, int(x), int(y), tuple(float(v) for v in rgb), float(hue)))
        self.refresh()
        self.on_released_slot()

    def remove_sample(self, slot: int) -> None:
        """Removes the sample in slot (its number becomes free) and records an undo step."""
        self.samples = [sample for sample in self.samples if sample.slot != slot]
        self.refresh()
        self.on_released_slot()

    def clear_samples(self) -> None:
        """Removes every sample without an undo step (a new image was opened)."""
        self.samples = []
        self.refresh()

    def on_mode_changed(self, mode: str) -> None:
        """Fits the rotation to the samples' current output wheel angles for the new mode and records one undo step."""
        self.set_rotation(best_rotation(self.output_angles(), HARMONY_MODES[mode]["offsets"]))
        self.refresh()
        self.on_released_slot()

    def on_rotation_changed(self, value: int) -> None:
        """Rotation slider moved: rotates the wedges (the undo step is recorded on release)."""
        self.rotation = value * ROTATION_STEP
        self.refresh()

    def reset(self) -> None:
        """Removes every sample, sets the mode to None and the rotation to 0, and records one undo step; the Color Mixer is not touched."""
        self.samples = []
        self.mode_dropdown.blockSignals(True)
        self.mode_dropdown.setCurrentText("None")
        self.mode_dropdown.blockSignals(False)
        self.set_rotation(0.0)
        self.refresh()
        self.on_released_slot()

    def show_sample_menu(self, slot: int, pos: QPoint) -> None:
        """Context menu with 'Remove sample N' at the global position pos."""
        menu = QMenu(self)
        menu.addAction(f"Remove sample {slot}", lambda: self.remove_sample(slot))
        menu.exec(pos)

    def show_sample_info(self, slot: int) -> None:
        """Opens the info popup of the sample in slot."""
        sample = next(sample for sample in self.samples if sample.slot == slot)
        current = self.current_rgb(sample.x, sample.y)
        self.info_dialog = SampleInfoDialog(slot, sample.rgb, None if current is None else tuple(current), self)
        self.info_dialog.show()

    def get_state(self) -> dict:
        """The harmony snapshot {STATE_KEY: {samples: [Sample fields], mode, rotation}}."""
        return {self.STATE_KEY: {"samples": [sample._asdict() for sample in self.samples], "mode": self.mode, "rotation": self.rotation}}

    def restore_state(self, full_state_dict: dict) -> None:
        """Sets samples, mode and rotation from a snapshot without recording an undo step; snapshots without the harmony are ignored."""
        if self.STATE_KEY in full_state_dict:
            state = full_state_dict[self.STATE_KEY]
            self.samples = [Sample(d["slot"], d["x"], d["y"], tuple(d["rgb"]), d["hue"]) for d in state["samples"]]
            self.mode_dropdown.blockSignals(True)
            self.mode_dropdown.setCurrentText(state["mode"])
            self.mode_dropdown.blockSignals(False)
            self.set_rotation(state["rotation"])
            self.refresh()
