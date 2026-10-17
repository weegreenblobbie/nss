"""The Color Mixer panel (Hue / Saturation / Luminance tabs of 8 band sliders) and the cache of mixed preview images, for nss.color_mixer.ColorMixer."""
import threading
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QMouseEvent
from PyQt6.QtWidgets import QGroupBox, QHBoxLayout, QLabel, QPushButton, QSlider, QTabWidget, QVBoxLayout, QWidget

from numpy.typing import ArrayLike

from nss.color_mixer import ColorMixer
from nss.color_mixer.bands import CENTRES, hue_shift
from nss.color_mixer.mixer import PROPHOTO_XYZ, hexagon_hue, hsv_to_rgb
from nss.icc import gamma_icc_bytes, icc_to_srgb
from nss.color_gui.wheel_lut_tables import RIM_RGB

BAND_NAMES = ("Red", "Orange", "Yellow", "Green", "Aqua", "Blue", "Purple", "Magenta")
KINDS = ("hue", "sat", "lum")
KIND_TITLES = {"hue": "Hue", "sat": "Saturation", "lum": "Luminance"}
Levels = Tuple[Tuple[float, ...], Tuple[float, ...], Tuple[float, ...]]     # (hue, sat, lum), eight values each, Red to Magenta

# Track painting.  A mixer hue (hexagon hue of linear ProPhoto) is shown as the wheel rim colour whose screen hue matches the screen hue of a
# ProPhoto colour of that hue, taken at a saturation that stays inside sRGB so the screen hue is not bent by clipping.
LINEAR_PROPHOTO = gamma_icc_bytes("linear ProPhoto RGB", [tuple(column) for column in PROPHOTO_XYZ.T], 1.0)
PROBE_SAT, PROBE_VALUE = 0.35, 0.8      # the ProPhoto colour whose screen hue names a mixer hue
HUE_TRACK_STOPS = 17                    # Hue track gradient stops, slider levels -100..100
LUM_DARK = 0.3                          # Luminance track left end: this fraction of the band colour
LUM_LIGHT = 0.6                         # Luminance track right end: the band colour blended this far toward white


def screen_hue(rgb: ArrayLike) -> np.ndarray:
    """HSV hue in degrees of display RGB (..., 3), flattened to (n,)."""
    p = np.asarray(rgb, float).reshape(-1, 3)
    v = p.max(axis=1)
    return hexagon_hue(p, v, v - p.min(axis=1))


RIM_HUES = screen_hue(RIM_RGB)          # screen hue of each of the wheel's 360 rim colours


def band_color(hue: ArrayLike) -> np.ndarray:
    """Display RGB 0..1 (n, 3) of mixer hues (degrees, linear ProPhoto): the wheel rim colour with the same screen hue."""
    h = np.atleast_1d(np.asarray(hue, float)).ravel()
    probe = icc_to_srgb(hsv_to_rgb(h, np.full(h.shape, PROBE_SAT), np.full(h.shape, PROBE_VALUE)), LINEAR_PROPHOTO, clip=True)
    distance = np.abs((RIM_HUES[None, :] - screen_hue(probe)[:, None] + 180.0) % 360.0 - 180.0)
    return RIM_RGB[np.argmin(distance, axis=1)]


def band_hues(band: int, levels: ArrayLike) -> np.ndarray:
    """Mixer hue that the centre of `band` moves to at each of its Hue slider levels (-100..100)."""
    out = []
    for level in np.asarray(levels, float):
        sliders = np.zeros(8)
        sliders[band] = level
        out.append(CENTRES[band] + float(hue_shift(CENTRES[band], sliders)))
    return np.array(out)


def track_colors(kind: str, band: int) -> np.ndarray:
    """Display RGB stops (n, 3), left to right, of a slider track: Hue = the hues the band moves to; Sat = grey of the band colour's
    brightness to the band colour; Lum = dark to light band colour."""
    if kind == "hue":
        return band_color(band_hues(band, np.linspace(-100.0, 100.0, HUE_TRACK_STOPS)))
    color = band_color(CENTRES[band])[0]
    if kind == "sat":
        grey = float(np.dot(color, (0.2126, 0.7152, 0.0722)))
        return np.array([(grey, grey, grey), color])
    return np.array([LUM_DARK * color, color, color + LUM_LIGHT * (1.0 - color)])


def track_stylesheet(colors: np.ndarray) -> str:
    """Style sheet giving a horizontal QSlider a groove painted with evenly spaced gradient stops (display RGB 0..1), styled as the wheels' Hue slider."""
    n = len(colors)
    stops = ", ".join(f"stop:{i / (n - 1):.4f} rgb({r}, {g}, {b})" for i, (r, g, b) in enumerate(np.rint(np.clip(colors, 0.0, 1.0) * 255).astype(int)))
    return ("QSlider::groove:horizontal { height: 8px; border: 1px solid #8c8c8c; border-radius: 3px; "
            f"background: qlineargradient(x1:0, y1:0, x2:1, y2:0, {stops}); }}"
            "QSlider::handle:horizontal { width: 10px; margin: -4px 0; border: 1px solid #5c5c5c; border-radius: 3px; background: #f0f0f0; }")


def mixer_levels(state: dict) -> Levels:
    """(hue, sat, lum) slider tuples of a state's mixer_hue / mixer_sat / mixer_lum; missing keys count as all zero."""
    return tuple(tuple(float(v) for v in state.get(f"mixer_{kind}", (0.0,) * 8)) for kind in KINDS)


class ResetSlider(QSlider):
    """A slider that emits doubleClicked when double-clicked."""
    doubleClicked = pyqtSignal()

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        """Emits doubleClicked."""
        self.doubleClicked.emit()


class MixerPanel(QGroupBox):
    """The Color Mixer controls: Hue / Saturation / Luminance tabs, each with a full-width -100..100 slider per band under its name (tinted with the
    band colour) and value; then Reset Mixer."""
    STATE_KEY = "color_mixer"

    def __init__(self, on_changed_slot: Callable[[], None], on_released_slot: Callable[[], None], parent: Optional[QWidget] = None) -> None:
        """Builds the three tabs; value changes call on_changed_slot; releases, double-click resets and Reset Mixer call on_released_slot."""
        super().__init__("COLOR MIXER", parent)
        self.on_released_slot = on_released_slot
        self.setStyleSheet(
            "QGroupBox { font-weight: bold; border: 1px solid #cccccc; border-radius: 4px; margin-top: 10px; padding: 10px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 3px; }"
        )
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # Equal-width tabs in the panel's light-grey border style, as the wheel tabs
        self.tabs = QTabWidget()
        self.tabs.setUsesScrollButtons(False)
        self.tabs.setDocumentMode(True)
        self.tabs.tabBar().setExpanding(True)
        self.tabs.setStyleSheet(
            "QTabWidget::pane { border: 1px solid #cccccc; border-radius: 4px; border-top-left-radius: 0px; top: -1px; background: #ffffff; }"
            "QTabBar::tab { font-size: 11px; font-weight: normal; padding: 4px 0px; margin: 0px; border: 1px solid #cccccc; border-bottom: none;"
            " border-top-left-radius: 4px; border-top-right-radius: 4px; background: #ececec; color: #444444; }"
            "QTabBar::tab:!first { margin-left: -1px; }"
            "QTabBar::tab:selected { background: #ffffff; color: #000000; font-weight: bold; }"
            "QTabBar::tab:hover:!selected { background: #f6f6f6; }"
        )
        layout.addWidget(self.tabs)

        self.sliders: Dict[str, List[ResetSlider]] = {kind: [] for kind in KINDS}
        self.value_labels: Dict[str, List[QLabel]] = {kind: [] for kind in KINDS}
        for kind in KINDS:
            page = QWidget()
            page_layout = QVBoxLayout(page)
            page_layout.setContentsMargins(6, 6, 6, 6)
            page_layout.setSpacing(2)
            for k, band in enumerate(BAND_NAMES):
                r, g, b = np.rint(band_color(CENTRES[k])[0] * 255).astype(int)
                name_lbl = QLabel(band)
                name_lbl.setStyleSheet(f"font-size: 11px; font-weight: normal; border: none; border-left: 6px solid rgb({r}, {g}, {b}); padding-left: 3px;")
                value_lbl = QLabel("+0")
                value_lbl.setStyleSheet("font-size: 11px; font-weight: normal; border: none;")
                value_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                header = QHBoxLayout()
                header.addWidget(name_lbl)
                header.addWidget(value_lbl)
                page_layout.addLayout(header)

                slider = ResetSlider(Qt.Orientation.Horizontal)
                slider.setRange(-100, 100)
                slider.setValue(0)
                slider.setStyleSheet(track_stylesheet(track_colors(kind, k)))
                slider.setToolTip(f"Double-click to reset {band} {KIND_TITLES[kind]} to 0")
                slider.valueChanged.connect(lambda value, lbl=value_lbl: lbl.setText(f"{value:+d}"))
                slider.valueChanged.connect(on_changed_slot)
                slider.sliderReleased.connect(on_released_slot)
                slider.doubleClicked.connect(lambda s=slider: self.reset_slider(s))
                page_layout.addWidget(slider)
                self.sliders[kind].append(slider)
                self.value_labels[kind].append(value_lbl)
            self.tabs.addTab(page, KIND_TITLES[kind])

        self.reset_btn = QPushButton("Reset Mixer")
        self.reset_btn.setToolTip("Reset all 24 Color Mixer sliders to 0")
        self.reset_btn.clicked.connect(self.reset_all)
        layout.addWidget(self.reset_btn)

    def reset_slider(self, slider: QSlider) -> None:
        """Sets slider to 0 and records an undo step."""
        slider.setValue(0)
        self.on_released_slot()

    def reset_all(self) -> None:
        """Sets every mixer slider to 0 and records one undo step."""
        for kind in KINDS:
            for slider in self.sliders[kind]:
                slider.setValue(0)
        self.on_released_slot()

    def is_dragging(self) -> bool:
        """True while the user holds any mixer slider."""
        return any(slider.isSliderDown() for kind in KINDS for slider in self.sliders[kind])

    def get_state(self) -> dict:
        """The mixer snapshot {STATE_KEY: {hue, sat, lum: eight ints -100..100 each, Red to Magenta}}."""
        return {self.STATE_KEY: {kind: [slider.value() for slider in self.sliders[kind]] for kind in KINDS}}

    def restore_state(self, full_state_dict: dict) -> None:
        """Sets the sliders and value labels from a snapshot without emitting signals; snapshots without the mixer are ignored."""
        if self.STATE_KEY in full_state_dict:
            state = full_state_dict[self.STATE_KEY]
            for kind in KINDS:
                for slider, value_lbl, value in zip(self.sliders[kind], self.value_labels[kind], state[kind]):
                    slider.blockSignals(True)
                    slider.setValue(int(value))
                    value_lbl.setText(f"{int(value):+d}")
                    slider.blockSignals(False)


class MixerCache:
    """The last ColorMixer result of each preview image, so grading-only changes and the 3x3 grid reuse it instead of re-running the mixer."""

    def __init__(self) -> None:
        """An empty cache; the lock lets the preview and grid threads share it (the second caller waits, then gets the cached result)."""
        self.mixer = ColorMixer()
        self.lock = threading.Lock()
        self.entries: Dict[int, tuple] = {}         # id(image) -> (image, levels, mixed); holding the image keeps its id from being reused

    def mixed(self, img: np.ndarray, icc: bytes, levels: Levels) -> np.ndarray:
        """img (encoded in the profile icc) through the mixer at levels (hue, sat, lum)."""
        with self.lock:
            entry = self.entries.get(id(img))
            if entry is not None and entry[0] is img and entry[1] == levels:
                return entry[2]
            out = self.mixer.apply(img, icc, *levels)
            self.entries[id(img)] = (img, levels, out)
            return out

    def clear(self) -> None:
        """Forgets every cached image (call when a new file is opened)."""
        with self.lock:
            self.entries.clear()
