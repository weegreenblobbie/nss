"""The Color Grading Explorer main window: wheel tabs, master controls, explore grid, file IO and the worker threads that grade off the GUI thread."""
import numpy as np
from typing import Callable, List, Optional, Literal
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QPointF, QTimer
from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QLabel,
    QFrame,
    QToolBar,
    QStatusBar,
    QFileDialog,
    QHBoxLayout,
    QVBoxLayout,
    QSlider,
    QTabWidget,
    QColorDialog,
    QMenu,
    QDialog,
    QLineEdit,
    QDialogButtonBox,
    QPushButton,
    QCheckBox,
    QStackedWidget,
    QGroupBox,
    QScrollArea,
    QToolButton,
)
from PyQt6.QtGui import (
    QMouseEvent,
    QAction,
    QImage,
    QPainter,
    QConicalGradient,
    QRadialGradient,
    QPen,
    QDrag,
    QIntValidator,
    QCloseEvent,
)

from nss.tiff_file import TiffFile
from nss.image_utils import downsample_image
from nss.color_math import (
    StateNode,
    MutationAxis,
    create_default_state,
    generate_mutations,
    ShiftColors,
    shift_colors,
    random_shift,
    explore_states,
)
from nss.color_grading.grading import Grading
from nss.mru import MruManager

from nss.color_gui.color_wheel import ColorWheel, ResetLabel
from nss.color_gui.wheel_lut import hue_groove_stylesheet
from nss.color_gui.explore_widget import ExploreWidget
from nss.color_gui.state_manager import StateManager
from nss.color_gui.grading_state import grade, grading_kwargs
from nss.color_gui.presets import save_preset, load_preset
from nss.color_gui.mixer_panel import MixerPanel, MixerCache, mixer_levels, KINDS
from nss.color_gui.harmony_panel import HarmonyPanel, wheel_angle
from nss.color_gui.sample_view import SampleView
from nss.color_harmony import median_rgb
from plugin import lr_bridge
from nss.color_mixer.hue_remap import native_hue

RENDER_DEBOUNCE_MS = 150    # a slider drag re-grades the preview once it pauses this long
DRAG_PREVIEW_DIM = 600      # max dimension of the preview rendered while a Color Mixer slider is held; release renders the 1200 px proxy
# Lightroom wheel name, grading_kwargs prefix, StateNode prefix
LIGHTROOM_WHEELS = (("shadows", "sh", "shadow"), ("midtones", "mi", "midtone"), ("highlights", "hi", "highlight"), ("global", "gl", "global"))


class ZoneTabWidget(QWidget):
    """
    Base class for a stateful Zone tab widget.
    Enforces Memento state management with a unique STATE_KEY.
    """
    STATE_KEY = "base_zone"  # Subclasses must override this!
    
    def __init__(self, prefix: str, default_h: int, on_changed_slot: Callable[[], None], parent: Optional[QWidget] = None) -> None:
        """Builds the wheel and the Hue / Sat / Luma sliders; slider changes call on_changed_slot."""
        super().__init__(parent)
        self.prefix = prefix.lower()
        self.default_h = default_h
        self.on_changed_slot = on_changed_slot
        
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(10, 10, 10, 10)
        self.layout.setSpacing(5)
        
        self.wheel = ColorWheel()
        wheel_container = QHBoxLayout()
        wheel_container.addStretch()
        wheel_container.addWidget(self.wheel)
        wheel_container.addStretch()
        self.layout.addLayout(wheel_container)
        
        self.hue_lbl = ResetLabel(f"Hue: {default_h}°")
        self.hue_lbl.setStyleSheet("font-size: 11px;")
        self.hue_lbl.setToolTip("Double-click to reset Hue to default")
        self.hue_slider = QSlider(Qt.Orientation.Horizontal)
        self.hue_slider.setRange(0, 360)
        self.hue_slider.setValue(default_h)
        self.hue_slider.setStyleSheet(hue_groove_stylesheet())      # pure hue HSV (h, 1, 1) along the slider, as Lightroom shows a hue
        
        self.sat_lbl = ResetLabel("Sat: 0.00")
        self.sat_lbl.setStyleSheet("font-size: 11px;")
        self.sat_lbl.setToolTip("Double-click to reset Saturation to 0.00")
        self.sat_slider = QSlider(Qt.Orientation.Horizontal)
        self.sat_slider.setRange(0, 100)
        self.sat_slider.setValue(0)
        
        self.light_lbl = ResetLabel("Luma: 0.00")
        self.light_lbl.setStyleSheet("font-size: 11px;")
        self.light_lbl.setToolTip("Double-click to reset Luminance to 0.00")
        self.light_slider = QSlider(Qt.Orientation.Horizontal)
        self.light_slider.setRange(-100, 100)
        self.light_slider.setValue(0)
        
        self.layout.addWidget(self.hue_lbl)
        self.layout.addWidget(self.hue_slider)
        self.layout.addWidget(self.sat_lbl)
        self.layout.addWidget(self.sat_slider)
        self.layout.addWidget(self.light_lbl)
        self.layout.addWidget(self.light_slider)
        
        self.layout.addStretch()
        
        # Wire signals
        self.hue_slider.valueChanged.connect(self.on_changed_slot)
        self.sat_slider.valueChanged.connect(self.on_changed_slot)
        self.light_slider.valueChanged.connect(self.on_changed_slot)
        
        self.hue_slider.sliderReleased.connect(self.parent_slider_released)
        self.sat_slider.sliderReleased.connect(self.parent_slider_released)
        self.light_slider.sliderReleased.connect(self.parent_slider_released)
        self.wheel.interactionFinished.connect(self.parent_wheel_released)

    def parent_slider_released(self) -> None:
        """Records an undo step on the main window when a slider is released."""
        parent_mw = self.window()
        if parent_mw and hasattr(parent_mw, "on_discrete_action"):
            parent_mw.on_discrete_action()

    def parent_wheel_released(self, h: float, s: float) -> None:
        """Records an undo step on the main window when a wheel drag ends."""
        parent_mw = self.window()
        if parent_mw and hasattr(parent_mw, "on_discrete_action"):
            parent_mw.on_discrete_action()

    def get_state(self) -> dict:
        """This zone's snapshot {STATE_KEY: {hue, sat 0..1, light -1..1}}."""
        return {
            self.STATE_KEY: {
                "hue": float(self.hue_slider.value()),
                "sat": self.sat_slider.value() / 100.0,
                "light": self.light_slider.value() / 100.0,
            }
        }
        
    def restore_state(self, full_state_dict: dict) -> None:
        """Sets the wheel, sliders and labels from a snapshot without emitting signals; snapshots without this zone are ignored."""
        if self.STATE_KEY in full_state_dict:
            state = full_state_dict[self.STATE_KEY]
            self.wheel.blockSignals(True)
            self.sat_slider.blockSignals(True)
            self.light_slider.blockSignals(True)
            self.hue_slider.blockSignals(True)
            
            self.wheel.set_color(state["hue"], state["sat"])
            self.hue_slider.setValue(int(state["hue"]))
            self.sat_slider.setValue(int(state["sat"] * 100.0))
            self.light_slider.setValue(int(state["light"] * 100.0))
            
            # Update labels
            self.hue_lbl.setText(f"Hue: {state['hue']:.0f}°")
            self.sat_lbl.setText(f"Sat: {state['sat']:.2f}")
            self.light_lbl.setText(f"Luma: {state['light']:+.2f}")
            
            self.wheel.blockSignals(False)
            self.sat_slider.blockSignals(False)
            self.light_slider.blockSignals(False)
            self.hue_slider.blockSignals(False)


class ShadowsZoneWidget(ZoneTabWidget):
    """The Shadows tab."""
    STATE_KEY = "shadows_zone"


class MidtonesZoneWidget(ZoneTabWidget):
    """The Midtones tab."""
    STATE_KEY = "midtones_zone"


class HighlightsZoneWidget(ZoneTabWidget):
    """The Highlights tab."""
    STATE_KEY = "highlights_zone"


class GlobalZoneWidget(ZoneTabWidget):
    """The Global tab (the whole tonal range)."""
    STATE_KEY = "global_zone"


class MasterZoneWidget(QWidget):
    """Master controls: Blending, Balance, Rotation and Reset Grading."""
    STATE_KEY = "master_zone"
    
    def __init__(self, on_changed_slot: Callable[[], None], parent: Optional[QWidget] = None) -> None:
        """Builds the master sliders and the Reset Grading button; slider changes call on_changed_slot."""
        super().__init__(parent)
        self.on_changed_slot = on_changed_slot
        
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(10)
        
        # Master Blending
        self.blending_lbl = ResetLabel("Blending: 0.50")
        self.blending_lbl.setStyleSheet("font-size: 11px;")
        self.blending_lbl.setToolTip("Double-click to reset Blending to 0.50")
        self.blending_slider = QSlider(Qt.Orientation.Horizontal)
        self.blending_slider.setRange(0, 100)
        self.blending_slider.setValue(50)
        
        # Master Balance
        self.balance_lbl = ResetLabel("Balance: 0.00")
        self.balance_lbl.setStyleSheet("font-size: 11px;")
        self.balance_lbl.setToolTip("Double-click to reset Balance to 0.00")
        self.balance_slider = QSlider(Qt.Orientation.Horizontal)
        self.balance_slider.setRange(-100, 100)
        self.balance_slider.setValue(0)

        # Global Rotation
        self.rotation_lbl = ResetLabel("Rotation: 0°")
        self.rotation_lbl.setStyleSheet("font-size: 11px;")
        self.rotation_lbl.setToolTip("Double-click to reset Rotation to 0°")
        self.rotation_slider = QSlider(Qt.Orientation.Horizontal)
        self.rotation_slider.setRange(-180, 180)
        self.rotation_slider.setValue(0)

        self.layout.addWidget(self.blending_lbl)
        self.layout.addWidget(self.blending_slider)
        self.layout.addWidget(self.balance_lbl)
        self.layout.addWidget(self.balance_slider)
        self.layout.addWidget(self.rotation_lbl)
        self.layout.addWidget(self.rotation_slider)

        self.reset_all_btn = QPushButton("Reset Grading")
        self.reset_all_btn.setToolTip("Reset every wheel (Global included), Blending, Balance and Rotation to their defaults")
        self.layout.addWidget(self.reset_all_btn)

        self.blending_slider.valueChanged.connect(self.on_changed_slot)
        self.balance_slider.valueChanged.connect(self.on_changed_slot)
        self.rotation_slider.valueChanged.connect(self.on_changed_slot)
        
        # Release trigger for Memento
        self.blending_slider.sliderReleased.connect(self.parent_slider_released)
        self.balance_slider.sliderReleased.connect(self.parent_slider_released)
        self.rotation_slider.sliderReleased.connect(self.parent_slider_released)
        
        # Double click to reset
        self.blending_lbl.doubleClicked.connect(lambda: self.reset_slider(self.blending_slider, 50))
        self.balance_lbl.doubleClicked.connect(lambda: self.reset_slider(self.balance_slider, 0))
        self.rotation_lbl.doubleClicked.connect(lambda: self.reset_slider(self.rotation_slider, 0))

    def reset_slider(self, slider: QSlider, value: int) -> None:
        """Sets slider to value and records an undo step."""
        slider.setValue(value)
        self.parent_slider_released()

    def parent_slider_released(self) -> None:
        """Records an undo step on the main window when a slider is released."""
        parent_mw = self.window()
        if parent_mw and hasattr(parent_mw, "on_discrete_action"):
            parent_mw.on_discrete_action()

    def get_state(self) -> dict:
        """The master snapshot {STATE_KEY: {blending 0..1, balance -1..1, rotation in degrees}}."""
        return {
            self.STATE_KEY: {
                "blending": self.blending_slider.value() / 100.0,
                "balance": self.balance_slider.value() / 100.0,
                "rotation": float(self.rotation_slider.value()),
            }
        }
        
    def restore_state(self, full_state_dict: dict) -> None:
        """Sets the master sliders and labels from a snapshot without emitting signals."""
        if self.STATE_KEY in full_state_dict:
            state = full_state_dict[self.STATE_KEY]
            self.blending_slider.blockSignals(True)
            self.balance_slider.blockSignals(True)
            self.rotation_slider.blockSignals(True)
            
            self.blending_slider.setValue(int(state["blending"] * 100.0))
            self.balance_slider.setValue(int(state["balance"] * 100.0))
            self.rotation_slider.setValue(int(state.get("rotation", 0.0)))
            
            # Update labels
            self.blending_lbl.setText(f"Blending: {state['blending']:.2f}")
            self.balance_lbl.setText(f"Balance: {state['balance']:+.2f}")
            self.rotation_lbl.setText(f"Rotation: {state.get('rotation', 0.0):.0f}°")
            
            self.blending_slider.blockSignals(False)
            self.balance_slider.blockSignals(False)
            self.rotation_slider.blockSignals(False)


class GradingWorker(QThread):
    """
    Background worker thread to compute color grading for mutations asynchronously
    to prevent blocking the main PyQt6 GUI event thread.
    """
    progress = pyqtSignal(int, np.ndarray)
    finished_all = pyqtSignal()

    def __init__(self, working_image: np.ndarray, states: List[Optional[StateNode]], grade_fn: Callable[[np.ndarray, StateNode], np.ndarray], skip_center: bool = False, parent: Optional[QWidget] = None) -> None:
        """Grades up to 9 states of working_image with grade_fn; skip_center leaves index 4 out and None states are left out."""
        super().__init__(parent)
        self.working_image = working_image
        self.states = states
        self.grade_fn = grade_fn        # (image, state) -> graded image
        self.skip_center = skip_center
        self.is_cancelled = False

    def cancel(self) -> None:
        """Stops before the next state; the one being graded still finishes."""
        self.is_cancelled = True

    def run(self) -> None:
        """Grades the centre first, then the outer states, emitting progress(index, image) for each."""
        if self.skip_center:
            indices = [i for i in range(9) if i != 4]
        else:
            indices = [4] + [i for i in range(9) if i != 4]

        for i in indices:
            if self.is_cancelled:
                return
            if i >= len(self.states) or self.states[i] is None:
                continue
            try:
                graded = self.grade_fn(self.working_image, self.states[i])
                if self.is_cancelled:
                    return
                self.progress.emit(i, graded)
            except Exception:
                pass
        self.finished_all.emit()


class PreviewWorker(QThread):
    """Grades the single preview image off the GUI thread."""
    graded = pyqtSignal(np.ndarray)
    failed = pyqtSignal(str)

    def __init__(self, grade_fn: Callable[[np.ndarray, StateNode], np.ndarray], image: np.ndarray, state: StateNode, parent: Optional[QWidget] = None) -> None:
        """Grades `image` with `state` when started."""
        super().__init__(parent)
        self.grade_fn = grade_fn        # (image, state) -> graded image
        self.image = image
        self.state = state

    def run(self) -> None:
        """Emits graded(image), or failed(message) if grading raises."""
        try:
            self.graded.emit(self.grade_fn(self.image, self.state))
        except Exception as e:
            self.failed.emit(str(e))


class SaveWorker(QThread):
    """Grades the full-resolution image and writes it, off the GUI thread."""
    saved = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, grade_fn: Callable[[np.ndarray, StateNode], np.ndarray], image: np.ndarray, state: StateNode, tiff: TiffFile, path: str, parent: Optional[QWidget] = None) -> None:
        """Grades `image` with `state` and writes it to `path` through `tiff` when started."""
        super().__init__(parent)
        self.grade_fn = grade_fn
        self.image = image
        self.state = state
        self.tiff = tiff                # written with its own bit depth and ICC profile
        self.path = path

    def run(self) -> None:
        """Emits saved(path), or failed(message) if grading or writing raises."""
        try:
            self.tiff.array = self.grade_fn(self.image, self.state)
            self.tiff.saveas(self.path)
            self.saved.emit(self.path)
        except Exception as e:
            self.failed.emit(str(e))


class MainWindow(QMainWindow):
    """
    The Main Window for the 16-bit Color Grading Explorer with Memento State Management & Explore Mode.
    """
    def __init__(self) -> None:
        """Creates the managers, the grading engine, the debounce timer and the UI, and registers the stateful widgets."""
        super().__init__()
        self.setWindowTitle("16-bit Color Grading Explorer")
        self.resize(1300, 850)

        # Core Managers
        self.state_manager = StateManager()
        self.mru_manager = MruManager()

        # State Variables
        self.master_image: Optional[np.ndarray] = None
        self.proxy_image: Optional[np.ndarray] = None
        self.drag_proxy: Optional[np.ndarray] = None
        self.tiff_obj: Optional[TiffFile] = None
        self.grid_states: Optional[List[Optional[StateNode]]] = None     # None cells: Explore Mode could not shift
        self.rng = np.random.default_rng()                                 # Random Shift and the explore grid draw from it
        self.grading_worker: Optional[GradingWorker] = None
        self.preview_worker: Optional[PreviewWorker] = None
        self.pending_preview_state: Optional[StateNode] = None       # newest request made while a preview was being graded
        self.save_worker: Optional[SaveWorker] = None
        self.grading = Grading()
        self.mixer_cache = MixerCache()
        self.lightroom_request: Optional[lr_bridge.Request] = None   # set by open_lightroom_request
        self.lightroom_applied = False                                 # True once Apply to Lightroom wrote the result file

        # Slider drags restart this timer; the preview is re-graded when it fires
        self.render_timer = QTimer(self)
        self.render_timer.setSingleShot(True)
        self.render_timer.setInterval(RENDER_DEBOUNCE_MS)
        self.render_timer.timeout.connect(self.render_center_preview)

        # Setup UI
        self.init_ui()

        # Enforce unique state widget registrations
        self.state_manager.register_widget(self.sh_zone)
        self.state_manager.register_widget(self.mid_zone)
        self.state_manager.register_widget(self.hi_zone)
        self.state_manager.register_widget(self.gl_zone)
        self.state_manager.register_widget(self.master_zone)
        self.state_manager.register_widget(self.explore_widget)
        self.state_manager.register_widget(self.mixer_panel)
        self.state_manager.register_widget(self.harmony_panel)

    def init_ui(self) -> None:
        """Builds the menus, the undo / redo corner, the preview / explore views and the right-hand panel."""
        # Create Menu Bar
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("&File")

        open_action = QAction("&Open...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.setStatusTip("Open a 16-bit TIFF image")
        open_action.triggered.connect(lambda: self.open_file())
        file_menu.addAction(open_action)

        self.recent_menu = file_menu.addMenu("Recent &Directories")
        self.update_recent_directories_menu()

        save_action = QAction("&Save As...", self)
        save_action.setShortcut("Ctrl+S")
        save_action.setStatusTip("Save the active color graded image")
        save_action.setEnabled(False)
        save_action.triggered.connect(self.save_file)
        self.save_action = save_action
        file_menu.addAction(save_action)

        file_menu.addSeparator()

        save_preset_action = QAction("Save &Preset...", self)
        save_preset_action.setStatusTip("Save the current grading settings to a JSON preset")
        save_preset_action.triggered.connect(self.on_save_preset)
        file_menu.addAction(save_preset_action)

        load_preset_action = QAction("&Load Preset...", self)
        load_preset_action.setStatusTip("Load grading settings from a JSON preset")
        load_preset_action.triggered.connect(self.on_load_preset)
        file_menu.addAction(load_preset_action)

        file_menu.addSeparator()

        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Alt+F4")
        exit_action.setStatusTip("Exit the application")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Memento Undo/Redo Controls as Actions (for shortcuts & tracking)
        self.back_action = QAction("Back (Undo)", self)
        self.back_action.setShortcut("Ctrl+Z")
        self.back_action.setEnabled(False)
        self.back_action.triggered.connect(self.on_undo_clicked)

        self.forward_action = QAction("Forward (Redo)", self)
        self.forward_action.setShortcut("Ctrl+Y")
        self.forward_action.setEnabled(False)
        self.forward_action.triggered.connect(self.on_redo_clicked)

        # Create container for MenuBar corner widget
        corner_widget = QWidget()
        corner_layout = QHBoxLayout(corner_widget)
        corner_layout.setContentsMargins(0, 0, 10, 0)
        corner_layout.setSpacing(5)

        self.back_btn = QToolButton(self)
        self.back_btn.setDefaultAction(self.back_action)
        corner_layout.addWidget(self.back_btn)

        self.history_label = QLabel("Step: 0 of 0")
        self.history_label.setStyleSheet(
            "font-weight: bold; margin-left: 10px; margin-right: 10px;"
        )
        corner_layout.addWidget(self.history_label)

        self.forward_btn = QToolButton(self)
        self.forward_btn.setDefaultAction(self.forward_action)
        corner_layout.addWidget(self.forward_btn)

        menu_bar.setCornerWidget(corner_widget, Qt.Corner.TopRightCorner)

        # 2. Central Widget Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(15)

        # Left Side View Stack (Normal Mode single preview vs Explore Mode 3x3)
        self.view_stack = QStackedWidget()
        main_layout.addWidget(self.view_stack, stretch=4)

        # Normal View (Page 0)
        self.single_preview = SampleView(self)
        self.view_stack.addWidget(self.single_preview)

        # Explore Mode Grid View (Page 1)
        self.explore_widget = ExploreWidget(self)
        self.explore_widget.container_clicked.connect(self.on_explore_container_clicked)
        self.explore_widget.container_edit_requested.connect(self.on_explore_edit_requested)
        self.explore_widget.variation_changed.connect(self.on_explore_variation_changed)
        self.explore_widget.var_slider.sliderReleased.connect(self.on_discrete_action)
        self.view_stack.addWidget(self.explore_widget)

        # Right Side: Manual Grading & Harmony Randomizer Scroll Area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setFixedWidth(300)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)

        self.inspector_panel = QFrame()
        self.inspector_panel.setFrameShape(QFrame.Shape.StyledPanel)
        self.inspector_panel.setFrameShadow(QFrame.Shadow.Raised)
        self.inspector_panel.setStyleSheet(
            "border: 1px solid #cccccc; border-radius: 4px;"
        )
        inspector_layout = QVBoxLayout(self.inspector_panel)
        inspector_layout.setContentsMargins(15, 15, 15, 15)
        inspector_layout.setSpacing(12)
        scroll_area.setWidget(self.inspector_panel)

        main_layout.addWidget(scroll_area, stretch=1)

        # Harmony Randomizer / Explore Mode Section
        harmony_box = QGroupBox("EXPLORE & HARMONY")
        harmony_box.setStyleSheet(
            "QGroupBox { font-weight: bold; border: 1px solid #cccccc; border-radius: 4px; margin-top: 10px; padding: 10px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 3px; }"
        )
        harmony_layout = QVBoxLayout(harmony_box)
        harmony_layout.setSpacing(8)

        # What Random Shift may change
        targets_row = QHBoxLayout()
        targets_row.addWidget(self.explore_widget.mixer_checkbox)
        targets_row.addWidget(self.explore_widget.grading_checkbox)
        harmony_layout.addLayout(targets_row)
        self.explore_widget.mixer_checkbox.stateChanged.connect(self.update_random_shift_enabled)
        self.explore_widget.grading_checkbox.stateChanged.connect(self.update_random_shift_enabled)

        # Explore Toggle
        self.explore_checkbox = QCheckBox("Activate Explore Mode")
        self.explore_checkbox.setStyleSheet("font-weight: bold;")
        self.explore_checkbox.stateChanged.connect(self.on_explore_mode_toggled)
        harmony_layout.addWidget(self.explore_checkbox)

        # Randomize Button
        self.randomize_btn = QPushButton("Random Shift")
        self.randomize_btn.setStyleSheet("background-color: #2b5b2b; color: white; font-weight: bold; padding: 4px;")
        self.randomize_btn.clicked.connect(self.on_randomize_clicked)
        harmony_layout.addWidget(self.randomize_btn)

        # Variation Strength Slider positioned below Generate button
        harmony_layout.addWidget(self.explore_widget.var_label)
        harmony_layout.addWidget(self.explore_widget.var_slider)

        inspector_layout.addWidget(harmony_box)

        # Color Harmony: eyedropper samples of the preview against harmony wedges warped by the Color Mixer's Hue sliders
        self.harmony_panel = HarmonyPanel(self.on_discrete_action, self.displayed_rgb, self)
        self.harmony_panel.sampling_changed.connect(self.single_preview.set_sampling)
        self.harmony_panel.samples_changed.connect(self.on_harmony_samples_changed)
        self.single_preview.sample_picked.connect(self.on_sample_picked)
        self.single_preview.sample_clicked.connect(self.harmony_panel.show_sample_info)
        self.single_preview.sample_menu_requested.connect(self.harmony_panel.show_sample_menu)
        inspector_layout.addWidget(self.harmony_panel)
        self.update_random_shift_enabled()

        # Color Mixer (applied before the grade), above the wheel tabs
        self.mixer_panel = MixerPanel(self.on_manual_slider_changed, self.on_mixer_released, self)
        inspector_layout.addWidget(self.mixer_panel)

        # Manual 3-Way Tabs Setup
        self.tabs = QTabWidget()
        # Four equal-width wheel tabs across the 300 px panel (document mode stretches the bar), drawn in the panel's light-grey border style
        self.tabs.setUsesScrollButtons(False)
        self.tabs.setDocumentMode(True)
        self.tabs.tabBar().setExpanding(True)
        self.tabs.setStyleSheet(
            "QTabWidget::pane { border: 1px solid #cccccc; border-radius: 4px; border-top-left-radius: 0px; top: -1px; background: #ffffff; }"
            "QTabBar::tab { font-size: 11px; padding: 4px 0px; margin: 0px; border: 1px solid #cccccc; border-bottom: none;"
            " border-top-left-radius: 4px; border-top-right-radius: 4px; background: #ececec; color: #444444; }"
            "QTabBar::tab:!first { margin-left: -1px; }"
            "QTabBar::tab:selected { background: #ffffff; color: #000000; font-weight: bold; }"
            "QTabBar::tab:hover:!selected { background: #f6f6f6; }"
        )
        self.tabs.tabBarClicked.connect(self.on_tab_bar_clicked)
        self.show_all_tones = False

        # Color Grading group: the wheel tabs and the master controls, framed like the Color Mixer
        self.grading_box = QGroupBox("COLOR GRADING")
        self.grading_box.setStyleSheet(
            "QGroupBox { font-weight: bold; border: 1px solid #cccccc; border-radius: 4px; margin-top: 10px; padding: 10px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 3px; }"
        )
        grading_layout = QVBoxLayout(self.grading_box)
        grading_layout.setSpacing(12)
        grading_layout.addWidget(self.tabs)
        inspector_layout.addWidget(self.grading_box)

        # Stateful tab widgets
        self.sh_zone = ShadowsZoneWidget("Shadows", 240, self.on_manual_slider_changed, self)
        self.tabs.addTab(self.sh_zone, "Shadows")
        self.sh_zone.wheel.colorChanged.connect(self.on_sh_wheel_changed)
        self.sh_zone.hue_lbl.doubleClicked.connect(lambda: self.reset_hue(self.sh_zone, 240))
        self.sh_zone.sat_lbl.doubleClicked.connect(lambda: self.reset_slider(self.sh_zone.sat_slider, 0))
        self.sh_zone.light_lbl.doubleClicked.connect(lambda: self.reset_slider(self.sh_zone.light_slider, 0))

        self.mid_zone = MidtonesZoneWidget("Midtones", 120, self.on_manual_slider_changed, self)
        self.tabs.addTab(self.mid_zone, "Midtones")
        self.mid_zone.wheel.colorChanged.connect(self.on_mid_wheel_changed)
        self.mid_zone.hue_lbl.doubleClicked.connect(lambda: self.reset_hue(self.mid_zone, 120))
        self.mid_zone.sat_lbl.doubleClicked.connect(lambda: self.reset_slider(self.mid_zone.sat_slider, 0))
        self.mid_zone.light_lbl.doubleClicked.connect(lambda: self.reset_slider(self.mid_zone.light_slider, 0))

        self.hi_zone = HighlightsZoneWidget("Highlights", 60, self.on_manual_slider_changed, self)
        self.tabs.addTab(self.hi_zone, "Highlights")
        self.hi_zone.wheel.colorChanged.connect(self.on_hi_wheel_changed)
        self.hi_zone.hue_lbl.doubleClicked.connect(lambda: self.reset_hue(self.hi_zone, 60))
        self.hi_zone.sat_lbl.doubleClicked.connect(lambda: self.reset_slider(self.hi_zone.sat_slider, 0))
        self.hi_zone.light_lbl.doubleClicked.connect(lambda: self.reset_slider(self.hi_zone.light_slider, 0))

        self.gl_zone = GlobalZoneWidget("Global", 0, self.on_manual_slider_changed, self)
        self.tabs.addTab(self.gl_zone, "Global")
        self.gl_zone.wheel.colorChanged.connect(self.on_gl_wheel_changed)
        self.gl_zone.hue_lbl.doubleClicked.connect(lambda: self.reset_hue(self.gl_zone, 0))
        self.gl_zone.sat_lbl.doubleClicked.connect(lambda: self.reset_slider(self.gl_zone.sat_slider, 0))
        self.gl_zone.light_lbl.doubleClicked.connect(lambda: self.reset_slider(self.gl_zone.light_slider, 0))

        # Master Global Controls Section
        self.master_zone = MasterZoneWidget(self.on_manual_slider_changed, self)
        self.master_zone.reset_all_btn.clicked.connect(self.on_reset_all)
        grading_layout.addWidget(self.master_zone)
        inspector_layout.addStretch()

        self.tabs.currentChanged.connect(self.on_manual_slider_changed)
        self.statusBar().showMessage("Ready. Load a 16-bit TIFF image to begin.")

    # ----------------------------------------------------
    # Memento System Integration
    # ----------------------------------------------------
    def on_discrete_action(self) -> None:
        """
        Gathers snapshots on discrete events and pushes onto undo_stack.
        """
        snapshot = self.state_manager.capture_snapshot()
        
        # Keep a constant step size default for backwards compat
        snapshot["step_size"] = 0.2
        
        # Avoid duplicate pushes
        if self.state_manager.undo_stack and self.state_manager.undo_stack[-1] == snapshot:
            return
            
        self.state_manager.push_state(snapshot)
        self.update_memento_navigation_ui()
        
        # Re-render everything from new state!
        self.render_view_from_memento(snapshot)

    def on_mixer_released(self) -> None:
        """A Color Mixer slider was released or reset: records an undo step and renders the 1200 px proxy, replacing the drag preview."""
        self.render_timer.start()       # renders when the snapshot equals the last one; otherwise on_discrete_action renders now and stops the timer
        self.on_discrete_action()

    def on_undo_clicked(self) -> None:
        """Steps back one snapshot, restores the widgets and re-renders."""
        if self.state_manager.can_undo():
            snapshot = self.state_manager.undo()
            if snapshot:
                self.state_manager.restore_snapshot(snapshot)
                self.on_manual_slider_changed()
                
                self.render_view_from_memento(snapshot)
                self.update_memento_navigation_ui()
                self.statusBar().showMessage("Undo performed.")

    def on_redo_clicked(self) -> None:
        """Steps forward one snapshot, restores the widgets and re-renders."""
        if self.state_manager.can_redo():
            snapshot = self.state_manager.redo()
            if snapshot:
                self.state_manager.restore_snapshot(snapshot)
                self.on_manual_slider_changed()
                
                self.render_view_from_memento(snapshot)
                self.update_memento_navigation_ui()
                self.statusBar().showMessage("Redo performed.")

    def update_memento_navigation_ui(self) -> None:
        """Enables Back / Forward / Save and shows the history position."""
        self.back_action.setEnabled(self.state_manager.can_undo())
        self.forward_action.setEnabled(self.state_manager.can_redo())
        self.save_action.setEnabled(True)
        
        undo_len = len(self.state_manager.undo_stack)
        redo_len = len(self.state_manager.redo_stack)
        total = undo_len + redo_len
        self.history_label.setText(f"Step: {undo_len} of {total}")

    def rebuild_state_node_from_memento(self, snapshot: dict) -> StateNode:
        """
        Translates a Memento Snapshot back to a legacy StateNode dictionary for math operations.
        """
        state = create_default_state()

        state["step_size"] = snapshot.get("step_size", 0.2)

        # Shadows
        sh_state = snapshot.get("shadows_zone", {})
        state["shadow_hue"] = sh_state.get("hue", 240.0)
        state["shadow_sat"] = sh_state.get("sat", 0.0)
        state["shadow_light"] = sh_state.get("light", 0.0)

        # Midtones
        mid_state = snapshot.get("midtones_zone", {})
        state["midtone_hue"] = mid_state.get("hue", 120.0)
        state["midtone_sat"] = mid_state.get("sat", 0.0)
        state["midtone_light"] = mid_state.get("light", 0.0)

        # Highlights
        hi_state = snapshot.get("highlights_zone", {})
        state["highlight_hue"] = hi_state.get("hue", 60.0)
        state["highlight_sat"] = hi_state.get("sat", 0.0)
        state["highlight_light"] = hi_state.get("light", 0.0)

        # Global (snapshots from before the Global wheel have no global_zone)
        gl_state = snapshot.get("global_zone", {})
        state["global_hue"] = gl_state.get("hue", 0.0)
        state["global_sat"] = gl_state.get("sat", 0.0)
        state["global_light"] = gl_state.get("light", 0.0)

        # Master
        m_state = snapshot.get("master_zone", {})
        state["blending"] = m_state.get("blending", 0.5)
        state["balance"] = m_state.get("balance", 0.0)
        state["rotation"] = m_state.get("rotation", 0.0)

        # Color Mixer (snapshots from before the mixer have no color_mixer)
        mixer_state = snapshot.get("color_mixer", {})
        for kind in KINDS:
            state[f"mixer_{kind}"] = [float(v) for v in mixer_state.get(kind, [0.0] * 8)]

        return state

    def apply_state_node_to_widgets(self, state: StateNode) -> None:
        """
        Applies a StateNode dict parameters to stateful widgets without triggering snapshot capture.
        """
        snapshot = {
            "shadows_zone": {"hue": state["shadow_hue"], "sat": state["shadow_sat"], "light": state["shadow_light"]},
            "midtones_zone": {"hue": state["midtone_hue"], "sat": state["midtone_sat"], "light": state["midtone_light"]},
            "highlights_zone": {"hue": state["highlight_hue"], "sat": state["highlight_sat"], "light": state["highlight_light"]},
            "global_zone": {"hue": state.get("global_hue", 0.0), "sat": state.get("global_sat", 0.0), "light": state.get("global_light", 0.0)},
            "master_zone": {"blending": state["blending"], "balance": state["balance"], "rotation": state.get("rotation", 0.0)},
            "explore_widget": {"variation_strength": self.explore_widget.variation_strength},
        }
        if "mixer_hue" in state:        # states from presets and Reset Grading carry no mixer values: the Color Mixer keeps its own
            snapshot["color_mixer"] = {kind: state[f"mixer_{kind}"] for kind in KINDS}
        self.state_manager.restore_snapshot(snapshot)
        self.on_manual_slider_changed()

    # ----------------------------------------------------
    # Rendering & Asynchronous Workers
    # ----------------------------------------------------
    def graded(self, img: np.ndarray, state: StateNode) -> np.ndarray:
        """img through the Color Mixer (cached per image) then graded by state, in the file's own ICC working space; returned unchanged when the file has no ICC profile (sRGB is never assumed)."""
        icc = None if self.tiff_obj is None else self.tiff_obj.iccprofile
        if icc is None:
            return img
        return grade(self.grading, self.mixer_cache.mixed(img, icc, mixer_levels(state)), icc, state)

    def graded_full(self, img: np.ndarray, state: StateNode) -> np.ndarray:
        """The full-resolution img through the Color Mixer (not cached, so no full-size copy is kept) then graded by state; needs the file's ICC profile."""
        icc = self.tiff_obj.iccprofile
        return grade(self.grading, self.mixer_cache.mixer.apply(img, icc, *mixer_levels(state)), icc, state)

    def preview_source(self) -> Optional[np.ndarray]:
        """The image the single preview renders: the small drag proxy while a Color Mixer slider is held, else the 1200 px proxy."""
        return self.drag_proxy if self.mixer_panel.is_dragging() else self.proxy_image

    def render_center_preview(self) -> None:
        """Debounced re-grade of the single preview from the current widget values."""
        if self.proxy_image is None:
            return
        state = self.rebuild_state_node_from_memento(self.state_manager.capture_snapshot())
        self.request_preview(state)

    def request_preview(self, state: StateNode) -> None:
        """Grades the single preview in a PreviewWorker.  While one is running, only the newest request is kept; it starts when that one finishes."""
        if self.proxy_image is None:
            return
        if self.tiff_obj is None or self.tiff_obj.iccprofile is None:     # nothing to grade: no thread needed
            self.single_preview.set_image(self.proxy_image)
            return
        if self.preview_worker is not None and self.preview_worker.isRunning():
            self.pending_preview_state = state
            return
        self.pending_preview_state = None
        self.preview_worker = PreviewWorker(self.graded,self.preview_source(), state, parent=self)
        self.preview_worker.graded.connect(self.single_preview.set_image)
        self.preview_worker.failed.connect(lambda msg: self.statusBar().showMessage(f"Error grading preview: {msg}"))
        self.preview_worker.finished.connect(self.on_preview_finished)
        self.preview_worker.start()

    def on_preview_finished(self) -> None:
        """Starts the newest preview request made while the last one was grading."""
        if self.pending_preview_state is not None:
            self.request_preview(self.pending_preview_state)

    def render_view_from_memento(self, snapshot: dict) -> None:
        """Re-renders the preview (and the 3x3 grid in Explore Mode) for a snapshot now, cancelling a pending debounce."""
        self.render_timer.stop()
        if self.proxy_image is None:
            return

        state = self.rebuild_state_node_from_memento(snapshot)

        # 1. Update Single Preview (Normal Mode)
        self.request_preview(state)
        
        # 2. Update 3x3 Look-Matrix (Explore Mode)
        if self.explore_widget.isVisible():
            self.start_explore_mutations_render(state)

    def start_explore_mutations_render(self, center_state: StateNode) -> None:
        """Cancels a running grid render, generates 8 random shifts around center_state and grades them in a GradingWorker.  When Random Shift is
        not ready only the centre is graded and the status bar says why."""
        if self.proxy_image is None:
            return
            
        # Cancel active background workers
        if self.grading_worker is not None:
            if self.grading_worker.isRunning():
                self.grading_worker.cancel()
                self.grading_worker.wait()
            self.grading_worker = None

        # Generate 8 random shifts around center
        explore = self.explore_widget
        if self.random_shift_ready():
            self.grid_states = explore_states(center_state, self.selected_colors(center_state), explore.variation_strength,
                                              explore.mixer_checkbox.isChecked(), explore.grading_checkbox.isChecked(), self.rng)
        else:
            self.grid_states = [center_state.copy() if i == 4 else None for i in range(9)]
            self.statusBar().showMessage("Explore Mode needs a colour sampled in Color Harmony and Mixer or Grading checked: showing the centre only.")

        # Explicitly clear the 8 outer image containers (excluding center index 4) to indicate a visual refresh
        for idx in range(9):
            if idx != 4:
                self.explore_widget.containers[idx].set_image(None)

        # Fire async worker
        self.grading_worker = GradingWorker(
            self.proxy_image,
            self.grid_states,
            self.graded,
            skip_center=False,
            parent=self
        )
        self.grading_worker.progress.connect(self.on_worker_progress)
        self.grading_worker.finished_all.connect(self.on_worker_finished)
        self.grading_worker.start()

    def on_worker_progress(self, index: int, arr: np.ndarray) -> None:
        """Shows a graded grid cell; the centre cell also updates the single preview."""
        self.explore_widget.containers[index].set_image(arr)
        if index == 4:
            self.single_preview.set_image(arr)

    def on_worker_finished(self) -> None:
        """Reports that the grid is done."""
        self.statusBar().showMessage("Ready.")

    # ----------------------------------------------------
    # Color Harmony Sampling
    # ----------------------------------------------------
    def displayed_rgb(self, x: int, y: int) -> Optional[np.ndarray]:
        """Median colour (3,) of the preview on screen around proxy pixel (x column, y row); None without a preview."""
        shown = self.single_preview.image_array
        if shown is None or self.proxy_image is None:
            return None
        sy, sx = shown.shape[0] / self.proxy_image.shape[0], shown.shape[1] / self.proxy_image.shape[1]     # the drag preview is smaller
        return median_rgb(shown, int(x * sx), int(y * sy))

    def on_sample_picked(self, u: float, v: float) -> None:
        """The eyedropper clicked the preview at image fractions (u, v): adds a sample with the displayed colour and the mixer-input hue read from
        the pre-mixer proxy at the same pixel (the grade after the mixer also moves hues, so the displayed hue cannot be mapped back through f)."""
        if self.proxy_image is None:
            return
        icc = None if self.tiff_obj is None else self.tiff_obj.iccprofile
        if icc is None:
            self.statusBar().showMessage("Sampling needs the file's ICC profile: the mixer hue is defined in the image's own working space")
            return
        h, w = self.proxy_image.shape[:2]
        x, y = min(int(u * w), w - 1), min(int(v * h), h - 1)
        hue = float(native_hue(median_rgb(self.proxy_image, x, y), icc)[0])
        self.harmony_panel.add_sample(x, y, self.displayed_rgb(x, y), hue)

    def on_harmony_samples_changed(self) -> None:
        """Places the numbered sample circles on the preview at their proxy pixels and enables Random Shift while there are samples."""
        self.update_random_shift_enabled()
        if self.proxy_image is None:
            self.single_preview.set_markers([])
            return
        h, w = self.proxy_image.shape[:2]
        self.single_preview.set_markers([(s.slot, (s.x + 0.5) / w, (s.y + 0.5) / h) for s in self.harmony_panel.samples])

    # ----------------------------------------------------
    # Explore Mode Actions & Harmonies
    # ----------------------------------------------------
    def on_explore_mode_toggled(self, state_val: int) -> None:
        """Switches between the single preview and the 3x3 grid and records an undo step."""
        is_active = state_val == 2  # Checked represents value 2
        
        if is_active:
            self.view_stack.setCurrentIndex(1)
            # Render mutations immediately
            snapshot = self.state_manager.capture_snapshot()
            state = self.rebuild_state_node_from_memento(snapshot)
            self.start_explore_mutations_render(state)
        else:
            self.view_stack.setCurrentIndex(0)
            
        self.on_discrete_action()

    def random_shift_ready(self) -> bool:
        """True while Random Shift has a colour to aim at and something to change: a Color Harmony sample and Mixer or Grading checked."""
        explore = self.explore_widget
        return bool(self.harmony_panel.samples) and (explore.mixer_checkbox.isChecked() or explore.grading_checkbox.isChecked())

    def update_random_shift_enabled(self) -> None:
        """Enables the Random Shift button only while random_shift_ready."""
        self.randomize_btn.setEnabled(self.random_shift_ready())

    def selected_colors(self, state: StateNode) -> ShiftColors:
        """The Color Harmony samples as random_shift colours: each sample's mixer hue, that hue on the grading wheel less state's Rotation (so it
        grades at the sample's hue), and the HSV saturation, value and tonal zone (state's Balance and Blending) of its displayed colour."""
        samples = self.harmony_panel.samples
        hue = np.array([sample.hue for sample in samples])
        rgb = np.array([sample.rgb for sample in samples])
        return shift_colors(hue, (wheel_angle(hue) - state["rotation"]) % 360.0, rgb, self.tiff_obj.iccprofile, self.grading,
                            state["balance"], state["blending"])

    def on_randomize_clicked(self) -> None:
        """Random Shift: in Explore Mode re-rolls the 8 outer images around the centre, otherwise shifts the current sliders (random_shift) as one undo step."""
        if self.proxy_image is None:
            return
        state = self.rebuild_state_node_from_memento(self.state_manager.capture_snapshot())
        if self.explore_checkbox.isChecked():
            self.start_explore_mutations_render(state)
            return
        explore = self.explore_widget
        self.apply_state_node_to_widgets(random_shift(state, self.selected_colors(state), explore.variation_strength,
                                                      explore.mixer_checkbox.isChecked(), explore.grading_checkbox.isChecked(), self.rng))
        self.on_discrete_action()
        self.statusBar().showMessage("Random Shift applied.")

    def on_explore_container_clicked(self, index: int) -> None:
        """
        Left-click on 3x3 outer slot promotions.
        """
        if self.proxy_image is None or self.grid_states is None:
            return
            
        if index == 4:
            # Re-roll at current baseline strength
            snapshot = self.state_manager.capture_snapshot()
            state = self.rebuild_state_node_from_memento(snapshot)
            self.start_explore_mutations_render(state)
            return
            
        # Left click outer: make center baseline, shrink Variation Strength, update Memento
        chosen_state = self.grid_states[index]
        if chosen_state is None:
            return

        # Halve the slider's integer value programmatically
        current_val = self.explore_widget.var_slider.value()
        self.explore_widget.variation_strength = (current_val // 2) / 100.0
        
        # Promote & Capture
        self.apply_state_node_to_widgets(chosen_state)
        self.on_discrete_action()
        self.statusBar().showMessage(f"Promoted mutation {index} to center baseline. Tightening variation range...")

    def on_explore_variation_changed(self, value: float) -> None:
        """Re-rolls the grid at the new Variation Strength while Explore Mode is on."""
        if self.explore_checkbox.isChecked():
            snapshot = self.state_manager.capture_snapshot()
            center_state = self.rebuild_state_node_from_memento(snapshot)
            self.start_explore_mutations_render(center_state)

    def on_explore_edit_requested(self, index: int) -> None:
        """
        Right-click outer slot context menu "Edit" clicked.
        Loads variation settings, exits Explore Mode, and goes back to single view.
        """
        if self.grid_states is None or index >= len(self.grid_states) or self.grid_states[index] is None:
            return
            
        chosen_state = self.grid_states[index]
        
        # Exit Explore Mode
        self.explore_checkbox.setChecked(False)
        self.view_stack.setCurrentIndex(0)
        
        # Apply & Push State
        self.apply_state_node_to_widgets(chosen_state)
        self.on_discrete_action()
        self.statusBar().showMessage(f"Loaded mutation {index} into manual editor. Exiting Explore Mode.")

    # ----------------------------------------------------
    # Manual Sliders & Color Wheels Callbacks
    # ----------------------------------------------------
    def on_tab_bar_clicked(self, index: int) -> None:
        """Re-clicking the active tab toggles the all-tones view of the wheels."""
        if index == self.tabs.currentIndex():
            self.show_all_tones = not self.show_all_tones
            self.on_manual_slider_changed()

    def on_manual_slider_changed(self) -> None:
        """
        Triggered when manual sliders or wheels values are dragged/shifted in real-time.
        Renders the active center single image immediately without capturing discrete Memento snapshots.
        """
        # Build StateNode from current widget states directly
        snapshot = self.state_manager.capture_snapshot()
        state = self.rebuild_state_node_from_memento(snapshot)
        
        rot = state["rotation"]
        eff_sh_hue = (state["shadow_hue"] + rot) % 360.0
        eff_mid_hue = (state["midtone_hue"] + rot) % 360.0
        eff_hi_hue = (state["highlight_hue"] + rot) % 360.0
        eff_gl_hue = (state["global_hue"] + rot) % 360.0

        # Update numerical labels dynamically in real-time with effective hues
        self.sh_zone.hue_lbl.setText(f"Hue: {eff_sh_hue:.0f}°")
        self.sh_zone.sat_lbl.setText(f"Sat: {state['shadow_sat']:.2f}")
        self.sh_zone.light_lbl.setText(f"Luma: {state['shadow_light']:+.2f}")
        
        self.mid_zone.hue_lbl.setText(f"Hue: {eff_mid_hue:.0f}°")
        self.mid_zone.sat_lbl.setText(f"Sat: {state['midtone_sat']:.2f}")
        self.mid_zone.light_lbl.setText(f"Luma: {state['midtone_light']:+.2f}")
        
        self.hi_zone.hue_lbl.setText(f"Hue: {eff_hi_hue:.0f}°")
        self.hi_zone.sat_lbl.setText(f"Sat: {state['highlight_sat']:.2f}")
        self.hi_zone.light_lbl.setText(f"Luma: {state['highlight_light']:+.2f}")

        self.gl_zone.hue_lbl.setText(f"Hue: {eff_gl_hue:.0f}°")
        self.gl_zone.sat_lbl.setText(f"Sat: {state['global_sat']:.2f}")
        self.gl_zone.light_lbl.setText(f"Luma: {state['global_light']:+.2f}")

        self.master_zone.blending_lbl.setText(f"Blending: {state['blending']:.2f}")
        self.master_zone.balance_lbl.setText(f"Balance: {state['balance']:+.2f}")
        self.master_zone.rotation_lbl.setText(f"Rotation: {rot:.0f}°")

        # The harmony wheel and wedges follow the Color Mixer's Hue sliders
        self.harmony_panel.set_hue_levels(state["mixer_hue"])

        # Update visual color wheel indicators/rotation offsets/saturation/hue in real-time
        self.sh_zone.wheel.rotation_offset = rot
        self.mid_zone.wheel.rotation_offset = rot
        self.hi_zone.wheel.rotation_offset = rot
        self.gl_zone.wheel.rotation_offset = rot

        self.sh_zone.wheel.sat = state["shadow_sat"]
        self.mid_zone.wheel.sat = state["midtone_sat"]
        self.hi_zone.wheel.sat = state["highlight_sat"]
        self.gl_zone.wheel.sat = state["global_sat"]

        self.sh_zone.wheel.hue = state["shadow_hue"]
        self.mid_zone.wheel.hue = state["midtone_hue"]
        self.hi_zone.wheel.hue = state["highlight_hue"]
        self.gl_zone.wheel.hue = state["global_hue"]

        # Update multi-view tone data for the wheels in real-time
        active_idx = self.tabs.currentIndex()
        for idx, zone in enumerate([self.sh_zone, self.mid_zone, self.hi_zone]):
            zone.wheel.show_all_tones = (self.show_all_tones and idx == active_idx)
            if zone.wheel.show_all_tones:
                zone.wheel.all_tones_data = {
                    'S': (state["shadow_hue"], state["shadow_sat"]),
                    'M': (state["midtone_hue"], state["midtone_sat"]),
                    'H': (state["highlight_hue"], state["highlight_sat"])
                }
            else:
                zone.wheel.all_tones_data = {}
        
        self.sh_zone.wheel.update()
        self.mid_zone.wheel.update()
        self.hi_zone.wheel.update()
        self.gl_zone.wheel.update()

        if self.master_image is None or self.proxy_image is None:
            return

        self.render_timer.start()

    def on_sh_wheel_changed(self, hue: float, sat: float) -> None:
        """The Shadows wheel moved: syncs its Hue and Sat sliders without signals, then refreshes."""
        self.sh_zone.sat_slider.blockSignals(True)
        self.sh_zone.sat_slider.setValue(int(sat * 100.0))
        self.sh_zone.sat_slider.blockSignals(False)
        self.sh_zone.hue_slider.blockSignals(True)
        self.sh_zone.hue_slider.setValue(int(hue))
        self.sh_zone.hue_slider.blockSignals(False)
        self.on_manual_slider_changed()

    def on_mid_wheel_changed(self, hue: float, sat: float) -> None:
        """The Midtones wheel moved: syncs its Hue and Sat sliders without signals, then refreshes."""
        self.mid_zone.sat_slider.blockSignals(True)
        self.mid_zone.sat_slider.setValue(int(sat * 100.0))
        self.mid_zone.sat_slider.blockSignals(False)
        self.mid_zone.hue_slider.blockSignals(True)
        self.mid_zone.hue_slider.setValue(int(hue))
        self.mid_zone.hue_slider.blockSignals(False)
        self.on_manual_slider_changed()

    def on_hi_wheel_changed(self, hue: float, sat: float) -> None:
        """The Highlights wheel moved: syncs its Hue and Sat sliders without signals, then refreshes."""
        self.hi_zone.sat_slider.blockSignals(True)
        self.hi_zone.sat_slider.setValue(int(sat * 100.0))
        self.hi_zone.sat_slider.blockSignals(False)
        self.hi_zone.hue_slider.blockSignals(True)
        self.hi_zone.hue_slider.setValue(int(hue))
        self.hi_zone.hue_slider.blockSignals(False)
        self.on_manual_slider_changed()

    def on_gl_wheel_changed(self, hue: float, sat: float) -> None:
        """The Global wheel moved: syncs its Hue and Sat sliders without signals, then refreshes."""
        self.gl_zone.sat_slider.blockSignals(True)
        self.gl_zone.sat_slider.setValue(int(sat * 100.0))
        self.gl_zone.sat_slider.blockSignals(False)
        self.gl_zone.hue_slider.blockSignals(True)
        self.gl_zone.hue_slider.setValue(int(hue))
        self.gl_zone.hue_slider.blockSignals(False)
        self.on_manual_slider_changed()

    def on_reset_all(self) -> None:
        """Every wheel (Global included), Blending, Balance and Rotation back to their defaults, as one undoable step."""
        self.apply_state_node_to_widgets(create_default_state())
        self.on_discrete_action()
        self.statusBar().showMessage("All grading controls reset.")

    def on_save_preset(self) -> None:
        """Asks for a path and saves the current settings as a JSON preset."""
        path, _ = QFileDialog.getSaveFileName(self, "Save Grading Preset", self.mru_manager.get_most_recent_directory(), "Presets (*.json)")
        if path:
            try:
                save_preset(path, self.rebuild_state_node_from_memento(self.state_manager.capture_snapshot()))
                self.statusBar().showMessage(f"Saved preset {path}")
            except Exception as e:
                self.statusBar().showMessage(f"Error saving preset: {str(e)}")

    def on_load_preset(self) -> None:
        """Asks for a JSON preset, applies it to the widgets and records an undo step."""
        path, _ = QFileDialog.getOpenFileName(self, "Load Grading Preset", self.mru_manager.get_most_recent_directory(), "Presets (*.json)")
        if path:
            try:
                self.apply_state_node_to_widgets(load_preset(path))
                self.on_discrete_action()
                self.statusBar().showMessage(f"Loaded preset {path}")
            except Exception as e:
                self.statusBar().showMessage(f"Error loading preset: {str(e)}")

    def reset_hue(self, zone: ZoneTabWidget, default_hue: float) -> None:
        """Resets a zone's hue to default_hue and records an undo step."""
        zone.hue_slider.blockSignals(True)
        zone.hue_slider.setValue(int(default_hue))
        zone.hue_slider.blockSignals(False)
        zone.wheel.set_color(default_hue, zone.wheel.sat)
        self.on_manual_slider_changed()
        self.on_discrete_action()

    def reset_slider(self, slider: QSlider, value: int) -> None:
        """Sets slider to value and records an undo step."""
        slider.setValue(value)
        self.on_discrete_action()

    # ----------------------------------------------------
    # File I/O MRU
    # ----------------------------------------------------
    def update_recent_directories_menu(self) -> None:
        """Rebuilds the Recent Directories menu from the MRU list."""
        self.recent_menu.clear()
        paths = self.mru_manager.get_mru_directories()
        
        if not paths:
            no_recent_action = QAction("No Recent Directories", self)
            no_recent_action.setEnabled(False)
            self.recent_menu.addAction(no_recent_action)
            return

        for i, path in enumerate(paths):
            action = QAction(f"&{i + 1}: {path}", self)
            action.triggered.connect(lambda checked=False, p=path: self.open_file_dialog_at_dir(p))
            self.recent_menu.addAction(action)

    def open_file_dialog_at_dir(self, start_dir: str) -> None:
        """Opens the file dialog in start_dir."""
        self.open_file(start_dir=start_dir)

    def open_file(self, start_dir: Optional[str] = None) -> None:
        """Asks for an RGB TIFF and loads it with load_file."""
        if start_dir is None:
            start_dir = self.mru_manager.get_most_recent_directory()

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open 16-bit TIFF",
            start_dir,
            "TIFF Images (*.tif *.tiff);;All Files (*)"
        )
        if file_path:
            self.load_file(file_path)

    def load_file(self, file_path: str) -> None:
        """Loads the RGB TIFF at file_path (values as stored, ICC profile kept), makes the preview proxy and starts a fresh history with default settings."""
        try:
            tiff = TiffFile().read(file_path)
            if tiff.array.ndim != 3 or tiff.array.shape[-1] != 3:
                raise ValueError(f"expected an RGB image, got shape {tiff.array.shape}")
            self.tiff_obj = tiff
            self.master_image = tiff.array
            self.proxy_image = downsample_image(self.master_image, max_dim=1200)
            self.drag_proxy = downsample_image(self.proxy_image, max_dim=DRAG_PREVIEW_DIM)
            self.mixer_cache.clear()
            
            self.mru_manager.add_path(file_path)
            self.update_recent_directories_menu()
            
            # Push initial starting state onto the StateManager
            initial_state = create_default_state("Monochromatic")
            self.apply_state_node_to_widgets(initial_state)
            self.harmony_panel.clear_samples()     # samples are pixels of the old image; mode and rotation are kept
            
            # Create initial snapshot capture
            self.state_manager.undo_stack.clear()
            self.state_manager.redo_stack.clear()
            
            snapshot = self.state_manager.capture_snapshot()
            snapshot["step_size"] = 0.20
            self.state_manager.push_state(snapshot)
            
            # Render View
            self.render_view_from_memento(snapshot)
            self.update_memento_navigation_ui()
            
            shape_str = "x".join(map(str, self.master_image.shape))
            proxy_str = "x".join(map(str, self.proxy_image.shape))
            if tiff.iccprofile is None:
                self.statusBar().showMessage(f"Loaded {file_path}, but it has no embedded ICC profile: grading is disabled (sRGB is never assumed)")
            else:
                self.statusBar().showMessage(f"Loaded {file_path} ({tiff.icc_name}; Master: {shape_str}, Proxy: {proxy_str})")
        except Exception as e:
            self.statusBar().showMessage(f"Error loading image: {str(e)}")

    def save_file(self) -> None:
        """Asks for a path, then grades and writes the full image in a SaveWorker (needs the file's ICC profile)."""
        if self.master_image is None or self.tiff_obj is None:
            return
        if self.tiff_obj.iccprofile is None:
            self.statusBar().showMessage("Cannot save a graded image: the file has no embedded ICC profile.")
            return

        start_dir = self.mru_manager.get_most_recent_directory()

        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save 16-bit TIFF",
            start_dir,
            "TIFF Images (*.tif *.tiff);;All Files (*)"
        )
        if save_path:
            snapshot = self.state_manager.capture_snapshot()
            center_state = self.rebuild_state_node_from_memento(snapshot)

            self.save_action.setEnabled(False)
            self.statusBar().showMessage(f"Grading and saving {save_path} ...")
            self.save_worker = SaveWorker(self.graded_full, self.master_image, center_state, self.tiff_obj, save_path, parent=self)
            self.save_worker.saved.connect(self.on_save_done)
            self.save_worker.failed.connect(lambda msg: self.statusBar().showMessage(f"Error saving image: {msg}"))
            self.save_worker.finished.connect(lambda: self.save_action.setEnabled(True))
            self.save_worker.start()

    def on_save_done(self, save_path: str) -> None:
        """Adds the saved file to the recent directories and reports success."""
        self.mru_manager.add_path(save_path)
        self.update_recent_directories_menu()
        self.statusBar().showMessage(f"Successfully saved graded image to {save_path}")

    def open_lightroom_request(self, request: lr_bridge.Request) -> None:
        """Loads the request's TIFF, sets the Color Mixer, the four wheels, Blending and Balance to its Lightroom values (one undo step) and adds the Apply to Lightroom button."""
        self.lightroom_request = request
        self.load_file(request.tiff)
        state = self.rebuild_state_node_from_memento(self.state_manager.capture_snapshot())
        for name, _, prefix in LIGHTROOM_WHEELS:
            hue, sat, lum = request.wheels[name]
            state[f"{prefix}_hue"], state[f"{prefix}_sat"], state[f"{prefix}_light"] = hue, sat / 100.0, lum / 100.0
        state["blending"], state["balance"] = request.blending / 100.0, request.balance / 100.0
        for kind in KINDS:
            state[f"mixer_{kind}"] = [float(v) for v in request.mixer[kind]]
        self.apply_state_node_to_widgets(state)
        self.on_discrete_action()
        apply_btn = QPushButton("Apply to Lightroom")
        apply_btn.clicked.connect(self.apply_to_lightroom)
        self.statusBar().addPermanentWidget(apply_btn)

    def apply_to_lightroom(self) -> None:
        """Writes the current mixer and grading to the request's result file, the hues with Rotation added as they are graded, and closes the window."""
        state = self.rebuild_state_node_from_memento(self.state_manager.capture_snapshot())
        kw = grading_kwargs(state)
        mixer = {kind: state[f"mixer_{kind}"] for kind in KINDS}
        wheels = {name: (kw[f"{short}_h"], kw[f"{short}_s"], kw[f"{short}_l"]) for name, short, _ in LIGHTROOM_WHEELS}
        lr_bridge.write_result(self.lightroom_request.result, mixer, wheels, kw["blend"], kw["balance"])
        self.lightroom_applied = True
        self.close()

    def closeEvent(self, event: QCloseEvent) -> None:
        """Waits for running workers (cancelling the grid render) before the window closes."""
        if self.grading_worker is not None:
            if self.grading_worker.isRunning():
                self.grading_worker.cancel()
                self.grading_worker.wait()
        self.render_timer.stop()
        self.pending_preview_state = None
        for worker in (self.preview_worker, self.save_worker):     # neither can be interrupted mid-grade; a save must finish writing
            if worker is not None and worker.isRunning():
                worker.wait()
        super().closeEvent(event)
