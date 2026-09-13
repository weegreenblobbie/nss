import numpy as np
import json
from typing import List, Optional, Literal
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QSettings, QPointF
from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QGridLayout,
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
    QComboBox,
    QPushButton,
    QCheckBox,
    QStackedWidget,
    QGroupBox,
)
from PyQt6.QtGui import (
    QMouseEvent,
    QAction,
    QImage,
    QPixmap,
    QColor,
    QPainter,
    QConicalGradient,
    QRadialGradient,
    QPen,
    QDrag,
    QIntValidator,
)

from nss.utils import TiffFile
from nss.color_math import (
    StateNode,
    MutationAxis,
    create_default_state,
    apply_grading,
    generate_mutations,
    generate_random_harmony_state,
    generate_explore_mutations,
)
from nss.mru import MruManager

from nss.color_gui.color_wheel import ColorWheel, ResetLabel
from nss.color_gui.swatches import ClickableSwatchLabel, ColorSwatch, ColorModifyDialog
from nss.color_gui.explore_widget import ExploreWidget, ImageContainer
from nss.color_gui.state_manager import StateManager


class ZoneTabWidget(QWidget):
    """
    Base class for a stateful Zone tab widget.
    Enforces Memento state management with a unique STATE_KEY.
    """
    STATE_KEY = "base_zone"  # Subclasses must override this!
    
    def __init__(self, prefix: str, default_h: int, on_changed_slot, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.prefix = prefix.lower()
        self.default_h = default_h
        self.on_changed_slot = on_changed_slot
        
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(10, 10, 10, 10)
        self.layout.setSpacing(5)
        
        self.swatch_lbl = ClickableSwatchLabel()
        self.swatch_lbl.setFixedSize(24, 24)
        self.swatch_lbl.setStyleSheet("border: 1px solid #cccccc; border-radius: 4px;")
        self.swatch_lbl.setCursor(Qt.CursorShape.PointingHandCursor)
        self.swatch_lbl.setToolTip("Click to open color picker wheel")
        
        self.wheel = ColorWheel()
        wheel_container = QHBoxLayout()
        wheel_container.addStretch()
        wheel_container.addWidget(self.wheel)
        wheel_container.addStretch()
        self.layout.addLayout(wheel_container)
        
        self.hue_lbl = ResetLabel(f"Hue: {default_h}°")
        self.hue_lbl.setStyleSheet("font-size: 11px;")
        self.hue_lbl.setToolTip("Double-click to reset Hue to default")
        
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
        self.layout.addWidget(self.sat_lbl)
        self.layout.addWidget(self.sat_slider)
        self.layout.addWidget(self.light_lbl)
        self.layout.addWidget(self.light_slider)
        
        # Saved custom colors section
        recent_lbl = QLabel("Saved Custom Colors:")
        recent_lbl.setStyleSheet("font-size: 10px; font-weight: bold; margin-top: 5px;")
        self.layout.addWidget(recent_lbl)
        
        patch_layout = QHBoxLayout()
        patch_layout.setContentsMargins(0, 0, 0, 0)
        patch_layout.setSpacing(6)
        
        drag_lbl = QLabel("Drag to save")
        drag_lbl.setStyleSheet("font-size: 10px; font-style: italic;")
        
        patch_layout.addWidget(self.swatch_lbl)
        patch_layout.addWidget(drag_lbl)
        patch_layout.addStretch()
        self.layout.addLayout(patch_layout)
        
        self.history_grid = QGridLayout()
        self.history_grid.setContentsMargins(0, 2, 0, 2)
        self.history_grid.setSpacing(4)
        grid_widget = QWidget()
        grid_widget.setLayout(self.history_grid)
        self.layout.addWidget(grid_widget)
        self.layout.addStretch()
        
        # Wire signals
        self.sat_slider.valueChanged.connect(self.on_changed_slot)
        self.light_slider.valueChanged.connect(self.on_changed_slot)
        
        self.sat_slider.sliderReleased.connect(self.parent_slider_released)
        self.light_slider.sliderReleased.connect(self.parent_slider_released)
        self.wheel.interactionFinished.connect(self.parent_wheel_released)

    def parent_slider_released(self) -> None:
        parent_mw = self.window()
        if parent_mw and hasattr(parent_mw, "on_discrete_action"):
            parent_mw.on_discrete_action()

    def parent_wheel_released(self, h, s) -> None:
        parent_mw = self.window()
        if parent_mw and hasattr(parent_mw, "on_discrete_action"):
            parent_mw.on_discrete_action()

    def get_state(self) -> dict:
        return {
            self.STATE_KEY: {
                "hue": float(self.wheel.hue),
                "sat": self.sat_slider.value() / 100.0,
                "light": self.light_slider.value() / 100.0,
            }
        }
        
    def restore_state(self, full_state_dict: dict) -> None:
        if self.STATE_KEY in full_state_dict:
            state = full_state_dict[self.STATE_KEY]
            self.wheel.blockSignals(True)
            self.sat_slider.blockSignals(True)
            self.light_slider.blockSignals(True)
            
            self.wheel.set_color(state["hue"], state["sat"])
            self.sat_slider.setValue(int(state["sat"] * 100.0))
            self.light_slider.setValue(int(state["light"] * 100.0))
            
            # Update labels
            self.hue_lbl.setText(f"Hue: {state['hue']:.0f}°")
            self.sat_lbl.setText(f"Sat: {state['sat']:.2f}")
            self.light_lbl.setText(f"Luma: {state['light']:+.2f}")
            
            self.wheel.blockSignals(False)
            self.sat_slider.blockSignals(False)
            self.light_slider.blockSignals(False)


class ShadowsZoneWidget(ZoneTabWidget):
    STATE_KEY = "shadows_zone"


class MidtonesZoneWidget(ZoneTabWidget):
    STATE_KEY = "midtones_zone"


class HighlightsZoneWidget(ZoneTabWidget):
    STATE_KEY = "highlights_zone"


class MasterZoneWidget(QWidget):
    STATE_KEY = "master_zone"
    
    def __init__(self, on_changed_slot, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.on_changed_slot = on_changed_slot
        
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(10)
        
        # Title
        m_title = QLabel("MASTER DOCK CONTROLS")
        m_title.setStyleSheet("font-weight: bold; border: none; border-bottom: 1px solid #cccccc; padding-bottom: 2px;")
        self.layout.addWidget(m_title)

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
        slider.setValue(value)
        self.parent_slider_released()

    def parent_slider_released(self) -> None:
        parent_mw = self.window()
        if parent_mw and hasattr(parent_mw, "on_discrete_action"):
            parent_mw.on_discrete_action()

    def get_state(self) -> dict:
        return {
            self.STATE_KEY: {
                "blending": self.blending_slider.value() / 100.0,
                "balance": self.balance_slider.value() / 100.0,
                "rotation": float(self.rotation_slider.value()),
            }
        }
        
    def restore_state(self, full_state_dict: dict) -> None:
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

    def __init__(self, working_image: np.ndarray, states: List[StateNode], skip_center: bool = False, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.working_image = working_image
        self.states = states
        self.skip_center = skip_center
        self.is_cancelled = False

    def cancel(self) -> None:
        self.is_cancelled = True

    def run(self) -> None:
        if self.skip_center:
            indices = [i for i in range(9) if i != 4]
        else:
            indices = [4] + [i for i in range(9) if i != 4]

        for i in indices:
            if self.is_cancelled:
                return
            if i >= len(self.states):
                continue
            try:
                graded = apply_grading(self.working_image, self.states[i])
                if self.is_cancelled:
                    return
                self.progress.emit(i, graded)
            except Exception:
                pass
        self.finished_all.emit()


class MainWindow(QMainWindow):
    """
    The Main Window for the 16-bit Color Grading Explorer with Memento State Management & Explore Mode.
    """
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("16-bit Color Grading Explorer")
        self.resize(1300, 850)

        # Core Managers
        self.state_manager = StateManager()
        self.mru_manager = MruManager()

        # State Variables
        self.master_image: Optional[np.ndarray] = None
        self.proxy_image: Optional[np.ndarray] = None
        self.tiff_obj: Optional[TiffFile] = None
        self.grid_states: Optional[List[StateNode]] = None
        self.grading_worker: Optional[GradingWorker] = None

        # Setup UI
        self.init_ui()
        
        # Enforce unique state widget registrations
        self.state_manager.register_widget(self.sh_zone)
        self.state_manager.register_widget(self.mid_zone)
        self.state_manager.register_widget(self.hi_zone)
        self.state_manager.register_widget(self.master_zone)
        self.state_manager.register_widget(self.explore_widget)

    def init_ui(self) -> None:
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

        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Alt+F4")
        exit_action.setStatusTip("Exit the application")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # 1. Create Toolbar with Back/Forward Undo/Redo & Intensity slider
        toolbar = QToolBar("Navigation and Controls")
        self.addToolBar(toolbar)

        # Memento Undo/Redo Controls
        self.back_action = QAction("Back (Undo)", self)
        self.back_action.setShortcut("Ctrl+Z")
        self.back_action.setEnabled(False)
        self.back_action.triggered.connect(self.on_undo_clicked)
        toolbar.addAction(self.back_action)

        self.history_label = QLabel("Step: 0 of 0")
        self.history_label.setStyleSheet(
            "font-weight: bold; margin-left: 10px; margin-right: 10px;"
        )
        toolbar.addWidget(self.history_label)

        self.forward_action = QAction("Forward (Redo)", self)
        self.forward_action.setShortcut("Ctrl+Y")
        self.forward_action.setEnabled(False)
        self.forward_action.triggered.connect(self.on_redo_clicked)
        toolbar.addAction(self.forward_action)

        toolbar.addSeparator()

        # Intensity Label
        self.intensity_label = QLabel(" Intensity: 0.20x ")
        self.intensity_label.setStyleSheet("font-weight: bold;")
        toolbar.addWidget(self.intensity_label)

        self.intensity_slider = QSlider(Qt.Orientation.Horizontal)
        self.intensity_slider.setRange(10, 200)
        self.intensity_slider.setValue(20)
        self.intensity_slider.setFixedWidth(120)
        self.intensity_slider.valueChanged.connect(self.on_intensity_changed)
        self.intensity_slider.sliderReleased.connect(self.on_discrete_action)
        toolbar.addWidget(self.intensity_slider)

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
        self.single_preview = ImageContainer(-1, self)
        self.view_stack.addWidget(self.single_preview)

        # Explore Mode Grid View (Page 1)
        self.explore_widget = ExploreWidget(self)
        self.explore_widget.container_clicked.connect(self.on_explore_container_clicked)
        self.explore_widget.container_edit_requested.connect(self.on_explore_edit_requested)
        self.view_stack.addWidget(self.explore_widget)

        # Right Side: Manual Grading & Harmony Randomizer Dock
        self.inspector_panel = QFrame()
        self.inspector_panel.setFrameShape(QFrame.Shape.StyledPanel)
        self.inspector_panel.setFrameShadow(QFrame.Shadow.Raised)
        self.inspector_panel.setFixedWidth(300)
        self.inspector_panel.setStyleSheet(
            "border: 1px solid #cccccc; border-radius: 4px;"
        )
        inspector_layout = QVBoxLayout(self.inspector_panel)
        inspector_layout.setContentsMargins(15, 15, 15, 15)
        inspector_layout.setSpacing(12)
        main_layout.addWidget(self.inspector_panel, stretch=1)

        # Harmony Randomizer / Explore Mode Section
        harmony_box = QGroupBox("EXPLORE & HARMONY")
        harmony_box.setStyleSheet(
            "QGroupBox { font-weight: bold; border: 1px solid #cccccc; border-radius: 4px; margin-top: 10px; padding: 10px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 3px; }"
        )
        harmony_layout = QVBoxLayout(harmony_box)
        harmony_layout.setSpacing(8)

        # Dropdown
        dropdown_row = QHBoxLayout()
        dropdown_row.addWidget(QLabel("Mode:"))
        self.harmony_dropdown = QComboBox()
        self.harmony_dropdown.addItems(["Complementary", "Analogous", "Triadic", "Monochromatic"])
        self.harmony_dropdown.currentTextChanged.connect(self.on_harmony_mode_dropdown_changed)
        dropdown_row.addWidget(self.harmony_dropdown)
        harmony_layout.addLayout(dropdown_row)

        # Explore Toggle
        self.explore_checkbox = QCheckBox("Activate Explore Mode")
        self.explore_checkbox.setStyleSheet("font-weight: bold;")
        self.explore_checkbox.stateChanged.connect(self.on_explore_mode_toggled)
        harmony_layout.addWidget(self.explore_checkbox)

        # Randomize Button
        self.randomize_btn = QPushButton("Generate New Random Look")
        self.randomize_btn.setStyleSheet("background-color: #2b5b2b; color: white; font-weight: bold; padding: 4px;")
        self.randomize_btn.clicked.connect(self.on_randomize_clicked)
        harmony_layout.addWidget(self.randomize_btn)

        inspector_layout.addWidget(harmony_box)

        # Manual 3-Way Tabs Setup
        self.tabs = QTabWidget()
        inspector_layout.addWidget(self.tabs)

        # Stateful tab widgets
        self.sh_zone = ShadowsZoneWidget("Shadows", 240, self.on_manual_slider_changed, self)
        self.tabs.addTab(self.sh_zone, "Shadows")
        self.sh_zone.wheel.colorChanged.connect(self.on_sh_wheel_changed)
        self.sh_zone.hue_lbl.doubleClicked.connect(lambda: self.reset_wheel_hue(self.sh_zone.wheel, 240))
        self.sh_zone.sat_lbl.doubleClicked.connect(lambda: self.reset_slider(self.sh_zone.sat_slider, 0))
        self.sh_zone.light_lbl.doubleClicked.connect(lambda: self.reset_slider(self.sh_zone.light_slider, 0))

        self.mid_zone = MidtonesZoneWidget("Midtones", 120, self.on_manual_slider_changed, self)
        self.tabs.addTab(self.mid_zone, "Midtones")
        self.mid_zone.wheel.colorChanged.connect(self.on_mid_wheel_changed)
        self.mid_zone.hue_lbl.doubleClicked.connect(lambda: self.reset_wheel_hue(self.mid_zone.wheel, 120))
        self.mid_zone.sat_lbl.doubleClicked.connect(lambda: self.reset_slider(self.mid_zone.sat_slider, 0))
        self.mid_zone.light_lbl.doubleClicked.connect(lambda: self.reset_slider(self.mid_zone.light_slider, 0))

        self.hi_zone = HighlightsZoneWidget("Highlights", 60, self.on_manual_slider_changed, self)
        self.tabs.addTab(self.hi_zone, "Highlights")
        self.hi_zone.wheel.colorChanged.connect(self.on_hi_wheel_changed)
        self.hi_zone.hue_lbl.doubleClicked.connect(lambda: self.reset_wheel_hue(self.hi_zone.wheel, 60))
        self.hi_zone.sat_lbl.doubleClicked.connect(lambda: self.reset_slider(self.hi_zone.sat_slider, 0))
        self.hi_zone.light_lbl.doubleClicked.connect(lambda: self.reset_slider(self.hi_zone.light_slider, 0))

        # Populate custom swatches
        self.update_history_swatches_ui("shadows", self.load_custom_colors("shadows"))
        self.update_history_swatches_ui("midtones", self.load_custom_colors("midtones"))
        self.update_history_swatches_ui("highlights", self.load_custom_colors("highlights"))

        # Master Global Controls Section
        self.master_zone = MasterZoneWidget(self.on_manual_slider_changed, self)
        inspector_layout.addWidget(self.master_zone)
        inspector_layout.addStretch()

        self.statusBar().showMessage("Ready. Load a 16-bit TIFF image to begin.")

    # ----------------------------------------------------
    # Memento System Integration
    # ----------------------------------------------------
    def on_discrete_action(self) -> None:
        """
        Gathers snapshots on discrete events and pushes onto undo_stack.
        """
        snapshot = self.state_manager.capture_snapshot()
        
        # Verify step size slider value
        snapshot["step_size"] = self.intensity_slider.value() / 100.0
        
        # Avoid duplicate pushes
        if self.state_manager.undo_stack and self.state_manager.undo_stack[-1] == snapshot:
            return
            
        self.state_manager.push_state(snapshot)
        self.update_memento_navigation_ui()
        
        # Re-render everything from new state!
        self.render_view_from_memento(snapshot)

    def on_undo_clicked(self) -> None:
        if self.state_manager.can_undo():
            snapshot = self.state_manager.undo()
            if snapshot:
                self.state_manager.restore_snapshot(snapshot)
                
                # Sync step size slider
                self.intensity_slider.blockSignals(True)
                self.intensity_slider.setValue(int(snapshot.get("step_size", 0.2) * 100.0))
                self.intensity_label.setText(f" Intensity: {snapshot.get('step_size', 0.2):.2f}x ")
                self.intensity_slider.blockSignals(False)
                
                self.render_view_from_memento(snapshot)
                self.update_memento_navigation_ui()
                self.statusBar().showMessage("Undo performed.")

    def on_redo_clicked(self) -> None:
        if self.state_manager.can_redo():
            snapshot = self.state_manager.redo()
            if snapshot:
                self.state_manager.restore_snapshot(snapshot)
                
                # Sync step size slider
                self.intensity_slider.blockSignals(True)
                self.intensity_slider.setValue(int(snapshot.get("step_size", 0.2) * 100.0))
                self.intensity_label.setText(f" Intensity: {snapshot.get('step_size', 0.2):.2f}x ")
                self.intensity_slider.blockSignals(False)
                
                self.render_view_from_memento(snapshot)
                self.update_memento_navigation_ui()
                self.statusBar().showMessage("Redo performed.")

    def update_memento_navigation_ui(self) -> None:
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
        
        # Explore mode parameters
        explore_state = snapshot.get("explore_widget", {})
        state["harmony_mode"] = explore_state.get("harmony_mode", "Monochromatic")
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

        # Master
        m_state = snapshot.get("master_zone", {})
        state["blending"] = m_state.get("blending", 0.5)
        state["balance"] = m_state.get("balance", 0.0)
        state["rotation"] = m_state.get("rotation", 0.0)

        return state

    def apply_state_node_to_widgets(self, state: StateNode) -> None:
        """
        Applies a StateNode dict parameters to stateful widgets without triggering snapshot capture.
        """
        snapshot = {
            "shadows_zone": {"hue": state["shadow_hue"], "sat": state["shadow_sat"], "light": state["shadow_light"]},
            "midtones_zone": {"hue": state["midtone_hue"], "sat": state["midtone_sat"], "light": state["midtone_light"]},
            "highlights_zone": {"hue": state["highlight_hue"], "sat": state["highlight_sat"], "light": state["highlight_light"]},
            "master_zone": {"blending": state["blending"], "balance": state["balance"], "rotation": state.get("rotation", 0.0)},
            "explore_widget": {"harmony_mode": state["harmony_mode"], "variation_strength": self.explore_widget.variation_strength},
        }
        self.state_manager.restore_snapshot(snapshot)

    # ----------------------------------------------------
    # Rendering & Asynchronous Workers
    # ----------------------------------------------------
    def render_view_from_memento(self, snapshot: dict) -> None:
        if self.proxy_image is None:
            return
            
        state = self.rebuild_state_node_from_memento(snapshot)
        
        # 1. Update Single Preview (Normal Mode)
        graded_single = apply_grading(self.proxy_image, state)
        self.single_preview.set_image(graded_single)
        
        # 2. Update 3x3 Look-Matrix (Explore Mode)
        if self.explore_widget.isVisible():
            self.start_explore_mutations_render(state)

    def start_explore_mutations_render(self, center_state: StateNode) -> None:
        if self.proxy_image is None:
            return
            
        # Cancel active background workers
        if self.grading_worker is not None:
            if self.grading_worker.isRunning():
                self.grading_worker.cancel()
                self.grading_worker.wait()
            self.grading_worker = None

        # Generate 8 mutations around center
        mode = self.explore_widget.harmony_mode
        self.grid_states = generate_explore_mutations(center_state, mode, self.explore_widget.variation_strength)
        self.grid_states[4] = center_state.copy()

        # Fire async worker
        self.grading_worker = GradingWorker(
            self.proxy_image,
            self.grid_states,
            skip_center=False,
            parent=self
        )
        self.grading_worker.progress.connect(self.on_worker_progress)
        self.grading_worker.finished_all.connect(self.on_worker_finished)
        self.grading_worker.start()

    def on_worker_progress(self, index: int, arr: np.ndarray) -> None:
        self.explore_widget.containers[index].set_image(arr)
        if index == 4:
            self.single_preview.set_image(arr)

    def on_worker_finished(self) -> None:
        self.statusBar().showMessage("Ready.")

    # ----------------------------------------------------
    # Explore Mode Actions & Harmonies
    # ----------------------------------------------------
    def on_explore_mode_toggled(self, state_val: int) -> None:
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

    def on_harmony_mode_dropdown_changed(self, text: str) -> None:
        self.explore_widget.harmony_mode = text
        self.on_discrete_action()

    def on_randomize_clicked(self) -> None:
        if self.proxy_image is None:
            return
            
        # Capture current blending and balance before randomizing so they remain exactly where they are
        snapshot = self.state_manager.capture_snapshot()
        curr_state = self.rebuild_state_node_from_memento(snapshot)
        curr_blending = curr_state.get("blending", 0.5)
        curr_balance = curr_state.get("balance", 0.0)
        
        # Zero out the Rotation slider/label automatically
        self.master_zone.rotation_slider.blockSignals(True)
        self.master_zone.rotation_slider.setValue(0)
        self.master_zone.rotation_lbl.setText("Rotation: 0°")
        self.master_zone.rotation_slider.blockSignals(False)
        
        if self.explore_checkbox.isChecked():
            # Lock/keep the center baseline state, with rotation reset to 0, and regenerate the 8 outer mutations
            curr_state["rotation"] = 0.0
            self.start_explore_mutations_render(curr_state)
            self.statusBar().showMessage("Regenerated 8 outer mutations based on current center baseline.")
        else:
            mode = self.harmony_dropdown.currentText()
            random_state = generate_random_harmony_state(mode)
            
            # Keep global blending and balance exactly where they were
            random_state["blending"] = curr_blending
            random_state["balance"] = curr_balance
            random_state["rotation"] = 0.0
            
            # Reset variation strength for first generation
            self.explore_widget.variation_strength = 1.0
            
            # Sync widget values and record discrete state
            self.apply_state_node_to_widgets(random_state)
            self.on_discrete_action()
            self.statusBar().showMessage("Generated a brand new random harmony look!")

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
        self.explore_widget.variation_strength *= 0.5  # ShrinkVariation Strength for tighter drill down
        
        # Promote & Capture
        self.apply_state_node_to_widgets(chosen_state)
        self.on_discrete_action()
        self.statusBar().showMessage(f"Promoted mutation {index} to center baseline. Tightening variation range...")

    def on_explore_edit_requested(self, index: int) -> None:
        """
        Right-click outer slot context menu "Edit" clicked.
        Loads variation settings, exits Explore Mode, and goes back to single view.
        """
        if self.grid_states is None or index >= len(self.grid_states):
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
        
        self.master_zone.blending_lbl.setText(f"Blending: {state['blending']:.2f}")
        self.master_zone.balance_lbl.setText(f"Balance: {state['balance']:+.2f}")
        self.master_zone.rotation_lbl.setText(f"Rotation: {rot:.0f}°")

        # Update visual color wheel indicators/rotation offsets/saturation in real-time
        self.sh_zone.wheel.rotation_offset = rot
        self.mid_zone.wheel.rotation_offset = rot
        self.hi_zone.wheel.rotation_offset = rot
        
        self.sh_zone.wheel.sat = state["shadow_sat"]
        self.mid_zone.wheel.sat = state["midtone_sat"]
        self.hi_zone.wheel.sat = state["highlight_sat"]
        
        self.sh_zone.wheel.update()
        self.mid_zone.wheel.update()
        self.hi_zone.wheel.update()

        if self.master_image is None or self.proxy_image is None:
            return

        # Update swatches and single view preview instantly
        self.sh_zone.swatch_lbl.hue = eff_sh_hue
        self.sh_zone.swatch_lbl.sat = state["shadow_sat"]
        
        self.mid_zone.swatch_lbl.hue = eff_mid_hue
        self.mid_zone.swatch_lbl.sat = state["midtone_sat"]
        
        self.hi_zone.swatch_lbl.hue = eff_hi_hue
        self.hi_zone.swatch_lbl.sat = state["highlight_sat"]
        
        self.sh_zone.swatch_lbl.update_color(eff_sh_hue, state["shadow_sat"])
        self.mid_zone.swatch_lbl.update_color(eff_mid_hue, state["midtone_sat"])
        self.hi_zone.swatch_lbl.update_color(eff_hi_hue, state["highlight_sat"])
        
        graded_center = apply_grading(self.proxy_image, state)
        self.single_preview.set_image(graded_center)

    def on_sh_wheel_changed(self, hue: float, sat: float) -> None:
        self.sh_zone.sat_slider.blockSignals(True)
        self.sh_zone.sat_slider.setValue(int(sat * 100.0))
        self.sh_zone.sat_slider.blockSignals(False)
        self.on_manual_slider_changed()

    def on_mid_wheel_changed(self, hue: float, sat: float) -> None:
        self.mid_zone.sat_slider.blockSignals(True)
        self.mid_zone.sat_slider.setValue(int(sat * 100.0))
        self.mid_zone.sat_slider.blockSignals(False)
        self.on_manual_slider_changed()

    def on_hi_wheel_changed(self, hue: float, sat: float) -> None:
        self.hi_zone.sat_slider.blockSignals(True)
        self.hi_zone.sat_slider.setValue(int(sat * 100.0))
        self.hi_zone.sat_slider.blockSignals(False)
        self.on_manual_slider_changed()

    def reset_wheel_hue(self, wheel: ColorWheel, default_hue: float) -> None:
        wheel.set_color(default_hue, wheel.sat)
        self.on_manual_slider_changed()
        self.on_discrete_action()

    def reset_slider(self, slider: QSlider, value: int) -> None:
        slider.setValue(value)
        self.on_discrete_action()

    def on_intensity_changed(self, value: int) -> None:
        step_size = value / 100.0
        self.intensity_label.setText(f" Intensity: {step_size:.2f}x ")
        
        if self.master_image is None:
            return
            
        snapshot = self.state_manager.capture_snapshot()
        self.render_view_from_memento(snapshot)

    # ----------------------------------------------------
    # File I/O MRU
    # ----------------------------------------------------
    def update_recent_directories_menu(self) -> None:
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
        self.open_file(start_dir=start_dir)

    def open_file(self, start_dir: Optional[str] = None) -> None:
        if start_dir is None:
            start_dir = self.mru_manager.get_most_recent_directory()

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open 16-bit TIFF",
            start_dir,
            "TIFF Images (*.tif *.tiff);;All Files (*)"
        )
        if file_path:
            try:
                from nss.image_utils import load_tiff_to_float32, downsample_image
                
                self.master_image, self.tiff_obj = load_tiff_to_float32(file_path)
                self.proxy_image = downsample_image(self.master_image, max_dim=1200)
                
                self.mru_manager.add_path(file_path)
                self.update_recent_directories_menu()
                
                # Push initial starting state onto the StateManager
                initial_state = create_default_state("Monochromatic")
                self.apply_state_node_to_widgets(initial_state)
                
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
                self.statusBar().showMessage(f"Loaded {file_path} (Master: {shape_str}, Proxy: {proxy_str})")
            except Exception as e:
                self.statusBar().showMessage(f"Error loading image: {str(e)}")

    def save_file(self) -> None:
        if self.master_image is None or self.tiff_obj is None:
            return

        start_dir = self.mru_manager.get_most_recent_directory()

        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save 16-bit TIFF",
            start_dir,
            "TIFF Images (*.tif *.tiff);;All Files (*)"
        )
        if save_path:
            try:
                snapshot = self.state_manager.capture_snapshot()
                center_state = self.rebuild_state_node_from_memento(snapshot)
                
                graded_image = apply_grading(self.master_image, center_state)
                self.tiff_obj.array = graded_image
                self.tiff_obj.saveas(save_path)
                
                self.mru_manager.add_path(save_path)
                self.update_recent_directories_menu()
                self.statusBar().showMessage(f"Successfully saved graded image to {save_path}")
            except Exception as e:
                self.statusBar().showMessage(f"Error saving image: {str(e)}")

    # ----------------------------------------------------
    # Swatch Grid Controls Persistence & Loaders
    # ----------------------------------------------------
    def load_custom_colors(self, zone: str) -> list:
        settings = QSettings("NSS", "ColorGradingExplorer")
        data = settings.value(f"history_grid_{zone}")
        if data:
            try:
                res = json.loads(str(data))
                if len(res) == 16:
                    return res
                if len(res) > 16:
                    return res[:16]
                else:
                    defaults = self.get_default_colors(zone)
                    return res + defaults[len(res):]
            except Exception:
                pass
        return self.get_default_colors(zone)

    def get_default_colors(self, zone: str) -> list:
        if zone == "shadows":
            return [
                [240.0, 0.0], [240.0, 0.1], [240.0, 0.2], [240.0, 0.3],
                [240.0, 0.4], [240.0, 0.5], [200.0, 0.1], [200.0, 0.2],
                [200.0, 0.3], [200.0, 0.4], [270.0, 0.1], [270.0, 0.2],
                [270.0, 0.3], [270.0, 0.4], [180.0, 0.1], [180.0, 0.2]
            ]
        elif zone == "midtones":
            return [
                [120.0, 0.0], [120.0, 0.1], [120.0, 0.2], [120.0, 0.3],
                [120.0, 0.4], [120.0, 0.5], [80.0, 0.1], [80.0, 0.2],
                [80.0, 0.3], [80.0, 0.4], [160.0, 0.1], [160.0, 0.2],
                [160.0, 0.3], [160.0, 0.4], [100.0, 0.1], [100.0, 0.2]
            ]
        else:
            return [
                [60.0, 0.0], [60.0, 0.1], [60.0, 0.2], [60.0, 0.3],
                [60.0, 0.4], [60.0, 0.5], [30.0, 0.1], [30.0, 0.2],
                [30.0, 0.3], [30.0, 0.4], [90.0, 0.1], [90.0, 0.2],
                [90.0, 0.3], [90.0, 0.4], [45.0, 0.1], [45.0, 0.2]
            ]

    def add_custom_color(self, zone: str, hue: float, sat: float) -> None:
        h = float(round(hue)) % 360
        s = float(round(sat, 2))
        
        history = self.load_custom_colors(zone)
        new_history = []
        for item in history:
            if item is None or not isinstance(item, list) or len(item) < 2:
                continue
            if abs(item[0] - h) < 1.0 and abs(item[1] - s) < 0.01:
                continue
            new_history.append(item)
            
        new_history.insert(0, [h, s])
        new_history = new_history[:16]
        
        settings = QSettings("NSS", "ColorGradingExplorer")
        settings.setValue(f"history_grid_{zone}", json.dumps(new_history))
        self.update_history_swatches_ui(zone, new_history)

    def update_history_swatches_ui(self, zone: str, history: list) -> None:
        if zone == "shadows":
            grid = self.sh_zone.history_grid
        elif zone == "midtones":
            grid = self.mid_zone.history_grid
        else:
            grid = self.hi_zone.history_grid

        while grid.count():
            item = grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        for i, item in enumerate(history):
            row = i // 8
            col = i % 8
            if item is None or not isinstance(item, list) or len(item) < 2:
                h, s = None, None
            else:
                h, s = item[0], item[1]
                
            swatch = ColorSwatch(h, s, self)
            swatch.index = i
            swatch.clicked.connect(lambda hue, sat, z=zone: self.on_swatch_clicked(z, hue, sat))
            swatch.overwritten.connect(lambda idx, hue, sat, z=zone: self.on_swatch_overwritten(z, idx, hue, sat))
            swatch.edited.connect(lambda idx, hue, sat, z=zone: self.on_swatch_overwritten(z, idx, hue, sat))
            swatch.forgotten.connect(lambda idx, z=zone: self.on_swatch_forgotten(z, idx))
            grid.addWidget(swatch, row, col)

    def on_swatch_clicked(self, zone: str, hue: float, sat: float) -> None:
        if zone == "shadows":
            target_zone = self.sh_zone
        elif zone == "midtones":
            target_zone = self.mid_zone
        else:
            target_zone = self.hi_zone

        target_zone.wheel.set_color(hue, sat)
        target_zone.sat_slider.setValue(int(sat * 100.0))
        self.on_discrete_action()

    def on_swatch_overwritten(self, zone: str, index: int, hue: float, sat: float) -> None:
        history = self.load_custom_colors(zone)
        if index < len(history):
            history[index] = [float(round(hue)) % 360, float(round(sat, 2))]
            settings = QSettings("NSS", "ColorGradingExplorer")
            settings.setValue(f"history_grid_{zone}", json.dumps(history))

    def on_swatch_forgotten(self, zone: str, index: int) -> None:
        history = self.load_custom_colors(zone)
        if index < len(history):
            history[index] = None
            settings = QSettings("NSS", "ColorGradingExplorer")
            settings.setValue(f"history_grid_{zone}", json.dumps(history))
            self.update_history_swatches_ui(zone, history)

    def update_zone_swatches(self, state: StateNode) -> None:
        is_comp = state.get("harmony_mode") == "Complementary"

        # Shadows
        sh_hue = state.get("shadow_hue", 240.0)
        sh_sat = 0.15 if is_comp else state.get("shadow_sat", 0.0)
        sh_color = QColor.fromHslF(sh_hue / 360.0, sh_sat, 1.0 - 0.5 * sh_sat)
        sh_pix = QPixmap(24, 24)
        sh_pix.fill(sh_color)
        self.sh_zone.swatch_lbl.setPixmap(sh_pix)
        self.sh_zone.swatch_lbl.hue = sh_hue
        self.sh_zone.swatch_lbl.sat = sh_sat
        
        # Midtones
        mid_hue = state.get("midtone_hue", 120.0)
        mid_sat = state.get("midtone_sat", 0.0)
        mid_color = QColor.fromHslF(mid_hue / 360.0, mid_sat, 1.0 - 0.5 * mid_sat)
        mid_pix = QPixmap(24, 24)
        mid_pix.fill(mid_color)
        self.mid_zone.swatch_lbl.setPixmap(mid_pix)
        self.mid_zone.swatch_lbl.hue = mid_hue
        self.mid_zone.swatch_lbl.sat = mid_sat

        # Highlights
        hi_hue = state.get("highlight_hue", 60.0)
        hi_sat = 0.15 if is_comp else state.get("highlight_sat", 0.0)
        hi_color = QColor.fromHslF(hi_hue / 360.0, hi_sat, 1.0 - 0.5 * hi_sat)
        hi_pix = QPixmap(24, 24)
        hi_pix.fill(hi_color)
        self.hi_zone.swatch_lbl.setPixmap(hi_pix)
        self.hi_zone.swatch_lbl.hue = hi_hue
        self.hi_zone.swatch_lbl.sat = hi_sat

    def closeEvent(self, event) -> None:
        if self.grading_worker is not None:
            if self.grading_worker.isRunning():
                self.grading_worker.cancel()
                self.grading_worker.wait()
        super().closeEvent(event)
