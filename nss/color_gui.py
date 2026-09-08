import numpy as np
from typing import List, Optional
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
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread
from PyQt6.QtGui import QMouseEvent, QAction, QImage, QPixmap, QColor

from nss.utils import TiffFile
from nss.color_math import (
    StateNode,
    MutationAxis,
    create_default_state,
    apply_grading,
    generate_mutations,
)
from nss.history import HistoryManager
from nss.mru import MruManager

class ImageContainer(QLabel):
    """
    A custom clickable image container for the 3x3 color grading explorer.
    """
    clicked = pyqtSignal(int)  # Emits the index of the container (0-8)

    def __init__(self, index: int, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.index: int = index
        self.master_pixmap: Optional[QPixmap] = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(150, 150)
        
        # Initial aesthetic styling
        self.setFrameStyle(QFrame.Shape.Panel | QFrame.Shadow.Sunken)
        self.setLineWidth(2)
        self.setStyleSheet(
            "background-color: #2e2e2e; color: #d0d0d0; "
            "border: 1px solid #555555; border-radius: 4px;"
        )
        self.setText(f"Image {index}")

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.index)

    def set_active(self, is_active: bool) -> None:
        """
        Sets active (center) or inactive style for the container.
        """
        if is_active:
            self.setStyleSheet(
                "background-color: #1a3a1a; color: #40ff40; "
                "border: 3px solid #00ff00; border-radius: 4px;"
            )
            self.setFrameStyle(QFrame.Shape.Panel | QFrame.Shadow.Raised)
        else:
            self.setStyleSheet(
                "background-color: #2e2e2e; color: #d0d0d0; "
                "border: 1px solid #555555; border-radius: 4px;"
            )
            self.setFrameStyle(QFrame.Shape.Panel | QFrame.Shadow.Sunken)

    def set_image(self, arr: Optional[np.ndarray]) -> None:
        """
        Converts a NumPy array (Grayscale, RGB, or float32) to QPixmap and displays it.
        If arr is None, clears the image and displays placeholder text.
        """
        if arr is None:
            self.master_pixmap = None
            self.clear()
            if self.index == 4:
                self.setText("Center (Active)")
            else:
                self.setText(f"Mutation {self.index}")
            return

        try:
            # Handle float32 automatic display mapping
            if arr.dtype == np.float32:
                from nss.image_utils import to_uint8_display
                arr = to_uint8_display(arr)
                
            if arr.ndim == 2:
                h, w = arr.shape
                # Use tobytes() to ensure memory safety
                qimg = QImage(arr.tobytes(), w, h, w, QImage.Format.Format_Grayscale8)
            elif arr.ndim == 3:
                h, w, c = arr.shape
                if c == 3:
                    qimg = QImage(arr.tobytes(), w, h, w * 3, QImage.Format.Format_RGB888)
                elif c == 4:
                    qimg = QImage(arr.tobytes(), w, h, w * 4, QImage.Format.Format_RGBA8888)
                else:
                    raise ValueError(f"Unsupported number of channels: {c}")
            else:
                raise ValueError(f"Unsupported dimensions: {arr.ndim}")

            self.master_pixmap = QPixmap.fromImage(qimg)
            self.update_display_pixmap()
        except Exception as e:
            self.setText(f"Error: {str(e)}")

    def update_display_pixmap(self) -> None:
        """
        Re-scales and displays the master pixmap inside the label, maintaining aspect ratio.
        """
        if self.master_pixmap is not None and not self.master_pixmap.isNull():
            scaled = self.master_pixmap.scaled(
                self.width() - 8,
                self.height() - 8,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.setPixmap(scaled)

    def resizeEvent(self, event) -> None:
        """
        Handle container resizing by scaling the display image.
        """
        super().resizeEvent(event)
        self.update_display_pixmap()


class ResetLabel(QLabel):
    """
    A custom QLabel that detects double clicks and emits a signal to reset controls.
    """
    doubleClicked = pyqtSignal()

    def mouseDoubleClickEvent(self, event) -> None:
        self.doubleClicked.emit()


class ClickableSwatchLabel(QLabel):
    """
    A custom QLabel that acts as a clickable color swatch to open QColorDialog.
    """
    clicked = pyqtSignal()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()


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
        """
        Signals the worker thread to stop processing.
        """
        self.is_cancelled = True

    def run(self) -> None:
        # Determine index processing sequence (render center index 4 first unless skipped)
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
                # Core NumPy and OpenCV image processing grading happens in background
                graded = apply_grading(self.working_image, self.states[i])
                if self.is_cancelled:
                    return
                self.progress.emit(i, graded)
            except Exception:
                pass
        self.finished_all.emit()


class MainWindow(QMainWindow):
    """
    The Main Window for the 16-bit Color Grading Explorer with Lightroom-style controls.
    """
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("16-bit Color Grading Explorer")
        self.resize(1300, 850)

        # State Variables
        self.master_image: Optional[np.ndarray] = None
        self.proxy_image: Optional[np.ndarray] = None
        self.tiff_obj: Optional[TiffFile] = None
        self.history_manager: HistoryManager = HistoryManager()
        self.mru_manager: MruManager = MruManager()
        self.grid_states: Optional[List[StateNode]] = None
        self.grading_worker: Optional[GradingWorker] = None

        # Setup UI Components
        self.init_ui()

    def init_ui(self) -> None:
        # Create Menu Bar
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("&File")

        open_action = QAction("&Open...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.setStatusTip("Open a 16-bit TIFF image")
        open_action.triggered.connect(lambda: self.open_file())
        file_menu.addAction(open_action)

        # Recent Directories Submenu
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

        # 1. Create Toolbar with Back/Forward, Step-size slider and History label
        toolbar = QToolBar("Navigation and Controls")
        self.addToolBar(toolbar)

        # "Back" button
        self.back_action = QAction("Back", self)
        self.back_action.setToolTip("Go to previous color state (Undo)")
        self.back_action.setEnabled(False)
        self.back_action.triggered.connect(self.on_back_clicked)
        toolbar.addAction(self.back_action)

        # History Index Display Label
        self.history_label = QLabel("Step: 0 of 0")
        self.history_label.setStyleSheet(
            "font-weight: bold; margin-left: 10px; margin-right: 10px; color: #a0a0a0;"
        )
        toolbar.addWidget(self.history_label)

        # "Forward" button
        self.forward_action = QAction("Forward", self)
        self.forward_action.setToolTip("Go to next color state (Redo)")
        self.forward_action.setEnabled(False)
        self.forward_action.triggered.connect(self.on_forward_clicked)
        toolbar.addAction(self.forward_action)

        toolbar.addSeparator()

        # Mutation Intensity (Step Size) Slider - Defaults to 0.20x magnitude (subtle steps)
        self.intensity_label = QLabel(" Intensity: 0.20x ")
        self.intensity_label.setStyleSheet("color: #a0a0a0; font-weight: bold;")
        toolbar.addWidget(self.intensity_label)

        self.intensity_slider = QSlider(Qt.Orientation.Horizontal)
        self.intensity_slider.setRange(10, 200)
        self.intensity_slider.setValue(20)  # Default value 20 represents 0.20
        self.intensity_slider.setFixedWidth(120)
        self.intensity_slider.setToolTip("Slide to scale mutation offset magnitude (0.10x to 2.00x)")
        self.intensity_slider.valueChanged.connect(self.on_intensity_changed)
        toolbar.addWidget(self.intensity_slider)

        # 2. Setup Central Widget with QHBoxLayout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(15)

        # Left Side: 3x3 Grid Layout
        grid_widget = QWidget()
        grid_layout = QGridLayout(grid_widget)
        grid_layout.setContentsMargins(0, 0, 0, 0)
        grid_layout.setSpacing(10)
        main_layout.addWidget(grid_widget, stretch=4)

        self.containers: List[ImageContainer] = []
        for i in range(9):
            container = ImageContainer(i, self)
            container.clicked.connect(self.on_container_clicked)
            self.containers.append(container)

            # Map index to row and col
            row = i // 3
            col = i % 3
            grid_layout.addWidget(container, row, col)

            # Mark center container (index 4) as active
            if i == 4:
                container.set_active(True)
                container.setText("Center (Active)")
            else:
                container.set_active(False)
                container.setText(f"Mutation {i}")

        # Right Side: Structured Lightroom-Style Manual 3-Way Color Grading Control Dock
        self.inspector_panel = QFrame()
        self.inspector_panel.setFrameShape(QFrame.Shape.StyledPanel)
        self.inspector_panel.setFrameShadow(QFrame.Shadow.Raised)
        self.inspector_panel.setFixedWidth(300)
        self.inspector_panel.setStyleSheet(
            "background-color: #252525; border: 1px solid #444; border-radius: 4px;"
        )
        
        inspector_layout = QVBoxLayout(self.inspector_panel)
        inspector_layout.setContentsMargins(15, 15, 15, 15)
        inspector_layout.setSpacing(12)
        
        main_layout.addWidget(self.inspector_panel, stretch=1)

        # Dock Title Header
        header = QLabel("3-WAY COLOR GRADING")
        header.setStyleSheet("font-weight: bold; font-size: 14px; color: #40ff40; border: none; margin-bottom: 5px;")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        inspector_layout.addWidget(header)

        # 3-Zone Tabs Setup
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet(
            "QTabWidget::pane { border: 1px solid #444; background-color: #2b2b2b; }"
            "QTabBar::tab { background-color: #333; color: #aaa; padding: 6px 12px; border: 1px solid #444; }"
            "QTabBar::tab:selected { background-color: #2b2b2b; color: #eee; font-weight: bold; }"
        )
        inspector_layout.addWidget(self.tabs)

        # Tab Helper Function
        def create_zone_tab(title_prefix: str, default_h: int) -> tuple[QWidget, QSlider, ResetLabel, QSlider, ResetLabel, QSlider, ResetLabel, ClickableSwatchLabel]:
            tab_widget = QWidget()
            tab_layout = QVBoxLayout(tab_widget)
            tab_layout.setContentsMargins(12, 12, 12, 12)
            tab_layout.setSpacing(8)

            # Top Header Row with Swatch Label
            top_row = QHBoxLayout()
            title_lbl = QLabel(f"{title_prefix} Color parameters")
            title_lbl.setStyleSheet("font-weight: bold; color: #ddd; font-size: 11px;")
            swatch_lbl = ClickableSwatchLabel()
            swatch_lbl.setFixedSize(36, 14)
            swatch_lbl.setStyleSheet("border: 1px solid #555; border-radius: 2px;")
            swatch_lbl.setCursor(Qt.CursorShape.PointingHandCursor)
            swatch_lbl.setToolTip("Click to open color picker wheel")
            top_row.addWidget(title_lbl)
            top_row.addStretch()
            top_row.addWidget(swatch_lbl)
            tab_layout.addLayout(top_row)

            # Hue Control (Double click label to reset to default)
            hue_lbl = ResetLabel(f"Hue: {default_h}°")
            hue_lbl.setStyleSheet("color: #ccc; font-size: 11px;")
            hue_lbl.setToolTip("Double-click to reset Hue to default")
            hue_sld = QSlider(Qt.Orientation.Horizontal)
            hue_sld.setRange(0, 360)
            hue_sld.setValue(default_h)
            hue_sld.valueChanged.connect(self.on_manual_slider_changed)

            # Saturation Control
            sat_lbl = ResetLabel("Sat: 0.00")
            sat_lbl.setStyleSheet("color: #ccc; font-size: 11px;")
            sat_lbl.setToolTip("Double-click to reset Saturation to 0.00")
            sat_sld = QSlider(Qt.Orientation.Horizontal)
            sat_sld.setRange(0, 100)
            sat_sld.setValue(0)
            sat_sld.valueChanged.connect(self.on_manual_slider_changed)

            # Luminance Control
            lum_lbl = ResetLabel("Luma: 0.00")
            lum_lbl.setStyleSheet("color: #ccc; font-size: 11px;")
            lum_lbl.setToolTip("Double-click to reset Luminance to 0.00")
            lum_sld = QSlider(Qt.Orientation.Horizontal)
            lum_sld.setRange(-100, 100)
            lum_sld.setValue(0)
            lum_sld.valueChanged.connect(self.on_manual_slider_changed)

            tab_layout.addWidget(hue_lbl)
            tab_layout.addWidget(hue_sld)
            tab_layout.addWidget(sat_lbl)
            tab_layout.addWidget(sat_sld)
            tab_layout.addWidget(lum_lbl)
            tab_layout.addWidget(lum_sld)
            tab_layout.addStretch()

            return tab_widget, hue_sld, hue_lbl, sat_sld, sat_lbl, lum_sld, lum_lbl, swatch_lbl

        # Shadows Tab (default blue 240)
        sh_tab, self.sh_hue_slider, self.sh_hue_lbl, self.sh_sat_slider, self.sh_sat_lbl, self.sh_light_slider, self.sh_light_lbl, self.sh_swatch_label = create_zone_tab("Shadows", 240)
        self.tabs.addTab(sh_tab, "Shadows")
        self.sh_hue_lbl.doubleClicked.connect(lambda: self.reset_slider(self.sh_hue_slider, 240))
        self.sh_sat_lbl.doubleClicked.connect(lambda: self.reset_slider(self.sh_sat_slider, 0))
        self.sh_light_lbl.doubleClicked.connect(lambda: self.reset_slider(self.sh_light_slider, 0))
        self.sh_swatch_label.clicked.connect(self.pick_shadow_color)

        # Midtones Tab (default green 120)
        mid_tab, self.mid_hue_slider, self.mid_hue_lbl, self.mid_sat_slider, self.mid_sat_lbl, self.mid_light_slider, self.mid_light_lbl, self.mid_swatch_label = create_zone_tab("Midtones", 120)
        self.tabs.addTab(mid_tab, "Midtones")
        self.mid_hue_lbl.doubleClicked.connect(lambda: self.reset_slider(self.mid_hue_slider, 120))
        self.mid_sat_lbl.doubleClicked.connect(lambda: self.reset_slider(self.mid_sat_slider, 0))
        self.mid_light_lbl.doubleClicked.connect(lambda: self.reset_slider(self.mid_light_slider, 0))
        self.mid_swatch_label.clicked.connect(self.pick_midtone_color)

        # Highlights Tab (default yellow/gold 60)
        hi_tab, self.hi_hue_slider, self.hi_hue_lbl, self.hi_sat_slider, self.hi_sat_lbl, self.hi_light_slider, self.hi_light_lbl, self.hi_swatch_label = create_zone_tab("Highlights", 60)
        self.tabs.addTab(hi_tab, "Highlights")
        self.hi_hue_lbl.doubleClicked.connect(lambda: self.reset_slider(self.hi_hue_slider, 60))
        self.hi_sat_lbl.doubleClicked.connect(lambda: self.reset_slider(self.hi_sat_slider, 0))
        self.hi_light_lbl.doubleClicked.connect(lambda: self.reset_slider(self.hi_light_slider, 0))
        self.hi_swatch_label.clicked.connect(self.pick_highlight_color)

        # Master Global Controls Section
        master_widget = QWidget()
        master_widget.setStyleSheet("border: none; background: transparent;")
        master_widget.setContentsMargins(0, 0, 0, 0)
        master_layout = QVBoxLayout(master_widget)
        master_layout.setSpacing(10)
        inspector_layout.addWidget(master_widget)

        # Title
        m_title = QLabel("MASTER DOCK CONTROLS")
        m_title.setStyleSheet("font-weight: bold; color: #a0a0a0; border: none; border-bottom: 1px solid #444; padding-bottom: 2px;")
        master_layout.addWidget(m_title)

        # Master Blending
        self.blending_lbl = ResetLabel("Blending: 0.50")
        self.blending_lbl.setStyleSheet("color: #ccc; font-size: 11px;")
        self.blending_lbl.setToolTip("Double-click to reset Blending to 0.50")
        self.blending_slider = QSlider(Qt.Orientation.Horizontal)
        self.blending_slider.setRange(0, 100)
        self.blending_slider.setValue(50)
        self.blending_slider.valueChanged.connect(self.on_manual_slider_changed)
        self.blending_lbl.doubleClicked.connect(lambda: self.reset_slider(self.blending_slider, 50))
        
        # Master Balance
        self.balance_lbl = ResetLabel("Balance: 0.00")
        self.balance_lbl.setStyleSheet("color: #ccc; font-size: 11px;")
        self.balance_lbl.setToolTip("Double-click to reset Balance to 0.00")
        self.balance_slider = QSlider(Qt.Orientation.Horizontal)
        self.balance_slider.setRange(-100, 100)
        self.balance_slider.setValue(0)
        self.balance_slider.valueChanged.connect(self.on_manual_slider_changed)
        self.balance_lbl.doubleClicked.connect(lambda: self.reset_slider(self.balance_slider, 0))

        master_layout.addWidget(self.blending_lbl)
        master_layout.addWidget(self.blending_slider)
        master_layout.addWidget(self.balance_lbl)
        master_layout.addWidget(self.balance_slider)

        inspector_layout.addStretch()

        # 3. Setup Status Bar Message via built-in QMainWindow statusBar()
        self.statusBar().showMessage("Ready. Load a 16-bit TIFF image to begin.")

    def reset_slider(self, slider: QSlider, value: int) -> None:
        """
        Helper method to reset a slider back to its default value on label double-click.
        """
        slider.setValue(value)

    def on_manual_slider_changed(self) -> None:
        """
        Reads values from all manual grading sliders, updates active center StateNode,
        re-renders center image preview immediately, and updates swatches.
        Isolates manual changes exclusively to the center image to preserve snappiness.
        """
        if self.master_image is None:
            return

        state = self.history_manager.get_current_state()
        if state is None:
            return

        # 1. Read values from sliders and update state node
        state["shadow_hue"] = float(self.sh_hue_slider.value())
        state["shadow_sat"] = self.sh_sat_slider.value() / 100.0
        state["shadow_light"] = self.sh_light_slider.value() / 100.0

        state["midtone_hue"] = float(self.mid_hue_slider.value())
        state["midtone_sat"] = self.mid_sat_slider.value() / 100.0
        state["midtone_light"] = self.mid_light_slider.value() / 100.0

        state["highlight_hue"] = float(self.hi_hue_slider.value())
        state["highlight_sat"] = self.hi_sat_slider.value() / 100.0
        state["highlight_light"] = self.hi_light_slider.value() / 100.0

        state["blending"] = self.blending_slider.value() / 100.0
        state["balance"] = self.balance_slider.value() / 100.0

        # Update text readouts
        self.sh_hue_lbl.setText(f"Hue: {state['shadow_hue']:.0f}°")
        self.sh_sat_lbl.setText(f"Sat: {state['shadow_sat']:.2f}")
        self.sh_light_lbl.setText(f"Luma: {state['shadow_light']:+.2f}")

        self.mid_hue_lbl.setText(f"Hue: {state['midtone_hue']:.0f}°")
        self.mid_sat_lbl.setText(f"Sat: {state['midtone_sat']:.2f}")
        self.mid_light_lbl.setText(f"Luma: {state['midtone_light']:+.2f}")

        self.hi_hue_lbl.setText(f"Hue: {state['highlight_hue']:.0f}°")
        self.hi_sat_lbl.setText(f"Sat: {state['highlight_sat']:.2f}")
        self.hi_light_lbl.setText(f"Luma: {state['highlight_light']:+.2f}")

        self.blending_lbl.setText(f"Blending: {state['blending']:.2f}")
        self.balance_lbl.setText(f"Balance: {state['balance']:+.2f}")

        # 2. Update visual color swatches live
        self.update_zone_swatches(state)

        # 3. Recalculate and update the center tile immediately!
        # Applying color grading math to proxy_image for real-time snappy feedback
        if self.proxy_image is not None:
            graded_center = apply_grading(self.proxy_image, state)
            self.containers[4].set_image(graded_center)

    def pick_shadow_color(self) -> None:
        """
        Opens QColorDialog to select a shadow tint color, and updates the state.
        """
        self.pick_zone_color("shadow_hue", "shadow_sat", self.sh_hue_slider, self.sh_sat_slider, 240.0)

    def pick_midtone_color(self) -> None:
        """
        Opens QColorDialog to select a midtone tint color, and updates the state.
        """
        self.pick_zone_color("midtone_hue", "midtone_sat", self.mid_hue_slider, self.mid_sat_slider, 120.0)

    def pick_highlight_color(self) -> None:
        """
        Opens QColorDialog to select a highlight tint color, and updates the state.
        """
        self.pick_zone_color("highlight_hue", "highlight_sat", self.hi_hue_slider, self.hi_sat_slider, 60.0)

    def pick_zone_color(self, hue_key: str, sat_key: str, hue_slider: QSlider, sat_slider: QSlider, default_hue: float) -> None:
        """
        Generic helper to open QColorDialog and apply HSL values to sliders.
        """
        state = self.history_manager.get_current_state()
        if state is None:
            return

        current_hue = state.get(hue_key, default_hue)
        current_sat = state.get(sat_key, 0.0)
        initial_color = QColor.fromHslF(current_hue / 360.0, current_sat, 0.5)

        color = QColorDialog.getColor(initial_color, self, "Select Zone Tint Color")
        if color.isValid():
            h, s, l, a = color.getHslF()
            hue = h * 360.0 if h >= 0.0 else current_hue
            sat = s
            
            # Sync sliders
            self.block_manual_signals(True)
            hue_slider.setValue(int(hue))
            sat_slider.setValue(int(sat * 100.0))
            self.block_manual_signals(False)
            
            # Recalculate
            self.on_manual_slider_changed()

    def update_zone_swatches(self, state: StateNode) -> None:
        """
        Updates the three tab color swatches live based on HSL.
        """
        from PyQt6.QtGui import QColor, QPixmap
        
        is_comp = state.get("harmony_mode") == "Complementary"

        # Shadows
        sh_hue = state.get("shadow_hue", 240.0)
        sh_sat = 0.15 if is_comp else state.get("shadow_sat", 0.0)
        sh_color = QColor.fromHslF(sh_hue / 360.0, sh_sat, 0.5)
        sh_pix = QPixmap(36, 14)
        sh_pix.fill(sh_color)
        self.sh_swatch_label.setPixmap(sh_pix)
        
        # Midtones
        mid_hue = state.get("midtone_hue", 120.0)
        mid_sat = state.get("midtone_sat", 0.0)
        mid_color = QColor.fromHslF(mid_hue / 360.0, mid_sat, 0.5)
        mid_pix = QPixmap(36, 14)
        mid_pix.fill(mid_color)
        self.mid_swatch_label.setPixmap(mid_pix)

        # Highlights
        hi_hue = state.get("highlight_hue", 60.0)
        hi_sat = 0.15 if is_comp else state.get("highlight_sat", 0.0)
        hi_color = QColor.fromHslF(hi_hue / 360.0, hi_sat, 0.5)
        hi_pix = QPixmap(36, 14)
        hi_pix.fill(hi_color)
        self.hi_swatch_label.setPixmap(hi_pix)

    def sync_sliders_with_state(self, state: StateNode) -> None:
        """
        Synchronizes all UI sliders and swatches to match the active StateNode parameters.
        Blocks signals to avoid triggering recursive render events during synchronization.
        """
        self.block_manual_signals(True)

        self.sh_hue_slider.setValue(int(state.get("shadow_hue", 240.0)))
        self.sh_sat_slider.setValue(int(state.get("shadow_sat", 0.0) * 100.0))
        self.sh_light_slider.setValue(int(state.get("shadow_light", 0.0) * 100.0))

        self.mid_hue_slider.setValue(int(state.get("midtone_hue", 120.0)))
        self.mid_sat_slider.setValue(int(state.get("midtone_sat", 0.0) * 100.0))
        self.mid_light_slider.setValue(int(state.get("midtone_light", 0.0) * 100.0))

        self.hi_hue_slider.setValue(int(state.get("highlight_hue", 60.0)))
        self.hi_sat_slider.setValue(int(state.get("highlight_sat", 0.0) * 100.0))
        self.hi_light_slider.setValue(int(state.get("highlight_light", 0.0) * 100.0))

        self.blending_slider.setValue(int(state.get("blending", 0.5) * 100.0))
        self.balance_slider.setValue(int(state.get("balance", 0.0) * 100.0))

        # Update labels readouts
        self.sh_hue_lbl.setText(f"Hue: {state.get('shadow_hue', 240.0):.0f}°")
        self.sh_sat_lbl.setText(f"Sat: {state.get('shadow_sat', 0.0):.2f}")
        self.sh_light_lbl.setText(f"Luma: {state.get('shadow_light', 0.0):+.2f}")

        self.mid_hue_lbl.setText(f"Hue: {state.get('midtone_hue', 120.0):.0f}°")
        self.mid_sat_lbl.setText(f"Sat: {state.get('midtone_sat', 0.0):.2f}")
        self.mid_light_lbl.setText(f"Luma: {state.get('midtone_light', 0.0):+.2f}")

        self.hi_hue_lbl.setText(f"Hue: {state.get('highlight_hue', 60.0):.0f}°")
        self.hi_sat_lbl.setText(f"Sat: {state.get('highlight_sat', 0.0):.2f}")
        self.hi_light_lbl.setText(f"Luma: {state.get('highlight_light', 0.0):+.2f}")

        self.blending_lbl.setText(f"Blending: {state.get('blending', 0.5):.2f}")
        self.balance_lbl.setText(f"Balance: {state.get('balance', 0.0):+.2f}")

        self.update_zone_swatches(state)

        self.block_manual_signals(False)

    def block_manual_signals(self, block: bool) -> None:
        """
        Helper to block/unblock signals on all manual controls.
        """
        self.sh_hue_slider.blockSignals(block)
        self.sh_sat_slider.blockSignals(block)
        self.sh_light_slider.blockSignals(block)

        self.mid_hue_slider.blockSignals(block)
        self.mid_sat_slider.blockSignals(block)
        self.mid_light_slider.blockSignals(block)

        self.hi_hue_slider.blockSignals(block)
        self.hi_sat_slider.blockSignals(block)
        self.hi_light_slider.blockSignals(block)

        self.blending_slider.blockSignals(block)
        self.balance_slider.blockSignals(block)

    def update_recent_directories_menu(self) -> None:
        """
        Clears and repopulates the Recent Directories submenu from MruManager paths.
        """
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
        """
        Convenience method to trigger the open file dialog at a specific path.
        """
        self.open_file(start_dir=start_dir)

    def open_file(self, start_dir: Optional[str] = None) -> None:
        """
        Opens a file dialog to load a 16-bit TIFF file.
        """
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
                
                # Load using the scaling pipeline (high resolution master)
                self.master_image, self.tiff_obj = load_tiff_to_float32(file_path)
                
                # Automatically generate a downsampled working proxy (1200px max dimension)
                self.proxy_image = downsample_image(self.master_image, max_dim=1200)
                
                # Update MRU directories
                self.mru_manager.add_path(file_path)
                self.update_recent_directories_menu()
                
                # Initialize history with default state
                self.history_manager.clear()
                initial_state = create_default_state("Monochromatic")
                self.history_manager.push_state(initial_state)
                
                # Render grid
                self.render_grid_from_current_state()
                
                # Update status bar
                shape_str = "x".join(map(str, self.master_image.shape))
                proxy_str = "x".join(map(str, self.proxy_image.shape))
                self.statusBar().showMessage(f"Loaded {file_path} (Master: {shape_str}, Proxy: {proxy_str})")
            except Exception as e:
                self.statusBar().showMessage(f"Error loading image: {str(e)}")

    def save_file(self) -> None:
        """
        Saves the current color graded image as a 16-bit TIFF.
        """
        if self.master_image is None or self.tiff_obj is None or self.grid_states is None:
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
                # Apply current center grading to the full-resolution master image!
                center_state = self.grid_states[4]
                graded_image = apply_grading(self.master_image, center_state)
                
                # Store the graded image onto our tiff object
                self.tiff_obj.array = graded_image
                
                # Save using the original metadata on tiff_obj
                self.tiff_obj.saveas(save_path)
                
                # Update MRU directories
                self.mru_manager.add_path(save_path)
                self.update_recent_directories_menu()
                
                self.statusBar().showMessage(f"Successfully saved graded image to {save_path}")
            except Exception as e:
                self.statusBar().showMessage(f"Error saving image: {str(e)}")

    def on_back_clicked(self) -> None:
        """
        Navigates back in history.
        """
        if self.history_manager.can_undo():
            self.history_manager.undo()
            self.render_grid_from_current_state()
            self.statusBar().showMessage("Undo action.")

    def on_forward_clicked(self) -> None:
        """
        Navigates forward in history.
        """
        if self.history_manager.can_redo():
            self.history_manager.redo()
            self.render_grid_from_current_state()
            self.statusBar().showMessage("Redo action.")

    def on_harmony_mode_changed(self, text: str) -> None:
        """
        Handler for when the user selects a different harmony mode from the dropdown.
        Updates active state's harmony mode and triggers re-generation of outer tiles,
        keeping the center image completely untouched.
        """
        if self.master_image is None:
            return
            
        state = self.history_manager.get_current_state()
        if state is not None:
            state["harmony_mode"] = text  # type: ignore
            # Trigger outer progressive render, skipping the center!
            self.start_asynchronous_render_for_state(state, skip_center=True)

    def on_axis_toggled(self, checked: bool) -> None:
        """
        Triggered when a mutation axis radio button is toggled.
        Only re-renders outer tiles (skipping center) if checked is True.
        Does NOT push a new state to history.
        """
        if not checked or self.master_image is None:
            return
            
        state = self.history_manager.get_current_state()
        if state is not None:
            self.start_asynchronous_render_for_state(state, skip_center=True)

    def on_intensity_changed(self, value: int) -> None:
        """
        Triggered when the user slides the mutation intensity control.
        Updates the intensity label, updates active state's step_size,
        and triggers a progressive re-render of the 8 surrounding outer tiles.
        """
        step_size = value / 100.0
        self.intensity_label.setText(f" Intensity: {step_size:.2f}x ")
        
        if self.master_image is None:
            return
            
        state = self.history_manager.get_current_state()
        if state is not None:
            state["step_size"] = step_size
            # Re-render outer tiles only, keeping the center invariant
            self.start_asynchronous_render_for_state(state, skip_center=True)

    def get_active_mutation_axis(self) -> MutationAxis:
        """
        Returns the active mutation axis. Defaults to 'All' now that controls are removed.
        """
        return "All"

    def render_grid_from_current_state(self) -> None:
        """
        Renders all 9 containers asynchronously from current active state in history.
        """
        state = self.history_manager.get_current_state()
        if state is not None:
            self.start_asynchronous_render_for_state(state, skip_center=False)
            self.sync_sliders_with_state(state)

    def start_asynchronous_render_for_state(self, state: StateNode, skip_center: bool = False) -> None:
        """
        Generates 9 mutations based on the input state, and fires up a background
        GradingWorker thread to compute and render the tiles progressively.
        """
        if self.proxy_image is None:
            return

        # 1. Sync intensity slider only if skip_center is False (load/undo/redo)
        if not skip_center:
            self.intensity_slider.blockSignals(True)
            self.intensity_slider.setValue(int(state.get("step_size", 0.2) * 100.0))
            self.intensity_slider.blockSignals(False)
            self.intensity_label.setText(f" Intensity: {state.get('step_size', 0.2):.2f}x ")

        # 2. Update navigation controls and history index display immediately
        self.back_action.setEnabled(self.history_manager.can_undo())
        self.forward_action.setEnabled(self.history_manager.can_redo())
        self.save_action.setEnabled(True)
        self.history_label.setText(self.history_manager.get_history_display_text())

        # 3. Generate the 9 state nodes for the grid using current state's active mode
        mutations = generate_mutations(state, state.get("harmony_mode", "Monochromatic"), axis="All")
        
        # Lock index 4 to be EXACTLY the original unchanged history state!
        mutations[4] = state.copy()
        self.grid_states = mutations

        # 4. Cancel any running background worker thread safely
        if self.grading_worker is not None:
            if self.grading_worker.isRunning():
                self.grading_worker.cancel()
                self.grading_worker.wait()
            self.grading_worker = None

        # 5. Create and launch the new worker thread using proxy_image for snappy UI calculation
        self.grading_worker = GradingWorker(
            self.proxy_image,
            self.grid_states,
            skip_center=skip_center,
            parent=self
        )
        self.grading_worker.progress.connect(self.on_worker_progress)
        self.grading_worker.finished_all.connect(self.on_worker_finished)
        self.grading_worker.start()

    def on_worker_progress(self, index: int, arr: np.ndarray) -> None:
        """
        Updates a specific container tile in the grid as soon as it is computed.
        """
        if self.proxy_image is not None:
            self.containers[index].set_image(arr)

    def on_worker_finished(self) -> None:
        """
        Updates the status bar when all outer mutations are finished processing.
        """
        self.statusBar().showMessage("Ready.")

    def on_container_clicked(self, index: int) -> None:
        """
        Handler for when an image container in the 3x3 grid is clicked.
        Promotes selected outer variation to center immediately, or re-rolls mutation proposals
        if the center tile itself is clicked.
        """
        if self.proxy_image is None or self.grid_states is None:
            return

        if index == 4:
            # Clicking the center image tile triggers a re-roll of the 8 outer tiles
            self.statusBar().showMessage("Re-rolling mutation proposals around center...")
            state = self.history_manager.get_current_state()
            if state is not None:
                # Clear all surrounding cells immediately to give instant visual feedback that a re-roll has begun
                for i in range(9):
                    if i != 4:
                        self.containers[i].set_image(None)
                # Fire up background progressive rendering of the 8 outer tiles, skipping center
                self.start_asynchronous_render_for_state(state, skip_center=True)
            return

        # 1. Promote clicked state parameters to center state
        chosen_state = self.grid_states[index]

        # 2. Update center tile immediately with pre-computed high-res pixmap of the clicked tile!
        clicked_container = self.containers[index]
        if clicked_container.master_pixmap is not None:
            self.containers[4].master_pixmap = clicked_container.master_pixmap
            self.containers[4].update_display_pixmap()
            self.containers[4].set_active(True)
        else:
            # Fallback if somehow pixmap is missing (e.g. still rendering)
            self.containers[4].set_image(apply_grading(self.proxy_image, chosen_state))

        # Clear other cells immediately to show we are generating new mutations around the new center
        for i in range(9):
            if i != 4:
                self.containers[i].set_image(None)

        # 3. Push the new state to history
        self.history_manager.push_state(chosen_state)
        self.sync_sliders_with_state(chosen_state)

        # 4. Fire up background asynchronous worker to render other 8 tiles progressively, skipping the center!
        self.start_asynchronous_render_for_state(chosen_state, skip_center=True)
        
        self.statusBar().showMessage(f"Promoted mutation {index} to center immediately. Computing new mutations...")

    def closeEvent(self, event) -> None:
        """
        Safely stops active background thread when application is closed.
        """
        if self.grading_worker is not None:
            if self.grading_worker.isRunning():
                self.grading_worker.cancel()
                self.grading_worker.wait()
        super().closeEvent(event)
