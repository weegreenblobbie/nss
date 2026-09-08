import numpy as np
from typing import List, Optional
from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QGridLayout,
    QComboBox,
    QLabel,
    QFrame,
    QToolBar,
    QStatusBar,
    QFileDialog,
    QRadioButton,
    QButtonGroup,
    QHBoxLayout,
    QVBoxLayout,
    QSlider,
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread
from PyQt6.QtGui import QMouseEvent, QAction, QImage, QPixmap

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
    The Main Window for the 16-bit Color Grading Explorer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("16-bit Color Grading Explorer")
        self.resize(1200, 800)

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

        # 1. Create Toolbar with buttons, combobox, radio button group, and step-size slider
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

        # Harmony Mode label and dropdown
        mode_label = QLabel(" Harmony Mode: ")
        toolbar.addWidget(mode_label)

        self.harmony_combo = QComboBox()
        self.harmony_combo.addItems(["Monochromatic", "Analogous", "Complementary"])
        self.harmony_combo.currentTextChanged.connect(self.on_harmony_mode_changed)
        toolbar.addWidget(self.harmony_combo)

        toolbar.addSeparator()

        # Mutation Axis group
        axis_label = QLabel(" Mutation Axis: ")
        toolbar.addWidget(axis_label)

        axis_widget = QWidget()
        axis_layout = QHBoxLayout(axis_widget)
        axis_layout.setContentsMargins(0, 0, 0, 0)
        axis_layout.setSpacing(5)

        self.axis_group = QButtonGroup(self)
        self.radio_all = QRadioButton("All")
        self.radio_hue = QRadioButton("Hue")
        self.radio_sat = QRadioButton("Saturation")
        self.radio_lum = QRadioButton("Luminance")

        self.radio_all.setChecked(True)

        self.axis_group.addButton(self.radio_all)
        self.axis_group.addButton(self.radio_hue)
        self.axis_group.addButton(self.radio_sat)
        self.axis_group.addButton(self.radio_lum)

        # Bind toggle triggers
        self.radio_all.toggled.connect(self.on_axis_toggled)
        self.radio_hue.toggled.connect(self.on_axis_toggled)
        self.radio_sat.toggled.connect(self.on_axis_toggled)
        self.radio_lum.toggled.connect(self.on_axis_toggled)

        axis_layout.addWidget(self.radio_all)
        axis_layout.addWidget(self.radio_hue)
        axis_layout.addWidget(self.radio_sat)
        axis_layout.addWidget(self.radio_lum)

        toolbar.addWidget(axis_widget)

        toolbar.addSeparator()

        # Mutation Intensity (Step Size) Slider
        self.intensity_label = QLabel(" Intensity: 1.00x ")
        self.intensity_label.setStyleSheet("color: #a0a0a0; font-weight: bold;")
        toolbar.addWidget(self.intensity_label)

        self.intensity_slider = QSlider(Qt.Orientation.Horizontal)
        self.intensity_slider.setRange(10, 200)
        self.intensity_slider.setValue(100)
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

        # Right Side: Vertical Read-only Inspector Panel
        self.inspector_panel = QFrame()
        self.inspector_panel.setFrameShape(QFrame.Shape.StyledPanel)
        self.inspector_panel.setFrameShadow(QFrame.Shadow.Raised)
        self.inspector_panel.setFixedWidth(280)
        self.inspector_panel.setStyleSheet(
            "background-color: #252525; border: 1px solid #444; border-radius: 4px;"
        )
        
        inspector_layout = QVBoxLayout(self.inspector_panel)
        inspector_layout.setContentsMargins(15, 15, 15, 15)
        inspector_layout.setSpacing(12)
        
        main_layout.addWidget(self.inspector_panel, stretch=1)

        # Setup Inspector Widgets
        header = QLabel("STATE INSPECTOR")
        header.setStyleSheet("font-weight: bold; font-size: 14px; color: #40ff40; border: none; margin-bottom: 5px;")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        inspector_layout.addWidget(header)

        def add_section(title: str) -> None:
            sec_label = QLabel(title)
            sec_label.setStyleSheet(
                "font-weight: bold; color: #a0a0a0; border: none; "
                "border-bottom: 1px solid #444; margin-top: 10px; padding-bottom: 2px;"
            )
            inspector_layout.addWidget(sec_label)

        def add_row(label_text: str) -> QLabel:
            row_widget = QWidget()
            row_widget.setStyleSheet("border: none; background: transparent;")
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            
            lbl = QLabel(label_text)
            lbl.setStyleSheet("color: #888; font-size: 11px;")
            val = QLabel("-")
            val.setStyleSheet("color: #eee; font-weight: bold; font-size: 11px;")
            val.setAlignment(Qt.AlignmentFlag.AlignRight)
            
            row_layout.addWidget(lbl)
            row_layout.addWidget(val)
            inspector_layout.addWidget(row_widget)
            return val

        add_section("GLOBAL CONFIG")
        self.mode_val = add_row("Harmony Mode:")
        self.step_size_val = add_row("Mutation Intensity:")
        self.glob_hue_val = add_row("Global Hue:")
        self.glob_sat_val = add_row("Global Sat:")
        self.glob_light_val = add_row("Global Light:")

        add_section("SHADOWS (L < 0.3)")
        self.sh_hue_val = add_row("Shadow Hue:")
        self.sh_sat_val = add_row("Shadow Sat (Tint):")
        self.sh_light_val = add_row("Shadow Light:")

        add_section("MIDTONES (0.3 - 0.7)")
        self.mid_hue_val = add_row("Midtone Hue:")
        self.mid_sat_val = add_row("Midtone Sat (Tint):")
        self.mid_light_val = add_row("Midtone Light:")

        add_section("HIGHLIGHTS (L > 0.7)")
        self.hi_hue_val = add_row("Highlight Hue:")
        self.hi_sat_val = add_row("Highlight Sat (Tint):")
        self.hi_light_val = add_row("Highlight Light:")

        inspector_layout.addStretch()

        # 3. Setup Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready. Load a 16-bit TIFF image to begin.")

    def update_inspector_panel(self, state: Optional[StateNode]) -> None:
        """
        Updates all text readouts in the Right-Hand Inspector panel to match the input StateNode.
        """
        if state is None:
            self.mode_val.setText("-")
            self.step_size_val.setText("-")
            self.glob_hue_val.setText("-")
            self.glob_sat_val.setText("-")
            self.glob_light_val.setText("-")
            self.sh_hue_val.setText("-")
            self.sh_sat_val.setText("-")
            self.sh_light_val.setText("-")
            self.mid_hue_val.setText("-")
            self.mid_sat_val.setText("-")
            self.mid_light_val.setText("-")
            self.hi_hue_val.setText("-")
            self.hi_sat_val.setText("-")
            self.hi_light_val.setText("-")
            return

        self.mode_val.setText(state.get("harmony_mode", "Monochromatic"))
        self.step_size_val.setText(f"{state.get('step_size', 1.0):.2f}x")
        self.glob_hue_val.setText(f"{state.get('hue_shift', 0.0):.1f}°")
        self.glob_sat_val.setText(f"{state.get('sat_shift', 0.0):+.2f}")
        self.glob_light_val.setText(f"{state.get('light_shift', 0.0):+.2f}")
        
        self.sh_hue_val.setText(f"{state.get('shadow_hue', 240.0):.1f}°")
        self.sh_sat_val.setText(f"{state.get('shadow_sat', 0.0):.2f}")
        self.sh_light_val.setText(f"{state.get('shadow_light', 0.0):+.2f}")

        self.mid_hue_val.setText(f"{state.get('midtone_hue', 120.0):.1f}°")
        self.mid_sat_val.setText(f"{state.get('midtone_sat', 0.0):.2f}")
        self.mid_light_val.setText(f"{state.get('midtone_light', 0.0):+.2f}")

        self.hi_hue_val.setText(f"{state.get('highlight_hue', 60.0):.1f}°")
        self.hi_sat_val.setText(f"{state.get('highlight_sat', 0.0):.2f}")
        self.hi_light_val.setText(f"{state.get('highlight_light', 0.0):+.2f}")

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
                initial_state = create_default_state(self.harmony_combo.currentText())  # type: ignore
                self.history_manager.push_state(initial_state)
                
                # Render grid
                self.render_grid_from_current_state()
                
                # Update status bar
                shape_str = "x".join(map(str, self.master_image.shape))
                proxy_str = "x".join(map(str, self.proxy_image.shape))
                self.status_bar.showMessage(f"Loaded {file_path} (Master: {shape_str}, Proxy: {proxy_str})")
            except Exception as e:
                self.status_bar.showMessage(f"Error loading image: {str(e)}")

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
                
                self.status_bar.showMessage(f"Successfully saved graded image to {save_path}")
            except Exception as e:
                self.status_bar.showMessage(f"Error saving image: {str(e)}")

    def on_back_clicked(self) -> None:
        """
        Navigates back in history.
        """
        if self.history_manager.can_undo():
            self.history_manager.undo()
            self.render_grid_from_current_state()
            self.status_bar.showMessage("Undo action.")

    def on_forward_clicked(self) -> None:
        """
        Navigates forward in history.
        """
        if self.history_manager.can_redo():
            self.history_manager.redo()
            self.render_grid_from_current_state()
            self.status_bar.showMessage("Redo action.")

    def on_harmony_mode_changed(self, text: str) -> None:
        """
        Handler for when the user selects a different harmony mode from the dropdown.
        Updates the active state's harmony mode and triggers re-generation of outer tiles,
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
        Updates the intensity label, updates the active state's step_size,
        and triggers a progressive re-render of the 8 surrounding outer tiles.
        """
        step_size = value / 100.0
        self.intensity_label.setText(f" Intensity: {step_size:.2f}x ")
        
        if self.master_image is None:
            return
            
        state = self.history_manager.get_current_state()
        if state is not None:
            state["step_size"] = step_size
            self.update_inspector_panel(state)
            # Re-render outer tiles only, keeping the center invariant
            self.start_asynchronous_render_for_state(state, skip_center=True)

    def get_active_mutation_axis(self) -> MutationAxis:
        """
        Returns the string key of the active mutation axis.
        """
        if self.radio_hue.isChecked():
            return "Hue"
        elif self.radio_sat.isChecked():
            return "Saturation"
        elif self.radio_lum.isChecked():
            return "Luminance"
        return "All"

    def render_grid_from_current_state(self) -> None:
        """
        Renders all 9 containers asynchronously from the current active state in history.
        """
        state = self.history_manager.get_current_state()
        if state is not None:
            self.start_asynchronous_render_for_state(state, skip_center=False)
            self.update_inspector_panel(state)

    def start_asynchronous_render_for_state(self, state: StateNode, skip_center: bool = False) -> None:
        """
        Generates 9 mutations based on the input state, and fires up a background
        GradingWorker thread to compute and render the tiles progressively.
        """
        if self.proxy_image is None:
            return

        # 1. Sync the UI controls with the state parameters if we are loading/undoing/redoing
        if not skip_center:
            self.harmony_combo.blockSignals(True)
            self.harmony_combo.setCurrentText(state["harmony_mode"])
            self.harmony_combo.blockSignals(False)
            
            self.intensity_slider.blockSignals(True)
            self.intensity_slider.setValue(int(state.get("step_size", 1.0) * 100.0))
            self.intensity_slider.blockSignals(False)
            self.intensity_label.setText(f" Intensity: {state.get('step_size', 1.0):.2f}x ")

        # 2. Update navigation controls and history index display immediately
        self.back_action.setEnabled(self.history_manager.can_undo())
        self.forward_action.setEnabled(self.history_manager.can_redo())
        self.save_action.setEnabled(True)
        self.history_label.setText(self.history_manager.get_history_display_text())

        # 3. Generate the 9 state nodes for the grid using active UI mode and axis
        active_mode = self.harmony_combo.currentText()  # type: ignore
        axis = self.get_active_mutation_axis()
        mutations = generate_mutations(state, active_mode, axis=axis)
        
        # Lock index 4 to be EXACTLY the original unchanged history state!
        # This guarantees center image invariance on dropdown/axis changes!
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
        self.status_bar.showMessage("Ready.")

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
            self.status_bar.showMessage("Re-rolling mutation proposals around center...")
            state = self.history_manager.get_current_state()
            if state is not None:
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
        self.update_inspector_panel(chosen_state)

        # 4. Fire up background asynchronous worker to render other 8 tiles progressively, skipping the center!
        self.start_asynchronous_render_for_state(chosen_state, skip_center=True)
        
        self.status_bar.showMessage(f"Promoted mutation {index} to center immediately. Computing new mutations...")

    def closeEvent(self, event) -> None:
        """
        Safely stops active background thread when application is closed.
        """
        if self.grading_worker is not None:
            if self.grading_worker.isRunning():
                self.grading_worker.cancel()
                self.grading_worker.wait()
        super().closeEvent(event)
