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
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QMouseEvent, QAction, QImage, QPixmap

from nss.utils import TiffFile
from nss.color_math import (
    StateNode,
    create_default_state,
    apply_grading,
    generate_mutations,
)
from nss.history import HistoryManager

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


class MainWindow(QMainWindow):
    """
    The Main Window for the 16-bit Color Grading Explorer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("16-bit Color Grading Explorer")
        self.resize(800, 800)

        # State Variables
        self.master_image: Optional[np.ndarray] = None
        self.tiff_obj: Optional[TiffFile] = None
        self.history_manager: HistoryManager = HistoryManager()
        self.grid_states: Optional[List[StateNode]] = None

        # Setup UI Components
        self.init_ui()

    def init_ui(self) -> None:
        # Create Menu Bar
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("&File")

        open_action = QAction("&Open...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.setStatusTip("Open a 16-bit TIFF image")
        open_action.triggered.connect(self.open_file)
        file_menu.addAction(open_action)

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

        # 1. Create Toolbar with buttons and combobox
        toolbar = QToolBar("Navigation and Controls")
        self.addToolBar(toolbar)

        # "Back" button
        self.back_action = QAction("Back", self)
        self.back_action.setToolTip("Go to previous color state (Undo)")
        self.back_action.setEnabled(False)
        self.back_action.triggered.connect(self.on_back_clicked)
        toolbar.addAction(self.back_action)

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

        # 2. Setup Central Widget and 3x3 Grid Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        grid_layout = QGridLayout(central_widget)
        grid_layout.setSpacing(10)

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

        # 3. Setup Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready. Load a 16-bit TIFF image to begin.")

    def open_file(self) -> None:
        """
        Opens a file dialog to load a 16-bit TIFF file.
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open 16-bit TIFF",
            "",
            "TIFF Images (*.tif *.tiff);;All Files (*)"
        )
        if file_path:
            try:
                from nss.image_utils import load_tiff_to_float32
                
                # Load using the scaling pipeline
                self.master_image, self.tiff_obj = load_tiff_to_float32(file_path)
                
                # Initialize history with default state
                self.history_manager.clear()
                initial_state = create_default_state(self.harmony_combo.currentText())  # type: ignore
                self.history_manager.push_state(initial_state)
                
                # Render grid
                self.render_grid_from_current_state()
                
                # Update status bar
                shape_str = "x".join(map(str, self.master_image.shape))
                self.status_bar.showMessage(f"Successfully loaded {file_path} ({shape_str})")
            except Exception as e:
                self.status_bar.showMessage(f"Error loading image: {str(e)}")

    def save_file(self) -> None:
        """
        Saves the current color graded image as a 16-bit TIFF.
        """
        if self.master_image is None or self.tiff_obj is None or self.grid_states is None:
            return

        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save 16-bit TIFF",
            "",
            "TIFF Images (*.tif *.tiff);;All Files (*)"
        )
        if save_path:
            try:
                # Apply current center grading to the master image
                center_state = self.grid_states[4]
                graded_image = apply_grading(self.master_image, center_state)
                
                # Store the graded image onto our tiff object
                self.tiff_obj.array = graded_image
                
                # Save using the original metadata on tiff_obj
                self.tiff_obj.saveas(save_path)
                
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
        """
        if self.master_image is None:
            return
            
        current = self.history_manager.get_current_state()
        if current is not None:
            # Create a new state based on current but with the new mode
            new_state = create_default_state(text)  # type: ignore
            new_state["hue_shift"] = current["hue_shift"]
            new_state["sat_shift"] = current["sat_shift"]
            new_state["light_shift"] = current["light_shift"]
            new_state["highlight_hue"] = current["highlight_hue"]
            new_state["shadow_hue"] = current["shadow_hue"]
            
            self.history_manager.push_state(new_state)
            self.render_grid_from_current_state()

    def render_grid_from_current_state(self) -> None:
        """
        Renders all 9 containers from the master float32 image according to 
        the current state in the history stack.
        """
        if self.master_image is None:
            return

        state = self.history_manager.get_current_state()
        if state is None:
            return

        # 1. Update combobox without triggering event loops
        self.harmony_combo.blockSignals(True)
        self.harmony_combo.setCurrentText(state["harmony_mode"])
        self.harmony_combo.blockSignals(False)

        # 2. Generate 9 mutations
        self.grid_states = generate_mutations(state, state["harmony_mode"])

        # 3. Apply grading and set image to each container
        for i in range(9):
            graded = apply_grading(self.master_image, self.grid_states[i])
            self.containers[i].set_image(graded)

        # 4. Update UI element enabled states
        self.back_action.setEnabled(self.history_manager.can_undo())
        self.forward_action.setEnabled(self.history_manager.can_redo())
        self.save_action.setEnabled(True)

    def on_container_clicked(self, index: int) -> None:
        """
        Handler for when an image container in the 3x3 grid is clicked.
        Promotes the selected variation to the center.
        """
        if self.master_image is None or self.grid_states is None:
            return

        if index == 4:
            # Clicking the center does nothing
            return

        # Get the chosen StateNode from the clicked grid item
        chosen_state = self.grid_states[index]

        # Push to history
        self.history_manager.push_state(chosen_state)

        # Re-render
        self.render_grid_from_current_state()
        
        self.status_bar.showMessage(f"Promoted mutation {index} to center.")
