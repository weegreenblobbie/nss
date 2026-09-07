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
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QMouseEvent, QAction

class ImageContainer(QLabel):
    """
    A custom clickable image container for the 3x3 color grading explorer.
    """
    clicked = pyqtSignal(int)  # Emits the index of the container (0-8)

    def __init__(self, index: int, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.index: int = index
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


class MainWindow(QMainWindow):
    """
    The Main Window for the 16-bit Color Grading Explorer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("16-bit Color Grading Explorer")
        self.resize(800, 800)

        # Setup UI Components
        self.init_ui()

    def init_ui(self) -> None:
        # 1. Create Toolbar with buttons and combobox
        toolbar = QToolBar("Navigation and Controls")
        self.addToolBar(toolbar)

        # "Back" button
        self.back_action = QAction("Back", self)
        self.back_action.setToolTip("Go to previous color state (Undo)")
        self.back_action.setEnabled(False)
        toolbar.addAction(self.back_action)

        # "Forward" button
        self.forward_action = QAction("Forward", self)
        self.forward_action.setToolTip("Go to next color state (Redo)")
        self.forward_action.setEnabled(False)
        toolbar.addAction(self.forward_action)

        toolbar.addSeparator()

        # Harmony Mode label and dropdown
        mode_label = QLabel(" Harmony Mode: ")
        toolbar.addWidget(mode_label)

        self.harmony_combo = QComboBox()
        self.harmony_combo.addItems(["Monochromatic", "Analogous", "Complementary"])
        toolbar.addWidget(self.harmony_combo)

        # 2. Setup Central Widget and 3x3 Grid Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        grid_layout = QGridLayout(central_widget)
        grid_layout.setSpacing(10)

        self.containers: List[ImageContainer] = []
        for i in range(9):
            container = ImageContainer(i, self)
            # Connect the click event for exploration in later phases
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

    def on_container_clicked(self, index: int) -> None:
        """
        Handler for when an image container in the 3x3 grid is clicked.
        """
        # For Phase 1, just display a status bar message showing which was clicked.
        self.status_bar.showMessage(f"Container {index} clicked (Harmony: {self.harmony_combo.currentText()})")
