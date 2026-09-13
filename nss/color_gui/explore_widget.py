import numpy as np
from typing import List, Optional
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget,
    QGridLayout,
    QLabel,
    QFrame,
    QMenu,
)
from PyQt6.QtGui import (
    QMouseEvent,
    QImage,
    QPixmap,
    QAction,
)
from nss.color_math import StateNode

class ImageContainer(QLabel):
    """
    A custom clickable/right-clickable image container for the 3x3 color grading explorer.
    """
    clicked = pyqtSignal(int)  # Emits the index of the container (0-8)
    edit_requested = pyqtSignal(int)  # Emits the index on right-click -> "Edit"

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
            "border: 1px solid #cccccc; border-radius: 4px;"
        )
        self.setText(f"Image {index}")

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.index)

    def contextMenuEvent(self, event) -> None:
        # Context menu is only available for outer slots (not center index 4)
        if self.index == 4:
            return
            
        menu = QMenu(self)
        edit_action = QAction("Edit", self)
        edit_action.triggered.connect(lambda: self.edit_requested.emit(self.index))
        menu.addAction(edit_action)
        menu.exec(event.globalPos())

    def set_active(self, is_active: bool) -> None:
        """
        Sets active (center) or inactive style for the container.
        """
        if is_active:
            self.setStyleSheet(
                "border: 3px solid #008000; border-radius: 4px;"
            )
            self.setFrameStyle(QFrame.Shape.Panel | QFrame.Shadow.Raised)
        else:
            self.setStyleSheet(
                "border: 1px solid #cccccc; border-radius: 4px;"
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


class ExploreWidget(QWidget):
    """
    A stateful widget containing a 3x3 visual look-matrix.
    Enforces Memento state management with a unique STATE_KEY.
    """
    STATE_KEY = "explore_widget"
    container_clicked = pyqtSignal(int)
    container_edit_requested = pyqtSignal(int)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.harmony_mode: str = "Monochromatic"
        self.variation_strength: float = 1.0  # Starts large, shrinks on promote
        
        # Grid Layout
        self.grid_layout = QGridLayout(self)
        self.grid_layout.setContentsMargins(0, 0, 0, 0)
        self.grid_layout.setSpacing(10)

        # 3x3 Containers
        self.containers: List[ImageContainer] = []
        for i in range(9):
            container = ImageContainer(i, self)
            container.clicked.connect(self.container_clicked.emit)
            container.edit_requested.connect(self.container_edit_requested.emit)
            self.containers.append(container)

            # Map index to row/col
            row = i // 3
            col = i % 3
            self.grid_layout.addWidget(container, row, col)

            if i == 4:
                container.set_active(True)
                container.setText("Center (Active)")
            else:
                container.set_active(False)
                container.setText(f"Mutation {i}")

    def get_state(self) -> dict:
        """
        Memento pattern: captures self state.
        """
        return {
            self.STATE_KEY: {
                "harmony_mode": self.harmony_mode,
                "variation_strength": self.variation_strength,
            }
        }

    def restore_state(self, full_state_dict: dict) -> None:
        """
        Memento pattern: restores self state.
        """
        if self.STATE_KEY in full_state_dict:
            state = full_state_dict[self.STATE_KEY]
            self.harmony_mode = state.get("harmony_mode", "Monochromatic")
            self.variation_strength = state.get("variation_strength", 1.0)
