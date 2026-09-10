import pytest
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt, QPointF
from PyQt6.QtGui import QMouseEvent
from nss.color_gui.explore_widget import ExploreWidget, ImageContainer

def test_image_container_signals() -> None:
    app = QApplication.instance() or QApplication([])
    
    container = ImageContainer(3)
    
    clicked_data = []
    container.clicked.connect(clicked_data.append)
    
    # Simulate left click
    event = QMouseEvent(
        QMouseEvent.Type.MouseButtonPress,
        QPointF(10.0, 10.0),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier
    )
    container.mousePressEvent(event)
    assert clicked_data == [3]


def test_explore_widget_state() -> None:
    app = QApplication.instance() or QApplication([])
    
    widget = ExploreWidget()
    assert widget.STATE_KEY == "explore_widget"
    assert widget.harmony_mode == "Monochromatic"
    assert widget.variation_strength == 1.0
    
    # Check get_state
    state = widget.get_state()
    assert state == {
        "explore_widget": {
            "harmony_mode": "Monochromatic",
            "variation_strength": 1.0
        }
    }
    
    # Check restore_state
    new_state = {
        "explore_widget": {
            "harmony_mode": "Triadic",
            "variation_strength": 0.5
        }
    }
    widget.restore_state(new_state)
    assert widget.harmony_mode == "Triadic"
    assert widget.variation_strength == 0.5
