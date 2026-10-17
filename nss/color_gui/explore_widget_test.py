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
    assert widget.variation_strength == 0.05
    
    # Check get_state
    state = widget.get_state()
    assert state == {
        "explore_widget": {
            "variation_strength": 0.05
        }
    }
    
    # Check restore_state
    new_state = {
        "explore_widget": {
            "variation_strength": 0.5
        }
    }
    widget.restore_state(new_state)
    assert widget.variation_strength == 0.5


def test_explore_widget_variation_strength_slider() -> None:
    app = QApplication.instance() or QApplication([])
    
    widget = ExploreWidget()
    assert hasattr(widget, "var_slider")
    
    # 1. Default should be 5% (0.05 strength)
    assert widget.var_slider.value() == 5
    assert widget.variation_strength == 0.05
    
    # 2. Set to 80% (0.8 strength)
    widget.var_slider.setValue(80)
    assert widget.variation_strength == 0.8
    assert widget.var_label.text() == "Variation Strength: 80%"


def test_explore_widget_mixer_and_grading_checked_by_default() -> None:
    """The Mixer and Grading checkboxes start checked."""
    app = QApplication.instance() or QApplication([])
    widget = ExploreWidget()
    assert widget.mixer_checkbox.text() == "Mixer" and widget.mixer_checkbox.isChecked()
    assert widget.grading_checkbox.text() == "Grading" and widget.grading_checkbox.isChecked()
