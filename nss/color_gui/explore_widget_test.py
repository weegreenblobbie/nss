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


def test_explore_widget_variation_strength_slider() -> None:
    from nss.color_math import create_default_state, generate_explore_mutations
    app = QApplication.instance() or QApplication([])
    
    widget = ExploreWidget()
    assert hasattr(widget, "var_slider")
    
    # 1. Default should be 100% (1.0 strength)
    assert widget.var_slider.value() == 100
    assert widget.variation_strength == 1.0
    
    # 2. Set to 80% (0.8 strength)
    widget.var_slider.setValue(80)
    assert widget.variation_strength == 0.8
    assert widget.var_label.text() == "Variation Strength: 80%"
    
    # 3. Verify math scaling: 0.8 strength means max Hue delta = 45 * 0.8 = 36.0,
    # and max Saturation delta = 0.25 * 0.8 = 0.20
    center = create_default_state("Monochromatic")
    # Set non-zero base saturation to allow full delta verification
    center["shadow_sat"] = 0.15
    center["midtone_sat"] = 0.15
    center["highlight_sat"] = 0.15
    
    mutations = generate_explore_mutations(center, "Monochromatic", widget.variation_strength)
    assert len(mutations) == 9
    for i, s in enumerate(mutations):
        if i != 4:
            # Hue delta check (<= 36.0)
            hue_diff = abs(s["shadow_hue"] - center["shadow_hue"])
            hue_dist = min(hue_diff, 360.0 - hue_diff)
            assert hue_dist <= 36.0 + 1e-5
            
            # Saturation delta check (<= 0.20)
            sat_diff = abs(s["shadow_sat"] - center["shadow_sat"])
            assert sat_diff <= 0.20 + 1e-5

