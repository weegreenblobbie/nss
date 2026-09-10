import pytest
from unittest.mock import patch
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt, QMimeData, QPointF
from PyQt6.QtGui import QMouseEvent, QColor
from nss.color_gui.swatches import ClickableSwatchLabel, ColorSwatch, ColorModifyDialog

def test_color_swatch() -> None:
    app = QApplication.instance() or QApplication([])
    
    swatch = ColorSwatch(120.0, 0.4)
    assert swatch.hue == 120.0
    assert swatch.sat == 0.4

    swatch.update_color(60.0, 0.8)
    assert swatch.hue == 60.0
    assert swatch.sat == 0.8


def test_color_swatch_drag_and_drop() -> None:
    app = QApplication.instance() or QApplication([])

    # Create a draggable ClickableSwatchLabel
    active_swatch = ClickableSwatchLabel()
    active_swatch.hue = 300.0
    active_swatch.sat = 0.75

    # Create a drop-accepting ColorSwatch
    grid_swatch = ColorSwatch(120.0, 0.5)
    grid_swatch.index = 3

    # Connect to check overridden signals
    overwritten_data = []
    grid_swatch.overwritten.connect(lambda idx, h, s: overwritten_data.append((idx, h, s)))

    # Setup Mime Data
    mime_data = QMimeData()
    mime_data.setText("300.0,0.75")

    class MockEvent:
        def __init__(self, mime):
            self._mime = mime
            self.proposed_accepted = False

        def mimeData(self):
            return self._mime

        def acceptProposedAction(self):
            self.proposed_accepted = True

    mock_event = MockEvent(mime_data)
    grid_swatch.dropEvent(mock_event)

    # Verify that the color swatch is updated with the dragged color
    assert grid_swatch.hue == 300.0
    assert grid_swatch.sat == 0.75
    assert mock_event.proposed_accepted
    assert overwritten_data == [(3, 300.0, 0.75)]


def test_color_modify_dialog() -> None:
    app = QApplication.instance() or QApplication([])
    
    # Create dialog with red color (RGB: 255, 0, 0)
    initial_color = QColor(255, 0, 0)
    dialog = ColorModifyDialog(initial_color)
    
    # Assert initial fields
    assert dialog.r_edit.text() == "255"
    assert dialog.g_edit.text() == "0"
    assert dialog.b_edit.text() == "0"
    assert dialog.hex_edit.text() == "#FF0000"
    
    # Edit RGB to Green (0, 255, 0)
    dialog.r_edit.setText("0")
    dialog.g_edit.setText("255")
    dialog.b_edit.setText("0")
    
    assert dialog.current_color.red() == 0
    assert dialog.current_color.green() == 255
    assert dialog.current_color.blue() == 0
    assert dialog.hex_edit.text() == "#00FF00"
    
    # Edit Hex to Blue (#0000FF)
    dialog.hex_edit.setText("#0000FF")
    assert dialog.current_color.red() == 0
    assert dialog.current_color.green() == 0
    assert dialog.current_color.blue() == 255
    assert dialog.r_edit.text() == "0"
    assert dialog.g_edit.text() == "0"
    assert dialog.b_edit.text() == "255"


def test_color_swatch_forget_and_none_slot() -> None:
    app = QApplication.instance() or QApplication([])
    
    # Create an empty swatch
    swatch = ColorSwatch(None, None)
    assert swatch.hue is None
    assert swatch.sat is None
    assert "Empty Slot" in swatch.toolTip()
    
    # Ensure clicking on it does not raise errors or emit signals
    clicked_data = []
    swatch.clicked.connect(lambda h, s: clicked_data.append((h, s)))
    
    # Simulate a left click event
    event = QMouseEvent(
        QMouseEvent.Type.MouseButtonPress,
        QPointF(5.0, 5.0),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier
    )
    swatch.mousePressEvent(event)
    assert len(clicked_data) == 0  # No signal emitted
    
    # Create a filled swatch
    swatch_filled = ColorSwatch(240.0, 0.5)
    swatch_filled.index = 5
    
    # Register forgotten signal
    forgotten_data = []
    swatch_filled.forgotten.connect(lambda idx: forgotten_data.append(idx))
    
    # Simulate right-click forget action trigger
    swatch_filled._on_forget_triggered()
    assert forgotten_data == [5]
