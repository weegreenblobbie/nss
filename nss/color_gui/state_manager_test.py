import pytest
from PyQt6.QtWidgets import QApplication, QWidget
from nss.color_gui.state_manager import StateManager

class MockStatefulWidget(QWidget):
    STATE_KEY = "mock_widget"
    def __init__(self, parent=None):
        super().__init__(parent)
        self.val = 42

    def get_state(self) -> dict:
        return {self.STATE_KEY: {"val": self.val}}

    def restore_state(self, state: dict) -> None:
        if self.STATE_KEY in state:
            self.val = state[self.STATE_KEY]["val"]


def test_state_manager_registration() -> None:
    app = QApplication.instance() or QApplication([])
    manager = StateManager()
    
    widget = MockStatefulWidget()
    manager.register_widget(widget)
    
    assert "mock_widget" in manager.get_registered_widgets()
    
    # Test collision
    with pytest.raises(ValueError, match="Duplicate STATE_KEY collision"):
        manager.register_widget(widget)


def test_state_manager_undo_redo_flow() -> None:
    app = QApplication.instance() or QApplication([])
    manager = StateManager()
    
    widget = MockStatefulWidget()
    manager.register_widget(widget)
    
    # Capture first state
    snapshot1 = manager.capture_snapshot()
    assert snapshot1 == {"mock_widget": {"val": 42}}
    manager.push_state(snapshot1)
    
    # Mutate & push second state
    widget.val = 100
    snapshot2 = manager.capture_snapshot()
    manager.push_state(snapshot2)
    
    assert manager.can_undo()
    assert not manager.can_redo()
    
    # Undo
    prev_snapshot = manager.undo()
    assert prev_snapshot == {"mock_widget": {"val": 42}}
    manager.restore_snapshot(prev_snapshot)
    assert widget.val == 42
    
    assert not manager.can_undo()
    assert manager.can_redo()
    
    # Redo
    next_snapshot = manager.redo()
    assert next_snapshot == {"mock_widget": {"val": 100}}
    manager.restore_snapshot(next_snapshot)
    assert widget.val == 100
