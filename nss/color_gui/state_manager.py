from typing import Dict, List, Any, Optional

class StateManager:
    """
    Manages global undo and redo stacks using the Memento pattern.
    Stores full state snapshot dictionaries containing states of all registered stateful widgets.
    """
    def __init__(self) -> None:
        self.undo_stack: List[Dict[str, Any]] = []
        self.redo_stack: List[Dict[str, Any]] = []
        self._registered_widgets: Dict[str, Any] = {}

    def register_widget(self, widget: Any) -> None:
        """
        Registers a stateful widget. Enforces that each widget defines a unique class-level STATE_KEY.
        """
        state_key = getattr(widget, "STATE_KEY", None)
        if not state_key:
            raise ValueError(f"Widget {widget} must define a class-level STATE_KEY")
        if state_key in self._registered_widgets:
            raise ValueError(f"Duplicate STATE_KEY collision: {state_key}")
        self._registered_widgets[state_key] = widget

    def get_registered_widgets(self) -> Dict[str, Any]:
        return self._registered_widgets

    def capture_snapshot(self) -> Dict[str, Any]:
        """
        Gathers states from all registered stateful widgets and returns a full state snapshot.
        """
        snapshot = {}
        for state_key, widget in self._registered_widgets.items():
            widget_state = widget.get_state()
            if state_key in widget_state:
                snapshot.update(widget_state)
            else:
                snapshot[state_key] = widget_state
        return snapshot

    def push_state(self, snapshot: Dict[str, Any]) -> None:
        """
        Pushes a new state snapshot onto the undo stack. Clears the redo stack.
        """
        self.undo_stack.append(snapshot)
        self.redo_stack.clear()

    def undo(self) -> Optional[Dict[str, Any]]:
        """
        Pops the top state from undo_stack, pushes it to redo_stack, and returns the previous state snapshot.
        """
        if len(self.undo_stack) <= 1:
            return None
        
        # Pop current state and push to redo stack
        current = self.undo_stack.pop()
        self.redo_stack.append(current)
        
        # Return previous state (the top of undo_stack now represents the newly active state)
        return self.undo_stack[-1]

    def redo(self) -> Optional[Dict[str, Any]]:
        """
        Pops from redo_stack, pushes to undo_stack, and returns the newly active state snapshot.
        """
        if not self.redo_stack:
            return None
        
        state = self.redo_stack.pop()
        self.undo_stack.append(state)
        return state

    def can_undo(self) -> bool:
        return len(self.undo_stack) > 1

    def can_redo(self) -> bool:
        return len(self.redo_stack) > 0

    def restore_snapshot(self, snapshot: Dict[str, Any]) -> None:
        """
        Distributes the snapshot keys to registered widgets.
        """
        for state_key, widget in self._registered_widgets.items():
            widget.restore_state(snapshot)
