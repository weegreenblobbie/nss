from typing import List, Optional
from nss.color_math import StateNode

class HistoryManager:
    """
    Manages the Undo/Redo parameter history stack of StateNodes.
    """
    def __init__(self) -> None:
        self.history: List[StateNode] = []
        self.current_index: int = -1

    def clear(self) -> None:
        """
        Resets the history.
        """
        self.history.clear()
        self.current_index = -1

    def push_state(self, state: StateNode) -> None:
        """
        Pushes a new state. Clears any redo history forward of the current index.
        """
        # Clear anything after current index
        if self.current_index < len(self.history) - 1:
            self.history = self.history[:self.current_index + 1]
        
        self.history.append(state.copy())
        self.current_index = len(self.history) - 1

    def undo(self) -> Optional[StateNode]:
        """
        Navigates back in history. Returns the newly active state or None.
        """
        if self.can_undo():
            self.current_index -= 1
            return self.get_current_state()
        return None

    def redo(self) -> Optional[StateNode]:
        """
        Navigates forward in history. Returns the newly active state or None.
        """
        if self.can_redo():
            self.current_index += 1
            return self.get_current_state()
        return None

    def can_undo(self) -> bool:
        """
        Returns True if undo is available.
        """
        return self.current_index > 0

    def can_redo(self) -> bool:
        """
        Returns True if redo is available.
        """
        return self.current_index < len(self.history) - 1

    def get_current_state(self) -> Optional[StateNode]:
        """
        Returns the currently active StateNode, or None if history is empty.
        """
        if 0 <= self.current_index < len(self.history):
            return self.history[self.current_index]
        return None

    def get_history_display_text(self) -> str:
        """
        Returns a formatted 1-based history index and total stack size.
        """
        if len(self.history) == 0:
            return "Step: 0 of 0"
        return f"Step: {self.current_index + 1} of {len(self.history)}"
