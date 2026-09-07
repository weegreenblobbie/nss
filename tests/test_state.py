import pytest
from nss.color_math import create_default_state
from nss.history import HistoryManager

def test_history_manager_basic_flow() -> None:
    hm = HistoryManager()
    
    assert not hm.can_undo()
    assert not hm.can_redo()
    assert hm.get_current_state() is None
    
    # 1. Push state A
    state_a = create_default_state("Monochromatic")
    state_a["hue_shift"] = 10.0
    hm.push_state(state_a)
    
    assert not hm.can_undo()  # First state cannot undo
    assert not hm.can_redo()
    assert hm.get_current_state() == state_a
    
    # 2. Push state B
    state_b = create_default_state("Analogous")
    state_b["hue_shift"] = 20.0
    hm.push_state(state_b)
    
    assert hm.can_undo()
    assert not hm.can_redo()
    assert hm.get_current_state() == state_b
    
    # 3. Undo
    undone = hm.undo()
    assert undone == state_a
    assert not hm.can_undo()
    assert hm.can_redo()
    
    # 4. Redo
    redone = hm.redo()
    assert redone == state_b
    assert hm.can_undo()
    assert not hm.can_redo()
    
    # 5. Undo and then Push state C (clears B redo history)
    hm.undo()  # now at state_a
    state_c = create_default_state("Complementary")
    state_c["hue_shift"] = 30.0
    hm.push_state(state_c)
    
    assert hm.get_current_state() == state_c
    assert hm.can_undo()
    assert not hm.can_redo()  # state_b redo is cleared!
    
    hm.undo()
    assert hm.get_current_state() == state_a
