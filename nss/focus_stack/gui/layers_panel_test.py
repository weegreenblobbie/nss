"""Tests for nss.focus_stack.gui.layers_panel."""
import numpy as np
import pytest
from PyQt6.QtCore import QEvent, QSize, Qt
from PyQt6.QtGui import QIcon
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from nss.focus_stack.gui.layers_panel import LayersPanel

APP = QApplication.instance() or QApplication([])  # kept alive for every test in this module
PANELS: list[LayersPanel] = []


@pytest.fixture(autouse=True)
def delete_panels():
    """Deletes every panel a test made, so Python's garbage collector never frees one in the middle of a paint."""
    yield
    for panel in PANELS:
        panel.hide()
        panel.deleteLater()
    PANELS.clear()
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def make_panel() -> LayersPanel:
    """A shown panel for a target and two frames, with differently coloured non-square thumbnails."""
    thumbnails = []
    for color in ((200, 40, 40), (40, 200, 40), (40, 40, 200)):
        thumb = np.zeros((12, 20, 3), dtype=np.uint8)
        thumb[:] = color
        thumbnails.append(thumb)
    panel = LayersPanel()
    PANELS.append(panel)
    panel.set_layers(["one.tif", "two.tif"], thumbnails, 2)
    panel.resize(240, 400)
    panel.show()
    QApplication.processEvents()
    return panel


def row_centre(panel: LayersPanel, row: int):
    """The centre of a row in the list's viewport coordinates."""
    return panel.list.visualItemRect(panel.list.item(row)).center()


def test_the_target_is_always_the_first_row_followed_by_the_frames() -> None:
    """Rows are the target and then each frame, with its name and a thumbnail."""
    panel = make_panel()
    assert panel.list.count() == 3
    assert [panel.list.item(i).text() for i in range(3)] == ["Target", "one.tif", "two.tif"]
    assert all(not panel.list.item(i).icon().isNull() for i in range(3))
    assert panel.list.item(1).icon().pixmap(20, 12).toImage().pixelColor(5, 5).green() == 200


def test_twenty_frames_are_listed_by_name_after_the_target() -> None:
    """Twenty-one rows: the target, then every frame in order, with nothing but its name."""
    panel = LayersPanel()
    PANELS.append(panel)
    names = [f"f{i}.tif" for i in range(1, 21)]
    panel.set_layers(names, [np.zeros((4, 6, 3), dtype=np.uint8)] * 21, 20)
    assert [panel.list.item(i).text() for i in range(21)] == ["Target"] + names


def test_show_active_highlights_a_row_without_announcing_a_choice() -> None:
    """Showing the active patch's layer selects its row, or clears the selection, and never emits layer_chosen."""
    panel = make_panel()
    chosen = []
    panel.layer_chosen.connect(chosen.append)
    panel.show_active(2)
    assert panel.list.currentRow() == 2
    panel.show_active(0)
    assert panel.list.currentRow() == 0
    panel.show_active(None)
    assert panel.list.currentRow() == -1 and not panel.list.selectedItems()
    assert chosen == []


def test_clicking_a_row_announces_that_layer_once() -> None:
    """A real click on the second frame's row emits layer_chosen(2), and the list does not take keyboard focus."""
    panel = make_panel()
    chosen = []
    panel.layer_chosen.connect(chosen.append)
    QTest.mouseClick(panel.list.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, row_centre(panel, 2))
    assert chosen == [2]
    assert panel.list.focusPolicy() == Qt.FocusPolicy.NoFocus


def test_a_selected_rows_thumbnail_keeps_its_own_colours() -> None:
    """Qt tints the icon of a selected item; the thumbnails must look the same selected or not."""
    panel = make_panel()
    icon = panel.list.item(1).icon()
    normal = icon.pixmap(QSize(20, 12), QIcon.Mode.Normal).toImage().pixelColor(5, 5)
    selected = icon.pixmap(QSize(20, 12), QIcon.Mode.Selected).toImage().pixelColor(5, 5)
    assert normal.green() == 200 and selected == normal


def test_the_hint_says_how_to_finish_a_patch_and_how_to_edit_an_old_one() -> None:
    """The hint names Esc for finishing a patch and the pins for going back to one."""
    text = make_panel().hint.text()
    assert "Esc" in text and "pin" in text and "Tab" in text and "Ctrl" in text and "target" in text


def test_ctrl_click_views_a_layer_and_a_plain_click_chooses_it() -> None:
    """Ctrl+click announces layer_viewed and not layer_chosen; a plain click is the other way round."""
    panel = make_panel()
    chosen, viewed = [], []
    panel.layer_chosen.connect(chosen.append)
    panel.layer_viewed.connect(viewed.append)
    QTest.mouseClick(panel.list.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ControlModifier, row_centre(panel, 2))
    assert viewed == [2] and chosen == []
    QTest.mouseClick(panel.list.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, row_centre(panel, 1))
    assert viewed == [2] and chosen == [1]


def test_the_right_click_menu_offers_set_as_target_for_frames_only() -> None:
    """Frames get a 'Set ... as the target' action that announces target_requested; the target row (and rows past the frames) get none."""
    panel = make_panel()
    requested = []
    panel.target_requested.connect(requested.append)
    assert panel.target_menu(0) is None and panel.target_menu(3) is None
    menu = panel.target_menu(2)
    assert [action.text() for action in menu.actions()] == ["Set two.tif as the target"]
    menu.actions()[0].trigger()
    assert requested == [2]


def test_the_viewed_layers_row_is_bold() -> None:
    """show_viewed puts one row in bold, and None clears it."""
    panel = make_panel()
    panel.show_viewed(1)
    assert [panel.list.item(i).font().bold() for i in range(3)] == [False, True, False]
    panel.show_viewed(None)
    assert not any(panel.list.item(i).font().bold() for i in range(3))
