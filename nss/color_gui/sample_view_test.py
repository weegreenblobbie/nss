"""Tests of nss.color_gui.sample_view: click mapping to image fractions, sample circle hits and the sampling cursor."""
import numpy as np
from PyQt6.QtCore import QPoint, QPointF, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from nss.color_gui.sample_view import SampleView


def _view() -> SampleView:
    """A 400 x 300 SampleView showing a (60, 160, 3) image, so the image is letterboxed top and bottom."""
    view = SampleView()
    view.resize(400, 300)
    view.set_image(np.linspace(0.0, 1.0, 60 * 160 * 3, dtype=np.float32).reshape(60, 160, 3))
    view.show()
    QApplication.processEvents()
    return view


def test_click_maps_to_image_fractions_only_when_sampling() -> None:
    """A click on the image emits (u, v) only in sampling mode; outside the image nothing is emitted."""
    app = QApplication.instance() or QApplication([])
    view = _view()
    picked = []
    view.sample_picked.connect(lambda u, v: picked.append((u, v)))
    rect = view.image_rect()
    assert rect.width() > 2.0 * rect.height()                         # aspect kept: 160 x 60
    target = QPoint(int(rect.left() + 0.25 * rect.width()), int(rect.top() + 0.75 * rect.height()))
    QTest.mouseClick(view, Qt.MouseButton.LeftButton, pos=target)
    assert picked == []
    view.set_sampling(True)
    QTest.mouseClick(view, Qt.MouseButton.LeftButton, pos=target)
    QTest.mouseClick(view, Qt.MouseButton.LeftButton, pos=QPoint(200, int(rect.top()) - 5))     # letterbox
    assert len(picked) == 1
    u, v = picked[0]
    assert abs(u - 0.25) < 0.02 and abs(v - 0.75) < 0.05
    view.close()


def test_sample_circle_takes_the_click() -> None:
    """A click on a circle emits sample_clicked with its slot, not sample_picked; circles follow the image rectangle."""
    app = QApplication.instance() or QApplication([])
    view = _view()
    view.set_sampling(True)
    view.set_markers([(4, 0.5, 0.5), (7, 0.1, 0.2)])
    clicked, picked = [], []
    view.sample_clicked.connect(lambda slot, pos: clicked.append(slot))
    view.sample_picked.connect(lambda u, v: picked.append(u))
    rect = view.image_rect()
    c = view.marker_center(rect, 0.1, 0.2)
    QTest.mouseClick(view, Qt.MouseButton.LeftButton, pos=c.toPoint())
    assert clicked == [7] and picked == []
    assert view.marker_at(QPointF(rect.center())) == 4
    assert view.marker_at(QPointF(rect.left() + 0.8 * rect.width(), rect.center().y())) is None
    view.close()


def test_sampling_cursor_toggles() -> None:
    """Sampling sets a bitmap cursor; leaving it restores the default."""
    app = QApplication.instance() or QApplication([])
    view = _view()
    view.set_sampling(True)
    assert view.cursor().shape() == Qt.CursorShape.BitmapCursor
    view.set_sampling(False)
    assert view.cursor().shape() == Qt.CursorShape.ArrowCursor and not view.sampling
    view.close()
