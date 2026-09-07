import sys
from unittest.mock import MagicMock
import numpy as np

# --- Mock tifffile ---
class MockTiffPage:
    def __init__(self) -> None:
        self.colormap = None
        self.photometric = None
        self.iccprofile = None

class MockTiffPages:
    def __init__(self) -> None:
        self.first = MockTiffPage()

class MockTiffFile:
    def __init__(self, filename: str, **kwargs) -> None:
        self.filename = filename
        self.pages = MockTiffPages()

    def __enter__(self) -> "MockTiffFile":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        pass

    def asarray(self) -> np.ndarray:
        # Return a standard 16-bit array for testing
        return np.array([[0, 32768, 65535]], dtype=np.uint16)

mock_tf = MagicMock()
mock_tf.TiffFile = MockTiffFile
mock_tf.imwrite = MagicMock()
sys.modules['tifffile'] = mock_tf
sys.modules['tifffile.tifffile'] = mock_tf

# --- Mock matplotlib ---
matplotlib_mock = MagicMock()
sys.modules['matplotlib'] = matplotlib_mock
sys.modules['matplotlib.pyplot'] = matplotlib_mock

# --- Mock PIL ---
pil_mock = MagicMock()
sys.modules['PIL'] = pil_mock
sys.modules['PIL.Image'] = pil_mock

# --- Mock skimage ---
skimage_mock = MagicMock()
sys.modules['skimage'] = skimage_mock
sys.modules['skimage.feature'] = skimage_mock

# --- Mock scipy ---
scipy_mock = MagicMock()
sys.modules['scipy'] = scipy_mock
sys.modules['scipy.ndimage'] = MagicMock()
sys.modules['scipy.special'] = MagicMock()

# Mock scipy.signal and convolve2d
scipy_signal_mock = MagicMock()
scipy_signal_mock.convolve2d = MagicMock()
sys.modules['scipy.signal'] = scipy_signal_mock

# --- Mock PyQt6 ---
pyqt6_mock = MagicMock()
sys.modules['PyQt6'] = pyqt6_mock
sys.modules['PyQt6.QtCore'] = pyqt6_mock

class MockQSettings:
    """
    In-memory mock for QSettings to support MRU persistence testing.
    """
    def __init__(self, org: str, app: str) -> None:
        self.org = org
        self.app = app
        # Initial dummy data to match test expectations, using platform-normalized absolute paths
        import os
        self._store = {
            "mru_directories": [
                os.path.abspath("dummy/existing/dir1"),
                os.path.abspath("dummy/existing/dir2")
            ]
        }

    def value(self, key: str, default=None):
        return self._store.get(key, default)

    def setValue(self, key: str, value) -> None:
        self._store[key] = value

pyqt6_mock.QSettings = MockQSettings
