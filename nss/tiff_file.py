"""TIFF reading and writing that keeps the stored pixel values and the embedded ICC profile."""
import warnings
import numpy as np
from tifffile import imwrite as tf_imwrite
from tifffile import TiffFile as tf_TiffFile

from .icc import icc_description


class TiffFile:
    """
    TIFF reader/writer that does NOT change pixel values and carries the ICC profile.

    - read():   integer data is scaled by the dtype's full range (uint16 -> /65535). No min/max stretch.
    - saveas(): values are clipped to [0, 1], scaled back by the same constant and rounded, so
                read() -> saveas() reproduces the file exactly. The embedded ICC profile is written back.
    """

    def __init__(self) -> None:
        """An empty file object; fill it with read() or from_array()."""
        self._filename = None
        self._array = None
        self._colormap = None
        self._photometric = None
        self._iccprofile = None
        self._dtype = np.dtype(np.uint16)

    @property
    def array(self) -> np.ndarray | None:
        """The pixel data, float 0..1 after read()."""
        return self._array

    @array.setter
    def array(self, new_array: np.ndarray) -> None:
        """Replaces the pixel data written by saveas()."""
        self._array = new_array

    @property
    def iccprofile(self) -> bytes | None:
        """The embedded ICC profile bytes, or None."""
        return self._iccprofile

    @iccprofile.setter
    def iccprofile(self, icc: bytes | None) -> None:
        """Sets the ICC profile written by saveas()."""
        self._iccprofile = icc

    @property
    def icc_name(self) -> str | None:
        """The ICC profile's description, or None without a profile."""
        return None if self._iccprofile is None else icc_description(self._iccprofile)

    @classmethod
    def from_array(cls, array: np.ndarray, icc: bytes | None = None, dtype: type = np.uint16) -> "TiffFile":
        """A file object holding `array` (0..1), to be written as `dtype` with the profile `icc`."""
        obj = cls()
        obj._array = np.asarray(array)
        obj._iccprofile = icc
        obj._dtype = np.dtype(dtype)
        return obj

    def read(self, filename: str, dtype: type = np.float32) -> "TiffFile":
        """Reads `filename`: integer data divided by its full range (no min/max stretch), the ICC profile kept.  Returns self."""
        self._filename = filename
        with tf_TiffFile(filename) as fin:
            page = fin.pages.first
            raw = fin.asarray()
            self._colormap = page.colormap
            self._photometric = page.photometric
            self._iccprofile = page.iccprofile
        self._dtype = raw.dtype
        arr = raw.astype(dtype)
        if raw.dtype.kind in "ui":
            arr /= dtype(np.iinfo(raw.dtype).max)
        self._array = arr
        return self

    def saveas(self, filename: str) -> None:
        """Writes the array with the ICC profile in the original format: integer formats are clipped to 0..1 and scaled back, float data as is."""
        a = np.asarray(self._array, dtype=np.float64)
        if np.isnan(a).any():
            raise ValueError("array contains NaN; refusing to write")
        if self._dtype.kind in "ui":
            out = np.round(np.clip(a, 0.0, 1.0) * np.iinfo(self._dtype).max).astype(self._dtype)
        else:
            out = a.astype(self._dtype)
        photometric = self._photometric
        if photometric is None:
            photometric = "rgb" if out.ndim == 3 and out.shape[-1] == 3 else "minisblack"
        tf_imwrite(
            filename,
            out,
            photometric=photometric,
            colormap=self._colormap,
            iccprofile=self._iccprofile,
            software="https://github.com/weegreenblobbie/nss",
        )