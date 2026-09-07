import numpy as np
from nss.utils import TiffFile

def load_tiff_to_float32(filepath: str) -> tuple[np.ndarray, TiffFile]:
    """
    Loads a 16-bit TIFF file using the existing TiffFile class.
    Returns a tuple of (np.ndarray of float32 in range [0.0, 1.0], TiffFile object).
    """
    tiff = TiffFile()
    tiff.read(filepath)
    arr = tiff.array
    if arr is None:
        raise ValueError(f"Failed to read array from file {filepath}")
    
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
        
    # Standardize scale to 0.0 - 1.0 range
    arr_min = np.nanmin(arr)
    arr_max = np.nanmax(arr)
    if arr_max > arr_min:
        arr = (arr - arr_min) / (arr_max - arr_min)
    else:
        arr = np.zeros_like(arr)
    
    # Store standardized array back on the tiff object
    tiff.array = arr
    return arr, tiff

def to_uint8_display(arr: np.ndarray) -> np.ndarray:
    """
    Converts a float32 array in the 0.0 - 1.0 range to a uint8 array (0 - 255).
    Guarantees no out-of-bounds values by clipping to 0.0 - 1.0 before scaling.
    Does NOT overwrite the original array.
    """
    clipped = np.clip(arr, 0.0, 1.0)
    scaled = (clipped * 255.0) + 0.5
    return scaled.astype(np.uint8)
