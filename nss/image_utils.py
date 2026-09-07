import cv2
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

def downsample_image(img: np.ndarray, max_dim: int = 1200) -> np.ndarray:
    """
    Downsamples the input image so that its maximum dimension does not exceed max_dim.
    Uses cv2.INTER_AREA interpolation for high quality downsampling of float32 arrays.
    """
    h, w = img.shape[:2]
    current_max = max(h, w)
    if current_max <= max_dim:
        return img.copy()
        
    scale = max_dim / current_max
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Ensure minimum dimensions of 1x1
    new_w = max(1, new_w)
    new_h = max(1, new_h)
    
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

def to_uint8_display(arr: np.ndarray) -> np.ndarray:
    """
    Converts a float32 array in the 0.0 - 1.0 range to a uint8 array (0 - 255).
    Guarantees no out-of-bounds values by clipping to 0.0 - 1.0 before scaling.
    Does NOT overwrite the original array.
    """
    clipped = np.clip(arr, 0.0, 1.0)
    scaled = (clipped * 255.0) + 0.5
    return scaled.astype(np.uint8)
