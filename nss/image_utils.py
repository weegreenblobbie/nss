import cv2
import numpy as np

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
