import numpy as np


def rgb_to_lab_pure(rgb_img: np.ndarray) -> np.ndarray:
    """
    Converts a 16-bit RGB image array to float64 CIE L*a*b* color space.
    
    The conversion performs standard IEC 61966-2-1 sRGB linearization (gamma expansion),
    maps to CIE 1931 XYZ tristimulus values using standard sRGB transformation matrix,
    normalizes against the standard D65 illuminant white point (Xn=0.95047, Yn=1.00000, Zn=1.08883),
    and applies standard CIE L*a*b* cube-root non-linear mapping.

    Args:
        rgb_img (np.ndarray): Input RGB array scaled to 16-bit range [0, 65535].
            Can be 2D slice or 3D image of shape (..., 3).

    Returns:
        np.ndarray: Converted array in CIE L*a*b* space with dtype np.float64,
            with L* in [0, 100], a* in [-128, 127], and b* in [-128, 127].
    """
    rgb = rgb_img.astype(np.float64) / 65535.0
    mask = rgb > 0.04045
    rgb[mask] = ((rgb[mask] + 0.055) / 1.055) ** 2.4
    rgb[~mask] = rgb[~mask] / 12.92
    matrix = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ])
    xyz = np.dot(rgb, matrix.T)
    xyz /= np.array([0.95047, 1.00000, 1.08883])
    mask = xyz > 0.008856
    f_xyz = np.empty_like(xyz)
    f_xyz[mask] = np.cbrt(xyz[mask])
    f_xyz[~mask] = (7.787 * xyz[~mask]) + (16.0 / 116.0)
    L = (116.0 * f_xyz[..., 1]) - 16.0
    a = 500.0 * (f_xyz[..., 0] - f_xyz[..., 1])
    b = 200.0 * (f_xyz[..., 1] - f_xyz[..., 2])
    return np.stack([L, a, b], axis=-1)
