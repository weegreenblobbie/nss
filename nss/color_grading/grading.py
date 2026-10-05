import cv2
import numpy as np

from nss.color_grading.shadows import Shadows
from nss.color_grading.midtones import Midtones
from nss.color_grading.highlights import Highlights


class Grading:
    def __init__(self):
        self.shadows = Shadows()
        self.midtones = Midtones()
        self.highlights = Highlights()
        self.luma_coeffs_rgb = np.array([0.2909, 0.3338, 0.3753], dtype=np.float32)
        self._init_cielab_lut()

    def _init_cielab_lut(self):
        lr_rgb_targets = {
            0: [0.4010, 0.1360, 0.1977], 10: [0.3867, 0.1707, 0.1759], 20: [0.3808, 0.2110, 0.1664],
            30: [0.3779, 0.2490, 0.1619], 40: [0.3749, 0.2828, 0.1571], 50: [0.3716, 0.3140, 0.1517],
            60: [0.3681, 0.3434, 0.1457], 70: [0.3376, 0.3485, 0.1468], 80: [0.3054, 0.3533, 0.1479],
            90: [0.2709, 0.3577, 0.1489], 100: [0.2323, 0.3618, 0.1498], 110: [0.1869, 0.3657, 0.1506],
            120: [0.1257, 0.3693, 0.1514], 130: [0.1122, 0.3693, 0.1949], 140: [0.0955, 0.3692, 0.2307],
            150: [0.0724, 0.3692, 0.2623], 160: [0.0132, 0.3691, 0.2914], 170: [0.0000, 0.3691, 0.3188],
            180: [0.0000, 0.3690, 0.3451], 190: [0.0000, 0.3422, 0.3473], 200: [0.0000, 0.3145, 0.3493],
            210: [0.0919, 0.2965, 0.3616], 220: [0.1192, 0.2716, 0.3683], 230: [0.1299, 0.2386, 0.3700],
            240: [0.1392, 0.2013, 0.3716], 250: [0.1972, 0.1933, 0.3713], 260: [0.2418, 0.1845, 0.3710],
            270: [0.2800, 0.1747, 0.3706], 280: [0.3147, 0.1635, 0.3703], 290: [0.3470, 0.1504, 0.3699],
            300: [0.3777, 0.1350, 0.3696], 310: [0.3823, 0.1351, 0.3445], 320: [0.3866, 0.1353, 0.3187],
            330: [0.3906, 0.1355, 0.2918], 340: [0.3943, 0.1357, 0.2633], 350: [0.3978, 0.1359, 0.2324],
            360: [0.4010, 0.1360, 0.1977]
        }
        
        base_rgb = np.array([[[0.2501, 0.2501, 0.2501]]], dtype=np.float32)
        base_lab = cv2.cvtColor(base_rgb, cv2.COLOR_RGB2Lab)[0, 0]
        
        actual_mask_weight = self.shadows.get_mask(np.array([[0.2501]], dtype=np.float32), 0.0, 50.0)[0, 0]
        true_multiplier = 1.0 / max(actual_mask_weight, 1e-6)
        
        self.cielab_lut = {}
        for h, rgb_tgt in lr_rgb_targets.items():
            lr_rgb = np.array([[rgb_tgt]], dtype=np.float32)
            lr_lab = cv2.cvtColor(lr_rgb, cv2.COLOR_RGB2Lab)[0, 0]
            self.cielab_lut[h] = (lr_lab - base_lab) * true_multiplier

    def _get_shift_vector(self, hue: float, sat: float, lum: float) -> np.ndarray:
        if sat == 0 and lum == 0:
            return np.zeros(3, dtype=np.float32)

        hue = float(hue) % 360.0
        keys = sorted(self.cielab_lut.keys())
        chromatic_shift = np.zeros(3, dtype=np.float32)
        for i in range(len(keys) - 1):
            if keys[i] <= hue <= keys[i+1]:
                k1, k2 = keys[i], keys[i+1]
                v1, v2 = self.cielab_lut[k1], self.cielab_lut[k2]
                chromatic_shift = v1 + (v2 - v1) * ((hue - k1) / (k2 - k1))
                break

        # UI slider scaling with linear magnitude preserve
        normalized_sat = sat / 100.0
        chromatic_shift = chromatic_shift * normalized_sat
        luma_shift = np.array([lum * 0.2, 0.0, 0.0], dtype=np.float32)
        
        return (chromatic_shift + luma_shift).astype(np.float32)

    def apply(self, img_rgb: np.ndarray, balance: float, blend: float,
              sh_h: float = 0.0, sh_s: float = 0.0, sh_l: float = 0.0,
              mi_h: float = 0.0, mi_s: float = 0.0, mi_l: float = 0.0,
              hi_h: float = 0.0, hi_s: float = 0.0, hi_l: float = 0.0) -> np.ndarray:
        img_luma = np.sum(img_rgb * self.luma_coeffs_rgb, axis=-1).astype(np.float32)
        m_sh = self.shadows.get_mask(img_luma, balance, blend)[..., np.newaxis]
        m_mi = self.midtones.get_mask(img_luma, balance, blend)[..., np.newaxis]
        m_hi = self.highlights.get_mask(img_luma, balance, blend)[..., np.newaxis]
        
        v_sh = self._get_shift_vector(sh_h, sh_s, sh_l)
        v_mi = self._get_shift_vector(mi_h, mi_s, mi_l)
        v_hi = self._get_shift_vector(hi_h, hi_s, hi_l)
        
        total_raw_shift = (m_sh * v_sh) + (m_mi * v_mi) + (m_hi * v_hi)
        if np.all(total_raw_shift == 0):
            return img_rgb
        
        # Adobe Gamut Compression Curve
        rgb_sat = (np.max(img_rgb, axis=-1) - np.min(img_rgb, axis=-1)).astype(np.float32)
        x_data = np.array([0.0, 0.25, 0.50, 0.75, 1.0], dtype=np.float32)
        y_data = np.array([1.0, 1.3356, 1.2230, 0.9717, 0.7500], dtype=np.float32)
        
        blend_multiplier = np.interp(rgb_sat, x_data, y_data)[..., np.newaxis]
        
        total_lab_shift = np.copy(total_raw_shift)
        total_lab_shift[..., 1:] *= blend_multiplier
        
        img_lab = cv2.cvtColor(img_rgb.astype(np.float32), cv2.COLOR_RGB2Lab)
        graded_lab = img_lab + total_lab_shift
        
        graded_rgb_raw = cv2.cvtColor(graded_lab.astype(np.float32), cv2.COLOR_Lab2RGB)
        
        # Directional Vector Scaling (Gamut Boundary Protection without veiling glare)
        delta = (graded_rgb_raw - img_rgb).astype(np.float32)
        delta_safe = np.where(delta == 0, 1e-9, delta)
        scale_down = -img_rgb / delta_safe
        scale_up = (1.0 - img_rgb) / delta_safe
        
        scale = np.ones_like(img_rgb)
        scale[delta < 0] = scale_down[delta < 0]
        scale[delta > 0] = scale_up[delta > 0]
        pixel_scale = np.clip(np.min(scale, axis=-1, keepdims=True), 0.0, 1.0)
        graded_rgb = np.clip(img_rgb + delta * pixel_scale, 0.0, 1.0).astype(np.float32)
        
        return graded_rgb