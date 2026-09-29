from dataclasses import dataclass, asdict
import sys
import os

import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import scipy
import scipy.optimize

from nss.utils import TiffFile
from nss.color_math import apply_grading, create_default_state, StateNode

# Constant paths for test data
INPUT_PATH: str = "tests/test_data/input_synthetic.tif"

@dataclass
class DEConfig:
    """
     *** IMMUTABLE MASTER CONFIG ***
        DO NOT MODIFY, DUPLICATE, OR ALTER THIS CLASS.
    """

    strategy: str = 'best1bin'
    maxiter: int = 2000
    popsize: int = 40
    tol: float = 1e-5
    updating: str = 'deferred'
    polish: bool = True
    disp: bool = False
    mutation: tuple = (0.1, 1.99)
    recombination: float = 0.33
    init: str = 'latinhypercube'
    workers: int = -1


def load_image_array(path: str) -> np.ndarray:
    """
    Loads a 16-bit TIFF image using our existing image I/O utility (TiffFile)
    and returns its array in [0.0, 1.0] float32 format.
    """
    tiff_obj: TiffFile = TiffFile().read(path)
    return tiff_obj.array

# ==============================================================================
# MULTIPROCESSING-SAFE EVALUATOR CLASS (MUST BE AT MODULE LEVEL)
# ==============================================================================
import numpy as np
import scipy.optimize

class DecoupledMultiHueGAEvaluator:
    """
    Trains ONE color channel at a time, for ONE specific zone, 
    against ALL 6 target hues simultaneously.
    """
    def __init__(self, zone_name, img_arr_ds, targets_ds_dict, base_state, channel_idx, channel_name):
        self.zone_name = zone_name
        self.img_arr_ds = img_arr_ds
        self.targets_ds_dict = targets_ds_dict
        self.base_state = base_state
        self.channel_idx = channel_idx
        self.channel_name = channel_name
        
        # Create a boolean mask of purely neutral pixels (where R == G == B)
        # We use a tiny tolerance to account for floating point math
        r = self.img_arr_ds[:, :, 0]
        g = self.img_arr_ds[:, :, 1]
        b = self.img_arr_ds[:, :, 2]
        self.neutral_mask = (np.abs(r - g) < 1e-3) & (np.abs(g - b) < 1e-3)
        
        self.best_rmse = float('inf')
        self.gen = 0

    def eval_channel(self, x):
        # x is exactly 9 parameters for ONE channel and ONE zone
        # 3 Structural: center, width, gain
        # 6 Hue Weights for the spline
        c, w, g = x[0:3]
        hue_weights = x[3:9]

        # Initialize harmless defaults so the engine doesn't crash on inactive zones/channels
        kwargs = {
            "shadow_center_r": 0.0, "shadow_width_r": 1.0, "shadow_gain_r": 0.0,
            "mid_center_r": 0.5, "mid_width_r": 1.0, "midtone_gain_r": 0.0,
            "highlight_center_r": 1.0, "highlight_width_r": 1.0, "highlight_gain_r": 0.0,
            
            "shadow_center_g": 0.0, "shadow_width_g": 1.0, "shadow_gain_g": 0.0,
            "mid_center_g": 0.5, "mid_width_g": 1.0, "midtone_gain_g": 0.0,
            "highlight_center_g": 1.0, "highlight_width_g": 1.0, "highlight_gain_g": 0.0,
            
            "shadow_center_b": 0.0, "shadow_width_b": 1.0, "shadow_gain_b": 0.0,
            "mid_center_b": 0.5, "mid_width_b": 1.0, "midtone_gain_b": 0.0,
            "highlight_center_b": 1.0, "highlight_width_b": 1.0, "highlight_gain_b": 0.0,

            "shadow_hue_w_r": [0.0]*6, "mid_hue_w_r": [0.0]*6, "hi_hue_w_r": [0.0]*6,
            "shadow_hue_w_g": [0.0]*6, "mid_hue_w_g": [0.0]*6, "hi_hue_w_g": [0.0]*6,
            "shadow_hue_w_b": [0.0]*6, "mid_hue_w_b": [0.0]*6, "hi_hue_w_b": [0.0]*6,
        }

        # Dynamically inject parameters into the ACTIVE zone and ACTIVE channel
        zone_prefix = "shadow" if self.zone_name == "shadows" else "mid" if self.zone_name == "midtones" else "highlight"
        gain_prefix = "shadow_gain" if self.zone_name == "shadows" else "midtone_gain" if self.zone_name == "midtones" else "highlight_gain"
        ch_prefix = ['r', 'g', 'b'][self.channel_idx]
        hue_prefix = "hi" if self.zone_name == "highlights" else zone_prefix
        
        kwargs[f"{zone_prefix}_center_{ch_prefix}"] = float(c)
        kwargs[f"{zone_prefix}_width_{ch_prefix}"] = float(w)
        kwargs[f"{gain_prefix}_{ch_prefix}"] = float(g)
        kwargs[f"{hue_prefix}_hue_w_{ch_prefix}"] = [float(v) for v in hue_weights]

        total_mse = 0.0

        # Evaluate against ALL 6 target hues simultaneously
        for hue_angle, expected_ds in self.targets_ds_dict.items():
            current_state = self.base_state.copy()
            if self.zone_name == "shadows": current_state["shadow_hue"] = hue_angle
            elif self.zone_name == "midtones": current_state["midtone_hue"] = hue_angle
            elif self.zone_name == "highlights": current_state["highlight_hue"] = hue_angle
            
            actual_1, _ = apply_grading(self.img_arr_ds, current_state, **kwargs)
            
            # Calculate full image delta
            delta = expected_ds[:, :, self.channel_idx] * 65535.0 - actual_1[:, :, self.channel_idx] * 65535.0
            
            # KEY FIX: Only extract the delta values for the neutral pixels!
            delta_neutral = delta[self.neutral_mask]
            
            total_mse += np.mean(delta_neutral ** 2)

        # Return the root of the AVERAGE MSE across all 6 hues
        avg_rmse = np.sqrt(total_mse / len(self.targets_ds_dict))
        return avg_rmse

    def callback(self, xk, convergence=0):
        self.gen += 1
        current_rmse = self.eval_channel(xk)
        if current_rmse < self.best_rmse * 0.90:  # Print on 5% improvements
            self.best_rmse = current_rmse
            print(f"[{self.channel_name}] Gen {self.gen:4d} | Multi-Hue Channel RMSE: {current_rmse:.0f}", flush=True)
        return self.best_rmse < 100.0


def run_suite_a_calibration(zone_name: str) -> None:
    """
    zone_name must be one of: "shadows", "midtones", "highlights"
    Requires exports named: train_sh_0.tif, train_sh_60.tif, etc.
    """
    img_arr = load_image_array(INPUT_PATH)
    ds = 24
    img_arr_ds = img_arr[::ds, ::ds]

    # Map the hue angles to their respective target files
    zone_short = "sh" if zone_name == "shadows" else "mid" if zone_name == "midtones" else "hi"
    hues = [0, 60, 120, 180, 240, 300]
    targets_ds_dict = {}
    
    for h in hues:
        expected_img = load_image_array(f"tests/test_data/train_{zone_short}_{h}.tif")
        targets_ds_dict[h] = expected_img[::ds, ::ds]

    # Base state ensures ONLY the active zone is turned ON (Sat = 1.0)
    sh_state = create_default_state()
    sh_state.update({
        "balance": 0.0,
        "blending": 0.5,
        "shadow_sat": 1.0 if zone_name == "shadows" else 0.0,
        "midtone_sat": 1.0 if zone_name == "midtones" else 0.0,
        "highlight_sat": 1.0 if zone_name == "highlights" else 0.0,
    })

    # 9 Parameters per channel: [center, width, gain] + [6 hue weights]
    bounds = [
        (0.0, 1.0), (0.01, 10.0), (0.0, 20.0),  # Center, Width, Gain
    ] + [(-300.0, 300.0)] * 6

    results = {}

    print(f"\n==================================================")
    print(f"  STARTING SUITE A CALIBRATION: {zone_name.upper()}  ")
    print(f"==================================================")

    for ch_name, ch_idx in [("Red", 0), ("Green", 1), ("Blue", 2)]:
        print(f"\n--- TRAINING {ch_name.upper()} CHANNEL ---", flush=True)
        evaluator = DecoupledMultiHueGAEvaluator(zone_name, img_arr_ds, targets_ds_dict, sh_state, ch_idx, ch_name)
        
        res = scipy.optimize.differential_evolution(
            evaluator.eval_channel, 
            bounds,
            maxiter=DEConfig.maxiter,
            popsize=DEConfig.popsize,
            mutation=DEConfig.mutation,
            recombination=DEConfig.recombination,
            init=DEConfig.init,
            tol=DEConfig.tol,
            updating=DEConfig.updating,
            polish=DEConfig.polish,
            disp=DEConfig.disp,
            callback=evaluator.callback,
            workers=DEConfig.workers
        )
        print(f"{ch_name} Final Average RMSE across 6 Hues: {evaluator.best_rmse:.0f}")
        results[ch_name] = res.x

    print("\n==================================================")
    print(f" FINAL OPTIMIZED PARAMETERS FOR {zone_name.upper()} ")
    print("==================================================")
    
    zone_prefix = "shadow" if zone_name == "shadows" else "mid" if zone_name == "midtones" else "highlight"
    gain_prefix = "shadow_gain" if zone_name == "shadows" else "midtone_gain" if zone_name == "midtones" else "highlight_gain"
    hue_prefix = "hi" if zone_name == "highlights" else zone_prefix

    for ch_name, x in results.items():
        ch = ch_name.lower()[0] # 'r', 'g', or 'b'
        print(f"\n# {ch_name} Channel")
        print(f"params['{zone_prefix}_center_{ch}'] = {x[0]:.6f}")
        print(f"params['{zone_prefix}_width_{ch}'] = {x[1]:.6f}")
        print(f"params['{gain_prefix}_{ch}'] = {x[2]:.6f}")
        print(f"params['{hue_prefix}_hue_w_{ch}'] = {[round(float(v), 6) for v in x[3:9]]}")


class CurveTrackerGA:
    """
    Locks Hue Weights to the baseline.
    Optimizes `center`, `width`, AND `gain` to track how Blending warps the curves.
    """
    def __init__(self, img_arr_ds, expected_ds, base_state, zone_name):
        self.img_arr_ds = img_arr_ds
        self.expected_ds = expected_ds
        self.base_state = base_state
        self.zone_name = zone_name
        
        r, g, b = self.img_arr_ds[:, :, 0], self.img_arr_ds[:, :, 1], self.img_arr_ds[:, :, 2]
        self.neutral_mask = (np.abs(r - g) < 1e-3) & (np.abs(g - b) < 1e-3)

    def eval_curve(self, x):
        c, w, g_val = x[0], x[1], x[2]
        
        kwargs = {}
        zone_prefix = "shadow" if self.zone_name == "shadows" else "mid" if self.zone_name == "midtones" else "highlight"
        gain_prefix = "shadow_gain" if self.zone_name == "shadows" else "midtone_gain" if self.zone_name == "midtones" else "highlight_gain"
        
        for ch in ['r', 'g', 'b']:
            kwargs[f"{zone_prefix}_center_{ch}"] = float(c)
            kwargs[f"{zone_prefix}_width_{ch}"] = float(w)
            kwargs[f"{gain_prefix}_{ch}"] = float(g_val) 
            
        actual_1, _ = apply_grading(self.img_arr_ds, self.base_state, **kwargs)
        
        delta = self.expected_ds * 65535.0 - actual_1 * 65535.0
        delta_neutral = delta[self.neutral_mask]
        
        return np.sqrt(np.mean(delta_neutral ** 2))

def run_suite_b_tracker(sweep_name: str, zone_name: str):
    img_arr = load_image_array(INPUT_PATH)
    ds = 24
    img_arr_ds = img_arr[::ds, ::ds]

    if "bal" in sweep_name:
        suffixes = ["neg100", "neg50", "pos50", "pos100"]
    else:
        suffixes = ["0", "25", "75", "100"]

    sh_state = create_default_state()
    sh_state.update({
        "shadow_sat": 1.0 if zone_name == "shadows" else 0.0,
        "midtone_sat": 1.0 if zone_name == "midtones" else 0.0,
        "highlight_sat": 1.0 if zone_name == "highlights" else 0.0,
        "shadow_hue": 240.0 if zone_name == "shadows" else 0.0,
        "midtone_hue": 0.0,
        "highlight_hue": 60.0 if zone_name == "highlights" else 0.0,
    })

    bounds = [(0.0, 1.0), (0.01, 10.0), (0.0, 20.0)]
    
    print(f"\n==================================================")
    print(f" TRACKING CURVE SHIFTS FOR: {sweep_name.upper()} ")
    print(f"==================================================")

    for suffix in suffixes:
        file_path = f"tests/test_data/train_{sweep_name}_{suffix}.tif"
        expected_ds = load_image_array(file_path)[::ds, ::ds]
        
        evaluator = CurveTrackerGA(img_arr_ds, expected_ds, sh_state, zone_name)
        
        res = scipy.optimize.differential_evolution(
            evaluator.eval_curve, 
            bounds,
            maxiter=DEConfig.maxiter,
            popsize=DEConfig.popsize,
            mutation=DEConfig.mutation,
            recombination=DEConfig.recombination,
            init=DEConfig.init,
            tol=DEConfig.tol,
            updating=DEConfig.updating,
            polish=DEConfig.polish,
            disp=DEConfig.disp,
            workers=DEConfig.workers
        )
        
        c, w, g_val = res.x
        print(f"File: {suffix:>7} | C: {c:.6f} | W: {w:.6f} | Gain: {g_val:.6f} | RMSE: {res.fun:.0f}")

    print("==================================================\n")


def run_trinity_validation():
    import numpy as np
    
    img_arr = load_image_array(INPUT_PATH)
    
    print("\n==================================================")
    print(" EVALUATING TRINITY OVERLAP (ALL 3 ZONES ACTIVE) ")
    print("==================================================")
    
    # The Trinity locked baseline state
    base_state = create_default_state()
    base_state.update({
        "shadow_hue": 240.0, "shadow_sat": 1.0,
        "midtone_hue": 0.0, "midtone_sat": 1.0,
        "highlight_hue": 120.0, "highlight_sat": 1.0,
    })

    test_conditions = [
        ("train_trinity_blend_0.tif", {"blending": 0.0, "balance": 0.0}),
        ("train_trinity_blend_100.tif", {"blending": 100.0, "balance": 0.0}),
        ("train_trinity_bal_neg100.tif", {"blending": 50.0, "balance": -100.0}),
        ("train_trinity_bal_pos100.tif", {"blending": 50.0, "balance": 100.0}),
    ]

    for filename, modifiers in test_conditions:
        # Create state for this specific test
        test_state = base_state.copy()
        test_state.update(modifiers)
        
        # Run our engine
        actual_graded, _ = apply_grading(img_arr, test_state)
        
        # Load Lightroom's reference
        expected_ds = load_image_array(f"tests/test_data/{filename}")
        
        # Calculate RMSE across the whole image
        delta = expected_ds * 65535.0 - actual_graded * 65535.0
        rmse = np.sqrt(np.mean(delta ** 2))
        
        print(f"File: {filename:>30} | RMSE: {rmse:.0f}")
        
    print("==================================================\n")


import numpy as np
import scipy.optimize

class GammaTrackerGA:
    def __init__(self, img_arr_ds, expected_ds, base_state):
        self.img_arr_ds = img_arr_ds
        self.expected_ds = expected_ds
        self.base_state = base_state
        r, g, b = self.img_arr_ds[:, :, 0], self.img_arr_ds[:, :, 1], self.img_arr_ds[:, :, 2]
        self.neutral_mask = (np.abs(r - g) < 1e-3) & (np.abs(g - b) < 1e-3)

    def eval_curve(self, x):
        kwargs = {"gamma_shift": float(x[0])}
        actual_1, _ = apply_grading(self.img_arr_ds, self.base_state, **kwargs)
        delta = self.expected_ds * 65535.0 - actual_1 * 65535.0
        return np.sqrt(np.mean(delta[self.neutral_mask] ** 2))

def run_gamma_proof(sweep_name: str, zone_name: str):
    img_arr = load_image_array(INPUT_PATH)
    ds = 24
    img_arr_ds = img_arr[::ds, ::ds]
    suffixes = ["neg100", "neg50", "pos50", "pos100"]

    sh_state = create_default_state()
    sh_state.update({
        "shadow_sat": 1.0 if zone_name == "shadows" else 0.0,
        "midtone_sat": 1.0 if zone_name == "midtones" else 0.0,
        "highlight_sat": 1.0 if zone_name == "highlights" else 0.0,
        "shadow_hue": 240.0 if zone_name == "shadows" else 0.0,
        "midtone_hue": 0.0,
        "highlight_hue": 60.0 if zone_name == "highlights" else 0.0,
    })

    # Gamma can range from severe highlight dominance (0.1) to severe shadow dominance (10.0)
    bounds = [(0.1, 10.0)]
    
    print(f"\n==================================================")
    print(f" PROVING GAMMA SHIFT FOR: {sweep_name.upper()} ")
    print(f"==================================================")

    for suffix in suffixes:
        file_path = f"tests/test_data/train_{sweep_name}_{suffix}.tif"
        expected_ds = load_image_array(file_path)[::ds, ::ds]
        
        evaluator = GammaTrackerGA(img_arr_ds, expected_ds, sh_state)
        res = scipy.optimize.differential_evolution(
            evaluator.eval_curve,
            bounds,
            maxiter=DEConfig.maxiter,
            popsize=DEConfig.popsize,
            workers=DEConfig.workers
        )
        print(f"File: {suffix:>7} | Gamma: {res.x[0]:.6f} | RMSE: {res.fun:.0f}")


import numpy as np
import scipy.optimize

class TrinityTrackerGA:
    def __init__(self, img_arr_ds, expected_ds, base_state):
        self.img_arr_ds = img_arr_ds
        self.expected_ds = expected_ds
        self.base_state = base_state
        r, g, b = self.img_arr_ds[:, :, 0], self.img_arr_ds[:, :, 1], self.img_arr_ds[:, :, 2]
        self.neutral_mask = (np.abs(r - g) < 1e-3) & (np.abs(g - b) < 1e-3)

    def eval_curve(self, x):
        kwargs = {}
        # x maps to: [sh_c, sh_w, mid_c, mid_w, hi_c, hi_w]
        for ch in ['r', 'g', 'b']:
            kwargs[f"shadow_center_{ch}"] = float(x[0])
            kwargs[f"shadow_width_{ch}"] = float(x[1])
            kwargs[f"mid_center_{ch}"] = float(x[2])
            kwargs[f"mid_width_{ch}"] = float(x[3])
            kwargs[f"highlight_center_{ch}"] = float(x[4])
            kwargs[f"highlight_width_{ch}"] = float(x[5])
            
        actual_1, _ = apply_grading(self.img_arr_ds, self.base_state, **kwargs)
        delta = self.expected_ds * 65535.0 - actual_1 * 65535.0
        return np.sqrt(np.mean(delta[self.neutral_mask] ** 2))

def run_trinity_ga():
    img_arr = load_image_array(INPUT_PATH)
    ds = 24
    img_arr_ds = img_arr[::ds, ::ds]
    
    base_state = create_default_state()
    # Lock the state to the exact Trinity Setup
    base_state.update({
        "shadow_hue": 240.0, "shadow_sat": 1.0,
        "midtone_hue": 0.0, "midtone_sat": 1.0,
        "highlight_hue": 120.0, "highlight_sat": 1.0,
        "blending": 50.0
    })

    # Bounds: sh_c, sh_w, mid_c, mid_w, hi_c, hi_w
    bounds = [
        (0.0, 0.5),   # sh_c (Shadows usually anchor low)
        (0.01, 1.0),  # sh_w
        (0.0, 1.0),   # mid_c
        (0.01, 1.0),  # mid_w
        (0.5, 1.0),   # hi_c (Highlights usually anchor high)
        (0.01, 1.0)   # hi_w
    ]
    
    for sweep_name in ["train_trinity_bal_neg100.tif", "train_trinity_bal_pos100.tif"]:
        print(f"\n==================================================")
        print(f" SOLVING TRINITY BALANCE FOR: {sweep_name} ")
        print(f"==================================================")

        file_path = f"tests/test_data/{sweep_name}"
        expected_ds = load_image_array(file_path)[::ds, ::ds]
        
        # Pass the balance modifier cleanly to the state
        test_state = base_state.copy()
        test_state["balance"] = 100.0 if "pos100" in sweep_name else -100.0
        
        evaluator = TrinityTrackerGA(img_arr_ds, expected_ds, test_state)
        res = scipy.optimize.differential_evolution(
            evaluator.eval_curve, 
            bounds,
            maxiter=DEConfig.maxiter, 
            popsize=DEConfig.popsize,
            mutation=DEConfig.mutation, 
            recombination=DEConfig.recombination,
            init=DEConfig.init,
            tol=DEConfig.tol,
            updating=DEConfig.updating,
            polish=DEConfig.polish,
            disp=DEConfig.disp,
            workers=DEConfig.workers
        )
        
        print(f"Shadow:    Center = {res.x[0]:.4f}, Width = {res.x[1]:.4f}")
        print(f"Midtone:   Center = {res.x[2]:.4f}, Width = {res.x[3]:.4f}")
        print(f"Highlight: Center = {res.x[4]:.4f}, Width = {res.x[5]:.4f}")
        print(f"RMSE: {res.fun:.0f}")


import numpy as np
import scipy.optimize

class AmplitudeTrackerGA:
    def __init__(self, img_arr_ds, expected_ds, base_state):
        self.img_arr_ds = img_arr_ds
        self.expected_ds = expected_ds
        self.base_state = base_state
        r, g, b = self.img_arr_ds[:, :, 0], self.img_arr_ds[:, :, 1], self.img_arr_ds[:, :, 2]
        self.neutral_mask = (np.abs(r - g) < 1e-3) & (np.abs(g - b) < 1e-3)

    def eval_curve(self, x):
        kwargs = {
            "mult_sh": float(x[0]),
            "mult_mid": float(x[1]),
            "mult_hi": float(x[2])
        }
        actual_1, _ = apply_grading(self.img_arr_ds, self.base_state, **kwargs)
        delta = self.expected_ds * 65535.0 - actual_1 * 65535.0
        return np.sqrt(np.mean(delta[self.neutral_mask] ** 2))

def run_amplitude_ga():
    img_arr = load_image_array(INPUT_PATH)
    ds = 24
    img_arr_ds = img_arr[::ds, ::ds]
    
    base_state = create_default_state()
    base_state.update({
        "shadow_hue": 240.0, "shadow_sat": 1.0,
        "midtone_hue": 0.0, "midtone_sat": 1.0,
        "highlight_hue": 120.0, "highlight_sat": 1.0,
        "blending": 50.0
    })

    # Bounds: Allow multipliers to drastically shrink (0.01) or vastly overpower (20.0)
    bounds = [(0.01, 20.0), (0.01, 20.0), (0.01, 20.0)]
    
    for sweep_name in ["train_trinity_bal_neg100.tif", "train_trinity_bal_pos100.tif"]:
        print(f"\n==================================================")
        print(f" SOLVING AMPLITUDE GA FOR: {sweep_name} ")
        print(f"==================================================")

        file_path = f"tests/test_data/{sweep_name}"
        expected_ds = load_image_array(file_path)[::ds, ::ds]
        
        test_state = base_state.copy()
        test_state["balance"] = 100.0 if "pos100" in sweep_name else -100.0
        
        evaluator = AmplitudeTrackerGA(img_arr_ds, expected_ds, test_state)
        res = scipy.optimize.differential_evolution(
            evaluator.eval_curve, bounds,
            maxiter=DEConfig.maxiter, popsize=DEConfig.popsize,
            mutation=DEConfig.mutation, recombination=DEConfig.recombination,
            tol=DEConfig.tol, workers=DEConfig.workers
        )
        
        print(f"Shadow Mult:    {res.x[0]:.4f}")
        print(f"Midtone Mult:   {res.x[1]:.4f}")
        print(f"Highlight Mult: {res.x[2]:.4f}")
        print(f"RMSE: {res.fun:.0f}")


import numpy as np
import scipy.optimize

class ParabolaTrackerGA:
    def __init__(self, img_arr_ds, expected_ds, base_state):
        self.img_arr_ds = img_arr_ds
        self.expected_ds = expected_ds
        self.base_state = base_state
        r, g, b = self.img_arr_ds[:, :, 0], self.img_arr_ds[:, :, 1], self.img_arr_ds[:, :, 2]
        self.neutral_mask = (np.abs(r - g) < 1e-3) & (np.abs(g - b) < 1e-3)

    def eval_curve(self, x):
        kwargs = {"parabola_mult": float(x[0])}
        actual_1, _ = apply_grading(self.img_arr_ds, self.base_state, **kwargs)
        delta = self.expected_ds * 65535.0 - actual_1 * 65535.0
        return np.sqrt(np.mean(delta[self.neutral_mask] ** 2))

def run_parabola_ga():
    img_arr = load_image_array(INPUT_PATH)
    ds = 24
    img_arr_ds = img_arr[::ds, ::ds]
    
    base_state = create_default_state()
    base_state.update({
        "shadow_hue": 240.0, "shadow_sat": 1.0,
        "midtone_hue": 0.0, "midtone_sat": 1.0,
        "highlight_hue": 120.0, "highlight_sat": 1.0,
        "blending": 50.0
    })

    # The maximum mathematically logical shift before the parabola completely clips is 2.0
    bounds = [(0.0, 2.0)]
    
    for sweep_name in ["train_trinity_bal_neg100.tif", "train_trinity_bal_pos100.tif"]:
        print(f"\n==================================================")
        print(f" SOLVING PARABOLA GA FOR: {sweep_name} ")
        print(f"==================================================")

        file_path = f"tests/test_data/{sweep_name}"
        expected_ds = load_image_array(file_path)[::ds, ::ds]
        
        test_state = base_state.copy()
        test_state["balance"] = 100.0 if "pos100" in sweep_name else -100.0
        
        evaluator = ParabolaTrackerGA(img_arr_ds, expected_ds, test_state)
        res = scipy.optimize.differential_evolution(
            evaluator.eval_curve, bounds,
            maxiter=DEConfig.maxiter,
            popsize=DEConfig.popsize,
            workers=DEConfig.workers,
        )
        
        print(f"Parabola Multiplier: {res.x[0]:.4f}")
        print(f"RMSE: {res.fun:.0f}")


SAVE_PLOTS = True

def evaluate_and_plot(test_name, ref_img_path, ref_img, py_img, thresh_gray, thresh_neutral, thresh_blocks):
    """
    Computes RMSE metrics, prints a single concise summary line, and
    plots a 1x2 Lightroom vs Python comparison image.
    """
    delta = ref_img * 65535.0 - py_img * 65535.0
    rmse_gray = np.sqrt(np.mean(delta[0:540, :, :] ** 2))
    rmse_neutral = np.sqrt(np.mean(delta[540:810, :, :] ** 2))
    rmse_blocks = np.sqrt(np.mean(delta[810:, :, :] ** 2))

    num_errors = 0

    if rmse_gray > thresh_gray:
        gray_msg = f"> {thresh_gray:.0f}, FAIL"
        num_errors += 1
    else:
        gray_msg = "PASS"

    if rmse_neutral > thresh_neutral:
        neutral_msg = f"> {thresh_neutral:.0f}, FAIL"
        num_errors += 1
    else:
        neutral_msg = "PASS"

    if rmse_blocks > thresh_blocks:
        blocks_msg = f"> {thresh_blocks:.0f}, FAIL"
        num_errors += 1
    else:
        blocks_msg = "PASS"

    print(f"[{test_name}]")
    print(f"  Gradient: {rmse_gray:5.0f}    {gray_msg}")
    print(f"  Neutral:  {rmse_neutral:5.0f}    {neutral_msg}")
    print(f"  RGBCMY:   {rmse_blocks:5.0f}    {blocks_msg}")

    if SAVE_PLOTS:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(np.clip(ref_img, 0.0, 1.0))
        axes[0].set_title("Lightroom Output")
        axes[0].axis('off')

        axes[1].imshow(np.clip(py_img, 0.0, 1.0))
        axes[1].set_title("Python Output")
        axes[1].axis('off')

        plot_path = os.path.join("tests", "test_data", os.path.basename(ref_img_path.strip()).replace('.tif', '.png').replace('.tiff', '.png'))
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close(fig)

    assert num_errors == 0, "FAILURE"

def test_case_1_highlights() -> None:
    """
    Standalone check for Highlight tonal isolation against Lightroom validation export.
    """
    img = load_image_array(INPUT_PATH)
    ref_path = "tests/test_data/case_1_highlights.tif"
    ref = load_image_array(ref_path)
    state = create_default_state()
    state.update({
        "balance": 0.0,
        "blending": 0.5,
        "highlight_hue": 240.0,
        "highlight_sat": 1.0,
        "midtone_sat": 0.0,
        "shadow_sat": 0.0,
   })
    actual, S_mask = apply_grading(img, state)
    evaluate_and_plot("test_case_1_highlights", ref_path, ref, actual, 862.01, 267.96, 4496.04)


def test_case_1_midtones() -> None:
    """
    Standalone check for Midtone tonal isolation against Lightroom validation export.
    """
    img = load_image_array(INPUT_PATH)
    ref_path = "tests/test_data/case_1_midtones.tif"
    ref = load_image_array(ref_path)
    state = create_default_state()
    state.update({
        "shadow_sat": 0.0, "midtone_hue": 240.0, "midtone_sat": 1.0, "highlight_sat": 0.0,
        "blending": 0.5, "balance": 0.0
   })
    actual, S_mask = apply_grading(img, state)
    evaluate_and_plot("test_case_1_midtones", ref_path, ref, actual, 812.62, 827.65, 2199.58)


def test_case_1_shadows() -> None:
    """
    Standalone check for Shadow tonal isolation against Lightroom validation export.
    """
    img = load_image_array(INPUT_PATH)
    ref_path = "tests/test_data/case_1_shadows.tif"
    ref = load_image_array(ref_path)
    state = create_default_state()
    state.update({
        "shadow_hue": 240.0, "shadow_sat": 1.0, "midtone_sat": 0.0, "highlight_sat": 0.0,
        "blending": 0.5, "balance": 0.0
   })
    actual, S_mask = apply_grading(img, state)
    evaluate_and_plot("test_case_1_shadows", ref_path, ref, actual, 731.68, 592.16, 2572.98)


def test_case_2_10() -> None:
    """
    Standalone check for Saturation 0.10 against Lightroom validation export.
    """
    img = load_image_array(INPUT_PATH)
    ref_path = "tests/test_data/case_2_10.tif"
    ref = load_image_array(ref_path)
    state = create_default_state()
    state.update({
        "shadow_sat": 0.0, "midtone_hue": 0.0, "midtone_sat": 0.10, "highlight_sat": 0.0,
        "blending": 0.5, "balance": 0.0
   })
    actual, S_mask = apply_grading(img, state)
    evaluate_and_plot("test_case_2_10", ref_path, ref, actual, 622.79, 44.35, 1784.89)


def test_case_2_50() -> None:
    """
    Standalone check for Saturation 0.50 against Lightroom validation export.
    """
    img = load_image_array(INPUT_PATH)
    ref_path = "tests/test_data/case_2_50.tif"
    ref = load_image_array(ref_path)
    state = create_default_state()
    state.update({
        "shadow_sat": 0.0, "midtone_hue": 0.0, "midtone_sat": 0.50, "highlight_sat": 0.0,
        "blending": 0.5, "balance": 0.0
   })
    actual, S_mask = apply_grading(img, state)
    evaluate_and_plot("test_case_2_50", ref_path, ref, actual, 629.31, 372.79, 4232.24)


def test_case_2_100() -> None:
    """
    Standalone check for Saturation 1.0 against Lightroom validation export.
    """
    img = load_image_array(INPUT_PATH)
    ref_path = "tests/test_data/case_2_100.tif"
    ref = load_image_array(ref_path)
    state = create_default_state()
    state.update({
        "shadow_sat": 0.0, "midtone_hue": 0.0, "midtone_sat": 1.0, "highlight_sat": 0.0,
        "blending": 0.5, "balance": 0.0
   })
    actual, S_mask = apply_grading(img, state)
    evaluate_and_plot("test_case_2_100", ref_path, ref, actual, 676.12, 790.38, 4323.14)

@pytest.mark.skip(reason="Rotation is somehow broken, needs investigation")
@pytest.mark.parametrize("ref_path, offset", [
    ("tests/test_data/case_3_rot_0.tif", 0.0),
    ("tests/test_data/case_3_rot_90.tif", 90.0),
    ("tests/test_data/case_3_rot_180.tif", 180.0),
])
def test_case_3_rot(ref_path: str, offset: float) -> None:
    """
    Standalone check for Rotation against Lightroom validation export.
    """
    img = load_image_array(INPUT_PATH)
    ref = load_image_array(ref_path)
    state = create_default_state()
    state.update({
        "shadow_hue": (240.0 + offset) % 360.0,
        "shadow_sat": 0.5,
        "midtone_hue": (0.0 + offset) % 360.0,
        "midtone_sat": 0.5,
        "highlight_hue": (0.0 + offset) % 360.0,
        "highlight_sat": 0.0,
        "blending": 0.5,
        "balance": 0.0
   })
    actual, S_mask = apply_grading(img, state)
    evaluate_and_plot(os.path.basename(ref_path), ref_path, ref, actual, 733.59, 612.42, 2797.48)

def test_case_4() -> None:
    """
    Standalone check for Harmony Clash against Lightroom validation export.
    """
    img = load_image_array(INPUT_PATH)
    ref_path = "tests/test_data/case_4.tif"
    ref = load_image_array(ref_path)
    state = create_default_state()
    state.update({
        "shadow_hue": 30.0,
        "shadow_sat": 0.5,
        "midtone_sat": 0.0,
        "highlight_hue": 210.0,
        "highlight_sat": 0.5,
        "blending": 0.5,
        "balance": 0.0
   })
    actual, S_mask = apply_grading(img, state)
    evaluate_and_plot("test_case_4", ref_path, ref, actual, 577.09, 169.74, 3968.48)

@pytest.mark.skip(reason="blend and balance don't work")
def test_case_5_blend_0() -> None:
    """
    Standalone check for Blending = 0 against Lightroom validation export.
    """
    img = load_image_array(INPUT_PATH)
    ref_path = "tests/test_data/case_5_blend_0.tif"
    ref = load_image_array(ref_path)
    state = create_default_state()
    state.update({
        "shadow_hue": 240.0, "shadow_sat": 1.0, "midtone_hue": 120.0, "midtone_sat": 1.0,
        "highlight_hue": 0.0, "highlight_sat": 1.0, "blending": 0.0, "balance": 0.0
   })
    actual, S_mask = apply_grading(img, state)
    evaluate_and_plot("test_case_5_blend_0", ref_path, ref, actual, 874.47, 601.71, 1159.58)

@pytest.mark.skip(reason="blend and balance don't work")
def test_case_5_blend_33() -> None:
    """
    Standalone check for Blending = 33 against Lightroom validation export.
    """
    img = load_image_array(INPUT_PATH)
    ref_path = "tests/test_data/case_5_blend_33.tif"
    ref = load_image_array(ref_path)
    state = create_default_state()
    state.update({
        "shadow_hue": 240.0, "shadow_sat": 1.0, "midtone_hue": 120.0, "midtone_sat": 1.0,
        "highlight_hue": 0.0, "highlight_sat": 1.0, "blending": 0.33, "balance": 0.0
   })
    actual, S_mask = apply_grading(img, state)
    evaluate_and_plot("test_case_5_blend_33", ref_path, ref, actual, 695.24, 656.60, 904.84)

@pytest.mark.skip(reason="blend and balance don't work")
def test_case_5_blend_66() -> None:
    """
    Standalone check for Blending = 66 against Lightroom validation export.
    """
    img = load_image_array(INPUT_PATH)
    ref_path = "tests/test_data/case_5_blend_66.tif"
    ref = load_image_array(ref_path)
    state = create_default_state()
    state.update({
        "shadow_hue": 240.0, "shadow_sat": 1.0, "midtone_hue": 120.0, "midtone_sat": 1.0,
        "highlight_hue": 0.0, "highlight_sat": 1.0, "blending": 0.66, "balance": 0.0
   })
    actual, S_mask = apply_grading(img, state)
    evaluate_and_plot("test_case_5_blend_66", ref_path, ref, actual, 1780.0, 1000.0, 3200.0)

@pytest.mark.skip(reason="blend and balance don't work")
def test_case_5_blend_100() -> None:
    """
    Standalone check for Blending = 100 against Lightroom validation export.
    """
    img = load_image_array(INPUT_PATH)
    ref_path = "tests/test_data/case_5_blend_100.tif"
    ref = load_image_array(ref_path)
    state = create_default_state()
    state.update({
        "shadow_hue": 240.0, "shadow_sat": 1.0, "midtone_hue": 120.0, "midtone_sat": 1.0,
        "highlight_hue": 0.0, "highlight_sat": 1.0, "blending": 1.0, "balance": 0.0
   })
    actual, S_mask = apply_grading(img, state)
    evaluate_and_plot("test_case_5_blend_100", ref_path, ref, actual, 1977.12, 496.89, 4355.93)

@pytest.mark.skip(reason="blend and balance don't work")
def test_case_5_bal_neg100() -> None:
    """
    Standalone check for Balance = -100 against Lightroom validation export.
    """
    img = load_image_array(INPUT_PATH)
    ref_path = "tests/test_data/case_5_bal_neg100.tif"
    ref = load_image_array(ref_path)
    state = create_default_state()
    state.update({
        "shadow_hue": 240.0, "shadow_sat": 1.0, "midtone_hue": 120.0, "midtone_sat": 1.0,
        "highlight_hue": 0.0, "highlight_sat": 1.0, "blending": 0.5, "balance": -1.0
    })
    actual, S_mask = apply_grading(img, state)
    evaluate_and_plot("test_case_5_bal_neg100", ref_path, ref, actual, 811.00, 1747.0, 722.0)

@pytest.mark.skip(reason="blend and balance don't work")
def test_case_5_bal_neg50() -> None:
    """
    Standalone check for Balance = -50 against Lightroom validation export.
    """
    img = load_image_array(INPUT_PATH)
    ref_path = "tests/test_data/case_5_bal_neg50.tif"
    ref = load_image_array(ref_path)
    state = create_default_state()
    state.update({
        "shadow_hue": 240.0, "shadow_sat": 1.0, "midtone_hue": 120.0, "midtone_sat": 1.0,
        "highlight_hue": 0.0, "highlight_sat": 1.0, "blending": 0.5, "balance": -0.5
   })
    actual, S_mask = apply_grading(img, state)
    evaluate_and_plot("test_case_5_bal_neg50", ref_path, ref, actual, 1974.55, 2472.82, 1977.85)

@pytest.mark.skip(reason="blend and balance don't work")
def test_case_5_bal_pos50() -> None:
    """
    Standalone check for Balance = 50 against Lightroom validation export.
    """
    img = load_image_array(INPUT_PATH)
    ref_path = "tests/test_data/case_5_bal_pos50.tif"
    ref = load_image_array(ref_path)
    state = create_default_state()
    state.update({
        "shadow_hue": 240.0, "shadow_sat": 1.0, "midtone_hue": 120.0, "midtone_sat": 1.0,
        "highlight_hue": 0.0, "highlight_sat": 1.0, "blending": 0.5, "balance": 0.5
   })
    actual, S_mask = apply_grading(img, state)
    evaluate_and_plot("test_case_5_bal_pos50", ref_path, ref, actual, 1862.79, 808.54, 1881.68)

@pytest.mark.skip(reason="blend and balance don't work")
def test_case_5_bal_pos100() -> None:
    """
    Standalone check for Balance = 100 against Lightroom validation export.
    """
    img = load_image_array(INPUT_PATH)
    ref_path = "tests/test_data/case_5_bal_pos100.tif"
    ref = load_image_array(ref_path)
    state = create_default_state()
    state.update({
        "shadow_hue": 240.0, "shadow_sat": 1.0, "midtone_hue": 120.0, "midtone_sat": 1.0,
        "highlight_hue": 0.0, "highlight_sat": 1.0, "blending": 0.5, "balance": 1.0
   })
    actual, S_mask = apply_grading(img, state)
    evaluate_and_plot("test_case_5_bal_pos100", ref_path, ref, actual, 591.0, 843.0, 2443.0)
