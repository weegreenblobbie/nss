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
INPUT_PATH: str = "tests/test_data/reference_color_grading_input.tif"

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

class GAEvaluator:
    """
    *** IMMUTABLE CLASS ***
    Trains one color channel at a time over the entire image.
    Optimizes all 3 tonal zones (Shadows, Midtones, Highlights) simultaneously for the active channel.
    """
    def __init__(self, case_name, img_arr_ds, expected_ds, sh_state, channel_idx, channel_name):
        self.case_name = case_name
        self.img_arr_ds = img_arr_ds
        self.expected_ds = expected_ds
        self.sh_state = sh_state
        self.channel_idx = channel_idx
        self.channel_name = channel_name
        
        self.best_rmse = float('inf')
        self.gen = 0

    def eval_channel(self, x):
        # x is exactly 27 parameters for ONE channel
        # 9 Structural: sh_c, sh_w, sh_g, mid_c, mid_w, mid_g, hi_c, hi_w, hi_g
        # 18 Hue Weights: sh (6), mid (6), hi (6)
        c_sh, w_sh, g_sh, c_mid, w_mid, g_mid, c_hi, w_hi, g_hi = x[0:9]
        hue_sh = x[9:15]
        hue_mid = x[15:21]
        hue_hi = x[21:27]

        # Fill the INACTIVE channels with harmless defaults so the engine doesn't crash.
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

        # Inject the ACTIVE channel's live parameters
        ch_prefix = ['r', 'g', 'b'][self.channel_idx]
        
        kwargs[f"shadow_center_{ch_prefix}"] = float(c_sh)
        kwargs[f"shadow_width_{ch_prefix}"] = float(w_sh)
        kwargs[f"shadow_gain_{ch_prefix}"] = float(g_sh)
        kwargs[f"mid_center_{ch_prefix}"] = float(c_mid)
        kwargs[f"mid_width_{ch_prefix}"] = float(w_mid)
        kwargs[f"midtone_gain_{ch_prefix}"] = float(g_mid)
        kwargs[f"highlight_center_{ch_prefix}"] = float(c_hi)
        kwargs[f"highlight_width_{ch_prefix}"] = float(w_hi)
        kwargs[f"highlight_gain_{ch_prefix}"] = float(g_hi)

        kwargs[f"shadow_hue_w_{ch_prefix}"] = [float(v) for v in hue_sh]
        kwargs[f"mid_hue_w_{ch_prefix}"] = [float(v) for v in hue_mid]
        kwargs[f"hi_hue_w_{ch_prefix}"] = [float(v) for v in hue_hi]

        actual_1, _ = apply_grading(self.img_arr_ds, self.sh_state, **kwargs)
        
        # KEY DIFFERENCE: We only calculate RMSE for the current channel!
        delta = self.expected_ds[:, :, self.channel_idx] * 65535.0 - actual_1[:, :, self.channel_idx] * 65535.0
        return np.sqrt(np.mean(delta ** 2))

    def callback(self, xk, convergence=0):
        self.gen += 1
        current_rmse = self.eval_channel(xk)
        if current_rmse < self.best_rmse * 0.9:  # Print on 10% improvements
            self.best_rmse = current_rmse
            print(f"[{self.channel_name}] Gen {self.gen:4d} | Channel RMSE: {current_rmse:.0f}", flush=True)
        return self.best_rmse < 100.0


def run_ga_optimize(case_name: str, state_updates: dict, init_struct: list = None, init_hues: list = None) -> None:
    """
    *** IMMUTABLE MASTER RUNNER ***
    DO NOT MODIFY, DUPLICATE, OR ALTER THIS FUNCTION.
    """
    img_arr = load_image_array(INPUT_PATH)
    expected_img = load_image_array(f"tests/test_data/{case_name}.tif")

    ds = 24
    img_arr_ds = img_arr[::ds, ::ds]
    expected_ds = expected_img[::ds, ::ds]

    sh_state = create_default_state()
    sh_state.update(state_updates)

    # 27 Parameters per channel: 9 Structural, 18 Hue
    bounds = [
        (0.0, 0.5), (0.01, 10.0), (0.0, 20.0),  # Shadow: center, width, gain
        (0.2, 0.8), (0.01, 10.0), (0.0, 20.0),  # Midtone: center, width, gain
        (0.5, 1.0), (0.01, 10.0), (0.0, 20.0),  # Highlight: center, width, gain
    ] + [(-300.0, 300.0)] * 18

    results = {}

    for ch_name, ch_idx in [("Blue", 2)]: #[("Red", 0), ("Green", 1), ("Blue", 2)]:
        print(f"\n--- TRAINING {ch_name.upper()} CHANNEL ({case_name}) ---", flush=True)
        evaluator = GAEvaluator(case_name, img_arr_ds, expected_ds, sh_state, ch_idx, ch_name)
        
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
        print(f"{ch_name} Final RMSE: {evaluator.best_rmse:.0f}")
        results[ch_name] = res.x

    print("\n==================================================")
    print(f"ALL CHANNELS OPTIMIZED FOR: {case_name}")
    print("==================================================")
    
    for ch_name, x in results.items():
        ch = ch_name.lower()[0] # 'r', 'g', or 'b'
        print(f"\n--- {ch_name.upper()} PARAMETERS ---")
        print(f"Optimal shadow_center_{ch}: {x[0]:.6f} | width: {x[1]:.6f} | gain: {x[2]:.6f}")
        print(f"Optimal midtone_center_{ch}: {x[3]:.6f} | width: {x[4]:.6f} | gain: {x[5]:.6f}")
        print(f"Optimal highlight_center_{ch}: {x[6]:.6f} | width: {x[7]:.6f} | gain: {x[8]:.6f}")
        print(f"Optimal shadow_hue_w_{ch}: {[round(float(v), 6) for v in x[9:15]]}")
        print(f"Optimal mid_hue_w_{ch}: {[round(float(v), 6) for v in x[15:21]]}")
        print(f"Optimal hi_hue_w_{ch}: {[round(float(v), 6) for v in x[21:27]]}")

def run_ga_optimize_v2(case_name: str, state_updates: dict) -> None:
    """
    *** IMMUTABLE MASTER RUNNER V2 (SINGLE-STEP HIGHLIGHTS) ***
    DO NOT MODIFY, DUPLICATE, OR ALTER THIS FUNCTION.
    """
    img_arr = load_image_array(INPUT_PATH)
    expected_img = load_image_array(f"tests/test_data/{case_name}.tif")

    ds = 24
    img_arr_ds = img_arr[::ds, ::ds]
    expected_ds = expected_img[::ds, ::ds]

    sh_state = create_default_state()
    sh_state.update(state_updates)

    # Initialize Evaluator Class
    evaluator = GAEvaluator(case_name, img_arr_ds, expected_ds, sh_state, ds)

    # --------------------------------------------------------------------------
    # UNIFIED STEP: 27 PARAMETERS (9 STRUCTURAL + 18 HUE WEIGHTS)
    # --------------------------------------------------------------------------
    # 9 Structural: [center, width, gain] for R, G, B
    bounds_struct = [
        (0.5, 1.0), (0.01, 10.0), (0.0, 20.0),  # Red center, width, gain
        (0.5, 1.0), (0.01, 10.0), (0.0, 20.0),  # Green center, width, gain
        (0.5, 1.0), (0.01, 10.0), (0.0, 20.0)   # Blue center, width, gain
    ]
    # 18 Hue Weights: [hi_hue_w_r (6), hi_hue_w_g (6), hi_hue_w_b (6)]
    bounds_hue = [(-300.0, 300.0)] * 18
    
    bounds_total = bounds_struct + bounds_hue

    print(f"\n--- SINGLE-STEP OPTIMIZATION: 27 HIGHLIGHT PARAMETERS ({case_name}) ---", flush=True)
    res = scipy.optimize.differential_evolution(
        evaluator.eval_highlight_isolation, 
        bounds_total,
        maxiter=DEConfig.maxiter,
        popsize=DEConfig.popsize,
        mutation=DEConfig.mutation,
        recombination=DEConfig.recombination,
        init=DEConfig.init,
        tol=DEConfig.tol,
        updating=DEConfig.updating,
        polish=DEConfig.polish,
        disp=DEConfig.disp,
        callback=evaluator.callback_highlight_isolation,
        workers=DEConfig.workers
    )
    
    print(f"Final Evaluations: {res.nfev:6d} | Best RGBCMY RMSE: {evaluator.best_rmse:.0f}")
    x = res.x
    
    print("\n==================================================")
    print(f"OPTIMIZED HIGHLIGHT PARAMETERS ({case_name}):")
    print(f"Optimal highlight_center_r: {x[0]:.6f}")
    print(f"Optimal highlight_width_r:  {x[1]:.6f}")
    print(f"Optimal highlight_gain_r:   {x[2]:.6f}")
    print(f"Optimal highlight_center_g: {x[3]:.6f}")
    print(f"Optimal highlight_width_g:  {x[4]:.6f}")
    print(f"Optimal highlight_gain_g:   {x[5]:.6f}")
    print(f"Optimal highlight_center_b: {x[6]:.6f}")
    print(f"Optimal highlight_width_b:  {x[7]:.6f}")
    print(f"Optimal highlight_gain_b:   {x[8]:.6f}\n")
    
    print(f"OPTIMIZED 18 HIGHLIGHT HUE WEIGHTS ({case_name}):")
    print(f"Optimal hi_hue_w_r: {[round(float(v), 6) for v in x[9:15]]}")
    print(f"Optimal hi_hue_w_g: {[round(float(v), 6) for v in x[15:21]]}")
    print(f"Optimal hi_hue_w_b: {[round(float(v), 6) for v in x[21:27]]}")
    print("==================================================")


def run_ga_optimize_case_1_midtones() -> None:
    run_ga_optimize(
        case_name="case_1_midtones",
        state_updates={
            "balance": 0.0,
            "blending": 0.5,
            "highlight_sat": 0.0, 
            "midtone_hue": 240.0,
            "midtone_sat": 1.0,
            "shadow_sat": 0.0, 
        }
    )

def run_ga_optimize_case_1_highlights() -> None:
    run_ga_optimize(
        case_name="case_1_highlights",
        state_updates={
        "balance": 0.0,
        "blending": 0.5,
        "highlight_hue": 240.0,
        "highlight_sat": 1.0,
        "midtone_sat": 0.0,
        "shadow_sat": 0.0,
        }
    )


def run_ga_optimize_bal_pos100_protocol() -> None:  
    run_ga_optimize(
        case_name="case_5_bal_pos100",
        state_updates={"shadow_hue": 240.0, "shadow_sat": 1.0, "midtone_hue": 120.0, "midtone_sat": 1.0, "highlight_hue": 0.0, "highlight_sat": 1.0, "blending": 0.5, "balance": 1.0},
        init_struct=[0.306010, 1.124182, 0.616380, 0.172157, 0.953856, 0.595673, 2.843971, 2.652393, 0.665725, 0.021758, 0.128765, 1.538227, 0.188538, 1.481302, 0.859915, 2.883242, 0.224509, 0.164233, 1.002839, 1.158010, 2.924640]
    )


def run_ga_optimize_bal_neg100_protocol() -> None:
    run_ga_optimize(
        case_name="case_5_bal_neg100",
        state_updates={
            "shadow_hue": 240.0, "shadow_sat": 1.0, 
            "midtone_hue": 120.0, "midtone_sat": 1.0, 
            "highlight_hue": 0.0, "highlight_sat": 1.0, 
            "blending": 0.5, "balance": -1.0
        }
        # No initial seeds: let it explore the full 5.0 gain / 300.0 hue space
    )

def run_ga_optimize_bal_pos100_protocol() -> None:
    run_ga_optimize(
        case_name="case_5_bal_pos100",
        state_updates={
            "shadow_hue": 240.0, "shadow_sat": 1.0, 
            "midtone_hue": 120.0, "midtone_sat": 1.0, 
            "highlight_hue": 0.0, "highlight_sat": 1.0, 
            "blending": 0.5, "balance": -1.0
        }
        # No initial seeds: let it explore the full 5.0 gain / 300.0 hue space
    )

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

    print(f"[{test_name}] Computed Grayscale: {rmse_gray:.2f} (Threshold: {thresh_gray}) | Neutral: {rmse_neutral:.2f} (Threshold: {thresh_neutral}) | RGBCMY: {rmse_blocks:.2f} (Threshold: {thresh_blocks})")

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

    assert rmse_gray < thresh_gray, f"failed Grayscale: {rmse_gray}"
    assert rmse_neutral < thresh_neutral, f"failed Neutral Gray: {rmse_neutral}"
    assert rmse_blocks < thresh_blocks, f"failed Blocks: {rmse_blocks}"






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


def test_case_3_rot_0() -> None:
    """
    Standalone check for Rotation 0 against Lightroom validation export.
    """
    img = load_image_array(INPUT_PATH)
    ref_path = "tests/test_data/case_3_rot_0.tif"
    ref = load_image_array(ref_path)
    state = create_default_state()
    state.update({
        "shadow_hue": 240.0, "shadow_sat": 0.5, "midtone_hue": 0.0, "midtone_sat": 0.5,
        "highlight_sat": 0.0, "blending": 0.5, "balance": 0.0
   })
    actual, S_mask = apply_grading(img, state)
    evaluate_and_plot("test_case_3_rot_0", ref_path, ref, actual, 733.59, 612.42, 2797.48)


def test_case_3_rot_90() -> None:
    """
    Standalone check for Rotation 90 against Lightroom validation export.
    """
    img = load_image_array(INPUT_PATH)
    ref_path = "tests/test_data/case_3_rot_90.tif"
    ref = load_image_array(ref_path)
    state = create_default_state()
    state.update({
        "shadow_hue": 330.0, "shadow_sat": 0.5, "midtone_hue": 90.0, "midtone_sat": 0.5,
        "highlight_sat": 0.0, "blending": 0.5, "balance": 0.0
   })
    actual, S_mask = apply_grading(img, state)
    evaluate_and_plot("test_case_3_rot_90", ref_path, ref, actual, 1074.60, 581.86, 11615.92)


def test_case_3_rot_180() -> None:
    """
    Standalone check for Rotation 180 against Lightroom validation export.
    """
    img = load_image_array(INPUT_PATH)
    ref_path = "tests/test_data/case_3_rot_180.tif"
    ref = load_image_array(ref_path)
    state = create_default_state()
    state.update({
        "shadow_hue": 60.0, "shadow_sat": 0.5, "midtone_hue": 180.0, "midtone_sat": 0.5,
        "highlight_sat": 0.0, "blending": 0.5, "balance": 0.0
   })
    actual, S_mask = apply_grading(img, state)
    evaluate_and_plot("test_case_3_rot_180", ref_path, ref, actual, 1383.09, 907.41, 11074.65)


def test_case_4() -> None:
    """
    Standalone check for Harmony Clash against Lightroom validation export.
    """
    img = load_image_array(INPUT_PATH)
    ref_path = "tests/test_data/case_4.tif"
    ref = load_image_array(ref_path)
    state = create_default_state()
    state.update({
        "shadow_hue": 30.0, "shadow_sat": 0.5, "midtone_sat": 0.0, "highlight_hue": 210.0,
        "highlight_sat": 0.5, "blending": 0.5, "balance": 0.0
   })
    actual, S_mask = apply_grading(img, state)
    evaluate_and_plot("test_case_4", ref_path, ref, actual, 577.09, 169.74, 3968.48)


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
