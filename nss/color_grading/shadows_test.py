import os
import cv2
import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nss.color_grading.math import rgb_to_lab_pure
from nss.color_grading.shadows import LumaMaskShadows


def test_assertion_1d_array_raises():
    """Test that a 1D array raises an AssertionError requiring a 2D array."""
    engine = LumaMaskShadows()
    arr_1d = np.zeros(10, dtype=np.float32)
    with pytest.raises(AssertionError, match="must be a 2D array"):
        engine.get_mask(arr_1d)


def test_assertion_3d_array_raises():
    """Test that a 3D array raises an AssertionError requiring a 2D array."""
    engine = LumaMaskShadows()
    arr_3d = np.zeros((13, 11, 3), dtype=np.float32)
    with pytest.raises(AssertionError, match="must be a 2D array"):
        engine.get_mask(arr_3d)


def test_assertion_float64_dtype_raises():
    """Test that a float64 array raises an AssertionError requiring np.float32."""
    engine = LumaMaskShadows()
    arr_f64 = np.zeros((13, 11), dtype=np.float64)
    with pytest.raises(AssertionError, match="must be np.float32"):
        engine.get_mask(arr_f64)


def test_assertion_uint8_dtype_raises():
    """Test that a uint8 array raises an AssertionError requiring np.float32."""
    engine = LumaMaskShadows()
    arr_u8 = np.zeros((13, 11), dtype=np.uint8)
    with pytest.raises(AssertionError, match="must be np.float32"):
        engine.get_mask(arr_u8)


def test_assertion_non_array_raises():
    """Test that non-numpy array input raises an AssertionError."""
    engine = LumaMaskShadows()
    with pytest.raises(AssertionError, match="must be a numpy array"):
        engine.get_mask([[0.1, 0.2], [0.3, 0.4]])


def test_valid_2d_array_execution():
    """Test that a valid 2D float32 array produces a matching 2D mask."""
    engine = LumaMaskShadows()
    # Non-square dimensions with non-zero distinct values (Rules 20.1 & 20.2)
    lum = np.linspace(0.05, 0.95, 13 * 11, dtype=np.float32).reshape(13, 11)
    mask = engine.get_mask(lum, balance=0, blend=50)

    assert isinstance(mask, np.ndarray)
    assert mask.shape == (13, 11)
    assert mask.dtype == np.float32 or mask.dtype == np.float64
    assert np.all(mask >= 0.0) and np.all(mask <= 1.0)
    assert np.max(mask) > 0.0


def test_engine_shadows_lightroom_validation():
    """Validate standalone engine accuracy against Lightroom synthetic ground truth."""
    input_path = "tests/test_data/input_synthetic.tif"
    if not os.path.exists(input_path):
        input_path = "/workspace/tests/test_data/input_synthetic.tif"

    im_in = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    orig_rgb = im_in[100:101, :, ::-1]
    orig_lab = rgb_to_lab_pure(orig_rgb)
    lum_array_2d = im_in[100:101, :, 0].astype(np.float32) / 65535.0

    engine = LumaMaskShadows()

    test_cases = [
        {
            "name": "Balance -50 (Blend 50)",
            "file": "tests/test_data/set_c_sh_bal_neg50.tif",
            "balance": -50,
            "blend": 50,
            "color": "#1E88E5",
        },
        {
            "name": "Balance +100 (Blend 50)",
            "file": "tests/test_data/set_c_sh_bal_pos100.tif",
            "balance": 100,
            "blend": 50,
            "color": "#D81B60",
        },
        {
            "name": "Blend 75 (Balance 0)",
            "file": "tests/test_data/set_c_sh_blend_75.tif",
            "balance": 0,
            "blend": 75,
            "color": "#7CB342",
        },
    ]

    fig, axes = plt.subplots(3, 1, figsize=(11, 13), sharex=True)
    results = []

    for i, tc in enumerate(test_cases):
        fpath = tc["file"]
        if not os.path.exists(fpath):
            fpath = os.path.join("/workspace", fpath)

        im_test = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
        test_rgb = im_test[100:101, :, ::-1]
        test_lab = rgb_to_lab_pure(test_rgb)
        delta_b = np.abs(test_lab[0, :, 2] - orig_lab[0, :, 2])
        actual_norm = delta_b / np.max(delta_b)

        engine_mask_2d = engine.get_mask(lum_array_2d, balance=tc["balance"], blend=tc["blend"])
        engine_mask_1d = engine_mask_2d[0]
        rmse_16bit = np.sqrt(np.mean((actual_norm - engine_mask_1d) ** 2)) * 65535.0
        results.append((tc["name"], rmse_16bit))

        assert rmse_16bit < 150.0, f"RMSE {rmse_16bit} exceeded threshold for {tc['name']}"

        ax = axes[i]
        ax.plot(lum_array_2d[0], actual_norm, color=tc["color"], linewidth=2.2, linestyle="-", label="Actual Lightroom")
        ax.plot(lum_array_2d[0], engine_mask_1d, color="black", linewidth=1.8, linestyle="--", label="Engine (LumaMaskShadows)")
        ax.set_title(f"{tc['name']} — RMSE: {rmse_16bit:.1f} DN ({rmse_16bit / 655.35:.2f}% error)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Normalized Mask [0, 1]", fontsize=10)
        ax.set_ylim(-0.05, 1.08)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper right", fontsize=10)

    axes[-1].set_xlabel("Relative Input Luminance (0.0 = Black, 1.0 = White)", fontsize=11)
    axes[-1].set_xlim(0.0, 1.0)
    plt.tight_layout()

    out_paths = [
        "tests/test_data/nss_color_grading_shadows_test.png",
    ]
    for p in out_paths:
        try:
            fig.savefig(p, dpi=150)
        except Exception:
            pass

    plt.close(fig)

    print("\n================ FINAL ENGINE VALIDATION RESULTS ================")
    for name, rmse in results:
        pct = rmse / 655.35
        print(f"  {name:30s} -> RMSE: {rmse:6.1f} DN ({pct:5.2f}% error)")
    print("=================================================================\n")


def test_blind_off_grid_validation():
    """Blind off-grid validation against unseen Lightroom parameter states."""
    import sys

    input_path = "tests/test_data/input_synthetic.tif"
    if not os.path.exists(input_path):
        input_path = "/workspace/tests/test_data/input_synthetic.tif"

    im_in = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    orig_rgb = im_in[100:101, :, ::-1]
    orig_lab = rgb_to_lab_pure(orig_rgb)
    luminance_array = np.linspace(0.0, 1.0, 2048, dtype=np.float32).reshape(1, -1)

    engine = LumaMaskShadows()

    test_cases = [
        ("val_sh_bal_neg33_blend_63.tif", -33, 63),
        ("val_sh_bal_pos42_blend_18.tif", 42, 18),
        ("val_sh_bal_pos86_blend_91.tif", 86, 91),
    ]

    colors = ["#1E88E5", "#FB8C00", "#D81B60"]
    fig, axes = plt.subplots(3, 1, figsize=(11, 13), sharex=True)
    results = []

    for idx, (filename, bal, blend) in enumerate(test_cases):
        fpath = os.path.join("tests/test_data", filename)
        if not os.path.exists(fpath):
            fpath = os.path.join("/workspace/tests/test_data", filename)

        im_val = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
        test_rgb = im_val[100:101, :, ::-1]
        test_lab = rgb_to_lab_pure(test_rgb)
        delta_b = np.abs(test_lab[0, :, 2] - orig_lab[0, :, 2])
        ground_truth = delta_b / np.max(delta_b)

        engine_mask = engine.get_mask(luminance_array, balance=bal, blend=blend)[0]
        rmse_16bit = np.sqrt(np.mean((ground_truth - engine_mask) ** 2)) * 65535.0
        results.append((filename, bal, blend, rmse_16bit))

        ax = axes[idx]
        x_axis = luminance_array[0]
        ax.plot(x_axis, ground_truth, color=colors[idx], linewidth=2.2, linestyle="-", label="Ground Truth (Lightroom)")
        ax.plot(x_axis, engine_mask, color="black", linewidth=1.8, linestyle="--", label="Engine (LumaMaskShadows)")
        ax.set_title(
            f"Blind Test: Bal {bal:+d}, Blend {blend:d} ({filename}) — RMSE: {rmse_16bit:.1f} DN ({rmse_16bit / 655.35:.2f}% error)",
            fontsize=11,
            fontweight="bold",
        )
        ax.set_ylabel("Normalized Mask [0, 1]", fontsize=10)
        ax.set_ylim(-0.05, 1.08)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper right", fontsize=10)

    axes[-1].set_xlabel("Relative Input Luminance (0.0 = Black, 1.0 = White)", fontsize=11)
    axes[-1].set_xlim(0.0, 1.0)
    plt.tight_layout()

    out_paths = [
        "tests/test_data/nss_color_grading_shadows_test_validation.png",
        "/workspace/tests/test_data/nss_color_grading_shadows_test_validation.png",
        "/workspace/nss_color_grading_shadows_test_validation.png",
        "/home/ubuntu/.gemini/antigravity-cli/brain/15b4b398-4b8f-443b-ae9f-5358cfa63f07/nss_color_grading_shadows_test_validation.png",
    ]
    for p in out_paths:
        try:
            fig.savefig(p, dpi=150)
        except Exception:
            pass

    plt.close(fig)

    # Print a Markdown table of the results to sys.stdout
    md_table = [
        "\n### Blind Off-Grid Validation Results\n",
        "| File | Balance | Blending | 16-bit DN RMSE | Relative Error |",
        "|:---|:---:|:---:|:---:|:---:|",
    ]
    for filename, bal, blend, rmse in results:
        err_pct = rmse / 655.35
        md_table.append(f"| `{filename}` | {bal:+d} | {blend:d} | **{rmse:.1f} DN** | **{err_pct:.2f}%** |")
    md_output = "\n".join(md_table) + "\n"
    sys.stdout.write(md_output)
    sys.stdout.flush()


if __name__ == "__main__":
    print("Running assertion tests...")
    test_assertion_1d_array_raises()
    print("  ✓ 1D array raises AssertionError")
    test_assertion_3d_array_raises()
    print("  ✓ 3D array raises AssertionError")
    test_assertion_float64_dtype_raises()
    print("  ✓ float64 dtype raises AssertionError")
    test_assertion_uint8_dtype_raises()
    print("  ✓ uint8 dtype raises AssertionError")
    test_assertion_non_array_raises()
    print("  ✓ Non-array raises AssertionError")
    test_valid_2d_array_execution()
    print("  ✓ Valid 2D array execution passed")
    print("\nRunning Lightroom synthetic validation...")
    test_engine_shadows_lightroom_validation()
    print("\nRunning Blind Off-Grid validation...")
    test_blind_off_grid_validation()
