"""Tests of nss.color_explore.harmony: wrapped differences, the best-fit rotation and the warped wedges."""
import numpy as np

from nss.color_explore.harmony import HARMONY_MODES, angdiff, best_rotation, in_wedge, median_rgb, nearest_wedge, wedge_arcs
from nss.color_mixer.hue_remap import hue_remap_forward, hue_remap_lut


def test_median_rgb_ignores_outliers_and_reads_rows_and_columns() -> None:
    """The 5 x 5 median at column 4, row 2 of a (6, 9, 3) image: one hot pixel does not move it; rows and columns are not swapped; borders cut."""
    img = np.zeros((6, 9, 3))
    img[..., 0] = np.arange(9)[None, :]                 # red = column
    img[..., 1] = np.arange(6)[:, None]                 # green = row
    img[..., 2] = 0.25
    img[2, 4] = (100.0, 100.0, 100.0)
    assert np.allclose(median_rgb(img, 4, 2), (4.0, 2.0, 0.25))
    assert np.allclose(median_rgb(img, 8, 0), (7.0, 1.0, 0.25))       # 3 columns (6..8) x 3 rows (0..2)


def band(k: int, v: float) -> np.ndarray:
    """Eight Hue levels, band k at v, the others 0."""
    a = np.zeros(8)
    a[k] = v
    return a


def test_angdiff_wraps_to_half_open_range() -> None:
    """Differences land in [-180, 180): 350 - 10 = -20, 10 - 350 = 20, a half turn is -180."""
    assert np.allclose(angdiff([350.0, 10.0, 180.0, 95.0], [10.0, 350.0, 0.0, 40.0]), [-20.0, 20.0, -180.0, 55.0])


def test_best_rotation_finds_the_samples() -> None:
    """Complementary samples at 37 and 217 fit R = 37; Triadic samples near 20 / 140 / 260 fit R = 20."""
    assert best_rotation([37.0, 217.0], HARMONY_MODES["Complementary"]["offsets"]) == 37.0
    assert best_rotation([18.0, 142.0, 260.0], HARMONY_MODES["Triadic"]["offsets"]) == 20.0


def test_best_rotation_ties_and_empty() -> None:
    """No samples (or mode None) gives 0; a tie picks the lowest R (one sample, Complementary: 70 and 250 both fit, 70 wins)."""
    assert best_rotation([], HARMONY_MODES["Triadic"]["offsets"]) == 0.0
    assert best_rotation([45.0], HARMONY_MODES["None"]["offsets"]) == 0.0
    assert best_rotation([250.0], HARMONY_MODES["Complementary"]["offsets"]) == 70.0


def test_nearest_wedge_assignment() -> None:
    """With R = 10, Split-Complementary centres are 10, 160, 220."""
    assert list(nearest_wedge([5.0, 150.0, 230.0, 350.0], 10.0, HARMONY_MODES["Split-Complementary"]["offsets"])) == [0, 1, 2, 0]


def test_neutral_mixer_wedge_is_undistorted() -> None:
    """With all Hue sliders at 0 the drawn wedge is the target wedge itself, also across 0 degrees."""
    lut = hue_remap_lut(np.zeros(8))
    (s, e), = wedge_arcs(lut, 100.0, 30.0)
    assert abs(s - 85.0) < 1e-2 and abs(e - 115.0) < 1e-2
    (s, e), = wedge_arcs(lut, 5.0, 30.0)
    assert abs(s - 350.0) < 1e-2 and abs(e - 380.0) < 1e-2


def test_warped_wedge_maps_into_the_target() -> None:
    """Green +80 warps the wedge at 120: every input hue inside the drawn arcs maps into the target, those just outside do not."""
    lut = hue_remap_lut(band(3, 80.0))
    arcs = wedge_arcs(lut, 120.0, 30.0)
    assert len(arcs) == 1
    s, e = arcs[0]
    assert abs((e - s) - 30.0) > 3.0                                      # stretched or compressed
    inside = np.linspace(s + 0.1, e - 0.1, 50)
    assert in_wedge(hue_remap_forward(lut, inside), 120.0, 30.0).all()
    assert not in_wedge(hue_remap_forward(lut, [s - 1.0, e + 1.0]), 120.0, 30.0).any()


def test_fold_over_wedge_has_several_arcs() -> None:
    """Orange +100 with Yellow -100 folds f over near 50: the wedge at 50 (width 6) is drawn as more than one arc."""
    lut = hue_remap_lut(band(1, 100.0) + band(2, -100.0))
    arcs = wedge_arcs(lut, 50.0, 6.0)
    assert len(arcs) > 1
    for s, e in arcs:
        assert in_wedge(hue_remap_forward(lut, 0.5 * (s + e)), 50.0, 6.0)
