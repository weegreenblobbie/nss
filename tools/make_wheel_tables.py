"""
Writes nss/color_gui/wheel_lut_tables.py: how Lightroom's Color Grading wheel is PAINTED, measured from screenshots of the active wheel.

Input: tests/test_data/lr_wheel_<H>.png for H = 0, 60, 120, 180, 240, 300 (each crop differs; every disk is fitted on its own).  The six images are
aligned on their disk centres and their per-pixel median is the handle-free composite.  From it:
    RIM_RGB      (360, 3) display RGB 0..1 at 0.95 R, one row per screen degree (0 at 3 o'clock, counter-clockwise), 5 angles x 4 radii averaged
    CENTRE_GREY  the grey the wheel blends to at its centre (least squares)
    BLEND_R, BLEND_G   g(r) on r/R = 0..1: colour = CENTRE_GREY + g(r) (RIM_RGB(angle) - CENTRE_GREY) in encoded RGB (least squares per radius)

    python -m tools.make_wheel_tables [composite.npy]      # optional: also save the composite (float RGB 0..255) for tools/validate_wheel_lut.py
"""
import os
import sys

import cv2
import numpy as np

HUES = (0, 60, 120, 180, 240, 300)
SHOTS = os.path.join("tests", "test_data", "lr_wheel_{}.png")
OUT = os.path.join("nss", "color_gui", "wheel_lut_tables.py")
N, C0 = 480, 239.5                  # common canvas and disk centre of the aligned images
RIM_R = 0.95                        # r/R of the rim samples (inside the antialiased edge)
BLEND_RADII = np.linspace(0.0, 1.0, 41)


def fit_disk(rgb: np.ndarray) -> tuple[float, float, float]:
    """(cx, cy, r) of the wheel disk: least-squares circle through its edge, refitted without points far off the circle."""
    mask = (np.abs(rgb - rgb[5, 5]).max(-1) > 12).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask)
    disk = (lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])).astype(np.uint8) * 255
    ey, ex = np.nonzero(cv2.Canny(disk, 50, 150))
    for _ in range(3):
        sol = np.linalg.lstsq(np.c_[2 * ex, 2 * ey, np.ones_like(ex)], (ex**2 + ey**2).astype(float), rcond=None)[0]
        r = float(np.sqrt(sol[2] + sol[0]**2 + sol[1]**2))
        keep = np.abs(np.hypot(ex - sol[0], ey - sol[1]) - r) < 1.5
        ex, ey = ex[keep], ey[keep]
    return float(sol[0]), float(sol[1]), r


def composite() -> tuple[np.ndarray, float]:
    """Median of the six aligned screenshots (RGB float 0..255, centre at (C0, C0)) and the mean disk radius in pixels."""
    stack, radii = [], []
    for h in HUES:
        rgb = cv2.cvtColor(cv2.imread(SHOTS.format(h)), cv2.COLOR_BGR2RGB).astype(np.float32)
        cx, cy, r = fit_disk(rgb)
        stack.append(cv2.warpAffine(rgb, np.float32([[1, 0, C0 - cx], [0, 1, C0 - cy]]), (N, N), flags=cv2.INTER_LINEAR, borderValue=(71, 71, 71)))
        radii.append(r)
    return np.median(np.stack(stack), 0), float(np.mean(radii))


def rim_table(comp: np.ndarray, radius: float) -> np.ndarray:
    """(360, 3) rim colour 0..1 per screen degree: mean of 5 angles (+-1 degree) x 4 radii (RIM_R +- 0.02)."""
    rows = []
    for deg in range(360):
        a = np.radians(np.linspace(deg - 1, deg + 1, 5))[:, None]
        q = np.linspace(RIM_R - 0.02, RIM_R + 0.02, 4)[None, :] * radius
        x, y = np.rint(C0 + q * np.cos(a)).astype(int), np.rint(C0 - q * np.sin(a)).astype(int)
        rows.append(comp[y, x].reshape(-1, 3).mean(0))
    return np.array(rows) / 255.0


def blend_profile(comp: np.ndarray, radius: float, rim: np.ndarray) -> tuple[float, np.ndarray]:
    """(centre grey, g on BLEND_RADII) minimising the RGB error of grey + g(r) (rim(angle) - grey) over the disc."""
    img = comp / 255.0
    yy, xx = np.mgrid[0:N, 0:N]
    rpx = np.hypot(xx - C0, yy - C0)
    r = rpx / radius
    ang = np.degrees(np.arctan2(C0 - yy, xx - C0)) % 360
    i = np.floor(ang).astype(int) % 360
    f = (ang - np.floor(ang))[..., None]
    rim_at = rim[i] * (1 - f) + rim[(i + 1) % 360] * f
    disc = (rpx < radius - 2) & (rpx > 8)          # no antialiased edge, no grey centre dot
    best = None
    for grey in np.linspace(0.90, 1.0, 51):
        X, Y = img - grey, rim_at - grey
        g = np.full(len(BLEND_RADII), np.nan)
        for k, q in enumerate(BLEND_RADII):
            sel = disc & (np.abs(r - q) < 0.0125)
            if sel.any():
                g[k] = (X[sel] * Y[sel]).sum() / (Y[sel] ** 2).sum()
        ok = np.isfinite(g)
        g = np.interp(BLEND_RADII, BLEND_RADII[ok], g[ok])
        g[0] = 0.0
        err = (((grey + np.interp(r, BLEND_RADII, g)[..., None] * Y) - img)[disc] ** 2).mean()
        if best is None or err < best[0]:
            best = (err, float(grey), g)
    return best[1], best[2]


def fmt(name: str, a: np.ndarray, per: int = 6) -> str:
    """`name = np.array([...])` with one table row (or `per` values) per line."""
    a = np.asarray(a, float)
    if a.ndim == 2:
        lines = [f"    [{', '.join(f'{v:.6f}' for v in row)}]," for row in a]
    else:
        lines = ["    " + ", ".join(f"{v:.6f}" for v in a[i:i + per]) + "," for i in range(0, len(a), per)]
    return f"{name} = np.array([\n" + "\n".join(lines) + "\n])"


def main() -> None:
    """Measures the tables and writes OUT (and the composite if a path is given)."""
    comp, radius = composite()
    rim = rim_table(comp, radius)
    grey, g = blend_profile(comp, radius, rim)
    src = ('"""Lightroom Color Grading wheel painting, measured from the active-wheel screenshots.\n\n'
           'GENERATED by tools/make_wheel_tables.py; do not edit by hand.  Used through nss/color_gui/wheel_lut.py.\n"""\n'
           'import numpy as np\n\n'
           f'CENTRE_GREY = {grey:.4f}\n\n'
           + fmt("BLEND_R", BLEND_RADII) + "\n\n" + fmt("BLEND_G", g) + "\n\n" + fmt("RIM_RGB", rim) + "\n")
    with open(OUT, "w") as f:
        f.write(src)
    print(f"wrote {OUT}: rim {rim.shape}, centre grey {grey:.4f}, g(0.5) {np.interp(0.5, BLEND_RADII, g):.3f}, disk radius {radius:.2f} px")
    if len(sys.argv) > 1:
        np.save(sys.argv[1], comp)
        print("saved composite", sys.argv[1])


if __name__ == "__main__":
    main()
