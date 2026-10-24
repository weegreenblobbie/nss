"""
Centre the moon: find it in rendered 16-bit TIFFs with nss.utils.detect_moon (stars and other outlier edges dropped)
and compute crops centred exactly on it, for the Lightroom plug-in's "Moon: Center" / "Moon: Verify centering" (request /
result files: plugin/moon_bridge.py).

    python moon.py --request <request file>         one process for all the photos of the request; exit 0 once the result file is written
    python moon.py a.tif b.tif ... [--debug]        prints the findings; --debug also writes <name>_moon.png overlays next to the inputs

Coordinates: x = column, y = row in a continuous frame where pixel (row, col) covers [col, col + 1) x [row, row + 1), so detect_moon's
(row, col) centre is at x = col + 0.5, y = row + 0.5 and a crop fraction f is the pixel position f * width.  Lightroom stores the crop in the
photo's unrotated frame; the image is rendered in the displayed orientation, and Lightroom's orientation code (AB, BC, ...) maps between the
two through the measured table ORIENTATIONS.
"""
import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
import os
import sys
import warnings

import cv2
import numpy as np

from nss import utils
from nss.tiff_file import TiffFile
from plugin import moon_bridge

MIN_RADIUS = 3.0                # pixels; a smaller circle is not a moon
# Lightroom orientation code -> where the stored frame lands in the displayed frame, in displayed fractions: (origin = the stored top-left
# corner, u = the stored rightward unit edge, v = the stored downward unit edge); a stored point (s, t) is shown at origin + s u + t v.
# All measured in Lightroom Classic (center moon + verify).  Lightroom's letters follow no single corner rule:
# Flip Horizontal + Rotate Right reports AD and is the anti-diagonal flip.
ORIENTATIONS = {
    "AB": ((0, 0), (1, 0), (0, 1)),       # as shot
    "BC": ((1, 0), (0, 1), (-1, 0)),      # Rotate Right (clockwise)
    "CD": ((1, 1), (-1, 0), (0, -1)),     # 180
    "DA": ((0, 1), (0, -1), (1, 0)),      # Rotate Left (counter-clockwise)
    "BA": ((1, 0), (-1, 0), (0, 1)),      # Flip Horizontal
    "DC": ((0, 1), (1, 0), (0, -1)),      # Flip Vertical
    "AD": ((1, 1), (0, -1), (-1, 0)),     # Flip Horizontal + Rotate Right (= Flip Vertical + Rotate Left): anti-diagonal flip
    "CB": ((0, 0), (0, 1), (1, 0)),       # Flip Horizontal + Rotate Left: main-diagonal flip (transpose)
}


def find_moon(image: str | np.ndarray, size: int = 121,
              min_confidence: float = moon_bridge.DEFAULT_MIN_CONFIDENCE) -> tuple[float, float, float, float, str]:
    """(x, y, radius, confidence, problem) of the moon in a TIFF path or an image array (pixels), from detect_moon.  problem is '' for a
    moon, else why there is none (x, y, radius NaN): detect_moon raised, the circle is implausible (radius under MIN_RADIUS or the
    centre outside the image) or its confidence is below min_confidence."""
    array = TiffFile().read(image).array if isinstance(image, str) else np.asarray(image, np.float32)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)         # an image without edges divides 0 by 0 before the fit raises
            circle, confidence = utils.detect_moon(array, size)
    except ValueError as e:
        return np.nan, np.nan, np.nan, np.nan, f"no moon found ({e})"
    x, y, radius = float(circle.center[1]) + 0.5, float(circle.center[0]) + 0.5, float(circle.radius)
    height, width = array.shape[:2]
    if not (np.isfinite([x, y, radius]).all() and radius >= MIN_RADIUS and 0.0 < x < width and 0.0 < y < height):
        return np.nan, np.nan, np.nan, confidence, f"implausible circle: centre ({x:.1f}, {y:.1f}), radius {radius:.1f}"
    if confidence < min_confidence:
        return np.nan, np.nan, np.nan, confidence, f"confidence {confidence:.3f} below {min_confidence:.3f}"
    return x, y, radius, confidence, ""


def _axes(orientation: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(origin, u axis, v axis) in displayed fractions of the stored frame's top-left corner, rightward and downward unit edges."""
    if orientation not in ORIENTATIONS:
        raise ValueError(f"unknown orientation {orientation!r}")
    origin, u, v = ORIENTATIONS[orientation]
    return np.array(origin, np.float64), np.array(u, np.float64), np.array(v, np.float64)


def to_display(crop: tuple[float, float, float, float], orientation: str) -> tuple[float, float, float, float]:
    """A (left, top, right, bottom) crop in stored fractions as fractions of the displayed frame."""
    origin, u, v = _axes(orientation)
    corners = origin + np.array([crop[0], crop[2]])[:, None] * u + np.array([crop[1], crop[3]])[:, None] * v
    return float(corners[:, 0].min()), float(corners[:, 1].min()), float(corners[:, 0].max()), float(corners[:, 1].max())


def to_stored(crop: tuple[float, float, float, float], orientation: str) -> tuple[float, float, float, float]:
    """A (left, top, right, bottom) crop in displayed-frame fractions as Lightroom's stored fractions."""
    origin, u, v = _axes(orientation)
    corners = np.array([[crop[0], crop[1]], [crop[2], crop[3]]]) - origin
    du, dv = corners @ u, corners @ v
    return float(du.min()), float(dv.min()), float(du.max()), float(dv.max())


# With a crop angle, Lightroom stores the turned crop's top-left corner as (CropLeft, CropTop) and its bottom-right corner as (CropRight,
# CropBottom), in fractions of the unrotated photo: a crop w x h px turned by CropAngle has (right - left, bottom - top) = (w cos - h sin,
# w sin + h cos) px.  Measured in Lightroom Classic: an 8256 x 5504 photo, angle -2.21, stored 0.034218 0.069086 0.989082 0.946113, exported
# 7691 x 5127 px.  The centre is the midpoint of the two corners at any angle.

def _stored_size(width: int, height: int, orientation: str) -> tuple[int, int]:
    """The unrotated photo's (width, height) in pixels from the displayed one's."""
    return (height, width) if _axes(orientation)[1][0] == 0 else (width, height)


def crop_size(crop: tuple[float, float, float, float], width: int, height: int, orientation: str, angle: float) -> tuple[float, float]:
    """(w, h) in displayed pixels of a Lightroom crop (stored fractions, CropAngle in degrees) on a photo displayed width x height."""
    stored_w, stored_h = _stored_size(width, height, orientation)
    dx, dy = (crop[2] - crop[0]) * stored_w, (crop[3] - crop[1]) * stored_h
    c, s = np.cos(np.radians(angle)), np.sin(np.radians(angle))
    w, h = dx * c + dy * s, dy * c - dx * s
    return (h, w) if (stored_w, stored_h) != (width, height) else (w, h)


def lightroom_crop(crop: tuple[float, float, float, float], width: int, height: int, orientation: str, angle: float) -> tuple[float, float, float, float]:
    """Lightroom's stored (left, top, right, bottom) for an untilted crop in displayed fractions that Lightroom turns by angle (CropAngle,
    degrees) about its centre: the stored frame's top-left and bottom-right corners of the turned crop."""
    left, top, right, bottom = to_stored(crop, orientation)
    stored_w, stored_h = _stored_size(width, height, orientation)
    w, h = (right - left) * stored_w, (bottom - top) * stored_h
    c, s = np.cos(np.radians(angle)), np.sin(np.radians(angle))
    half_x, half_y = (w * c - h * s) / (2 * stored_w), (w * s + h * c) / (2 * stored_h)
    cx, cy = (left + right) / 2, (top + bottom) / 2
    return cx - half_x, cy - half_y, cx + half_x, cy + half_y


def max_crop(cx: float, cy: float, width: int, height: int, aspect: float, angle: float = 0.0) -> tuple[float, float]:
    """(w, h) in pixels of the largest crop of aspect w / h, turned by angle degrees about its centre (cx, cy), whose four corners stay inside
    the image: its half-extents w/2 |cos| + h/2 |sin| and w/2 |sin| + h/2 |cos| are at most min(cx, width - cx) and min(cy, height - cy)."""
    c, s = abs(np.cos(np.radians(angle))), abs(np.sin(np.radians(angle)))
    w = min(2.0 * min(cx, width - cx) / (c + s / aspect), 2.0 * min(cy, height - cy) / (s + c / aspect))
    return w, w / aspect


def crop_for(cx: float, cy: float, width: int, height: int, aspect: float, size_px: tuple[float, float] | None = None,
             angle: float = 0.0) -> tuple[float, float, float, float]:
    """(left, top, right, bottom) fractions of a crop centred exactly on (cx, cy) pixels, before Lightroom turns it by angle degrees about its
    centre: size_px (w, h) when it fits (turned) inside the image, else the max_crop of aspect w / h (0 = the image's own aspect)."""
    aspect = aspect or width / height
    w, h = max_crop(cx, cy, width, height, aspect, angle)
    if size_px is not None and size_px[0] <= w * (1 + 1e-9) and size_px[1] <= h * (1 + 1e-9):
        w, h = size_px
    return (cx - w / 2) / width, (cy - h / 2) / height, (cx + w / 2) / width, (cy + h / 2) / height


def common_size(found: list[dict], aspect: float) -> tuple[float, float]:
    """(w, h): the largest crop of this aspect that fits every found moon (turned by its own crop angle; the minimum of their max_crop widths)."""
    w = min(max_crop(r["cx"], r["cy"], r["width"], r["height"], aspect, r["angle"])[0] for r in found)
    return w, w / aspect


def measure(photo: moon_bridge.Photo, size: int, min_confidence: float) -> dict:
    """A photo's measure result: status, message, the moon and its confidence, the image size, the request crop's size (crop_width,
    crop_height, image pixels) and the moon's offset (dx, dy) from that crop's centre; orientation and angle are kept for add_crops."""
    try:
        array = TiffFile().read(photo.image).array
        height, width = array.shape[:2]
        left, top, right, bottom = to_display(photo.crop, photo.orientation)
        cx, cy, radius, confidence, problem = find_moon(array, size, min_confidence)
    except Exception as e:      # an unreadable image or a bad request line fails this photo only
        return {"status": "error", "message": f"{type(e).__name__}: {e}"}
    ok = not problem
    crop_width, crop_height = crop_size(photo.crop, width, height, photo.orientation, photo.angle)
    result = {"status": "ok" if ok else "notfound", "message": problem or "moon found", "width": width, "height": height,
              "crop_width": crop_width, "crop_height": crop_height, "orientation": photo.orientation, "angle": photo.angle}
    if np.isfinite(confidence):
        result["confidence"] = confidence
    if ok:
        result.update(cx=cx, cy=cy, radius=radius, dx=cx - (left + right) / 2 * width, dy=cy - (top + bottom) / 2 * height)
    return result


def write_progress(path: str, done: int, total: int) -> None:
    """Replaces the progress file with '<done> <total>' in one step, so a reader never sees half a line.  While the plug-in has the file
    open Windows refuses the replace; that update is skipped (the next one, or the result file, follows)."""
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        f.write(f"{done} {total}\n")
    try:
        os.replace(path + ".tmp", path)
    except PermissionError:
        pass


def measure_all(request: moon_bridge.Request) -> list[dict]:
    """measure() of each photo of a measure request, results in request order, request.workers photos at a time: in that many processes,
    or in one thread of this process for 1 worker.  A photo is started only when a worker is free, so once the cancel file (if given) exists
    no further photo starts and those left are status canceled.  After each finished photo the progress file (if given) gets
    '<done> <count>'."""
    results = [{"status": "canceled", "message": "canceled"} for _ in request.photos]
    waiting = list(enumerate(request.photos))
    done = 0
    pool = ProcessPoolExecutor(request.workers) if request.workers > 1 else ThreadPoolExecutor(1)
    with pool:
        running = {}
        while waiting or running:
            while waiting and len(running) < request.workers and not (request.cancel and os.path.exists(request.cancel)):
                i, photo = waiting.pop(0)
                running[pool.submit(measure, photo, request.size, request.min_confidence)] = i
            if not running:
                break
            finished, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in finished:
                i = running.pop(future)
                try:
                    results[i] = future.result()
                except Exception as e:      # a worker process that died (e.g. out of memory) fails its photo
                    results[i] = {"status": "error", "message": f"{type(e).__name__}: {e}"}
                done += 1
                if request.progress:
                    write_progress(request.progress, done, len(request.photos))
    return results


def add_crops(results: list[dict], mode: str) -> None:
    """Adds the new crop (Lightroom's stored fractions; the crop angle is kept) to each ok result.  single: each keeps its current crop size if
    it fits, else the largest of its aspect; batch: one common pixel size, the aspect of the first found photo."""
    found = [r for r in results if r["status"] == "ok"]
    if not found:
        return
    aspect = found[0]["crop_width"] / found[0]["crop_height"]
    size = common_size(found, aspect) if mode == "batch" else None
    for r in found:
        own_aspect = aspect if mode == "batch" else r["crop_width"] / r["crop_height"]
        crop = crop_for(r["cx"], r["cy"], r["width"], r["height"], own_aspect, size or (r["crop_width"], r["crop_height"]), r["angle"])
        r.update(zip(("left", "top", "right", "bottom"), lightroom_crop(crop, r["width"], r["height"], r["orientation"], r["angle"])))
        r["display_crop"] = crop


def run(request: moon_bridge.Request) -> list[dict]:
    """A request's results, in order: measure_all for measure; for single / batch, its measurements with add_crops' new crops."""
    if request.mode == "measure":
        return measure_all(request)
    add_crops(request.measured, request.mode)
    return request.measured


def write_overlay(path: str, image: np.ndarray, result: dict) -> str:
    """Writes <name>_moon.png next to path: the image with the detected circle and centre (red) and the new crop (green).  Returns its path."""
    rgb = np.clip(image if image.ndim == 3 else np.repeat(image[..., None], 3, axis=2), 0.0, 1.0)
    out = cv2.cvtColor((rgb * 255.0 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR)
    height, width = out.shape[:2]
    thick = max(1, max(width, height) // 600)
    if result["status"] == "ok":
        centre = (int(result["cx"]), int(result["cy"]))
        cv2.circle(out, centre, int(result["radius"] + 0.5), (0, 0, 255), thick)
        cv2.drawMarker(out, centre, (0, 0, 255), cv2.MARKER_CROSS, 8 * thick, thick)
    if "display_crop" in result:
        left, top, right, bottom = result["display_crop"]
        cv2.rectangle(out, (int(left * width), int(top * height)), (int(right * width), int(bottom * height)), (0, 255, 0), thick)
    png = os.path.splitext(path)[0] + "_moon.png"
    cv2.imwrite(png, out)
    return png


def main() -> int:
    """Command line: --request <file> for the plug-in, or image paths to check by eye."""
    parser = argparse.ArgumentParser(description="Find the moon and crop centred on it")
    parser.add_argument("images", nargs="*", help="TIFF files to check (batch mode when several)")
    parser.add_argument("--request", help="request file written by the Lightroom plug-in (plugin/moon_bridge.py)")
    parser.add_argument("--size", type=int, default=121, help="detect_moon grid size in pixels (default 121)")
    parser.add_argument("--workers", type=int, default=1, help="processes measuring photos at the same time (default 1; about 1 GB of memory each)")
    parser.add_argument("--min-confidence", type=float, default=moon_bridge.DEFAULT_MIN_CONFIDENCE,
                        help=f"share of edge points on the circle below which the moon is not found (default {moon_bridge.DEFAULT_MIN_CONFIDENCE})")
    parser.add_argument("--debug", action="store_true", help="also write <name>_moon.png overlays next to the inputs")
    args = parser.parse_args()
    if args.request is not None:
        request = moon_bridge.read_request(args.request)
        moon_bridge.write_result(request.result, run(request))
        return 0
    if not args.images:
        parser.error("give --request or image files")
    photos = [moon_bridge.Photo(path, path, (0.0, 0.0, 1.0, 1.0), "AB") for path in args.images]
    results = measure_all(moon_bridge.Request("", "measure", args.size, photos, args.min_confidence, workers=args.workers))
    add_crops(results, "batch" if len(photos) > 1 else "single")
    for photo, r in zip(photos, results):
        line = f"{photo.image}: {r['status']}  {r['message']}"
        if r["status"] == "ok":
            line += f"  centre ({r['cx']:.2f}, {r['cy']:.2f}) radius {r['radius']:.2f} confidence {r['confidence']:.3f}  crop {r['left']:.4f} {r['top']:.4f} {r['right']:.4f} {r['bottom']:.4f}"
        if args.debug and r["status"] != "error":
            line += f"  -> {write_overlay(photo.image, TiffFile().read(photo.image).array, r)}"
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
