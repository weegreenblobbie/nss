"""Command line: python -m nss.focus_stack [--mode focus|clean] target.tif frame1.tif [frame2.tif ...] -o out.tif"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from nss.focus_stack.clean import stack_clean
from nss.focus_stack.register import RegistrationError
from nss.focus_stack.stack import REFERENCE_DIM, StackResult, render_full, stack_frames
from nss.image_utils import to_uint8_display
from nss.tiff_file import TiffFile

LABEL_COLORS = np.array(  # BGR: target black, then one colour per frame, repeating
    [(0, 0, 0), (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)], dtype=np.uint8
)


def write_debug_images(folder: Path, out: StackResult) -> None:
    """Writes the target, the label map, each frame's aligned picture and, when there are any, its agreement map and the median as 8-bit PNGs."""
    folder.mkdir(parents=True, exist_ok=True)

    def save_rgb(name: str, image: np.ndarray) -> None:
        """Writes a float RGB image as an 8-bit PNG."""
        cv2.imwrite(str(folder / name), cv2.cvtColor(to_uint8_display(np.clip(image, 0.0, 1.0)), cv2.COLOR_RGB2BGR))

    save_rgb("target.png", out.target)
    cv2.imwrite(str(folder / "labels.png"), LABEL_COLORS[out.labels % len(LABEL_COLORS)])
    for k, frame in enumerate(out.frames, start=1):
        save_rgb(f"aligned_{k}.png", frame.aligned)
        if frame.agreement is not None:
            cv2.imwrite(str(folder / f"agreement_{k}.png"), np.clip((frame.agreement + 1.0) * 127.5, 0, 255).astype(np.uint8))
    if out.median is not None:
        save_rgb("median.png", out.median)


def main(argv: list[str] | None = None) -> int:
    """Runs the command line and returns the exit code: 0 on success, 1 when a frame cannot be registered."""
    parser = argparse.ArgumentParser(
        prog="python -m nss.focus_stack",
        description="Replace regions of a target picture with the same regions of extra frames of the same scene: the sharper ones "
        "(--mode focus) or, to remove people and anything else that moved, the ones behind them (--mode clean, frames from one spot).",
    )
    parser.add_argument("--mode", choices=["focus", "clean"], default="focus", help="what to stack for (default %(default)s)")
    parser.add_argument("target", help="the picture to repair, e.g. the Lightroom panorama")
    parser.add_argument("frames", nargs="+", help="extra frames covering the same scene")
    parser.add_argument("-o", "--output", required=True, help="where to write the result")
    parser.add_argument("--work-dim", type=int, default=REFERENCE_DIM, help="long side in pixels of the working pictures (default %(default)s)")
    parser.add_argument("--full-res", action="store_true", help="write the result at the target's own size instead of the working size")
    parser.add_argument("--tile-rows", type=int, default=1024, help="rows of the full-size picture rendered at a time with --full-res; the picture does not change, the memory used does (default %(default)s)")
    parser.add_argument("--debug-dir", type=Path, help="folder for the target, label map, each frame's aligned picture (and agreement map, or the median) as images")
    args = parser.parse_args(argv)

    target_file = TiffFile().read(args.target)
    frame_arrays = [TiffFile().read(path).array for path in args.frames]
    try:
        out = (stack_clean if args.mode == "clean" else stack_frames)(target_file.array, frame_arrays, args.work_dim)
    except RegistrationError as error:
        print(f"registration failed: {error}", file=sys.stderr)
        return 1

    for k, (path, frame) in enumerate(zip(args.frames, out.frames), start=1):
        print(f"{path}: {frame.inliers} inliers, used for {100.0 * (out.labels == k).mean():.1f}% of the picture")
    if out.median is not None:
        print(f"clean plate (median): used for {100.0 * (out.labels == len(out.frames) + 1).mean():.1f}% of the picture")
    if args.debug_dir is not None:
        write_debug_images(args.debug_dir, out)
    target_file.array = render_full(target_file.array, frame_arrays, out.frames, out.weights, args.tile_rows) if args.full_res else out.result
    target_file.saveas(args.output)
    rows, cols = target_file.array.shape[:2]
    print(f"replaced {100.0 * out.replaced_fraction:.1f}% of the covered area, wrote {cols}x{rows} to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
