"""Focus Stack: correct a picture by painting patches that show other frames of the same scene.

    python focus_stack.py [--mode focus|clean|clean2] target.tif frame1.tif [frame2.tif ...]
    python focus_stack.py --lightroom <request file>          from the Lightroom plug-in (plugin/stack_bridge.py)

focus (the default) repairs blurred regions of a panorama with sharper frames; clean removes people and anything else that
moved, using frames shot from the same spot (clean2 builds its clean plate differently).  With --lightroom the pictures, mode and output come from the request file, the window has
Save to Lightroom and Cancel buttons, and the exit code is 0 (saved), 1 (closed without saving) or 2 (error).
"""
import argparse
import sys

from PyQt6.QtWidgets import QApplication

from nss.focus_stack.gui.main_window import MainWindow
from nss.focus_stack.stack import REFERENCE_DIM
from plugin import stack_bridge


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """The parsed command line; exits with a usage error unless it has a target and frames, or a Lightroom request, and not both."""
    parser = argparse.ArgumentParser(description="Focus Stack: paint patches of other frames over a picture")
    parser.add_argument("--mode", choices=["focus", "clean", "clean2"], default="focus", help="what to stack for; clean2 is clean with a plate made from the closest pair of frames at each pixel (default %(default)s)")
    parser.add_argument("target", nargs="?", help="the picture to repair, e.g. the Lightroom panorama")
    parser.add_argument("frames", nargs="*", help="extra frames covering the same scene")
    parser.add_argument("--work-dim", type=int, default=REFERENCE_DIM, help="long side in pixels of the preview you paint on (default %(default)s)")
    parser.add_argument("--lightroom", metavar="REQUEST", help="request file written by the Lightroom plug-in (plugin/stack_bridge.py)")
    args = parser.parse_args(argv)
    if args.lightroom is not None and args.target is not None:
        parser.error("--lightroom replaces the target and the frames")
    if args.lightroom is None and (args.target is None or not args.frames):
        parser.error("give a target and at least one frame, or --lightroom REQUEST")
    return args


def main() -> None:
    """Opens the window and exits with its code: 1 when a frame cannot be registered to the target, otherwise 0; with --lightroom see plugin/stack_bridge.py."""
    args = parse_args()
    request = None
    if args.lightroom is not None:
        try:
            request = stack_bridge.read_request(args.lightroom)
        except (OSError, ValueError) as error:
            print(f"Cannot read the request: {error}", file=sys.stderr)
            sys.exit(stack_bridge.EXIT_CODES[stack_bridge.ERROR])
    app = QApplication(sys.argv)
    window = MainWindow()
    if request is None:
        window.open(args.target, args.frames, args.work_dim, args.mode)
    else:
        window.use_lightroom(request)
        window.open(request.target, request.frames, request.work_dim, request.mode)
    window.show()
    code = app.exec()
    sys.exit(window.finish_lightroom() if request is not None else (window.exit_code or code))


if __name__ == "__main__":
    main()
