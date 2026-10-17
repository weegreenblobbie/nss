import argparse
import sys
from PyQt6.QtWidgets import QApplication
from nss.color_gui import MainWindow
from plugin import lr_bridge

def main() -> None:
    """
    Main entry point for the Color Grading Explorer application.
    With --lightroom it edits a Lightroom request file and exits 0 after Apply to Lightroom wrote the result, 1 when the window is closed otherwise.
    """
    parser = argparse.ArgumentParser(description="Color Grading Explorer")
    parser.add_argument("--lightroom", metavar="REQUEST", help="request file written by the Lightroom plug-in (plugin/lr_bridge.py)")
    args = parser.parse_args()
    app = QApplication(sys.argv)
    window = MainWindow()
    if args.lightroom is not None:
        window.open_lightroom_request(lr_bridge.read_request(args.lightroom))
    window.show()
    code = app.exec()
    if args.lightroom is not None:
        code = 0 if window.lightroom_applied else 1
    sys.exit(code)

if __name__ == "__main__":
    main()
