import sys
from PyQt6.QtWidgets import QApplication
from nss.color_gui import MainWindow

def main() -> None:
    """
    Main entry point for the Color Grading Explorer application.
    """
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
