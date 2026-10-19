"""Settings for every test in the repo: Qt runs headless, so the GUI tests in nss/color_gui open no windows."""
import os

os.environ["QT_QPA_PLATFORM"] = "offscreen"
