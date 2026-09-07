# Project Roadmap & Tasks

This document tracks the implementation progress of the Color Grading Explorer. The agent should complete these tasks sequentially. Do not move to a new phase until the current one is functional and (where applicable) tested via `pytest`.

## Phase 1: Basic GUI Scaffold
*Goal: Set up the desktop application shell and layout.*
- [x] Initialize a basic PyQt6 `QMainWindow` in `nss/color_gui.py`.
- [x] Create a central widget with a `QGridLayout` (3x3 grid).
- [x] Create a custom `QFrame` or `QLabel` subclass to act as a clickable image container.
- [x] Populate the 3x3 grid with these containers using placeholder colors or text.
- [x] Add basic "Back" and "Forward" buttons to a top or bottom toolbar (disabled by default).
- [x] Add a dropdown/combobox to select the Harmony Mode (Monochromatic, Analogous, Complementary).
*(Note: No unit tests for Phase 1 as it is strictly GUI layout).*

## Phase 2: Core Image Engine & I/O
*Goal: Successfully load a 16-bit TIFF and process it as float32.*
- [x] Integrate the existing TIFF reading function from the `nss` repository.
- [x] Write a pipeline utility function in a separate module (e.g., `nss/image_utils.py`) to convert the loaded 16-bit array to `float32` (scaled 0.0 to 1.0).
- [x] Write a utility function to map the `float32` array back to an 8-bit `uint8` array format suitable for display.
- [x] Wire up a "File -> Open" menu action in the GUI to load a TIFF and display it in the center grid cell.
- [x] **TEST:** Write `tests/test_io.py` to verify the 16-bit array scaling math and conversions. **Do NOT import or test any PyQt6 objects in this test.** Run `pytest` to confirm.

## Phase 3: Color Science & Mutation Math
*Goal: Implement the NumPy/OpenCV math for color grading and mutation generation.*
- [x] Write a function to convert the `float32` image between BGR and HLS/LAB color spaces using OpenCV.
- [x] Implement NumPy luminosity masking utilities to isolate Shadows (e.g., L < 0.3), Midtones, and Highlights (e.g., L > 0.7).
- [x] Define the `StateNode` dictionary structure (e.g., tracking hue shifts and mode).
- [x] Implement the `Monochromatic` mutation generator logic.
- [x] Implement the `Analogous` mutation generator logic.
- [x] Implement the `Complementary` mutation generator logic.
- [x] **TEST:** Write `tests/test_math.py` to assert that the mutation and masking functions return mathematically valid arrays within expected bounds. **Do NOT import PyQt6.** Run `pytest` to confirm.

## Phase 4: Interactivity & State Management
*Goal: Connect the math to the UI and enable exploration.*
- [ ] Implement the history list logic to store `StateNode` dictionaries (the Undo/Redo stack).
- [ ] Wire up the click event on the 8 outer image containers to promote that variation to the center.
- [ ] On click, push the new center parameters to the history stack and generate 8 new random mutations.
- [ ] Apply the parameters to the original `float32` master image to render the 9 variations to the screen.
- [ ] Enable and wire up the Back and Forward buttons to traverse the history stack and re-render the grid.
- [ ] **TEST:** Write `tests/test_state.py` to verify pushing, undoing, and redoing `StateNode` dictionaries in pure Python lists works correctly. Run `pytest` to confirm.
