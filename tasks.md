# Project Roadmap & Tasks

This document tracks the implementation progress of the Color Grading Explorer. The agent should complete these tasks sequentially. Do not move to a new phase until the current one is functional and tested.

## Phase 1: Basic GUI Scaffold
*Goal: Set up the desktop application shell and layout.*
- [ ] Initialize a basic PyQt6 `QMainWindow`.
- [ ] Create a central widget with a `QGridLayout` (3x3 grid).
- [ ] Create a custom `QFrame` or `QLabel` subclass to act as a clickable image container.
- [ ] Populate the 3x3 grid with these containers using placeholder colors or text.
- [ ] Add basic "Back" and "Forward" buttons to a top or bottom toolbar (disabled by default).
- [ ] Add a dropdown/combobox to select the Harmony Mode (Monochromatic, Analogous, Complementary).

## Phase 2: Core Image Engine & I/O
*Goal: Successfully load a 16-bit TIFF and display it in the UI.*
- [ ] Integrate the existing TIFF reading function from the `nss` repository.
- [ ] Write a pipeline function to convert the loaded 16-bit array to `float32` (scaled 0.0 to 1.0).
- [ ] Write a utility function to convert the `float32` array back to an 8-bit `QImage` or `QPixmap` for the PyQt6 canvas.
- [ ] Wire up a "File -> Open" menu action to load a TIFF and display it in the center grid cell.

## Phase 3: Color Science & Mutation Math
*Goal: Implement the NumPy/OpenCV math for color grading and mutation generation.*
- [ ] Write a function to convert the `float32` image between BGR and HSL/LAB color spaces using OpenCV.
- [ ] Implement NumPy luminosity masking to isolate Shadows (e.g., L < 0.3), Midtones, and Highlights (e.g., L > 0.7).
- [ ] Define the `StateNode` dictionary structure (e.g., tracking hue shifts and mode).
- [ ] Implement the `Monochromatic` mutation generator (randomizes S and L, locks H).
- [ ] Implement the `Analogous` mutation generator (shifts H by a small random degree, alters S and L).
- [ ] Implement the `Complementary` mutation generator (forces highlight H to target, shadow H to opposite).

## Phase 4: Interactivity & State Management
*Goal: Connect the math to the UI and enable exploration.*
- [ ] Implement the history list to store `StateNode` dictionaries (the Undo/Redo stack).
- [ ] Wire up the click event on the 8 outer image containers to promote that variation to the center.
- [ ] On click, push the new center parameters to the history stack and generate 8 new random `StateNode` mutations for the outer grid.
- [ ] Apply the parameters to the original `float32` master image to render the 9 variations to the screen.
- [ ] Enable and wire up the Back and Forward buttons to traverse the history stack and re-render the grid.