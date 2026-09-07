# Color Grading Explorer - System Architecture

## Overview
A desktop application that loads 16-bit TIFF images and provides a mutation-based color grading UI. The user explores color harmony adjustments (Monochromatic, Analogous, Complementary) through a 3x3 interactive image grid. 

## Tech Stack
*   **GUI Framework:** PyQt6 (Native desktop application)
*   **Image Processing:** NumPy and OpenCV (`cv2` for color space conversions only)
*   **Image I/O:** The existing TIFF reading/writing modules from the `nss` repository.
*   **Language:** Python 3.11+

## 1. The Data Pipeline (Crucial)
Preserving the 16-bit dynamic range is the highest priority. The pipeline must follow these exact stages:
1.  **Ingest:** Load and save 16-bit TIFFs using the existing TIFF I/O utilities provided in the `nss.utils.TiffFile` class (do NOT use standard OpenCV `imread`/`imwrite`). 
2.  **Math Space:** Ensure the ingested array is cast to `float32` and scaled to a `0.0` to `1.0` range. All color math happens in this floating-point space.
3.  **Color Space:** Convert BGR/RGB to HLS or LAB for grading operations using OpenCV.
4.  **Display:** To render the 9 variations in the PyQt UI, map the `float32` arrays down to 8-bit `uint8` (0-255) and format for the PyQt canvas. *Never overwrite the `float32` master data.*

## 2. State Management & Navigation
To maintain low memory footprint and high performance, the system uses **Parameter-Based State**.
*   **Do not store image arrays in the history stack.** 
*   A "State Node" is a dictionary of the adjustments applied to reach that state (e.g., `{hue_shift: 10, mode: 'analogous', shadow_tint: ...}`).
*   **Undo/Redo:** The Forward and Back buttons navigate a list of these parameter dictionaries. When a state becomes active, the UI recalculates the image from the master `float32` original using those parameters.

## 3. UI Layout
*   **The Grid:** A 3x3 layout. The center image is the current active state.
*   **Interaction:** Clicking one of the 8 surrounding images promotes it to the center. The system pushes the new parameters to the history stack and generates 8 new random mutations around it.

## 4. Color Grading Logic
*   **Luminosity Masks:** Use NumPy array masking to isolate tonal ranges (e.g., Shadows: Lightness < 0.3, Highlights: Lightness > 0.7).
*   **Harmony Modes:** 
    *   *Monochromatic:* Lock hue; mutate only saturation and lightness.
    *   *Analogous:* Mutate hue within a narrow adjacent band (e.g., ±30 degrees).
    *   *Complementary:* Force highlight hues to a target, and shadow hues to target + 180 degrees.

## 5. File Structure & Organization
To maintain a clean repository, all code generation must adhere to the following file structure. Do not create files outside of this established map unless explicitly requested.

*   `color.py` 
    *   **Location:** Root directory.
    *   **Purpose:** The main entry point of the application. It should be as minimal as possible, responsible only for initializing the PyQt6 `QApplication`, instantiating the main window from `color_gui.py`, and executing the event loop.
*   `nss/color_gui.py`
    *   **Location:** Inside the `nss/` directory.
    *   **Purpose:** Contains the core PyQt6 application logic, the `QMainWindow` class, and the grid layout UI code.
*   `nss/` (Existing Directory)
    *   **Purpose:** Contains the existing TIFF reading/writing modules which `color_gui.py` will import and utilize.

## 6. Environment & Dependencies
*   **Virtual Environment:** Assume the application will be run on a Windows host inside a standard Python virtual environment (`venv`).
*   **Requirements:** Maintain all third-party dependencies strictly within a `requirements.txt` file in the root directory. 
*   **Agent Constraint:** Do not attempt to execute `pip install` or run Python scripts directly. When introducing a new dependency (like PyQt6 or OpenCV), update the `requirements.txt` file and instruct the user to install it.

## 7. Testing & Quality Assurance
*   **Framework:** Use `pytest`. All tests must reside in a `_tests.py` file next to the code module it is testing.
*   **Decoupling:** Strictly decouple the core math and state logic (NumPy/OpenCV) from the PyQt6 GUI. The GUI should only act as a view layer.
*   **Execution:** The agent is running in a Linux container equipped with Python, `pytest`, `pytest-qt`, and `pytest-xvfb`. The agent is highly encouraged to run `pytest` to verify its own logic as it builds.
