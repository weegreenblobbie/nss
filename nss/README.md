# nss - the image tools package

Tools for 16-bit TIFF photographs as numpy arrays: star and moon alignment, moon centering, a Lightroom colour model with its GUI, and an
intentional camera movement (ICM) simulator.  Run everything from the repo root (`import nss` must work); the Lightroom Classic plug-in that
drives several of these tools lives in `plugin/` (see `plugin/README.md`).

## Conventions

* **Image IO:** always `nss.tiff_file.TiffFile`.  `read()` gives float32 0..1 holding the stored values (integer data divided by its full
  range, never stretched) and keeps the embedded ICC profile; `saveas()` writes the same bit depth and profile back, uncompressed.  sRGB is
  never assumed: colour maths reads the working space from the profile (`nss.icc`).
* **Previews:** GUIs work on a downsampled copy (`nss.image_utils.downsample_image`, shown with `to_uint8_display`) and render the
  full-resolution array only when saving.
* **Maths apart from widgets:** the numpy / OpenCV transforms are in plain modules with tests (`*_test.py` next to each module); the PyQt6
  code only gathers settings, runs the transforms (in background threads where they are slow) and shows the result.

## Modules

| Module | What it does |
| --- | --- |
| `tiff_file.py` | 16-bit TIFF reading / writing that keeps the stored values and the ICC profile. |
| `icc.py` | Matrix/TRC ICC profiles: describe, convert to and from linear light. |
| `image_utils.py` | Preview downsampling and the float to 8-bit display mapping. |
| `utils.py` | Moon detection (`detect_moon`), circle fitting, alignment helpers, `timeit`; used by `nss.py`, `moon.py` and `center.py`. |
| `align.py`, `align_moon.py` | Brute-force alignment of star-field / moon frames to a target frame (`nss.py` runs them over many files). |
| `center.py` | Centre the moon or sun in a TIFF (`python -m nss.center`). |
| `squash.cpp` | pybind11 helper for `utils.squash` (built with `build.py`). |
| `color_grading/` | Model of Lightroom's Color Grading panel; see its README. |
| `color_mixer/` | Model of Lightroom's Color Mixer (HSL), table driven. |
| `color_explore/` | The colour GUI's state, colour harmonies and random shifts. |
| `color_gui/` | The Color Grading Explorer (`python color.py`); see its README. |
| `mru.py` | Recently used directories for the colour GUI. |
| `icm_render.py` | ICM maths: path fitting, the per-frame camera motion and the averaged render. |
| `icm.py` | The ICM simulator GUI (`python -m nss.icm`). |

## ICM simulator (`icm.py`, `icm_render.py`)

    python -m nss.icm picture.tif [-o out.tif]
    python -m nss.icm --lightroom <request file>      (the plug-in's "ICM: Intentional Camera Movement", plugin/icm_bridge.py)

1. **Draw** the camera's motion path on the picture.  The stroke becomes a smoothed spline (`fit_path`) stored in fractions of the picture's
   width and height, so it is independent of the window size and of the resolution it is rendered at.
2. **Shape** it: the shutter time and samples per second (or the total frames of a multi-exposure) give the frame count; the speed curve sets
   how far along the path each frame is, the rotation curve turns -45..45 degrees and the zoom curve scales 0.5..1.5 (`sample_motion`).
3. **Preview:** after each change (200 ms debounce) the picture downsampled to the screen size is rendered in a `RenderThread`; a new change
   cancels a running preview.
4. **Save:** the full-size picture is rendered in the background with a progress bar (`render`: every frame turned and scaled about the path's
   mean point, shifted along the path, edges reflected, averaged) and written with the source's bit depth and ICC profile.  In Lightroom mode
   it goes to the request's output, the result file says saved / cancelled / error and the exit code is 0 / 1 / 2.
5. **Reset** drops the path and its preview and flattens the three curves (the exposure settings stay).  Closing the window (Escape or its close button) while a path has
   unsaved changes asks Save (Save to Lightroom) / Discard / Cancel first; Save closes the window once the picture is written.
