# nss.focus_stack - repair a picture from other frames of the same scene

A panorama stitched from several shots can have regions out of focus.  An extra frame of the whole scene may be sharper there.  This
package aligns the extra frame to the panorama (the *target*) and blends in only the regions where the frame is clearly sharper.

There are two modes, chosen with `--mode` on the command line and in the window: **focus** (the default), described first, and **clean**
(see "Clean mode" below), which removes people and anything else that moved from a picture using frames shot from the same spot.

    python -m nss.focus_stack [--mode focus|clean] target.tif frame1.tif [frame2.tif ...] -o out.tif [--work-dim 3000] [--full-res] [--debug-dir folder]

The alignment, brightness gain and mask are always found on copies `--work-dim` px on the long side (default 3000).  By default the result is
written at that working size, which takes seconds and is for judging.  `--full-res` applies the same alignment, gain and mask to the full-size
pictures (flow, gain and mask are smooth, so they are upsampled; the frame is warped once with Lanczos) and writes the target's own size.  For
an 8165 x 11358 target it took about 20 s and 8.4 GiB of memory, because the whole picture is processed at once.  Either way the file has the
target's bit depth and ICC profile.  Exit code 0 is success, 1 means the frame could not be registered (nothing is written).  One line is
printed per frame with its match count and the share of the picture it is used for, then one with the share of the covered area replaced.
The first frame that cannot be registered aborts the whole run.  `--debug-dir` writes `target.png`, `labels.png` (black target, then red, green,
blue ... per frame) and, per frame, `aligned_N.png` and `agreement_N.png` (working size) for judging the result.

`--full-res` renders `--tile-rows` rows of the target at a time (default 1024).  Each pixel is computed on its own, so the strip height changes
memory and not the picture: on the 8165 x 11358 test pair with two frames, 1024-row strips and one strip covering everything gave identical files.
The peak was 9.3 GiB with strips against 11.0 GiB without, and about 45 s either way; most of the memory is the pictures and the alignment
stage, not the render.

## Pipeline (`stack.py` chains these; each module has a `*_test.py`)

| Step | Module | What it does |
| --- | --- | --- |
| 1 | `register.py` | SIFT matches on downsampled luma, ratio test, RANSAC homography of frame onto target. Raises `RegistrationError` with fewer than 30 inliers. |
| 2 | `flow.py` | DIS dense flow on locally normalised luma (immune to gain and lighting) for what the homography leaves (lens, projection, parallax); the homography and flow are applied in one Lanczos resample. |
| 3 | `blend.py` `source_coverage`, `match_gain` | Where the frame has pixels, and a smooth per-channel brightness match to the target. |
| 4 | `sharpness.py` | Mean absolute Laplacian of the lightly blurred luma, averaged over a Gaussian window. |
| 5 | `blend.py` `agreement_map` | Local normalised correlation of low-passed luma: near 1 where both frames show the same structure (even if one is blurred), near 0 where the frame shows content the target lacks, such as the side of an object seen from another angle. |
| 6 | `labels.py` | A label per pixel (0 target, k frame k): the sharpest frame that agrees and beats the target by more than `BIAS_STOPS`; holes and thin strips are tidied per label. The labels are feathered into weights that add up to 1. |
| 7 | `blend.py` `composite` | The weighted sum of the target and the aligned frames. |

## Choices worth knowing

* **Working resolution matters.** At 1600 px the blur on the test pair was too fine to see; around 3000 px the sharpness ratio separated hand
  labelled regions best.  Above the frame's own resolution it got worse.  All pixel sizes in `stack.py` are defined at 3000 px and scaled.
* **The frames have different native resolution.** On the test pair one frame pixel covers about 1.9 target pixels, so patches from the frame are upsampled
  and cannot be sharper than the frame itself.  Compare sharpness at the coarser frame's scale, never at the finer one.
* **Do not compare sharpness at full size.** There the target scored higher than the visibly sharper result, most likely because its
  pixel-level grain counts as detail while the upsampled frame has none.  Judge by eye at 100%, or measure at the working size.
* **Sharpness is not contrast.** A blurred edge has the same peak-to-valley as a sharp one; the Laplacian measures how steeply it changes.
* **Agreement gate.** Without it the sharpness test replaced regions along object edges with misaligned content from the frame.

## The window (`python focus_stack.py [--mode focus|clean] target.tif frame1.tif [frame2.tif ...]`)

The automatic merge is the starting point; the window lets you correct it by painting **patches**, like Lightroom's Remove tool.  A patch is a
painted region plus the layer it shows (the target or one frame).  Patches stay: select one by clicking its pin, change it, or delete it.
All the maths is in `edit.py` (`EditState`); `gui/canvas.py` turns mouse and keys into calls on it and `gui/main_window.py` adds the menus and the
threads (aligning on start, saving at the end).  You paint on the working-size preview (`--work-dim`, default 3000); Ctrl+S renders the full size.

| Input | Action |
| --- | --- |
| left-drag | paint: starts a patch showing the next layer after the one under the first dab, or adds to the active patch |
| Shift- or Alt-drag | erase from the active patch (those pixels return to the automatic choice; an empty patch is deleted) |
| hover a pin, then click it | select that patch, even after Esc; the pin grows and the cursor becomes a pointing hand (the brush circle hides); then Delete removes it |
| Tab, Shift+Tab | cycle the active patch through the layers: the target, the frames in the order given, then the median in clean mode, and round again |
| `[`, `]` | brush size |
| Esc | deselect, so the next stroke starts a new patch |
| Delete | delete the active patch |
| Ctrl+Z, Ctrl+Shift+Z | undo, redo (a stroke is one step; cycling one patch repeatedly is one step) |
| hold T | show the original target |
| L | tint the picture by which layer each region uses; the active patch is left untinted so you see the picture under it |
| two-finger scroll or wheel | pan (a touchpad's pixel delta, or 60 pixels a mouse notch; sideways scrolling pans sideways) |
| Ctrl+scroll or a pinch | zoom about the cursor or the fingers |
| Space, or the Pan button | toggle pan mode: a left-drag moves the picture (open hand; closed while dragging) and nothing is painted or selected; Space or Esc leaves it |
| middle-drag | pan, without pan mode |
| toolbar or Ctrl+0, 1, 2, 3 | Fit, 100%, 200%, 400% (Ctrl+= and Ctrl+- step between 12.5% and 3200%, about the middle of the view) |
| Ctrl+S | save the full-size picture with the target's bit depth and ICC profile |

**Layers panel** (right): the target first, then one row per frame with a thumbnail and its name (and, in clean mode, a last row "Clean plate (median)").  The active patch's layer is
highlighted; clicking a row gives the active patch that layer, like Tab (with no patch active it only shows a hint).
**Ctrl+click a row to look at that layer**: the canvas shows its whole picture (aligned and brightness matched, so the frames can be compared),
its row goes bold and the status line says which layer; Ctrl+click it again or press Esc to go back.  Nothing can be painted while a layer is
viewed.  **Right-click a frame row and choose "Set ... as the target"** to correct that picture instead: everything is aligned again in the
background (about 20 seconds for five frames), the old target takes the frame's place so the Tab order stays put, and the patches painted so far
are discarded (it asks first when there are any).  Use this to find the frame that is clear where you need it and make it the target.

**Zoom is relative to the preview** you paint on (3000 px on the long side by default, not the full size), so "100%" shows one preview pixel per
screen pixel.  The status line gives the share of the full size as well, for example "zoom 200% (53% of full size)".  A larger `--work-dim` gives a
sharper picture when zoomed in, at the cost of alignment time and memory.

* **A hand-painted patch may use a frame where the automatic agreement check said no** (you judged it by eye), but never where the frame has
  no pixels.  Where patches overlap, the newer is on top.
* **A patch is opaque inside its outline and ends at the brush circle.** Its soft edge lies outside the outline and is narrow, `PATCH_FEATHER`
  (0.15) of the automatic regions' feather: the patch grows by two of those narrow sigmas (about 3 px at the 3000 px working size) and only that
  is blurred, so even a small patch hides what is under it completely.  (Blurring the painted shape itself left the middle of a small patch at only
  about 50% strength, so the person under it showed through as a ghost; the automatic regions' own 22 px growth made a painted patch much
  larger than its circle.)
* **Only the changed region is recomputed** after an edit (`EditState.result_region`, padded by the feathering's reach so it equals the full
  render), which is why painting stays interactive (about 10 ms per brush move at 3000 px).
* **Not saved between sessions:** patches live in memory.  Quitting with patches that were not saved asks first.

## Clean mode (`--mode clean`): removing people

For pictures of one scene shot from the same spot, with people or traffic moving through it.  The target is the first picture; the other
frames show what is behind the people in it.  Nothing is aligned by flow here, because a dense flow is dragged around by whoever is in one
frame and not another and bends the background into their outline; a homography is accurate enough for same-spot handheld shots.

1. **Align** each frame to the target by homography (`clean.align_frame_homography`).
2. **The median layer:** the per-pixel, per-channel median of the target and the frames, over the frames that cover each pixel.  It shows the empty
   scene wherever a strict majority of the frames do (`blend.masked_median`).  It is the last layer, after the frames.
3. **Brightness:** each frame is matched to the target in two passes.  The first is one gain per channel, the *median* of the per-pixel ratios
   (`clean.global_gain`), which a big object cannot skew as long as it covers under half the picture.  (A ratio smoothed over a local window
   was used here at first: a car much larger than the window made it crush every frame to a black oval there, and since the median then agreed
   with the dark car, the car was never seen as foreground.)  The second pass is local and ignores the foreground of both pictures
   (`clean.correct_gains`).
4. **Trust:** the median is *trusted* at a pixel when a strict majority of the covering layers agree with it (`trusted_median`).  The target is
   only called foreground where the median is trusted.
5. **Blobs:** the foreground of the target is grouped into blobs (`foreground_blobs`).  A blob is left alone unless the median is trusted over at
   least 70% of the pixels where the target has a person: if most frames disagree about what is behind the people, they are people who stay.
6. **Automatic labels** (`auto_labels`): a blob goes whole to the covering frame closest to the median, if it is clean over the blob and does not
   show a person of its own at the same place as the target's.  Where that frame has a person of its own, the pixel takes the median (if trusted).
   People who stand in most frames cannot be removed and are left exactly as shot; those are for your patches.

In the window, a new patch starts on the layer `suggest_layer` picks for the painted spot (the covering frame closest to the median there, if it
is clean, otherwise the median) instead of the next layer; Tab still cycles in plain order.  The median is rendered at full size strip by strip
from that strip's own warped frames, and only for strips where some pixel uses it; a frame with no weight in a strip is not read.

The constants (`clean.py`) were tuned on one scene of five frames and are the first thing to retune.  Use 3 or more extra frames; 5 or more is better.

## Clean mode 2 (`--mode clean2`): a plate from the closest pairs

Same inputs as clean mode, working size only.  It differs in how the clean plate is made and how a blob is filled.

1. **Align** and match brightness as in clean mode.
2. **Pairwise plate** (`pairwise_plate`): at each pixel the mean of the two covering layers that are closest to each other (distance smoothed by
   `DEVIATION_SIGMA`).  Two layers that match are most likely both the background, so people who walk through are removed even when they
   are in most of the frames.  It is wrong where a person is at the same place in two layers.
3. **Check against the median** (`choose_plate`): the pairwise plate and the median fail in different places, so they mostly agree.  Each
   connected area where they differ by more than `PLATE_DIFF` takes whichever fits the calm pixels around it better (`ring_prior`, the
   plates' agreeing pixels extended smoothly into the area).  The plate is in `StackResult.median` and is the last layer.
4. **Blobs** (`foreground_blobs`): where the target is far from the plate, with the thresholds of clean mode.
5. **Labels** (`plate_labels`): a blob where the target's mean distance from the plate is already under `CLEAN_MEAN` is left as shot: it is people
   who stay, and the plate is made of them (the pair of frames that matched is the same group), so replacing them only corrupts them.  Any
   other blob goes whole to the covering frame closest to the plate, if it is clean over the blob; where that frame has a person of its
   own the pixel takes the plate.  A blob with no clean frame takes the plate whole.  People left as shot are for the patch tool.

In the window, `python focus_stack.py --mode clean2 target.tif frame1.tif ...` starts from the same stack, and the last layer is listed as "Clean plate"
(not "(median)") because it is not a median.  The plate is chosen at the working size but built at the full size: at each pixel it is
either the mean of two particular layers or the median, and `StackResult.plate_mix` records how much of each layer (and of the median) that is.
A full-size render (Ctrl+S, or `--full-res`) applies the mix, upsampled, to each strip's own warped full-size frames, so the plate is as sharp as
the frames; a frame the mix does not use in a strip is not read there.  Which pair wins is still decided at the working size, so the choice
changes smoothly and not with the strip height.  Setting a frame as the target restacks with clean2 again.
