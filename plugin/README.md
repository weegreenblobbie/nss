# plugin - Lightroom Classic plug-in `nss` (nss.Color, moon centering, ICM and Stack)

The plug-in **nss** has four parts.  **Color** opens the selected Lightroom
Classic photo in the nss.Color GUI (`color.py`) and writes its Color Mixer and
Color Grading values back to the photo.  **Moon** crops photos so the moon is
centred (see Center the moon).  **ICM** simulates intentional camera movement and **Stack** removes people from, or
repairs the focus of, a series of photos of one scene (see Stack).

**Only tested on Windows** (Lightroom Classic 15.6).  The launcher is a Windows
batch file; other platforms would need their own launcher script (see Configure
below) and are untested.

## Files

* `nss.lrdevplugin/` - the plug-in folder you add to Lightroom (`Info.lua`,
  `Config.lua`, `Common.lua`, `EditWithNss.lua`, `DumpSettings.lua`, `Keys.lua`,
  `CenterMoon.lua`, `VerifyMoon.lua`, `Moon.lua`, `ICM.lua`, `Stack.lua`, `StackClean.lua`, `StackFocus.lua`).
* `nss_color.bat` - the launcher Lightroom runs: starts
  `color.py --lightroom <request file>` with the repo's `venv` Python.
* `nss_moon.bat` - the moon launcher: starts `moon.py --request <request file>`.
* `nss_icm.bat` - the ICM launcher: starts `python -m nss.icm --lightroom <request file>`.
* `nss_stack.bat` - the Stack launcher: starts `focus_stack.py --lightroom <request file>`.
* `lr_bridge.py`, `moon_bridge.py`, `icm_bridge.py`, `stack_bridge.py` - read the request file and
  write the result file on the Python side (formats in their docstrings);
  `handover.py` reads the `key=value` files.

## Install

1. **Python environment.**  Create the `venv` at the repo root and install the
   dependencies:

       cd C:\path\to\nss
       python -m venv venv
       venv\Scripts\python -m pip install -r requirements.txt

   Check that `venv\Scripts\python color.py` opens the GUI.

2. **Configure the paths.**  These files contain the author's repo path
   (`C:\users\weegr\code\nss`); change it to your clone:
   * `nss_color.bat`, `nss_moon.bat`, `nss_icm.bat` and `nss_stack.bat`: the `cd /d ...` line and
     the path to `venv\scripts\python.exe`.
   * `nss.lrdevplugin\Config.lua`: `launcher`, `moonLauncher`, `icmLauncher` and `stackLauncher`,
     the full paths of `nss_color.bat`, `nss_moon.bat`, `nss_icm.bat` and `nss_stack.bat`.

3. **Add the plug-in to Lightroom Classic.**  File > Plug-in Manager > Add,
   select the `plugin\nss.lrdevplugin` folder, and check that `nss` is
   listed as installed and running.

## Use

1. Select one photo and choose Library > Plug-in Extras > **Color: Edit...**.
2. The plug-in exports a 16-bit Adobe RGB TIFF with the Color Mixer and Color
   Grading set to neutral (everything else baked in) into Lightroom's temp
   folder (`nss_color`), then puts the photo's values back.  The GUI opens with
   its sliders at the photo's values.
3. Edit, then press **Apply to Lightroom** in the GUI's status bar.  The values
   are applied to the photo as one undo step, `nss.Color`.  Closing the window
   without Apply leaves the photo unchanged.

## Intentional camera movement

1. Select a photo and choose Library > Plug-in Extras > **ICM: Intentional Camera Movement**
   (with several selected, only the active one is opened).
2. The plug-in exports it as a 16-bit Adobe RGB TIFF with all its edits and the
   crop baked in (Lightroom's temp folder, `nss_icm`) and opens it in
   `nss.icm`.
3. Draw the motion path, set the curves, and press **Save to Lightroom**: the
   ICM picture is rendered at full size and written as `<name>_icm.tif` next to
   the original (`_icm_2`, `_icm_3` ... when taken; never overwritten), then
   imported stacked above the original and selected.  When the original's
   folder is not writable it goes to `Config.lua`'s `icmFallbackFolder`
   (default `Documents\nss_icm`), unstacked, and a dialog says where.  Closing
   the window with an unsaved motion asks whether to save to Lightroom first;
   closing it without saving changes nothing.

## Stack: Clean and Stack: Focus

Library > Plug-in Extras > **Stack: Clean (remove people)...** or **Stack: Focus...**, with two or more photos selected (the tool is described in
`nss/focus_stack/README.md`; clean needs photos shot from one spot).

1. The **active photo starts as the target**, the picture that is corrected: the one Lightroom highlights as the most selected, or the first selected
   photo when Lightroom reports none.  The other selected photos are the frames, in the order Lightroom lists them (the filmstrip order, which
   is the Tab order in the window).  **You choose the real target in the window**: Ctrl+click a layer in the Layers panel to look at it, then
   right-click it and choose "Set ... as the target" (about 20 seconds to align again).  The trace file (`Documents\nss_color_trace.txt`) names
   the photo the plug-in started with.  At most 21 photos (the target and 20 frames, `stackMaxFrames` in `Config.lua`): work in batches beyond
   that.  Select the photos first: with no selection Lightroom may report the whole folder.
2. The plug-in exports every selected photo as a 16-bit Adobe RGB TIFF with all its edits and the crop baked in (Lightroom's temp folder,
   `nss_stack`; about 150 to 300 MB each at full size, so up to a few GB while the window is open) and opens them in `focus_stack.py`.
   The progress bar can cancel the export.
3. Work in the window (paint patches, Tab through the layers), then press **Save to Lightroom** (Ctrl+S): the picture is rendered at full
   size and written as `<name>_clean.tif` (`_focus.tif` for Stack: Focus), where `<name>` is the photo that is the target *when you save*,
   next to that photo's original (`_clean_2`, `_clean_3` ... when taken; never overwritten), then imported stacked above that photo and
   selected.  The plug-in prepares such a path for every photo it exports, and the window says which one it used.  When the folder of the
   target is not writable the picture goes to `Config.lua`'s `stackFallbackFolder` (default `Documents\nss_stack`), unstacked, and a dialog
   says where.
4. **Cancel** (the toolbar button) or closing the window leaves everything unchanged; with patches painted it asks first.  Esc keeps its meaning
   inside the window (deselect the patch, leave pan mode).  If the frames cannot be aligned or the picture cannot be written, a dialog says why.
5. The exported TIFFs are deleted in every case.

## Center the moon

* Library > Plug-in Extras > **Moon: Center** crops the selected
  photos so the moon's centre is the crop's centre.  The photos are exported at
  full size and full frame, `moon.py` finds the moon
  (`nss.utils.detect_moon`) and the new crop is applied as one undo step
  per photo.  One photo keeps its crop size if it fits, else gets the largest
  crop of its aspect; several photos get one common crop size in pixels (the
  first photo's aspect) so they line up when stacked.  A crop angle is kept:
  the crop is turned about its centre and sized so its corners stay inside the
  photo.  No dialog: the offset corrected per photo and the skipped ones with
  the reason are written to `Documents\nss_color_trace.txt`, which each run of
  Moon: Center starts afresh.
* **Moon: Verify centering** exports the photos with their current
  crop and shows how far the moon's centre is from the crop's centre (under
  1 px when centering worked).
* Settings: File > Plug-in Manager > nss, section **Moon**:
  * **Photos per chunk** (default 16): both menu items export and measure this
    many photos at a time and delete each chunk's TIFFs once measured, so
    the temp folder holds one chunk at most (about 270 MB per photo).
  * **Worker processes** (default 8): photos measured at the same time, each
    in its own Python process of about 1 GB of memory.

  The defaults are in `Config.lua`.
* The progress bar covers the export, the measuring and the applying; cancel
  stops after the current photo.  Canceled before the crops are applied,
  Moon: Center changes no photo; while they are applied, the photos done so far
  keep their new crop.
* `python moon.py a.tif b.tif --debug` prints the findings and writes
  `a_moon.png` overlays (circle, centre, crop) to check the detection by eye.

## Troubleshooting

* `Documents\nss_color_bat.txt`, `Documents\nss_moon_bat.txt`,
  `Documents\nss_icm_bat.txt`, `Documents\nss_stack_bat.txt` - what the launchers printed, and the exit code.
* `Documents\nss_color_trace.txt` - the plug-in's steps and Moon: Center's
  results (export, command run, result); Moon: Center empties it first, the
  other menu items append.
* "Unknown develop settings": your Lightroom names a setting differently.  Run
  Library > Plug-in Extras > **Color: Write develop settings to the log**,
  which writes every setting of the selected photo to
  `Documents\nss_develop_settings.txt`, and correct the names in
  `nss.lrdevplugin\Keys.lua` (and `WHEEL_KEYS` / `MIXER` in `lr_bridge.py`).
* If Lightroom stops between the neutral export and the restore, the photo's
  original values are in `backup.txt` in the same temp folder.
