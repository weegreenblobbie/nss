# plugin - Lightroom Classic plug-in for nss.Color

Opens the selected Lightroom Classic photo in the nss.Color GUI (`color.py`)
and writes the GUI's Color Mixer and Color Grading values back to the photo.

**Only tested on Windows** (Lightroom Classic 15.6).  The launcher is a Windows
batch file; other platforms would need their own launcher script (see Configure
below) and are untested.

## Files

* `nss.lrdevplugin/` - the plug-in folder you add to Lightroom (`Info.lua`,
  `Config.lua`, `EditWithNss.lua`, `DumpSettings.lua`, `Keys.lua`).
* `nss_color.bat` - the launcher Lightroom runs: starts
  `color.py --lightroom <request file>` with the repo's `venv` Python.
* `lr_bridge.py` - reads the request file and writes the result file on the
  Python side (format in its docstring).

## Install

1. **Python environment.**  Create the `venv` at the repo root and install the
   dependencies:

       cd C:\path\to\nss
       python -m venv venv
       venv\Scripts\python -m pip install -r requirements.txt

   Check that `venv\Scripts\python color.py` opens the GUI.

2. **Configure the paths.**  Both files contain the author's repo path
   (`C:\users\weegr\code\nss`); change it to your clone:
   * `nss_color.bat`: the `cd /d ...` line and the path to
     `venv\scripts\python.exe`.
   * `nss.lrdevplugin\Config.lua`: `launcher`, the full path of
     `nss_color.bat`.

3. **Add the plug-in to Lightroom Classic.**  File > Plug-in Manager > Add,
   select the `plugin\nss.lrdevplugin` folder, and check that `nss.Color` is
   listed as installed and running.

## Use

1. Select one photo and choose Library > Plug-in Extras > **Edit with
   nss.Color**.
2. The plug-in exports a 16-bit Adobe RGB TIFF with the Color Mixer and Color
   Grading set to neutral (everything else baked in) into Lightroom's temp
   folder (`nss_color`), then puts the photo's values back.  The GUI opens with
   its sliders at the photo's values.
3. Edit, then press **Apply to Lightroom** in the GUI's status bar.  The values
   are applied to the photo as one undo step, `nss.Color`.  Closing the window
   without Apply leaves the photo unchanged.

## Troubleshooting

* `Documents\nss_color_bat.txt` - what the launcher and the GUI printed, and
  the exit code.
* `Documents\nss_color_trace.txt` - the plug-in's steps (export, command run,
  result).
* "Unknown develop settings": your Lightroom names a setting differently.  Run
  Library > Plug-in Extras > **nss.Color: write develop settings to the log**,
  which writes every setting of the selected photo to
  `Documents\nss_develop_settings.txt`, and correct the names in
  `nss.lrdevplugin\Keys.lua` (and `WHEEL_KEYS` / `MIXER` in `lr_bridge.py`).
* If Lightroom stops between the neutral export and the restore, the photo's
  original values are in `backup.txt` in the same temp folder.
