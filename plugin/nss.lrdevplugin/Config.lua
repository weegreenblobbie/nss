-- The one place to edit.  `launcher` is run as:  launcher "<request file>"
-- and must start the nss color GUI on that request (see nss_color.bat next to this folder), wait until the window is closed,
-- and exit with code 0 after the result file was written (any other code or no result file = leave the photo alone).
local LrPathUtils = import 'LrPathUtils'

return {
  launcher = [[C:\users\weegr\code\nss\plugin\nss_color.bat]],
  workFolderName = 'nss_color',     -- inside Lightroom's temp folder
  -- Moon: Center / Moon: Verify centering: `moonLauncher` is run as  moonLauncher "<request file>"  (see nss_moon.bat) and exits 0 once the
  -- result file is written.
  moonLauncher = [[C:\users\weegr\code\nss\plugin\nss_moon.bat]],
  moonWorkFolderName = 'nss_moon',  -- inside Lightroom's temp folder
  -- defaults of the Plug-in Manager's Moon settings (Settings.lua)
  moonChunkSize = 16,               -- photos exported and measured at a time; each full-size TIFF of an 8256 x 5504 photo is about 270 MB
  moonWorkers = 8,                  -- photos measured at the same time, each in its own process of about 1 GB
  moonSize = 121,                   -- detect_moon grid size in pixels of the full-size export
  moonMinConfidence = 0.9,          -- share of the edge points on the fitted circle below which a photo is skipped (moon not found)
  verifyTolerance = 1.0,            -- pixels; verify flags a moon centre further than this from the crop centre
  -- ICM: `icmLauncher` is run as  icmLauncher "<request file>"  (see nss_icm.bat) and starts python -m nss.icm --lightroom on it;
  -- the result file says saved, cancelled or error (plugin/icm_bridge.py).
  icmLauncher = [[C:\users\weegr\code\nss\plugin\nss_icm.bat]],
  icmWorkFolderName = 'nss_icm',    -- inside Lightroom's temp folder
  -- where the ICM picture goes when the original photo's folder is not writable (normally it goes next to the original)
  icmFallbackFolder = LrPathUtils.child(LrPathUtils.getStandardFilePath('documents'), 'nss_icm'),
}
