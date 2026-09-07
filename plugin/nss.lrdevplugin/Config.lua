-- The one place to edit.  `launcher` is run as:  launcher "<request file>"
-- and must start the nss color GUI on that request (see nss_color.bat next to this folder), wait until the window is closed,
-- and exit with code 0 after the result file was written (any other code or no result file = leave the photo alone).
return {
  launcher = [[C:\users\weegr\code\nss\plugin\nss_color.bat]],
  workFolderName = 'nss_color',     -- inside Lightroom's temp folder
}
