-- Library > Plug-in Extras > nss > ICM
--   1. export the target photo (the active one when several are selected) as a 16-bit Adobe RGB TIFF, every edit and the crop baked in
--   2. pick the output <name>_icm.tif next to the original (then _icm_2, _icm_3 ...; never overwrites), or in Config.icmFallbackFolder
--      when the original's folder is not writable
--   3. write the request file (plugin/icm_bridge.py), run Config.icmLauncher (python -m nss.icm --lightroom) and wait for it
--   4. saved: import the output stacked above the original and select it; cancelled: nothing; error or no result: a dialog.
--      The exported TIFF is deleted in every case.
local LrApplication = import 'LrApplication'
local LrDialogs = import 'LrDialogs'
local LrExportSession = import 'LrExportSession'
local LrFileUtils = import 'LrFileUtils'
local LrFunctionContext = import 'LrFunctionContext'
local LrPathUtils = import 'LrPathUtils'
local LrProgressScope = import 'LrProgressScope'
local LrTasks = import 'LrTasks'
local Common = require 'Common'
local Config = require 'Config'

local TITLE = 'nss.ICM'
local trace = Common.trace

-- true when a file can be created in folder (checked with a tiny file, deleted again)
local function writable(folder)
  local probe = LrPathUtils.child(folder, 'nss_icm_write_test.tmp')
  local f = io.open(probe, 'w')
  if not f then return false end
  f:close()
  LrFileUtils.delete(probe)
  return true
end

-- The first of <base>_icm.tif, <base>_icm_2.tif, <base>_icm_3.tif ... in folder that does not exist yet.
local function unusedName(folder, base)
  local path = LrPathUtils.child(folder, base .. '_icm.tif')
  local n = 1
  while LrFileUtils.exists(path) do
    n = n + 1
    path = LrPathUtils.child(folder, base .. '_icm_' .. n .. '.tif')
  end
  return path
end

-- Runs the ICM window on tiff and takes its answer: imports output when it was saved.  notes: lines for the closing dialog.
local function handOver(catalog, photo, tiff, output, stack, notes, progress)
  local work = LrPathUtils.parent(tiff)
  local requestPath = LrPathUtils.child(work, 'request.txt')
  local resultPath = LrPathUtils.child(work, 'result.txt')
  LrFileUtils.delete(resultPath)
  local wrote, werr = Common.writeLines(requestPath, { 'tiff=' .. tiff, 'output=' .. output, 'result=' .. resultPath })
  if not wrote then
    LrDialogs.message(TITLE, 'Cannot write ' .. requestPath .. ': ' .. tostring(werr), 'critical')
    return
  end
  local command = Common.quote(Config.icmLauncher) .. ' ' .. Common.quote(requestPath)
  if WIN_ENV then command = '"' .. command .. '"' end     -- cmd /c strips one pair of quotes
  progress:setCaption('Waiting for the ICM window to close')
  trace('run: ' .. command)
  local code = LrTasks.execute(command)
  local values = Common.readValues(resultPath) or {}
  trace('launcher exit code ' .. tostring(code) .. ', status ' .. tostring(values.status) .. ', output exists: ' .. tostring(LrFileUtils.exists(output)))

  if values.status == 'cancelled' then return end
  if values.status ~= 'saved' or not LrFileUtils.exists(output) then
    LrDialogs.message(TITLE, 'nss.icm finished with exit code ' .. tostring(code) .. ', status ' .. tostring(values.status) .. ':\n' ..
                      (values.message or 'no output was written') .. '\n\nWhat the launcher printed: Documents\\nss_icm_bat.txt\nTrace: ' ..
                      Common.tracePath, 'critical')
    return
  end

  progress:setCaption('Importing ' .. LrPathUtils.leafName(output))
  local newPhoto
  catalog:withWriteAccessDo(TITLE, function()
    if stack then newPhoto = catalog:addPhoto(output, photo, 'above') else newPhoto = catalog:addPhoto(output) end
  end, { timeout = 10 })
  catalog:setSelectedPhotos(newPhoto, { newPhoto })
  trace('imported ' .. output .. (stack and ', stacked above the original' or ''))
  if #notes > 0 then LrDialogs.message(TITLE, table.concat(notes, '\n\n'), 'info') end
end

local function run(context)
  trace('--- ICM')
  local catalog = LrApplication.activeCatalog()
  local photo = catalog:getTargetPhoto()
  if not photo then
    LrDialogs.message(TITLE, 'Select a photo first.', 'info')
    return
  end
  local notes = {}
  local selected = #catalog:getTargetPhotos()
  if selected > 1 then
    table.insert(notes, selected .. ' photos were selected; only the active one, ' .. photo:getFormattedMetadata('fileName') .. ', was opened.')
  end

  -- the output: next to the original, or in the fallback folder (a photo can only be stacked with one in the same folder)
  local original = photo:getRawMetadata('path')
  local folder, stack = LrPathUtils.parent(original), true
  if not writable(folder) then
    folder, stack = Config.icmFallbackFolder, false
    LrFileUtils.createAllDirectories(folder)
  end
  local output = unusedName(folder, LrPathUtils.removeExtension(LrPathUtils.leafName(original)))
  if not stack then table.insert(notes, 'The folder of the original is not writable; the ICM picture was saved as\n' .. output) end
  trace('output ' .. output)

  -- 1. export
  local progress = LrProgressScope({ title = TITLE, functionContext = context })
  progress:setCaption('Exporting ' .. photo:getFormattedMetadata('fileName'))
  local work = LrPathUtils.child(LrPathUtils.getStandardFilePath('temp'), Config.icmWorkFolderName)
  LrFileUtils.createAllDirectories(work)
  local tiff = LrPathUtils.child(work, 'nss_icm_input.tif')
  LrFileUtils.delete(tiff)
  local exportSettings = Common.tiffExportSettings(work)
  exportSettings.LR_collisionHandling = 'overwrite'
  exportSettings.LR_renamingTokensOn = true
  exportSettings.LR_tokens = 'nss_icm_input'
  local session = LrExportSession({ photosToExport = { photo }, exportSettings = exportSettings })
  local ok, err = LrTasks.pcall(function() session:doExportOnCurrentTask() end)
  trace('export ok: ' .. tostring(ok) .. ', tiff exists: ' .. tostring(LrFileUtils.exists(tiff)))
  if not ok or not LrFileUtils.exists(tiff) then
    LrDialogs.message(TITLE, 'Export failed: ' .. (ok and ('no file ' .. tiff) or tostring(err)), 'critical')
    return
  end

  -- 2. the ICM window, then the exported TIFF goes whatever happened
  local done, handOverErr = LrTasks.pcall(handOver, catalog, photo, tiff, output, stack, notes, progress)
  LrFileUtils.delete(tiff)
  trace('deleted ' .. tiff)
  if not done then error(handOverErr, 0) end
end

LrTasks.startAsyncTask(function()
  local ok, err = LrTasks.pcall(LrFunctionContext.callWithContext, 'nssIcm', run)
  if not ok then
    trace('error: ' .. tostring(err))
    LrDialogs.message(TITLE, 'Error: ' .. tostring(err), 'critical')
  end
end)
