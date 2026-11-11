-- Library > Plug-in Extras > nss > Stack: Clean / Stack: Focus (StackClean.lua and StackFocus.lua call run; the work is here)
--   1. the selected photos: the active one (catalog:getTargetPhoto, the one Lightroom highlights as the most selected) starts as the target,
--      or the first selected when Lightroom reports none; the others are the frames, in the order Lightroom lists them (at most
--      Config.stackMaxFrames of them).  The target can be changed in the window, so this only has to be a good guess.
--   2. export every photo as a 16-bit Adobe RGB TIFF, every edit and the crop baked in
--   3. pick an output <name>_clean.tif (_clean2.tif, _focus.tif) for EVERY photo, next to its original (then _clean_2 ...; never overwrites), or in
--      Config.stackFallbackFolder when that folder is not writable: the window writes the one of whichever photo ends up as the target
--   4. write the request file (plugin/stack_bridge.py), run Config.stackLauncher (python focus_stack.py --lightroom) and wait for it
--   5. saved: import the output stacked above the photo that was the target (the result says which) and select it; cancelled: nothing;
--      error or no result: a dialog.  The exported TIFFs are deleted in every case.
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

local M = {}
local TITLE = 'nss.Stack'
local trace = Common.trace
M.trace = trace

-- true when a file can be created in folder (checked with a tiny file, deleted again)
local function writable(folder)
  local probe = LrPathUtils.child(folder, 'nss_stack_write_test.tmp')
  local f = io.open(probe, 'w')
  if not f then return false end
  f:close()
  LrFileUtils.delete(probe)
  return true
end

-- The first of <base>_<suffix>.tif, <base>_<suffix>_2.tif, <base>_<suffix>_3.tif ... in folder that does not exist yet.
local function unusedName(folder, base, suffix)
  local path = LrPathUtils.child(folder, base .. '_' .. suffix .. '.tif')
  local n = 1
  while LrFileUtils.exists(path) do
    n = n + 1
    path = LrPathUtils.child(folder, base .. '_' .. suffix .. '_' .. n .. '.tif')
  end
  return path
end

-- Deletes the exported TIFFs.
local function deleteAll(tiffs)
  for _, tiff in ipairs(tiffs) do LrFileUtils.delete(tiff) end
end

-- Exports the photos into folder work as nss_stack_0.tif (the target), nss_stack_1.tif ...; the list of the paths made so far, then true
-- when all went well, false when the user canceled, or false and a message when an export failed.
local function exportAll(photos, work, progress)
  local tiffs = {}
  for i, photo in ipairs(photos) do
    if progress:isCanceled() then return tiffs, false end
    progress:setCaption(string.format('Exporting %s (%d of %d)', photo:getFormattedMetadata('fileName'), i, #photos))
    progress:setPortionComplete(i - 1, #photos)
    local name = 'nss_stack_' .. (i - 1)
    local tiff = LrPathUtils.child(work, name .. '.tif')
    LrFileUtils.delete(tiff)
    local exportSettings = Common.tiffExportSettings(work)
    exportSettings.LR_collisionHandling = 'overwrite'
    exportSettings.LR_renamingTokensOn = true
    exportSettings.LR_tokens = name
    local session = LrExportSession({ photosToExport = { photo }, exportSettings = exportSettings })
    local ok, err = LrTasks.pcall(function() session:doExportOnCurrentTask() end)
    trace('export ' .. i .. ' ok: ' .. tostring(ok) .. ', tiff exists: ' .. tostring(LrFileUtils.exists(tiff)))
    if not ok or not LrFileUtils.exists(tiff) then
      return tiffs, false, 'Export of ' .. photo:getFormattedMetadata('fileName') .. ' failed: ' .. (ok and ('no file ' .. tiff) or tostring(err))
    end
    tiffs[i] = tiff
  end
  progress:setPortionComplete(#photos, #photos)
  return tiffs, true
end

-- Runs the stack window on the exported tiffs (the target first) and takes its answer: imports the output of the photo that was the
-- target when it was saved.  outputs[i] and stacks[i] belong to photos[i]: where its result goes and whether it can be stacked with it.
local function handOver(catalog, photos, mode, tiffs, outputs, stacks, progress)
  local work = LrPathUtils.parent(tiffs[1])
  local requestPath = LrPathUtils.child(work, 'request.txt')
  local resultPath = LrPathUtils.child(work, 'result.txt')
  LrFileUtils.delete(resultPath)
  local lines = { 'mode=' .. mode, 'target=' .. tiffs[1], 'count=' .. (#tiffs - 1) }
  for i = 2, #tiffs do lines[#lines + 1] = 'frame.' .. (i - 1) .. '=' .. tiffs[i] end
  lines[#lines + 1] = 'output=' .. outputs[1]
  for i = 2, #tiffs do lines[#lines + 1] = 'output.' .. (i - 1) .. '=' .. outputs[i] end
  lines[#lines + 1] = 'result=' .. resultPath
  local wrote, werr = Common.writeLines(requestPath, lines)
  if not wrote then
    LrDialogs.message(TITLE, 'Cannot write ' .. requestPath .. ': ' .. tostring(werr), 'critical')
    return
  end
  local command = Common.quote(Config.stackLauncher) .. ' ' .. Common.quote(requestPath)
  if WIN_ENV then command = '"' .. command .. '"' end     -- cmd /c strips one pair of quotes
  progress:setCaption('Waiting for the Stack window to close')
  trace('run: ' .. command)
  local code = LrTasks.execute(command)
  local values = Common.readValues(resultPath) or {}
  local index = (tonumber(values.target) or 0) + 1       -- the result counts photos from 0
  local output = values.output
  trace('launcher exit code ' .. tostring(code) .. ', status ' .. tostring(values.status) .. ', target ' .. tostring(values.target) ..
        ', output exists: ' .. tostring(output ~= nil and LrFileUtils.exists(output)))

  if values.status == 'cancelled' then return end
  if values.status ~= 'saved' or not output or not LrFileUtils.exists(output) or not photos[index] then
    LrDialogs.message(TITLE, 'focus_stack finished with exit code ' .. tostring(code) .. ', status ' .. tostring(values.status) .. ':\n' ..
                      (values.message or 'no output was written') .. '\n\nWhat the launcher printed: Documents\\nss_stack_bat.txt\nTrace: ' ..
                      Common.tracePath, 'critical')
    return
  end

  progress:setCaption('Importing ' .. LrPathUtils.leafName(output))
  local target, stack = photos[index], stacks[index]
  local newPhoto
  catalog:withWriteAccessDo(TITLE, function()
    if stack then newPhoto = catalog:addPhoto(output, target, 'above') else newPhoto = catalog:addPhoto(output) end
  end, { timeout = 10 })
  catalog:setSelectedPhotos(newPhoto, { newPhoto })
  trace('imported ' .. output .. (stack and (', stacked above ' .. target:getFormattedMetadata('fileName')) or ''))
  if not stack then
    LrDialogs.message(TITLE, 'The folder of ' .. target:getFormattedMetadata('fileName') .. ' is not writable; the picture was saved as\n' .. output, 'info')
  end
end

local function runIn(context, mode)
  trace('--- Stack: ' .. mode)
  local catalog = LrApplication.activeCatalog()
  local selected = catalog:getTargetPhotos()
  if #selected < 2 then
    LrDialogs.message(TITLE, 'Select two or more photos of the same scene first. The active one starts as the target; you can change it in the window.', 'info')
    return
  end
  if #selected > Config.stackMaxFrames + 1 then
    LrDialogs.message(TITLE, 'Select at most ' .. (Config.stackMaxFrames + 1) .. ' photos (the target and ' .. Config.stackMaxFrames ..
                      ' frames); ' .. #selected .. ' are selected. Work in batches.', 'info')
    return
  end
  -- the target starts as the active photo (the first selected one when Lightroom reports none); the frames are the others in the order
  -- Lightroom lists them.  The window lets the user choose another target.
  local active = catalog:getTargetPhoto()
  local photos = {}
  for _, photo in ipairs(selected) do
    if active and photo.localIdentifier == active.localIdentifier then table.insert(photos, 1, photo) else table.insert(photos, photo) end
  end
  local names = {}
  for i, photo in ipairs(photos) do names[i] = photo:getFormattedMetadata('fileName') end
  trace('target (' .. ((active and photos[1].localIdentifier == active.localIdentifier) and 'the active photo' or 'the first selected') ..
        ') and frames: ' .. table.concat(names, ', '))

  -- an output for every photo, for when it is made the target in the window: next to its original, or in the fallback folder (a photo
  -- can only be stacked with one in the same folder)
  local outputs, stacks = {}, {}
  for i, photo in ipairs(photos) do
    local original = photo:getRawMetadata('path')
    local folder, stack = LrPathUtils.parent(original), true
    if not writable(folder) then
      folder, stack = Config.stackFallbackFolder, false
      LrFileUtils.createAllDirectories(folder)
    end
    outputs[i] = unusedName(folder, LrPathUtils.removeExtension(LrPathUtils.leafName(original)), mode)
    stacks[i] = stack
  end
  trace('outputs: ' .. table.concat(outputs, ', '))

  -- 1. export
  local progress = LrProgressScope({ title = TITLE, functionContext = context })
  local work = LrPathUtils.child(LrPathUtils.getStandardFilePath('temp'), Config.stackWorkFolderName)
  LrFileUtils.createAllDirectories(work)
  local tiffs, complete, problem = exportAll(photos, work, progress)
  if not complete then
    deleteAll(tiffs)
    if problem then LrDialogs.message(TITLE, problem, 'critical') end
    return
  end

  -- 2. the stack window, then the exported TIFFs go whatever happened
  local done, handOverErr = LrTasks.pcall(handOver, catalog, photos, mode, tiffs, outputs, stacks, progress)
  deleteAll(tiffs)
  trace('deleted the exported TIFFs')
  if not done then error(handOverErr, 0) end
end

-- mode is 'clean', 'clean2' or 'focus'; call from a task.
function M.run(mode)
  LrFunctionContext.callWithContext('nssStack', function(context) runIn(context, mode) end)
end

return M
