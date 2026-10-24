-- Library > Plug-in Extras > nss > Color: Edit...
--   1. remember the photo's mixer / color grading values, set them to neutral, export a 16-bit Adobe RGB TIFF (everything else baked in), put the values back
--   2. write the request file, run the launcher (the nss color GUI) and wait for it
--   3. read the values the GUI wrote and apply them to the photo (one undo step "nss.Color")
local LrApplication = import 'LrApplication'
local LrTasks = import 'LrTasks'
local LrFileUtils = import 'LrFileUtils'
local LrPathUtils = import 'LrPathUtils'
local LrDialogs = import 'LrDialogs'
local LrExportSession = import 'LrExportSession'
local LrLogger = import 'LrLogger'
local Keys = require 'Keys'
local Config = require 'Config'

local logger = LrLogger('nssColor')
logger:enable('logfile')

local tracePath = LrPathUtils.child(LrPathUtils.getStandardFilePath('documents'), 'nss_color_trace.txt')
local function trace(msg)
  local f = io.open(tracePath, 'a')
  if f then
    f:write(os.date('%H:%M:%S '), msg, '\n')
    f:close()
  end
  logger:info(msg)
end

local function writeLines(path, lines)
  local f, err = io.open(path, 'w')
  if not f then return false, err end
  f:write(table.concat(lines, '\n'), '\n')
  f:close()
  return true
end

local function readValues(path)
  local f = io.open(path, 'r')
  if not f then return nil end
  local values = {}
  for line in f:lines() do
    local k, v = line:match('^%s*([%w_]+)%s*=%s*(.-)%s*$')
    if k and tonumber(v) then values[k] = tonumber(v) end
  end
  f:close()
  return values
end

local function applyTo(catalog, photo, settings, name)
  catalog:withWriteAccessDo(name, function()
    photo:applyDevelopSettings(settings, name)
  end, { timeout = 10 })
end

local function quote(s) return '"' .. s .. '"' end

local function run()
  trace('--- Color: Edit')
  local catalog = LrApplication.activeCatalog()
  local photo = catalog:getTargetPhoto()
  if not photo then
    LrDialogs.message('nss.Color', 'Select a photo first.', 'info')
    return
  end

  local current = photo:getDevelopSettings()
  local missing = {}
  for _, k in ipairs(Keys.all) do
    if current[k] == nil then table.insert(missing, k) end
  end
  if #missing > 0 then
    LrDialogs.message('nss.Color', 'Unknown develop settings (nothing was changed):\n' .. table.concat(missing, ', ') .. '\nRun "Color: Write develop settings to the log" and correct Keys.lua.', 'critical')
    return
  end

  local work = LrPathUtils.child(LrPathUtils.getStandardFilePath('temp'), Config.workFolderName)
  LrFileUtils.createAllDirectories(work)
  local tiff = LrPathUtils.child(work, 'nss_input.tif')
  local requestPath = LrPathUtils.child(work, 'request.txt')
  local resultPath = LrPathUtils.child(work, 'result.txt')
  local backupPath = LrPathUtils.child(work, 'backup.txt')
  LrFileUtils.delete(tiff)
  LrFileUtils.delete(resultPath)

  -- the photo's own values (also saved to disk: if Lightroom dies in the next seconds they can be restored by hand)
  local saved, backup = {}, {}
  for _, k in ipairs(Keys.all) do
    saved[k] = current[k]
    table.insert(backup, k .. '=' .. tostring(current[k]))
  end
  writeLines(backupPath, backup)

  -- 1. export without the mixer and the wheels
  trace('exporting to ' .. tiff)
  applyTo(catalog, photo, Keys.neutral, 'nss.Color: export without mixer and wheels')
  local session = LrExportSession({
    photosToExport = { photo },
    exportSettings = {
      LR_format = 'TIFF',
      LR_export_colorSpace = 'AdobeRGB',
      LR_export_bitDepth = 16,
      LR_tiff_compressionMethod = 'compressionMethod_None',
      LR_export_destinationType = 'specificFolder',
      LR_export_destinationPathPrefix = work,
      LR_export_useSubfolder = false,
      LR_collisionHandling = 'overwrite',
      LR_renamingTokensOn = true,
      LR_tokens = 'nss_input',
      LR_size_doConstrain = false,
      LR_outputSharpeningOn = false,
      LR_reimportExportedPhoto = false,
      LR_useWatermark = false,
    },
  })
  local ok, err = LrTasks.pcall(function() session:doExportOnCurrentTask() end)
  applyTo(catalog, photo, saved, 'nss.Color: restore')      -- the photo is back as it was, whatever happened
  if not ok then
    logger:error('export failed: ' .. tostring(err))
    LrDialogs.message('nss.Color', 'Export failed: ' .. tostring(err), 'critical')
    return
  end
  trace('export done, tiff exists: ' .. tostring(LrFileUtils.exists(tiff)))
  if not LrFileUtils.exists(tiff) then
    LrDialogs.message('nss.Color', 'The export did not produce ' .. tiff, 'critical')
    return
  end

  -- 2. hand over to the GUI
  local lines = { 'tiff=' .. tiff, 'result=' .. resultPath }
  for _, k in ipairs(Keys.all) do table.insert(lines, k .. '=' .. tostring(saved[k])) end
  local wrote, werr = writeLines(requestPath, lines)
  if not wrote then
    LrDialogs.message('nss.Color', 'Cannot write ' .. requestPath .. ': ' .. tostring(werr), 'critical')
    return
  end
  local command = quote(Config.launcher) .. ' ' .. quote(requestPath)
  if WIN_ENV then command = '"' .. command .. '"' end     -- cmd /c strips one pair of quotes
  trace('run: ' .. command)
  local code = LrTasks.execute(command)
  trace('launcher exit code ' .. tostring(code) .. ', result file exists: ' .. tostring(LrFileUtils.exists(resultPath)))

  -- 3. take the answer
  if code ~= 0 or not LrFileUtils.exists(resultPath) then
    LrDialogs.message('nss.Color', 'The GUI finished with exit code ' .. tostring(code) .. ' and wrote no result, so nothing was applied.\n\nWhat the launcher printed: Documents\\nss_color_bat.txt\nTrace: ' .. tracePath, 'info')
    return
  end
  local values = readValues(resultPath)
  local settings, count = {}, 0
  for _, k in ipairs(Keys.all) do
    if values and values[k] then
      settings[k] = math.floor(values[k] + 0.5)
      count = count + 1
    end
  end
  if count ~= #Keys.all then
    LrDialogs.message('nss.Color', 'The result file has ' .. count .. ' of ' .. #Keys.all .. ' values; nothing was applied.', 'critical')
    return
  end
  applyTo(catalog, photo, settings, 'nss.Color')
  trace('applied ' .. count .. ' settings')
end

LrTasks.startAsyncTask(function()
  local ok, err = LrTasks.pcall(run)
  if not ok then
    logger:error(tostring(err))
    LrDialogs.message('nss.Color', 'Error: ' .. tostring(err), 'critical')
  end
end)