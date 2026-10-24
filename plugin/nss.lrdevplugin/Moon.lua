-- Library > Plug-in Extras > nss > Moon: Center / Moon: Verify centering (CenterMoon.lua, VerifyMoon.lua).
--   center: back up every photo's crop keys, then chunk by chunk (Settings: photos per chunk): set the full frame, export the photos at
--           full size as 16-bit TIFFs (all other edits baked in), put the crops back (also on errors), measure the moons with
--           moon.py (request / result files: plugin/moon_bridge.py) and delete the exported files.  Then moon.py computes all the crops from
--           the measurements in one go (several photos = batch mode: one common crop size) and they are applied, one undo step
--           "nss.Moon: center" per photo.  No dialog: the results go to the trace file, which each center run starts afresh.
--   verify: the same chunks exported WITH the current crop; a dialog reports how far the moon's centre is from the crop's centre
--           (its lines are appended to the trace).
-- moon.py runs in a second task while this one moves the progress bar and turns a cancel into moon.py's cancel file.
-- Only the crop keys (Keys.crop, Keys.cropPrefix) are touched, never the mixer or grading keys.
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
local Keys = require 'Keys'
local Settings = require 'Settings'

local M = {}
local HISTORY = 'nss.Moon: center'
local RESULT_KEYS = { 'status', 'message', 'confidence', 'cx', 'cy', 'radius', 'width', 'height', 'crop_width', 'crop_height', 'dx', 'dy',
                      'left', 'top', 'right', 'bottom' }

local tracePath = Common.tracePath
local writeLines, readValues, applyTo, quote = Common.writeLines, Common.readValues, Common.applyTo, Common.quote

-- Empties the trace file and writes msg as its first line: Moon: Center starts each run's trace afresh (verify and Color: Edit... append).
function M.startTrace(msg)
  local f = io.open(tracePath, 'w')
  if f then f:close() end
  M.trace(msg)
end

-- Appends a line to the trace file (and Lightroom's log).
M.trace = Common.trace

local function photoName(photo) return photo:getFormattedMetadata('fileName') end

-- the work folder inside Lightroom's temp folder, emptied of an earlier run's files
local function workFolder()
  local work = LrPathUtils.child(LrPathUtils.getStandardFilePath('temp'), Config.moonWorkFolderName)
  LrFileUtils.createAllDirectories(work)
  local old = {}
  for file in LrFileUtils.files(work) do table.insert(old, file) end
  for _, file in ipairs(old) do LrFileUtils.delete(file) end
  return work
end

-- The progress bar over the whole run: bar = { scope = LrProgressScope, done = steps done, total = steps }; one step per photo exported,
-- measured or applied.
local function tick(bar, steps)
  bar.done = bar.done + (steps or 1)
  bar.scope:setPortionComplete(bar.done, bar.total)
end

-- Exports photos as 16-bit Adobe RGB TIFFs, full size, no compression, into work.  Returns (paths, failures) indexed like photos (a failed
-- render has a failures entry instead of a path), or nil when the progress bar was canceled (the remaining renders are skipped).
local function export(photos, work, bar)
  local index = {}
  for i, photo in ipairs(photos) do index[photo.localIdentifier] = i end
  local exportSettings = Common.tiffExportSettings(work)       -- full size: detect_moon's grid size is in pixels
  exportSettings.LR_collisionHandling = 'rename'               -- two photos with the same file name get distinct TIFFs
  exportSettings.LR_renamingTokensOn = false
  local session = LrExportSession({ photosToExport = photos, exportSettings = exportSettings })
  local paths, failures, canceled = {}, {}, false
  for _, rendition in session:renditions() do
    canceled = canceled or bar.scope:isCanceled()
    if canceled then
      rendition:skipRender()
    else
      local i = index[rendition.photo.localIdentifier]
      local ok, pathOrMessage = rendition:waitForRender()
      if ok then paths[i] = pathOrMessage else failures[i] = tostring(pathOrMessage) end
      tick(bar)
    end
  end
  if canceled then return nil end
  return paths, failures
end

-- Runs moon.py on a request (header: mode= ... lines; photoLines: the photo.i.* lines of count photos) in a second task.  Meanwhile, every
-- half second, the progress file moves the bar one step per photo measured, and a cancel of the bar writes moon.py's cancel file.
-- Returns (results per photo, {key = string} from RESULT_KEYS, canceled), or nil and a message when moon.py wrote no result.
local function runMoon(work, header, photoLines, count, bar)
  local requestPath = LrPathUtils.child(work, 'request.txt')
  local resultPath = LrPathUtils.child(work, 'result.txt')
  local progressPath = LrPathUtils.child(work, 'progress.txt')
  local cancelPath = LrPathUtils.child(work, 'cancel.txt')
  for _, path in ipairs({ resultPath, progressPath, cancelPath }) do LrFileUtils.delete(path) end
  local lines = { 'result=' .. resultPath, 'progress=' .. progressPath, 'cancel=' .. cancelPath, 'count=' .. count }
  for _, line in ipairs(header) do table.insert(lines, line) end
  for _, line in ipairs(photoLines) do table.insert(lines, line) end
  local wrote, werr = writeLines(requestPath, lines)
  if not wrote then return nil, 'Cannot write ' .. requestPath .. ': ' .. tostring(werr) end
  local command = quote(Config.moonLauncher) .. ' ' .. quote(requestPath)
  if WIN_ENV then command = '"' .. command .. '"' end     -- cmd /c strips one pair of quotes
  M.trace('run: ' .. command)

  local finished, code = false, nil
  LrTasks.startAsyncTask(function()
    local ok, result = LrTasks.pcall(LrTasks.execute, command)
    code = ok and result or tostring(result)
    finished = true
  end)
  local measured, canceled = 0, false
  while not finished do
    LrTasks.sleep(0.5)
    if not canceled and bar.scope:isCanceled() then
      canceled = true
      writeLines(cancelPath, { 'cancel' })
      M.trace('cancel requested')
    end
    local f = io.open(progressPath, 'r')
    if f then
      local done = tonumber(((f:read('*l') or ''):match('^(%d+)')))
      f:close()
      if done and done > measured then
        tick(bar, done - measured)
        measured = done
      end
    end
  end
  M.trace('launcher exit code ' .. tostring(code) .. ', result file exists: ' .. tostring(LrFileUtils.exists(resultPath)))
  local values = code == 0 and readValues(resultPath) or nil
  if not values then
    return nil, 'moon.py finished with exit code ' .. tostring(code) .. ' and wrote no result.\n\nWhat it printed: Documents\\nss_moon_bat.txt\nTrace: ' .. tracePath
  end
  local results = {}
  for i = 1, count do
    local r = {}
    for _, key in ipairs(RESULT_KEYS) do r[key] = values['photo.' .. i .. '.' .. key] end
    r.status = r.status or 'error'
    r.message = r.message or 'no result for this photo'
    results[i] = r
  end
  return results, canceled
end

-- Exports and measures photos, Settings' photos per chunk at a time, deleting each chunk's exported files once measured.  With saved (crop keys per
-- localIdentifier) a chunk is exported at the full frame and its crops put back right after its export; without, as cropped.
-- Returns the measure results indexed like photos, each with orientation and angle added; nil when canceled; nil and a message on failure.
local function measureInChunks(catalog, photos, work, bar, saved)
  local results = {}
  local chunkSize = Settings.get('moonChunkSize')
  for first = 1, #photos, chunkSize do
    if bar.scope:isCanceled() then return nil end
    local last = math.min(first + chunkSize - 1, #photos)
    local chunk = {}
    for i = first, last do table.insert(chunk, photos[i]) end
    bar.scope:setCaption(string.format('Exporting photos %d to %d of %d', first, last, #photos))
    M.trace(string.format('chunk %d .. %d: export', first, last))
    local ok, paths, failures
    if saved then
      ok, paths, failures = LrTasks.pcall(function()
        for _, photo in ipairs(chunk) do applyTo(catalog, photo, Keys.cropFull, 'nss.Moon: center, full frame for export') end
        return export(chunk, work, bar)
      end)
      local restoreFailed = {}
      for _, photo in ipairs(chunk) do
        local restored, rerr = LrTasks.pcall(applyTo, catalog, photo, saved[photo.localIdentifier], 'nss.Moon: center, restore crop')
        if not restored then table.insert(restoreFailed, photoName(photo) .. ': ' .. tostring(rerr)) end
      end
      M.trace('export ok: ' .. tostring(ok) .. ', crops restored, failures: ' .. #restoreFailed)
      if #restoreFailed > 0 then
        return nil, 'Could not restore the crop of:\n' .. table.concat(restoreFailed, '\n') .. '\n\nThe saved values are in ' .. LrPathUtils.child(work, 'crop_backup.txt')
      end
    else
      ok, paths, failures = LrTasks.pcall(export, chunk, work, bar)
    end
    if not ok then error('export failed: ' .. tostring(paths)) end
    if paths == nil then return nil end

    -- measure the chunk's exported files; an export failure counts as that photo's measure step
    local photoLines, measuredIndex, canceled = {}, {}, false
    for j, photo in ipairs(chunk) do
      local c = saved and saved[photo.localIdentifier]
      local orientation = c and (photo:getRawMetadata('orientation') or 'AB') or 'AB'      -- verify: the exported image is the crop as shown
      local angle = c and c.CropAngle or 0
      local crop = c and { c.CropLeft, c.CropTop, c.CropRight, c.CropBottom } or { 0, 0, 1, 1 }
      if paths[j] then
        table.insert(measuredIndex, j)
        local p = 'photo.' .. #measuredIndex .. '.'
        table.insert(photoLines, p .. 'id=' .. tostring(photo.localIdentifier))
        table.insert(photoLines, p .. 'image=' .. paths[j])
        table.insert(photoLines, p .. 'left=' .. tostring(crop[1]))
        table.insert(photoLines, p .. 'top=' .. tostring(crop[2]))
        table.insert(photoLines, p .. 'right=' .. tostring(crop[3]))
        table.insert(photoLines, p .. 'bottom=' .. tostring(crop[4]))
        table.insert(photoLines, p .. 'orientation=' .. orientation)
        table.insert(photoLines, p .. 'angle=' .. tostring(angle))
      else
        results[first + j - 1] = { status = 'error', message = 'export failed: ' .. tostring(failures[j]), orientation = orientation, angle = angle }
        tick(bar)
      end
    end
    if #measuredIndex > 0 then
      bar.scope:setCaption(string.format('Finding the moon in photos %d to %d of %d', first, last, #photos))
      local header = { 'mode=measure', 'size=' .. tostring(Config.moonSize), 'min_confidence=' .. tostring(Config.moonMinConfidence),
                       'workers=' .. tostring(Settings.get('moonWorkers')) }
      local measured, canceledOrMessage = runMoon(work, header, photoLines, #measuredIndex, bar)
      if not measured then return nil, canceledOrMessage end
      canceled = canceledOrMessage
      for k, j in ipairs(measuredIndex) do
        local r = measured[k]
        local c = saved and saved[chunk[j].localIdentifier]
        r.orientation = c and (chunk[j]:getRawMetadata('orientation') or 'AB') or 'AB'
        r.angle = c and c.CropAngle or 0
        results[first + j - 1] = r
      end
    end
    for _, path in pairs(paths) do LrFileUtils.delete(path) end
    if canceled then return nil end
  end
  return results
end

local function offsetText(r) return string.format('%.1f px, %.1f px', tonumber(r.dx), tonumber(r.dy)) end

local function summary(title, lines)
  LrDialogs.message('nss.Moon', title .. '\n\n' .. table.concat(lines, '\n'), 'info')
end

-- Writes a result title and its lines to the trace instead of a dialog.
local function traceSummary(title, lines)
  M.trace(title)
  for _, line in ipairs(lines) do M.trace('  ' .. line) end
end

function M.center()
  LrFunctionContext.callWithContext('nssCenterMoon', function(context)
    M.startTrace('--- Moon: Center')
    local catalog = LrApplication.activeCatalog()
    local photos = catalog:getTargetPhotos()
    if #photos == 0 then
      LrDialogs.message('nss.Moon', 'Select one or more photos first.', 'info')
      return
    end
    local progress = LrProgressScope({ title = HISTORY, functionContext = context })

    -- the photos to center and their crop keys (a crop angle is kept: the new crop is turned by it about its centre)
    local todo, saved, skipped, backup = {}, {}, {}, {}
    for _, photo in ipairs(photos) do
      local settings = photo:getDevelopSettings()
      local missing = false
      for _, k in ipairs(Keys.crop) do missing = missing or settings[k] == nil end
      if missing then
        table.insert(skipped, photoName(photo) .. ': no crop settings (run "Color: Write develop settings to the log" and check Keys.crop)')
      else
        local crop = {}
        for k, v in pairs(settings) do
          if k:sub(1, #Keys.cropPrefix) == Keys.cropPrefix then
            crop[k] = v
            table.insert(backup, tostring(photo.localIdentifier) .. '.' .. k .. '=' .. tostring(v))
          end
        end
        table.insert(todo, photo)
        saved[photo.localIdentifier] = crop
      end
    end
    M.trace(#todo .. ' to center, ' .. #skipped .. ' skipped before export')
    if #todo == 0 then
      traceSummary('0 centered.', skipped)
      return
    end
    local mode = #todo == 1 and 'single' or 'batch'

    -- 1. export and measure in chunks; the crops go back after each chunk's export (all saved to disk first, to restore by hand if
    --    Lightroom dies meanwhile)
    local work = workFolder()
    writeLines(LrPathUtils.child(work, 'crop_backup.txt'), backup)
    local bar = { scope = progress, done = 0, total = 3 * #todo }
    local results, message = measureInChunks(catalog, todo, work, bar, saved)
    if not results then
      if message then
        LrDialogs.message('nss.Moon', message, 'critical')
      else
        LrDialogs.message('nss.Moon', 'Canceled; no photo was changed.', 'info')
      end
      return
    end

    -- 2. the crops of all the found moons at once (batch: one common size over every chunk)
    local found, photoLines = {}, {}
    for i, photo in ipairs(todo) do
      local r = results[i]
      if r.status == 'ok' then
        table.insert(found, { photo = photo, r = r })
        local p = 'photo.' .. #found .. '.'
        table.insert(photoLines, p .. 'id=' .. tostring(photo.localIdentifier))
        for _, key in ipairs({ 'cx', 'cy', 'width', 'height', 'crop_width', 'crop_height' }) do table.insert(photoLines, p .. key .. '=' .. r[key]) end
        table.insert(photoLines, p .. 'orientation=' .. r.orientation)
        table.insert(photoLines, p .. 'angle=' .. tostring(r.angle))
      else
        table.insert(skipped, photoName(photo) .. ': ' .. r.status .. ', ' .. r.message)
        tick(bar)
      end
    end
    if #found > 0 then
      progress:setCaption('Computing the crops')
      local crops, cropMessage = runMoon(work, { 'mode=' .. mode }, photoLines, #found, bar)
      if not crops then
        LrDialogs.message('nss.Moon', cropMessage, 'critical')
        return
      end
      for k, f in ipairs(found) do
        for _, key in ipairs({ 'left', 'top', 'right', 'bottom' }) do f.r[key] = crops[k][key] end
      end
    end

    -- 3. apply the crops, one undo step per photo; canceling stops before the next photo
    progress:setCaption('Applying the crops')
    local centered = {}
    for _, f in ipairs(found) do
      local r = f.r
      if progress:isCanceled() then
        table.insert(skipped, photoName(f.photo) .. ': canceled')
      else
        local asked = { CropLeft = tonumber(r.left), CropTop = tonumber(r.top), CropRight = tonumber(r.right), CropBottom = tonumber(r.bottom) }
        applyTo(catalog, f.photo, asked, HISTORY)
        table.insert(centered, photoName(f.photo) .. ': moved by ' .. offsetText(r) .. ', confidence ' .. r.confidence .. ' (' .. r.orientation .. ', angle ' .. tostring(r.angle) .. ')')
        -- Lightroom may move or shrink a crop it finds outside the photo: report what it stored when that differs from what was asked
        local stored = f.photo:getDevelopSettings()
        local changed = false
        for _, k in ipairs({ 'CropLeft', 'CropTop', 'CropRight', 'CropBottom' }) do changed = changed or math.abs(stored[k] - asked[k]) > 1e-5 end
        if changed then
          local function ltrb(t) return string.format('%.5f %.5f %.5f %.5f', t.CropLeft, t.CropTop, t.CropRight, t.CropBottom) end
          table.insert(centered, '    Lightroom changed the crop: asked ' .. ltrb(asked) .. ', stored ' .. ltrb(stored))
        end
      end
      tick(bar)
    end
    local lines = centered
    if #skipped > 0 then
      table.insert(lines, #skipped .. ' skipped:')
      for _, line in ipairs(skipped) do table.insert(lines, line) end
    end
    traceSummary(#centered .. ' centered, ' .. #skipped .. ' skipped (' .. mode .. ' mode):', lines)
  end)
end

function M.verify()
  LrFunctionContext.callWithContext('nssVerifyMoon', function(context)
    M.trace('--- Moon: Verify centering')
    local catalog = LrApplication.activeCatalog()
    local photos = catalog:getTargetPhotos()
    if #photos == 0 then
      LrDialogs.message('nss.Moon', 'Select one or more photos first.', 'info')
      return
    end
    local progress = LrProgressScope({ title = 'nss.Moon: verify centering', functionContext = context })
    local work = workFolder()
    M.trace('export with the current crop to ' .. work)
    local bar = { scope = progress, done = 0, total = 2 * #photos }
    local results, message = measureInChunks(catalog, photos, work, bar, nil)
    if not results then
      LrDialogs.message('nss.Moon', message or 'Canceled.', message and 'critical' or 'info')
      return
    end
    local lines = {}
    for i, photo in ipairs(photos) do
      local r = results[i]
      if r.status == 'ok' then
        local distance = math.sqrt(tonumber(r.dx) ^ 2 + tonumber(r.dy) ^ 2)
        local verdict = distance <= Config.verifyTolerance and 'centred' or string.format('NOT centred (over %s px)', tostring(Config.verifyTolerance))
        table.insert(lines, string.format('%s: %s, moon centre off the crop centre by %s (%.2f px), confidence %.3f, orientation %s', photoName(photo), verdict,
                                          offsetText(r), distance, tonumber(r.confidence), photo:getRawMetadata('orientation') or '?'))
        local s = photo:getDevelopSettings()
        M.trace(string.format('verify %s: crop %.6f %.6f %.6f %.6f angle %s, export %s x %s px, moon at %s %s, offset %s %s', photoName(photo),
                              s.CropLeft, s.CropTop, s.CropRight, s.CropBottom, tostring(s.CropAngle), r.width, r.height, r.cx, r.cy, r.dx, r.dy))
      else
        table.insert(lines, photoName(photo) .. ': ' .. r.status .. ', ' .. r.message)
      end
    end
    M.trace('verified ' .. #photos .. ' photos')
    summary('Moon centre vs crop centre (tolerance ' .. Config.verifyTolerance .. ' px):', lines)
  end)
end

return M
