-- What the menu items share (EditWithNss.lua, Moon.lua, ICM.lua): the trace file, the key=value hand-over files, applying develop settings,
-- quoting a command's paths and the TIFF export settings.
local LrLogger = import 'LrLogger'
local LrPathUtils = import 'LrPathUtils'

local M = {}

M.logger = LrLogger('nssColor')
M.logger:enable('logfile')

M.tracePath = LrPathUtils.child(LrPathUtils.getStandardFilePath('documents'), 'nss_color_trace.txt')

-- Appends a line to the trace file (and Lightroom's log).
function M.trace(msg)
  local f = io.open(M.tracePath, 'a')
  if f then
    f:write(os.date('%H:%M:%S '), msg, '\n')
    f:close()
  end
  M.logger:info(msg)
end

-- Writes lines to path; false and the error when it cannot.
function M.writeLines(path, lines)
  local f, err = io.open(path, 'w')
  if not f then return false, err end
  f:write(table.concat(lines, '\n'), '\n')
  f:close()
  return true
end

-- The key=value lines of a file as strings, nil when it cannot be read.
function M.readValues(path)
  local f = io.open(path, 'r')
  if not f then return nil end
  local values = {}
  for line in f:lines() do
    local k, v = line:match('^%s*([%w_.]+)%s*=%s*(.-)%s*$')
    if k then values[k] = v end
  end
  f:close()
  return values
end

-- Applies develop settings to a photo as one undo step called name.
function M.applyTo(catalog, photo, settings, name)
  catalog:withWriteAccessDo(name, function()
    photo:applyDevelopSettings(settings, name)
  end, { timeout = 10 })
end

function M.quote(s) return '"' .. s .. '"' end

-- Export settings of a full-size, uncompressed 16-bit Adobe RGB TIFF into folder, every edit (crop included) baked in, not re-imported.
-- A new table each call: the caller adds how the file is named (LR_collisionHandling, LR_renamingTokensOn, LR_tokens).
function M.tiffExportSettings(folder)
  return {
    LR_format = 'TIFF',
    LR_export_colorSpace = 'AdobeRGB',
    LR_export_bitDepth = 16,
    LR_tiff_compressionMethod = 'compressionMethod_None',
    LR_export_destinationType = 'specificFolder',
    LR_export_destinationPathPrefix = folder,
    LR_export_useSubfolder = false,
    LR_size_doConstrain = false,
    LR_outputSharpeningOn = false,
    LR_reimportExportedPhoto = false,
    LR_useWatermark = false,
  }
end

return M
