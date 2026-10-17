-- Writes every develop setting of the selected photo to Documents\nss_develop_settings.txt and says which nss.Color keys are missing.
local LrApplication = import 'LrApplication'
local LrTasks = import 'LrTasks'
local LrDialogs = import 'LrDialogs'
local LrPathUtils = import 'LrPathUtils'
local Keys = require 'Keys'

LrTasks.startAsyncTask(function()
  local photo = LrApplication.activeCatalog():getTargetPhoto()
  if not photo then
    LrDialogs.message('nss.Color', 'Select a photo first.', 'info')
    return
  end
  local settings = photo:getDevelopSettings()
  local names = {}
  for k in pairs(settings) do table.insert(names, k) end
  table.sort(names)

  local missing = {}
  for _, k in ipairs(Keys.all) do
    if settings[k] == nil then table.insert(missing, k) end
  end

  local path = LrPathUtils.child(LrPathUtils.getStandardFilePath('documents'), 'nss_develop_settings.txt')
  local f, err = io.open(path, 'w')
  if not f then
    LrDialogs.message('nss.Color', 'Cannot write ' .. path .. ': ' .. tostring(err), 'critical')
    return
  end
  f:write('missing nss.Color keys: ', #missing == 0 and '(none)' or table.concat(missing, ', '), '\n\n')
  for _, k in ipairs(names) do f:write(k, ' = ', tostring(settings[k]), '\n') end
  f:close()

  if #missing == 0 then
    LrDialogs.message('nss.Color', 'All ' .. #Keys.all .. ' keys exist.\n' .. #names .. ' settings written to\n' .. path, 'info')
  else
    LrDialogs.message('nss.Color', 'Missing keys: ' .. table.concat(missing, ', ') .. '\n\nAll settings written to\n' .. path, 'warning')
  end
end)