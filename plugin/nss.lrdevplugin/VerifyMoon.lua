-- Library > Plug-in Extras > nss > Moon: Verify centering (the work is in Moon.lua)
local LrTasks = import 'LrTasks'
local LrDialogs = import 'LrDialogs'
local Moon = require 'Moon'

LrTasks.startAsyncTask(function()
  local ok, err = LrTasks.pcall(Moon.verify)
  if not ok then
    Moon.trace('error: ' .. tostring(err))
    LrDialogs.message('nss.Moon', 'Error: ' .. tostring(err), 'critical')
  end
end)
