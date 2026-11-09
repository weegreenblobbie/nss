-- Library > Plug-in Extras > nss > Stack: Focus (the work is in Stack.lua)
local LrTasks = import 'LrTasks'
local LrDialogs = import 'LrDialogs'
local Stack = require 'Stack'

LrTasks.startAsyncTask(function()
  local ok, err = LrTasks.pcall(Stack.run, 'focus')
  if not ok then
    Stack.trace('error: ' .. tostring(err))
    LrDialogs.message('nss.Stack', 'Error: ' .. tostring(err), 'critical')
  end
end)
