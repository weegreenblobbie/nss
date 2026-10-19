-- The moon settings a user changes in File > Plug-in Manager > nss (PluginInfoProvider.lua), kept in Lightroom's preferences
-- for this plug-in.  Config.lua holds their defaults.
local LrPrefs = import 'LrPrefs'
local Config = require 'Config'

local M = {}
M.KEYS = { 'moonChunkSize', 'moonWorkers' }      -- whole numbers, at least 1

-- The setting's value: the Plug-in Manager's when it is a number of at least 1, else Config.lua's default.
function M.get(key)
  local value = LrPrefs.prefsForPlugin()[key]
  if type(value) == 'number' and value >= 1 then return math.floor(value) end
  return Config[key]
end

-- Puts Config.lua's default into each setting not set yet, so the Plug-in Manager shows it.
function M.fillDefaults(prefs)
  for _, key in ipairs(M.KEYS) do
    if prefs[key] == nil then prefs[key] = Config[key] end
  end
end

return M
