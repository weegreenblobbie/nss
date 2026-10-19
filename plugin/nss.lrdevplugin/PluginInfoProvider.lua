-- The Moon section in File > Plug-in Manager > nss: photos per chunk and worker processes (Settings.lua).
local LrPrefs = import 'LrPrefs'
local LrView = import 'LrView'
local Settings = require 'Settings'

return {
  sectionsForTopOfDialog = function(f, propertyTable)
    local prefs = LrPrefs.prefsForPlugin()
    Settings.fillDefaults(prefs)
    local bind = LrView.bind
    local label = LrView.share('nssMoonLabel')
    return {
      {
        title = 'Moon',
        bind_to_object = prefs,
        f:row {
          f:static_text { title = 'Photos per chunk:', alignment = 'right', width = label },
          f:edit_field { value = bind 'moonChunkSize', min = 1, max = 1000, precision = 0, width_in_digits = 5 },
          f:static_text { title = 'exported and measured at a time; each full-size TIFF needs about 270 MB of disk' },
        },
        f:row {
          f:static_text { title = 'Worker processes:', alignment = 'right', width = label },
          f:edit_field { value = bind 'moonWorkers', min = 1, max = 64, precision = 0, width_in_digits = 5 },
          f:static_text { title = 'photos measured at the same time; each needs about 1 GB of memory' },
        },
      },
    }
  end,
}
