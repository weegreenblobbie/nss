-- Lightroom develop-setting keys that nss.Color owns: the 24 Color Mixer sliders and the Color Grading wheels (shadows, midtones, highlights, global, blending, balance).
-- `neutral` is what the picture is exported with, so that the GUI can apply them itself.
-- If a key name is wrong for your Lightroom, EditWithNss stops before changing anything; run "write develop settings to the log" and fix the name here.
local M = { all = {}, neutral = {} }

local function add(key, neutral)
  table.insert(M.all, key)
  M.neutral[key] = neutral
end

local bands = { 'Red', 'Orange', 'Yellow', 'Green', 'Aqua', 'Blue', 'Purple', 'Magenta' }
for _, band in ipairs(bands) do
  add('HueAdjustment' .. band, 0)
  add('SaturationAdjustment' .. band, 0)
  add('LuminanceAdjustment' .. band, 0)
end
-- Lightroom still keeps the shadow and highlight hue / saturation under the old split-toning names (read from the settings of a Classic 15.6 photo)
add('SplitToningShadowHue', 0)
add('SplitToningShadowSaturation', 0)
add('ColorGradeShadowLum', 0)
add('ColorGradeMidtoneHue', 0)
add('ColorGradeMidtoneSat', 0)
add('ColorGradeMidtoneLum', 0)
add('SplitToningHighlightHue', 0)
add('SplitToningHighlightSaturation', 0)
add('ColorGradeHighlightLum', 0)
add('ColorGradeGlobalHue', 0)
add('ColorGradeGlobalSat', 0)
add('ColorGradeGlobalLum', 0)
add('ColorGradeBlending', 50)
add('SplitToningBalance', 0)

return M