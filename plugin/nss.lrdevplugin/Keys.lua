-- Lightroom develop-setting keys that nss.Color owns: the 24 Color Mixer sliders and the Color Grading wheels (shadows, midtones, highlights, global, blending, balance).
-- `neutral` is what the picture is exported with, so that the GUI can apply them itself.
-- `grading` is the wheel keys alone: a black and white photo (ConvertToGrayscale) has no Color Mixer keys, Lightroom gives it the B&W mixer (GrayMixer...) instead.
-- If a key name is wrong for your Lightroom, EditWithNss stops before changing anything; run "Color: Write develop settings to the log" and fix the name here.
local M = { all = {}, grading = {}, neutral = {} }

local function add(key, neutral)
  table.insert(M.all, key)
  M.neutral[key] = neutral
end

local function addGrading(key, neutral)
  add(key, neutral)
  table.insert(M.grading, key)
end

local bands = { 'Red', 'Orange', 'Yellow', 'Green', 'Aqua', 'Blue', 'Purple', 'Magenta' }
for _, band in ipairs(bands) do
  add('HueAdjustment' .. band, 0)
  add('SaturationAdjustment' .. band, 0)
  add('LuminanceAdjustment' .. band, 0)
end
-- Lightroom still keeps the shadow and highlight hue / saturation under the old split-toning names (read from the settings of a Classic 15.6 photo)
addGrading('SplitToningShadowHue', 0)
addGrading('SplitToningShadowSaturation', 0)
addGrading('ColorGradeShadowLum', 0)
addGrading('ColorGradeMidtoneHue', 0)
addGrading('ColorGradeMidtoneSat', 0)
addGrading('ColorGradeMidtoneLum', 0)
addGrading('SplitToningHighlightHue', 0)
addGrading('SplitToningHighlightSaturation', 0)
addGrading('ColorGradeHighlightLum', 0)
addGrading('ColorGradeGlobalHue', 0)
addGrading('ColorGradeGlobalSat', 0)
addGrading('ColorGradeGlobalLum', 0)
addGrading('ColorGradeBlending', 50)
addGrading('SplitToningBalance', 0)

-- The crop keys "Moon: Center" sets (Lightroom's fractions of the photo in its stored, unrotated orientation) and their full-frame values.
-- The backup holds every develop setting whose name starts with `cropPrefix` (CropConstrain... included); the mixer and grading keys are never touched.
M.crop = { 'CropLeft', 'CropTop', 'CropRight', 'CropBottom', 'CropAngle' }
M.cropFull = { CropLeft = 0, CropTop = 0, CropRight = 1, CropBottom = 1, CropAngle = 0 }
M.cropPrefix = 'Crop'

return M