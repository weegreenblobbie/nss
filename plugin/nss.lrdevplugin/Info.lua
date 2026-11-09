return {
  LrSdkVersion = 6.0,
  LrSdkMinimumVersion = 6.0,
  LrToolkitIdentifier = 'com.nss.color',
  LrPluginName = 'nss',
  LrLibraryMenuItems = {
    { title = 'Color: Edit...', file = 'EditWithNss.lua' },
    { title = 'Color: Write develop settings to the log', file = 'DumpSettings.lua' },
    { title = 'Moon: Center', file = 'CenterMoon.lua' },
    { title = 'Moon: Verify centering', file = 'VerifyMoon.lua' },
    { title = 'ICM: Intentional Camera Movement', file = 'ICM.lua' },
    { title = 'Stack: Clean (remove people)...', file = 'StackClean.lua' },
    { title = 'Stack: Focus...', file = 'StackFocus.lua' },
  },
  LrPluginInfoProvider = 'PluginInfoProvider.lua',
  VERSION = { major = 1, minor = 0, revision = 0 },
}
