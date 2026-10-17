@echo off
rem Lightroom runs this as:  nss_color.bat "<request file>"   (edit the module name and the folder below)
cd /d C:\users\weegr\code\nss
echo %date% %time% start %1 > "%USERPROFILE%\Documents\nss_color_bat.txt"
"C:\users\weegr\code\nss\venv\scripts\python.exe" color.py --lightroom %1 >> "%USERPROFILE%\Documents\nss_color_bat.txt" 2>&1
set code=%errorlevel%
echo exit code %code% >> "%USERPROFILE%\Documents\nss_color_bat.txt"
exit /b %code%