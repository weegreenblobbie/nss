@echo off
rem Lightroom runs this as:  nss_moon.bat "<request file>"   (edit the folder below, as in nss_color.bat)
cd /d C:\users\weegr\code\nss
echo %date% %time% start %1 > "%USERPROFILE%\Documents\nss_moon_bat.txt"
"C:\users\weegr\code\nss\venv\scripts\python.exe" moon.py --request %1 >> "%USERPROFILE%\Documents\nss_moon_bat.txt" 2>&1
set code=%errorlevel%
echo exit code %code% >> "%USERPROFILE%\Documents\nss_moon_bat.txt"
exit /b %code%