@echo off
rem Lightroom runs this as:  nss_icm.bat "<request file>"   (edit the folder below, as in nss_color.bat)
cd /d C:\users\weegr\code\nss
echo %date% %time% start %1 > "%USERPROFILE%\Documents\nss_icm_bat.txt"
"C:\users\weegr\code\nss\venv\scripts\python.exe" -m nss.icm --lightroom %1 >> "%USERPROFILE%\Documents\nss_icm_bat.txt" 2>&1
set code=%errorlevel%
echo exit code %code% >> "%USERPROFILE%\Documents\nss_icm_bat.txt"
exit /b %code%
