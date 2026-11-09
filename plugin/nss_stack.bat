@echo off
rem Lightroom runs this as:  nss_stack.bat "<request file>"   (edit the folder below, as in nss_color.bat)
cd /d C:\users\weegr\code\nss
echo %date% %time% start %1 > "%USERPROFILE%\Documents\nss_stack_bat.txt"
"C:\users\weegr\code\nss\venv\scripts\python.exe" focus_stack.py --lightroom %1 >> "%USERPROFILE%\Documents\nss_stack_bat.txt" 2>&1
set code=%errorlevel%
echo exit code %code% >> "%USERPROFILE%\Documents\nss_stack_bat.txt"
exit /b %code%
