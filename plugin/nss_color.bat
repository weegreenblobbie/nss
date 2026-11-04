@echo off
rem Lightroom runs this as:  nss_color.bat "<request file>"   (edit the module name and the folder below)
cd /d C:\users\weegr\code\nss
rem one OpenBLAS thread per call: the explore grid grades on 8 threads and OpenBLAS crashes (access violation) when they all call it with its own thread pool
set OPENBLAS_NUM_THREADS=1
echo %date% %time% start %1 > "%USERPROFILE%\Documents\nss_color_bat.txt"
"C:\users\weegr\code\nss\venv\scripts\python.exe" color.py --lightroom %1 >> "%USERPROFILE%\Documents\nss_color_bat.txt" 2>&1
set code=%errorlevel%
echo exit code %code% >> "%USERPROFILE%\Documents\nss_color_bat.txt"
exit /b %code%