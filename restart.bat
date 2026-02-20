@echo off
setlocal

call stop.bat
if errorlevel 1 exit /b 1

call start.bat
