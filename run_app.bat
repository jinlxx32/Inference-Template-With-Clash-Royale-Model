@echo off
cd /d "%~dp0"

:: Launch the application in the background and close this command prompt window immediately
start "" pythonw app.py
exit
