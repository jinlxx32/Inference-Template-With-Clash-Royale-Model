@echo off
cd /d "%~dp0"

:: Launch the live scrcpy overlay app and close this command prompt window immediately
start "" pythonw live_app.py
exit
