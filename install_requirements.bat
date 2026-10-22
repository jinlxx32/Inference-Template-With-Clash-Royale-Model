@echo off
title Install Clash Royale Model Studio Dependencies
cd /d "%~dp0"

echo ========================================================
echo   Installing Dependencies for Clash Royale Model Studio
echo ========================================================
echo.

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo.
if %ERRORLEVEL% EQU 0 (
    echo ========================================================
    echo   [SUCCESS] All dependencies installed successfully!
    echo   You can now launch the app using 'run_app.bat'.
    echo ========================================================
) else (
    echo ========================================================
    echo   [ERROR] Failed to install one or more packages.
    echo   Please make sure Python is installed and in your PATH.
    echo ========================================================
)

pause
