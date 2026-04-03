@echo off
echo [INFO] Launching Dolphin multi-instance script...
echo [INFO] Script location: %~dp0
echo.

PowerShell.exe -ExecutionPolicy Bypass -File "%~dp0launch_dolphin_instances.ps1"
set EXIT_CODE=%ERRORLEVEL%

if %EXIT_CODE% neq 0 (
    echo.
    echo [ERROR] Script failed with exit code %EXIT_CODE%
    echo [INFO]  Common causes on a new PC:
    echo         - Dolphin.exe not found next to this .bat
    echo         - User folder not found ^(launch Dolphin once first^)
    echo         - ROM not found ^(place in a Jeux or Games folder^)
    echo.
    pause
)

exit /b %EXIT_CODE%
