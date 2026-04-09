@echo off
echo [INFO] Launching Dolphin multi-instance script...
echo [INFO] Script location: %~dp0
echo.

:: Use full system path — PowerShell.exe may not be in PATH on all systems
set "PS_EXE=%SystemRoot%\System32\WindowsPowerShell\v1.0\PowerShell.exe"
if not exist "%PS_EXE%" set "PS_EXE=%SystemRoot%\SysWOW64\WindowsPowerShell\v1.0\PowerShell.exe"
if not exist "%PS_EXE%" (
    echo [ERROR] PowerShell not found at expected locations:
    echo         %SystemRoot%\System32\WindowsPowerShell\v1.0\PowerShell.exe
    echo         %SystemRoot%\SysWOW64\WindowsPowerShell\v1.0\PowerShell.exe
    pause
    exit /b 1
)
"%PS_EXE%" -ExecutionPolicy Bypass -File "%~dp0launch_dolphin_instances.ps1"
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