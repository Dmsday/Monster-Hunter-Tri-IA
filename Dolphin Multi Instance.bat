@echo off
PowerShell.exe -ExecutionPolicy Bypass -File "%~dp0launch_dolphin_instances.ps1"
REM Exit immediately after PowerShell finishes (no timeout needed)
REM PowerShell script handles its own auto-close behavior
exit /b 0