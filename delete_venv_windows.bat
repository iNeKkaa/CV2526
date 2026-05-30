@echo off
cd /d "%~dp0"

echo This script will delete the local Python virtual environment: .venv
if not exist ".venv" (
    echo No .venv folder found. Nothing to delete.
    pause
    exit /b 0
)

choice /M "Do you really want to delete .venv"
if errorlevel 2 (
    echo Cancelled. The virtual environment was not deleted.
    pause
    exit /b 0
)

echo Deleting .venv ...
rmdir /s /q ".venv"

if exist ".venv" (
    echo Failed to delete .venv. Close terminals/editors using it and try again.
) else (
    echo Done. The virtual environment has been deleted.
    echo To recreate it later, run setup_windows.bat again.
)

pause
