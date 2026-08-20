@echo off
cd /d "%~dp0"
where pythonw >nul 2>nul
if %errorlevel%==0 (
    start "" pythonw "%~dp0gesture_mouse.py"
) else (
    start "" python "%~dp0gesture_mouse.py"
)
