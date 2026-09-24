@echo off
set SCRIPT_DIR=%~dp0
start "" "%SCRIPT_DIR%.venv\Scripts\pythonw.exe" "%SCRIPT_DIR%run_gui.py"
