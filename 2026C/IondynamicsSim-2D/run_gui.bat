@echo off
set PYTHONPATH=src
python run_gui.py
if %errorlevel% neq 0 (
    echo.
    echo Python command failed, trying 'py' launcher...
    py run_gui.py
)
pause
