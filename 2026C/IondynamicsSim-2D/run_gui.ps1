$env:PYTHONPATH = "src"
try {
    python run_gui.py
} catch {
    Write-Host "Python command failed, trying 'py' launcher..."
    py run_gui.py
}
pause
