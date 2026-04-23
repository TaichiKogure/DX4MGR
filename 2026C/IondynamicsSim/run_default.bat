@echo off
set PYTHONPATH=src
python -m iondynamics.cli run --config configs/default.yaml
pause
