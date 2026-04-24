@echo off
cd /d "%~dp0"
echo Starting Omini Vaani System...
uv run python start_vaani_web.py
pause
