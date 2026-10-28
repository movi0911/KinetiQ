@echo off
title PerceptionGuard Unified Application (Frontend + Backend)
cd /d "%~dp0"
echo ======================================================================
echo    PERCEPTIONGUARD: UNIFIED 3D FRONTEND + AI BACKEND
echo ======================================================================
echo.
echo [1/2] Starting local Python AI server on http://localhost:8000 ...
echo [2/2] Launching your web browser automatically...
echo.
start http://localhost:8000
python server.py
pause
