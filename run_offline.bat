@echo off
title PerceptionGuard Local Offline Server
cd /d "%~dp0frontend"
echo ========================================================
echo   PERCEPTIONGUARD: 3D OFFLINE FRONTEND LAUNCHER
echo ========================================================
echo.
echo Launching local server on http://localhost:3000 ...
echo Opening your web browser automatically...
echo.
start http://localhost:3000
python -m http.server 3000
pause
