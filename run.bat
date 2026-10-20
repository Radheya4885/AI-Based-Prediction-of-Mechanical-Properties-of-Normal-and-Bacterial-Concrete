@echo off
cd /d "%~dp0"
title Concrete Strength Predictor

echo ========================================================
echo   Concrete Strength Predictor - Web Application
echo ========================================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found in .venv.
    echo Please make sure the setup completed properly.
    pause
    exit /b 1
)

echo Initializing AI models (TabPFN, Random Forest, XGBoost)...
echo The web browser will open automatically at:
echo   http://127.0.0.1:8000/
echo.

:: Launch browser in background after 2 seconds
start /b cmd /c "timeout /t 2 /nobreak >nul & start http://127.0.0.1:8000/"

:: Start the server
.venv\Scripts\python.exe src\api\app.py

pause
