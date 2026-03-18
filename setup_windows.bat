@echo off
echo ============================================
echo  Vehicle Counter — One-click Setup (Windows)
echo ============================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found!
    echo Please install Python 3.10 from https://python.org
    echo Make sure to tick "Add Python to PATH" during install.
    pause
    exit /b 1
)

echo [OK] Python found.
echo.
echo Installing required packages...
echo (This may take 2-5 minutes on first run)
echo.

pip install --upgrade pip
pip install ultralytics deep-sort-realtime opencv-python numpy pandas matplotlib flask Pillow

echo.
echo ============================================
echo  Setup complete!
echo  Run the project: python main.py
echo ============================================
pause
