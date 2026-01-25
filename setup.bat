@echo off
echo ========================================
echo Voice Clone Studio - Setup Script
echo ========================================
echo.

REM Check Python version
echo [1/6] Checking Python installation...
python --version
if %errorlevel% neq 0 (
    echo ERROR: Python not found! Please install Python 3.12+ first.
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)
echo.

REM Install SOX using winget
echo [2/6] Installing SOX (audio processing library)...
echo Installing via winget...
winget install -e --id ChrisBagwell.SoX --accept-source-agreements --accept-package-agreements
if %errorlevel% neq 0 (
    echo WARNING: SOX installation may have failed.
    echo You can also install manually from: https://sourceforge.net/projects/sox/files/sox/
    echo Or using Chocolatey: choco install sox
)
echo.

REM Create virtual environment
echo [3/6] Creating virtual environment...
if exist venv (
    echo Virtual environment already exists, skipping...
) else (
    python -m venv venv
    if %errorlevel% neq 0 (
        echo ERROR: Failed to create virtual environment!
        pause
        exit /b 1
    )
    echo Virtual environment created successfully!
)
echo.

REM Activate virtual environment
echo [4/6] Activating virtual environment...
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo ERROR: Failed to activate virtual environment!
    pause
    exit /b 1
)
echo.

REM Install PyTorch with CUDA support
echo [5/6] Installing PyTorch with CUDA 13.0 support...
echo This may take several minutes...
pip install torch==2.9.1 torchaudio --index-url https://download.pytorch.org/whl/cu130
if %errorlevel% neq 0 (
    echo ERROR: Failed to install PyTorch!
    pause
    exit /b 1
)
echo.

REM Install requirements
echo [6/6] Installing requirements...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install requirements!
    pause
    exit /b 1
)
echo.

echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo Python version being used:
python --version
echo.
echo OPTIONAL: Install Flash Attention 2 for better performance
echo.
echo Option 1 - Build from source (requires C++ compiler):
echo   pip install flash-attn --no-build-isolation
echo.
echo Option 2 - Use prebuilt wheel (faster, no compiler needed):
echo   Visit: https://github.com/bdashore3/flash-attention/releases
echo   Download the wheel matching your Python version
echo   Then: pip install downloaded-wheel-file.whl
echo.
echo ========================================
echo.
echo To launch Voice Clone Studio:
echo   1. Make sure virtual environment is activated: venv\Scripts\activate
echo   2. Run: python Voice_Clone_Studio.py
echo   3. Or use: Launch_UI.bat
echo.
pause
