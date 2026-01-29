@echo off
echo ========================================
echo Voice Clone Studio - Setup Script
echo ========================================
echo.
echo Select CUDA version for PyTorch (press number or wait 10 seconds for default):
echo   1. CUDA 13.0 (latest, for newest GPUs - DEFAULT)
echo   2. CUDA 12.8 (for newer GPUs)
echo   3. CUDA 12.1 (for older GPUs, GTX 10-series and newer)
echo.
choice /C 123 /T 10 /D 1 /M "Enter choice"
set CUDA_CHOICE=%errorlevel%
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

REM Install media processing tools
echo [2/6] Installing media processing tools...
echo Installing SOX...
winget install -e --id ChrisBagwell.SoX --accept-source-agreements --accept-package-agreements
if %errorlevel% neq 0 (
    echo WARNING: SOX installation may have failed.
    echo You can also install manually from: https://sourceforge.net/projects/sox/files/sox/
    echo Or using Chocolatey: choco install sox
)
echo.
echo Installing ffmpeg...
winget install -e --id Gyan.FFmpeg --accept-source-agreements --accept-package-agreements
if %errorlevel% neq 0 (
    echo WARNING: ffmpeg installation may have failed.
    echo You can also install manually from: https://ffmpeg.org/download.html
    echo Or using Chocolatey: choco install ffmpeg
)
echo.

REM Create virtual environment
echo [3/6] Creating virtual environment...
if exist venv (
    echo Virtual environment already exists, skipping...
) else (
    python -m venv venv
    if not exist venv (
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

REM Update pip
echo Updating pip...
python.exe -m pip install --upgrade pip
if %errorlevel% neq 0 (
    echo ERROR: Failed to update pip!
    pause
    exit /b 1
)

REM Install PyTorch
echo [5/6] Installing PyTorch...
setlocal enabledelayedexpansion

if "%CUDA_CHOICE%"=="1" set CUDA_VER=cu130
if "%CUDA_CHOICE%"=="2" set CUDA_VER=cu128
if "%CUDA_CHOICE%"=="3" set CUDA_VER=cu121

if defined CUDA_VER (
    echo Installing PyTorch with !CUDA_VER!...
    echo This may take several minutes...
    pip install torch==2.9.1 torchaudio --index-url https://download.pytorch.org/whl/!CUDA_VER!
    if !errorlevel! neq 0 (
        echo ERROR: Failed to install PyTorch!
        pause
        exit /b 1
    )
) else (
    echo Invalid CUDA choice! Please run setup again.
    pause
    exit /b 1
)
endlocal
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
echo   3. Or use: launch.bat
echo.
pause
