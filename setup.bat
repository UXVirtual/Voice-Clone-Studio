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

REM Install PyTorch with appropriate CUDA version
echo [5/6] Installing PyTorch...
echo Checking for NVIDIA GPU support...
nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    setlocal enabledelayedexpansion
    echo GPU detected!
    echo.
    echo NVIDIA GPU Info:
    nvidia-smi
    echo.
    
    REM Extract CUDA version from nvidia-smi output
    for /f "tokens=* usebackq" %%i in (`nvidia-smi ^| findstr /i "cuda version"`) do set CUDA_LINE=%%i
    
    REM Parse CUDA version using string replacement
    REM Remove everything up to and including "CUDA Version: "
    set CUDA_VERSION=!CUDA_LINE:*CUDA Version: =!
    REM Take only the first token (version number)
    for /f "tokens=1" %%i in ("!CUDA_VERSION!") do set CUDA_VERSION=%%i
    
    if defined CUDA_VERSION (
        echo Detected CUDA Version: !CUDA_VERSION!
        
        REM Determine which CUDA wheel to use (max cu130)
        REM Extract major version (everything before the dot)
        for /f "tokens=1 delims=." %%i in ("!CUDA_VERSION!") do set CUDA_MAJOR=%%i
        
        REM Extract minor version (everything after the dot)
        for /f "tokens=2 delims=." %%i in ("!CUDA_VERSION!") do set CUDA_MINOR=%%i
        
        if !CUDA_MAJOR! geq 13 (
            set CUDA_WHEEL=cu130
            echo Will use: CUDA 13.0 wheel (detected version is 13.x, capping at max supported)
        ) else if !CUDA_MAJOR! equ 12 (
            set CUDA_WHEEL=cu12!CUDA_MINOR!
            echo Will use: CUDA 12.!CUDA_MINOR! wheel
        ) else (
            set CUDA_WHEEL=cu130
            echo CUDA version too old, will attempt CUDA 13.0
        )
    ) else (
        echo Could not detect CUDA version, will attempt CUDA 13.0
        set CUDA_WHEEL=cu130
    )
    echo.
    
    REM Try detected CUDA version first
    echo Attempting !CUDA_WHEEL! installation...
    echo This may take several minutes...
    pip install torch==2.9.1 torchaudio --index-url https://download.pytorch.org/whl/!CUDA_WHEEL!
    if !errorlevel! equ 0 (
        echo !CUDA_WHEEL! installation successful!
    ) else (
        echo !CUDA_WHEEL! failed, trying CPU-only version...
        pip install torch==2.9.1 torchaudio
        if !errorlevel! neq 0 (
            echo ERROR: Failed to install PyTorch!
            pause
            exit /b 1
        )
    )
    endlocal
) else (
    echo No GPU detected or NVIDIA driver not found, installing CPU-only version...
    echo This may take several minutes...
    pip install torch==2.9.1 torchaudio
    if %errorlevel% neq 0 (
        echo ERROR: Failed to install PyTorch!
        pause
        exit /b 1
    )
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
