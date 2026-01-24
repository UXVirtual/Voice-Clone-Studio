@echo off
echo ========================================
echo   Qwen3-TTS Voice Clone Studio
echo ========================================
echo.

call venv\Scripts\activate.bat

echo Starting the UI...
echo.
echo The web interface will open in your browser at:
echo   http://127.0.0.1:7860
echo.
echo Press Ctrl+C to stop the server.
echo.

python voice_clone_ui.py

pause
