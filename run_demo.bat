@echo off
echo =============================================
echo  BioHybridKG Demo App — Starting Server
echo =============================================

cd /d "%~dp0"

echo [1/2] Installing dependencies...
pip install fastapi "uvicorn[standard]" python-multipart requests >nul 2>&1

echo [2/2] Starting FastAPI server...
echo.
echo  Open in browser: http://127.0.0.1:8000
echo.
python -m uvicorn app.backend.main:app --reload --host 127.0.0.1 --port 8000

pause
