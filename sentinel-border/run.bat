@echo off
:: SentinelBorder — One-click launch script (Windows / uv)
:: Requires: uv installed (https://docs.astral.sh/uv/)

title SentinelBorder — Triage System
color 0B

echo.
echo  ====================================================================
echo   SENTINELBORDER v1.0 ^| SIH 26188 ^| MHA / SSB
echo   Autonomous Document Screening ^& Biometric Triage System
echo  ====================================================================
echo.

:: ── Check uv ──────────────────────────────────────────────────────────
where uv >nul 2>&1
if %errorlevel% NEQ 0 (
  echo  [ERROR] 'uv' not found. Install from https://docs.astral.sh/uv/
  pause
  exit /b 1
)
echo  [OK] uv found.

:: ── Create / verify venv ──────────────────────────────────────────────
if not exist ".venv" (
  echo  [SETUP] Creating virtual environment with Python 3.11...
  uv venv --python 3.11
)
echo  [OK] Virtual environment ready.

:: ── Install dependencies ───────────────────────────────────────────────
echo  [SETUP] Installing Python dependencies (this may take a while on first run)...
uv pip install -r backend\requirements.txt
if %errorlevel% NEQ 0 (
  echo  [ERROR] Dependency installation failed. See output above.
  pause
  exit /b 1
)
echo  [OK] Dependencies installed.

:: ── Check Tesseract (optional) ─────────────────────────────────────────
where tesseract >nul 2>&1
if %errorlevel% NEQ 0 (
  echo  [WARN] Tesseract OCR not found in PATH.
  echo         Install from: https://github.com/UB-Mannheim/tesseract/wiki
  echo         PaddleOCR and PassportEye will still function.
)

:: ── Load .env for GEMINI_API_KEY (optional) ─────────────────────────────
set "ENV_FILE="
if exist ".env" set "ENV_FILE=.env"
if exist "backend\.env" set "ENV_FILE=backend\.env"

if defined ENV_FILE (
  echo  [INFO] Loading environment variables from %ENV_FILE%...
  for /f "usebackq tokens=1,* delims==" %%A in ("%ENV_FILE%") do (
    if not "%%A"=="" if not "%%B"=="" (
      :: Strip spaces from key (lazy way: just check the main keys we care about)
      echo %%A | findstr /C:"GEMINI_API_KEY" >nul
      if not errorlevel 1 (
        set "GEMINI_API_KEY=%%B"
      )
      echo %%A | findstr /C:"STRUCTURED_OCR_PROVIDER" >nul
      if not errorlevel 1 (
        set "STRUCTURED_OCR_PROVIDER=%%B"
      )
    )
  )
)

:: Clean up the value in case it has leading spaces (e.g. GEMINI_API_KEY = key)
for /f "tokens=* delims= " %%a in ("%GEMINI_API_KEY%") do set "GEMINI_API_KEY=%%a"
for /f "tokens=* delims= " %%a in ("%STRUCTURED_OCR_PROVIDER%") do set "STRUCTURED_OCR_PROVIDER=%%a"

:: Show selected non-MRZ vision provider
if /I "%STRUCTURED_OCR_PROVIDER%"=="ollama" (
  echo  [OK] Ollama Vision selected — ensure Ollama is running locally.
) else if "%GEMINI_API_KEY%"=="" (
  echo  [ERROR] GEMINI_API_KEY not set.
  echo          Set STRUCTURED_OCR_PROVIDER=ollama or configure GEMINI_API_KEY.
) else (
  echo  [OK] GEMINI_API_KEY found — Gemini Vision OCR enabled.
)


:: ── Launch server ──────────────────────────────────────────────────────
echo.
echo  [START] Launching SentinelBorder on http://127.0.0.1:8000
echo  [INFO]  Open your browser to: http://127.0.0.1:8000
echo  [INFO]  Press Ctrl+C to stop.
echo.
.venv\Scripts\uvicorn.exe backend.app:app --host 127.0.0.1 --port 8000 --reload --app-dir .

pause
