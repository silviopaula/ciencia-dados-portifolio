@echo off
chcp 65001 >nul
title App Forecasts - Python + Modern Frontend (Porta 8060)
cd /d "%~dp0"

echo ====================================================================
echo   Verificando e encerrando processos anteriores na porta 8060...
echo ====================================================================
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8060" ^| findstr "LISTENING"') do (
    echo Encerrando PID %%a na porta 8060...
    taskkill /F /PID %%a >nul 2>&1
)

echo.
echo ====================================================================
echo   Iniciando App Forecasts (FastAPI Backend + Modern Web Frontend)
echo   URL: http://127.0.0.1:8060
echo ====================================================================
echo.

:: Abre o navegador automaticamente apos 2 segundos em background
start "" cmd /c "timeout /t 2 /nobreak >nul && start http://127.0.0.1:8060"

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" run_app.py
) else (
    python run_app.py
)

pause
