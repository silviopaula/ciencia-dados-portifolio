@echo off
setlocal
title App Forecasts - AutoML, ML Recipes, ARIMA e AutoTS (Porta 8060)

cd /d "%~dp0"

echo =======================================================================
echo   App Forecasts - Previsao de Series Temporais (AutoML / ARIMA / AutoTS)
echo =======================================================================
echo.

set "PY_CMD=python"
if exist ".venv\Scripts\python.exe" (
    set "PY_CMD=.venv\Scripts\python.exe"
) else (
    where python >nul 2>nul
    if errorlevel 1 (
        echo [ERRO] Python nao encontrado no PATH e .venv inexistente. Instale Python 3.10+.
        pause
        exit /b 1
    )
)

echo [1/3] Liberando porta dedicada 8060 (se ocupada)...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8060" ^| findstr "LISTENING"') do (
    taskkill /F /PID %%a >nul 2>nul
)

echo [2/3] Iniciando servidor FastAPI (Uvicorn) em http://127.0.0.1:8060 ...
start "App Forecasts - API (8060)" /min cmd /c ""%PY_CMD%" run_app.py"

echo [3/3] Aguardando /api/health ficar pronto...
set READY=0
for /L %%i in (1,1,30) do (
    powershell -NoProfile -Command "try { $r = Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:8060/api/health' -TimeoutSec 2; if ($r.StatusCode -eq 200) { exit 0 } else { exit 1 } } catch { exit 1 }" >nul 2>nul
    if not errorlevel 1 (
        set READY=1
        goto :open_browser
    )
    timeout /t 1 /nobreak >nul
)

:open_browser
if "%READY%"=="1" (
    echo [OK] Aplicacao pronta em http://127.0.0.1:8060
    start "" "http://127.0.0.1:8060"
) else (
    echo [AVISO] O servidor demorou para responder. Verifique a janela minimizada do Uvicorn.
)

echo.
echo Pressione qualquer tecla para encerrar esta janela (o servidor continua na janela minimizada).
pause >nul
endlocal
