@echo off
setlocal
title TimeSeries Studio Pro - Exploracao, Anomalias e Previsao

cd /d "%~dp0"

echo =======================================================================
echo   TimeSeries Studio Pro - Suite Analitica de Series Temporais
echo =======================================================================
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo [ERRO] Python nao encontrado no PATH. Instale Python 3.10+ ou ative o venv.
    pause
    exit /b 1
)

echo [1/3] Liberando porta dedicada 8055 (se ocupada)...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8055" ^| findstr "LISTENING"') do (
    taskkill /F /PID %%a >nul 2>nul
)

echo [2/3] Iniciando servidor FastAPI (Uvicorn) em http://127.0.0.1:8055 ...
start "TimeSeries Studio Pro - API (8055)" /min cmd /c "python run.py"

echo [3/3] Aguardando /api/health ficar pronto...
set READY=0
for /L %%i in (1,1,30) do (
    powershell -NoProfile -Command "try { $r = Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:8055/api/health' -TimeoutSec 2; if ($r.StatusCode -eq 200) { exit 0 } else { exit 1 } } catch { exit 1 }" >nul 2>nul
    if not errorlevel 1 (
        set READY=1
        goto :open_browser
    )
    timeout /t 1 /nobreak >nul
)

:open_browser
if "%READY%"=="1" (
    echo [OK] Aplicacao pronta em http://127.0.0.1:8055
    start "" "http://127.0.0.1:8055"
) else (
    echo [AVISO] O servidor demorou para responder. Verifique a janela minimizada do Uvicorn.
)

echo.
echo Pressione qualquer tecla para encerrar esta janela (o servidor continua na janela minimizada).
pause >nul
endlocal
