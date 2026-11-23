@echo off
chcp 65001 >nul
title TimeSeries Studio Pro — Servidor Web

cd /d "%~dp0"

echo ====================================================================
echo   TimeSeries Studio Pro — Suite Unificada de Series Temporais
echo ====================================================================
echo.
echo [1/3] Encerrando sessoes anteriores nas portas 8000 e 8055...

REM Encerra qualquer processo que esteja escutando nas portas 8000 ou 8055
for %%p in (8000 8055) do (
    for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":%%p " ^| findstr "LISTENING"') do (
        if not "%%a"=="0" (
            echo       Finalizando processo anterior na porta %%p ^(PID: %%a^)...
            taskkill /F /T /PID %%a >nul 2>&1
        )
    )
)

REM Pequena pausa para garantir a liberacao da porta pelo Windows
timeout /t 1 /nobreak >nul

echo [2/3] Abrindo navegador em http://127.0.0.1:8055 ...
start "" /b cmd /c "timeout /t 2 /nobreak >nul & start http://127.0.0.1:8055"

echo [3/3] Iniciando nova sessao do servidor Python...
echo.
python run.py

if %errorlevel% neq 0 (
    echo.
    echo [ERRO] Falha ao iniciar o aplicativo. Verifique se o Python esta instalado no PATH.
    pause
)
