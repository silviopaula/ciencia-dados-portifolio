@echo off
title RAG - Assistente de Artigos Cientificos
color 0B

:menu
cls
echo ========================================
echo   RAG - Assistente de Artigos
echo ========================================
echo.

REM Verificar status dos containers
 docker ps --filter "name=app_rag_papers" --format "{{.Names}}" 2>nul | findstr "app_rag_papers" >nul 2>&1
if errorlevel 1 (
    set STATUS=PARADO
    set STATUS_COLOR=0C
) else (
    set STATUS=RODANDO
    set STATUS_COLOR=0A
)

color %STATUS_COLOR%
echo Status: %STATUS%
color 0B
echo.
echo ========================================
echo.
echo [1] Iniciar RAG
echo [2] Parar RAG
echo [3] Reiniciar RAG
echo [4] Status completo
echo [5] Ver logs
echo.
echo [I] Instalar (primeira vez)
echo [G] Configurar GPU
echo [L] Limpar e otimizar
echo.
echo [A] Abrir no navegador
echo [0] Sair
echo.
echo ========================================
set /p opcao="Escolha uma opcao: "

if "%opcao%"=="1" goto iniciar
if "%opcao%"=="2" goto parar
if "%opcao%"=="3" goto reiniciar
if "%opcao%"=="4" goto status
if "%opcao%"=="5" goto logs
if /i "%opcao%"=="I" goto instalar
if /i "%opcao%"=="G" goto gpu
if /i "%opcao%"=="L" goto limpar
if /i "%opcao%"=="A" goto abrir
if "%opcao%"=="0" exit
goto menu

:iniciar
cls
echo ========================================
echo   Iniciando RAG
echo ========================================
echo.

REM Verificar Docker
echo [1/7] Verificando Docker...
docker info >nul 2>&1
if errorlevel 1 (
    echo ERRO: Docker nao esta rodando!
    echo Inicie o Docker Desktop primeiro.
    pause
    goto menu
)
echo OK
timeout /t 1 /nobreak >nul

REM Verificar GPU
echo.
echo [2/7] Verificando GPU...
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo CPU mode
    set USE_GPU=0
) else (
    docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi >nul 2>&1
    if errorlevel 1 (
        echo GPU detectada mas nao acessivel no Docker
        echo Use opcao [G] para configurar
        set USE_GPU=0
    ) else (
        echo GPU NVIDIA detectada
        set USE_GPU=1
    )
)
timeout /t 1 /nobreak >nul

REM Parar containers antigos
echo.
echo [3/7] Verificando containers antigos...
docker-compose down >nul 2>&1
echo OK
timeout /t 1 /nobreak >nul

REM Iniciar
echo.
echo [4/7] Iniciando containers...
docker-compose up -d
if errorlevel 1 (
    echo ERRO ao iniciar!
    pause
    goto menu
)
echo OK
timeout /t 5 /nobreak >nul

REM Aguardar Ollama
echo.
echo [5/7] Aguardando LLM...
set /a tentativas=0
:loop_start
set /a tentativas+=1
curl -s http://localhost:11434/api/tags >nul 2>&1
if errorlevel 1 (
    if %tentativas% LSS 30 (
        timeout /t 2 /nobreak >nul
        goto loop_start
    ) else (
        echo TIMEOUT!
        pause
        goto menu
    )
)
echo OK

REM Verificar modelo
echo.
echo [6/7] Verificando modelo...
docker exec app_rag_papers-ollama-1 ollama list 2>nul | findstr /C:"qwen" >nul 2>&1
if errorlevel 1 (
    echo Modelo nao encontrado. Baixando qwen2.5:7b...
    docker exec app_rag_papers-ollama-1 ollama pull qwen2.5:7b
    if errorlevel 1 (
        echo ERRO: Falha ao baixar modelo!
        pause
        goto menu
    ) else (
        echo OK - Modelo baixado com sucesso!
    )
) else (
    echo OK - Modelo ja instalado
)

REM Status final
echo.
echo [7/7] Verificando aplicacao...
timeout /t 3 /nobreak >nul
curl -s http://localhost:8501 >nul 2>&1
if errorlevel 1 (
    echo AVISO: App pode nao estar pronto ainda
) else (
    echo OK
)

echo.
echo ========================================
if %USE_GPU%==1 (
    echo   RAG Iniciado com GPU
) else (
    echo   RAG Iniciado com CPU
)
echo ========================================
echo.
echo Abrindo navegador...
timeout /t 2 /nobreak >nul
start http://localhost:8501
echo.
echo Pressione qualquer tecla para voltar ao menu...
pause >nul
goto menu

:parar
cls
echo ========================================
echo   Parando RAG
echo ========================================
echo.
docker-compose down
if errorlevel 1 (
    echo ERRO ao parar!
) else (
    echo.
    echo RAG parado com sucesso!
)
echo.
timeout /t 2 /nobreak >nul
goto menu

:reiniciar
cls
echo ========================================
echo   Reiniciando RAG
echo ========================================
echo.
echo Parando...
docker-compose down >nul 2>&1
timeout /t 2 /nobreak >nul
echo Iniciando...
docker-compose up -d
timeout /t 10 /nobreak >nul
echo.
echo RAG reiniciado!
echo.
timeout /t 2 /nobreak >nul
goto menu

:status
cls
echo ========================================
echo   Status Completo
echo ========================================
echo.
echo CONTAINERS:
 docker ps -a --filter "name=app_rag_papers"
echo.
echo IMAGENS:
 docker images | findstr "app_rag_papers\|ollama\|REPOSITORY"
echo.
echo MODELOS LLM:
 docker exec app_rag_papers-ollama-1 ollama list 2>nul
echo.
echo USO DE ESPACO:
docker system df
echo.
echo GPU:
nvidia-smi 2>nul || echo Nao disponivel
echo.
pause
goto menu

:logs
cls
echo ========================================
echo   Logs (Ctrl+C para sair)
echo ========================================
echo.
docker-compose logs -f
goto menu

:instalar
cls
echo ========================================
echo   Instalacao Completa
echo ========================================
echo.
echo Redirecionando para instalador...
timeout /t 2 /nobreak >nul
call INSTALAR-RAG.bat
goto menu

:gpu
cls
echo ========================================
echo   Configurar GPU
echo ========================================
echo.
echo Redirecionando para configuracao GPU...
timeout /t 2 /nobreak >nul
call HABILITAR-GPU.bat
goto menu

:limpar
cls
echo ========================================
echo   Limpar e Otimizar
echo ========================================
echo.
echo Redirecionando para limpeza...
timeout /t 2 /nobreak >nul
call LIMPAR-E-REBUILD.bat
goto menu

:abrir
start http://localhost:8501
echo.
echo Abrindo navegador...
timeout /t 2 /nobreak >nul
goto menu