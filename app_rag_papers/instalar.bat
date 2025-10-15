@echo off
REM instalar.bat - Instala dependências e prepara ambiente

REM Cria ambiente virtual Python
python -m venv .venv
if exist .venv\Scripts\activate (
    call .venv\Scripts\activate
    echo Ambiente virtual ativado.
) else (
    echo Falha ao criar ambiente virtual.
    exit /b 1
)

REM Atualiza pip e instala dependências
pip install --upgrade pip
pip install -r requirements.txt

REM Verifica Docker Compose
where docker-compose >nul 2>nul
if %errorlevel% neq 0 (
    echo Por favor, instale o Docker Compose manualmente.
) else (
    echo Docker Compose já instalado.
)

REM Baixa imagens Docker necessárias
docker compose pull

echo Instalacao concluida!
pause
