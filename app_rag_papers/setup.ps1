Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "  Setup RAG com Docker, Ollama e GPU" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""

# 0. Verificar GPU
Write-Host "[0/7] Verificando GPU NVIDIA..." -ForegroundColor Yellow
try {
    docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi 2>&1 | Out-Null
    Write-Host "OK - GPU detectada" -ForegroundColor Green
    $useGPU = $true
} catch {
    Write-Host "AVISO - GPU não detectada, usando CPU" -ForegroundColor Yellow
    $useGPU = $false
}
Start-Sleep -Seconds 2

# 1. Para containers existentes
Write-Host "[1/7] Parando containers existentes..." -ForegroundColor Yellow
docker-compose down
Write-Host "OK" -ForegroundColor Green
Start-Sleep -Seconds 2

# 2. Build das imagens
Write-Host "[2/7] Construindo imagens Docker..." -ForegroundColor Yellow
docker-compose build
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERRO no build!" -ForegroundColor Red
    exit 1
}
Write-Host "OK" -ForegroundColor Green
Start-Sleep -Seconds 2

# 3. Sobe os containers
Write-Host "[3/7] Subindo containers..." -ForegroundColor Yellow
docker-compose up -d
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERRO ao subir containers!" -ForegroundColor Red
    exit 1
}
Write-Host "OK" -ForegroundColor Green
Start-Sleep -Seconds 5

# 4. Aguarda Ollama ficar pronto
Write-Host "[4/7] Aguardando Ollama inicializar..." -ForegroundColor Yellow
$maxAttempts = 30
$attempt = 0
$ollamaReady = $false

while ($attempt -lt $maxAttempts -and -not $ollamaReady) {
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -TimeoutSec 2 -ErrorAction SilentlyContinue
        if ($response.StatusCode -eq 200) {
            $ollamaReady = $true
        }
    } catch {
        $attempt++
        Write-Host "  Tentativa $attempt/$maxAttempts..." -ForegroundColor Gray
        Start-Sleep -Seconds 2
    }
}

if (-not $ollamaReady) {
    Write-Host "ERRO: Ollama não inicializou!" -ForegroundColor Red
    exit 1
}
Write-Host "OK" -ForegroundColor Green

# 5. Verificar GPU no Ollama
if ($useGPU) {
    Write-Host "[5/7] Verificando GPU no Ollama..." -ForegroundColor Yellow
    try {
        docker exec app_rag_did-ollama-1 nvidia-smi | Out-Null
        Write-Host "OK - Ollama com GPU habilitada" -ForegroundColor Green
    } catch {
        Write-Host "AVISO - GPU não acessível no Ollama" -ForegroundColor Yellow
    }
} else {
    Write-Host "[5/7] GPU não disponível, usando CPU" -ForegroundColor Yellow
}

# 6. Baixa modelo qwen2.5:7b
Write-Host "[6/7] Baixando modelo qwen2.5:7b (~4.7GB)..." -ForegroundColor Yellow
if ($useGPU) {
    Write-Host "  Com GPU: Respostas em 5-10 segundos" -ForegroundColor Cyan
} else {
    Write-Host "  Sem GPU: Respostas em 30-60 segundos" -ForegroundColor Gray
}
docker exec app_rag_did-ollama-1 ollama pull qwen2.5:7b
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERRO ao baixar qwen2.5:7b!" -ForegroundColor Red
    exit 1
}
Write-Host "OK" -ForegroundColor Green

# 7. Verifica modelos instalados
Write-Host "[7/7] Modelos instalados:" -ForegroundColor Yellow
docker exec app_rag_did-ollama-1 ollama list

Write-Host ""
Write-Host "=====================================" -ForegroundColor Cyan
if ($useGPU) {
    Write-Host "  Setup concluído com GPU" -ForegroundColor Green
} else {
    Write-Host "  Setup concluído (CPU)" -ForegroundColor Green
}
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Acesse: http://localhost:8501" -ForegroundColor Cyan
Write-Host ""
if ($useGPU) {
    Write-Host "Configurações recomendadas (GPU):" -ForegroundColor Yellow
    Write-Host "  Chunk size: 1200" -ForegroundColor Gray
    Write-Host "  Overlap: 200" -ForegroundColor Gray
    Write-Host "  k_docs: 10" -ForegroundColor Gray
    Write-Host "  Modelo: qwen2.5:7b" -ForegroundColor Gray
    Write-Host ""
    Write-Host "Para modelo maior (14B - limite de 6GB):" -ForegroundColor Yellow
    Write-Host "  docker exec app_rag_did-ollama-1 ollama pull qwen2.5:14b" -ForegroundColor Gray
} else {
    Write-Host "Para habilitar GPU:" -ForegroundColor Yellow
    Write-Host "  1. Instale NVIDIA Container Toolkit" -ForegroundColor Gray
    Write-Host "  2. Execute novamente o setup" -ForegroundColor Gray
}
Write-Host ""
Write-Host "Comandos úteis:" -ForegroundColor Yellow
Write-Host "  Ver logs:       docker-compose logs -f" -ForegroundColor Gray
Write-Host "  Parar tudo:     docker-compose down" -ForegroundColor Gray
Write-Host "  Reiniciar:      docker-compose restart" -ForegroundColor Gray
if ($useGPU) {
    Write-Host "  Monitorar GPU:  docker exec app_rag_did-ollama-1 watch -n 1 nvidia-smi" -ForegroundColor Gray
    Write-Host "  Testar GPU:     .\test-gpu.ps1" -ForegroundColor Gray
}
Write-Host ""Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "  Setup RAG com Docker, Ollama e GPU" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""

# 0. Verificar GPU
Write-Host "[0/7] Verificando GPU NVIDIA..." -ForegroundColor Yellow
try {
    docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi 2>&1 | Out-Null
    Write-Host "OK - GPU detectada! 🚀" -ForegroundColor Green
    $useGPU = $true
} catch {
    Write-Host "AVISO - GPU não detectada, usando CPU" -ForegroundColor Yellow
    $useGPU = $false
}
Start-Sleep -Seconds 2

# 1. Para containers existentes
Write-Host "[1/7] Parando containers existentes..." -ForegroundColor Yellow
docker-compose down
Write-Host "OK" -ForegroundColor Green
Start-Sleep -Seconds 2

# 2. Build das imagens
Write-Host "[2/7] Construindo imagens Docker..." -ForegroundColor Yellow
docker-compose build
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERRO no build!" -ForegroundColor Red
    exit 1
}
Write-Host "OK" -ForegroundColor Green
Start-Sleep -Seconds 2

# 3. Sobe os containers
Write-Host "[3/7] Subindo containers..." -ForegroundColor Yellow
docker-compose up -d
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERRO ao subir containers!" -ForegroundColor Red
    exit 1
}
Write-Host "OK" -ForegroundColor Green
Start-Sleep -Seconds 5

# 4. Aguarda Ollama ficar pronto
Write-Host "[4/7] Aguardando Ollama inicializar..." -ForegroundColor Yellow
$maxAttempts = 30
$attempt = 0
$ollamaReady = $false

while ($attempt -lt $maxAttempts -and -not $ollamaReady) {
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -TimeoutSec 2 -ErrorAction SilentlyContinue
        if ($response.StatusCode -eq 200) {
            $ollamaReady = $true
        }
    } catch {
        $attempt++
        Write-Host "  Tentativa $attempt/$maxAttempts..." -ForegroundColor Gray
        Start-Sleep -Seconds 2
    }
}

if (-not $ollamaReady) {
    Write-Host "ERRO: Ollama não inicializou!" -ForegroundColor Red
    exit 1
}
Write-Host "OK" -ForegroundColor Green

# 5. Verificar GPU no Ollama
if ($useGPU) {
    Write-Host "[5/7] Verificando GPU no Ollama..." -ForegroundColor Yellow
    try {
        docker exec app_rag_did-ollama-1 nvidia-smi | Out-Null
        Write-Host "OK - Ollama com GPU habilitada! 🚀" -ForegroundColor Green
    } catch {
        Write-Host "AVISO - GPU não acessível no Ollama" -ForegroundColor Yellow
    }
} else {
    Write-Host "[5/7] GPU não disponível, usando CPU" -ForegroundColor Yellow
}

# 6. Baixa modelo qwen2.5:7b
Write-Host "[6/7] Baixando modelo qwen2.5:7b (~4.7GB)..." -ForegroundColor Yellow
if ($useGPU) {
    Write-Host "  Com GPU: Respostas em 5-10 segundos! 🚀" -ForegroundColor Cyan
} else {
    Write-Host "  Sem GPU: Respostas em 30-60 segundos" -ForegroundColor Gray
}
docker exec app_rag_did-ollama-1 ollama pull qwen2.5:7b
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERRO ao baixar qwen2.5:7b!" -ForegroundColor Red
    exit 1
}
Write-Host "OK" -ForegroundColor Green

# 7. Verifica modelos instalados
Write-Host "[7/7] Modelos instalados:" -ForegroundColor Yellow
docker exec app_rag_did-ollama-1 ollama list

Write-Host ""
Write-Host "=====================================" -ForegroundColor Cyan
if ($useGPU) {
    Write-Host "  Setup concluído com GPU! 🚀" -ForegroundColor Green
} else {
    Write-Host "  Setup concluído (CPU)!" -ForegroundColor Green
}
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Acesse: http://localhost:8501" -ForegroundColor Cyan
Write-Host ""
if ($useGPU) {
    Write-Host "Configurações recomendadas (GPU):" -ForegroundColor Yellow
    Write-Host "  Chunk size: 1200" -ForegroundColor Gray
    Write-Host "  Overlap: 200" -ForegroundColor Gray
    Write-Host "  k_docs: 10" -ForegroundColor Gray
    Write-Host "  Modelo: qwen2.5:7b" -ForegroundColor Gray
    Write-Host ""
    Write-Host "Para modelo maior (14B - limite de 6GB):" -ForegroundColor Yellow
    Write-Host "  docker exec app_rag_did-ollama-1 ollama pull qwen2.5:14b" -ForegroundColor Gray
} else {
    Write-Host "Para habilitar GPU:" -ForegroundColor Yellow
    Write-Host "  1. Instale NVIDIA Container Toolkit" -ForegroundColor Gray
    Write-Host "  2. Execute novamente o setup" -ForegroundColor Gray
}
Write-Host ""
Write-Host "Comandos úteis:" -ForegroundColor Yellow
Write-Host "  Ver logs:       docker-compose logs -f" -ForegroundColor Gray
Write-Host "  Parar tudo:     docker-compose down" -ForegroundColor Gray
Write-Host "  Reiniciar:      docker-compose restart" -ForegroundColor Gray
if ($useGPU) {
    Write-Host "  Monitorar GPU:  docker exec app_rag_did-ollama-1 watch -n 1 nvidia-smi" -ForegroundColor Gray
    Write-Host "  Testar GPU:     .\test-gpu.ps1" -ForegroundColor Gray
}
Write-Host ""