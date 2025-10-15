Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "  Teste de GPU - NVIDIA 1660 Super" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""

# 1. Testar NVIDIA Docker
Write-Host "[1/5] Testando NVIDIA Docker..." -ForegroundColor Yellow
try {
    docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
    Write-Host "OK - GPU acessível no Docker" -ForegroundColor Green
} catch {
    Write-Host "ERRO - GPU não acessível" -ForegroundColor Red
    Write-Host "Instale NVIDIA Container Toolkit" -ForegroundColor Yellow
    exit 1
}
Write-Host ""

# 2. Verificar se containers estão rodando
Write-Host "[2/5] Verificando containers..." -ForegroundColor Yellow
$containers = docker ps --format "{{.Names}}"
if ($containers -match "ollama" -and $containers -match "app") {
    Write-Host "OK - Containers rodando" -ForegroundColor Green
} else {
    Write-Host "AVISO - Containers não encontrados" -ForegroundColor Yellow
    Write-Host "Execute: docker-compose up -d" -ForegroundColor Gray
}
Write-Host ""

# 3. Testar GPU no Ollama
Write-Host "[3/5] Testando GPU no Ollama..." -ForegroundColor Yellow
try {
    docker exec app_rag_did-ollama-1 nvidia-smi
    Write-Host "OK - Ollama com acesso a GPU" -ForegroundColor Green
} catch {
    Write-Host "ERRO - Ollama sem GPU" -ForegroundColor Red
    Write-Host "Verifique docker-compose.yml" -ForegroundColor Yellow
}
Write-Host ""

# 4. Testar PyTorch no app
Write-Host "[4/5] Testando PyTorch/CUDA no app..." -ForegroundColor Yellow
try {
    $cuda_test = docker exec app_rag_did-app-1 python -c "import torch; print(torch.cuda.is_available())"
    if ($cuda_test -match "True") {
        Write-Host "OK - PyTorch detectou GPU" -ForegroundColor Green
        
        $gpu_name = docker exec app_rag_did-app-1 python -c "import torch; print(torch.cuda.get_device_name(0))"
        Write-Host "GPU: $gpu_name" -ForegroundColor Cyan
    } else {
        Write-Host "AVISO - PyTorch não detectou GPU" -ForegroundColor Yellow
    }
} catch {
    Write-Host "ERRO ao testar PyTorch" -ForegroundColor Red
}
Write-Host ""

# 5. Listar modelos instalados
Write-Host "[5/5] Modelos Ollama instalados:" -ForegroundColor Yellow
try {
    docker exec app_rag_did-ollama-1 ollama list
} catch {
    Write-Host "Nenhum modelo instalado" -ForegroundColor Gray
}
Write-Host ""

Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "  Teste Concluído" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Recomendações:" -ForegroundColor Yellow
Write-Host "  - Use qwen2.5:7b (4.5GB VRAM)" -ForegroundColor Gray
Write-Host "  - Pode testar qwen2.5:14b (5.8GB VRAM - limite)" -ForegroundColor Gray
Write-Host "  - Ajuste k_docs para 10-15" -ForegroundColor Gray
Write-Host "  - Chunk size pode ser 1200-1500" -ForegroundColor Gray
Write-Host ""
Write-Host "Para instalar qwen2.5:7b:" -ForegroundColor Yellow
Write-Host "  docker exec app_rag_did-ollama-1 ollama pull qwen2.5:7b" -ForegroundColor Gray
Write-Host ""Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "  Teste de GPU - NVIDIA 1660 Super" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""

# 1. Testar NVIDIA Docker
Write-Host "[1/5] Testando NVIDIA Docker..." -ForegroundColor Yellow
try {
    docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
    Write-Host "OK - GPU acessível no Docker" -ForegroundColor Green
} catch {
    Write-Host "ERRO - GPU não acessível!" -ForegroundColor Red
    Write-Host "Instale NVIDIA Container Toolkit" -ForegroundColor Yellow
    exit 1
}
Write-Host ""

# 2. Verificar se containers estão rodando
Write-Host "[2/5] Verificando containers..." -ForegroundColor Yellow
$containers = docker ps --format "{{.Names}}"
if ($containers -match "ollama" -and $containers -match "app") {
    Write-Host "OK - Containers rodando" -ForegroundColor Green
} else {
    Write-Host "AVISO - Containers não encontrados" -ForegroundColor Yellow
    Write-Host "Execute: docker-compose up -d" -ForegroundColor Gray
}
Write-Host ""

# 3. Testar GPU no Ollama
Write-Host "[3/5] Testando GPU no Ollama..." -ForegroundColor Yellow
try {
    docker exec app_rag_did-ollama-1 nvidia-smi
    Write-Host "OK - Ollama com acesso a GPU" -ForegroundColor Green
} catch {
    Write-Host "ERRO - Ollama sem GPU!" -ForegroundColor Red
    Write-Host "Verifique docker-compose.yml" -ForegroundColor Yellow
}
Write-Host ""

# 4. Testar PyTorch no app
Write-Host "[4/5] Testando PyTorch/CUDA no app..." -ForegroundColor Yellow
try {
    $cuda_test = docker exec app_rag_did-app-1 python -c "import torch; print(torch.cuda.is_available())"
    if ($cuda_test -match "True") {
        Write-Host "OK - PyTorch detectou GPU" -ForegroundColor Green
        
        $gpu_name = docker exec app_rag_did-app-1 python -c "import torch; print(torch.cuda.get_device_name(0))"
        Write-Host "GPU: $gpu_name" -ForegroundColor Cyan
    } else {
        Write-Host "AVISO - PyTorch não detectou GPU" -ForegroundColor Yellow
    }
} catch {
    Write-Host "ERRO ao testar PyTorch" -ForegroundColor Red
}
Write-Host ""

# 5. Listar modelos instalados
Write-Host "[5/5] Modelos Ollama instalados:" -ForegroundColor Yellow
try {
    docker exec app_rag_did-ollama-1 ollama list
} catch {
    Write-Host "Nenhum modelo instalado" -ForegroundColor Gray
}
Write-Host ""

Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "  Teste Concluído!" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Recomendações:" -ForegroundColor Yellow
Write-Host "  - Use qwen2.5:7b (4.5GB VRAM)" -ForegroundColor Gray
Write-Host "  - Pode testar qwen2.5:14b (5.8GB VRAM - limite!)" -ForegroundColor Gray
Write-Host "  - Ajuste k_docs para 10-15" -ForegroundColor Gray
Write-Host "  - Chunk size pode ser 1200-1500" -ForegroundColor Gray
Write-Host ""
Write-Host "Para instalar qwen2.5:7b:" -ForegroundColor Yellow
Write-Host "  docker exec app_rag_did-ollama-1 ollama pull qwen2.5:7b" -ForegroundColor Gray
Write-Host ""