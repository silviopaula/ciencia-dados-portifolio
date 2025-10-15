#!/bin/bash

echo "====================================="
echo "  Setup RAG com Docker e Ollama"
echo "====================================="
echo ""

# 1. Para containers existentes
echo "[1/6] Parando containers existentes..."
docker-compose down
echo "OK"
sleep 2

# 2. Build das imagens
echo "[2/6] Construindo imagens Docker..."
docker-compose build
if [ $? -ne 0 ]; then
    echo "ERRO no build!"
    exit 1
fi
echo "OK"
sleep 2

# 3. Sobe os containers
echo "[3/6] Subindo containers..."
docker-compose up -d
if [ $? -ne 0 ]; then
    echo "ERRO ao subir containers!"
    exit 1
fi
echo "OK"
sleep 5

# 4. Aguarda Ollama ficar pronto
echo "[4/6] Aguardando Ollama inicializar..."
max_attempts=30
attempt=0
ollama_ready=false

while [ $attempt -lt $max_attempts ] && [ "$ollama_ready" = false ]; do
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        ollama_ready=true
    else
        attempt=$((attempt + 1))
        echo "  Tentativa $attempt/$max_attempts..."
        sleep 2
    fi
done

if [ "$ollama_ready" = false ]; then
    echo "ERRO: Ollama não inicializou!"
    exit 1
fi
echo "OK"

# 5. Baixa modelo qwen2.5 (melhor multilíngue)
echo "[5/6] Baixando modelo qwen2.5:7b (pode demorar - ~4.7GB)..."
docker exec app_rag_did-ollama-1 ollama pull qwen2.5:7b
if [ $? -ne 0 ]; then
    echo "ERRO ao baixar qwen2.5!"
    exit 1
fi
echo "OK"

# 6. Verifica modelos instalados
echo "[6/6] Modelos instalados:"
docker exec app_rag_did-ollama-1 ollama list

echo ""
echo "====================================="
echo "  Setup concluído com sucesso!"
echo "====================================="
echo ""
echo "Acesse: http://localhost:8501"
echo ""
echo "Comandos úteis:"
echo "  Ver logs:    docker-compose logs -f"
echo "  Parar tudo:  docker-compose down"
echo "  Reiniciar:   docker-compose restart"
echo ""