# RAG para Artigos Científicos 📚

Sistema de RAG (Retrieval-Augmented Generation) para análise de artigos científicos com interface web via Streamlit, processamento de PDFs, base vetorial FAISS e integração com LLM via Ollama.

## 🌟 Características

- Interface web amigável com Streamlit
- Suporte a GPU NVIDIA para aceleração
- Processamento de múltiplos PDFs
- Embeddings otimizados com FAISS
- LLM local via Ollama (qwen2.5:7b)
- Containerização com Docker
- Sistema de menu interativo
- Detecção automática de GPU

## 📂 Estrutura do Projeto

```
app_rag_papers/
├── app.py              # Aplicação Streamlit principal
├── docker-compose.yml  # Configuração dos containers
├── Dockerfile         # Build da imagem do app
├── requirements.txt   # Dependências Python
├── rag.bat           # Menu interativo (Windows)
├── setup.ps1         # Script de setup PowerShell
├── test-gpu.ps1      # Teste de GPU
├── artigos/          # Pasta para PDFs
└── faiss_index/      # Base vetorial persistente
```

## 🚀 Instalação

### Pré-requisitos

- Python 3.10+
- Docker Desktop
- NVIDIA GPU (opcional)
- NVIDIA Container Toolkit (para GPU)

### Instalação Rápida

1. Clone o repositório
2. Execute `rag.bat`
3. Selecione opção [1] para iniciar

O sistema automaticamente:
- Verifica Docker e GPU
- Baixa e configura containers
- Instala modelo LLM
- Abre interface web

## 💻 Uso

### Interface Web

1. **Gerenciar Base**
   - Upload de PDFs em `/artigos`
   - Processamento de documentos
   - Criação/atualização de índice FAISS
   - Parâmetros ajustáveis (chunk size, overlap)

2. **Consultas**
   - Perguntas em português
   - Respostas baseadas no contexto
   - Citação de fontes
   - Trechos relevantes expandíveis

### Configurações

- **Chunk size**: 1200 (recomendado)
- **Overlap**: 200 (recomendado)
- **k_docs**: 10 (ajustável)
- **Temperatura**: 0.15 (ajustável)

### Menu RAG (rag.bat)

```
[1] Iniciar RAG
[2] Parar RAG
[3] Reiniciar RAG
[4] Status completo
[5] Ver logs
[I] Instalar
[G] Configurar GPU
[A] Abrir navegador
```

## 🔧 Tecnologias

- **Frontend**: Streamlit
- **Embeddings**: HuggingFace (all-MiniLM-L6-v2)
- **Base Vetorial**: FAISS
- **LLM**: Ollama (qwen2.5:7b)
- **Containers**: Docker
- **PDF**: PyPDF + LangChain
- **GPU**: NVIDIA CUDA

## ⚡ Performance

### Com GPU
- Embeddings: ~2-3s por página
- LLM: 5-10s por resposta
- VRAM: 4.5GB (modelo base)

### Sem GPU
- Embeddings: ~5-7s por página
- LLM: 30-60s por resposta
- RAM: 8GB+ recomendado

## 🛠️ Desenvolvimento

### Estrutura do Código

- **app.py**
  ```python
  # Componentes principais
  - Interface Streamlit
  - Processamento de PDFs
  - Gerenciamento FAISS
  - Integração Ollama
  - Sistema de cache
  ```

- **docker-compose.yml**
  ```yaml
  # Serviços
  - app: Interface web
  - ollama: Servidor LLM
  ```

### Scripts Auxiliares

- **setup.ps1**: Setup completo
- **test-gpu.ps1**: Diagnóstico GPU
- **rag.bat**: Interface CLI

## 📊 Limitações e Recomendações

### GPU
- VRAM mínima: 4GB
- Recomendado: 6GB+
- qwen2.5:7b: 4.5GB VRAM
- qwen2.5:14b: 5.8GB VRAM

### PDFs
- Tamanho: < 100MB por arquivo
- Páginas: < 1000 por sessão
- Formatos: PDF texto (não imagens)

### Base FAISS
- Chunks: < 10000 recomendado
- Persistente entre sessões
- Limpeza manual quando necessário

## 🔄 Manutenção

### Limpeza
```bash
# Via menu RAG
[2] Parar RAG
[L] Limpar e otimizar
[1] Iniciar RAG
```

### Atualização
```bash
git pull
docker-compose pull
docker-compose build --no-cache
```

## 🔍 Troubleshooting

### Problemas Comuns

1. **GPU não detectada**
   - Verificar drivers NVIDIA
   - Instalar NVIDIA Container Toolkit
   - Usar opção [G] no menu

2. **Erro de modelo**
   - Verificar conexão Ollama
   - Reinstalar modelo: opção [I]
   - Checar logs: opção [5]

3. **Base FAISS**
   - Limpar base existente
   - Reprocessar PDFs
   - Verificar permissões pasta

### Logs
```bash
# Ver logs em tempo real
rag.bat > opção [5]
```

## 📝 Notas

- Mantenha PDFs organizados em `/artigos`
- Backup periódico de `/faiss_index`
- Monitore uso de GPU/RAM
- Ajuste parâmetros conforme necessidade
- Use branch stable para produção

## 🤝 Contribuição

1. Fork o projeto
2. Crie branch (`git checkout -b feature/nova-feature`)
3. Commit (`git commit -m 'Adiciona nova feature'`)
4. Push (`git push origin feature/nova-feature`)
5. Abra Pull Request

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.