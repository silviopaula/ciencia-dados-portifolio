# RAG para Livros de Macroeconomia

Aplicação de Retrieval-Augmented Generation (RAG) para consultar livros e notas, com um exemplo para livros de macroeconomia em PDF usando IA generativa.

## Descrição

Sistema que permite fazer upload de PDFs, processar o conteúdo, indexar em um banco vetorial e realizar perguntas contextualizadas usando o modelo Gemini do Google. As respostas são geradas com base no conteúdo dos documentos carregados, com citação das fontes.

## Tecnologias

### Framework e Interface
- **Streamlit**: Interface web interativa e responsiva
- **Python 3.10+**: Linguagem base

### Processamento de Documentos
- **LangChain**: Framework para aplicações com LLMs
- **PyPDFLoader**: Extração de texto de PDFs mantendo metadados (página, fonte)
- **RecursiveCharacterTextSplitter**: Divisão inteligente de texto em chunks

### Embeddings e Armazenamento Vetorial
- **HuggingFace Embeddings**: Modelo `sentence-transformers/all-MiniLM-L6-v2`
- **Chroma**: Banco de dados vetorial para busca semântica

### LLM
- **Google Gemini**: Modelo `gemini-2.5-flash` para geração de respostas

## Como Foi Construído

### Arquitetura do Sistema
```
┌─────────────┐
│   PDF Upload │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│  PyPDFLoader    │  Lê PDFs e extrai texto + metadados
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  Text Splitter  │  Divide em chunks de 500 caracteres (overlap 50)
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  HF Embeddings  │  Converte chunks em vetores (384 dimensões)
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│     Chroma      │  Armazena vetores + texto original
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  Pergunta User  │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│   Retriever     │  Busca k=4 chunks mais similares
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  Gemini LLM     │  Gera resposta baseada nos chunks recuperados
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  Resposta +     │
│  Fontes         │
└─────────────────┘
```

### Fluxo de Dados

1. **Ingestão**
   - Usuário faz upload de 1 ou mais PDFs
   - PyPDFLoader processa cada página e mantém metadados
   - Texto é dividido em chunks de 500 caracteres com overlap de 50

2. **Indexação**
   - Cada chunk é convertido em embedding (vetor de 384 dimensões)
   - Vetores são armazenados no Chroma com o texto original
   - Index fica em memória durante a sessão

3. **Consulta**
   - Usuário faz uma pergunta em linguagem natural
   - Pergunta é convertida em embedding
   - Sistema busca os 4 chunks mais similares (busca por similaridade de cosseno)
   - Chunks recuperados são enviados como contexto para o Gemini
   - LLM gera resposta fundamentada no contexto
   - Sistema exibe resposta e as fontes (arquivo + página)

### Decisões Técnicas

- **Chunk size 500**: Balanceio entre contexto e precisão
- **Overlap 50**: Evita perda de informação nas bordas
- **k=4 docs**: Suficiente para contexto sem sobrecarregar o prompt
- **all-MiniLM-L6-v2**: Rápido e eficiente para português/inglês
- **gemini-2.5-flash**: Custo-benefício e latência baixa
- **Temperature 0.2**: Respostas mais determinísticas e factuais

## Instalação

### Pré-requisitos
- Python 3.10 ou superior
- Chave de API do Google Gemini

## Como Usar

### 1. Iniciar a aplicação
```powershell
# Ative o ambiente virtual
.venv\Scripts\activate

# Execute o Streamlit
streamlit run app.py
```

### 2. Configurar

Na sidebar:
- Ajuste chunk_size e chunk_overlap se necessário
- Defina k (número de documentos recuperados)
- Cole sua GOOGLE_API_KEY

### 3. Processar PDFs

- Faça upload dos PDFs
- Clique em "Processar PDFs"
- Aguarde a indexação

### 4. Fazer Perguntas

- Digite sua pergunta na caixa de texto
- Clique em "Perguntar"
- Veja a resposta e as fontes citadas

## Exemplos de Perguntas

- "Explique o modelo IS-LM"
- "Qual a diferença entre inflação de demanda e inflação de custos?"
- "Como funciona a política monetária expansionista?"
- "Diferencie as versões da Curva de Phillips"

## Configurações Avançadas

### Personalizar Embeddings

Troque o modelo na função `build_vectordb`:
```python
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
```

### Persistência do Banco

Para manter o índice entre sessões, use um diretório fixo:
```python
persist_dir = "./chroma_db"
```

## Limitações

- Banco vetorial em diretório temporário (perde ao reiniciar)
- Não suporta imagens ou tabelas complexas dos PDFs
- Limitado pela janela de contexto do LLM

## Melhorias Futuras

- Persistência permanente do banco vetorial
- Cache de embeddings para reprocessamento rápido
- Suporte a outros formatos (DOCX, TXT)
- Interface para gerenciar documentos indexados
- Histórico de conversas