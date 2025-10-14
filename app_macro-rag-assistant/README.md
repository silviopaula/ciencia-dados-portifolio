# RAG para Livros de Macroeconomia

Aplicação de **Retrieval-Augmented Generation (RAG)** para consultar livros e notas em PDF, com um exemplo voltado para **livros de macroeconomia** usando IA generativa.

---

## Interface

![Interface da Aplicação](https://github.com/silviopaula/ciencia-dados-portifolio/blob/main/app_macro-rag-assistant/img/img.png)

---

## Descrição

Sistema que permite fazer **upload de PDFs**, processar o conteúdo, indexar em um **banco vetorial** e realizar **perguntas contextualizadas** usando o modelo **Gemini** do Google.  
As respostas são geradas com base no conteúdo dos documentos carregados, com **citação das fontes**.

---

## Tecnologias

### Framework e Interface
- **Streamlit**: Interface web interativa e responsiva  
- **Python 3.10+**: Linguagem base  

### Processamento de Documentos
- **LangChain**: Framework para aplicações com LLMs  
- **PyPDFLoader**: Extração de texto de PDFs com metadados  
- **RecursiveCharacterTextSplitter**: Divisão em *chunks* de texto  

### Embeddings e Armazenamento Vetorial
- **HuggingFace Embeddings**: Modelo `all-MiniLM-L6-v2` (384 dimensões)  
- **ChromaDB**: Banco vetorial local e leve  

### LLM
- **Google Gemini (`gemini-2.5-flash`)**: geração de respostas rápidas e custo-efetivas  

---

## Arquitetura do Sistema

PDF Upload → PyPDFLoader → Text Splitter → HF Embeddings → Chroma                               
↓                          
Gemini LLM (consulta)                         
↓                         
Resposta + Fontes Citadas                         


---

## Fluxo de Dados

1. **Ingestão:** upload e extração de texto via PyPDFLoader  
2. **Indexação:** conversão em embeddings e armazenamento no Chroma  
3. **Consulta:** busca semântica e resposta gerada pelo Gemini  

---

## Decisões Técnicas

| Parâmetro | Valor | Justificativa |
|------------|--------|----------------|
| `chunk_size` | 500 | Equilíbrio entre contexto e precisão |
| `chunk_overlap` | 50 | Evita perda de informação |
| `k` | 4 | Retorna os 4 trechos mais relevantes |
| Embedding | `all-MiniLM-L6-v2` | Rápido e eficiente |
| Modelo LLM | `gemini-2.5-flash` | Baixa latência |
| Temperature | 0.2 | Respostas mais factuais |

---

## Privacidade e Segurança

Durante a consulta:
- Apenas **trechos relevantes (chunks)** são enviados ao LLM  
- **Documentos completos e embeddings** permanecem locais  

| Item | Vai para o LLM externo? |
|------|--------------------------|
| Documentos inteiros | ❌ Não |
| Trechos recuperados (chunks) | ✅ Sim |
| Embeddings vetoriais | ❌ Não |
| Logs temporários | ⚠️ Possível (depende do provedor) |

### Boas Práticas
- Usar **LLM local (Ollama, LM Studio, Llama.cpp, Gemma)** para dados sensíveis  
- **RAG híbrido:** busca local + API externa apenas para perguntas genéricas  
- **Filtrar contexto** para evitar envio de informações confidenciais  
- **On-premise:** hospedar modelo e banco vetorial internamente  

---

## Desempenho e Custos
- **Indexação:** 2–5 s por PDF de 100 páginas  
- **Busca:** 100–200 ms  
- **Geração:** 2–5 s  
- **Custo:** embeddings e Chroma gratuitos; API do Gemini com camada gratuita  

---

## Limitações
- Banco vetorial temporário (volátil)  
- Não processa imagens nem tabelas complexas  
- Limitado à janela de contexto do LLM (~32 k tokens)  
- Qualidade depende do texto dos PDFs  

---

## Melhorias Futuras
- Persistência permanente do banco vetorial  
- Cache de embeddings  
- Suporte a DOCX e TXT  
- Interface para gerenciar PDFs indexados  
- Histórico de conversas  

---

## Exemplos de Perguntas
- “Explique o modelo IS-LM”  
- “Qual a diferença entre inflação de demanda e de custos?”  
- “Como funciona a política monetária expansionista?”  
- “Diferencie as versões da Curva de Phillips”

---

**Benefício:** ler centenas de páginas em segundos, com respostas baseadas nas fontes reais.
