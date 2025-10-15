import os
from typing import List
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

def test_ollama_connection(url: str) -> bool:
    import requests
    try:
        response = requests.get(f"{url}/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False

FAISS_PATH = "./faiss_index"

def check_and_pull_model(model_name: str, base_url: str) -> bool:
    """Verifica se o modelo existe e faz download se necessário."""
    import requests
    try:
        response = requests.post(f"{base_url}/api/show", json={"name": model_name})
        if response.status_code == 200:
            return True
            
        st.warning(f"Modelo {model_name} não encontrado. Iniciando download...")
        response = requests.post(f"{base_url}/api/pull", json={"name": model_name})
        if response.status_code == 200:
            st.success(f"Modelo {model_name} baixado com sucesso!")
            return True
        else:
            st.error(f"Erro ao baixar modelo: {response.text}")
            return False
    except Exception as e:
        st.error(f"Erro ao verificar/baixar modelo: {str(e)}")
        return False

def check_gpu():
    """Verifica disponibilidade real da GPU testando CUDA."""
    import torch
    try:
        # Testa CUDA
        if not torch.cuda.is_available():
            return False, "GPU não detectada"
            
        # Testa acesso via Docker
        import subprocess
        result = subprocess.run(["docker", "run", "--rm", "--gpus", "all", "nvidia/cuda:11.8.0-base-ubuntu22.04", "nvidia-smi"],
                              capture_output=True, text=True)
        if result.returncode != 0:
            return False, "GPU detectada mas não acessível no Docker"
            
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return True, f"GPU: {gpu_name} ({gpu_memory:.1f}GB)"
    except:
        return False, "Erro ao verificar GPU"

# Configuração inicial
st.set_page_config(page_title="RAG - Assistente de Artigos Científicos", layout="wide")
st.title("RAG para Artigos Científicos")

# Verifica LLM antes de tudo
ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
if not test_ollama_connection(ollama_url):
    st.error("⚠️ LLM não está acessível. Verifique se o Ollama está rodando.")
    st.stop()

# Verifica/baixa modelo padrão
if not check_and_pull_model("qwen2.5:7b", ollama_url):
    st.error("⚠️ Não foi possível carregar o modelo. Use a opção [I] no menu principal para instalar.")
    st.stop()

if "vectordb" not in st.session_state:
    st.session_state.vectordb = None

with st.sidebar:
    st.header("Configurações")
    
    # Status GPU
    has_gpu, gpu_message = check_gpu()
    if has_gpu:
        st.success(gpu_message)
    else:
        st.warning(f"{gpu_message} - usando CPU")
    
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_ok = test_ollama_connection(ollama_url)
    
    if ollama_ok:
        st.success(f"LLM conectado")
    else:
        st.error(f"LLM offline: {ollama_url}")
    
    # Debug da base
    if os.path.exists(FAISS_PATH):
        st.info(f"Base encontrada: {FAISS_PATH}")
        files = os.listdir(FAISS_PATH)
        st.text(f"Arquivos: {len(files)}")
    else:
        st.warning(f"Base não existe: {FAISS_PATH}")
    
    # Status do vectordb
    if st.session_state.vectordb:
        st.success("VectorDB carregado na memória")
    else:
        st.warning("VectorDB não carregado")
    
    chunk_size = st.number_input("Chunk size", 100, 3000, 1200, 50)
    chunk_overlap = st.number_input("Overlap", 0, 500, 200, 10)
    k_docs = st.number_input("k docs", 1, 20, 10, 1)
    
    ollama_model = st.selectbox(
        "Modelo LLM",
        ["qwen2.5:7b", "qwen2.5:14b"]
    )
    temperature = st.slider("Temperature", 0.0, 1.0, 0.15, 0.05)
    
    use_existing = st.checkbox("Usar banco existente", value=True)

def load_pdfs(directory: str) -> List:
    docs = []
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            filepath = os.path.join(directory, filename)
            loader = PyPDFLoader(filepath)
            docs.extend(loader.load())
    return docs

def split_docs(docs, csize: int, coverlap: int):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=csize,
        chunk_overlap=coverlap,
        add_start_index=True
    )
    return splitter.split_documents(docs)

def get_embeddings():
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True}
    )

def load_or_create_vectordb(chunks=None):
    """Carrega ou cria uma base FAISS para armazenar embeddings."""
    try:
        embeddings = get_embeddings()
        index_file = os.path.join(FAISS_PATH, "index.faiss")
        
        # Apenas carrega se existir e use_existing for True
        if use_existing:
            if os.path.exists(index_file):
                try:
                    st.info("Carregando base FAISS existente...")
                    db = FAISS.load_local(FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
                    return db
                except Exception as e:
                    st.error(f"Erro ao carregar base: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
                    return None
            else:
                st.warning("Índice FAISS não existe ainda. Crie a base processando os PDFs.")
                return None
                
        # Criar nova base
        if chunks is None:
            st.error("Precisa processar PDFs primeiro")
            return None
            
        # Garantir que a pasta existe
        if not os.path.exists(FAISS_PATH):
            os.makedirs(FAISS_PATH)
            
        st.info(f"Criando nova base FAISS com {len(chunks)} chunks...")
        db = FAISS.from_documents(chunks, embeddings)
        
        st.info("Salvando índice FAISS...")
        db.save_local(FAISS_PATH)
        
        st.success("Base FAISS criada e salva com sucesso")
        return db
            
    except Exception as e:
        st.error(f"Erro ao processar base FAISS: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None

def check_and_pull_model(model_name: str, base_url: str) -> bool:
    """Verifica se o modelo existe e faz download se necessário."""
    import requests
    try:
        # Verifica se o modelo existe
        response = requests.post(f"{base_url}/api/show", json={"name": model_name})
        if response.status_code == 200:
            return True
            
        # Se não existe, tenta baixar
        st.warning(f"Modelo {model_name} não encontrado. Iniciando download...")
        response = requests.post(f"{base_url}/api/pull", json={"name": model_name})
        if response.status_code == 200:
            st.success(f"Modelo {model_name} baixado com sucesso!")
            return True
        else:
            st.error(f"Erro ao baixar modelo: {response.text}")
            return False
    except Exception as e:
        st.error(f"Erro ao verificar/baixar modelo: {str(e)}")
        return False

def make_qa_chain(vectordb, model: str, temp: float):
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    if not test_ollama_connection(ollama_url):
        st.error(f"LLM não está acessível em {ollama_url}")
        return None
        
    # Verifica e baixa o modelo se necessário
    if not check_and_pull_model(model, ollama_url):
        return None
    
    llm = Ollama(
        model=model,
        temperature=temp,
        base_url=ollama_url,
        callbacks=[StreamingStdOutCallbackHandler()]
    )
    
    from langchain.prompts import PromptTemplate
    
    template = """Você é um assistente especializado em análise de artigos científicos e acadêmicos.

IMPORTANTE:
- O contexto abaixo está em INGLÊS (de artigos acadêmicos)
- A pergunta está em PORTUGUÊS
- Você deve responder em PORTUGUÊS BRASILEIRO, traduzindo as informações relevantes do contexto

Use o contexto fornecido para responder à pergunta de forma clara e completa.
Se não souber a resposta ou se o contexto não contiver informações suficientes, diga claramente.
Cite conceitos, teorias, metodologias ou autores mencionados nos artigos quando relevante.

Contexto (em inglês):
{context}

Pergunta: {question}

Resposta detalhada em português brasileiro:"""
    
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    
    retriever = vectordb.as_retriever(search_kwargs={"k": k_docs})
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Gerenciar Base")
    
    pdf_dir = st.text_input("Diretório dos PDFs", "./artigos")
    
    if st.button("Processar Artigos"):
        if not os.path.exists(pdf_dir):
            st.error(f"Diretório não existe: {pdf_dir}")
        else:
            try:
                # Desativa 'usar banco existente' ao processar novos artigos
                use_existing = False
                
                with st.spinner("Lendo PDFs..."):
                    raw_docs = load_pdfs(pdf_dir)
                
                if not raw_docs:
                    st.warning(f"Nenhum PDF encontrado em {pdf_dir}")
                else:
                    st.info(f"PDFs encontrados: {len(raw_docs)} documentos")
                    
                    with st.spinner("Criando chunks..."):
                        chunks = split_docs(raw_docs, chunk_size, chunk_overlap)
                    st.write(f"Docs: {len(raw_docs)} | Chunks: {len(chunks)}")
                    
                    with st.spinner("Baixando modelo de embeddings (primeira vez)..."):
                        embeddings = get_embeddings()
                        st.success("Embeddings carregado")
                    
                    with st.spinner("Indexando..."):
                        # Garante que a pasta existe
                        if not os.path.exists(FAISS_PATH):
                            os.makedirs(FAISS_PATH)
                            
                        try:
                            st.info("Gerando embeddings dos chunks...")
                            vectordb = FAISS.from_documents(chunks, embeddings)
                            
                            st.info("Salvando índice FAISS...")
                            vectordb.save_local(FAISS_PATH)
                            
                            st.session_state.vectordb = vectordb
                            st.success("Base criada e carregada com sucesso!")
                        except Exception as e:
                            st.error(f"Erro ao criar índice FAISS: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())
            except Exception as e:
                st.error(f"Erro geral: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    
    if st.button("Carregar Base Existente"):
        if os.path.exists(FAISS_PATH):
            with st.spinner("Carregando base..."):
                vectordb = load_or_create_vectordb()
            
            if vectordb:
                st.session_state.vectordb = vectordb
                st.success("Base carregada")
            else:
                st.error("Erro ao carregar base")
        else:
            st.error(f"Base não existe em: {FAISS_PATH}")
    
    if st.button("Limpar Base"):
        if os.path.exists(FAISS_PATH):
            import shutil
            shutil.rmtree(FAISS_PATH)
            st.session_state.vectordb = None
            st.success("Base removida")

with col2:
    st.subheader("Consultas")
    
    query = st.text_area(
        "Pergunta",
        placeholder="Ex: Quais são as principais contribuições metodológicas discutidas nos artigos?"
    )
    
    if st.button("Perguntar"):
        if st.session_state.vectordb is None:
            st.warning("Carregue ou crie a base primeiro")
        else:
            qa = make_qa_chain(st.session_state.vectordb, ollama_model, temperature)
            
            if qa is None:
                st.stop()
            
            with st.spinner("Gerando resposta..."):
                result = qa({"query": query})
            
            st.markdown("### Resposta")
            st.write(result["result"])
            
            st.markdown("### Fontes")
            for i, doc in enumerate(result["source_documents"], 1):
                meta = doc.metadata
                source = meta.get("source", "").split("/")[-1]
                page = meta.get("page", "n/a")
                st.markdown(f"**{i}.** {source} (pág. {page})")
                with st.expander(f"Ver trecho {i}"):
                    st.text(doc.page_content[:500])