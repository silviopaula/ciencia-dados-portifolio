import os
import tempfile
from typing import List

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

st.set_page_config(page_title="RAG - Macroeconomia (PDF)", layout="wide")
st.title("RAG para Livros/Notas de Macroeconomia (PDF)")

with st.sidebar:
    st.header("Configurações")
    chunk_size = st.number_input("Chunk size", min_value=100, max_value=2000, value=500, step=50)
    chunk_overlap = st.number_input("Chunk overlap", min_value=0, max_value=500, value=50, step=10)
    k_docs = st.number_input("k (docs recuperados)", min_value=1, max_value=10, value=4, step=1)
    gemini_model = st.text_input("Gemini model", value="gemini-1.5-flash")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
    apikey = st.text_input("GOOGLE_API_KEY", type="password")

    if apikey:
        os.environ["GOOGLE_API_KEY"] = apikey

if "vectordb" not in st.session_state:
    st.session_state.vectordb = None
if "persist_dir" not in st.session_state:
    st.session_state.persist_dir = None

uploaded_files = st.file_uploader(
    "Carregue 1+ PDFs (livros, capítulos, notas de aula)", type=["pdf"], accept_multiple_files=True
)

def load_pdfs_to_docs(files) -> List:
    docs = []
    for f in files:
        tmp_path = os.path.join(tempfile.gettempdir(), f.name)
        with open(tmp_path, "wb") as out:
            out.write(f.read())
        loader = PyPDFLoader(tmp_path)
        docs.extend(loader.load())
    return docs

def split_docs(docs, csize: int, coverlap: int):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=csize,
        chunk_overlap=coverlap,
        add_start_index=True
    )
    return splitter.split_documents(docs)

def build_vectordb(chunks, persist_dir: str):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": False}
    )
    db = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=persist_dir)
    db.persist()
    return db

def make_qa_chain(vectordb, model_name: str, temp: float):
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=temp)
    retriever = vectordb.as_retriever(search_kwargs={"k": k_docs})
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

col_ing, col_q = st.columns([1, 2])

with col_ing:
    st.subheader("1) Ingestão e Indexação")
    if st.button("Processar PDFs"):
        if not uploaded_files:
            st.warning("Carregue ao menos um PDF.")
        else:
            with st.spinner("Lendo PDFs..."):
                raw_docs = load_pdfs_to_docs(uploaded_files)
            with st.spinner("Fazendo chunking..."):
                chunks = split_docs(raw_docs, chunk_size, chunk_overlap)
            st.write(f"Documentos: {len(raw_docs)} | Chunks: {len(chunks)}")

            with st.spinner("Construindo vetorstore (Chroma)..."):
                persist_dir = tempfile.mkdtemp(prefix="chroma_macro_")
                vectordb = build_vectordb(chunks, persist_dir)

            st.session_state.vectordb = vectordb
            st.session_state.persist_dir = persist_dir
            st.success("Index pronto.")

with col_q:
    st.subheader("2) Perguntas")
    query = st.text_input("Faça uma pergunta (ex.: Explique o modelo IS-LM; Diferencie as versões da Curva de Phillips).")
    ask = st.button("Perguntar")

    if ask:
        if st.session_state.vectordb is None:
            st.warning("Crie o índice primeiro (botão 'Processar PDFs').")
        else:
            if not os.environ.get("GOOGLE_API_KEY"):
                st.error("Defina GOOGLE_API_KEY na sidebar ou no ambiente.")
            else:
                qa = make_qa_chain(st.session_state.vectordb, gemini_model, temperature)
                with st.spinner("Gerando resposta..."):
                    out = qa({"query": query})

                st.markdown("### Resposta")
                st.write(out.get("result", ""))

                st.markdown("### Fontes")
                src_docs = out.get("source_documents", [])
                for i, d in enumerate(src_docs, 1):
                    meta = d.metadata or {}
                    source = meta.get("source", "desconhecido")
                    page = meta.get("page", "n/a")
                    st.markdown(f"**{i}.** {source} | página: {page}")