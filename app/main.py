import os
import tempfile
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# Загружаем .env
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY не задан в .env")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")

# Директории
BASE_DIR = Path(__file__).resolve().parent.parent
VECTOR_DIR = BASE_DIR / "chroma_db"
VECTOR_DIR.mkdir(exist_ok=True)

STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(exist_ok=True)

app = FastAPI(title="PDF RAG Platform (MVP, multi-step)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # для MVP можно так, потом лучше ограничить
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Отдаём статику (index.html и т.п.)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def split_docs_from_pdf(pdf_path: str):
    """Загружает PDF и режет на чанки."""
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    splits = splitter.split_documents(docs)
    return splits


def get_embeddings():
    """Создаёт объект эмбеддингов."""
    return OpenAIEmbeddings(model=OPENAI_EMBED_MODEL)


def create_vector_store_for_kb(kb_id: str, splits):
    """Создаёт векторное хранилище для конкретной базы знаний (kb_id)."""
    embeddings = get_embeddings()
    vector_store = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        collection_name=kb_id,
        persist_directory=str(VECTOR_DIR),
    )
    return vector_store


def load_vector_store(kb_id: str) -> Chroma:
    """Загружает существующее векторное хранилище по kb_id."""
    embeddings = get_embeddings()
    vector_store = Chroma(
        collection_name=kb_id,
        embedding_function=embeddings,
        persist_directory=str(VECTOR_DIR),
    )
    return vector_store


def build_rag_chain(vector_store: Chroma):
    """RAG-цепочка: retriever → prompt → LLM → строковый ответ."""
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    prompt = ChatPromptTemplate.from_template(
        (
            "Ты помощник по документу. Отвечай строго на основе контекста.\n\n"
            "Контекст:\n{context}\n\n"
            "Вопрос: {question}\n\n"
            "Если ответа нет в документе, честно скажи, что не знаешь.\n"
            "Отвечай на том же языке, на котором задан вопрос."
        )
    )

    llm = ChatOpenAI(
        model=OPENAI_MODEL,
        temperature=0,
    )

    parser = StrOutputParser()

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | parser
    )
    return chain


@app.get("/")
def read_root():
    return {
        "status": "ok",
        "message": "PDF RAG Platform running (multi-step: /create-kb -> /ask, UI at /static/index.html)",
    }


@app.post("/create-kb")
async def create_kb(
    file: UploadFile = File(...),
):
    """
    1) Принимает один PDF
    2) Создаёт базу знаний (kb_id)
    3) Строит векторное хранилище и сохраняет его
    4) Возвращает kb_id, по которому потом можно задавать вопросы
    """

    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="PDF file required")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        splits = split_docs_from_pdf(tmp_path)
        kb_id = str(uuid.uuid4())

        create_vector_store_for_kb(kb_id, splits)

        return {"kb_id": kb_id}

    finally:
        if "tmp_path" in locals() and Path(tmp_path).exists():
            Path(tmp_path).unlink(missing_ok=True)


@app.post("/ask")
async def ask(
    kb_id: str = Form(...),
    question: str = Form(...),
):
    """
    1) Принимает kb_id и вопрос
    2) Загружает векторное хранилище по kb_id
    3) Отвечает на вопрос по этой базе знаний
    """

    if not kb_id:
        raise HTTPException(status_code=400, detail="kb_id is required")

    vector_store = load_vector_store(kb_id)
    rag_chain = build_rag_chain(vector_store)
    answer = rag_chain.invoke(question)

    return {"answer": answer}
