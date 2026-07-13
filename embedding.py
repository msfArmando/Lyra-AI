from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_postgres import PGVectorStore, PGEngine
import asyncio
import uuid
from langchain_core.documents import Document
import os
from pathlib import Path
from dotenv import load_dotenv
import chainlit as cl

async def Embedding(textembedd: str, category: str) -> None:
    load_dotenv()

    POSTGRES_USER = os.getenv("POSTGRES_USER")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
    POSTGRES_HOST = os.getenv("POSTGRES_HOST")
    POSTGRES_PORT = os.getenv("POSTGRES_PORT")
    POSTGRES_DB = os.getenv("POSTGRES_DB")
    TABLE_NAME = os.getenv("TABLE_NAME")
    SCHEMA_NAME = os.getenv("SCHEMA_NAME")

    CONNECTION_STRING = (
        f"postgresql+asyncpg://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}"
        f":{POSTGRES_PORT}/{POSTGRES_DB}"
    )

    files_path: str = './files_embedd'
    embedding_model = OllamaEmbeddings(model="bge-m3", base_url="http://192.168.1.128:11434/")
    pg_engine = PGEngine.from_connection_string(url=CONNECTION_STRING)

    text_list: str = ""
    textmetadata: str = ""
    textid: str = ""

    if os.listdir(files_path):
        print("Achei arquivos numa pasta...")
        for file in os.listdir(files_path):
            full_path: str = os.path.join(files_path, file)
            file_suffix: str = str(Path(full_path).suffix)

            if file_suffix == ".txt":
                if os.path.isfile(full_path):
                    with open(full_path, 'r', encoding='utf-8') as f:
                        content: str = f.read()
                        text_list: list[str] = [content]
    else:
        print("Não tem menhum arquivo na pasta...")
        text_list: list[str] = [textembedd]

    print(f"O texto que será salvo na base de dados é: {text_list}")
    textmetadata: list[dict[str, str]] = [{"category": f"{category}"}]
    textid = [str(uuid.uuid4()) for _ in text_list]
    
    try:
        await pg_engine.ainit_vectorstore_table(
            table_name=str(TABLE_NAME),
            vector_size=1024,
            schema_name=str(SCHEMA_NAME)
        )
    except Exception as e:
        if "already exists" in str(e):
            pass

    store = await PGVectorStore.create(
        engine=pg_engine,
        table_name=str(TABLE_NAME),
        schema_name=str(SCHEMA_NAME),
        embedding_service=embedding_model
    )
    print("Tentando adicionar O TEXTO na base de dados VETORIAL...")
    await store.aadd_texts(text_list, metadatas=textmetadata, ids=textid)