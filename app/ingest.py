from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-base-en-v1.5")

def embed_text(text: str):
    return model.encode(text).tolist()


from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)


import fitz

def extract_pdf_text(path):
    doc = fitz.open(path)
    return "\n".join(page.get_text() for page in doc)


from sqlalchemy import text
from app.db import engine

def insert_chunk(source, chunk_index, content, embedding):
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO documents (
                    source,
                    chunk_index,
                    content,
                    embedding
                )
                VALUES (
                    :source,
                    :chunk_index,
                    :content,
                    :embedding
                )
            """),
            {
                "source": source,
                "chunk_index": chunk_index,
                "content": content,
                "embedding": embedding,
            },
        )