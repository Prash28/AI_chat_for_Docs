from langchain_community.document_loaders import DirectoryLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

import os
import shutil

CHROMA_PATH = "chroma_db"
DATA_PATH = "data"

def load_documents():
    md_loader = DirectoryLoader(DATA_PATH, glob="*.md")
    md_docs = md_loader.load()
    print("Markdown files loaded")
    pdf_loader = DirectoryLoader(DATA_PATH, glob="*.pdf")
    pdf_docs = pdf_loader.load()
    print("PDF files loaded")

    all_docs = md_docs + pdf_docs
    return all_docs

documents = load_documents()
print("Documents loaded")
# print(documents)

def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function = len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} into {len(chunks)} chunks.")
    return chunks

chunks = split_text(documents)
print("Chunks created")
# print(chunks[4])

def save_to_chroma(chunks :list[Document]):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    
    hfEmbedding = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')
    db = Chroma.from_documents(
        documents=chunks, embedding=hfEmbedding, persist_directory=CHROMA_PATH
    )
    
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

save_to_chroma(chunks)
print("Database created")