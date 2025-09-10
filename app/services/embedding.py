# File: app/services/embedding.py
# Deskripsi: Mengelola model embedding dan penyimpanan/pemuatan vector store FAISS

import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

VECTOR_STORE_PATH = "app/db/vector_store_langchain"
MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"

print(f"INFO: Memuat model embedding '{MODEL_NAME}'...")
embedding_model = HuggingFaceEmbeddings(model_name=MODEL_NAME)
print("INFO: Model embedding berhasil dimuat.")

def save_vector_store(vector_store):
    print(f"INFO: Menyimpan vector store ke {VECTOR_STORE_PATH}...")
    os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
    vector_store.save_local(VECTOR_STORE_PATH)
    print("INFO: Vector store berhasil disimpan.")

def load_vector_store():
    if not os.path.exists(VECTOR_STORE_PATH):
        print("INFO: Vector store belum dibuat.")
        return None

    print(f"INFO: Memuat vector store dari {VECTOR_STORE_PATH}...")
    return FAISS.load_local(
        VECTOR_STORE_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )
