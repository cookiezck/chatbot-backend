# File: app/services/notion_sync.py
# Deskripsi: Versi dengan kategori & subkategori di metadata + path diisi ke content

import os
from dotenv import load_dotenv
from notion_client import Client
from notion_client.helpers import collect_paginated_api
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS

# Impor dari file embedding kita
from .embedding import embedding_model, save_vector_store

# --- Konfigurasi Awal ---
load_dotenv()
notion = Client(auth=os.getenv("NOTION_API_KEY"))
TOP_LEVEL_ID = os.getenv("NOTION_PAGE_ID")

# --- Variabel Global ---
all_documents = []
processed_ids = set()


def is_heading(block: dict) -> bool:
    """Memeriksa apakah sebuah blok adalah heading."""
    return block.get("type") in ["heading_1", "heading_2", "heading_3"]


def get_text_from_block(block: dict) -> str:
    """Mengekstrak teks dari berbagai jenis blok konten."""
    block_type = block.get("type")
    text_holding_blocks = [
        "paragraph", "heading_1", "heading_2", "heading_3",
        "bulleted_list_item", "numbered_list_item", "toggle", "quote", "callout"
    ]
    if block_type in text_holding_blocks:
        return "".join([text.get("plain_text", "") for text in block.get(block_type, {}).get("rich_text", [])])
    return ""


def save_chunk(path, title, category, subcategory, texts):
    """Helper untuk simpan chunk dengan path ke page_content."""
    if not texts:
        return
    joined_path = " > ".join(path)
    content_text = "\n".join(texts)  # bikin dulu
    all_documents.append(
        Document(
            page_content=f"Path: {joined_path}\n\n{content_text}",
            metadata={
                "source": joined_path,
                "title": title,
                "category": category,
                "subcategory": subcategory
            }
        )
    )


def process_item_recursively(item_id: str, path: list):
    """Fungsi penjelajah utama dengan kategori & subkategori."""
    if item_id in processed_ids:
        return
    processed_ids.add(item_id)
    print(f"-> Memproses Item: {item_id[:8]} | Path: {'/'.join(path) if path else 'ROOT'}")

    # Coba perlakukan sebagai DATABASE
    try:
        db_pages = collect_paginated_api(notion.databases.query, database_id=item_id)
        print(f"  -> Ditemukan Database. Memproses {len(db_pages)} halaman...")
        for page in db_pages:
            process_item_recursively(page["id"], path)
        return
    except Exception:
        pass

    try:
        # Ambil judul halaman
        current_title = ""
        try:
            page_obj = notion.pages.retrieve(page_id=item_id)
            properties = page_obj.get("properties", {})
            for prop_value in properties.values():
                if prop_value.get("type") == "title":
                    current_title = "".join([t.get("plain_text", "") for t in prop_value.get("title", [])])
                    break

            # Proses relation
            for prop_value in properties.values():
                if prop_value.get("type") == "relation":
                    for relation in prop_value.get("relation", []):
                        process_item_recursively(relation["id"], path + [current_title] if current_title else path)
        except Exception:
            pass

        new_path = path + [current_title] if current_title else path

        # Tentukan kategori dan subkategori
        category = new_path[0] if len(new_path) > 0 else "Uncategorized"
        subcategory = new_path[1] if len(new_path) > 1 else None

        # Ambil anak-anak blok
        children = collect_paginated_api(notion.blocks.children.list, block_id=item_id)
        if not children:
            return

        print(f"    -> Memproses {len(children)} blok anak di dalam '{current_title or item_id[:8]}'")

        current_chunk_texts = []
        for child in children:
            child_id = child["id"]
            child_type = child["type"]

            # Jika ada child page/database
            if child_type in ["child_database", "child_page"]:
                save_chunk(new_path, current_title, category, subcategory, current_chunk_texts)
                current_chunk_texts = []
                process_item_recursively(child_id, new_path)
                continue

            # Ambil teks
            text = get_text_from_block(child).strip()
            if is_heading(child) and text:
                save_chunk(new_path, current_title, category, subcategory, current_chunk_texts)
                current_chunk_texts = [text]
            elif text:
                current_chunk_texts.append(text)

            # Jika ada anak tambahan
            if child.get("has_children"):
                save_chunk(new_path, current_title, category, subcategory, current_chunk_texts)
                current_chunk_texts = []
                process_item_recursively(child_id, new_path)

        # Simpan chunk terakhir
        save_chunk(new_path, current_title, category, subcategory, current_chunk_texts)

    except Exception:
        pass


def sync_notion_to_vector_store():
    """Sinkronisasi Notion -> Vector Store FAISS."""
    global all_documents, processed_ids
    all_documents, processed_ids = [], set()

    if not TOP_LEVEL_ID:
        return {"status": "error", "message": "NOTION_PAGE_ID tidak ditemukan di file .env"}

    print("[INFO] Memulai sinkronisasi Notion...")
    process_item_recursively(TOP_LEVEL_ID, path=[])

    # Hapus duplikat
    unique_docs = []
    seen_content = set()
    for doc in all_documents:
        if doc.page_content not in seen_content:
            unique_docs.append(doc)
            seen_content.add(doc.page_content)

    print(f"\n[INFO] Proses selesai. Total dokumen unik: {len(unique_docs)}")

    if not unique_docs:
        return {"status": "warning", "message": "Tidak ada dokumen untuk di-embed."}

    print("[INFO] Membuat vector store dari dokumen...")
    vector_store = FAISS.from_documents(unique_docs, embedding_model)
    save_vector_store(vector_store)
    return {"status": "success", "message": f"Berhasil sinkron {len(unique_docs)} dokumen."}
