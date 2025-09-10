# File: app/services/retriever.py
from collections import Counter
from .embedding import load_vector_store

K_NEIGHBORS = 8    # kandidat awal
MAX_DOCS = 20      # batas dokumen final per retrieval


def search_relevant_context(query: str) -> str:
    """
    Cari dokumen relevan dengan fallback:
    1. Topic (paling spesifik)
    2. Subcategory
    3. Category
    """
    vector_store = load_vector_store()
    if vector_store is None:
        print("WARNING: Vector store belum tersedia.")
        return ""

    print(f"INFO: Mencari {K_NEIGHBORS} konteks awal untuk query: '{query}'")
    results = vector_store.similarity_search(query, k=K_NEIGHBORS)

    if not results:
        print("INFO: Tidak ada dokumen yang ditemukan.")
        return ""

    # --- Step 1: Cari topic dominan ---
    topics = [doc.metadata.get("topic") for doc in results if doc.metadata.get("topic")]
    subcategories = [doc.metadata.get("subcategory") for doc in results if doc.metadata.get("subcategory")]
    categories = [doc.metadata.get("category", "Uncategorized") for doc in results]

    chosen_level = None
    chosen_value = None

    if topics:
        top_topic, _ = Counter(topics).most_common(1)[0]
        chosen_level, chosen_value = "topic", top_topic
    elif subcategories:
        top_subcat, _ = Counter(subcategories).most_common(1)[0]
        chosen_level, chosen_value = "subcategory", top_subcat
    else:
        top_cat, _ = Counter(categories).most_common(1)[0]
        chosen_level, chosen_value = "category", top_cat

    print(f"INFO: Level dominan = {chosen_level} | Value = {chosen_value}")

    # --- Step 2: Ambil semua dokumen dalam level dominan ---
    all_in_scope = vector_store.similarity_search(chosen_value, k=MAX_DOCS * 2)
    filtered = [
        doc for doc in all_in_scope
        if doc.metadata.get(chosen_level) == chosen_value
    ]

    # Batasi jumlah final
    final_docs = filtered[:MAX_DOCS]

    # --- Step 3: Format jadi konteks ---
    context_parts = []
    print("\nDEBUG: Hasil retrieval final:")
    for doc in final_docs:
        source = doc.metadata.get("source", "Tidak diketahui")
        part = f"[path: {source}]\n{doc.page_content}"
        context_parts.append(part)
        print(f"- source: {source}")

    print()
    return "\n---\n".join(context_parts)
