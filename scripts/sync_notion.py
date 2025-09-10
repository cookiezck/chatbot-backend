# File: scripts/sync_notion.py
# Deskripsi: Skrip untuk menjalankan sinkronisasi Notion secara manual.

import sys
import os
from dotenv import load_dotenv

# Menambahkan path root proyek agar bisa mengimpor dari 'app'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Impor fungsi sinkronisasi setelah path diatur
from app.services.notion_sync import sync_notion_to_vector_store

def run_sync():
    """Fungsi utama untuk menjalankan sinkronisasi."""
    print("="*50)
    print("Memulai proses sinkronisasi manual dari Notion...")
    print("="*50)
    
    load_dotenv()
    result = sync_notion_to_vector_store()
    
    print("\n--- HASIL SINKRONISASI ---")
    print(f"Status: {result['status']}")
    print(f"Pesan: {result['message']}")
    print("--------------------------\n")

if __name__ == "__main__":
    run_sync()
