# File: app/services/llm_generator.py
# Deskripsi: Versi Final Multi-Turn dengan Klasifikasi Niat, Chat History, RAG + HyDE, Qwen API compatible, Session Management
# Catatan: Semua fungsi lama tetap ada, ditambah session relevansi multi-turn dan fallback jawaban untuk pertanyaan tidak relevan

import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI
from langchain_mistralai.chat_models import ChatMistralAI
from .retriever import search_relevant_context
from typing import List, Dict, Optional
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from datetime import datetime, timedelta

load_dotenv()

MAX_HISTORY = 6  # jumlah turn terakhir yang diingat
SESSION_TIMEOUT_MINUTES = 30  # hapus session jika idle lebih dari 30 menit

# ---------------- Session Management ----------------
SESSION_HISTORIES: Dict[str, Dict] = {}  # session_id -> {"messages": List[HumanMessage|AIMessage], "last_active": datetime}

def clean_expired_sessions():
    """Hapus session yang idle lebih dari SESSION_TIMEOUT_MINUTES."""
    now = datetime.now()
    for sid, data in list(SESSION_HISTORIES.items()):
        if data["last_active"] + timedelta(minutes=SESSION_TIMEOUT_MINUTES) < now:
            del SESSION_HISTORIES[sid]

def reset_session(session_id: str):
    """Hapus riwayat session tertentu."""
    if session_id in SESSION_HISTORIES:
        del SESSION_HISTORIES[session_id]

# ---------------- Helper LLM ----------------
def get_llm_instance():
    provider = os.getenv("LLM_PROVIDER")
    if provider == "ollama":
        print("INFO: Menginisialisasi LLM dari Ollama...")
        return Ollama(model=os.getenv("OLLAMA_MODEL"), base_url=os.getenv("OLLAMA_BASE_URL"))
    elif provider == "mistral_api":
        print("INFO: Menginisialisasi LLM dari Mistral API...")
        return ChatMistralAI(model=os.getenv("MISTRAL_API_MODEL"), api_key=os.getenv("MISTRAL_API_KEY"))
    elif provider == "qwen_api":
        print("INFO: Menginisialisasi LLM dari Qwen API (via OpenAI compatible endpoint)...")
        return ChatOpenAI(
            model=os.getenv("QWEN_API_MODEL"),
            api_key=os.getenv("QWEN_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            temperature=0.7,
            max_tokens=512
        )
    else:
        raise ValueError("LLM_PROVIDER tidak valid. Pilih 'ollama', 'mistral_api', atau 'qwen_api'.")

def normalize_history(raw_history, max_history: int = MAX_HISTORY):
    """
    Konversi history menjadi list HumanMessage / AIMessage.
    Bisa menerima list[dict] atau list[ChatMessage].
    """
    normalized = []
    if not raw_history:
        return normalized

    for item in raw_history[-max_history:]:
        # Jika item dict
        if isinstance(item, dict):
            role = item.get("role", "").lower()
            content = item.get("content") or item.get("text", "")
        else:  # item Pydantic ChatMessage
            role = item.role.value.lower()
            content = item.content

        if not content:
            continue
        if "user" in role:
            normalized.append(HumanMessage(content=content))
        elif "assistant" in role or "ai" in role:
            normalized.append(AIMessage(content=content))
        else:
            normalized.append(HumanMessage(content=content))

    return normalized

def classify_intent(question: str, llm) -> str:
    """Menggunakan LLM untuk mengklasifikasikan niat pengguna."""
    print(f"INFO: Mengklasifikasikan niat untuk: '{question}'")
    classifier_prompt = ChatPromptTemplate.from_template(
        "Klasifikasikan input pengguna berikut ke dalam salah satu kategori ini: 'sapaan', 'pertanyaan_spesifik', 'pertanyaan_umum', 'terima_kasih', 'tidak_relevan'. "
        "Jawab HANYA dengan satu kata.\n\n"
        "Input Pengguna: {question}\nOutput:"
    )
    classifier_chain = classifier_prompt | llm | StrOutputParser()
    intent = classifier_chain.invoke({"question": question}).strip().lower().split()[0]
    print(f"INFO: Niat terdeteksi: '{intent}'")
    return intent

def generate_hypothetical_document(question: str, llm) -> str:
    """Membuat dokumen hipotetis untuk pencarian (HyDE)."""
    print(f"INFO: Membuat dokumen hipotetis untuk query: '{question}'")
    hyde_prompt = ChatPromptTemplate.from_template(
        "Tulis paragraf jawaban ideal untuk pertanyaan pengguna berikut. Anggap ini ada di dokumen knowledge base. "
        "Tulis langsung, tanpa pembukaan.\n\nPertanyaan: {question}\nJawaban:"
    )
    hyde_chain = hyde_prompt | llm | StrOutputParser()
    hypothetical_document = hyde_chain.invoke({"question": question})
    print(f"INFO: Dokumen hipotetis dibuat: '{hypothetical_document[:100]}...'")
    return hypothetical_document

# ---------------- Main ----------------
def generate_answer(
    question: str,
    session_id: str,
    history: Optional[List[Dict[str, str]]] = None,
    image_url: Optional[str] = None
) -> str:
    """Fungsi utama multi-turn, mendukung teks atau multimodal + session management + handling pesan pertama."""
    try:
        llm = get_llm_instance()
        chat_history_messages = normalize_history(history)

        # --- Inisialisasi session jika belum ada ---
        if session_id not in SESSION_HISTORIES:
            SESSION_HISTORIES[session_id] = {"messages": [], "last_active": datetime.now()}
        else:
            SESSION_HISTORIES[session_id]["last_active"] = datetime.now()

        # --- Tambahkan history dari frontend ---
        if chat_history_messages:
            SESSION_HISTORIES[session_id]["messages"] += chat_history_messages

        clean_expired_sessions()

        # --- Mode Multimodal ---
        if image_url:
            message_content = [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]
            user_message = HumanMessage(content=message_content)
            response = llm.invoke(SESSION_HISTORIES[session_id]["messages"] + [user_message])
            SESSION_HISTORIES[session_id]["messages"].append(user_message)
            return response.content

        # --- Mode Teks Multi-turn (RAG + HyDE) ---
        intent = classify_intent(question, llm)

        # --- Respons default ---
        if "sapaan" in intent:
            answer_text = "Halo! Saya asisten virtual IOSS. Ada yang bisa saya bantu terkait dokumen panduan?"
        elif "terima_kasih" in intent:
            answer_text = "Sama-sama! Senang bisa membantu."
        elif "pertanyaan_umum" in intent:
            answer_text = "Saya adalah asisten virtual yang menjawab pertanyaan seputar sistem IOSS berdasarkan dokumen panduan."
        elif "tidak_relevan" in intent:
            # --- Ditambahkan fallback multi-turn ---
            answer_text = "Maaf, saya hanya dapat memberikan informasi yang berkaitan dengan panduan sistem IOSS."
        elif "pertanyaan_spesifik" in intent:
            hypothetical_document = generate_hypothetical_document(question, llm)
            context_text = search_relevant_context(hypothetical_document)

            # --- Jika context kosong ---
            if not context_text.strip():
                SESSION_HISTORIES[session_id]["messages"].append(HumanMessage(content=question))
                return "Maaf, saya tidak menemukan informasi tersebut dalam dokumen IOSS."

            # --- Jika context ada, bangun multi-turn messages ---
            system_prompt = """
            Anda adalah asisten AI untuk sistem IOSS.
Jawaban Anda HARUS selalu berdasarkan teks yang diberikan pada bagian 'Konteks Dokumen'. 
Pertimbangkan pertanyaan dan jawaban sebelumnya agar memahami konteks percakapan.
Jangan menebak atau menggunakan pengetahuan luar.

### ATURAN:
1. Jawab hanya dari 'Konteks Dokumen'. Jika tidak relevan, katakan: 
   "Maaf, saya tidak menemukan informasi tersebut dalam dokumen IOSS."
2. Jawaban ringkas, jelas, profesional. Gunakan format point bila perlu.
3. Jangan ulangi pertanyaan pengguna.
4. Jangan menambahkan detail yang tidak ada dalam konteks.
5. Bahasa Indonesia profesional, langsung ke inti, tanpa sapaan.
"""
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Gunakan hanya konteks berikut:\n\n{context_text}"}
            ]

            # --- Tambahkan history yang relevan (MAX_HISTORY terakhir) ---
            for msg in SESSION_HISTORIES[session_id]["messages"][-MAX_HISTORY:]:
                if isinstance(msg, HumanMessage):
                    messages.append({"role": "user", "content": msg.content})
                elif isinstance(msg, AIMessage):
                    messages.append({"role": "assistant", "content": msg.content})

            # --- Pertanyaan terbaru ---
            messages.append({"role": "user", "content": question})

            answer_obj = llm.invoke(messages)
            answer_text = answer_obj.content

        else:
            # --- Fallback jika intent tidak dikenali ---
            answer_text = "Maaf, saya kurang mengerti. Bisa coba tanyakan dengan cara lain?"

        # --- Update session history ---
        SESSION_HISTORIES[session_id]["messages"].append(HumanMessage(content=question))
        SESSION_HISTORIES[session_id]["messages"].append(AIMessage(content=answer_text))

        return answer_text

    except Exception as e:
        print(f"ERROR saat generate_answer: {e}")
        return "Terjadi kesalahan internal saat memproses permintaan Anda."
