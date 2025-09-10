# File: app/routes/chat.py
# Deskripsi: Endpoint API untuk chatbot, mendukung multi-user dan multi-turn

from fastapi import APIRouter, HTTPException
from ..models.chat import ChatRequest, Answer, SyncStatus, ChatMessage, Role
from ..services import llm_generator, notion_sync
import logging

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api",
    tags=["Chatbot API"]
)

@router.post("/ask", response_model=Answer)
async def ask_question(request: ChatRequest):
    """
    Endpoint untuk menerima pertanyaan dari user.
    - Mendukung multi-user dengan session_id
    - Mendukung multi-turn dengan history
    """
    if not request.question:
        raise HTTPException(status_code=400, detail="Teks pertanyaan tidak boleh kosong.")

    try:
        # Generate jawaban
        answer_text = llm_generator.generate_answer(
            question=request.question,
            session_id=request.session_id,
            history=request.history or [],
            image_url=request.image_url
        )

        # Ambil session history dan konversi ke ChatMessage Pydantic
        session_data = llm_generator.SESSION_HISTORIES.get(request.session_id, {})
        messages = session_data.get("messages", [])
        session_history_chatmessage = []
        for msg in messages:
            if isinstance(msg, llm_generator.HumanMessage):
                session_history_chatmessage.append(ChatMessage(role=Role.user, content=msg.content))
            elif isinstance(msg, llm_generator.AIMessage):
                session_history_chatmessage.append(ChatMessage(role=Role.assistant, content=msg.content))

        return Answer(text=answer_text, history=session_history_chatmessage)

    except Exception as e:
        logger.error(f"Error saat memproses pertanyaan: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Terjadi kesalahan internal pada server.")

# Endpoint reset session (tombol bersihkan chat)
@router.post("/reset-session/{session_id}")
async def reset_session(session_id: str):
    """
    Endpoint untuk menghapus riwayat session tertentu.
    Bisa dipanggil dari tombol 'Bersihkan Chat' atau otomatis saat idle.
    """
    llm_generator.reset_session(session_id)
    return {"status": "success", "message": f"Session {session_id} berhasil dihapus."}

@router.post("/sync-notion", response_model=SyncStatus)
async def sync_data():
    """
    Memicu proses sinkronisasi data dari Notion ke vector store.
    """
    try:
        result = notion_sync.sync_notion_to_vector_store()
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["message"])
        return SyncStatus(status=result["status"], message=result["message"])
    except Exception as e:
        logger.error(f"Error saat sinkronisasi Notion: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Gagal melakukan sinkronisasi data.")
