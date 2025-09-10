# File: app/models/chat.py

from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from enum import Enum
from datetime import datetime


class Role(str, Enum):
    user = "user"
    assistant = "assistant"


class ChatMessage(BaseModel):
    """Model untuk satu pesan dalam riwayat chat."""
    role: Role
    content: str
    timestamp: Optional[datetime] = None
    metadata: Optional[Dict[str, str]] = None


class ChatRequest(BaseModel):
    """Model untuk request ke endpoint /ask."""
    question: str
    session_id: str = Field(..., description="ID unik untuk setiap sesi percakapan.")
    history: Optional[List[ChatMessage]] = Field(None, description="Riwayat percakapan sebelumnya.")
    image_url: Optional[str] = Field(None, description="URL gambar opsional untuk input multimodal.")


class Answer(BaseModel):
    """Model untuk jawaban dari chatbot."""
    text: str
    history: Optional[List[ChatMessage]] = None


class SyncStatus(BaseModel):
    """Model untuk status hasil sinkronisasi."""
    status: str
    message: str
