"""WebSocket 消息模型"""
from typing import Optional, List
from pydantic import BaseModel


class ChatMessage(BaseModel):
    """对话消息"""
    role: str
    content: str


class ConversationRequest(BaseModel):
    """对话请求"""
    message: str
    conversation_history: Optional[List[ChatMessage]] = []
    llm_model: Optional[str] = None
    tts_voice: Optional[str] = None
    tts_speed: Optional[float] = None


class TTSRequest(BaseModel):
    """TTS 请求"""
    text: str
    voice: Optional[str] = None
    speed: Optional[float] = None
    pitch: Optional[float] = None
    volume: Optional[float] = None


class ErrorResponse(BaseModel):
    """错误响应"""
    error: str
    detail: Optional[str] = None
