"""Qwen3 Realtime TTS WebSocket 流式服务 - 使用 DashScope SDK"""
import asyncio
import base64
import logging
import queue
import threading
from typing import Optional

import dashscope
from dashscope.audio.qwen_tts_realtime import (
    QwenTtsRealtime,
    QwenTtsRealtimeCallback,
    AudioFormat,
)

from app.core.config import get_settings
from app.services.audio_processor import process_audio_chunk_speed

logger = logging.getLogger(__name__)
settings = get_settings()

# 设置 DashScope API Key
dashscope.api_key = settings.dashscope_api_key


class TTSCallback(QwenTtsRealtimeCallback):
    """TTS 回调类,接收实时音频数据"""
    
    def __init__(self):
        self.audio_queue = queue.Queue()
        self.error = None
        self.is_complete = False
        
    def on_open(self) -> None:
        logger.debug("[TTS-Realtime] WebSocket 连接已打开")
    
    def on_close(self, close_status_code, close_msg) -> None:
        logger.info(f"[TTS-Realtime] WebSocket 关闭: code={close_status_code}, msg={close_msg}")
        self.is_complete = True
    
    def on_event(self, response: dict) -> None:
        """处理事件消息"""
        try:
            event_type = response.get('type')
            
            if event_type == 'session.created':
                session_id = response.get('session', {}).get('id')
                logger.info(f"[TTS-Realtime] 会话创建: {session_id}")
            
            elif event_type == 'response.audio.delta':
                # 接收到音频数据 (Base64 编码)
                audio_b64 = response.get('delta')
                if audio_b64:
                    self.audio_queue.put(('audio', audio_b64))
            
            elif event_type == 'response.done':
                logger.info(f"[TTS-Realtime] ✅ TTS 合成完成")
                self.audio_queue.put(('done', None))
                self.is_complete = True
            
            elif event_type == 'error':
                error_msg = response.get('error', {}).get('message', 'Unknown error')
                logger.error(f"[TTS-Realtime] ❌ 错误: {error_msg}")
                self.error = error_msg
                self.audio_queue.put(('error', error_msg))
                
        except Exception as e:
            logger.exception(f"[TTS-Realtime] 处理事件异常: {e}")
            self.error = str(e)
            self.audio_queue.put(('error', str(e)))


class RealtimeTTSService:
    """Qwen3 TTS Realtime 服务 (使用 DashScope SDK)"""
    
    def __init__(self):
        self.conversation: Optional[QwenTtsRealtime] = None
        
    def synthesize_streaming(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
        pitch: Optional[float] = None,
        volume: Optional[float] = None,
    ):
        """
        流式合成语音
        
        Yields:
            str: Base64 编码的 PCM 音频数据
        """
        callback = TTSCallback()
        
        # 创建 TTS realtime 连接
        logger.info(f"[TTS-Realtime] 🎤 开始合成语音:")
        logger.info(f"[TTS-Realtime]   文本: {text}")
        logger.info(f"[TTS-Realtime]   音色: {voice or settings.tts_voice_name}")
        logger.info(f"[TTS-Realtime]   语速: {speed if speed is not None else settings.tts_speed}")
        
        try:
            conversation = QwenTtsRealtime(
                model=settings.default_tts_voice,
                callback=callback,
                url="wss://dashscope.aliyuncs.com/api-ws/v1/realtime",
            )
            
            # 连接
            conversation.connect()
            logger.info("[TTS-Realtime] WebSocket 连接成功")
            
            # 更新会话参数
            conversation.update_session(
                voice=voice or settings.tts_voice_name,
                response_format=AudioFormat.PCM_24000HZ_MONO_16BIT,
                speech_rate=speed if speed is not None else settings.tts_speed,
                pitch_rate=pitch if pitch is not None else settings.tts_pitch,
                volume=int((volume if volume is not None else settings.tts_volume) * 50),  # 转换到 0-100
            )
            
            # 发送文本
            conversation.append_text(text)
            conversation.commit()
            logger.debug("[TTS-Realtime] 已发送 TTS 请求")
            
            # 接收音频流
            audio_chunks = 0
            total_bytes = 0
            
            while not callback.is_complete:
                try:
                    event_type, data = callback.audio_queue.get(timeout=10.0)
                    
                    if event_type == 'error':
                        raise RuntimeError(f"TTS 错误: {data}")
                    
                    elif event_type == 'audio':
                        # Base64 编码的音频数据
                        audio_chunks += 1
                        audio_bytes = base64.b64decode(data)
                        total_bytes += len(audio_bytes)
                        
                        if audio_chunks == 1:
                            logger.info(f"[TTS-Realtime] 🔊 收到第一个音频块 ({len(audio_bytes)} 字节)")
                        elif audio_chunks % 10 == 0:
                            logger.debug(f"[TTS-Realtime] 已接收 {audio_chunks} 个音频块,累计 {total_bytes} 字节")
                        
                        yield data  # 直接返回 Base64 数据
                    
                    elif event_type == 'done':
                        logger.info(f"[TTS-Realtime] ✅ 合成完成,共 {audio_chunks} 个音频块,总计 {total_bytes} 字节")
                        break
                        
                except queue.Empty:
                    logger.warning("[TTS-Realtime] 等待音频数据超时")
                    break
            
            # 关闭连接
            conversation.close()
            
        except Exception as e:
            logger.exception(f"[TTS-Realtime] 合成失败: {e}")
            if 'conversation' in locals():
                conversation.close()
            raise


# 全局实例
_tts_service: Optional[RealtimeTTSService] = None


def get_tts_service() -> RealtimeTTSService:
    """获取 TTS 服务实例 (单例)"""
    global _tts_service
    if _tts_service is None:
        _tts_service = RealtimeTTSService()
    return _tts_service


async def synthesize_text(
    text: str,
    voice: Optional[str] = None,
    speed: Optional[float] = None,
    pitch: Optional[float] = None,
    volume: Optional[float] = None,
):
    """
    合成语音并流式返回 Base64 编码的音频块
    
    Yields:
        str: Base64 编码的 PCM 音频数据
    """
    service = get_tts_service()
    
    # 在线程中运行同步方法,避免阻塞事件循环
    loop = asyncio.get_event_loop()
    
    # 使用线程池执行器运行同步生成器
    def _sync_synthesize():
        return list(service.synthesize_streaming(
            text=text,
            voice=voice,
            speed=speed,
            pitch=pitch,
            volume=volume,
        ))
    
    # 在线程池中执行
    audio_chunks = await loop.run_in_executor(None, _sync_synthesize)
    
    # 逐个返回音频块
    for chunk in audio_chunks:
        # 如果启用了音频后处理,进行加速处理
        if settings.enable_audio_postprocess and settings.audio_speed_factor != 1.0:
            chunk = process_audio_chunk_speed(
                chunk,
                speed_factor=settings.audio_speed_factor,
                use_phase_vocoder=settings.use_phase_vocoder,
            )
        yield chunk
