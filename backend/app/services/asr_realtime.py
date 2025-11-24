"""Qwen3 ASR Flash Realtime - 实时语音识别服务 (WebSocket API)"""
import asyncio
import base64
import json
import logging
from typing import Optional

import websockets

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()




async def transcribe_audio_stream(
    audio_data: bytes,
    format: str = "pcm",
    sample_rate: int = 16000,
    language: str = "zh",
    enable_vad: bool = False,
) -> str:
    """
    使用 Qwen3 ASR Flash Realtime 识别音频流 (WebSocket API)
    
    Args:
        audio_data: PCM 音频数据 (bytes)
        format: 音频格式 (pcm, wav, opus 等)
        sample_rate: 采样率
        language: 语言 (zh, en, etc.)
        enable_vad: 是否启用服务端 VAD
    
    Returns:
        识别的完整文本
    """
    logger.info(
        f"[ASR-Realtime] �️ 开始识别音频流 "
        f"(格式={format}, 采样率={sample_rate}Hz, 大小={len(audio_data)} bytes, VAD={enable_vad})"
    )
    
    # 构建 WebSocket URL
    model = settings.default_asr_model
    base_url = "wss://dashscope.aliyuncs.com/api-ws/v1/realtime"
    ws_url = f"{base_url}?model={model}"
    
    # 构建请求头
    headers = {
        "Authorization": f"Bearer {settings.dashscope_api_key}",
        "OpenAI-Beta": "realtime=v1"
    }
    
    final_text = ""
    error_msg = None
    
    try:
        # 连接 WebSocket
        logger.debug(f"[ASR-Realtime] 🔌 连接到: {ws_url}")
        async with websockets.connect(
            ws_url,
            extra_headers=headers,
            ping_interval=None,
        ) as websocket:
            logger.info("[ASR-Realtime] ✅ WebSocket 连接成功")
            
            # 1. 发送 session.update 事件
            session_config = {
                "event_id": f"event_{int(asyncio.get_event_loop().time() * 1000)}",
                "type": "session.update",
                "session": {
                    "modalities": ["text"],
                    "input_audio_format": format,
                    "sample_rate": sample_rate,
                    "input_audio_transcription": {
                        "language": language,
                    }
                }
            }
            
            # 如果启用 VAD，添加 turn_detection
            if enable_vad:
                session_config["session"]["turn_detection"] = {
                    "type": "server_vad",
                    "threshold": 0.2,
                    "silence_duration_ms": 800
                }
            else:
                session_config["session"]["turn_detection"] = None
            
            logger.debug(f"[ASR-Realtime] � 发送配置: {json.dumps(session_config, ensure_ascii=False)}")
            await websocket.send(json.dumps(session_config))
            
            # 等待配置确认
            response = await websocket.recv()
            logger.debug(f"[ASR-Realtime] � 配置响应: {response}")
            
            # 2. 发送音频数据
            # 将音频数据分块发送 (每次 3200 字节，模拟实时流)
            chunk_size = 3200
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i + chunk_size]
                encoded_chunk = base64.b64encode(chunk).decode('utf-8')
                
                audio_event = {
                    "event_id": f"event_{int(asyncio.get_event_loop().time() * 1000)}",
                    "type": "input_audio_buffer.append",
                    "audio": encoded_chunk
                }
                
                await websocket.send(json.dumps(audio_event))
                logger.debug(f"[ASR-Realtime] � 发送音频块: {len(chunk)} bytes")
                
                # 模拟实时发送 (10ms 间隔)
                await asyncio.sleep(0.01)
            
            # 3. 如果未启用 VAD，发送 commit 事件
            if not enable_vad:
                commit_event = {
                    "event_id": f"event_{int(asyncio.get_event_loop().time() * 1000)}",
                    "type": "input_audio_buffer.commit"
                }
                logger.debug("[ASR-Realtime] 📤 发送 commit 事件")
                await websocket.send(json.dumps(commit_event))
            
            # 4. 接收识别结果
            logger.info("[ASR-Realtime] � 等待识别结果...")
            timeout = 30
            start_time = asyncio.get_event_loop().time()
            
            while asyncio.get_event_loop().time() - start_time < timeout:
                try:
                    # 设置短超时，避免长时间阻塞
                    response = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                    
                    try:
                        event = json.loads(response)
                        event_type = event.get("type", "")
                        
                        logger.debug(
                            f"[ASR-Realtime] � 收到事件: {event_type} | "
                            f"{json.dumps(event, ensure_ascii=False)}"
                        )
                        
                        # 识别转录事件
                        if event_type == "conversation.item.input_audio_transcription.completed":
                            transcript = event.get("transcript", "")
                            if transcript:
                                final_text = transcript
                                logger.info(f"[ASR-Realtime] 📝 识别文本: {transcript}")
                                # 收到完整识别结果，立即退出
                                break
                        
                        # 错误事件
                        elif event_type == "error":
                            error_data = event.get("error", {})
                            error_msg = error_data.get("message", "未知错误")
                            logger.error(f"[ASR-Realtime] ❌ 服务端错误: {error_msg}")
                            break
                        
                        # 会话结束
                        elif event_type in ["conversation.item.input_audio_transcription.failed", "done"]:
                            logger.info("[ASR-Realtime] 🏁 识别流程结束")
                            break
                    
                    except json.JSONDecodeError:
                        logger.warning(f"[ASR-Realtime] ⚠️ 无法解析响应: {response}")
                
                except asyncio.TimeoutError:
                    # 超时但继续等待
                    continue
                except websockets.exceptions.ConnectionClosed:
                    logger.info("[ASR-Realtime] 🔌 WebSocket 连接已关闭")
                    break
            
            # 检查是否超时
            if asyncio.get_event_loop().time() - start_time >= timeout:
                logger.warning("[ASR-Realtime] ⏰ 识别超时")
    
    except Exception as e:
        logger.exception(f"[ASR-Realtime] ❌ WebSocket 通信失败: {e}")
        raise RuntimeError(f"ASR 识别失败: {e}")
    
    # 返回结果
    if error_msg:
        raise RuntimeError(f"ASR 错误: {error_msg}")
    
    if not final_text:
        logger.warning("[ASR-Realtime] ⚠️ 未识别到任何文本")
        return ""
    
    logger.info(f"[ASR-Realtime] ✅ 识别成功: {final_text}")
    return final_text



async def transcribe_audio_base64(
    audio_b64: str,
    format: str = "pcm",
    sample_rate: int = 16000,
    language: str = "zh",
    enable_vad: bool = False,
) -> str:
    """
    识别 Base64 编码的音频
    
    Args:
        audio_b64: Base64 编码的音频数据
        format: 音频格式
        sample_rate: 采样率
        language: 语言
        enable_vad: 是否启用 VAD
    
    Returns:
        识别的文本
    """
    audio_data = base64.b64decode(audio_b64)
    return await transcribe_audio_stream(
        audio_data=audio_data,
        format=format,
        sample_rate=sample_rate,
        language=language,
        enable_vad=enable_vad,
    )


# 兼容旧接口的简化函数
async def recognize_speech(audio_data: bytes) -> str:
    """
    简化的语音识别接口
    
    Args:
        audio_data: PCM16 音频数据 (16kHz, 单声道)
    
    Returns:
        识别的文本
    """
    return await transcribe_audio_stream(
        audio_data=audio_data,
        format="pcm",
        sample_rate=16000,
        language="zh",
        enable_vad=False,
    )

