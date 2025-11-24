"""FastAPI WebSocket 后端 - Qwen3 多模态实时语音系统"""
import asyncio
import base64
import json
import logging
import time
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from starlette.websockets import WebSocketState
import websockets

from app.core.config import get_settings
from app.services import llm, tts_realtime
from app.services import history_pg
from app.utils.text_splitter import split_text_into_sentences

# 懒加载 GPT-SoVITS TTS
try:
    from app.services import tts_gptsovits
    GPTSOVITS_AVAILABLE = True
except ImportError:
    GPTSOVITS_AVAILABLE = False
    tts_gptsovits = None  # type: ignore

# 懒加载 ASR (避免启动时导入)
try:
    from app.services import asr_realtime
    ASR_AVAILABLE = True
except ImportError:
    ASR_AVAILABLE = False
    asr_realtime = None  # type: ignore

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    logger.info("🚀 Backend Realtime 启动 (多模态 + Agent 版本)")
    logger.info(f"   LLM 模型: {settings.default_llm_model}")
    logger.info(f"   ASR 模型: {settings.default_asr_model} {'✅' if ASR_AVAILABLE else '❌ (未安装)'}")
    logger.info(f"   TTS WebSocket URL: {settings.tts_realtime_url}")
    logger.info(f"   TTS 模型: {settings.default_tts_voice}")
    logger.info(f"   默认音色: {settings.tts_voice_name}")
    logger.info(f"   默认语速: {settings.tts_speed}")
    yield
    logger.info("🛑 Backend Realtime 关闭")


app = FastAPI(title="Qwen3 Multimodal Agent Voice Backend", lifespan=lifespan)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """根路径"""
    return {
        "service": "Qwen3 Multimodal Agent Voice Backend",
        "llm_model": settings.default_llm_model,
        "asr_model": settings.default_asr_model,
        "tts_model": settings.default_tts_voice,
        "asr_available": ASR_AVAILABLE,
        "gptsovits_available": GPTSOVITS_AVAILABLE,
        "features": ["voice_input", "multimodal_vision", "agent_planning", "realtime_tts", "gptsovits_tts"],
    }


def _parse_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).lower() in {"1", "true", "yes", "on"}


async def _recognize_audio_partial(audio_bytes: bytes, sample_rate: int = 16000) -> str:
    """
    调用真实 ASR 服务识别音频。
    
    Args:
        audio_bytes: PCM 音频字节
        sample_rate: 采样率
    
    Returns:
        识别出的文本字符串
    """
    if not ASR_AVAILABLE or not asr_realtime:
        return ""
    
    try:
        # 调用 asr_realtime 进行识别
        text = await asr_realtime.transcribe_audio_stream(
            audio_bytes,
            format='pcm',
            sample_rate=sample_rate,
            enable_vad=False,
        )
        return text or ""
    except Exception as e:
        logger.error(f"[Subtitles] ASR 识别失败: {e}")
        return ""


async def _open_asr_realtime_session(
    audio_format: str,
    sample_rate: int,
    language: str,
    enable_vad: bool,
):
    """建立与 DashScope Realtime ASR 的 WebSocket 连接并完成会话配置。"""
    model = settings.default_asr_model
    base_url = "wss://dashscope.aliyuncs.com/api-ws/v1/realtime"
    ws_url = f"{base_url}?model={model}"
    headers = {
        "Authorization": f"Bearer {settings.dashscope_api_key}",
        "OpenAI-Beta": "realtime=v1",
    }

    logger.info(
        "[Subtitles] 🛰️ 连接 ASR: model=%s, format=%s, sample_rate=%s, language=%s, vad=%s",
        model,
        audio_format,
        sample_rate,
        language,
        enable_vad,
    )

    asr_ws = await websockets.connect(
        ws_url,
        extra_headers=headers,
        ping_interval=None,
    )

    session_config = {
        "event_id": f"event_{int(asyncio.get_event_loop().time() * 1000)}",
        "type": "session.update",
        "session": {
            "modalities": ["text"],
            "input_audio_format": audio_format,
            "sample_rate": sample_rate,
            "input_audio_transcription": {
                "language": language,
            },
        },
    }

    if enable_vad:
        session_config["session"]["turn_detection"] = {
            "type": "server_vad",
            # 更敏感的阈值与更短静音窗口，促使更快出句末结果
            "threshold": 0.15,
            "silence_duration_ms": 300,
        }
    else:
        session_config["session"]["turn_detection"] = None

    await asr_ws.send(json.dumps(session_config))

    # 尝试读取一次响应，主要用于捕获配置异常
    try:
        resp = await asyncio.wait_for(asr_ws.recv(), timeout=5.0)
        try:
            event = json.loads(resp)
            if event.get("type") == "error":
                err = event.get("error", {})
                raise RuntimeError(err.get("message", "ASR 会话配置失败"))
        except json.JSONDecodeError:
            logger.debug("[Subtitles] ASR 配置响应非 JSON，忽略")
    except asyncio.TimeoutError:
        logger.warning("[Subtitles] 等待 ASR 会话配置确认超时，将继续流式转写")

    return asr_ws


@app.websocket("/ws/subtitles")
async def subtitle_stream(websocket: WebSocket):
    """
    实时字幕 WebSocket：支持增量(Partial) + 完整(Final)字幕流式修正。
    
    协议：
    - 前端 → 后端：{ "type": "audio", "data": "<base64 PCM>" } 或 { "type": "commit" }
    - 后端 → 前端：
      - { "type": "subtitle_delta", "text": "<new words>", "is_final": false }  [增量追加]
      - { "type": "subtitle", "text": "<full sentence>", "is_final": true }     [完整覆盖]
    """
    await websocket.accept()

    if not ASR_AVAILABLE:
        await websocket.send_json({"type": "error", "error": "ASR 服务未启用"})
        await websocket.close()
        return

    session_id = websocket.query_params.get("session_id", "subtitle")
    audio_format = websocket.query_params.get("format", settings.asr_format)
    language = websocket.query_params.get("language", "zh")
    enable_vad = _parse_bool(websocket.query_params.get("enable_vad"), True)
    try:
        sample_rate = int(websocket.query_params.get("sample_rate", settings.asr_sample_rate))
    except ValueError:
        sample_rate = settings.asr_sample_rate

    logger.info(
        "[Subtitles] 🔗 新字幕连接: session=%s, format=%s, sr=%s, lang=%s, vad=%s",
        session_id,
        audio_format,
        sample_rate,
        language,
        enable_vad,
    )

    try:
        asr_ws = await _open_asr_realtime_session(
            audio_format=audio_format,
            sample_rate=sample_rate,
            language=language,
            enable_vad=enable_vad,
        )
    except Exception as exc:
        logger.exception("[Subtitles] ASR 会话建立失败: %s", exc)
        await websocket.send_json({"type": "error", "error": f"无法连接 ASR: {exc}"})
        await websocket.close()
        return

    async def safe_send(payload: dict) -> bool:
        if websocket.client_state != WebSocketState.CONNECTED:
            return False
        try:
            await websocket.send_json(payload)
            return True
        except WebSocketDisconnect:
            return False
        except Exception as exc:
            logger.warning("[Subtitles] 发送失败: %s", exc)
            return False

    await safe_send({
        "type": "subtitle_ready",
        "model": settings.default_asr_model,
        "format": audio_format,
        "sample_rate": sample_rate,
        "language": language,
        "enable_vad": enable_vad,
    })

    stop_event = asyncio.Event()

    async def relay_client_audio():
        commit_sent = False
        try:
            while not stop_event.is_set():
                message_raw = await websocket.receive_text()
                logger.debug("[Subtitles] ⇢ 收到客户端消息: %s", message_raw[:120])
                try:
                    payload = json.loads(message_raw)
                except json.JSONDecodeError:
                    await safe_send({"type": "error", "error": "无效的 JSON 消息"})
                    continue

                msg_type = payload.get("type", "audio")
                audio_b64 = payload.get("data", "")

                if msg_type in {"audio", "audio_chunk"}:
                    if not audio_b64:
                        await safe_send({"type": "error", "error": "音频数据为空"})
                        continue

                    event = {
                        "event_id": f"audio_{int(time.time() * 1000)}",
                        "type": "input_audio_buffer.append",
                        "audio": audio_b64,
                    }
                    logger.debug("[Subtitles] ⇢ 转发音频块 (len=%d)", len(audio_b64))
                    await asr_ws.send(json.dumps(event))

                elif msg_type in {"end", "commit", "stop"}:
                    if not enable_vad:
                        logger.info("[Subtitles] 📝 收到结束/提交信号: %s", msg_type)
                        commit_event = {
                            "event_id": f"commit_{int(time.time() * 1000)}",
                            "type": "input_audio_buffer.commit",
                        }
                        await asr_ws.send(json.dumps(commit_event))
                        commit_sent = True

                elif msg_type == "ping":
                    logger.debug("[Subtitles] Pong → 客户端")
                    await safe_send({"type": "pong"})

                else:
                    await safe_send({"type": "error", "error": f"不支持的消息类型: {msg_type}"})

        except WebSocketDisconnect:
            logger.info("[Subtitles] 客户端断开: %s", session_id)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.exception("[Subtitles] 转发音频失败: %s", exc)
            await safe_send({"type": "error", "error": f"音频转发失败: {exc}"})
        finally:
            if not commit_sent and not enable_vad:
                try:
                    await asr_ws.send(json.dumps({
                        "event_id": f"commit_{int(time.time() * 1000)}",
                        "type": "input_audio_buffer.commit",
                    }))
                except Exception:
                    pass
            stop_event.set()

    async def relay_asr_results():
        try:
            while not stop_event.is_set():
                try:
                    resp = await asyncio.wait_for(asr_ws.recv(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                except websockets.exceptions.ConnectionClosed:
                    break

                try:
                    event = json.loads(resp)
                except json.JSONDecodeError:
                    logger.warning("[Subtitles] 无法解析 ASR 响应: %s", resp)
                    continue

                event_type = event.get("type", "")
                transcript = event.get("transcript") or event.get("text") or ""
                delta_text = event.get("delta") or ""

                if event_type.startswith("conversation.item.input_audio_transcription"):
                    is_final = event_type.endswith("completed")
                    if transcript:
                        logger.info(
                            "[Subtitles] 📝 字幕: %s | final=%s",
                            transcript,
                            is_final,
                        )
                        await safe_send({
                            "type": "subtitle",
                            "text": transcript,
                            "is_final": is_final,
                            "event": event_type,
                            "timestamp_ms": int(time.time() * 1000),
                        })
                    if is_final:
                        logger.debug("[Subtitles] 段落结束")
                        await safe_send({"type": "subtitle_segment_end"})
                elif event_type == "response.audio_transcript.delta" and delta_text:
                    logger.debug("[Subtitles] 🔤 增量字幕: %s", delta_text)
                    await safe_send({
                        "type": "subtitle_delta",
                        "text": delta_text,
                        "is_final": False,
                        "event": event_type,
                        "timestamp_ms": int(time.time() * 1000),
                    })
                elif enable_vad and event_type in {"input_audio_buffer.committed", "input_audio_buffer.stopped"}:
                    logger.debug("[Subtitles] 服务端 VAD 段落结束: %s", event_type)
                    await safe_send({"type": "subtitle_segment_end"})
                elif event_type == "error":
                    err_msg = event.get("error", {}).get("message", "ASR 错误")
                    logger.error("[Subtitles] ASR 错误: %s", err_msg)
                    await safe_send({"type": "error", "error": err_msg})
                    break
                elif event_type == "done":
                    break

        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.exception("[Subtitles] 读取 ASR 结果失败: %s", exc)
            await safe_send({"type": "error", "error": f"ASR 结果流失败: {exc}"})
        finally:
            stop_event.set()

    tasks = [
        asyncio.create_task(relay_client_audio()),
        asyncio.create_task(relay_asr_results()),
    ]

    try:
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    finally:
        stop_event.set()
        for task in tasks:
            task.cancel()
        try:
            await asr_ws.close()
        except Exception:
            pass
        await safe_send({"type": "subtitle_done"})
        logger.info("[Subtitles] 会话结束: %s", session_id)


async def collect_and_play_tts(
    text: str,
    voice: str,
    speed: float,
    websocket: WebSocket,
    tts_engine: str = "qwen"
) -> List[str]:
    """
    收集 TTS 音频并发送给前端播放
    
    Args:
        text: 待合成文本
        voice: 音色（仅 qwen-tts 使用）
        speed: 语速（仅 qwen-tts 使用）
        websocket: WebSocket 连接
        tts_engine: TTS 引擎 ("qwen" 或 "gptsovits")
    
    Returns:
        所有音频块的 Base64 列表
    """
    start_time = time.perf_counter()
    requested_engine = tts_engine or "qwen"
    engines_to_try: List[str] = []

    def _add_engine(name: str):
        if name not in engines_to_try:
            engines_to_try.append(name)

    if requested_engine == "gptsovits":
        if GPTSOVITS_AVAILABLE:
            _add_engine("gptsovits")
        else:
            logger.warning("[TTS] GPT-SoVITS 不可用，自动回退到 Qwen TTS")
        _add_engine("qwen")
    else:
        _add_engine("qwen")

    last_error: Optional[Exception] = None

    async def _stream_with_engine(engine_name: str) -> List[str]:
        streamed_chunks: List[str] = []
        if engine_name == "gptsovits":
            logger.info("[TTS] 使用 GPT-SoVITS 引擎")
            async for audio_b64 in tts_gptsovits.synthesize_text_stream(text):
                await websocket.send_json({"type": "audio", "data": audio_b64})
                streamed_chunks.append(audio_b64)
        else:
            logger.info("[TTS] 使用 Qwen TTS 引擎")
            async for audio_b64 in tts_realtime.synthesize_text(
                text=text,
                voice=voice,
                speed=speed,
            ):
                await websocket.send_json({"type": "audio", "data": audio_b64})
                streamed_chunks.append(audio_b64)
        return streamed_chunks

    for engine_name in engines_to_try:
        try:
            streamed = await _stream_with_engine(engine_name)
            if streamed:
                elapsed = time.perf_counter() - start_time
                logger.info(
                    "[TTS] 合成完成，共 %d 个音频块，用时 %.2fs，使用引擎=%s",
                    len(streamed),
                    elapsed,
                    engine_name,
                )
                return streamed
            logger.warning("[TTS] 引擎 %s 未返回音频数据，尝试下一候选", engine_name)
        except Exception as exc:
            last_error = exc
            logger.warning("[TTS] 引擎 %s 合成失败: %s", engine_name, exc)

    elapsed = time.perf_counter() - start_time
    if last_error:
        logger.error(f"[TTS] 所有 TTS 引擎合成失败 (耗时 {elapsed:.2f}s): {last_error}")
    else:
        logger.error(f"[TTS] TTS 引擎未返回音频数据 (耗时 {elapsed:.2f}s)")
    await websocket.send_json({
        "type": "error",
        "error": "TTS 合成失败，请稍后重试",
    })
    return []


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket 主入口 - 支持语音+文本+图像多模态输入"""
    await websocket.accept()
    connection_active = True

    async def safe_send(payload: dict) -> bool:
        """在连接仍然存活时发送消息，若失败则标记并停止后续发送。"""
        nonlocal connection_active
        if not connection_active:
            return False
        if websocket.client_state != WebSocketState.CONNECTED:
            connection_active = False
            logger.warning(
                "[WebSocket] 客户端已断开，跳过消息 type=%s",
                payload.get("type"),
            )
            return False
        try:
            await websocket.send_json(payload)
            return True
        except WebSocketDisconnect:
            connection_active = False
            raise
        except Exception as exc:  # RuntimeError/ClientDisconnected 等
            connection_active = False
            logger.warning(
                "[WebSocket] 发送消息失败 (type=%s): %s",
                payload.get("type"),
                exc,
            )
            return False
    
    # 全局会话ID（不因新窗口变化）
    session_id = settings.global_session_id
    
    logger.info(f"[WebSocket] 🔗 新连接建立,会话ID: {session_id}")
    
    # 从 SQLite 历史库加载最近窗口（全局合并）
    conversation_history: List[dict] = history_pg.fetch_recent_dialogs(
        session_id=session_id,
        limit_pairs=settings.history_context_window,
        include_all_sessions=True,
    )
    logger.info(f"[WebSocket] 📚 已加载历史对话(来自 DB): {len(conversation_history)} 条")
    
    try:
        while True:
            # 接收客户端消息
            data = await websocket.receive_text()
            process_start = time.perf_counter()
            logger.info(f"[WebSocket] 📨 收到消息,会话ID: {session_id}")
            
            try:
                request_data = json.loads(data)
                msg_type = request_data.get("type")
                
                # ===== 处理不同类型的输入 =====
                user_message = ""
                images: Optional[List[str]] = None
                
                # 1. 音频输入 (ASR)
                if msg_type == "audio":
                    if not ASR_AVAILABLE:
                        if not await safe_send({
                            "type": "error",
                            "error": "ASR 服务未启用，请安装 dashscope[audio]"
                        }):
                            break
                        continue
                    
                    audio_b64 = request_data.get("data", "")
                    if not audio_b64:
                        if not await safe_send({"type": "error", "error": "音频数据为空"}):
                            break
                        continue
                    
                    logger.info(f"[WebSocket] 🎤 处理音频输入")
                    asr_start = time.perf_counter()
                    
                    try:
                        # 调用 ASR 识别
                        user_message = await asr_realtime.transcribe_audio_base64(
                            audio_b64,
                            format=settings.asr_format,
                            sample_rate=settings.asr_sample_rate,
                        )
                        asr_elapsed = time.perf_counter() - asr_start
                        logger.info(f"[WebSocket] 🕒 ASR 识别耗时: {asr_elapsed:.2f}s")
                        
                        if not user_message:
                            if not await safe_send({
                                "type": "error",
                                "error": "未识别到任何语音内容"
                            }):
                                break
                            continue
                        
                        logger.info(f"[WebSocket] 🗣️ ASR 识别结果: {user_message}")
                        
                        # 发送识别结果给前端
                        if not await safe_send({
                            "type": "asr_result",
                            "text": user_message,
                        }):
                            break
                        
                    except Exception as e:
                        logger.exception(f"[WebSocket] ASR 识别失败: {e}")
                        if not await safe_send({
                            "type": "error",
                            "error": f"语音识别失败: {str(e)}"
                        }):
                            break
                        continue
                
                # 2. 文本输入
                elif msg_type == "text":
                    user_message = request_data.get("text", "")
                
                # 3. 兼容旧协议 (message 字段)
                else:
                    user_message = request_data.get("message", "")
                
                # 提取图片 (多模态)
                images = request_data.get("images")
                
                if not user_message:
                    if not await safe_send({"type": "error", "error": "消息不能为空"}):
                        break
                    continue
                
                logger.info(f"[WebSocket] 👤 用户消息: {user_message}")
                if images:
                    logger.info(f"[WebSocket] 🖼️  携带图片: {len(images)} 张")
                
                # 保存用户消息到历史上下文
                conversation_history.append({"role": "user", "content": user_message})
                if len(conversation_history) > settings.history_context_window * 2:
                    conversation_history = conversation_history[-settings.history_context_window * 2 :]
                
                # ===== 调用多模态 LLM (非流式，获取完整回复) =====
                logger.info(f"[WebSocket] 🤖 调用多模态 LLM (非流式),历史长度: {len(conversation_history)}")

                # 获取完整 LLM 响应
                llm_start = time.perf_counter()
                # 如果包含图片，禁用 Agent/工具，直接用多模态模型以确保图像被解析
                use_tools = request_data.get("enable_tools", settings.agent_enabled)
                if images:
                    use_tools = False
                full_response = await llm.chat(
                    conversation_history,
                    user_message,
                    images=images,
                    enable_tools=use_tools,
                    stream=False,  # 使用非流式
                )
                llm_elapsed = time.perf_counter() - llm_start
                logger.info(f"[WebSocket] 🧠 LLM 推理耗时: {llm_elapsed:.2f}s")
                
                logger.info(f"[WebSocket] 🤖 LLM 回复完成,长度: {len(full_response)} 字符")
                
                # TTS 配置
                voice = request_data.get("tts_voice", settings.tts_voice_name)
                speed = request_data.get("tts_speed", settings.tts_speed)
                tts_engine = request_data.get("tts_engine", settings.default_tts_engine)
                text_delay = request_data.get("text_delay", settings.text_stream_delay)

                # 句子级切分与口语化
                if tts_engine == "gptsovits":
                    # GPT-SoVITS 需要完整文本来保持与 test_tts.py 一致的语气/音色
                    sentences = [full_response]
                else:
                    sentences = split_text_into_sentences(full_response)
                    if not sentences:
                        sentences = [full_response]

                logger.info(
                    f"[WebSocket] 🎵 开始句子级 TTS 合成，共 {len(sentences)} 句，引擎: {tts_engine}"
                )

                all_audio_chunks: List[str] = []
                for idx, sentence in enumerate(sentences, start=1):
                    logger.info(f"[WebSocket] 🎵 合成第 {idx}/{len(sentences)} 句")
                    sentence_start = time.perf_counter()
                    sentence_audio = await collect_and_play_tts(
                        sentence,
                        voice,
                        speed,
                        websocket,
                        tts_engine=tts_engine,
                    )
                    sentence_elapsed = time.perf_counter() - sentence_start
                    logger.info(f"[WebSocket] 🎧 句子 {idx} 合成耗时: {sentence_elapsed:.2f}s")
                    all_audio_chunks.extend(sentence_audio)

                    if not await safe_send({
                        "type": "text_preview",
                        "text": sentence,
                        "from": "assistant",
                        "sentence_index": idx,
                        "sentence_total": len(sentences),
                    }):
                        connection_active = False
                        break
                    await asyncio.sleep(max(text_delay, 0.05))

                if not connection_active:
                    break

                logger.info(f"[WebSocket] 🎵 句子级 TTS 合成完成，共 {len(all_audio_chunks)} 个音频块")
                
                # 持久化到 SQLite（如启用）
                try:
                    history_pg.save_dialog(
                        session_id=session_id,
                        user_message=user_message,
                        assistant_message=full_response,
                        images=images,
                    )
                except Exception as exc:
                    logger.debug("[HistoryDB] 保存失败: %s", exc)
                
                # 更新内存上下文，保持窗口长度
                conversation_history.append({"role": "assistant", "content": full_response})
                if len(conversation_history) > settings.history_context_window * 2:
                    conversation_history = conversation_history[-settings.history_context_window * 2 :]
                
                # 发送完整文本回复 (用于前端确认)
                if not await safe_send({
                    "type": "final_text",
                    "text": full_response,
                    "from": "assistant",
                }):
                    break
                logger.info(f"[WebSocket] 📤 已发送完整文本")
                
                # 发送音频结束标志
                if not await safe_send({
                    "type": "audio_end",
                }):
                    break
                logger.info(f"[WebSocket] ✅ 流式输出完成")
                total_elapsed = time.perf_counter() - process_start
                logger.info(f"[WebSocket] 🕒 单轮流程总耗时: {total_elapsed:.2f}s")
                
            except json.JSONDecodeError:
                logger.error(f"[WebSocket] JSON 解析失败: {data}")
                if not await safe_send({"error": "无效的 JSON 格式"}):
                    break
            except Exception as e:
                logger.exception(f"[WebSocket] 处理消息时出错: {e}")
                if not await safe_send({
                    "error": f"处理失败: {str(e)}"
                }):
                    break
            finally:
                if "process_start" in locals():
                    total_elapsed = time.perf_counter() - process_start
                    logger.debug(f"[WebSocket] 本轮累计耗时: {total_elapsed:.2f}s (finally)")
    
    except WebSocketDisconnect:
        logger.info(f"[WebSocket] 🔌 连接断开,会话ID: {session_id}")
    except Exception as e:
        logger.exception(f"[WebSocket] 异常断开,会话ID: {session_id}: {e}")
    finally:
        logger.info(f"[WebSocket] 🧹 清理会话,ID: {session_id},历史长度: {len(conversation_history)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8044,
        reload=True,
        log_level="info",
    )
