"""Qwen3-VL-Plus 多模态推理封装 (Realtime Backend)"""
from __future__ import annotations

import asyncio
import base64
import logging
import time
from typing import List, Literal, Optional, Dict, Any, Union

from openai import AsyncOpenAI

from app.core.config import get_settings
from app.services.emotion_engine import emotion_engine
from app.services.info_agent import get_info_agent
from app.utils.persona_prompt import build_static_persona_prompts
from app.utils.text_cleaner import strip_stage_directions

logger = logging.getLogger(__name__)

settings = get_settings()
client = AsyncOpenAI(api_key=settings.dashscope_api_key, base_url=str(settings.qwen_base_url))
BASE_SYSTEM_PROMPT = (
    "你是一个强大的多模态 AI 助手，具备视觉理解、对话交互和任务规划能力。请简洁准确地回答用户问题。"
)
NO_STAGE_DIRECTION_PROMPT = (
    "与用户对话时请使用自然口语，不要使用舞台指令、*动作*、括号中的动作描写或旁白，"
    "只用直接的对话句子表达情绪。"
)


class SessionContext:
    """会话上下文 (简化版)"""
    def __init__(self):
        self.history = []


def _build_prompt(
    history: List[dict],
    user_text: str,
    images: Optional[List[str]] = None,
    persona_prompts: Optional[List[str]] = None,
    emotion_instruction: Optional[str] = None,
) -> List[dict]:
    """
    构建多模态提示词
    
    Args:
        history: 历史对话
        user_text: 用户文本
        images: 图片 URL 或 Base64 列表
        persona_prompts: 角色扮演提示词列表
        emotion_instruction: 当前心情指令
    
    Returns:
        OpenAI 格式的消息列表
    """
    system_prompt_blocks: List[str] = [BASE_SYSTEM_PROMPT]
    if persona_prompts:
        system_prompt_blocks.extend(persona_prompts)
    if emotion_instruction:
        system_prompt_blocks.append(emotion_instruction)
    system_prompt_blocks.append(NO_STAGE_DIRECTION_PROMPT)

    messages: List[dict] = [
        {
            "role": "system",
            "content": "\n\n".join(block.strip() for block in system_prompt_blocks if block.strip()),
        }
    ]
    
    # 保留最近6轮历史
    for turn in history[-6:]:
        messages.append({"role": turn["role"], "content": turn["content"]})
    
    # 构建用户消息 (支持多模态)
    if images and len(images) > 0:
        # 多模态消息: 文本 + 图像
        content = [{"type": "text", "text": user_text}]
        
        for img in images:
            if img.startswith("http://") or img.startswith("https://"):
                # URL 图片
                content.append({
                    "type": "image_url",
                    "image_url": {"url": img}
                })
            elif img.startswith("data:image"):
                # Data URL (Base64)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": img}
                })
            else:
                # 纯 Base64 (补充 data URL 前缀)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img}"}
                })
        
        messages.append({"role": "user", "content": content})
        logger.debug(f"[LLM] 构建多模态提示词: 文本 + {len(images)} 张图片")
    else:
        # 纯文本消息
        messages.append({"role": "user", "content": user_text})
        logger.debug(f"[LLM] 构建文本提示词")
    
    logger.debug(f"[LLM] 历史轮数: {len(history[-6:])}, 总消息数: {len(messages)}")
    return messages


async def chat(
    history: List[dict], 
    user_text: str,
    images: Optional[List[str]] = None,
    enable_tools: bool = False,
    stream: bool = False,
):
    """
    调用 Qwen3-VL-Plus 完成多模态对话推理
    
    Args:
        history: 历史对话
        user_text: 用户文本
        images: 图片列表 (URL 或 Base64)
        enable_tools: 是否启用工具调用 (Agent 能力)
        stream: 是否流式返回
    
    Returns:
        如果 stream=True，返回异步生成器 AsyncGenerator[str, None]
        如果 stream=False，返回完整文本 str
    """
    
    chat_start = time.perf_counter()

    persona_prompts = build_static_persona_prompts()
    if persona_prompts:
        logger.info("[LLM] 🧬 Persona prompts active: %d 条", len(persona_prompts))
    elif settings.persona_enabled:
        logger.warning("[LLM] ⚠️ Persona 已启用但未生成提示词, 请检查配置内容")
    else:
        logger.debug("[LLM] Persona 提示词已禁用")
    emotion_instruction: Optional[str] = None
    if settings.emotion_enabled:
        try:
            emotion_instruction = await emotion_engine.instruction_for(user_text)
        except Exception as exc:
            logger.warning("[Emotion] 指令生成失败: %s", exc)
    if emotion_instruction:
        logger.info("[LLM] 💓 Emotion instruction injected")

    messages = _build_prompt(
        history,
        user_text,
        images,
        persona_prompts=persona_prompts,
        emotion_instruction=emotion_instruction,
    )
    payload = {
        "model": settings.default_llm_model,
        "messages": messages,
        "temperature": 0.7,
        "stream": stream,
    }
    
    # TODO: 后续可扩展工具调用
    # if enable_tools:
    #     payload["tools"] = [...]
    
    # 分段日志，突出“历史对话”与“当前请求”，历史部分用块状标记显示（单条 INFO 避免重复时间戳）
    logger.info(f"[LLM] 调用模型: {settings.default_llm_model}, temperature=0.7, stream={stream}, 多模态={bool(images)}")
    logger.info("[LLM] 📤 发送给大模型 (分段展示)")
    # system
    logger.info("[LLM]   [system] %s", messages[0]["content"])
    # history (仅预览，完整内容写到 DEBUG)
    history_slice = history[-6:]
    if history_slice:
        preview_lines = []
        tail_count = min(3, len(history_slice))
        for msg in history_slice[-tail_count:]:
            snippet = str(msg.get("content", ""))[:120]
            tail = "..." if len(snippet) == 120 else ""
            preview_lines.append(f"- role={msg.get('role')}: {snippet}{tail}")
        block = "\n".join(preview_lines) if preview_lines else "无"
        logger.info("[LLM] ===== 历史对话预览 =====")
        logger.info("[LLM]   已加载历史条数: %d，以下仅展示末尾 %d 条", len(history_slice), tail_count)
        logger.info("[LLM] %s", block)
        logger.info("[LLM] ===== 预览结束 =====")
        # 详细内容转到 DEBUG，避免干扰当前交互阅读
        for i, msg in enumerate(history_slice, 1):
            logger.debug("[LLM][history-full] (%d/%d) role=%s content=%s", i, len(history_slice), msg.get("role"), msg.get("content"))
    else:
        logger.info("[LLM] ===== 历史对话预览 =====")
        logger.info("[LLM]   已加载历史条数: 0")
        logger.info("[LLM]   无")
        logger.info("[LLM] ===== 预览结束 =====")
    # current user (包含多模态信息)
    last_msg = messages[-1]
    if isinstance(last_msg.get("content"), list):
        text_parts = [item.get("text", "") for item in last_msg["content"] if item.get("type") == "text"]
        image_count = sum(1 for item in last_msg["content"] if item.get("type") == "image_url")
        logger.info("[LLM]   [user] text=%s | images=%d", " ".join(text_parts), image_count)
    else:
        logger.info("[LLM]   [user] %s", last_msg.get("content"))

    # 只按调用方开关决定是否走 Agent，避免多模态图片被 Agent 拦截
    use_agent = enable_tools
    if use_agent:
        agent_instance = get_info_agent()
        if agent_instance:
            try:
                agent_text = await agent_instance.arun(history, user_text)
                cleaned_agent_text = strip_stage_directions(agent_text)
                logger.info("[LLM] 🤖 Agent 返回内容:")
                logger.info(f"[LLM]   {cleaned_agent_text}")
                logger.info(f"[LLM] 推理成功(Agent),返回长度: {len(cleaned_agent_text)} 字符")
                return cleaned_agent_text
            except Exception as exc:
                logger.warning("[LLM] Agent 执行失败，回退到基础 LLM: %s", exc)
    
    try:
        if stream:
            # 流式模式：返回异步生成器
            return _stream_chat(payload, start_time=chat_start)
        else:
            # 非流式模式：返回完整文本
            response = await client.chat.completions.create(**payload)
            content = response.choices[0].message.content  # type: ignore[index]
            
            if isinstance(content, list):
                # OpenAI SDK 可能返回富文本数组,这里取文本片段并拼接
                result = "".join(block.get("text", "") for block in content)
                logger.debug(f"[LLM] 返回富文本数组,拼接后长度: {len(result)}")
            else:
                result = content or ""
            
            cleaned_result = strip_stage_directions(result)
            if cleaned_result != result:
                logger.info("[LLM] 🧹 Stage directions removed (%d -> %d chars)", len(result), len(cleaned_result))
            logger.info(f"[LLM] 📥 大模型返回的完整内容:")
            logger.info(f"[LLM]   {cleaned_result}")
            logger.info(f"[LLM] 推理成功,返回长度: {len(cleaned_result)} 字符")
            
            # 记录 token 使用情况
            logger.info(f"[LLM] 📥 大模型返回的完整内容:")
            logger.info(f"[LLM]   {cleaned_result}")

            if hasattr(response, 'usage') and response.usage:
                logger.info(f"[LLM] 💰 Token 使用: prompt={response.usage.prompt_tokens}, "
                           f"completion={response.usage.completion_tokens}, "
                           f"total={response.usage.total_tokens}")
            elapsed = time.perf_counter() - chat_start
            logger.info(f"[LLM] ⏱️ 推理耗时: {elapsed:.2f}s")
            
            return cleaned_result
        
    except Exception as exc:
        logger.exception(f"[LLM] 调用失败: {exc}")
        raise


async def _stream_chat(payload: dict, start_time: Optional[float] = None):
    """
    流式聊天的异步生成器
    
    Yields:
        每次生成的文本片段
    """
    logger.info(f"[LLM] 🌊 开始流式推理")
    full_text = ""
    stream_start = start_time if start_time is not None else time.perf_counter()
    
    try:
        response = await client.chat.completions.create(**payload)
        
        async for chunk in response:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if hasattr(delta, 'content') and delta.content:
                    content = delta.content
                    full_text += content
                    logger.debug(f"[LLM] 📤 流式片段: {content}")
                    yield content
        
        elapsed = time.perf_counter() - stream_start
        logger.info(f"[LLM] 🌊 流式推理完成,总长度: {len(full_text)} 字符,耗时 {elapsed:.2f}s")
        
    except Exception as exc:
        logger.exception(f"[LLM] 流式推理失败: {exc}")
        raise
