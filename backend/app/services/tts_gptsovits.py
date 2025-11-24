"""GPT-SoVITS 客户端：严格按照 test_tts.py 的调用方式封装"""
from __future__ import annotations

import base64
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator, Dict, Any

import httpx

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

DEBUG_DIR = Path(__file__).resolve().parents[2] / "debug_output" / "gptsovits"
DEBUG_DIR.mkdir(parents=True, exist_ok=True)


def build_payload(text: str) -> Dict[str, Any]:
	"""与 `model/Command/GPT-SoVITS/test_tts.py` 保持完全一致的 payload"""
	cleaned_text = text.strip()
	return {
		"text": cleaned_text,
		"text_lang": settings.gptsovits_text_lang,
		"ref_audio_path": settings.gptsovits_ref_audio_path,
		"prompt_text": settings.gptsovits_prompt_text,
		"prompt_lang": settings.gptsovits_prompt_lang,
		"top_k": settings.gptsovits_top_k,
		"top_p": 1.0,
		"temperature": 1.0,
		"text_split_method": settings.gptsovits_text_split_method,
		"batch_size": settings.gptsovits_batch_size,
		"speed_factor": settings.gptsovits_speed_factor,
		"seed": settings.gptsovits_seed,
		"streaming_mode": False,
		"media_type": "wav",
	}


async def request_tts(payload: Dict[str, Any]) -> bytes:
	"""向 GPT-SoVITS /tts 发起请求，返回 WAV 原始字节"""
	timeout = httpx.Timeout(60.0)
	async with httpx.AsyncClient(timeout=timeout) as client:
		logger.debug("[GPT-SoVITS] -> 请求体: %s", json.dumps(payload, ensure_ascii=False))
		response = await client.post(settings.gptsovits_api_url, json=payload)
		logger.debug("[GPT-SoVITS] <- 状态: %s, 长度: %s bytes", response.status_code, len(response.content))

	if response.status_code != 200:
		raise RuntimeError(
			f"GPT-SoVITS 请求失败: {response.status_code} {response.text}"
		)

	return response.content


def persist_debug_outputs(text: str, payload: Dict[str, Any], audio_bytes: bytes) -> Path:
	"""将调试文本、payload、音频落盘，方便对比 CLI 结果"""
	timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
	folder = DEBUG_DIR / f"tts_{timestamp}"
	folder.mkdir(parents=True, exist_ok=True)

	(folder / "text.txt").write_text(text, encoding="utf-8")
	(folder / "payload.json").write_text(
		json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
	)
	(folder / "output.wav").write_bytes(audio_bytes)

	return folder


async def synthesize_text_stream(text: str) -> AsyncGenerator[str, None]:
	"""对接 WebSocket 层的生成器，yield Base64 WAV 数据。"""
	if not text or not text.strip():
		logger.warning("[GPT-SoVITS] 空文本，跳过合成")
		return

	payload = build_payload(text)
	start_time = time.perf_counter()
	logger.info(
		"[GPT-SoVITS] 即将合成: len=%d, 前50字符=%.50s",
		len(text.strip()),
		text.strip(),
	)
	logger.info("[GPT-SoVITS] POST %s", settings.gptsovits_api_url)
	logger.info(
		"[GPT-SoVITS] 关键参数: ref_audio=%s, prompt=%s, top_k=%s, split=%s",
		settings.gptsovits_ref_audio_path,
		settings.gptsovits_prompt_text,
		settings.gptsovits_top_k,
		settings.gptsovits_text_split_method,
	)
	logger.debug("[GPT-SoVITS] Payload 全量: %s", json.dumps(payload, ensure_ascii=False))

	try:
		audio_bytes = await request_tts(payload)
	except Exception:
		logger.exception("[GPT-SoVITS] 请求失败")
		raise

	debug_folder = persist_debug_outputs(text.strip(), payload, audio_bytes)
	audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
	elapsed = time.perf_counter() - start_time
	logger.info(
		"[GPT-SoVITS] ✅ 合成成功，字节=%d，已写入 %s，用时 %.2fs",
		len(audio_bytes),
		debug_folder,
		elapsed,
	)
	yield audio_b64

 
