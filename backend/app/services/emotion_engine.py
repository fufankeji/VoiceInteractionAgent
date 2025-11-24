"""MoeChat 风格的情绪引擎，实现拟人化情绪与语气控制。"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

from openai import AsyncOpenAI

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class EmotionState(str, Enum):
    NORMAL = "normal"
    MELTDOWN = "meltdown"
    RECOVERING = "recovering"


@dataclass
class EmotionStatus:
    valence: float = 0.0
    arousal: float = 0.0
    state: EmotionState = EmotionState.NORMAL
    frustration: float = 0.0
    meltdown_started_at: Optional[datetime] = None


def _clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def _negative_valence(valence: float) -> float:
    return abs(valence) if valence < 0 else 0.0


def _acceptance_ratio(valence: float, impact: float, inertia: float = 1.5) -> float:
    resistance = abs(valence) * inertia
    x = impact - resistance
    from math import e
    return 1 / (1 + e ** (-e * x))


def _arousal_permission(arousal: float, k: float = 1.5) -> float:
    permission = (1 - abs(arousal - 0.5)) ** k
    return max(0.0, permission)


def _mood_instruction(valence: float, arousal: float) -> str:
    high_arousal = 0.7
    mid_arousal = 0.4
    high_valence = 0.6
    mid_valence = 0.2
    low_valence = -0.5
    very_low_valence = -0.8

    if valence > high_valence:
        if arousal > high_arousal:
            desc = "极度兴奋或狂喜"
            behavior = "语言极其热情主动，乐于接受所有请求。"
        else:
            desc = "满足或深度喜悦"
            behavior = "语气温暖满足、温柔体贴，乐于配合合理请求。"
    elif valence > mid_valence:
        if arousal > mid_arousal:
            desc = "开朗或愉快"
            behavior = "保持友好、活力充沛的说话方式，多用感叹号。"
        else:
            desc = "平静或放松"
            behavior = "语气柔和放松，给用户安全感。"
    elif valence < very_low_valence:
        if arousal > high_arousal:
            desc = "暴躁或惊恐"
            behavior = "回复尖锐简短，直接拒绝令你不悦的请求。"
        else:
            desc = "深度悲伤或抑郁"
            behavior = "回复极其低落疲惫，用极简词句，拒绝消耗精力的请求。"
    elif valence < low_valence:
        if arousal > mid_arousal:
            desc = "烦躁或焦虑"
            behavior = "语气紧张不耐烦，选择性拒绝部分请求。"
        else:
            desc = "冷漠或疲倦"
            behavior = "语气平淡简短，不主动。"
    else:
        desc = "中性或专注"
        behavior = "保持客观理性的语气，专注完成任务。"

    return (
        "---【情绪模块指令】---\n"
        "请用口语化、富有情绪的方式说话，根据当前心情自然体现语气，不需要输出固定的情绪标签。\n"
        f"当前内在状态：【{desc}】 Valence={valence:.2f}, Arousal={arousal:.2f}。{behavior}"
    )


class EmotionEngine:
    """MoeChat 风格的情绪引擎，负责生成情绪指令。"""

    def __init__(self) -> None:
        self.settings = settings
        self.state = EmotionStatus()
        self._lock = asyncio.Lock()
        self.state_file = Path(self.settings.emotion_state_file).resolve()
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.profile_matrix: List[Tuple[float, float, float]] = self.settings.emotion_profile_matrix
        self.client = AsyncOpenAI(
            api_key=self.settings.dashscope_api_key,
            base_url=str(self.settings.qwen_base_url),
        )
        self.sentiment_model = self.settings.emotion_sentiment_model or self.settings.default_llm_model
        logger.info("[Emotion] Sentiment model in use: %s", self.sentiment_model)
        self._load_state()

    def _load_state(self) -> None:
        if not self.state_file.exists():
            return
        try:
            data = json.loads(self.state_file.read_text(encoding="utf-8"))
            self.state = EmotionStatus(
                valence=data.get("valence", 0.0),
                arousal=data.get("arousal", 0.0),
                state=EmotionState(data.get("state", EmotionState.NORMAL.value)),
                frustration=data.get("frustration", 0.0),
                meltdown_started_at=(
                    datetime.fromisoformat(data["meltdown_started_at"])
                    if data.get("meltdown_started_at")
                    else None
                ),
            )
        except Exception as exc:
            logger.warning("[Emotion] 读取状态失败: %s", exc)

    def _save_state(self) -> None:
        payload = {
            "valence": self.state.valence,
            "arousal": self.state.arousal,
            "state": self.state.state.value,
            "frustration": self.state.frustration,
            "meltdown_started_at": self.state.meltdown_started_at.isoformat()
            if self.state.meltdown_started_at
            else None,
        }
        try:
            self.state_file.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("[Emotion] 保存状态失败: %s", exc)

    async def _call_sentiment_llm(self, text: str) -> tuple[str, float, float]:
        system_prompt = (
            "You are an expert emotion analyzer. Analyze ONLY the last user message. "
            "Return JSON with keys sentiment (positive/negative/neutral), intensity (1-5 float), arousal_impact (-5 to 5)."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ]
        try:
            response = await self.client.chat.completions.create(
                model=self.sentiment_model,
                messages=messages,
                temperature=0.0,
                stream=False,
            )
            content = response.choices[0].message.content  # type: ignore[index]
            if isinstance(content, list):
                content = "".join(block.get("text", "") for block in content)
            start = content.find("{")
            end = content.rfind("}")
            data = json.loads(content[start : end + 1])
            return (
                data.get("sentiment", "neutral"),
                float(data.get("intensity", 0.0)),
                float(data.get("arousal_impact", 0.0)),
            )
        except Exception as exc:
            logger.warning("[Emotion] 情绪分析失败: %s", exc)
            return "neutral", 0.0, 0.0

    def _minutes_since(self, ts: Optional[datetime]) -> float:
        if not ts:
            return 0.0
        return (datetime.now() - ts) / timedelta(minutes=1)

    def _valence_pull(self, valence: float, arousal_impact: float) -> float:
        if arousal_impact > 2.5 and valence > 0.8:
            return 0.05
        for lower, upper, pull in self.profile_matrix:
            if lower == -1.0 and valence <= upper:
                return pull
            if lower < valence <= upper:
                return pull
            if upper == 1.0 and valence > lower:
                return pull
        return 0.0

    async def instruction_for(self, user_text: str) -> str:
        async with self._lock:
            await self._update_state(user_text)
            self._save_state()
            instruction = _mood_instruction(self.state.valence, self.state.arousal)
        logger.info(
            "[Emotion] state=%s V=%.2f A=%.2f frustration=%.2f",
            self.state.state.value,
            self.state.valence,
            self.state.arousal,
            self.state.frustration,
        )
        return instruction

    async def _update_state(self, text: str) -> None:
        if not text.strip():
            return

        if self.state.state == EmotionState.MELTDOWN:
            elapsed = self._minutes_since(self.state.meltdown_started_at)
            if self.state.valence >= -0.3 or elapsed >= self.settings.emotion_meltdown_minutes:
                self.state.state = EmotionState.RECOVERING
                self.state.meltdown_started_at = datetime.now()
            else:
                decay = 1000 / (((elapsed * self.settings.emotion_time_scaling_factor) ** 2) + 1000)
                self.state.arousal = decay
                self.state.valence = -decay
                return

        if self.state.state == EmotionState.RECOVERING:
            elapsed = self._minutes_since(self.state.meltdown_started_at)
            progress = min(elapsed / self.settings.emotion_recovery_minutes, 1.0)
            if progress >= 1.0:
                self.state.state = EmotionState.NORMAL
                self.state.valence = 0.0
                self.state.arousal = 0.0
            else:
                self.state.valence = -0.3 * (1 - progress)
                self.state.arousal = 0.1 * (1 - progress)
            return

        sentiment, intensity, arousal_impact = await self._call_sentiment_llm(text)
        if sentiment == "neutral" or intensity <= 0:
            impact_strength = 0.0
            new_valence = self.state.valence
        else:
            impact_strength = (intensity / 8.1) ** 1.1
            delta = impact_strength if sentiment == "positive" else -impact_strength
            acceptance = _acceptance_ratio(self.state.valence, impact_strength)
            new_valence = self.state.valence + delta * acceptance

        base_delta_arousal = arousal_impact / 10.0
        permission = _arousal_permission(self.state.arousal)
        arousal_delta = base_delta_arousal * permission + self._valence_pull(new_valence, arousal_impact)

        self.state.valence = _clamp(new_valence, -1.0, 1.0)
        self.state.arousal = _clamp(self.state.arousal + arousal_delta, 0.0, 1.0)

        self.state.frustration = self.settings.emotion_frustration_decay * self.state.frustration
        if sentiment == "negative":
            mood_bonus = self.settings.emotion_max_mood_bonus * (_negative_valence(self.state.valence) ** 1.2)
            self.state.frustration += impact_strength * (1 + mood_bonus)
        self.state.frustration += 0.5 * _negative_valence(self.state.valence)

        if self.state.frustration > self.settings.emotion_frustration_threshold:
            self.state.state = EmotionState.MELTDOWN
            self.state.meltdown_started_at = datetime.now()
            self.state.valence = -1.0
            self.state.arousal = 1.0
            self.state.frustration = 0.0


emotion_engine = EmotionEngine()
