"""应用配置"""
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """全局配置"""

    model_config = SettingsConfigDict(
        env_file=str((BASE_DIR.parent / ".env").resolve()),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    dashscope_api_key: str = Field(..., description="DashScope API 密钥")
    qwen_base_url: str = Field(
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
        env="QWEN_BASE_URL",
    )
    default_llm_model: str = Field("qwen3-vl-plus", env="DEFAULT_LLM_MODEL")

    # ASR Realtime 配置
    asr_realtime_url: str = Field(
        "wss://dashscope.aliyuncs.com/api/v1/services/audio/asr/realtime",
        env="ASR_REALTIME_URL",
    )
    default_asr_model: str = Field("paraformer-realtime-v2", env="DEFAULT_ASR_MODEL")
    asr_format: str = Field("pcm", env="ASR_FORMAT")
    asr_sample_rate: int = Field(16000, env="ASR_SAMPLE_RATE")

    # TTS Realtime 配置
    tts_realtime_url: str = Field(
        "wss://dashscope.aliyuncs.com/api/v1/services/audio/tts/realtime",
        env="TTS_REALTIME_URL",
    )
    default_tts_voice: str = Field("qwen3-tts-flash-realtime", env="DEFAULT_TTS_VOICE")
    tts_voice_name: str = Field("Cherry", env="TTS_VOICE_NAME")
    tts_speed: float = Field(1.5, env="TTS_SPEED")
    tts_pitch: float = Field(0.0, env="TTS_PITCH")
    tts_volume: float = Field(1.0, env="TTS_VOLUME")
    tts_format: str = Field("pcm", env="TTS_FORMAT")
    tts_sample_rate: int = Field(24000, env="TTS_SAMPLE_RATE")

    # 音频后处理配置
    enable_audio_postprocess: bool = Field(False, env="ENABLE_AUDIO_POSTPROCESS")
    audio_speed_factor: float = Field(1.0, env="AUDIO_SPEED_FACTOR")
    use_phase_vocoder: bool = Field(False, env="USE_PHASE_VOCODER")

    # 文本流式显示配置
    text_stream_delay: float = Field(0.05, env="TEXT_STREAM_DELAY")
    text_stream_chunk_size: int = Field(3, env="TEXT_STREAM_CHUNK_SIZE")
    
    # TTS 引擎选择
    default_tts_engine: str = Field("qwen", env="DEFAULT_TTS_ENGINE")

    # 角色拟人化配置
    persona_enabled: bool = Field(False, env="PERSONA_ENABLED")
    persona_char_name: str = Field("N.E.X.U.S", env="PERSONA_CHAR_NAME")
    persona_user_name: str = Field("用户", env="PERSONA_USER_NAME")
    persona_backstory: str = Field("", env="PERSONA_BACKSTORY")
    persona_personality: str = Field("", env="PERSONA_PERSONALITY")
    persona_mask: str = Field("", env="PERSONA_MASK")
    persona_examples: str = Field("", env="PERSONA_EXAMPLES")
    persona_custom_prompt: str = Field("", env="PERSONA_CUSTOM_PROMPT")

    # GPT-SoVITS 配置
    gptsovits_api_url: str = Field(
        "http://127.0.0.1:9880/tts",
        env="GPTSOVITS_API_URL"
    )
    gptsovits_ref_audio_path: str = Field(
        "",
        env="GPTSOVITS_REF_AUDIO_PATH"
    )
    gptsovits_prompt_text: str = Field(
        "",
        env="GPTSOVITS_PROMPT_TEXT"
    )
    gptsovits_prompt_lang: str = Field("zh", env="GPTSOVITS_PROMPT_LANG")
    gptsovits_text_lang: str = Field("zh", env="GPTSOVITS_TEXT_LANG")
    gptsovits_top_k: int = Field(15, env="GPTSOVITS_TOP_K")
    gptsovits_batch_size: int = Field(20, env="GPTSOVITS_BATCH_SIZE")
    gptsovits_speed_factor: float = Field(1.0, env="GPTSOVITS_SPEED_FACTOR")
    gptsovits_seed: int = Field(-1, env="GPTSOVITS_SEED")
    gptsovits_text_split_method: str = Field("cut5", env="GPTSOVITS_TEXT_SPLIT_METHOD")

    # 情绪引擎配置
    emotion_enabled: bool = Field(False, env="EMOTION_ENABLED")
    emotion_state_file: str = Field("data/emotion_state.json", env="EMOTION_STATE_FILE")
    emotion_profile_matrix: List[Tuple[float, float, float]] = Field(
        default_factory=lambda: [
            (-1.0, -0.8, -0.05),
            (-0.8, -0.5, 0.03),
            (0.8, 1.0, 0.05),
        ],
        env="EMOTION_PROFILE_MATRIX",
    )
    emotion_frustration_threshold: float = Field(10.0, env="EMOTION_FRUSTRATION_THRESHOLD")
    emotion_frustration_decay: float = Field(0.95, env="EMOTION_FRUSTRATION_DECAY_RATE")
    emotion_max_mood_bonus: float = Field(0.75, env="EMOTION_MAX_MOOD_BONUS")
    emotion_meltdown_minutes: float = Field(90.0, env="EMOTION_MELTDOWN_MINUTES")
    emotion_recovery_minutes: float = Field(10.0, env="EMOTION_RECOVERY_MINUTES")
    emotion_time_scaling_factor: float = Field(5.0, env="EMOTION_TIME_SCALING_FACTOR")
    emotion_sentiment_model: str = Field("qwen2.5-7b-instruct", env="EMOTION_SENTIMENT_MODEL")

    # Agent 配置
    agent_enabled: bool = Field(False, env="AGENT_ENABLED")
    agent_trace_enabled: bool = Field(False, env="AGENT_TRACE_ENABLED")
    agent_recursion_limit: int = Field(20, env="AGENT_RECURSION_LIMIT")

    # 对话历史（数据库）配置
    history_db_enabled: bool = Field(False, env="HISTORY_DB_ENABLED")
    history_db_url: str = Field("", env="HISTORY_DB_URL")  # 例如: postgresql://user:pass@host:5432/dbname
    history_table_name: str = Field("conversation_history", env="HISTORY_TABLE_NAME")
    history_context_window: int = Field(10, env="HISTORY_CONTEXT_WINDOW")  # LLM 上下文窗口（条/turns 计数按 user+assistant）
    # 全局会话标识（避免前端新窗口造成 session_id 变化）
    global_session_id: str = Field("global", env="GLOBAL_SESSION_ID")


@lru_cache()
def get_settings() -> Settings:
    """获取 Settings 单例"""
    return Settings()
