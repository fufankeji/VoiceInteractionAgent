"""音频后处理工具 - 实现音频加速等功能"""
import base64
import logging
import numpy as np
from scipy import signal
from typing import Optional

logger = logging.getLogger(__name__)


def decode_pcm_base64(b64_data: str, dtype: str = '<i2') -> np.ndarray:
    """
    解码 Base64 PCM 音频为 numpy 数组
    
    Args:
        b64_data: Base64 编码的 PCM 数据
        dtype: PCM 数据类型,默认 '<i2' (16-bit little-endian)
    
    Returns:
        归一化的音频数组 (float32, -1.0 to 1.0)
    """
    raw_bytes = base64.b64decode(b64_data)
    pcm_int = np.frombuffer(raw_bytes, dtype=dtype)
    # 归一化到 -1.0 to 1.0
    return pcm_int.astype(np.float32) / 32768.0


def encode_pcm_base64(audio_array: np.ndarray, dtype: str = '<i2') -> str:
    """
    编码 numpy 音频数组为 Base64 PCM
    
    Args:
        audio_array: 归一化的音频数组 (float32, -1.0 to 1.0)
        dtype: 目标 PCM 数据类型
    
    Returns:
        Base64 编码的 PCM 字符串
    """
    # 限制范围并转换为 int16
    audio_clipped = np.clip(audio_array, -1.0, 1.0)
    pcm_int = (audio_clipped * 32767.0).astype(dtype)
    return base64.b64encode(pcm_int.tobytes()).decode('utf-8')


def speed_up_audio_resample(
    audio_array: np.ndarray,
    speed_factor: float,
    sample_rate: int = 24000,
) -> np.ndarray:
    """
    使用重采样方法加速音频 (简单快速,但会轻微改变音调)
    
    Args:
        audio_array: 音频数组 (float32, -1.0 to 1.0)
        speed_factor: 加速倍率 (>1.0 = 加速, <1.0 = 减速)
        sample_rate: 采样率
    
    Returns:
        加速后的音频数组
    """
    if speed_factor == 1.0:
        return audio_array
    
    # 计算新长度
    new_length = int(len(audio_array) / speed_factor)
    
    if new_length < 1:
        logger.warning(f"[AudioProcess] 加速倍率太大,音频过短: {new_length}")
        return audio_array
    
    # 使用高质量重采样
    resampled = signal.resample(audio_array, new_length)
    
    logger.debug(f"[AudioProcess] 音频加速: {len(audio_array)} -> {new_length} 样本 ({speed_factor}x)")
    
    return resampled.astype(np.float32)


def speed_up_audio_phase_vocoder(
    audio_array: np.ndarray,
    speed_factor: float,
    hop_length: int = 512,
) -> np.ndarray:
    """
    使用相位声码器方法加速音频 (保持音调,质量较好)
    
    注意: 需要 librosa 库,如果未安装则回退到重采样方法
    
    Args:
        audio_array: 音频数组
        speed_factor: 加速倍率
        hop_length: STFT 帧移
    
    Returns:
        加速后的音频数组
    """
    try:
        import librosa
        
        # librosa.effects.time_stretch 可以保持音调
        stretched = librosa.effects.time_stretch(
            audio_array,
            rate=speed_factor,
            hop_length=hop_length,
        )
        
        logger.debug(f"[AudioProcess] 使用 Phase Vocoder 加速: {speed_factor}x")
        return stretched.astype(np.float32)
        
    except ImportError:
        logger.warning("[AudioProcess] librosa 未安装,使用重采样方法")
        return speed_up_audio_resample(audio_array, speed_factor)


def process_audio_chunk_speed(
    b64_chunk: str,
    speed_factor: float = 1.0,
    use_phase_vocoder: bool = False,
) -> str:
    """
    处理单个 Base64 音频块,应用加速
    
    Args:
        b64_chunk: Base64 编码的 PCM 音频块
        speed_factor: 加速倍率
        use_phase_vocoder: 是否使用相位声码器 (质量更好但更慢)
    
    Returns:
        处理后的 Base64 音频块
    """
    if speed_factor == 1.0:
        return b64_chunk
    
    # 解码
    audio_array = decode_pcm_base64(b64_chunk)
    
    # 加速
    if use_phase_vocoder:
        processed = speed_up_audio_phase_vocoder(audio_array, speed_factor)
    else:
        processed = speed_up_audio_resample(audio_array, speed_factor)
    
    # 编码
    return encode_pcm_base64(processed)


# 批量处理
class AudioSpeedProcessor:
    """音频加速处理器 - 支持流式处理"""
    
    def __init__(
        self,
        speed_factor: float = 1.0,
        use_phase_vocoder: bool = False,
        buffer_size: int = 4800,  # 200ms @ 24kHz
    ):
        self.speed_factor = speed_factor
        self.use_phase_vocoder = use_phase_vocoder
        self.buffer_size = buffer_size
        self.buffer = np.array([], dtype=np.float32)
        
        logger.info(f"[AudioSpeedProcessor] 初始化: speed={speed_factor}, "
                   f"vocoder={use_phase_vocoder}")
    
    def process_chunk(self, b64_chunk: str) -> Optional[str]:
        """
        处理音频块 (流式)
        
        Returns:
            处理后的 Base64 块,如果缓冲区不足则返回 None
        """
        if self.speed_factor == 1.0:
            return b64_chunk
        
        # 解码并添加到缓冲区
        chunk_array = decode_pcm_base64(b64_chunk)
        self.buffer = np.concatenate([self.buffer, chunk_array])
        
        # 如果缓冲区足够,处理一部分
        if len(self.buffer) >= self.buffer_size:
            # 取出一部分处理
            to_process = self.buffer[:self.buffer_size]
            self.buffer = self.buffer[self.buffer_size:]
            
            # 加速处理
            if self.use_phase_vocoder:
                processed = speed_up_audio_phase_vocoder(to_process, self.speed_factor)
            else:
                processed = speed_up_audio_resample(to_process, self.speed_factor)
            
            return encode_pcm_base64(processed)
        
        return None
    
    def flush(self) -> Optional[str]:
        """清空缓冲区,返回剩余数据"""
        if len(self.buffer) == 0:
            return None
        
        if self.use_phase_vocoder:
            processed = speed_up_audio_phase_vocoder(self.buffer, self.speed_factor)
        else:
            processed = speed_up_audio_resample(self.buffer, self.speed_factor)
        
        self.buffer = np.array([], dtype=np.float32)
        return encode_pcm_base64(processed)
