"""
文本切句工具 - 复用自 MoeChat
用于将长文本切分成适合 TTS 合成的短句
"""
import re
from typing import Tuple, List


def remove_parentheses_content(text: str) -> str:
    """
    去除文本中的括号及括号内容
    
    Args:
        text: 原始文本
        
    Returns:
        处理后的文本
    """
    # 去除各种括号及其内容
    text = re.sub(r'\(.*?\)|（.*?）|【.*?】|\[.*?\]|\{.*?\}', '', text)
    return text


def split_by_punctuation(text: str, is_first: bool = False) -> Tuple[str, str]:
    """
    按照中文标点切分句子
    
    Args:
        text: 待切分文本
        is_first: 是否是首条消息（影响切分策略）
        
    Returns:
        (完整句子, 剩余缓冲)
    """
    # 中文句子结束符
    sentence_end_pattern = r'[。！？；\n]+'
    
    # 查找第一个句子结束位置
    match = re.search(sentence_end_pattern, text)
    
    if match:
        end_pos = match.end()
        complete_sentence = text[:end_pos].strip()
        remaining = text[end_pos:].strip()
        return complete_sentence, remaining
    else:
        # 没有找到结束符，全部作为缓冲
        return "", text


def remove_parentheses_content_and_split_v2(
    text: str,
    is_first_msg: bool = False
) -> Tuple[str, str]:
    """
    去除括号内容并切分句子（流式处理版本）
    
    Args:
        text: 输入文本（可能是不完整的流式片段）
        is_first_msg: 是否是第一条消息
        
    Returns:
        (完整句子, 剩余缓冲)
    """
    # 1. 去除括号内容
    cleaned_text = remove_parentheses_content(text)
    
    # 2. 按标点切句
    complete, remaining = split_by_punctuation(cleaned_text, is_first_msg)
    
    return complete, remaining


def remove_parentheses_content_and_split(
    text: str,
    is_remove_incomplete: bool = True
) -> list[str]:
    """
    去除括号内容并完整切分（批量处理版本）
    
    Args:
        text: 完整文本
        is_remove_incomplete: 是否移除不完整的句子
        
    Returns:
        句子列表
    """
    # 去除括号
    cleaned_text = remove_parentheses_content(text)
    
    # 按标点切分
    sentences = re.split(r'[。！？；\n]+', cleaned_text)
    
    # 过滤空句子
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # 如果需要移除不完整句子，去掉最后一个（可能不完整）
    if is_remove_incomplete and sentences:
        # 检查原文最后是否有标点
        if not re.search(r'[。！？；\n]$', text):
            sentences = sentences[:-1]
    
    return sentences


def clean_text_for_tts(text: str) -> str:
    """清理文本用于 TTS（尽量保留可读内容）"""

    if not text:
        return ""

    # 移除 Markdown 粗体
    cleaned = re.sub(r'\*\*(.*?)\*\*', r'\1', text)

    # 去掉括号符号但保留内部文字/表情，避免整句被抹掉
    cleaned = re.sub(r'[\(\)（）【】\[\]\{\}]', '', cleaned)

    # 移除多余空格和换行
    cleaned = cleaned.replace(' ', '').replace('\n', '')

    cleaned = cleaned.strip()

    # 如果清理后为空，退回到原始文本（去掉首尾空格）
    if not cleaned:
        return text.strip()
    return cleaned


def split_text_into_sentences(text: str) -> List[str]:
    """将完整文本切分为带标点的完整句子列表"""
    sentences: List[str] = []
    buffer = ""
    for char in text:
        buffer += char
        if char in "。！？!?\n":
            cleaned = buffer.strip()
            if cleaned:
                sentences.append(cleaned)
            buffer = ""
    # 处理剩余内容
    if buffer.strip():
        sentences.append(buffer.strip())
    return sentences
