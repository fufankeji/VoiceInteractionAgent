"""文本口语化处理工具"""
import random

FILLERS = ["嗯", "对啊", "其实呢", "说真的", "我想想", "行，那"]
ENDINGS = ["啦", "呢", "哦", "呀"]


def to_spoken_style(text: str, is_first: bool = False) -> str:
    """将文本润色为更口语化的表达"""
    spoken = text
    # 简单替换
    spoken = spoken.replace("因此", "所以")
    spoken = spoken.replace("因此", "所以")
    spoken = spoken.replace("首先", "先说")
    spoken = spoken.replace("其次", "另外")
    # 句尾加语气词
    if spoken and spoken[-1] in "。!.!?？":
        spoken = spoken[:-1] + random.choice(ENDINGS)
    # 句首加语气词
    if is_first:
        spoken = f"{random.choice(FILLERS)}，" + spoken
    return spoken