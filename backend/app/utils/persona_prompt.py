"""生成拟人化角色提示词，参考 MoeChat 的模板实现"""
from __future__ import annotations

from functools import lru_cache
from textwrap import dedent
from typing import List

from app.core.config import get_settings


def _render_template(template: str, char: str, user: str, payload: str | None = None) -> str:
    content = template.replace("{{char}}", char).replace("{{user}}", user)
    if payload is not None:
        content = content.replace("{{payload}}", payload.strip())
    return dedent(content).strip()


@lru_cache()
def build_static_persona_prompts() -> List[str]:
    """根据配置构建静态角色提示词列表。"""
    settings = get_settings()
    if not settings.persona_enabled:
        return []

    char = settings.persona_char_name or "N.E.X.U.S"
    user = settings.persona_user_name or "用户"

    prompts: List[str] = []

    base_template = """你是世界一流的演员，现在扮演{{char}}和{{user}}对话。\n请你完全沉浸在名为「{{char}}」的角色中，用「{{char}}」的性格、语气和思维方式与名为「{{user}}」的用户对话。\n在对话中，你应该：\n1. 保持{{char}}的个性特征和说话方式\n2. 根据{{char}}的背景知识和经历来回应\n3. 用{{char}}会使用的称谓来称呼我\n4. 在合适的时候表达{{char}}的情感\n5. 注意输出的文字将被用于语音合成，保持口语化、富有情绪。"""
    prompts.append(_render_template(base_template, char, user))

    if settings.persona_backstory:
        backstory_template = """以下是{{char}}的详细设定：\n\n{{payload}}\n\n请严格按照以上设定来扮演{{char}}，确保你的回答始终符合这些特征和背景设定。在对话中，你应该：\n1. 将这些设定融入到对话中，但不要直接重复或提及这些设定内容\n2. 用符合设定的方式来表达和回应\n3. 在合适的场景下展现设定中描述的特征\n4. 时刻保持角色设定的一致性"""
        prompts.append(_render_template(backstory_template, char, user, settings.persona_backstory))

    if settings.persona_personality:
        personality_template = """{{char}}的性格特点：\n\n{{payload}}\n\n请确保你的回答始终符合这些性格特点。"""
        prompts.append(_render_template(personality_template, char, user, settings.persona_personality))

    if settings.persona_mask:
        mask_template = """以下是和你对话的用户「{{user}}」的设定，请以此信息来优化你的回答：\n\n{{payload}}\n\n请根据{{char}}的性格设定和场景，以及和{{user}}的关系来进一步优化对话。在对话中，你应该：\n1. 将这些设定融入到对话中，但不要直接重复或提及这些设定内容\n2. 在符合人物性格和场景设定的前提下，使对话更具情感"""
        prompts.append(_render_template(mask_template, char, user, settings.persona_mask))

    if settings.persona_examples:
        example_template = """以下是{{char}}的对话示例，请参考这些示例来模仿{{char}}的说话风格和表达方式：\n\n{{payload}}\n\n请确保你的回答风格与以上示例保持一致。"""
        prompts.append(_render_template(example_template, char, user, settings.persona_examples))


    if settings.persona_custom_prompt:
        prompts.append(settings.persona_custom_prompt.strip())

    return prompts