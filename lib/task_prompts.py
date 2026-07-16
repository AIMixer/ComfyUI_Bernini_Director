"""Bernini task_type system prompts for T5 text encoding."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskPromptSpec:
    key: str
    label: str
    system_prompt: str
    description_zh: str


TASK_PROMPT_SPECS: tuple[TaskPromptSpec, ...] = (
    TaskPromptSpec(
        "default",
        "默认通用",
        "You are a helpful assistant.",
        "你是一名乐于助人的助手。",
    ),
    TaskPromptSpec(
        "t2i",
        "文生图(Text to Image)",
        "You are a helpful assistant specialized in text-to-image generation.",
        "专攻文生图生成的智能助手。",
    ),
    TaskPromptSpec(
        "t2v",
        "文生视频(Text to Video)",
        "You are a helpful assistant specialized in text-to-video generation.",
        "专攻文生视频生成的智能助手。",
    ),
    TaskPromptSpec(
        "i2i",
        "图生图(Image to Image)",
        "You are a helpful assistant specialized in image editing.",
        "专攻图片编辑的智能助手。",
    ),
    TaskPromptSpec(
        "r2i",
        "参考主体生图(Reference to Image)",
        "You are a helpful assistant specialized in subject-to-image generation.",
        "依托参考主体生成图像的智能助手。",
    ),
    TaskPromptSpec(
        "i2v",
        "图生视频(Image to Video) [实验性]",
        "You are a helpful assistant specialized in image-to-video generation.",
        "【实验性功能】Bernini 官方未提供 i2v 专用示例；源图作为单帧视频作为上下文生成视频。",
    ),
    TaskPromptSpec(
        "v2v",
        "视频转视频(Video to Video)",
        "You are a helpful assistant specialized in video editing.",
        "专攻视频编辑的智能助手。",
    ),
    TaskPromptSpec(
        "r2v",
        "参考主体生视频(Reference to Video)",
        "You are a helpful assistant specialized in subject-to-video generation.",
        "依托参考主体生成视频的智能助手。",
    ),
    TaskPromptSpec(
        "vi2v",
        "内容延展改视频",
        "You are a helpful assistant specialized in video editing on content propagation.",
        "基于画面内容延展进行视频编辑的智能助手。",
    ),
    TaskPromptSpec(
        "rv2v",
        "参考素材改视频",
        "You are a helpful assistant specialized in video editing with reference.",
        "依托参考素材进行视频编辑的智能助手。",
    ),
    TaskPromptSpec(
        "ads2v",
        "广告植入视频",
        "You are a helpful assistant specialized in ads insertion.",
        "专攻广告内容植入视频的智能助手。",
    ),
    TaskPromptSpec(
        "vrc2v",
        "主体位置动作微调",
        "You are a helpful assistant for editing. You may need to adjust the subject's action or position.",
        "视频编辑助手，负责调整画面主体动作、位置。",
    ),
    TaskPromptSpec(
        "mv2v",
        "全参数精细化改视频",
        "You are a helpful assistant for editing. You might need to adjust the video's style, lighting, colors, textures, and the subject's pose or action.",
        "视频编辑助手，可调风格、光影、色彩、材质、人物姿态动作。",
    ),
)

TASK_PROMPT_BY_KEY = {spec.key: spec for spec in TASK_PROMPT_SPECS}
_ALL_SYSTEM_PROMPTS = tuple(spec.system_prompt for spec in TASK_PROMPT_SPECS)

# Reserved for future task types not yet exposed in the Director UI (specs kept for prompt resolution).
HIDDEN_TASK_TYPE_KEYS: frozenset[str] = frozenset()


def task_type_option_label(spec: TaskPromptSpec) -> str:
    return f"{spec.key} — {spec.label}"


def task_type_combo_options() -> tuple[list[str], dict]:
    options = [
        task_type_option_label(spec)
        for spec in TASK_PROMPT_SPECS
        if spec.key not in HIDDEN_TASK_TYPE_KEYS
    ]

    default_spec = TASK_PROMPT_BY_KEY["rv2v"]
    return options, {
        "default": task_type_option_label(default_spec),
        "tooltip": (
            "选择任务类型后，节点会自动在正向提示词前拼接对应的 Bernini 系统提示词（英文），"
            "无需手动输入。您只需要关注正向提示词，无需关心系统提示词。"
        ),
    }


def resolve_task_key(task_type_value: str) -> str:
    value = task_type_value.split(",[object Object]", 1)[0].strip()
    if " · " in value:
        value = value.split(" · ", 1)[0].strip()
    for sep in (" — ", " —— ", " - ", " – "):
        if sep in value:
            return value.split(sep, 1)[0].strip()
    return value


def get_task_prompt_spec(task_type_value: str) -> TaskPromptSpec:
    key = resolve_task_key(task_type_value)
    return TASK_PROMPT_BY_KEY.get(key, TASK_PROMPT_BY_KEY["default"])


def apply_task_system_prompt(task_type_value: str, positive_prompt: str) -> str:
    spec = get_task_prompt_spec(task_type_value)
    system_prompt = spec.system_prompt
    user_prompt = positive_prompt.strip()

    if not system_prompt:
        return positive_prompt
    if not user_prompt:
        return system_prompt
    if user_prompt.startswith(system_prompt):
        return positive_prompt
    for known in _ALL_SYSTEM_PROMPTS:
        if known != system_prompt and user_prompt.startswith(known):
            user_prompt = user_prompt[len(known) :].lstrip()
            break
    return f"{system_prompt} {user_prompt}"
 