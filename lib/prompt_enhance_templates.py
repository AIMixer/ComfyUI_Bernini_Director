"""Bernini per-task LLM enhancement templates (aligned with official prompt_enhancer)."""

from __future__ import annotations

import re

from .official_pe_templates import (
    ENHANCE_TEMPLATES,
    JSON_MODE_TASKS,
    T2I_A14B_EN_SYS_PROMPT,
    T2I_SYSTEM_TASKS,
    T2V_A14B_EN_SYS_PROMPT,
    T2V_SYSTEM_TASKS,
)
from .task_prompts import get_task_prompt_spec

OUTPUT_LANGUAGE_EN = "English"
OUTPUT_LANGUAGE_ZH = "中文"

# Legacy combo values (workflows saved before checkbox migration).
CHARACTER_DETAIL_NORMAL = "一般"
CHARACTER_DETAIL_DETAILED = "详尽"

DETAILED_MIN_TOTAL_HAN = 300
DETAILED_MIN_APPEARANCE_HAN = 200

_DEFAULT_USER_TEMPLATE = (
    "You are a helpful assistant that enhances prompts for video generation and editing. "
    "Rewrite the following instruction to be more detailed and specific. "
    "English only.\n\nInstruction: {user_prompt}"
)

_TEMPLATE_ALIASES = {"mv2v": "v2v", "vrc2v": "rv2v"}
_TASKS_REQUIRE_IMAGE_SLOTS = frozenset({"rv2v", "r2v", "r2i", "vi2v"})


def _task_key(task_type: str) -> str:
    return _TEMPLATE_ALIASES.get(task_type, task_type)


def uses_json_mode(task_type: str) -> bool:
    return _task_key(task_type) in JSON_MODE_TASKS


def uses_t2v_system_prompt(task_type: str) -> bool:
    return _task_key(task_type) in T2V_SYSTEM_TASKS


def uses_t2i_system_prompt(task_type: str) -> bool:
    return _task_key(task_type) in T2I_SYSTEM_TASKS


def _append_image_slot_rules(template: str, task_key: str, language: str) -> str:
    key = _task_key(task_key)
    if key not in _TASKS_REQUIRE_IMAGE_SLOTS:
        return template
    if normalize_output_language(language) == "zh":
        rule = (
            "\n\n硬性要求（Bernini 参考图）：官方 prompt token 为 image0、image1…（小写），"
            "与 reference_image_0/1 输入对应；不要写 reference image0。"
            "用户输入 @imageN 时按该编号定位参考图；输出必须保留 imageN，并写出该参考图真实可见的外观特征"
            "（禁止臆造）；不要 @ 前缀，不要 slot 一词。"
            "源视频帧只能叫 frame0/frame1…，禁止占用 image0。"
        )
    else:
        rule = (
            "\n\nCRITICAL (Bernini reference slots): every reference to a reference image "
            'MUST use "image0", "image1", … (lowercase, upload order). '
            'Example: "Replace the man with the woman from image0, preserving pose…". '
            'Do NOT use only "reference image" without the imageN tag. '
            "image0 is the first reference image, not a source video frame."
        )
    if rule.strip() not in template:
        template += rule
    return template


def normalize_output_language(value: str) -> str:
    v = (value or "").strip().lower()
    if v in ("zh", "中文", "chinese", "cn", "简体中文", "chinese (simplified)"):
        return "zh"
    return "en"


def normalize_character_detail_level(value: str) -> str:
    """Legacy: return 'detailed' or 'normal'."""
    v = (value or "").strip().lower()
    if v in ("详尽", "详细", "detailed", "verbose", "full"):
        return "detailed"
    return "normal"


def is_detailed_character_level(level: str) -> bool:
    return normalize_character_detail_level(level) == "detailed"


def normalize_character_feature_enhance(value) -> bool:
    """Parse checkbox / legacy combo / bool."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    v = str(value).strip().lower()
    if v in ("true", "1", "yes", "on", "详尽", "详细", "detailed", "verbose", "full"):
        return True
    if v in ("false", "0", "no", "off", "一般", "normal", ""):
        return False
    return is_detailed_character_level(v)


def is_character_feature_enhance_enabled(
    feature_enhance=None,
    *,
    character_detail_level: str | None = None,
) -> bool:
    if feature_enhance is not None:
        return normalize_character_feature_enhance(feature_enhance)
    if character_detail_level is not None:
        return is_detailed_character_level(character_detail_level)
    return False


def count_han_chars(text: str) -> int:
    return len(re.findall(r"[\u4e00-\u9fff]", text or ""))


def build_character_detail_directive(
    enabled,
    *,
    output_language: str = OUTPUT_LANGUAGE_EN,
    task_key: str = "",
    character_detail_level: str | None = None,
) -> str:
    """Extra LLM instruction when「角色特征增强」is checked."""
    if not is_character_feature_enhance_enabled(
        enabled,
        character_detail_level=character_detail_level,
    ):
        return ""
    key = _task_key(task_key)
    if key not in _TASKS_REQUIRE_IMAGE_SLOTS:
        return ""
    if normalize_output_language(output_language) == "zh":
        return (
            "【角色特征增强 — 必须遵守】\n"
            "rewritten_text 须按三段写，**以 imageN 角色外观特征为核心**：\n"
            "① 一句编辑指令（含 imageN）；\n"
            f"② **大段人物角色外观描写**（单独≥{DETAILED_MIN_APPEARANCE_HAN}汉字，是全文的绝对主体）："
            "写「imageN 中的…」时**只看 imageN 参考图附件**，逐项写清该图中真实可见特征——\n"
            "  · 头部：短发/长发/束发、发色、头饰（若图中可见，无则不写）；\n"
            "  · 面容：脸型、肤色、眉眼唇等可见特征（无妆容则不写妆容）；\n"
            "  · 上装：实际款式（实验服/衬衫/T恤/长袍/汉服等）、颜色、领型袖型；\n"
            "  · 下装/全身：可见部分的颜色款式；腰带/首饰：仅写图中有的；\n"
            "  · 体态与背景类型：现代室内/古风/户外等。\n"
            "**先识别参考图是 modern 还是 ancient**，是什么写什么；"
            "禁止默认汉服古风；禁止把 frame 源视频特征写入 imageN；"
            "禁止无图依据的文学夸张（如「星辰大海」「温润如玉」）。\n"
            "③ 一句保留背景/镜头说明（简短即可）。\n"
            f"全文汉字必须≥{DETAILED_MIN_TOTAL_HAN}，② 须可逐条在参考图中找到依据。\n\n"
            "详尽示例 A（古风参考图）："
            "「…替换为 image1 中的女子。该女子乌黑长发、浅青汉服、金色发饰…与 image1 一致…」\n"
            "详尽示例 B（现代参考图）："
            "「…替换为 image0 中的男子。该男子黑发短发、白色实验服、现代室内…与 image0 一致…」\n\n"
        )
    return (
        "[Character feature enhance — REQUIRED]\n"
        "Structure: (1) one edit line with imageN; (2) a LONG character appearance paragraph "
        f"(≥{DETAILED_MIN_APPEARANCE_HAN} Chinese chars if output is Chinese) covering every visible "
        "trait from the reference image — hair, face, makeup, garments, accessories, pose, fabrics; "
        "(3) one brief preservation line. "
        f"Total ≥{DETAILED_MIN_TOTAL_HAN} Chinese characters if output is Chinese.\n\n"
    )


build_character_feature_enhance_directive = build_character_detail_directive


def build_detailed_retry_suffix(*, current_han: int, output_language: str) -> str:
    if normalize_output_language(output_language) == "zh":
        return (
            f"\n\n【扩写过短，必须重写】当前仅约 {current_han} 汉字，未达角色特征增强要求。"
            f"请重新输出 JSON，rewritten_text 总汉字≥{DETAILED_MIN_TOTAL_HAN}，"
            f"其中 imageN 对应**人物角色**的外观描写单独≥{DETAILED_MIN_APPEARANCE_HAN}汉字，"
            "必须逐项写清：发型/发饰/面容五官/妆容/上装/下装/腰带/首饰/体态/织物质感，"
            "禁止用一两句话或空词概括。"
        )
    return (
        f"\n\n[Too short — rewrite] Current output ~{current_han} chars. "
        f"Provide a much longer rewritten_text (≥{DETAILED_MIN_TOTAL_HAN} chars) with a "
        "dedicated long appearance section from the reference image."
    )


def localize_enhance_template(template: str, language: str) -> str:
    """Rewrite official English-only template instructions for Chinese output."""
    if normalize_output_language(language) != "zh":
        return template
    t = template
    replacements = (
        (r"输出必须是全英文", "输出必须是简体中文"),
        (r"输出必须是英文", "输出必须是简体中文"),
        (r"优质（英文）Prompt", "优质中文 Prompt"),
        (r"改写后的prompt字数控制在60-200字左右", "改写后的 prompt 字数控制在 60-200 汉字左右"),
        (r"\bEnglish only\.?", "仅使用简体中文。"),
        (r"\bin English\b", "使用简体中文"),
        (r"enhanced English prompt", "扩写后的中文提示词"),
        (r"final enhanced English prompt", "最终扩写后的中文提示词"),
        (r"final English prompt", "最终中文提示词"),
        (r"Generate an English prompt", "生成中文提示词"),
        (r"Generate a concise English", "生成简洁的中文"),
        (r"detailed English prompt", "详细的中文提示词"),
        (r"precise English prompt", "精确的中文提示词"),
        (
            r"generate a detailed V2V editing prompt in English",
            "生成详细的中文视频编辑提示词",
        ),
        (
            r"generate a detailed I2I editing prompt in English",
            "生成详细的中文图像编辑提示词",
        ),
        (r"The output must be entirely in English", "输出必须 entirely 使用简体中文"),
        (r"translate the intent into natural English", "将意图翻译为自然流畅的简体中文"),
    )
    for pattern, repl in replacements:
        t = re.sub(pattern, repl, t, flags=re.IGNORECASE)
    if "简体中文" not in t and "中文" not in t:
        suffix = (
            "\n\n输出语言：仅使用简体中文撰写全部扩写内容"
            "（若要求 JSON，则 rewritten_text 字段亦须为中文）。"
        )
        if "image0" in t:
            suffix += "保留 image0、image1 等参考图编号为英文小写，不要翻译成「参考图」。"
        t += suffix
    return t


_RV2V_STATIC_INTRO_RE = re.compile(
    r"1\. The first 3 images are uniformly sampled frames from the \*\*source video\*\*.*?"
    r"3\. An original editing instruction \(which may be in Chinese\)\.\s*\n",
    re.DOTALL,
)


def patch_rv2v_vision_intro(
    template: str,
    *,
    source_count: int,
    ref_slots: list[int],
    ref_images_first: bool = False,
    output_language: str = OUTPUT_LANGUAGE_EN,
) -> str:
    """Replace official static「前3张=源视频」intro with actual attachment order."""
    if source_count <= 0 and not ref_slots:
        return template
    if not _RV2V_STATIC_INTRO_RE.search(template):
        return template

    ref_slots = list(ref_slots or [])
    ref_count = len(ref_slots)
    zh = normalize_output_language(output_language) == "zh"
    lines: list[str] = []
    pos = 1
    if zh:
        lines.append(
            f"【Vision 附件顺序 — 必须以此为准，忽略下文模板中「前 3 张为源视频」等旧描述】"
            f"共 {source_count + ref_count} 张图："
        )
        if ref_images_first:
            for slot in ref_slots:
                lines.append(
                    f"{pos}. 参考图 image{slot}（写「image{slot} 中的…」时**只看本张**）"
                )
                pos += 1
            for i in range(source_count):
                lines.append(
                    f"{pos}. 源视频帧 frame{i}（写「将视频中…待替换对象」时**只看本张**）"
                )
                pos += 1
        else:
            for i in range(source_count):
                lines.append(
                    f"{pos}. 源视频帧 frame{i}（写「将视频中…待替换对象」时**只看本张**）"
                )
                pos += 1
            for slot in ref_slots:
                lines.append(
                    f"{pos}. 参考图 image{slot}（写「image{slot} 中的…」时**只看本张**）"
                )
                pos += 1
        lines.append("用户原始编辑指令如下。")
    else:
        lines.append(
            "Vision attachment order (MUST follow — ignore any other attachment-order text below):"
        )
        if ref_images_first:
            for slot in ref_slots:
                lines.append(
                    f"{pos}. Reference image image{slot} (appearance for 'from image{slot}' ONLY)."
                )
                pos += 1
            for i in range(source_count):
                lines.append(
                    f"{pos}. Source video frame{i} (subject to replace — 'in the video' clause ONLY)."
                )
                pos += 1
        else:
            for i in range(source_count):
                lines.append(
                    f"{pos}. Source video frame{i} (subject to replace — 'in the video' clause ONLY)."
                )
                pos += 1
            for slot in ref_slots:
                lines.append(
                    f"{pos}. Reference image image{slot} (appearance for 'from image{slot}' ONLY)."
                )
                pos += 1
        lines.append("Original editing instruction:")

    intro = "\n".join(lines) + "\n\n"
    return _RV2V_STATIC_INTRO_RE.sub(intro, template, count=1)


def _raw_template(task_key: str) -> str:
    key = _task_key(task_key)
    return ENHANCE_TEMPLATES.get(key, _DEFAULT_USER_TEMPLATE)


def resolve_enhance_template(
    task_type: str,
    *,
    custom_template: str = "",
    output_language: str = OUTPUT_LANGUAGE_EN,
) -> str:
    """Return the user-message template (official by default, localized for 中文)."""
    raw = (custom_template or "").strip()
    key = _task_key(task_type)
    if not raw:
        if key in T2V_SYSTEM_TASKS | T2I_SYSTEM_TASKS:
            return "{user_prompt}"
        raw = _raw_template(key)
        localized = localize_enhance_template(raw, output_language)
        return _append_image_slot_rules(localized, key, output_language)
    localized = localize_enhance_template(raw, output_language)
    return _append_image_slot_rules(localized, key, output_language)


def resolve_enhance_system_prompt(
    task_type: str,
    *,
    custom_template: str = "",
    output_language: str = OUTPUT_LANGUAGE_EN,
) -> str:
    """Official routing: t2v/t2i use A14B system prompts; others use task system prompts."""
    key = _task_key(task_type)
    if (custom_template or "").strip():
        return get_task_prompt_spec(key).system_prompt
    if key in T2V_SYSTEM_TASKS:
        return localize_enhance_template(T2V_A14B_EN_SYS_PROMPT, output_language)
    if key in T2I_SYSTEM_TASKS:
        return localize_enhance_template(T2I_A14B_EN_SYS_PROMPT, output_language)
    return get_task_prompt_spec(key).system_prompt


def format_enhance_user_content(
    template: str,
    *,
    user_prompt: str,
    image_num: int = 1,
) -> str:
    prompt = (user_prompt or "").strip()
    try:
        return template.format(
            user_prompt=prompt,
            original_text=prompt,
            image_num=image_num,
        )
    except KeyError:
        return template.format(user_prompt=prompt, image_num=image_num)


def get_enhance_template(task_type: str, *, output_language: str = OUTPUT_LANGUAGE_EN) -> str:
    return resolve_enhance_template(task_type, output_language=output_language)
