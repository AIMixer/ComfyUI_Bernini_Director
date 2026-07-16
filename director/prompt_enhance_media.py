"""Collect vision inputs and media helpers for director prompt enhancement."""

from __future__ import annotations

import base64
import io
import logging
import os
import re
import subprocess

import folder_paths
import numpy as np
import torch
from PIL import Image

from ..lib.prompt_enhance_templates import (
    DETAILED_MIN_APPEARANCE_HAN,
    DETAILED_MIN_TOTAL_HAN,
    OUTPUT_LANGUAGE_EN,
    normalize_character_feature_enhance,
    normalize_output_language,
)

log = logging.getLogger("ComfyUI-Bernini-Director.director")

LLM_VISION_MAX_SIDE = int(os.environ.get("BERNINI_PE_VISION_MAX_SIDE", "512"))
LLM_VISION_JPEG_QUALITY = int(os.environ.get("BERNINI_PE_VISION_JPEG_QUALITY", "65"))
LLM_VISION_MAX_IMAGES = int(os.environ.get("BERNINI_PE_VISION_MAX_IMAGES", "4"))


def downscale_b64_jpeg(
    b64: str,
    *,
    max_side: int = LLM_VISION_MAX_SIDE,
    quality: int = LLM_VISION_JPEG_QUALITY,
) -> str:
    """Shrink a base64 JPEG for LLM vision to reduce token / context usage."""
    if not b64:
        return b64
    try:
        raw = base64.b64decode(b64.split(",", 1)[-1] if b64.startswith("data:") else b64)
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        w, h = img.size
        if max(w, h) > max_side:
            scale = max_side / max(w, h)
            img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return b64.split(",", 1)[-1] if b64.startswith("data:") else b64


def prepare_llm_vision_images(
    images: list[str] | None,
    *,
    max_images: int = LLM_VISION_MAX_IMAGES,
    max_side: int = LLM_VISION_MAX_SIDE,
    quality: int = LLM_VISION_JPEG_QUALITY,
) -> list[str] | None:
    """Downscale and cap image count for local LLM vision (Ollama context limits)."""
    if not images:
        return None
    out = [
        downscale_b64_jpeg(img, max_side=max_side, quality=quality)
        for img in images
        if img
    ]
    if len(out) <= max_images:
        return out or None
    # Prefer: up to 2 source video frames + remaining reference slots.
    head = out[: min(2, max_images)]
    tail_n = max(0, max_images - len(head))
    trimmed = head + (out[-tail_n:] if tail_n else [])
    log.info(
        "LLM vision: trimmed %d -> %d images (max_side=%d) for context budget",
        len(out),
        len(trimmed),
        max_side,
    )
    return trimmed or None


def _tensor_frame_to_b64(frame: torch.Tensor, *, quality: int = 85) -> str | None:
    if frame is None or not isinstance(frame, torch.Tensor) or frame.numel() == 0:
        return None
    arr = frame.detach().float().cpu().numpy()
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim != 3:
        return None
    arr = np.clip(arr[..., :3], 0.0, 1.0)
    img = Image.fromarray((arr * 255.0).astype(np.uint8), mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def sample_video_tensor_frames(clip: torch.Tensor | None, *, max_frames: int = 3) -> list[str]:
    if clip is None or clip.shape[0] <= 0:
        return []
    total = int(clip.shape[0])
    if total <= max_frames:
        indices = list(range(total))
    else:
        step = total / (max_frames + 1)
        indices = [min(total - 1, max(0, int(step * (i + 1)))) for i in range(max_frames)]
    out: list[str] = []
    for idx in indices:
        b64 = _tensor_frame_to_b64(clip[idx : idx + 1])
        if b64:
            out.append(b64)
    return out


def parse_user_reference_slots(text: str) -> list[int]:
    """Extract Bernini image0–image4 indices from @imageN or imageN in user prompt."""
    body = text or ""
    indices: list[int] = []
    for match in re.finditer(r"@image(\d)(?!\d)", body, re.IGNORECASE):
        index = int(match.group(1))
        if 0 <= index < 5 and index not in indices:
            indices.append(index)
    if indices:
        return sorted(indices)
    for match in re.finditer(r"(?<![@\w])image(\d)(?!\d)", body, re.IGNORECASE):
        index = int(match.group(1))
        if 0 <= index < 5 and index not in indices:
            indices.append(index)
    return sorted(indices)


def filter_vision_for_user_slots(
    images: list[str],
    source_count: int,
    ref_slots: list[int],
    user_slots: list[int],
) -> tuple[list[str], list[int], int]:
    """When user names imageN, keep source frames but send only those reference slots."""
    if not user_slots or not ref_slots or not images:
        return images, ref_slots, source_count
    user_set = set(user_slots)
    n_refs = len(ref_slots)
    sources = images[:source_count]
    refs = images[source_count : source_count + n_refs]
    tail = images[source_count + n_refs :]
    filtered_refs: list[str] = []
    filtered_slots: list[int] = []
    for slot, b64 in zip(ref_slots, refs):
        if slot in user_set:
            filtered_refs.append(b64)
            filtered_slots.append(slot)
    if not filtered_refs:
        return images, ref_slots, source_count
    log.info(
        "LLM vision: user requested image(s) %s — frames + refs %s",
        user_slots,
        filtered_slots,
    )
    return sources + filtered_refs + tail, filtered_slots, source_count


def is_replace_task_prompt(text: str) -> bool:
    """True when user prompt looks like reference-driven replacement."""
    return bool(re.search(r"替换|换成|改为|换为", text or ""))


def reorder_vision_refs_first(
    images: list[str],
    source_count: int,
    ref_slots: list[int],
) -> tuple[list[str], int, list[int], bool]:
    """Put reference images before source frames so VL models anchor on imageN."""
    if source_count <= 0 or not ref_slots:
        return images, source_count, ref_slots, False
    sources = images[:source_count]
    n_refs = len(ref_slots)
    refs = images[source_count : source_count + n_refs]
    tail = images[source_count + n_refs :]
    return refs + sources + tail, source_count, ref_slots, True


def _has_image_tag(text: str, index: int) -> bool:
    """True if imageN appears (incl. image1中 / image1 中的 — \\b misses CJK suffixes)."""
    return bool(re.search(rf"(?<![@\w])image{index}(?!\d)", text or "", re.IGNORECASE))


def _normalize_reference_tags(text: str) -> str:
    """Normalize spacing and remove duplicated imageN phrases."""
    t = (text or "").strip()
    if not t:
        return t
    # image1 中的image1中 → image1 中的
    t = re.sub(
        r"(image(\d))\s*中的\s*\1\s*中",
        r"\1 中的",
        t,
        flags=re.IGNORECASE,
    )
    # image1 中的 image1 → image1 中的
    t = re.sub(
        r"(image(\d))\s*中的\s+\1(?!\d)",
        r"\1 中的",
        t,
        flags=re.IGNORECASE,
    )
    # image1中的 → image1 中的
    t = re.sub(r"(image(\d))中的", r"\1 中的", t, flags=re.IGNORECASE)
    t = re.sub(r"中的的+", "中的", t)
    t = re.sub(r"\s{2,}", " ", t)
    return t.strip()


def ensure_user_reference_tags(text: str, user_slots: list[int]) -> str:
    """Re-inject imageN tags when LLM drops them from enhanced output."""
    if not text or not user_slots:
        return text
    result = _normalize_reference_tags(text)
    for index in user_slots:
        if _has_image_tag(result, index):
            continue
        tag = f"image{index}"
        injected = False
        for pattern, repl in (
            (r"(将[^，。；\n]{0,80}?替换(?:为|成))", rf"\1 {tag} 中的"),
            (r"(换成)", rf"\1 {tag} 中的"),
            (r"(改为)", rf"\1 {tag} 中的"),
            (r"(使用)", rf"\1 {tag} 中的"),
        ):
            new_text, count = re.subn(pattern, repl, result, count=1)
            if count:
                result = new_text
                injected = True
                break
        if not injected:
            result = f"参考 {tag}，{result}"
        log.info("Enhanced prompt missing %s — re-injected reference tag", tag)
    return _normalize_reference_tags(result)


def build_replace_source_target_directive(
    user_slots: list[int],
    *,
    source_count: int = 0,
    output_language: str = OUTPUT_LANGUAGE_EN,
) -> str:
    """Split edit instruction: frame = subject to replace, imageN = replacement target."""
    if not user_slots or source_count <= 0:
        return ""
    slots_txt = "、".join(f"image{s}" for s in user_slots)
    if normalize_output_language(output_language) == "zh":
        return (
            "【替换任务 — 双源分工，严禁对调】\n"
            "扩写句式须为：「将视频中[源视频待替换对象特征]替换为 imageN 中的[参考图目标特征]…」\n"
            f"① **「将视频中…」至「替换为」之前**：只描述 frame0/frame1 源视频附件里**当前**待替换对象"
            "的真实可见特征（现行服饰、发型、姿态等）；**必须**来自源视频帧附件，"
            "**禁止**把 imageN 参考图附件里的外观写进这句。\n"
            f"② **「替换为 {slots_txt} 中的…」之后**：只描述 Vision 中对应 imageN 参考图附件里"
            "真实可见的外观（目标替换结果）；**必须**来自参考图附件，"
            "**禁止**把 frame 源视频里待替换对象的外观写进 imageN 句。\n"
            "写每句前先看对应附件：frame 与 imageN 外观不同时，两句分别写对，不得混用、对调或臆造。\n\n"
        )
    return (
        "[Replace task — dual sources]\n"
        f"Before 'replace with imageN': describe subject in source frames (frame0…) only.\n"
        f"After 'from imageN': describe target from reference image {slots_txt} only. Never swap.\n\n"
    )


def build_replace_structured_json_directive(
    ref_slots: list[int],
    *,
    output_language: str = OUTPUT_LANGUAGE_EN,
    character_feature_enhance: bool = False,
) -> str:
    """Force split JSON fields so Python assembles 将视频中…/imageN… without swap."""
    if not ref_slots:
        return ""
    feature_enhance = normalize_character_feature_enhance(character_feature_enhance)
    keys_lines: list[str] = []
    for slot in ref_slots:
        keys_lines.append(
            f'  "image{slot}_target": "ONLY the visible appearance from reference image{slot} attachment"'
        )
    keys_block = ",\n".join(keys_lines)
    if normalize_output_language(output_language) == "zh":
        target_hint = (
            "须大段逐项描写，总汉字≥300"
            if feature_enhance
            else "简洁准确即可"
        )
        return (
            "【替换扩写 — 分字段 JSON，禁止输出 rewritten_text】\n"
            "先逐张查看 Vision 附件（banner 中 frame=源视频，imageN=参考图），再填写 JSON。\n"
            "仅返回一个 JSON 对象，键名固定：\n"
            '  "frame_subject": "ONLY 源视频 frame 附件里待替换对象的可见外观（发型/服饰/场景），'
            "禁止写参考图外观\",\n"
            f"{keys_block}（{target_hint}；禁止写 frame 源视频外观）,\n"
            '  "preservation": "一句保留镜头/背景/光影/运镜",\n'
            '  "scene_detail": "可选，目标成片的场景与氛围（勿重复 frame_subject）"\n'
            "规则：frame_subject 必须来自 frame 图；imageN_target 必须来自 imageN 图；"
            "两者外观不同则分别写对，严禁对调。不要 rewritten_text 字段。\n\n"
        )
    return (
        "[Replace — structured JSON only, NO rewritten_text]\n"
        "Look at each vision attachment (frame* = source video, imageN = reference).\n"
        "Return one JSON object:\n"
        '  "frame_subject": "appearance TO REPLACE from frame attachments ONLY",\n'
        f"{keys_block},\n"
        '  "preservation": "one sentence on unchanged camera/background",\n'
        '  "scene_detail": "optional target scene description"\n'
        "Never swap frame vs imageN appearances. Do not include rewritten_text.\n\n"
    )


def assemble_replace_rv2v_prompt(
    data: dict,
    ref_slots: list[int],
    *,
    output_language: str = OUTPUT_LANGUAGE_EN,
) -> str:
    """Build final replace prompt from structured JSON (guaranteed clause order)."""
    if not data or not ref_slots:
        return ""
    frame = str(data.get("frame_subject") or data.get("frame_appearance") or "").strip()
    preserve = str(data.get("preservation") or "").strip()
    scene = str(data.get("scene_detail") or data.get("scene") or "").strip()
    slot = ref_slots[0]
    target = str(
        data.get(f"image{slot}_target")
        or data.get("reference_target")
        or data.get("image0_target")
        or ""
    ).strip()
    if not frame or not target:
        return ""
    zh = normalize_output_language(output_language) == "zh"
    if zh:
        core = f"将视频中{frame}替换为 image{slot} 中的{target}"
        if preserve:
            core += f"，{preserve}"
        if scene:
            core += f"。{scene}"
    else:
        core = (
            f"Replace {frame} in the video with {target} from image{slot}"
        )
        if preserve:
            core += f", {preserve}"
        if scene:
            core += f". {scene}"
    return _normalize_reference_tags(core)


def build_vision_attachment_banner(
    *,
    source_count: int = 0,
    ref_slots: list[int] | None = None,
    ref_video_count: int = 0,
    output_language: str = OUTPUT_LANGUAGE_EN,
) -> str:
    """Short banner matching actual vision array order: frames first, then refs (official)."""
    ref_slots = list(ref_slots or [])
    if source_count <= 0 and not ref_slots and ref_video_count <= 0:
        return ""
    lines: list[str] = []
    pos = 1
    zh = normalize_output_language(output_language) == "zh"
    if zh:
        lines.append("=== Vision 附件（顺序与下方图片数组一致，扩写前先看图）===")
        for i in range(source_count):
            lines.append(
                f"第{pos}张 = frame{i} 源视频帧 → 只用于 JSON 的 frame_subject /「将视频中…」"
            )
            pos += 1
        for slot in ref_slots:
            lines.append(
                f"第{pos}张 = image{slot} 参考图 → 只用于 JSON 的 image{slot}_target /「image{slot} 中的…」"
            )
            pos += 1
        for _ in range(ref_video_count):
            lines.append(f"第{pos}张 = 参考视频帧（非 image 编号）")
            pos += 1
    else:
        lines.append("=== Vision attachments (same order as image array below) ===")
        for i in range(source_count):
            lines.append(f"#{pos} = frame{i} source video → frame_subject / 'in the video' clause")
            pos += 1
        for slot in ref_slots:
            lines.append(f"#{pos} = image{slot} reference → image{slot}_target / 'from image{slot}'")
            pos += 1
        for _ in range(ref_video_count):
            lines.append(f"#{pos} = reference video frame")
            pos += 1
    return "\n".join(lines) + "\n\n"


def build_user_image_directive(
    user_slots: list[int],
    output_language: str,
    *,
    character_feature_enhance: bool = False,
) -> str:
    if not user_slots:
        return ""
    feature_enhance = normalize_character_feature_enhance(character_feature_enhance)
    if normalize_output_language(output_language) == "zh":
        if len(user_slots) == 1:
            s = user_slots[0]
            detail_hint = (
                f"须大段展开 image{s} 参考图可见外观（单独≥{DETAILED_MIN_APPEARANCE_HAN}汉字、"
                f"全文≥{DETAILED_MIN_TOTAL_HAN}汉字），逐项写清，但每项须在参考图中可见。"
                if feature_enhance
                else "外观须与参考图附件一致，禁止张冠李戴或臆造。"
            )
            return (
                f"【用户指定 @image{s} — 写 image{s} 前先看参考图附件】\n"
                f"扩写须保留 image{s}；「image{s} 中的…」后**仅**写 image{s} 参考图真实外观；{detail_hint}\n"
                f"写 image{s} 前对照参考图确认：① 短发还是长发 ② 上衣款式与颜色（实验服/衬衫/长袍等）"
                f" ③ 现代还是古风场景。**是什么写什么**，禁止默认汉服绿袍。\n"
                f"禁止：把 frame 源视频特征写到 image{s}；禁止「星辰大海」「温润如玉」等无图依据的文学句。\n"
                f"「将视频中…」只写 frame 待替换对象。输出写 image{s}（不要 @ 前缀）。\n\n"
            )
        joined = "、".join(f"@image{s}" for s in user_slots)
        return (
            f"【用户指定 {joined}】\n"
            f"每个 imageN 须保留编号，并写出对应参考图真实可见的外观特征，禁止张冠李戴或臆造。\n"
            f"输出使用 image0/image1…，不要写 @ 前缀，不要出现 slot 一词。\n\n"
        )
    if len(user_slots) == 1:
        s = user_slots[0]
        return (
            f"[User specified @image{s}]\n"
            f"Use image{s} only; appearance details must come from that reference, "
            f"not from other images. Output image{s} without @ prefix.\n\n"
        )
    joined = ", ".join(f"@image{s}" for s in user_slots)
    return (
        f"[User specified {joined}]\n"
        f"Describe each imageN using only its own reference image.\n\n"
    )


# Backward-compatible alias for internal imports
build_user_slot_directive = build_user_image_directive


def build_vision_slot_preamble(
    *,
    source_count: int = 0,
    ref_slots: list[int] | None = None,
    ref_video_count: int = 0,
    ref_images_first: bool = False,
    output_language: str = OUTPUT_LANGUAGE_EN,
) -> str:
    """Explain vision attachment order vs Bernini image0–4 slots for the LLM."""
    ref_slots = list(ref_slots or [])
    if source_count <= 0 and not ref_slots and ref_video_count <= 0:
        return ""
    lines: list[str] = []
    pos = 1
    if normalize_output_language(output_language) == "zh":
        lines.append("【Vision 附件 → 编号映射（扩写必须遵守）】")
        if ref_images_first:
            for slot in ref_slots:
                lines.append(
                    f"- 附件第 {pos} 张 = 参考图 image{slot}"
                    f"（写「image{slot} 中的…」时**只看这张**，逐条对照发型/服饰/颜色）"
                )
                pos += 1
            for i in range(source_count):
                lines.append(
                    f"- 附件第 {pos} 张 = 源视频帧 frame{i}（仅用于「将视频中[待替换对象]」）"
                )
                pos += 1
        else:
            for i in range(source_count):
                lines.append(f"- 附件第 {pos} 张 = 源视频帧 frame{i}（禁止称为 image0/image1）")
                pos += 1
            for slot in ref_slots:
                lines.append(
                    f"- 附件第 {pos} 张 = 参考图 image{slot}"
                    f"（扩写中必须写 image{slot}，不是其他编号）"
                )
                pos += 1
        for _ in range(ref_video_count):
            lines.append(f"- 附件第 {pos} 张 = 参考视频帧（不是 image 编号）")
            pos += 1
        if ref_slots and source_count > 0:
            lines.append(
                "替换分工：frame →「将视频中…」；"
                f"image{ref_slots[0]}"
                + (f"、image{ref_slots[1]}" if len(ref_slots) > 1 else "")
                + " 参考图 →「imageN 中的…」。"
                "参考图与 frame 外观不同时不得混用；写 imageN 禁止套用古风模板。"
            )
        elif ref_slots:
            lines.append(
                f"重要：参考图编号为 image{ref_slots[0]}"
                + (f"、image{ref_slots[1]}" if len(ref_slots) > 1 else "")
                + "…，与附件顺序无关。"
                "源视频帧只能叫 frame0/frame1…，绝不能占用 image0。"
            )
    else:
        lines.append("Vision attachment → index map (MUST follow in output):")
        if ref_images_first:
            for slot in ref_slots:
                lines.append(
                    f"- Attachment #{pos} = reference image image{slot} "
                    f"(for 'from image{slot}' appearance ONLY)"
                )
                pos += 1
            for i in range(source_count):
                lines.append(f"- Attachment #{pos} = source video frame{i}")
                pos += 1
        else:
            for i in range(source_count):
                lines.append(f"- Attachment #{pos} = source video frame{i} (NOT image0/image1)")
                pos += 1
            for slot in ref_slots:
                lines.append(
                    f"- Attachment #{pos} = reference image image{slot} "
                    f"(write exactly image{slot} in prompt)"
                )
                pos += 1
        for _ in range(ref_video_count):
            lines.append(f"- Attachment #{pos} = reference video frame (not imageN)")
            pos += 1
        if ref_slots:
            lines.append(
                f"CRITICAL: use image{ref_slots[0]} etc.; "
                "source frames are frame0/frame1 only — never image0."
            )
    return "\n".join(lines) + "\n\n"


def refs_tensors_to_b64_with_slots(refs) -> tuple[list[str], list[int]]:
    """Return (b64_list, bernini_slot_indices) sorted by slot index."""
    items: list[tuple[int, str]] = []
    for ref in refs or []:
        tensor = getattr(ref, "tensor", None)
        if tensor is None:
            continue
        slot = int(getattr(ref, "index", getattr(ref, "slot", len(items))))
        if tensor.ndim == 4 and tensor.shape[0] > 0:
            b64 = _tensor_frame_to_b64(tensor[0:1])
        else:
            b64 = _tensor_frame_to_b64(tensor)
        if b64:
            items.append((slot, b64))
    items.sort(key=lambda pair: pair[0])
    return [b64 for _, b64 in items], [slot for slot, _ in items]


def refs_tensors_to_b64(refs) -> list[str]:
    images, _ = refs_tensors_to_b64_with_slots(refs)
    return images


def collect_segment_vision_b64(
    *,
    source_clip: torch.Tensor | None,
    refs,
    reference_video: torch.Tensor | None = None,
    max_video_frames: int = 2,
) -> tuple[list[str], int, int, list[int]]:
    """Return (images, ref_count, source_count, ref_slot_indices)."""
    source_frames = sample_video_tensor_frames(source_clip, max_frames=max_video_frames)
    ref_images, ref_slots = refs_tensors_to_b64_with_slots(refs)
    ref_video_frames = sample_video_tensor_frames(reference_video, max_frames=max_video_frames)
    all_images = source_frames + ref_images + ref_video_frames
    return all_images, len(ref_images), len(source_frames), ref_slots


def extract_input_video_frames_b64(
    filename: str,
    *,
    subfolder: str = "",
    num_frames: int = 3,
) -> tuple[list[str], str | None]:
    """Extract uniformly sampled JPEG base64 frames from a file in ComfyUI input/."""
    if not filename:
        return [], "No filename"
    input_dir = folder_paths.get_input_directory()
    safe = os.path.normpath(str(filename)).replace("\\", "/")
    if safe.startswith("..") or os.path.isabs(safe):
        return [], "Invalid filename"
    video_path = os.path.join(input_dir, subfolder, safe) if subfolder else os.path.join(input_dir, safe)
    if not os.path.isfile(video_path):
        return [], f"File not found: {filename}"

    num_frames = max(1, min(int(num_frames), 5))
    try:
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-count_frames",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=nb_read_frames,nb_frames",
                "-of",
                "csv=p=0",
                video_path,
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        total_str = (probe.stdout or "").strip().split(",")[0].strip()
        total_frames = int(total_str) if total_str.isdigit() else 100
        if total_frames <= num_frames:
            indices = list(range(total_frames))
        else:
            step = total_frames / (num_frames + 1)
            indices = [int(step * (i + 1)) for i in range(num_frames)]

        frames_b64: list[str] = []
        for idx in indices:
            result = subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    video_path,
                    "-vf",
                    f"select=eq(n\\,{idx})",
                    "-frames:v",
                    "1",
                    "-f",
                    "image2pipe",
                    "-vcodec",
                    "mjpeg",
                    "-q:v",
                    "4",
                    "pipe:1",
                ],
                capture_output=True,
                timeout=15,
            )
            if result.returncode == 0 and result.stdout:
                frames_b64.append(base64.b64encode(result.stdout).decode("ascii"))
        return frames_b64, None
    except FileNotFoundError:
        return [], "ffmpeg/ffprobe not found"
    except Exception as exc:
        return [], f"{type(exc).__name__}: {exc}"


def load_input_image_b64(filename: str) -> tuple[str | None, str | None]:
    if not filename:
        return None, "No filename"
    input_dir = folder_paths.get_input_directory()
    safe = os.path.normpath(str(filename)).replace("\\", "/")
    if safe.startswith("..") or os.path.isabs(safe):
        return None, "Invalid filename"
    path = os.path.join(input_dir, safe)
    if not os.path.isfile(path):
        return None, f"File not found: {filename}"
    try:
        img = Image.open(path).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("ascii"), None
    except Exception as exc:
        return None, str(exc)
