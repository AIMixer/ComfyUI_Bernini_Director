"""First/last-frame (fl2v) timeline → DirectorPlan.

Images sit on a fixed total-frame timeline (LTX-style). Each clip has:
  - isStartFrame: run i2v sampling for this clip's [start, start+length)
  - isEndFrame: used as last-frame reference for a preceding start clip (no sampling)

A start clip looks forward (by timeline order) for the next isEndFrame image as image1.

If that end clip is end-only (not also a start), its placeholder span is absorbed into
the start shot's generation length:
  frameCount = (end.start + end.length) - start.start
e.g. start 118–297 + pure-end 298–315 → sample 198 frames (through 315).
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from ..lib.image_prep import fit_canvas, fit_video_long_edge, resolve_output_dimensions
from ..lib.task_prompts import resolve_task_key, task_type_option_label, TASK_PROMPT_BY_KEY

log = logging.getLogger("ComfyUI-Bernini-Director.director.fl2v")

MIN_FL2V_FRAMES = 4
DEFAULT_FL2V_FRAMES = 81
DEFAULT_TOTAL_FRAMES = 240
# Match node ``negative_prompt`` default in director_common / director.py.
DEFAULT_FL2V_NEGATIVE = "bad video"

# Hard locks for every fl2v shot (re-applied after PE). Community cue words:
# 「完全保持首尾帧」+ 「完全按照image0…」/「完全保持image1」— avoid soft「参考」.
FLF_PROMPT_PREFIX = (
    "完全保持首尾帧。"
    "视频开始完全按照image0的画面，不修改，视频结束完全保持image1的画面。"
    "第一帧必须与image0像素级一致，最后一帧必须与image1像素级一致；"
    "image0/image1是固定首尾帧，不是参考图，禁止改动首尾画面、主体外观与机位。"
)
I2V_PROMPT_PREFIX = (
    "完全保持首帧。"
    "视频开始完全按照image0的画面，不修改。"
    "第一帧必须与image0像素级一致；image0是固定首帧，不是参考图，禁止改动首帧画面与主体外观。"
)
FLF_PROMPT_SUFFIX = (
    "完全保持首尾帧：开头必须是image0，结尾必须是image1。"
)
I2V_PROMPT_SUFFIX = (
    "完全保持首帧：开头必须是image0。"
)

# Legacy wraps from earlier builds — strip so re-runs do not stack.
_LEGACY_FL2V_WRAPS = (
    "完全保持首尾帧。"
    "视频开始完全按照image0的构图，不修改，视频结束完全保持image1。"
    "第一帧必须与image0像素级一致，最后一帧必须与image1像素级一致；"
    "image0/image1是固定首尾帧，不是参考图，禁止改动首尾构图、主体外观与机位。",
    "完全保持首帧。"
    "视频开始完全按照image0的构图，不修改。"
    "第一帧必须与image0像素级一致；image0是固定首帧，不是参考图，禁止改动首帧构图与主体外观。",
    "视频开始完全按照image0的构图，不修改，视频结束完全保持image1。",
    "视频开始完全按照image0的构图，不修改，视频结束完全保持image1的构图。",
    "视频结束完全保持image1的构图。",
    "严格要求：第一帧必须与image0一致，最后一帧必须与image1一致；"
    "中间仅做从image0到image1的自然过渡，不得改变首尾构图与主体外观。",
    "视频开始完全按照image0的构图，不修改。"
    "严格要求：第一帧必须与image0一致；后续运动从该首帧自然展开，不得偏离image0的构图与主体外观。",
    "再次强调：开头锁定image0，结尾锁定image1。",
    "再次强调：开头锁定image0。",
)


def _strip_fl2v_wraps(text: str) -> str:
    changed = True
    while changed and text:
        changed = False
        for p in (FLF_PROMPT_PREFIX, I2V_PROMPT_PREFIX, *_LEGACY_FL2V_WRAPS):
            if text.startswith(p):
                text = text[len(p) :].strip()
                changed = True
        for s in (FLF_PROMPT_SUFFIX, I2V_PROMPT_SUFFIX, *_LEGACY_FL2V_WRAPS):
            if text.endswith(s):
                text = text[: -len(s)].strip()
                changed = True
    return text


def _sanitize_fl2v_body(text: str) -> str:
    """Rewrite soft「参考」wording that weakens first/last-frame locking."""
    if not text:
        return text
    replacements = (
        ("reference image0", "image0"),
        ("reference image1", "image1"),
        ("Reference image0", "image0"),
        ("Reference image1", "image1"),
        ("参考图 image0", "image0"),
        ("参考图 image1", "image1"),
        ("参考图image0", "image0"),
        ("参考图image1", "image1"),
        ("参考 image0", "image0"),
        ("参考 image1", "image1"),
        ("参考image0", "image0"),
        ("参考image1", "image1"),
        ("以image0为参考", "完全按照image0"),
        ("以image1为参考", "完全保持image1"),
        ("把image0当作参考", "完全按照image0"),
        ("把image1当作参考", "完全保持image1"),
        ("image0的构图", "image0的画面"),
        ("image1的构图", "image1的画面"),
        ("首尾构图", "首尾画面"),
        ("首帧构图", "首帧画面"),
    )
    out = text
    for old, new in replacements:
        out = out.replace(old, new)
    # Drop duplicated lock lines already present in the body (prefix/suffix will re-add).
    for marker in (
        "完全保持首尾帧。",
        "完全保持首帧。",
        "视频开始完全按照image0的画面，不修改，视频结束完全保持image1的画面。",
        "视频开始完全按照image0的画面，不修改，视频结束完全保持image1。",
        "视频开始完全按照image0的构图，不修改，视频结束完全保持image1。",
        "视频开始完全按照image0的画面，不修改。",
        "视频开始完全按照image0的构图，不修改。",
        "视频结束完全保持image1的画面。",
        "视频结束完全保持image1。",
    ):
        if out.startswith(marker):
            out = out[len(marker) :].strip()
    return out.strip()


def fl2v_prompt_body_only(prompt: str) -> str:
    """Strip hard-lock wraps / PE duplicates; keep only the motion body for UI storage."""
    text = _sanitize_fl2v_body(_strip_fl2v_wraps((prompt or "").strip()))
    if text.startswith("中间过程："):
        text = text[len("中间过程：") :].strip()
    return text


def reinforce_fl2v_prompt(prompt: str, *, has_end_frame: bool) -> str:
    """Ensure first/last-frame hard constraints wrap the (possibly PE-enhanced) prompt.

    Hard locks are placed *before* the motion body so they survive token truncation.
    """
    text = fl2v_prompt_body_only(prompt)
    prefix = FLF_PROMPT_PREFIX if has_end_frame else I2V_PROMPT_PREFIX
    suffix = FLF_PROMPT_SUFFIX if has_end_frame else I2V_PROMPT_SUFFIX
    if text:
        return f"{prefix}{suffix}中间过程：{text}"
    return f"{prefix}{suffix}"


def is_fl2v_timeline(timeline: dict, task_key: str = "") -> bool:
    mode = str(timeline.get("timelineMode") or "").lower()
    if mode == "fl2v":
        return True
    key = task_key or resolve_task_key(
        (timeline.get("global") or {}).get("taskType") or ""
    )
    return key == "fl2v"


def _load_image_ref(ref: dict) -> torch.Tensor:
    from .gen_timeline import _load_gen_image_tensor

    return _load_gen_image_tensor(ref)


def _fit_image(
    img: torch.Tensor,
    *,
    width: int,
    height: int,
    output_mode: str,
    ref_max_size: int,
) -> torch.Tensor:
    if img.ndim == 3:
        img = img.unsqueeze(0)
    if output_mode == "fixed":
        return fit_canvas(img, width, height)
    return fit_video_long_edge(img, ref_max_size)


def _build_fl2v_endpoint_source(
    start_img: torch.Tensor,
    end_img: torch.Tensor | None,
    frame_count: int,
) -> torch.Tensor:
    """Build source video with temporal endpoints for Bernini context encoding.

    Prompt/refs alone are soft. Encoding a full-length source whose first frame is
    image0 and last frame is image1 (middle held at image0) gives the model a much
    stronger first/last-frame signal — especially on long later shots.
    """
    from .plan import wan_align_frame_count

    if start_img.ndim == 3:
        start_img = start_img.unsqueeze(0)
    n = wan_align_frame_count(max(MIN_FL2V_FRAMES, int(frame_count)))
    # Hold start through the body (i2v-like), snap last frame to end when present.
    clip = start_img[:1].expand(n, -1, -1, -1).contiguous().clone()
    if end_img is not None:
        if end_img.ndim == 3:
            end_img = end_img.unsqueeze(0)
        clip[-1] = end_img[0]
    return clip


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _normalize_keyframes(raw: list[dict] | None) -> list[dict]:
    out: list[dict] = []
    for idx, item in enumerate(raw or []):
        if not isinstance(item, dict):
            continue
        image_file = (
            (item.get("imageFile") or item.get("image_file") or "").strip()
            or ((item.get("genImage") or {}).get("imageFile") or "").strip()
        )
        if not image_file and not (item.get("imageB64") or item.get("image_b64")):
            continue
        fc = int(
            item.get("frameCount")
            or item.get("frame_count")
            or item.get("length")
            or DEFAULT_FL2V_FRAMES
        )
        fc = max(MIN_FL2V_FRAMES, fc)
        start = max(0, int(item.get("start") or 0))

        if "isEndFrame" in item or "is_end_frame" in item:
            is_end = _as_bool(item.get("isEndFrame", item.get("is_end_frame")))
        elif "breakBefore" in item or "break_before" in item:
            is_end = idx > 0 and not _as_bool(item.get("breakBefore") or item.get("break_before"))
        else:
            is_end = False

        has_start_key = "isStartFrame" in item or "is_start_frame" in item
        is_start = (
            _as_bool(item.get("isStartFrame", item.get("is_start_frame")), True)
            if has_start_key
            else None
        )

        out.append(
            {
                "id": item.get("id") or "",
                "imageFile": image_file,
                "imageB64": item.get("imageB64") or item.get("image_b64") or "",
                "width": int(item.get("width") or (item.get("genImage") or {}).get("width") or 0),
                "height": int(item.get("height") or (item.get("genImage") or {}).get("height") or 0),
                "start": start,
                "length": fc,
                "frameCount": fc,
                "prompt": (item.get("prompt") or "").strip(),
                "negativePrompt": (
                    item.get("negativePrompt")
                    or item.get("negative_prompt")
                    or DEFAULT_FL2V_NEGATIVE
                ).strip()
                or DEFAULT_FL2V_NEGATIVE,
                "isStartFrame": is_start,
                "isEndFrame": is_end,
                "_breakBefore": _as_bool(
                    item.get("breakBefore") or item.get("break_before") or item.get("disconnect")
                ),
            }
        )
    out.sort(key=lambda k: (int(k.get("start") or 0), k.get("id") or ""))
    n = len(out)
    for i, kf in enumerate(out):
        if kf["isStartFrame"] is None:
            # Legacy: only the last continuous end-frame was non-start.
            end_only_last = (
                i > 0
                and i == n - 1
                and not kf.get("_breakBefore")
                and kf.get("isEndFrame")
            )
            kf["isStartFrame"] = not end_only_last
        kf.pop("_breakBefore", None)
    return out


def _shot_sample_frame_count(start_kf: dict, end_kf: dict | None) -> int:
    """Generation length for a start shot.

    Base = start clip length. If the paired end clip is end-only (not a start),
    extend through the end of that end clip's placeholder on the timeline.
    """
    base = max(MIN_FL2V_FRAMES, int(start_kf.get("frameCount") or start_kf.get("length") or DEFAULT_FL2V_FRAMES))
    if end_kf is None:
        return base
    # End clip is also a start → its span belongs to its own shot; only use as image1.
    if end_kf.get("isStartFrame"):
        return base
    start_t = int(start_kf.get("start") or 0)
    end_t = int(end_kf.get("start") or 0) + max(
        MIN_FL2V_FRAMES,
        int(end_kf.get("frameCount") or end_kf.get("length") or DEFAULT_FL2V_FRAMES),
    )
    return max(base, end_t - start_t)


def _expand_shots(keyframes: list[dict]) -> list[dict[str, Any]]:
    """Only isStartFrame clips produce sampling shots; next isEndFrame is image1."""
    if not keyframes:
        return []
    shots: list[dict[str, Any]] = []
    n = len(keyframes)
    for i, kf in enumerate(keyframes):
        if not kf.get("isStartFrame"):
            continue
        end = None
        for j in range(i + 1, n):
            if keyframes[j].get("isEndFrame"):
                end = keyframes[j]
                break
        fc = _shot_sample_frame_count(kf, end)
        shots.append(
            {
                "source_index": i,
                "start": kf,
                "end": end,
                "frameCount": fc,
                "timeline_start": int(kf.get("start") or 0),
                "prompt": kf.get("prompt") or "",
                "negativePrompt": kf.get("negativePrompt") or DEFAULT_FL2V_NEGATIVE,
            }
        )
    return shots


def count_fl2v_runnable_shots(timeline: dict) -> int:
    keys = _normalize_keyframes(timeline.get("keyframes") or timeline.get("segments") or [])
    return max(1, sum(1 for k in keys if k.get("isStartFrame")) or 1)


def build_fl2v_director_plan(
    timeline: dict,
    *,
    global_task_type: str,
    global_prompt: str,
    total_frames: int,
    frame_rate: float,
    width: int,
    height: int,
    ref_max_size: int,
):
    from .plan import (
        DirectorPlan,
        SegmentPlan,
        SegmentRef,
        _parse_run_selection,
        _resolve_export_mode,
    )

    global_block = timeline.get("global") or {}
    task_type = global_block.get("taskType") or global_task_type or task_type_option_label(
        TASK_PROMPT_BY_KEY["fl2v"]
    )
    task_key = resolve_task_key(task_type)
    if task_key != "fl2v":
        raise ValueError(f"fl2v plan builder received task_key={task_key}")

    keyframes = _normalize_keyframes(
        timeline.get("keyframes") or timeline.get("segments") or []
    )
    if not keyframes:
        raise ValueError(
            "fl2v: 请至少上传 1 张关键帧图片（可在首尾帧时间轴中拖入多张并设置首/尾帧）。"
        )

    shots = _expand_shots(keyframes)
    if not shots:
        raise ValueError(
            "fl2v: 没有勾选「首帧」的图片，无法采样。请至少将一张图片设为「首帧」。"
        )

    # runSelection uses keyframe / segment indices (timeline order).
    run_sel = _parse_run_selection(timeline, len(keyframes))
    if run_sel is not None:
        shots = [s for s in shots if int(s["source_index"]) in run_sel]
        if not shots:
            raise ValueError(
                "Bernini Director: 「选择运行」已开启，但勾选的片段中没有「首帧」。"
                "请勾选至少一张首帧图片再执行。"
            )

    output_block = timeline.get("output") or {}
    out_mode = str(output_block.get("mode") or "long_edge").lower()
    if out_mode not in ("fixed", "long_edge"):
        out_mode = "long_edge"
    src_w = int(keyframes[0].get("width") or 0)
    src_h = int(keyframes[0].get("height") or 0)
    out_w, out_h, ref_max, out_mode = resolve_output_dimensions(
        src_w or int(width or 832),
        src_h or int(height or 480),
        mode=out_mode,
        long_edge=int(
            output_block.get("longEdge")
            or output_block.get("long_edge")
            or ref_max_size
            or 848
        ),
        fixed_width=int(output_block.get("width") or timeline.get("width") or width),
        fixed_height=int(output_block.get("height") or timeline.get("height") or height),
    )

    export_mode = _resolve_export_mode(output_block)
    fallback_prompt = (global_block.get("prompt") or global_prompt or "").strip()
    fallback_negative = (
        global_block.get("negativePrompt")
        or global_block.get("negative_prompt")
        or ""
    ).strip()
    timeline_total = max(
        MIN_FL2V_FRAMES,
        int(timeline.get("totalFrames") or total_frames or DEFAULT_TOTAL_FRAMES),
    )

    segments: list[SegmentPlan] = []
    source_clips: list[torch.Tensor] = []
    for idx, shot in enumerate(shots):
        start_kf = shot["start"]
        end_kf = shot["end"]
        fc = max(MIN_FL2V_FRAMES, int(shot["frameCount"]))
        user_prompt = (shot.get("prompt") or "").strip() or fallback_prompt
        full_prompt = reinforce_fl2v_prompt(user_prompt, has_end_frame=end_kf is not None)
        shot_negative = (
            (shot.get("negativePrompt") or "").strip()
            or fallback_negative
            or DEFAULT_FL2V_NEGATIVE
        )

        start_ref = {
            "imageFile": start_kf.get("imageFile") or "",
            "imageB64": start_kf.get("imageB64") or "",
        }
        start_img = _fit_image(
            _load_image_ref(start_ref),
            width=out_w,
            height=out_h,
            output_mode=out_mode,
            ref_max_size=ref_max,
        )

        refs: list[SegmentRef] = [
            SegmentRef(index=0, tensor=start_img[:1].clone()),
        ]
        end_img = None
        if end_kf is not None:
            end_ref = {
                "imageFile": end_kf.get("imageFile") or "",
                "imageB64": end_kf.get("imageB64") or "",
            }
            end_img = _fit_image(
                _load_image_ref(end_ref),
                width=out_w,
                height=out_h,
                output_mode=out_mode,
                ref_max_size=ref_max,
            )
            refs.append(SegmentRef(index=1, tensor=end_img[:1].clone()))

        # Endpoint source (first=image0, last=image1) — stronger than prompt/refs alone.
        source_clip = _build_fl2v_endpoint_source(start_img, end_img, fc)
        source_clips.append(source_clip[:1].clone())

        start_f = int(shot.get("timeline_start") or 0)
        end_f = start_f + fc
        segments.append(
            SegmentPlan(
                index=idx,
                start_frame=start_f,
                end_frame=end_f,
                prompt=full_prompt,
                task_type=task_type,
                task_key="fl2v",
                use_global=False,
                refs=refs,
                negative_prompt=shot_negative,
                source_clip=source_clip,
            )
        )

    source_video = torch.full((len(segments), 16, 16, 3), 0.5, dtype=torch.float32)
    raw = dict(timeline)
    raw["timelineMode"] = "fl2v"
    raw["keyframes"] = keyframes
    raw["totalFrames"] = timeline_total

    return DirectorPlan(
        frame_rate=float(timeline.get("frameRate") or frame_rate or 24),
        total_frames=timeline_total,
        width=out_w,
        height=out_h,
        ref_max_size=ref_max,
        output_mode=out_mode,
        source_width=int(src_w or out_w),
        source_height=int(src_h or out_h),
        global_task_type=task_type,
        global_task_key="fl2v",
        global_prompt=fallback_prompt,
        global_refs=[],
        source_video=source_video,
        segments=segments,
        edit_mode="segment",
        raw=raw,
        export_mode=export_mode,
        # Already filtered by keyframe indices above.
        run_indices=None,
    )
