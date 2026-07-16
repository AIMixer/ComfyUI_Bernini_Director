"""Bernini Director — generation timeline (t2i / t2v / i2i / i2v) plan building."""

from __future__ import annotations

import logging

import torch

from ..lib.image_prep import fit_canvas, fit_video_long_edge, cat_frames_variable_size, resolve_output_dimensions
from ..lib.task_prompts import resolve_task_key

log = logging.getLogger("ComfyUI-Bernini-Director.director.gen")

GEN_BLANK_KEYS = frozenset({"t2i", "t2v", "r2i", "r2v"})
GEN_IMAGE_KEYS = frozenset({"i2i", "i2v"})
GEN_TASK_KEYS = GEN_BLANK_KEYS | GEN_IMAGE_KEYS
PROMPT_BATCH_KEYS = frozenset({"t2i", "i2i", "r2i", "t2v", "i2v", "r2v"})
VIDEO_BATCH_KEYS = frozenset({"t2v", "i2v", "r2v"})
IMAGE_BATCH_KEYS = frozenset({"t2i", "i2i", "r2i"})  # legacy alias

MIN_GEN_FRAMES = 1
MIN_GEN_VIDEO_FRAMES = 4


def is_gen_task_key(task_key: str) -> bool:
    return task_key in GEN_TASK_KEYS


def is_gen_timeline(timeline: dict, task_key: str) -> bool:
    mode = str(timeline.get("timelineMode") or "").lower()
    if mode in ("gen_blank", "gen_image", "image_batch", "prompt_batch"):
        return True
    if mode == "video":
        return False
    return is_gen_task_key(task_key)


def is_prompt_batch_timeline(timeline: dict, task_key: str) -> bool:
    mode = str(timeline.get("timelineMode") or "").lower()
    if mode in ("image_batch", "prompt_batch"):
        return True
    return task_key in PROMPT_BATCH_KEYS


def is_image_batch_timeline(timeline: dict, task_key: str) -> bool:
    return is_prompt_batch_timeline(timeline, task_key)


def is_video_batch_task_key(task_key: str) -> bool:
    return task_key in VIDEO_BATCH_KEYS


def gen_submode(timeline: dict, task_key: str) -> str:
    mode = str(timeline.get("timelineMode") or "").lower()
    if mode == "gen_image" or task_key in GEN_IMAGE_KEYS:
        return "gen_image"
    if mode == "gen_blank" or task_key in GEN_BLANK_KEYS:
        return "gen_blank"
    return "gen_blank"


def _min_frames_for_task(task_key: str) -> int:
    if task_key in IMAGE_BATCH_KEYS or task_key in ("t2i", "i2i"):
        return MIN_GEN_FRAMES
    if task_key in ("t2v", "i2v", "r2v"):
        return MIN_GEN_VIDEO_FRAMES
    return MIN_GEN_VIDEO_FRAMES


def _segment_frame_count(raw: dict, *, default: int, task_key: str) -> int:
    fc = int(raw.get("frameCount") or raw.get("frame_count") or raw.get("length") or default)
    return max(_min_frames_for_task(task_key), fc)


def _gen_segment_ranges(
    segments: list[dict],
    *,
    default_frame_count: int,
    task_key: str,
) -> list[tuple[int, int, dict]]:
    ranges: list[tuple[int, int, dict]] = []
    start = 0
    for raw in segments:
        fc = _segment_frame_count(raw, default=default_frame_count, task_key=task_key)
        ranges.append((start, start + fc, raw))
        start += fc
    if not ranges:
        fc = max(_min_frames_for_task(task_key), default_frame_count)
        ranges.append((0, fc, {}))
    return ranges


def _resolve_gen_image_ref(
    seg_data: dict,
    *,
    edit_mode: str,
    global_block: dict,
) -> dict | None:
    if edit_mode == "segment":
        img = seg_data.get("genImage") or {}
        if img.get("imageFile") or img.get("imageB64"):
            return img
        if seg_data.get("imageFile"):
            return {"imageFile": seg_data["imageFile"]}
        return None
    img = global_block.get("genImage") or {}
    if img.get("imageFile") or img.get("imageB64"):
        return img
    if global_block.get("imageFile"):
        return {"imageFile": global_block["imageFile"]}
    return None


def _load_gen_image_tensor(ref: dict) -> torch.Tensor:
    from .plan import load_reference_tensor

    tensor = load_reference_tensor(ref)
    if tensor is None:
        raise ValueError("Generation segment image could not be loaded.")
    return tensor


def _build_i2v_source_clip(
    img: torch.Tensor,
    _frame_count: int,
    *,
    width: int,
    height: int,
    output_mode: str,
    ref_max_size: int,
) -> torch.Tensor:
    """Use the source image as a one-frame source-video context."""
    if img.ndim == 3:
        img = img.unsqueeze(0)
    if output_mode == "fixed":
        return fit_canvas(img, width, height)
    return fit_video_long_edge(img, ref_max_size)


def _resolve_gen_image_source_dims(
    segment_ranges: list[tuple[int, int, dict]],
    global_block: dict,
    output_block: dict,
) -> tuple[int, int]:
    sw = int(global_block.get("sourceWidth") or output_block.get("sourceWidth") or 0)
    sh = int(global_block.get("sourceHeight") or output_block.get("sourceHeight") or 0)
    if sw > 0 and sh > 0:
        return sw, sh
    for _start, _end, seg_data in segment_ranges:
        gi = seg_data.get("genImage") or {}
        sw = int(gi.get("width") or 0)
        sh = int(gi.get("height") or 0)
        if sw > 0 and sh > 0:
            return sw, sh
    return 0, 0


def _build_gen_source_clips(
    ranges: list[tuple[int, int, dict]],
    *,
    task_key: str,
    submode: str,
    edit_mode: str,
    global_block: dict,
    height: int,
    width: int,
    output_mode: str,
    ref_max_size: int,
) -> list[torch.Tensor]:
    chunks: list[torch.Tensor] = []
    for _start, end, seg_data in ranges:
        frame_count = end - _start
        if frame_count <= 0:
            continue
        if submode == "gen_blank":
            clip = torch.full((frame_count, height, width, 3), 0.5, dtype=torch.float32)
        else:
            ref = _resolve_gen_image_ref(seg_data, edit_mode=edit_mode, global_block=global_block)
            if ref is None:
                seg_idx = len(chunks) + 1
                raise ValueError(
                    f"Segment #{seg_idx} has no source image. "
                    "Upload an image in the generation timeline (global or per-segment)."
                )
            img = _load_gen_image_tensor(ref)
            if task_key == "i2v":
                clip = _build_i2v_source_clip(
                    img,
                    frame_count,
                    width=width,
                    height=height,
                    output_mode=output_mode,
                    ref_max_size=ref_max_size,
                )
            else:
                clip = img.repeat(frame_count, 1, 1, 1)
                if output_mode == "fixed":
                    clip = fit_canvas(clip, width, height)
                else:
                    clip = fit_video_long_edge(clip, ref_max_size)
        chunks.append(clip)
    if not chunks:
        raise ValueError("Generation timeline has no frames.")
    return chunks


def _build_gen_source_video(
    ranges: list[tuple[int, int, dict]],
    *,
    task_key: str,
    submode: str,
    edit_mode: str,
    global_block: dict,
    height: int,
    width: int,
    output_mode: str,
    ref_max_size: int,
) -> torch.Tensor:
    return cat_frames_variable_size(
        _build_gen_source_clips(
            ranges,
            task_key=task_key,
            submode=submode,
            edit_mode=edit_mode,
            global_block=global_block,
            height=height,
            width=width,
            output_mode=output_mode,
            ref_max_size=ref_max_size,
        )
    )


def build_gen_director_plan(
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
    """Build DirectorPlan for generation timeline modes (lazy import avoids cycles)."""
    from .plan import (
        DirectorPlan,
        SegmentPlan,
        _load_refs,
        _parse_run_selection,
        _resolve_export_mode,
        segment_refs_for_context,
    )

    global_block = timeline.get("global") or {}
    edit_mode = timeline.get("editMode") or timeline.get("edit_mode") or "global"
    if is_prompt_batch_timeline(timeline, resolve_task_key(global_block.get("taskType") or global_task_type or "")):
        edit_mode = "segment"
    elif edit_mode not in ("global", "segment"):
        edit_mode = "global"

    task_type = global_block.get("taskType") or global_task_type or "t2v — 文生视频(Text to Video)"
    task_key = resolve_task_key(task_type)
    if not is_gen_task_key(task_key):
        raise ValueError(f"Task {task_key} is not supported on the generation timeline.")

    if task_key == "i2v":
        log.warning(
            "i2v is experimental: Bernini official has no dedicated i2v testcase or demo."
        )

    submode = gen_submode(timeline, task_key)
    prompt = global_block.get("prompt") or global_prompt or ""
    global_refs = _load_refs(global_block.get("refs") or [])

    output_block = timeline.get("output") or {}
    gen_block = timeline.get("gen") or {}
    default_fc = int(gen_block.get("defaultFrameCount") or total_frames or 81)

    segment_ranges = _gen_segment_ranges(
        timeline.get("segments") or [],
        default_frame_count=default_fc,
        task_key=task_key,
    )

    if submode == "gen_blank":
        out_mode = "fixed"
        fw = int(output_block.get("width") or timeline.get("width") or width or 0)
        fh = int(output_block.get("height") or timeline.get("height") or height or 0)
        if fw < 16 or fh < 16:
            raise ValueError(
                "t2i / t2v / r2i / r2v require fixed output width and height (≥16, multiples of 16). "
                "Set width and height in the generation timeline output panel."
            )
        out_w, out_h, ref_max, _ = resolve_output_dimensions(
            fw,
            fh,
            mode="fixed",
            long_edge=ref_max_size,
            fixed_width=fw,
            fixed_height=fh,
        )
    else:
        out_mode = str(output_block.get("mode") or "long_edge").lower()
        if out_mode not in ("fixed", "long_edge"):
            out_mode = "long_edge"
        src_w, src_h = _resolve_gen_image_source_dims(segment_ranges, global_block, output_block)
        out_w, out_h, ref_max, out_mode = resolve_output_dimensions(
            src_w or int(width or 832),
            src_h or int(height or 480),
            mode=out_mode,
            long_edge=int(output_block.get("longEdge") or output_block.get("long_edge") or ref_max_size or 848),
            fixed_width=int(output_block.get("width") or timeline.get("width") or width),
            fixed_height=int(output_block.get("height") or timeline.get("height") or height),
        )

    export_mode = _resolve_export_mode(output_block)
    # Image prompt-batch (t2i/i2i/r2i) always merges to images list; video batch (t2v/i2v/r2v) respects export mode.
    if is_prompt_batch_timeline(timeline, task_key) and not is_video_batch_task_key(task_key):
        export_mode = "all"

    source_clips = _build_gen_source_clips(
        segment_ranges,
        task_key=task_key,
        submode=submode,
        edit_mode=edit_mode,
        global_block=global_block,
        height=out_h,
        width=out_w,
        output_mode=out_mode,
        ref_max_size=ref_max,
    )
    attach_source_clips = is_prompt_batch_timeline(timeline, task_key) and task_key in ("i2i", "i2v")
    if attach_source_clips:
        # Placeholder timeline index only — spatial data comes from each segment's source_clip.
        source_video = torch.full((len(source_clips), 16, 16, 3), 0.5, dtype=torch.float32)
    else:
        source_video = cat_frames_variable_size(source_clips)

    segments: list[SegmentPlan] = []
    for idx, (start, end, seg_data) in enumerate(segment_ranges):
        if edit_mode == "global":
            seg_prompt = prompt
            seg_task = task_type
            seg_refs = list(global_refs)
            use_global = True
            seg_negative = ""
        else:
            use_global = False
            seg_prompt = (seg_data.get("prompt") or "").strip() or prompt
            seg_task = seg_data.get("taskType") or seg_data.get("task_type") or task_type
            seg_refs = _load_refs(seg_data.get("refs") or [])
            seg_negative = (
                (seg_data.get("negativePrompt") or seg_data.get("negative_prompt") or "").strip()
            )

        seg_task_key = resolve_task_key(seg_task)
        if seg_task_key == "i2v" and seg_refs:
            log.info(
                "i2v segment #%d: ignoring %d reference image(s); using source video context only",
                idx + 1,
                len(seg_refs),
            )
        seg_refs = segment_refs_for_context(seg_task_key, seg_refs)
        seg_source = source_clips[idx].clone() if idx < len(source_clips) else None

        segments.append(
            SegmentPlan(
                index=idx,
                start_frame=start,
                end_frame=end,
                prompt=seg_prompt,
                task_type=seg_task,
                task_key=seg_task_key,
                use_global=use_global,
                refs=seg_refs,
                negative_prompt=seg_negative,
                source_clip=seg_source,
            )
        )

    total = int(segment_ranges[-1][1]) if segment_ranges else int(source_video.shape[0])
    if is_prompt_batch_timeline(timeline, task_key):
        timeline_mode = "prompt_batch"
    else:
        timeline_mode = "gen_image" if submode == "gen_image" else "gen_blank"

    raw = dict(timeline)
    raw["timelineMode"] = timeline_mode
    src_w, src_h = _resolve_gen_image_source_dims(segment_ranges, global_block, output_block)

    return DirectorPlan(
        frame_rate=float(timeline.get("frameRate") or frame_rate or 24),
        total_frames=total,
        width=out_w,
        height=out_h,
        ref_max_size=ref_max,
        output_mode=out_mode,
        source_width=int(src_w or out_w),
        source_height=int(src_h or out_h),
        global_task_type=task_type,
        global_task_key=task_key,
        global_prompt=prompt,
        global_refs=global_refs,
        source_video=source_video,
        segments=segments,
        edit_mode=edit_mode,
        raw=raw,
        export_mode=export_mode,
        run_indices=_parse_run_selection(timeline, len(segments)),
    )
