"""Cross-segment continuity — **opt-in branch only**.

When continuity is **off**, the director must follow the official Studio /
per-segment path: each segment's own source + user refs → BerniniConditioning
→ dual-stage KSampler → VAEDecode. No prev-tail injection, no post luma.

When continuity is **on**, this module's branch runs instead and may:
1. Prefix ``source_video`` with luma-normalized prev-tail
2. Optionally inject at most one prev-tail frame as ``reference_image_*``
   (skipped when the segment already has user reference images)
3. After decode, trim overlap prefix and re-anchor mean luminance to source
"""

from __future__ import annotations

import logging

import torch

from ..lib.image_prep import cat_frames_variable_size, fit_canvas
from ..lib.ref_images import MAX_REFERENCE_IMAGES, REF_IMAGE_KEY_PREFIX
from .plan import DirectorPlan, SegmentPlan, wan_align_frame_count
from .segment_cache import load_segment_cache

log = logging.getLogger("ComfyUI-Bernini-Director.director.continuity")

CONTINUITY_TASK_KEYS = frozenset({"v2v", "rv2v", "vi2v", "vrc2v", "mv2v", "ads2v", "i2v"})

DEFAULT_CONTINUITY_OVERLAP = 9
MIN_CONTINUITY_OVERLAP = 1
MAX_CONTINUITY_OVERLAP = 81
# At most one generated continuity ref — multi-ref from prev output drifts face exposure.
MAX_CONTINUITY_REF_FRAMES = 1
# Source-stream anchor: replace opening frames of source_video with prev output.
MAX_CONTINUITY_SOURCE_ANCHOR = 17
# Post-decode: pull whole segment mean luma toward source (stops cumulative face brighten).
SEGMENT_SOURCE_LUMA_STRENGTH = 0.55
SEGMENT_SOURCE_LUMA_MAX_DELTA = 0.18


def resolve_continuity_settings(timeline: dict, *, segment_count: int) -> tuple[bool, int]:
    """Read segment continuity flags from timeline JSON (output only; default off)."""
    if segment_count < 2:
        return False, 0
    output = timeline.get("output") or {}
    enabled = bool(
        output.get("continuityEnabled") is True
        or output.get("continuity_enabled") is True
    )
    if not enabled:
        return False, 0
    raw = (
        output.get("continuityOverlapFrames")
        or output.get("continuity_overlap_frames")
        or DEFAULT_CONTINUITY_OVERLAP
    )
    overlap = max(MIN_CONTINUITY_OVERLAP, min(MAX_CONTINUITY_OVERLAP, int(raw)))
    return True, overlap


def resolve_continuity_lock_pixels(overlap_frames: int) -> int:
    """Wan-aligned frames taken from prev tail (shared budget for refs + source anchor)."""
    ov = max(MIN_CONTINUITY_OVERLAP, min(MAX_CONTINUITY_OVERLAP, int(overlap_frames)))
    return wan_align_frame_count(ov)


def resolve_continuity_ref_frames(overlap_frames: int) -> int:
    """How many prev-tail frames to inject as reference_image_*."""
    return min(resolve_continuity_lock_pixels(overlap_frames), MAX_CONTINUITY_REF_FRAMES)


def resolve_continuity_source_anchor_frames(overlap_frames: int) -> int:
    """How many opening source frames to hard-replace with prev output."""
    return min(resolve_continuity_lock_pixels(overlap_frames), MAX_CONTINUITY_SOURCE_ANCHOR)


def resolve_continuity_guide_frames(overlap_frames: int) -> tuple[int, int, int, int, int]:
    """Map UI overlap → (context_px, tail_refs, seam_blend, opening_blend, color_match)."""
    lock = resolve_continuity_lock_pixels(overlap_frames)
    refs = resolve_continuity_ref_frames(overlap_frames)
    # Post seam soft-blend / opening color-match disabled (drift). Whole-segment luma used instead.
    return lock, refs, 0, 0, 0


def resolve_segment_generation_frames(
    *,
    segment_frame_count: int,
    segment_index: int,
    continuity_enabled: bool,
    continuity_overlap: int,
) -> tuple[int, int]:
    """Return (gen_frames, prefix_trim_after_decode).

    Continuity segments generate overlap extra frames (prev-tail conditioned),
    then drop that prefix so the export does not repeat the previous ending.
    """
    base = wan_align_frame_count(max(1, int(segment_frame_count)))
    if not continuity_enabled or segment_index <= 0:
        return base, 0
    lock_px = resolve_continuity_source_anchor_frames(continuity_overlap)
    if lock_px <= 0:
        return base, 0
    return wan_align_frame_count(base + lock_px), lock_px


def is_continuity_active(plan: DirectorPlan, seg: SegmentPlan) -> bool:
    """True only on the opt-in continuity branch for segment index > 0."""
    return (
        plan.continuity_enabled
        and plan.segment_count >= 2
        and seg.index > 0
        and seg.task_key in CONTINUITY_TASK_KEYS
    )


def apply_continuity_conditioning_branch(
    *,
    plan: DirectorPlan,
    seg: SegmentPlan,
    all_segments: list[SegmentPlan],
    completed_outputs: dict[int, torch.Tensor],
    node_id: str | None,
    source_arg: torch.Tensor | None,
    ref_kwargs: dict[str, torch.Tensor],
    num_frames: int,
    target_len: int,
    ctx_w: int,
    ctx_h: int,
) -> tuple[torch.Tensor | None, dict[str, torch.Tensor], int, int, str | None]:
    """Opt-in branch: inject prev-tail into conditioning.

    Returns ``(source_for_cond, cond_refs, num_frames, prefix_trim, note)``.
    Callers must only invoke this when ``plan.continuity_enabled`` is True.
    If the segment is not eligible (first segment / wrong task), returns the
    official inputs unchanged (source_arg, ref_kwargs, num_frames, 0, None).
    """
    if not plan.continuity_enabled:
        return source_arg, dict(ref_kwargs), int(num_frames), 0, None

    prev_tail = resolve_prev_segment_output(
        plan, all_segments, seg.index, completed_outputs, node_id
    )
    if not is_continuity_active(plan, seg) or prev_tail is None:
        return source_arg, dict(ref_kwargs), int(num_frames), 0, None

    from ..lib.image_prep import cat_frames_variable_size

    anchor_n = resolve_continuity_source_anchor_frames(plan.continuity_overlap_frames)
    anchor_n = min(anchor_n, int(prev_tail.shape[0]))
    anchored = 0
    prefix_trim = 0
    out_frames = int(num_frames)
    source_for_cond = source_arg
    cond_refs = dict(ref_kwargs)

    guide_tail = prev_tail
    if source_arg is not None and int(source_arg.shape[0]) > 0:
        guide_tail = normalize_guide_luma_to_source(
            prev_tail,
            source_arg[0],
            width=ctx_w,
            height=ctx_h,
        )

    if source_arg is not None and anchor_n > 0:
        tail = fit_canvas(guide_tail[-anchor_n:], ctx_w, ctx_h).to(
            device=source_arg.device, dtype=source_arg.dtype
        )
        source_for_cond = cat_frames_variable_size([tail, source_arg])
        if int(source_for_cond.shape[0]) > out_frames:
            source_for_cond = source_for_cond[:out_frames]
        elif int(source_for_cond.shape[0]) < out_frames:
            pad = source_for_cond[-1:].repeat(
                out_frames - int(source_for_cond.shape[0]), 1, 1, 1
            )
            source_for_cond = torch.cat([source_for_cond, pad], dim=0)
        anchored = anchor_n
        prefix_trim = anchor_n
    elif anchor_n > 0:
        prefix_trim = 0
        out_frames = wan_align_frame_count(max(1, int(target_len)))

    ref_n = min(
        resolve_continuity_ref_frames(plan.continuity_overlap_frames),
        int(guide_tail.shape[0]),
    )
    cond_refs, added = continuity_ref_kwargs_from_prev(
        ref_kwargs,
        guide_tail,
        width=ctx_w,
        height=ctx_h,
        n_frames=ref_n,
        window_frames=anchor_n if anchor_n > 0 else ref_n,
    )

    bits = []
    if anchored > 0:
        bits.append(f"source-prefix {anchored}f")
    if added > 0:
        bits.append(f"+{added} ref")
    note = None
    if bits:
        note = (
            f"  continuity branch seg #{seg.index + 1}: "
            + ", ".join(bits)
            + (f", trim {prefix_trim}f" if prefix_trim > 0 else "")
            + f" (overlap {plan.continuity_overlap_frames})"
        )
    return source_for_cond, cond_refs, out_frames, prefix_trim, note


def apply_continuity_post_decode_branch(
    decoded: torch.Tensor,
    *,
    plan: DirectorPlan,
    seg: SegmentPlan,
    source_arg: torch.Tensor | None,
    ref_kwargs: dict[str, torch.Tensor],
    prefix_trim: int,
    target_len: int,
    ctx_w: int,
    ctx_h: int,
) -> torch.Tensor:
    """Opt-in branch: trim overlap prefix + optional source luma re-anchor.

    When continuity is off, callers must not use this — only pad/trim to
    ``target_len`` like the official Studio path.
    """
    out = decoded
    if prefix_trim > 0 and out.shape[0] > prefix_trim:
        out = out[prefix_trim:]
    if out.shape[0] > target_len:
        out = out[:target_len]
    elif out.shape[0] < target_len and out.shape[0] > 0:
        pad = out[-1:].repeat(target_len - out.shape[0], 1, 1, 1)
        out = torch.cat([out, pad], dim=0)

    if not plan.continuity_enabled or seg.index <= 0 or source_arg is None:
        return out

    user_ref_slots = sum(1 for k in ref_kwargs if k.startswith(REF_IMAGE_KEY_PREFIX))
    luma_strength = 0.28 if user_ref_slots > 0 else 0.55
    luma_delta = 0.10 if user_ref_slots > 0 else 0.18
    return match_segment_luminance_to_source(
        out,
        source_arg,
        width=ctx_w,
        height=ctx_h,
        strength=luma_strength,
        max_ratio_delta=luma_delta,
    )


def resolve_prev_segment_output(
    plan: DirectorPlan,
    all_segments: list[SegmentPlan],
    seg_index: int,
    completed: dict[int, torch.Tensor],
    node_id: str | None,
) -> torch.Tensor | None:
    prev_idx = seg_index - 1
    if prev_idx < 0:
        return None
    if prev_idx in completed:
        return completed[prev_idx]
    prev_seg = all_segments[prev_idx]
    cached = load_segment_cache(node_id, prev_seg, plan)
    if cached is not None:
        return cached
    if not plan.continuity_enabled:
        return None
    raise ValueError(
        f"段间连贯：片段 #{seg_index + 1} 需要上一段 #{prev_idx + 1} 的生成结果。"
        "请先运行上一段，或开启「全部运行」以生成完整序列；"
        "若使用「选择运行」，请确保上一段已有有效缓存。"
    )


def _occupied_ref_slots(ref_kwargs: dict[str, torch.Tensor]) -> set[int]:
    occupied: set[int] = set()
    for key in ref_kwargs:
        if key.startswith(REF_IMAGE_KEY_PREFIX):
            occupied.add(int(key.removeprefix(REF_IMAGE_KEY_PREFIX)))
    return occupied


def apply_source_continuity_anchor(
    source_video: torch.Tensor | None,
    prev_output: torch.Tensor,
    *,
    width: int,
    height: int,
    n_frames: int,
) -> tuple[torch.Tensor | None, int]:
    """Hard-replace opening of source_video with prev-tail (Bernini source stream)."""
    if source_video is None or int(n_frames) <= 0:
        return source_video, 0
    n = min(int(n_frames), int(source_video.shape[0]), int(prev_output.shape[0]))
    if n <= 0:
        return source_video, 0
    out = source_video.clone()
    tail = fit_canvas(prev_output[-n:], width, height).to(device=out.device, dtype=out.dtype)
    out[:n] = tail
    log.info("Segment continuity: anchored %d source opening frame(s) from prev tail", n)
    return out, n


def continuity_ref_kwargs_from_prev(
    ref_kwargs: dict[str, torch.Tensor],
    prev_output: torch.Tensor,
    *,
    width: int,
    height: int,
    n_frames: int,
    window_frames: int | None = None,
) -> tuple[dict[str, torch.Tensor], int]:
    """Append at most one prev-tail frame into a free reference_image_* slot.

    Skipped entirely when the segment already has user reference images — those
    own appearance; injecting generated frames there is the main face-brighten path.
    """
    occupied = _occupied_ref_slots(ref_kwargs)
    if occupied:
        log.info(
            "Segment continuity: skip continuity refs (%d user ref slot(s) present)",
            len(occupied),
        )
        return dict(ref_kwargs), 0

    n = min(int(n_frames), int(prev_output.shape[0]), MAX_CONTINUITY_REF_FRAMES)
    if n <= 0:
        return dict(ref_kwargs), 0

    free_slots = [i for i in range(MAX_REFERENCE_IMAGES) if i not in occupied]
    if not free_slots:
        return dict(ref_kwargs), 0

    n = min(n, len(free_slots))
    window = int(window_frames) if window_frames is not None else n
    window = min(max(n, window), int(prev_output.shape[0]))
    tail_window = fit_canvas(prev_output[-window:], width, height)

    # Prefer the last frame (seam).
    picks = [int(tail_window.shape[0]) - 1]

    merged = dict(ref_kwargs)
    for i, frame_i in enumerate(picks):
        slot = free_slots[i]
        merged[f"{REF_IMAGE_KEY_PREFIX}{slot}"] = tail_window[frame_i : frame_i + 1].contiguous()
    log.info("Segment continuity: injected %d Bernini reference_image stream(s)", len(picks))
    return merged, len(picks)


def _frame_mean_luminance(frame: torch.Tensor) -> float:
    """Mean Rec.601 luma for an HWC tensor in [0, 1]."""
    f = frame.float()
    luma = 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]
    return float(luma.mean().item())


def _tensor_mean_luminance(frames: torch.Tensor) -> float:
    """Mean Rec.601 luma over a frame batch (N,H,W,C) or single frame."""
    f = frames.float()
    if f.dim() == 3:
        f = f.unsqueeze(0)
    luma = 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]
    return float(luma.mean().item())


def normalize_guide_luma_to_source(
    guide: torch.Tensor,
    source_frame: torch.Tensor,
    *,
    width: int,
    height: int,
) -> torch.Tensor:
    """Fully match guide batch mean luma to a source frame (pre-conditioning)."""
    if guide is None or int(guide.shape[0]) <= 0:
        return guide
    ref = source_frame
    if ref.dim() == 4:
        ref = ref[0]
    ref = fit_canvas(ref.unsqueeze(0), width, height)[0]
    out = fit_canvas(guide, width, height).float()
    ref_mean = max(_frame_mean_luminance(ref), 1e-6)
    guide_mean = max(_tensor_mean_luminance(out), 1e-6)
    ratio = ref_mean / guide_mean
    # Cap extreme ratios (flash / near-black) but allow real exposure correction.
    ratio = max(0.55, min(1.8, ratio))
    if abs(ratio - 1.0) < 0.01:
        return out.to(dtype=guide.dtype)
    log.info(
        "Segment continuity: normalize guide luma ×%.3f (guide=%.3f → source=%.3f)",
        ratio,
        guide_mean,
        ref_mean,
    )
    return (out * ratio).clamp(0.0, 1.0).to(dtype=guide.dtype)


def match_segment_luminance_to_source(
    decoded: torch.Tensor,
    source_video: torch.Tensor | None,
    *,
    width: int,
    height: int,
    strength: float = SEGMENT_SOURCE_LUMA_STRENGTH,
    max_ratio_delta: float = SEGMENT_SOURCE_LUMA_MAX_DELTA,
) -> torch.Tensor:
    """Uniform RGB scale so decoded mean luma tracks the source clip (whole segment)."""
    if (
        source_video is None
        or int(decoded.shape[0]) <= 0
        or int(source_video.shape[0]) <= 0
        or strength <= 0.0
    ):
        return decoded
    src = fit_canvas(source_video, width, height)
    n = min(int(decoded.shape[0]), int(src.shape[0]))
    if n <= 0:
        return decoded
    ref_mean = max(_tensor_mean_luminance(src[:n]), 1e-6)
    dec_mean = max(_tensor_mean_luminance(decoded[:n]), 1e-6)
    ratio = ref_mean / dec_mean
    delta = max(0.0, float(max_ratio_delta))
    ratio = max(1.0 - delta, min(1.0 + delta, ratio))
    applied = 1.0 + (ratio - 1.0) * max(0.0, min(1.0, float(strength)))
    if abs(applied - 1.0) < 0.005:
        return decoded
    log.info(
        "Segment continuity: whole-segment luma ×%.3f toward source "
        "(decoded=%.3f source=%.3f)",
        applied,
        dec_mean,
        ref_mean,
    )
    return (decoded.float() * applied).clamp(0.0, 1.0).to(dtype=decoded.dtype)


def continuity_merged_frame_count(plan: DirectorPlan) -> int:
    return int(plan.total_frames)


def concat_continuous_chunks(
    chunks: list[torch.Tensor],
    segments: list[SegmentPlan],
    plan: DirectorPlan,
) -> torch.Tensor:
    """Concatenate full segments; plain join."""
    if not chunks:
        raise ValueError("concat_continuous_chunks: no chunks")
    if not plan.continuity_enabled or len(chunks) <= 1:
        return cat_frames_variable_size(chunks)

    merged = chunks[0]
    for seg, chunk in zip(segments[1:], chunks[1:]):
        merged = cat_frames_variable_size([merged, chunk])
        log.info(
            "Segment continuity merge: seg #%d +%d frame(s), plain concat",
            seg.index + 1,
            int(chunk.shape[0]),
        )
    return merged


def apply_cached_segment_continuity(
    chunk: torch.Tensor,
    seg: SegmentPlan,
    plan: DirectorPlan,
    completed_outputs: dict[int, torch.Tensor],
    *,
    width: int,
    height: int,
) -> torch.Tensor:
    del seg, plan, completed_outputs, width, height
    return chunk
