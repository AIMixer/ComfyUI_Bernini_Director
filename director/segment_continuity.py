"""Cross-segment continuity — **opt-in official Bernini path only**.

When continuity is **off**: Studio / per-segment path — segment source + user refs
→ BerniniConditioning → KSamplerAdvanced → VAEDecode. No cross-segment injection.

When continuity is **on** (WanSCAIL-style handoff on the official Bernini stack):
1. Prepend prev-tail pixels to ``source_video`` so canvas length matches generation
2. Generate ``prefix + segment`` frames (Wan 4n+1), trim prefix after decode
3. SCAIL latent prefix lock via ``noise_mask`` (prefix not resampled)
4. Append **one** appearance ref frame to ``context_latents``
5. Keep the timeline body in source order — do **not** MAD-skip leading body
   frames (that deletes the SCAIL handoff and looks like continuity is off)
6. Match source canvas length to ``gen_frames`` (lookahead / mirror) so Wan does not
   sample hollow end frames that freeze the segment tail
7. Carry ``noise_mask`` through **both** high and low sample stages (stripping it
   in low stage was the main “continuity looks like OFF” regression)
8. Feather lock edge; optional light nudge of body[0..] toward prev's *last*
   frame only — never sequence-lerp ``prev[-n:]`` into ``body[:n]`` (that
   replays the previous ending as duplicate frames at every cut)
9. Do **not** inject a second full motion stream (that duplicated / shifted
   source_id channels and caused seam jumps + temporal stutter)
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from ..lib.image_prep import cat_frames_variable_size, fit_canvas, fit_long_edge
from .plan import DirectorPlan, SegmentPlan, wan_align_frame_count
from .segment_cache import load_segment_cache

log = logging.getLogger("ComfyUI-Bernini-Director.director.continuity")

CONTINUITY_TASK_KEYS = frozenset({"v2v", "rv2v", "vi2v", "vrc2v", "mv2v", "ads2v", "i2v"})

DEFAULT_CONTINUITY_OVERLAP = 9
MIN_CONTINUITY_OVERLAP = 1
MAX_CONTINUITY_OVERLAP = 81
# Last N frames of prev as appearance refs (hair/outfit pop at joins needs >1).
MAX_CONTINUITY_REF_FRAMES = 2
# Leading-body MAD skip disabled: on successful rv2v/SCAIL handoff the body
# start *should* resemble prev tail; trimming it removes continuity.
CONTINUITY_SEAM_ECHO_BUDGET = 0
CONTINUITY_SEAM_ECHO_MAD = 8.0
CONTINUITY_SEAM_JOIN_MAX_SKIP = 0
CONTINUITY_SEAM_JOIN_MAD = 8.0
# Source-only lookahead past segment end for wan-align coverage (not an export skip).
CONTINUITY_SOURCE_LOOKAHEAD = 4
# Soften hard lock→body cliff (becomes the visible segment cut after prefix trim).
CONTINUITY_LOCK_FEATHER_LATENT = 1
CONTINUITY_LOCK_FEATHER_MASK = 0.35
# Do NOT sequence-lerp body[:n] with prev[-n:] — that replays the previous ending
# at every cut (visible duplicate frames). Optional: nudge first frames toward
# prev's *last* frame only (no temporal replay).
CONTINUITY_OPENING_LAST_FRAME_BLEND = 2
CONTINUITY_OPENING_LAST_FRAME_WEIGHT = 0.25


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
    """SCAIL prefix length in pixels (Wan 4n+1, for clean VAE round-trip)."""
    ov = max(MIN_CONTINUITY_OVERLAP, min(MAX_CONTINUITY_OVERLAP, int(overlap_frames)))
    return wan_align_frame_count(ov)


def resolve_continuity_guide_frames(overlap_frames: int) -> tuple[int, int, int, int, int]:
    """Map UI overlap → (context_px, tail_refs, seam_blend, opening_blend, color_match)."""
    lock = resolve_continuity_lock_pixels(overlap_frames)
    refs = min(MAX_CONTINUITY_REF_FRAMES, max(1, lock))
    return lock, refs, 0, 0, 0


def resolve_segment_generation_frames(
    *,
    segment_frame_count: int,
    segment_index: int,
    continuity_enabled: bool,
    continuity_overlap: int,
) -> tuple[int, int]:
    """Return (gen_frames, prefix_trim_after_decode).

    Continuity segments generate ``lock + body + source_lookahead`` (Wan-aligned).
    After decode we drop the lock prefix and keep the body in source order.
    """
    body = max(1, int(segment_frame_count))
    if not continuity_enabled or segment_index <= 0:
        return wan_align_frame_count(body), 0
    lock_px = resolve_continuity_lock_pixels(continuity_overlap)
    if lock_px <= 0:
        return wan_align_frame_count(body), 0
    # Include source lookahead so Wan 4n+1 padding is real timeline frames
    # (from 4ccd545), not hollow mirror tails that freeze near the cut.
    raw = lock_px + body + CONTINUITY_SOURCE_LOOKAHEAD
    return wan_align_frame_count(raw), lock_px


def _proxy_mad(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cheap mean-abs diff on a spatial proxy (RGB only)."""
    if a.shape[0] != b.shape[0]:
        return 1e9
    aa = a[..., :3].float()
    bb = b[..., :3].float()
    # Downsample for speed; values are 0..1 tensors from ComfyUI.
    step_h = max(1, int(aa.shape[1]) // 64)
    step_w = max(1, int(aa.shape[2]) // 64)
    aa = aa[:, ::step_h, ::step_w, :]
    bb = bb[:, ::step_h, ::step_w, :]
    # Scale to ~0..255 so thresholds match the offline video diagnostics.
    return float((aa - bb).abs().mean().item() * 255.0)


def count_leading_seam_echo_frames(
    body: torch.Tensor,
    prev_tail: torch.Tensor | None,
    *,
    max_skip: int = 4,
    mad_threshold: float = CONTINUITY_SEAM_ECHO_MAD,
) -> int:
    """Diagnostic: count leading body frames that replay ``prev_tail``.

    Not used to shift export/merge — dropping these removes the SCAIL handoff
    (result looks like continuity is off, especially on successful rv2v).
    """
    if (
        prev_tail is None
        or max_skip <= 0
        or int(body.shape[0]) <= 0
        or int(prev_tail.shape[0]) <= 0
    ):
        return 0
    limit = min(int(max_skip), int(body.shape[0]), int(prev_tail.shape[0]))
    if limit <= 0:
        return 0
    for k in range(limit, 0, -1):
        if _proxy_mad(body[:k], prev_tail[-k:]) <= mad_threshold:
            return k
    return 0


def match_clip_to_gen_length(clip: torch.Tensor, gen_frames: int) -> torch.Tensor:
    """Make source length equal ``gen_frames`` without freezing the last frame.

    Prefer truncating; if short, mirror-extend from existing motion (same idea as
    ``encode_tail_clip``) so Wan is not asked to invent hollow tail frames.
    """
    gen = max(1, int(gen_frames))
    n = int(clip.shape[0])
    if n == gen:
        return clip
    if n > gen:
        return clip[:gen]
    need = gen - n
    pad = clip[: min(need, n)].flip(0)
    while int(pad.shape[0]) < need:
        pad = torch.cat([pad, clip.flip(0)], dim=0)
    return torch.cat([clip, pad[:need]], dim=0)


def is_continuity_active(plan: DirectorPlan, seg: SegmentPlan) -> bool:
    return (
        plan.continuity_enabled
        and plan.segment_count >= 2
        and seg.index > 0
        and seg.task_key in CONTINUITY_TASK_KEYS
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


def _frame_mean_luminance(frame: torch.Tensor) -> float:
    """Mean Rec.601 luma for an HWC tensor in [0, 1] (from 8d2e27c)."""
    f = frame.float()
    luma = 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]
    return float(luma.mean().item())


def _tensor_mean_luminance(frames: torch.Tensor) -> float:
    """Mean Rec.601 luma over a frame batch (from 8d2e27c)."""
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
    """Match guide batch mean luma to a source frame (8d2e27c prepend path)."""
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


def prepend_continuity_source(
    clip_frames: torch.Tensor,
    prev_output: torch.Tensor | None,
    *,
    lock_px: int,
    width: int,
    height: int,
) -> torch.Tensor:
    """Prepend prev-tail so Bernini source canvas aligns with SCAIL-locked prefix.

    Without this, generation length is extended but source stays segment-only →
    temporal misalignment → hard seams / pose resets at segment boundaries.

    Restores 8d2e27c luma normalize on the guide prefix so exposure does not
    flash at the cut (no whole-segment post luma — that caused smile drift).
    """
    if prev_output is None or lock_px <= 0 or int(clip_frames.shape[0]) <= 0:
        return clip_frames
    n = min(int(lock_px), int(prev_output.shape[0]))
    if n <= 0:
        return clip_frames
    body = fit_canvas(clip_frames, width, height)
    prefix = normalize_guide_luma_to_source(
        prev_output[-n:],
        body[0],
        width=width,
        height=height,
    )
    prefix = prefix.to(device=body.device, dtype=body.dtype)
    out = torch.cat([prefix, body], dim=0)
    log.info(
        "Segment continuity: prepended %d source frame(s) from prev tail (%d body)",
        n,
        int(body.shape[0]),
    )
    return out


def _normalize_context_latent_5d(latent: torch.Tensor) -> torch.Tensor:
    """Match official BerniniConditioning: keep VAE encode as ``[1, C, F, H, W]``."""
    if latent.ndim == 3:
        latent = latent.unsqueeze(0).unsqueeze(2)
    elif latent.ndim == 4:
        latent = latent.unsqueeze(0)
    if latent.ndim != 5:
        raise ValueError(
            f"Context latent must be 5D [1,C,F,H,W] (got {tuple(latent.shape)})"
        )
    if int(latent.shape[0]) != 1:
        raise ValueError(f"Context latent batch must be 1, got {tuple(latent.shape)}")
    return latent


def _latent_frame_count(pixel_frames: int) -> int:
    return max(1, (max(1, int(pixel_frames)) - 1) // 4 + 1)


def _encode_video_latent(vae, frames: torch.Tensor) -> torch.Tensor:
    """Encode [F,H,W,C] frames with ComfyUI native VAE → 5D [1,C,F,H,W]."""
    lat = vae.encode(frames[..., :3])
    return _normalize_context_latent_5d(lat)


def _encode_reference_latent(vae, frame: torch.Tensor, ref_max_size: int) -> torch.Tensor:
    """Encode a single reference frame (long-edge resize) with native VAE."""
    resized = fit_long_edge(frame[..., :3], int(ref_max_size))
    lat = vae.encode(resized)
    return _normalize_context_latent_5d(lat)


def encode_tail_clip(
    tail_clip: torch.Tensor,
    *,
    vae,
    width: int,
    height: int,
) -> torch.Tensor:
    """VAE-encode prev-tail clip for SCAIL lock (must already be Wan 4n+1 length)."""
    clip = fit_canvas(tail_clip, width, height)
    aligned = wan_align_frame_count(int(clip.shape[0]))
    if int(clip.shape[0]) > aligned:
        clip = clip[:aligned]
    elif int(clip.shape[0]) < aligned:
        # Prefer mirror from existing motion over freezing last frame.
        need = aligned - int(clip.shape[0])
        pad = clip[: min(need, int(clip.shape[0]))].flip(0)
        while int(pad.shape[0]) < need:
            pad = torch.cat([pad, clip.flip(0)], dim=0)
        clip = torch.cat([clip, pad[:need]], dim=0)
    return _encode_video_latent(vae, clip)


def apply_scail_prefix_to_latent(
    latent: dict[str, Any],
    tail_latent: torch.Tensor,
    overlap_pixel_frames: int,
) -> tuple[dict[str, Any], bool]:
    """Write prev-tail prefix into latent + noise_mask=0 (official KSampler path).

    Returns ``(latent, applied)``. Spatial mismatch is resized instead of silently
    skipping (silent skip made continuity look identical to OFF).
    """
    tail_latent = _normalize_context_latent_5d(tail_latent)
    samples = latent["samples"]
    if samples.ndim == 4:
        samples = samples.unsqueeze(0)
    _, _c, t_total, h, w = samples.shape
    th, tw = int(tail_latent.shape[3]), int(tail_latent.shape[4])
    if th != h or tw != w:
        log.warning(
            "Segment continuity: SCAIL spatial mismatch tail %dx%d vs latent %dx%d "
            "— resizing tail latent (was a silent skip)",
            th,
            tw,
            h,
            w,
        )
        # [1,C,F,H,W] → interpolate over H,W per temporal slice.
        b, c, f, _, _ = tail_latent.shape
        flat = tail_latent.reshape(b * c * f, 1, th, tw)
        flat = torch.nn.functional.interpolate(
            flat.float(), size=(h, w), mode="bilinear", align_corners=False
        )
        tail_latent = flat.to(dtype=tail_latent.dtype).reshape(b, c, f, h, w)
    aligned_pixels = wan_align_frame_count(int(overlap_pixel_frames))
    t_tail = min(
        int(tail_latent.shape[2]),
        _latent_frame_count(aligned_pixels),
        t_total,
    )
    if t_tail <= 0:
        return latent, False

    out = dict(latent)
    patched = samples.clone()
    patched[:, :, :t_tail] = tail_latent[:, :, :t_tail].to(
        device=patched.device, dtype=patched.dtype
    )
    noise_mask = torch.ones(
        (1, 1, t_total, h, w),
        dtype=patched.dtype,
        device=patched.device,
    )
    noise_mask[:, :, :t_tail] = 0.0
    # Feather the last locked latent frame(s) so free body does not hard-cut away
    # from the frozen prefix (that cliff becomes the segment seam after trim).
    feather = min(int(CONTINUITY_LOCK_FEATHER_LATENT), max(0, t_tail - 1))
    if feather > 0:
        for i in range(feather):
            # ramp 0 → CONTINUITY_LOCK_FEATHER_MASK across the feather window
            t = t_tail - feather + i
            alpha = float(CONTINUITY_LOCK_FEATHER_MASK) * float(i + 1) / float(feather)
            noise_mask[:, :, t] = alpha
    out["samples"] = patched
    out["noise_mask"] = noise_mask
    log.info(
        "Segment continuity: SCAIL prefix lock %d latent frame(s) (%d px, feather=%d)",
        t_tail,
        aligned_pixels,
        feather,
    )
    return out, True


def _context_latents_from_conditioning(conditioning) -> list[torch.Tensor]:
    for _tensor, payload in conditioning or []:
        if isinstance(payload, dict):
            streams = payload.get("context_latents")
            if streams:
                return list(streams)
    return []


def append_tail_reference_latent(
    streams: list[torch.Tensor],
    prev_output: torch.Tensor,
    *,
    vae,
    width: int,
    height: int,
    ref_max_size: int,
    n_frames: int = 1,
) -> list[torch.Tensor]:
    """Append last ``n_frames`` of prev as appearance reference latents."""
    if int(prev_output.shape[0]) <= 0:
        return streams
    n = min(max(1, int(n_frames)), int(prev_output.shape[0]))
    out = list(streams)
    for frame in fit_canvas(prev_output[-n:], width, height):
        lat = _encode_reference_latent(vae, frame.unsqueeze(0), ref_max_size)
        out.append(lat)
    log.info("Segment continuity: appended %d appearance reference latent(s)", n)
    return out


def apply_continuity_to_core_conditioning(
    positive,
    negative,
    *,
    prev_output: torch.Tensor,
    vae,
    width: int,
    height: int,
    ref_max_size: int,
    n_ref_frames: int = 1,
):
    """Append appearance ref only — motion handoff lives in prepended source + SCAIL."""
    import node_helpers

    if int(n_ref_frames) <= 0:
        return positive, negative
    streams = _context_latents_from_conditioning(positive)
    streams = append_tail_reference_latent(
        streams,
        prev_output,
        vae=vae,
        width=width,
        height=height,
        ref_max_size=ref_max_size,
        n_frames=n_ref_frames,
    )
    payload = {"context_latents": streams}
    positive = node_helpers.conditioning_set_values(positive, payload)
    negative = node_helpers.conditioning_set_values(negative, payload)
    return positive, negative


def apply_scail_continuity_core(
    *,
    plan: DirectorPlan,
    seg: SegmentPlan,
    prev_output: torch.Tensor | None,
    positive,
    negative,
    vae,
    width: int,
    height: int,
    ref_max_size: int = 848,
    latent: dict[str, Any] | None = None,
) -> tuple[Any, Any, dict[str, Any] | None, str | None]:
    """SCAIL latent prefix + appearance refs. Source prepend happens in executor."""
    if not is_continuity_active(plan, seg) or prev_output is None:
        return positive, negative, latent, None

    lock_px = min(
        resolve_continuity_lock_pixels(plan.continuity_overlap_frames),
        int(prev_output.shape[0]),
    )
    if lock_px <= 0:
        return positive, negative, latent, None

    _, ref_frames, _, _, _ = resolve_continuity_guide_frames(plan.continuity_overlap_frames)

    tail_clip = fit_canvas(prev_output[-lock_px:], width, height)
    tail_latent = encode_tail_clip(
        tail_clip,
        vae=vae,
        width=width,
        height=height,
    )
    positive, negative = apply_continuity_to_core_conditioning(
        positive,
        negative,
        prev_output=prev_output,
        vae=vae,
        width=width,
        height=height,
        ref_max_size=ref_max_size,
        n_ref_frames=ref_frames,
    )
    scail_ok = False
    if latent is not None:
        latent, scail_ok = apply_scail_prefix_to_latent(latent, tail_latent, lock_px)

    t_lock = _latent_frame_count(wan_align_frame_count(lock_px))
    if scail_ok:
        note = (
            f"  continuity branch seg #{seg.index + 1}: source-prepend + SCAIL lock "
            f"{lock_px}f ({t_lock} latent, kept through dual-stage) + {ref_frames}f ref "
            f"(overlap {plan.continuity_overlap_frames})"
        )
    else:
        note = (
            f"  continuity branch seg #{seg.index + 1}: source-prepend + {ref_frames}f ref "
            f"— SCAIL lock FAILED (seg will jump like continuity OFF)"
        )
        log.error(
            "Segment continuity: SCAIL lock failed for seg #%d — handoff degraded",
            seg.index + 1,
        )
    return positive, negative, latent, note


def _blend_opening_toward_last_frame(
    body: torch.Tensor,
    guide: torch.Tensor,
    *,
    blend_frames: int = CONTINUITY_OPENING_LAST_FRAME_BLEND,
    max_weight: float = CONTINUITY_OPENING_LAST_FRAME_WEIGHT,
) -> torch.Tensor:
    """Nudge body head toward prev's last frame only — never replay prev[-n:]."""
    n = min(int(blend_frames), int(body.shape[0]))
    if n <= 0 or guide is None or int(guide.shape[0]) <= 0 or max_weight <= 0:
        return body
    out = body.clone()
    last = guide[-1:].to(device=out.device, dtype=out.dtype)
    if last.shape[1:] != out.shape[1:]:
        last = fit_canvas(last, int(out.shape[2]), int(out.shape[1]))
    for i in range(n):
        w = float(max_weight) * (1.0 - float(i) / float(n))
        out[i] = last[0] * w + out[i] * (1.0 - w)
    return out


def trim_decoded_for_continuity(
    decoded: torch.Tensor,
    *,
    prefix_trim: int,
    target_len: int,
    prev_tail: torch.Tensor | None = None,
    max_echo_skip: int = 0,
) -> torch.Tensor:
    """Drop SCAIL overlap prefix, optional last-frame nudge, keep body length.

    Sequence crossfade against ``prev[-n:]`` is intentionally NOT used — it
    replayed the previous ending at every cut (duplicate frames in 00101).
    """
    del max_echo_skip  # retained for call-site compat; not used
    prefix_trim = max(0, int(prefix_trim))
    out = decoded
    if prefix_trim > 0 and int(decoded.shape[0]) > prefix_trim:
        out = decoded[prefix_trim:]
    elif prefix_trim > 0 and int(decoded.shape[0]) == prefix_trim:
        out = decoded[:0]

    if target_len > 0 and int(out.shape[0]) > target_len:
        out = out[:target_len]

    guide = None
    if prev_tail is not None and int(prev_tail.shape[0]) > 0:
        guide = prev_tail
    elif prefix_trim > 0 and int(decoded.shape[0]) > prefix_trim:
        guide = decoded[:prefix_trim]

    if guide is not None and int(out.shape[0]) > 0:
        out = _blend_opening_toward_last_frame(out, guide)
    return out


def continuity_merged_frame_count(plan: DirectorPlan) -> int:
    return int(plan.total_frames)


def concat_continuous_chunks(
    chunks: list[torch.Tensor],
    segments: list[SegmentPlan],
    plan: DirectorPlan,
) -> torch.Tensor:
    """Concatenate segment bodies in order (no sequence crossfade / no replay)."""
    del segments, plan
    if not chunks:
        raise ValueError("concat_continuous_chunks: no chunks")
    return cat_frames_variable_size(chunks)


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
