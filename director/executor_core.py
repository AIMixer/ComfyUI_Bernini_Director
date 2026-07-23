"""Run Bernini Director segments through the official ComfyUI core pipeline."""

from __future__ import annotations

import logging

import torch

from ..lib.image_prep import fit_canvas, fit_video_long_edge
from ..nodes.conditioning import _run_conditioning
from .core_sampling import sample_dual_stage
from .core_text_encode import encode_core_conditioning
from .prompt_enhance_runtime import (
    PromptEnhanceSettings,
    maybe_enhance_segment_prompt,
    notify_prompt_enhanced,
)
from .segment_runtime import (
    frames_label,
    needs_source_video,
    resolve_segment_raw_clip,
    resolve_segment_raw_clip_with_lookahead,
    segment_passthrough_chunk,
    tensor_frame_to_jpeg_b64,
)
from .plan import (
    DirectorPlan,
    plan_summary,
    prepare_segment_clip,
    reference_video_for_segment,
    refs_to_kwargs_for_context,
    wan_align_frame_count,
)
from .progress import report_director_finish, report_director_progress, report_director_segment_preview
from .segment_cache import load_segment_cache, save_segment_cache
from .segment_continuity import (
    CONTINUITY_SOURCE_LOOKAHEAD,
    apply_cached_segment_continuity,
    apply_scail_continuity_core,
    concat_continuous_chunks,
    is_continuity_active,
    match_clip_to_gen_length,
    prepend_continuity_source,
    resolve_prev_segment_output,
    resolve_segment_generation_frames,
    trim_decoded_for_continuity,
)
from .vram_cleanup import cleanup_segment_vram

log = logging.getLogger("ComfyUI-Bernini-Director.director.core")


def execute_director_plan_core(
    plan: DirectorPlan,
    *,
    node_id: str | None = None,
    vae,
    model_high,
    model_low,
    clip,
    negative_prompt: str,
    high_noise_cfg: float = 1.0,
    high_noise_seed: int = 0,
    low_noise_cfg: float = 1.0,
    low_noise_seed: int = 0,
    steps: int = 6,
    split_step: int = 3,
    sampler: str = "res_multistep",
    scheduler: str = "simple",
    clear_vram_between_segments: bool = True,
    prompt_enhance: PromptEnhanceSettings | None = None,
) -> tuple[torch.Tensor, list[torch.Tensor], str]:
    """Process every segment with ComfyUI core Bernini conditioning + KSampler."""
    pe = prompt_enhance or PromptEnhanceSettings()
    from nodes import VAEDecode

    decoder = VAEDecode()

    all_segments = plan.segments
    run_indices = plan.run_indices if plan.run_indices is not None else frozenset(range(len(all_segments)))
    run_list = sorted(run_indices)
    seg_total = len(run_list)
    progress_pos = {idx: pos for pos, idx in enumerate(run_list)}

    output_chunks: list[torch.Tensor] = []
    segment_outputs: list[torch.Tensor] = []
    reports: list[str] = [plan_summary(plan), "", "Execution path: ComfyUI official Bernini-R"]
    if clear_vram_between_segments:
        reports.append(
            "VRAM: 段间清理显存已开启（多段时在 context 编码后卸载模型；"
            "单段时跳过采样前 unload，避免重载叠峰）"
        )
    if plan.run_indices is not None:
        skipped = [i + 1 for i in range(len(all_segments)) if i not in run_indices]
        reports.append(
            f"Run selection: {len(run_list)}/{len(all_segments)} segment(s) "
            f"(indices {[i + 1 for i in run_list]}; skipped {skipped or 'none'})"
        )

    completed_outputs: dict[int, torch.Tensor] = {}
    if plan.continuity_enabled:
        reports.append(
            "Segment continuity: ON — SCAIL prefix lock (both sample stages) + "
            "source prepend + luma-matched guide (no leading seam skip)"
        )
    else:
        reports.append(
            "Segment continuity: OFF — official Studio path "
            "(per-segment source + user refs only; no cross-segment injection)"
        )

    def _run_one_segment(seg, *, progress_index: int) -> torch.Tensor:
        meta = {
            "frames_label": frames_label(seg),
            "task_key": seg.task_key,
            "timeline_segment_index": seg.index,
            "timeline_segment_total": len(all_segments),
        }

        report_director_progress(
            node_id,
            segment_index=progress_index,
            segment_total=seg_total,
            phase="prepare",
            phase_value=0,
            phase_max=1,
            **meta,
        )

        is_one_frame_i2v = seg.task_key == "i2v" and seg.source_clip is not None
        # Per-segment gate: task must support continuity (v2v/rv2v/…) — never trim
        # t2v/r2v/gen heads just because the global checkbox is on.
        seg_continuity = is_continuity_active(plan, seg)
        # Continuity may need a few frames past the segment end so gen length
        # matches a fully-conditioned source canvas (avoids hollow-tail freeze).
        lookahead = 0
        if seg_continuity and not is_one_frame_i2v:
            # Conditioning-only frames past the body so gen length is covered.
            lookahead = CONTINUITY_SOURCE_LOOKAHEAD
        raw_clip = resolve_segment_raw_clip_with_lookahead(
            plan, seg, end_extra=lookahead
        )
        body_len = max(1, int(seg.frame_count or 0))
        if body_len <= 0:
            body_len = max(1, int(raw_clip.shape[0]) - lookahead)
        target_len = (
            max(1, seg.frame_count or body_len)
            if is_one_frame_i2v
            else body_len
        )
        # Body-only clip for export length; lookahead stays in raw for conditioning.
        body_raw = raw_clip[:target_len] if int(raw_clip.shape[0]) > target_len else raw_clip
        if seg.source_clip is not None:
            clip_frames = body_raw
        elif plan.output_mode == "fixed":
            clip_frames = fit_canvas(body_raw, plan.width, plan.height)
        else:
            clip_frames = fit_video_long_edge(body_raw, plan.ref_max_size)
        # Length prep: official Studio = this segment only; continuity prepends prev-tail.
        prev_tail_output = None
        if is_one_frame_i2v:
            num_frames = wan_align_frame_count(target_len)
            prefix_trim = 0
        elif seg_continuity:
            gen_frames, prefix_trim = resolve_segment_generation_frames(
                segment_frame_count=target_len,
                segment_index=seg.index,
                continuity_enabled=True,
                continuity_overlap=plan.continuity_overlap_frames,
            )
            # Keep body at the segment's own length (no last-frame pad).
            clip_frames, _ = prepare_segment_clip(clip_frames, max(1, int(target_len)))
            if needs_source_video(seg.task_key):
                if clip_frames is not None and clip_frames.shape[0] > 0:
                    ctx_h, ctx_w = int(clip_frames.shape[1]), int(clip_frames.shape[2])
                else:
                    ctx_w, ctx_h = plan.width, plan.height
                if prefix_trim > 0:
                    prev_tail_output = resolve_prev_segment_output(
                        plan, all_segments, seg.index, completed_outputs, node_id
                    )
                    # Prefer timeline lookahead for post-body conditioning frames.
                    if int(raw_clip.shape[0]) > target_len:
                        extra = raw_clip[target_len:]
                        if plan.output_mode == "fixed":
                            extra = fit_canvas(extra, plan.width, plan.height)
                        else:
                            extra = fit_video_long_edge(extra, plan.ref_max_size)
                        clip_frames = torch.cat([clip_frames, extra], dim=0)
                    clip_frames = prepend_continuity_source(
                        clip_frames,
                        prev_tail_output,
                        lock_px=prefix_trim,
                        width=ctx_w,
                        height=ctx_h,
                    )
                # Source must cover gen_frames; hollow tail → freeze / stutter near seams.
                clip_frames = match_clip_to_gen_length(clip_frames, int(gen_frames))
            num_frames = int(gen_frames)
        else:
            # Official Studio / Bernini single-clip path — no overlap padding.
            num_frames = wan_align_frame_count(max(1, int(target_len)))
            prefix_trim = 0
            clip_frames, _ = prepare_segment_clip(clip_frames, num_frames)

        report_director_progress(
            node_id,
            segment_index=progress_index,
            segment_total=seg_total,
            phase="prepare",
            phase_value=1,
            phase_max=1,
            **meta,
        )

        positive_prompt = seg.prompt
        seg_negative = (seg.negative_prompt or "").strip() or negative_prompt
        ref_video_pe = reference_video_for_segment(plan, seg, num_frames)
        source_pe = clip_frames if needs_source_video(seg.task_key) else None
        if pe.active:
            original = positive_prompt
            positive_prompt = maybe_enhance_segment_prompt(
                pe,
                task_type=seg.task_type,
                user_prompt=positive_prompt,
                source_clip=source_pe,
                refs=seg.refs,
                reference_video=ref_video_pe,
            )
            if positive_prompt != original:
                notify_prompt_enhanced(
                    node_id,
                    text=positive_prompt,
                    segment_index=seg.index,
                    field="segment" if not seg.use_global else "global",
                )

        report_director_progress(
            node_id,
            segment_index=progress_index,
            segment_total=seg_total,
            phase="text_encode",
            phase_value=0,
            phase_max=1,
            **meta,
        )
        positive, negative = encode_core_conditioning(
            clip,
            task_type=seg.task_type,
            positive_prompt=positive_prompt,
            negative_prompt=seg_negative,
        )
        report_director_progress(
            node_id,
            segment_index=progress_index,
            segment_total=seg_total,
            phase="text_encode",
            phase_value=1,
            phase_max=1,
            **meta,
        )

        if clear_vram_between_segments:
            cleanup_segment_vram(enabled=True)

        ref_kwargs = refs_to_kwargs_for_context(seg.task_key, seg.refs)
        source_arg = clip_frames if needs_source_video(seg.task_key) else None
        ref_video_arg = reference_video_for_segment(plan, seg, num_frames)

        if clip_frames is not None and clip_frames.shape[0] > 0:
            ctx_h, ctx_w = int(clip_frames.shape[1]), int(clip_frames.shape[2])
        else:
            ctx_w, ctx_h = plan.width, plan.height

        report_director_progress(
            node_id,
            segment_index=progress_index,
            segment_total=seg_total,
            phase="context_encode",
            phase_value=0,
            phase_max=1,
            **meta,
        )

        # BerniniConditioning on (possibly continuity-prepended) source + user refs.
        # SCAIL latent lock + appearance ref are applied immediately after.
        positive, negative, latent, task_hint = _run_conditioning(
            positive,
            negative,
            vae,
            ctx_w,
            ctx_h,
            num_frames,
            1,
            source_video=source_arg,
            reference_video=ref_video_arg,
            ref_max_size=plan.ref_max_size,
            **ref_kwargs,
        )
        if seg_continuity:
            if prev_tail_output is None:
                prev_tail_output = resolve_prev_segment_output(
                    plan, all_segments, seg.index, completed_outputs, node_id
                )
            positive, negative, latent, continuity_note = apply_scail_continuity_core(
                plan=plan,
                seg=seg,
                prev_output=prev_tail_output,
                positive=positive,
                negative=negative,
                vae=vae,
                width=ctx_w,
                height=ctx_h,
                ref_max_size=plan.ref_max_size,
                latent=latent,
            )
            if continuity_note:
                reports.append(continuity_note)
        report_director_progress(
            node_id,
            segment_index=progress_index,
            segment_total=seg_total,
            phase="context_encode",
            phase_value=1,
            phase_max=1,
            **meta,
        )

        if clear_vram_between_segments:
            cleanup_segment_vram(enabled=True, unload_models=seg_total > 1)

        def _report_sample_phase(phase: str, value: float) -> None:
            report_director_progress(
                node_id,
                segment_index=progress_index,
                segment_total=seg_total,
                phase=phase,
                phase_value=value,
                phase_max=1,
                **meta,
            )

        samples = sample_dual_stage(
            model_high=model_high,
            model_low=model_low,
            positive=positive,
            negative=negative,
            latent=latent,
            high_seed=high_noise_seed,
            low_seed=low_noise_seed,
            high_cfg=high_noise_cfg,
            low_cfg=low_noise_cfg,
            steps=steps,
            split_step=split_step,
            sampler_name=sampler,
            scheduler=scheduler,
            on_phase=_report_sample_phase,
        )

        report_director_progress(
            node_id,
            segment_index=progress_index,
            segment_total=seg_total,
            phase="decode",
            phase_value=0,
            phase_max=1,
            **meta,
        )
        decoded, = decoder.decode(vae, samples)
        report_director_progress(
            node_id,
            segment_index=progress_index,
            segment_total=seg_total,
            phase="decode",
            phase_value=1,
            phase_max=1,
            **meta,
        )

        # Length finalize only — no color/luma post (avoids smile drift / color shift).
        if seg_continuity:
            decoded = trim_decoded_for_continuity(
                decoded,
                prefix_trim=prefix_trim,
                target_len=target_len,
                prev_tail=prev_tail_output,
            )
        elif decoded.shape[0] > target_len:
            # Cut only — never pad with repeated last frames (visible stutter).
            decoded = decoded[:target_len]

        chunk = decoded.cpu().float()
        save_segment_cache(node_id, seg, plan, chunk)
        completed_outputs[seg.index] = chunk

        if plan.global_task_key in {"t2i", "i2i", "r2i"} and decoded.shape[0] >= 1:
            try:
                h, w = int(decoded.shape[1]), int(decoded.shape[2])
                report_director_segment_preview(
                    node_id,
                    segment_index=seg.index,
                    image_b64=tensor_frame_to_jpeg_b64(decoded[0]),
                    width=w,
                    height=h,
                )
            except Exception as exc:
                log.debug("Segment preview skipped: %s", exc)
        elif plan.global_task_key in {"t2v", "i2v", "r2v"} and decoded.shape[0] >= 1:
            try:
                frames_b64 = [
                    tensor_frame_to_jpeg_b64(decoded[i])
                    for i in range(int(decoded.shape[0]))
                ]
                h, w = int(decoded.shape[1]), int(decoded.shape[2])
                report_director_segment_preview(
                    node_id,
                    segment_index=seg.index,
                    image_b64=frames_b64[0],
                    width=w,
                    height=h,
                    frames=frames_b64,
                    fps=float(plan.frame_rate or 24),
                )
            except Exception as exc:
                log.debug("Segment video preview skipped: %s", exc)

        if clear_vram_between_segments:
            del positive, negative, latent, samples, decoded, clip_frames, source_arg, raw_clip
            cleanup_segment_vram(enabled=True)

        reports.append(
            f"Segment {seg.index + 1}/{len(all_segments)}: {task_hint} "
            f"({target_len} frames, high_seed={high_noise_seed}, low_seed={low_noise_seed})"
        )
        log.info(
            "Bernini Director [core] segment %d/%d done (%d frames, task=%s)",
            seg.index + 1,
            len(all_segments),
            target_len,
            seg.task_key,
        )
        return chunk

    for seg in all_segments:
        if seg.index in run_indices:
            if clear_vram_between_segments and segment_outputs:
                cleanup_segment_vram(enabled=True)
            chunk = _run_one_segment(seg, progress_index=progress_pos[seg.index])
            segment_outputs.append(chunk)
            if plan.export_mode == "all":
                output_chunks.append(chunk)
            continue

        if plan.export_mode != "all":
            continue

        cached = load_segment_cache(node_id, seg, plan)
        if cached is not None:
            cached = cached.float()
            cached = apply_cached_segment_continuity(
                cached, seg, plan, completed_outputs, width=plan.width, height=plan.height
            )
            completed_outputs[seg.index] = cached
            reports.append(
                f"Segment {seg.index + 1}/{len(all_segments)}: "
                f"loaded from cache ({cached.shape[0]} frames)"
            )
        elif needs_source_video(seg.task_key) or seg.source_clip is not None:
            try:
                cached = segment_passthrough_chunk(plan, seg)
                if cached is not None:
                    save_segment_cache(node_id, seg, plan, cached)
                    reports.append(
                        f"Segment {seg.index + 1}/{len(all_segments)}: "
                        f"source passthrough ({cached.shape[0]} frames, no prior cache)"
                    )
            except Exception as exc:
                log.warning("Segment %d source passthrough failed: %s", seg.index + 1, exc)
                cached = None
        if cached is None:
            raise ValueError(
                f"Segment {seg.index + 1} is not selected and has no valid cache. "
                "Run all segments once (全部运行), or include this segment in your run selection."
            )
        output_chunks.append(cached)

    if not output_chunks and not segment_outputs:
        raise ValueError("Director plan produced no segments.")

    report_director_finish(node_id, seg_total)
    export_chunks = output_chunks if output_chunks else segment_outputs
    export_segments = all_segments if output_chunks else [all_segments[i] for i in sorted(run_indices)]
    combined = concat_continuous_chunks(export_chunks, export_segments, plan)
    return combined, segment_outputs, "\n".join(reports)
