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
    apply_cached_segment_continuity,
    apply_continuity_conditioning_branch,
    apply_continuity_post_decode_branch,
    concat_continuous_chunks,
    resolve_segment_generation_frames,
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
            "Segment continuity: ON — opt-in branch "
            "(luma-normalized source-prefix / optional 1 ref / trim / source luma re-anchor)"
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

        raw_clip = resolve_segment_raw_clip(plan, seg)
        is_one_frame_i2v = seg.task_key == "i2v" and seg.source_clip is not None
        target_len = max(1, seg.frame_count or raw_clip.shape[0]) if is_one_frame_i2v else raw_clip.shape[0]
        if seg.source_clip is not None:
            clip_frames = raw_clip
        elif plan.output_mode == "fixed":
            clip_frames = fit_canvas(raw_clip, plan.width, plan.height)
        else:
            clip_frames = fit_video_long_edge(raw_clip, plan.ref_max_size)
        # Length prep: official Studio = this segment only; continuity may add overlap.
        if is_one_frame_i2v:
            num_frames = wan_align_frame_count(target_len)
            prefix_trim = 0
        elif plan.continuity_enabled:
            gen_frames, prefix_trim = resolve_segment_generation_frames(
                segment_frame_count=target_len,
                segment_index=seg.index,
                continuity_enabled=True,
                continuity_overlap=plan.continuity_overlap_frames,
            )
            base_frames = wan_align_frame_count(max(1, int(target_len)))
            clip_frames, _ = prepare_segment_clip(clip_frames, base_frames)
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

        # Default = official Studio inputs (this segment only). Continuity is a separate branch.
        cond_refs = dict(ref_kwargs)
        source_for_cond = source_arg
        continuity_note = None
        if plan.continuity_enabled:
            (
                source_for_cond,
                cond_refs,
                num_frames,
                prefix_trim,
                continuity_note,
            ) = apply_continuity_conditioning_branch(
                plan=plan,
                seg=seg,
                all_segments=all_segments,
                completed_outputs=completed_outputs,
                node_id=node_id,
                source_arg=source_arg,
                ref_kwargs=ref_kwargs,
                num_frames=num_frames,
                target_len=target_len,
                ctx_w=ctx_w,
                ctx_h=ctx_h,
            )

        positive, negative, latent, task_hint = _run_conditioning(
            positive,
            negative,
            vae,
            ctx_w,
            ctx_h,
            num_frames,
            1,
            source_video=source_for_cond,
            reference_video=ref_video_arg,
            ref_max_size=plan.ref_max_size,
            **cond_refs,
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

        report_director_progress(
            node_id,
            segment_index=progress_index,
            segment_total=seg_total,
            phase="high_noise",
            phase_value=0,
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
        )
        report_director_progress(
            node_id,
            segment_index=progress_index,
            segment_total=seg_total,
            phase="high_noise",
            phase_value=1,
            phase_max=1,
            **meta,
        )
        report_director_progress(
            node_id,
            segment_index=progress_index,
            segment_total=seg_total,
            phase="low_noise",
            phase_value=1,
            phase_max=1,
            **meta,
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

        if plan.continuity_enabled:
            # Continuity branch: trim overlap prefix + optional source luma re-anchor.
            decoded = apply_continuity_post_decode_branch(
                decoded,
                plan=plan,
                seg=seg,
                source_arg=source_arg,
                ref_kwargs=ref_kwargs,
                prefix_trim=prefix_trim,
                target_len=target_len,
                ctx_w=ctx_w,
                ctx_h=ctx_h,
            )
        else:
            # Official Studio path: length only — no cross-segment post-process.
            if decoded.shape[0] > target_len:
                decoded = decoded[:target_len]
            elif decoded.shape[0] < target_len and decoded.shape[0] > 0:
                pad = decoded[-1:].repeat(target_len - decoded.shape[0], 1, 1, 1)
                decoded = torch.cat([decoded, pad], dim=0)

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
