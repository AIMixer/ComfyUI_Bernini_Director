"""Build ComfyUI AUDIO outputs for Bernini Director source-video edit runs."""

from __future__ import annotations

from typing import Any

import torch

from ..lib.audio_io import diagnose_source_audio_failure, extract_timeline_audio

SILENT_SAMPLE_RATE = 44100


def task_passes_source_audio(task_key: str) -> bool:
    return task_key in {"v2v", "mv2v", "rv2v", "vi2v", "vrc2v", "ads2v"}


def empty_audio_dict(sample_rate: int = SILENT_SAMPLE_RATE) -> dict[str, Any]:
    """Silent placeholder — ComfyUI AUDIO outputs must not be None."""
    return {"waveform": torch.zeros(1, 1, 0), "sample_rate": int(sample_rate)}


def _coerce_audio_output(audio: dict[str, Any] | None, *, sample_rate: int) -> dict[str, Any]:
    if audio is None:
        return empty_audio_dict(sample_rate)
    wave = audio.get("waveform")
    if not isinstance(wave, torch.Tensor) or wave.numel() <= 0:
        return empty_audio_dict(int(audio.get("sample_rate") or sample_rate))
    return audio


def _audio_has_samples(audio: dict[str, Any] | None) -> bool:
    return (
        isinstance(audio, dict)
        and isinstance(audio.get("waveform"), torch.Tensor)
        and int(audio["waveform"].numel()) > 0
    )


def build_director_audio_outputs(
    plan,
    images_out: list,
    *,
    export_segments: bool,
    output_frame_end: int | None = None,
) -> list[dict[str, Any]]:
    """Return one AUDIO dict per images_out entry (never None)."""
    fps = float(plan.frame_rate or 24.0)
    silent_sample_rate = SILENT_SAMPLE_RATE
    if not task_passes_source_audio(plan.global_task_key):
        return [empty_audio_dict(silent_sample_rate) for _ in images_out]

    timeline = plan.raw or {}

    if export_segments:
        if plan.run_indices is not None:
            seg_indices = sorted(plan.run_indices)
        else:
            seg_indices = list(range(len(plan.segments)))
        outputs: list[dict[str, Any]] = []
        for i, _tensor in enumerate(images_out):
            if i >= len(seg_indices):
                outputs.append(empty_audio_dict(silent_sample_rate))
                continue
            seg = plan.segments[seg_indices[i]]
            outputs.append(
                _coerce_audio_output(
                    extract_timeline_audio(timeline, seg.start_frame, seg.end_frame, fps),
                    sample_rate=silent_sample_rate,
                )
            )
        return outputs

    end = max(0, int(output_frame_end if output_frame_end is not None else plan.total_frames))
    audio = extract_timeline_audio(timeline, 0, end, fps) if end > 0 else None
    merged = _coerce_audio_output(audio, sample_rate=silent_sample_rate)
    return [merged] if len(images_out) == 1 else [empty_audio_dict(silent_sample_rate) for _ in images_out]


def source_audio_report_note(
    plan,
    audio_out: list,
    *,
    export_segments: bool,
    output_frame_end: int | None = None,
) -> str:
    """Append-ready report line for source-audio extraction status."""
    if not task_passes_source_audio(plan.global_task_key):
        return ""
    if any(_audio_has_samples(a) for a in audio_out):
        return (
            "\n\nSource audio: extracted from input video "
            "(connect audio → VHS Video Combine)."
        )

    fps = float(plan.frame_rate or 24.0)
    timeline = plan.raw or {}
    if export_segments and plan.segments:
        if plan.run_indices is not None:
            seg_indices = sorted(plan.run_indices)
        else:
            seg_indices = list(range(len(plan.segments)))
        seg = plan.segments[seg_indices[0]] if seg_indices else plan.segments[0]
        start, end = int(seg.start_frame), int(seg.end_frame)
    else:
        start = 0
        end = max(0, int(output_frame_end if output_frame_end is not None else plan.total_frames))
    hint = diagnose_source_audio_failure(timeline, start, end, fps)
    return f"\n\nSource audio: none — {hint}."
