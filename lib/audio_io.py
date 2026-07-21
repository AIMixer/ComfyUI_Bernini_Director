"""Extract source audio aligned with Bernini Director timeline (v2v / rv2v)."""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from typing import Any

import torch

from .video_io import (
    ffprobe_bin,
    resolve_logical_frame_entry,
    resolve_video_path,
    video_clips_from_timeline,
)

log = logging.getLogger("ComfyUI-Bernini-Director.audio")

_ENCODE_ARGS = ("utf-8", "backslashreplace")


def _ffmpeg_bin() -> str | None:
    try:
        from imageio_ffmpeg import get_ffmpeg_exe

        return get_ffmpeg_exe()
    except ImportError:
        return shutil.which("ffmpeg")


def video_has_audio(path: str) -> bool | None:
    """Return True/False when ffprobe works; None when probe is unavailable."""
    probe = ffprobe_bin()
    if not probe:
        return None
    try:
        res = subprocess.run(
            [
                probe,
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=codec_type",
                "-of",
                "csv=p=0",
                path,
            ],
            capture_output=True,
            check=True,
        )
        return b"audio" in res.stdout.lower()
    except subprocess.CalledProcessError:
        return False


def _parse_ffmpeg_audio_info(stderr: str) -> tuple[int, int]:
    match = re.search(r", (\d+) Hz, (\w+), ", stderr)
    if match:
        ar = int(match.group(1))
        ac = {"mono": 1, "stereo": 2}.get(match.group(2), 2)
        return ar, ac
    return 44100, 2


def _probe_audio_stream(path: str) -> tuple[int, int]:
    """Return (sample_rate, channels) via ffprobe; fall back to stereo 44.1kHz."""
    probe = ffprobe_bin()
    if not probe or not path:
        return 44100, 2
    try:
        res = subprocess.run(
            [
                probe,
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=sample_rate,channels",
                "-of",
                "csv=p=0",
                path,
            ],
            capture_output=True,
            check=True,
        )
        text = res.stdout.decode(*_ENCODE_ARGS).strip()
        # csv: sample_rate,channels  (sometimes just one field)
        parts = [p.strip() for p in text.replace("\n", ",").split(",") if p.strip()]
        ar = int(float(parts[0])) if parts else 44100
        ac = int(parts[1]) if len(parts) > 1 else 2
        if ar <= 0:
            ar = 44100
        if ac <= 0:
            ac = 2
        return ar, ac
    except (subprocess.CalledProcessError, ValueError, IndexError):
        return 44100, 2


def extract_audio_segment(path: str, start_sec: float, duration_sec: float) -> dict[str, Any] | None:
    """Extract a slice of audio as ComfyUI AUDIO dict."""
    if duration_sec <= 0:
        return None
    ffmpeg = _ffmpeg_bin()
    if not ffmpeg or not path or not os.path.isfile(path):
        return None
    ar, _ac = _probe_audio_stream(path)
    # Force a known interleaved layout so reshape cannot disagree with ffmpeg output.
    # Stereo is the ComfyUI AUDIO convention used downstream.
    out_ac = 2
    args = [ffmpeg, "-v", "error", "-nostdin"]
    if start_sec > 0:
        args += ["-ss", str(max(0.0, start_sec))]
    args += [
        "-i",
        path,
        "-t",
        str(duration_sec),
        "-vn",
        "-ac",
        str(out_ac),
        "-ar",
        str(ar),
        "-f",
        "f32le",
        "-",
    ]
    try:
        res = subprocess.run(args, capture_output=True, check=True)
    except subprocess.CalledProcessError as exc:
        err = (exc.stderr or b"").decode(*_ENCODE_ARGS)
        log.debug("No audio extracted from %s: %s", path, err.strip())
        return None
    if not res.stdout:
        return None
    # Keep probe rate when stderr is silent (-v error); fall back to ffmpeg banner parse.
    parsed_ar, _parsed_ac = _parse_ffmpeg_audio_info(res.stderr.decode(*_ENCODE_ARGS))
    if parsed_ar > 0:
        ar = parsed_ar
    audio = torch.frombuffer(bytearray(res.stdout), dtype=torch.float32)
    # Drop dangling samples when byte length is not divisible by channel count
    # (odd mono leftovers, truncated pipes, etc.). reshape() requires an exact size.
    usable = (int(audio.numel()) // out_ac) * out_ac
    if usable < out_ac:
        return None
    if usable < int(audio.numel()):
        log.debug(
            "Truncating %d dangling audio value(s) from %s (not divisible by %d ch)",
            int(audio.numel()) - usable,
            path,
            out_ac,
        )
        audio = audio[:usable]
    audio = audio.reshape((-1, out_ac)).transpose(0, 1).unsqueeze(0)
    return {"waveform": audio, "sample_rate": ar}


def _merge_audio_spans(spans: list[tuple[str, float, float]]) -> dict[str, Any] | None:
    chunks: list[dict[str, Any]] = []
    for path, start, duration in spans:
        piece = extract_audio_segment(path, start, duration)
        if piece is not None:
            chunks.append(piece)
    if not chunks:
        return None
    sample_rate = chunks[0]["sample_rate"]
    channels = chunks[0]["waveform"].shape[1]
    merged: list[torch.Tensor] = []
    for chunk in chunks:
        wave = chunk["waveform"]
        if int(chunk["sample_rate"]) != sample_rate:
            log.warning(
                "Mixed sample rates in timeline audio (%s vs %s); skipping chunk.",
                chunk["sample_rate"],
                sample_rate,
            )
            continue
        if wave.shape[1] != channels:
            log.warning("Mixed channel counts in timeline audio; skipping chunk.")
            continue
        merged.append(wave)
    if not merged:
        return None
    return {"waveform": torch.cat(merged, dim=-1), "sample_rate": sample_rate}


def _timeline_audio_spans(
    timeline: dict,
    logical_start: int,
    logical_end: int,
    frame_rate: float,
) -> list[tuple[str, float, float]]:
    """Map logical frames to contiguous source-audio spans."""
    if logical_end <= logical_start or frame_rate <= 0:
        return []
    clips = video_clips_from_timeline(timeline)
    if not clips:
        return []

    fps = float(frame_rate)
    frame_dur = 1.0 / fps
    spans: list[tuple[str, float, float]] = []
    current_path: str | None = None
    current_start = 0.0
    current_end = 0.0

    def flush() -> None:
        nonlocal current_path, current_start, current_end
        if current_path and current_end > current_start:
            spans.append((current_path, current_start, current_end - current_start))
        current_path = None

    for logical in range(logical_start, logical_end):
        clip_idx, src_frame = resolve_logical_frame_entry(timeline, logical)
        if clip_idx < 0 or clip_idx >= len(clips):
            clip_idx = 0
        try:
            path = resolve_video_path(clips[clip_idx])
        except ValueError:
            flush()
            continue
        t0 = float(src_frame) / fps
        t1 = t0 + frame_dur
        if current_path == path and abs(t0 - current_end) <= frame_dur * 0.51:
            current_end = t1
        else:
            flush()
            current_path = path
            current_start = t0
            current_end = t1
    flush()
    return spans


def extract_timeline_audio(
    timeline: dict,
    logical_start: int,
    logical_end: int,
    frame_rate: float,
) -> dict[str, Any] | None:
    """Extract source audio for logical timeline range [start, end)."""
    spans = _timeline_audio_spans(timeline, logical_start, logical_end, frame_rate)
    if not spans:
        return None
    if not _ffmpeg_bin():
        log.warning(
            "Source audio skipped: ffmpeg unavailable "
            "(install FFmpeg on PATH or `pip install imageio-ffmpeg`)."
        )
        return None
    paths = {path for path, _, _ in spans}
    probes = [video_has_audio(p) for p in paths]
    # Only skip when ffprobe positively confirms every source has no audio track.
    # If ffprobe is missing (None), still try ffmpeg extraction.
    if probes and all(p is False for p in probes):
        return None
    return _merge_audio_spans(spans)


def diagnose_source_audio_failure(
    timeline: dict,
    logical_start: int,
    logical_end: int,
    frame_rate: float,
) -> str:
    """Human-readable reason when source-audio extraction produced silence."""
    if not _ffmpeg_bin():
        return (
            "ffmpeg unavailable (install FFmpeg on PATH or `pip install imageio-ffmpeg`)"
        )
    spans = _timeline_audio_spans(timeline, logical_start, logical_end, frame_rate)
    if not spans:
        return "could not map timeline frames to a source video path"
    paths = sorted({path for path, _, _ in spans})
    probes = [video_has_audio(p) for p in paths]
    if probes and all(p is False for p in probes):
        return "input video has no audio track"
    if ffprobe_bin() is None:
        return (
            "audio extraction failed — ffprobe not found "
            "(install a full FFmpeg build with ffprobe on PATH) "
            "and/or ffmpeg could not decode audio from the source"
        )
    if any(p is True for p in probes):
        return "ffmpeg failed to extract audio despite an audio stream being present"
    return "audio extraction failed"
