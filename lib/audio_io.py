"""Extract source audio aligned with Bernini Director timeline (v2v / rv2v)."""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from typing import Any

import torch

from .video_io import resolve_logical_frame_entry, resolve_video_path, video_clips_from_timeline

log = logging.getLogger("ComfyUI-Bernini-Director.audio")

_ENCODE_ARGS = ("utf-8", "backslashreplace")


def _ffmpeg_bin() -> str | None:
    try:
        from imageio_ffmpeg import get_ffmpeg_exe

        return get_ffmpeg_exe()
    except ImportError:
        return shutil.which("ffmpeg")


def _ffprobe_bin() -> str | None:
    return shutil.which("ffprobe")


def video_has_audio(path: str) -> bool:
    """Return True when the file has at least one audio stream."""
    probe = _ffprobe_bin()
    if not probe:
        return False
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


def extract_audio_segment(path: str, start_sec: float, duration_sec: float) -> dict[str, Any] | None:
    """Extract a slice of audio as ComfyUI AUDIO dict."""
    if duration_sec <= 0:
        return None
    ffmpeg = _ffmpeg_bin()
    if not ffmpeg or not path or not os.path.isfile(path):
        return None
    args = [ffmpeg, "-v", "error", "-nostdin"]
    if start_sec > 0:
        args += ["-ss", str(max(0.0, start_sec))]
    args += ["-i", path, "-t", str(duration_sec), "-f", "f32le", "-"]
    try:
        res = subprocess.run(args, capture_output=True, check=True)
    except subprocess.CalledProcessError as exc:
        err = (exc.stderr or b"").decode(*_ENCODE_ARGS)
        log.debug("No audio extracted from %s: %s", path, err.strip())
        return None
    if not res.stdout:
        return None
    ar, ac = _parse_ffmpeg_audio_info(res.stderr.decode(*_ENCODE_ARGS))
    audio = torch.frombuffer(bytearray(res.stdout), dtype=torch.float32)
    if audio.numel() < ac:
        return None
    frames = audio.numel() // ac
    audio = audio.reshape((frames, ac)).transpose(0, 1).unsqueeze(0)
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
    paths = {path for path, _, _ in spans}
    if not any(video_has_audio(p) for p in paths):
        return None
    return _merge_audio_spans(spans)
