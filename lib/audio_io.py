"""Extract source audio aligned with Bernini Director timeline (v2v / rv2v).

Independent of video tensors, but same frame selection + correct clocks:

  1. logical → frameMap → source index
  2. native = round((src / timeline_fps) * opencv_fps)  # same as video decode
  3. cut on the **file PTS clock** of those natives:
       start      = frame0_PTS + native0 * frame_dur
       media_dur  = count * frame_dur
     (frame_dur = measured PTS delta; this file uses 0.04s ticks, not 1/avg_fps)
  4. linear-resample to exactly N / timeline_fps (IMAGE / Combine clock)

Without step 3→4, audio rides the timeline clock while pictures are source
frames on the PTS grid → progressive “audio ahead” on long clips.
Do **not** pack one PTS step into each output frame slot (that accelerates).
"""

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

_FULL_AUDIO_CACHE: dict[str, dict[str, Any]] = {}
_OPENCV_FPS_CACHE: dict[str, float] = {}
# path -> (pts0, frame_dur)
_PTS_TIMING_CACHE: dict[str, tuple[float, float]] = {}


def _ffmpeg_bin() -> str | None:
    try:
        from imageio_ffmpeg import get_ffmpeg_exe

        return get_ffmpeg_exe()
    except ImportError:
        return shutil.which("ffmpeg")


def video_has_audio(path: str) -> bool | None:
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


def frames_to_audio_samples(frame_count: int, fps: float, sample_rate: int) -> int:
    frame_count = max(0, int(frame_count))
    fps = float(fps or 24.0)
    sample_rate = max(1, int(sample_rate))
    if frame_count <= 0 or fps <= 0:
        return 0
    return int(round(frame_count * sample_rate / fps))


def _clip_probe_fps(clip: dict, timeline_fps: float) -> float:
    file_fps = float(clip.get("nativeFps") or clip.get("native_fps") or 0.0)
    if file_fps <= 0:
        file_fps = float(timeline_fps or 24.0)
    return file_fps


def _opencv_file_fps(path: str, fallback: float) -> float:
    """Same FPS ``load_video_resampled`` reads via ``cv2.CAP_PROP_FPS``."""
    cached = _OPENCV_FPS_CACHE.get(path)
    if cached is not None:
        return cached
    fps = 0.0
    try:
        import cv2

        cap = cv2.VideoCapture(path)
        if cap.isOpened():
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            cap.release()
    except Exception as exc:
        log.debug("OpenCV FPS probe failed for %s: %s", path, exc)
    if fps <= 0:
        fps = float(fallback or 24.0)
    _OPENCV_FPS_CACHE[path] = fps
    return fps


def _src_frame_to_native(src_frame: int, *, timeline_fps: float, file_fps: float) -> int:
    """Identical to ``load_video_resampled`` index mapping."""
    t_sec = max(0.0, float(int(src_frame)) / float(timeline_fps or 24.0))
    return int(round(t_sec * float(file_fps or timeline_fps or 24.0)))


def _probe_pts_timing(path: str, *, fallback_fps: float) -> tuple[float, float]:
    """Return (frame0_pts, frame_dur) from measured file PTS.

    Many clips have avg_frame_rate ≈ 24.85 but PTS ticks of exactly 0.04s (25).
    Audio must follow the PTS grid of the decoded frame index, not 1/avg_fps.
    """
    cached = _PTS_TIMING_CACHE.get(path)
    if cached is not None:
        return cached

    pts0 = 0.0
    frame_dur = 1.0 / float(fallback_fps or 24.0) if fallback_fps > 0 else 1.0 / 24.0
    probe = ffprobe_bin()
    if probe and path and os.path.isfile(path):
        try:
            res = subprocess.run(
                [
                    probe,
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-read_intervals",
                    "%+#2",
                    "-show_entries",
                    "frame=pts_time",
                    "-of",
                    "csv=p=0",
                    path,
                ],
                capture_output=True,
                check=True,
            )
            times: list[float] = []
            for line in res.stdout.decode(*_ENCODE_ARGS).splitlines():
                val = line.split(",")[0].strip()
                if not val or val.upper() == "N/A":
                    continue
                try:
                    times.append(float(val))
                except ValueError:
                    continue
                if len(times) >= 2:
                    break
            if times:
                pts0 = float(times[0])
            if len(times) >= 2 and times[1] > times[0]:
                frame_dur = float(times[1] - times[0])
        except (subprocess.CalledProcessError, OSError, ValueError):
            try:
                import json

                res = subprocess.run(
                    [
                        probe,
                        "-v",
                        "error",
                        "-select_streams",
                        "v:0",
                        "-show_entries",
                        "stream=start_time,r_frame_rate",
                        "-of",
                        "json",
                        path,
                    ],
                    capture_output=True,
                    check=True,
                )
                streams = json.loads(res.stdout.decode(*_ENCODE_ARGS)).get("streams") or []
                if streams:
                    pts0 = float(streams[0].get("start_time") or 0.0)
                    rate = str(streams[0].get("r_frame_rate") or "")
                    if "/" in rate:
                        num, den = rate.split("/", 1)
                        try:
                            r = float(num) / float(den)
                            if r > 0:
                                frame_dur = 1.0 / r
                        except ValueError:
                            pass
            except (subprocess.CalledProcessError, OSError, ValueError, KeyError):
                pass

    log.info(
        "Source audio PTS timing for %s: pts0=%.6fs frame_dur=%.6fs (%.4f fps tick)",
        os.path.basename(path),
        pts0,
        frame_dur,
        (1.0 / frame_dur) if frame_dur > 0 else 0.0,
    )
    _PTS_TIMING_CACHE[path] = (pts0, frame_dur)
    return pts0, frame_dur


def _load_full_audio(path: str) -> dict[str, Any] | None:
    cached = _FULL_AUDIO_CACHE.get(path)
    if cached is not None:
        return cached
    ffmpeg = _ffmpeg_bin()
    if not ffmpeg or not path or not os.path.isfile(path):
        return None
    ar, _ac = _probe_audio_stream(path)
    out_ac = 2
    args = [
        ffmpeg,
        "-v",
        "error",
        "-nostdin",
        "-i",
        path,
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
        log.debug("No audio loaded from %s: %s", path, err.strip())
        return None
    if not res.stdout:
        return None
    parsed_ar, _ = _parse_ffmpeg_audio_info(res.stderr.decode(*_ENCODE_ARGS))
    if parsed_ar > 0:
        ar = parsed_ar
    audio = torch.frombuffer(bytearray(res.stdout), dtype=torch.float32)
    usable = (int(audio.numel()) // out_ac) * out_ac
    if usable < out_ac:
        return None
    if usable < int(audio.numel()):
        audio = audio[:usable]
    wave = audio.reshape((-1, out_ac)).transpose(0, 1).unsqueeze(0).contiguous()
    out = {"waveform": wave, "sample_rate": int(ar)}
    _FULL_AUDIO_CACHE[path] = out
    return out


def _slice_samples(wave: torch.Tensor, *, src_start: int, n_samples: int) -> torch.Tensor:
    channels = int(wave.shape[1])
    have = int(wave.shape[-1])
    n_samples = max(0, int(n_samples))
    if n_samples <= 0:
        return wave[..., :0]
    src_start = max(0, int(src_start))
    if src_start >= have:
        return torch.zeros(1, channels, n_samples, dtype=wave.dtype, device=wave.device)
    take = wave[..., src_start : src_start + n_samples]
    got = int(take.shape[-1])
    if got == n_samples:
        return take.contiguous()
    pad = torch.zeros(1, channels, n_samples - got, dtype=wave.dtype, device=wave.device)
    return torch.cat([take, pad], dim=-1)


def _fit_wave_to_samples(wave: torch.Tensor, want: int) -> torch.Tensor:
    have = int(wave.shape[-1])
    if want <= 0:
        return wave[..., :0]
    if have == want:
        return wave
    if have > want:
        return wave[..., :want].contiguous()
    pad = torch.zeros(
        1, int(wave.shape[1]), want - have, dtype=wave.dtype, device=wave.device
    )
    return torch.cat([wave, pad], dim=-1)


def _resample_wave_to_samples(wave: torch.Tensor, want: int) -> torch.Tensor:
    """Map PTS-clock audio onto the timeline/Combine clock (linear)."""
    have = int(wave.shape[-1])
    if want <= 0:
        return wave[..., :0]
    if have == want:
        return wave
    if have <= 1:
        return _fit_wave_to_samples(wave, want)
    out = torch.nn.functional.interpolate(
        wave.float(), size=int(want), mode="linear", align_corners=True
    )
    return out.to(dtype=wave.dtype)


def _timeline_audio_spans(
    timeline: dict,
    logical_start: int,
    logical_end: int,
    frame_rate: float,
) -> list[tuple[str, float, float, float]]:
    """Spans: (path, start_sec, media_dur_on_pts_clock, out_dur_on_timeline)."""
    if logical_end <= logical_start or frame_rate <= 0:
        return []
    clips = video_clips_from_timeline(timeline)
    if not clips:
        return []

    fps = float(frame_rate)
    spans: list[tuple[str, float, float, float]] = []
    run_path: str | None = None
    run_file_fps = fps
    run_n0 = 0
    run_count = 0
    run_pts0 = 0.0
    run_frame_dur = 1.0 / fps

    def flush_run() -> None:
        nonlocal run_path, run_file_fps, run_n0, run_count, run_pts0, run_frame_dur
        if run_path and run_count > 0 and run_frame_dur > 0:
            start = float(run_pts0) + float(run_n0) * float(run_frame_dur)
            media_dur = float(run_count) * float(run_frame_dur)
            out_dur = float(run_count) / fps
            if media_dur > 0 and out_dur > 0:
                spans.append((run_path, max(0.0, start), media_dur, out_dur))
        run_path = None
        run_count = 0

    for logical in range(logical_start, logical_end):
        clip_idx, src_frame = resolve_logical_frame_entry(timeline, logical)
        if clip_idx < 0 or clip_idx >= len(clips):
            clip_idx = 0
        clip = clips[clip_idx]
        try:
            path = resolve_video_path(clip)
        except ValueError:
            flush_run()
            continue
        fallback_fps = _clip_probe_fps(clip, fps)
        file_fps = _opencv_file_fps(path, fallback_fps)
        native = _src_frame_to_native(src_frame, timeline_fps=fps, file_fps=file_fps)
        pts0, frame_dur = _probe_pts_timing(path, fallback_fps=file_fps)
        if (
            run_path == path
            and run_count > 0
            and abs(run_file_fps - file_fps) < 1e-6
            and abs(run_frame_dur - frame_dur) < 1e-9
            and native == run_n0 + run_count
        ):
            run_count += 1
            continue
        flush_run()
        run_path = path
        run_file_fps = file_fps
        run_n0 = native
        run_count = 1
        run_pts0 = pts0
        run_frame_dur = frame_dur
    flush_run()
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
    paths = {path for path, _, _, _ in spans}
    probes = [video_has_audio(p) for p in paths]
    if probes and all(p is False for p in probes):
        return None

    chunks: list[torch.Tensor] = []
    sr = 44100
    resampled_spans = 0
    media_total = 0.0
    out_total = 0.0
    for path, start_sec, media_dur, out_dur in spans:
        full = _load_full_audio(path)
        if full is None:
            return None
        sr = int(full["sample_rate"] or sr)
        wave = full["waveform"]
        i0 = max(0, int(round(float(start_sec) * sr)))
        n_media = max(1, int(round(float(media_dur) * sr)))
        n_out = max(1, int(round(float(out_dur) * sr)))
        piece = _slice_samples(wave, src_start=i0, n_samples=n_media)
        if n_out != n_media:
            resampled_spans += 1
            media_total += float(media_dur)
            out_total += float(out_dur)
            piece = _resample_wave_to_samples(piece, n_out)
        chunks.append(piece)

    if resampled_spans > 0:
        log.info(
            "Source audio: PTS→timeline remap on %d span(s), "
            "media %.3fs → timeline %.3fs (%d frame window(s))",
            resampled_spans,
            media_total,
            out_total,
            len(spans),
        )
        if len(spans) > 50 and resampled_spans == len(spans):
            log.warning(
                "Source audio: %d one-shot spans (timeline fps likely ≠ source PTS). "
                "Prefer matching Director fps to source; sync still remaps but "
                "fragmented cuts are less ideal.",
                len(spans),
            )

    if not chunks:
        return None
    merged = torch.cat(chunks, dim=-1)
    fps = float(frame_rate or 24.0)
    n_frames = max(0, int(logical_end) - int(logical_start))
    want = frames_to_audio_samples(n_frames, fps, sr)
    have = int(merged.shape[-1])
    if want > 0 and have != want:
        rel = abs(have - want) / float(want)
        if rel > 0.0005:
            merged = _resample_wave_to_samples(merged, want)
        else:
            merged = _fit_wave_to_samples(merged, want)
    return {"waveform": merged, "sample_rate": sr}


def extract_audio_segment(path: str, start_sec: float, duration_sec: float) -> dict[str, Any] | None:
    """Extract a time range from a file (legacy helper)."""
    if duration_sec <= 0:
        return None
    full = _load_full_audio(path)
    if full is None:
        return None
    sr = int(full["sample_rate"])
    i0 = max(0, int(round(float(start_sec) * sr)))
    n = max(1, int(round(float(duration_sec) * sr)))
    return {
        "waveform": _slice_samples(full["waveform"], src_start=i0, n_samples=n),
        "sample_rate": sr,
    }


def diagnose_source_audio_failure(
    timeline: dict,
    logical_start: int,
    logical_end: int,
    frame_rate: float,
) -> str:
    if not _ffmpeg_bin():
        return (
            "ffmpeg unavailable (install FFmpeg on PATH or `pip install imageio-ffmpeg`)"
        )
    spans = _timeline_audio_spans(timeline, logical_start, logical_end, frame_rate)
    if not spans:
        return "could not map timeline frames to a source video path"
    paths = sorted({path for path, _, _, _ in spans})
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
