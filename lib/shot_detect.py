"""Shot-boundary detection via PySceneDetect (AdaptiveDetector)."""

from __future__ import annotations

import logging
import os
from typing import Any

log = logging.getLogger("ComfyUI-Bernini-Director.shot_detect")

MIN_SEG_FRAMES = 4

# AdaptiveDetector.adaptive_threshold — higher = fewer cuts.
_SENSITIVITY_THRESHOLD = {
    "low": 4.5,
    "medium": 3.0,
    "high": 2.0,
}


def scenedetect_available() -> bool:
    try:
        import scenedetect  # noqa: F401

        return True
    except ImportError:
        return False


def _require_scenedetect():
    try:
        from scenedetect import AdaptiveDetector, detect
    except ImportError as exc:
        raise ImportError(
            "PySceneDetect is required for smart shot split. "
            "Install: pip install 'scenedetect<0.8'"
        ) from exc
    return AdaptiveDetector, detect


def _src_frame_to_logical(
    src_frame: int,
    *,
    native_fps: float,
    timeline_fps: float,
    logical_start: int,
    logical_end: int,
) -> int:
    """Map a source-video frame index into the clip's logical timeline range."""
    span = max(0, int(logical_end) - int(logical_start))
    if span <= 0:
        return int(logical_start)
    nf = float(native_fps) if native_fps and native_fps > 0 else 0.0
    tf = float(timeline_fps) if timeline_fps and timeline_fps > 0 else 0.0
    if nf <= 0 or tf <= 0:
        # Fallback: treat source index as already on the timeline scale.
        logical = int(logical_start) + int(src_frame)
    else:
        logical = int(round((float(src_frame) / nf) * tf)) + int(logical_start)
    return max(int(logical_start), min(int(logical_end), logical))


def _merge_close_cuts(cuts: list[int], *, min_gap: int, total: int) -> list[int]:
    """Keep 0 and total; drop interior cuts closer than min_gap to a neighbor."""
    gap = max(MIN_SEG_FRAMES, int(min_gap))
    ordered = sorted({0, int(total), *[int(c) for c in cuts if 0 < int(c) < int(total)]})
    if len(ordered) <= 2:
        return ordered
    kept = [ordered[0]]
    for cut in ordered[1:-1]:
        if cut - kept[-1] < gap:
            continue
        kept.append(cut)
    if ordered[-1] - kept[-1] < gap and len(kept) > 1:
        # Prefer keeping the end bound; drop the last interior cut if too close to end.
        kept.pop()
    kept.append(ordered[-1])
    # Second pass: ensure every consecutive pair spans at least MIN_SEG_FRAMES when possible.
    final = [kept[0]]
    for cut in kept[1:-1]:
        if cut - final[-1] < MIN_SEG_FRAMES:
            continue
        final.append(cut)
    if kept[-1] - final[-1] < MIN_SEG_FRAMES and len(final) > 1:
        final.pop()
    final.append(kept[-1])
    return final


def detect_shots_in_file(
    path: str,
    *,
    sensitivity: str = "medium",
    min_scene_len_src: int = 15,
) -> tuple[list[int], dict[str, Any]]:
    """
    Detect hard cuts in a video file.

    Returns (source_cut_frames_including_0_and_end, meta).
    """
    AdaptiveDetector, detect = _require_scenedetect()
    key = str(sensitivity or "medium").strip().lower()
    threshold = _SENSITIVITY_THRESHOLD.get(key, _SENSITIVITY_THRESHOLD["medium"])
    min_len = max(1, int(min_scene_len_src))

    detector = AdaptiveDetector(
        adaptive_threshold=float(threshold),
        min_scene_len=min_len,
    )
    scenes = detect(path, detector, show_progress=False, start_in_scene=True)
    if not scenes:
        return [0], {"method": "pyscenedetect_adaptive", "threshold": threshold, "scene_count": 0}

    cuts = [int(scenes[0][0].get_frames())]
    for start_tc, _end_tc in scenes[1:]:
        cuts.append(int(start_tc.get_frames()))
    end_frame = int(scenes[-1][1].get_frames())
    if end_frame not in cuts:
        cuts.append(end_frame)
    cuts = sorted(set(cuts))
    meta = {
        "method": "pyscenedetect_adaptive",
        "threshold": threshold,
        "scene_count": len(scenes),
        "sensitivity": key if key in _SENSITIVITY_THRESHOLD else "medium",
    }
    return cuts, meta


def detect_timeline_shot_cuts(
    clips: list[dict[str, Any]],
    *,
    frame_rate: float,
    total_frames: int,
    sensitivity: str = "medium",
    min_shot_frames: int = 12,
) -> dict[str, Any]:
    """
    Detect shot cuts across one or more timeline clips.

    Each clip dict needs: path, logicalStart, logicalEnd, and optionally nativeFps.
    Returns cutFrames (logical, including 0 and totalFrames).
    """
    from .video_io import resolve_video_path

    total = max(0, int(total_frames))
    if total < MIN_SEG_FRAMES * 2:
        return {
            "cutFrames": [0, total] if total > 0 else [0],
            "shotCount": 1 if total > 0 else 0,
            "method": "pyscenedetect_adaptive",
            "warnings": ["Timeline too short for shot split."],
        }

    if not scenedetect_available():
        raise ImportError(
            "PySceneDetect is required for smart shot split. "
            "Install: pip install 'scenedetect<0.8'"
        )

    timeline_fps = float(frame_rate) if frame_rate and frame_rate > 0 else 24.0
    min_shot = max(MIN_SEG_FRAMES, int(min_shot_frames or MIN_SEG_FRAMES))
    # Source min_scene_len roughly matches logical min at native≈timeline fps.
    min_scene_len_src = max(MIN_SEG_FRAMES, min_shot)

    logical_cuts: list[int] = [0, total]
    warnings: list[str] = []
    method = "pyscenedetect_adaptive"
    scene_total = 0

    for raw in clips or []:
        try:
            path = resolve_video_path(raw) if not raw.get("path") else str(raw["path"])
        except Exception as exc:
            warnings.append(f"Skip clip: {exc}")
            continue
        logical_start = int(raw.get("logicalStart") or raw.get("logical_start") or 0)
        logical_end = int(raw.get("logicalEnd") or raw.get("logical_end") or total)
        logical_start = max(0, min(total, logical_start))
        logical_end = max(logical_start, min(total, logical_end))
        if logical_end - logical_start < MIN_SEG_FRAMES * 2:
            continue

        native_fps = float(
            raw.get("nativeFps")
            or raw.get("native_fps")
            or timeline_fps
        )
        try:
            src_cuts, meta = detect_shots_in_file(
                path,
                sensitivity=sensitivity,
                min_scene_len_src=min_scene_len_src,
            )
        except Exception as exc:
            log.warning("Shot detect failed for %s: %s", path, exc)
            warnings.append(f"Detect failed ({os.path.basename(path)}): {exc}")
            continue

        method = str(meta.get("method") or method)
        scene_total += int(meta.get("scene_count") or 0)
        # Interior source cuts only (skip file start/end markers).
        interior = src_cuts[1:-1] if len(src_cuts) > 2 else []
        for src in interior:
            logical = _src_frame_to_logical(
                int(src),
                native_fps=native_fps,
                timeline_fps=timeline_fps,
                logical_start=logical_start,
                logical_end=logical_end,
            )
            if logical_start < logical < logical_end:
                logical_cuts.append(logical)

        # Always keep clip seams as cuts when provided as interior bounds.
        if 0 < logical_start < total:
            logical_cuts.append(logical_start)
        if 0 < logical_end < total:
            logical_cuts.append(logical_end)

    merged = _merge_close_cuts(logical_cuts, min_gap=min_shot, total=total)
    shot_count = max(0, len(merged) - 1)
    return {
        "cutFrames": merged,
        "shotCount": shot_count,
        "method": method,
        "warnings": warnings,
        "sceneCountRaw": scene_total,
    }
