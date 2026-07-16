"""Disk cache for Bernini Director segment decode outputs (partial re-run + merge)."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import torch

import folder_paths

from .plan import DirectorPlan, SegmentPlan

log = logging.getLogger("ComfyUI-Bernini-Director.director.cache")


def _cache_root(node_id: str) -> Path:
    root = Path(folder_paths.get_output_directory()) / "bernini_seg_cache" / str(node_id)
    root.mkdir(parents=True, exist_ok=True)
    return root


def segment_cache_fingerprint(seg: SegmentPlan, plan: DirectorPlan) -> dict[str, Any]:
    """Stable identity for a segment — cache invalidates when edit params change."""
    ref_files = sorted(f"img{ref.index}" for ref in seg.refs)
    ref_video_file = (seg.reference_video_meta.get("videoFile") or seg.reference_video_meta.get("fileName") or "").strip()
    return {
        "index": seg.index,
        "start": seg.start_frame,
        "end": seg.end_frame,
        "prompt": seg.prompt,
        "negative": seg.negative_prompt,
        "task_key": seg.task_key,
        "width": plan.width,
        "height": plan.height,
        "output_mode": plan.output_mode,
        "ref_max": plan.ref_max_size,
        "refs": ref_files,
        "ref_video": ref_video_file,
        "ref_video_start": seg.reference_video_start_frame,
        "continuity": plan.continuity_enabled,
        "continuity_overlap": plan.continuity_overlap_frames if plan.continuity_enabled else 0,
    }


def save_segment_cache(
    node_id: str | None,
    seg: SegmentPlan,
    plan: DirectorPlan,
    tensor: torch.Tensor,
) -> None:
    if not node_id:
        return
    fp = segment_cache_fingerprint(seg, plan)
    root = _cache_root(node_id)
    idx = seg.index
    torch.save(tensor.cpu().float(), root / f"seg_{idx:04d}.pt")
    (root / f"seg_{idx:04d}.meta.json").write_text(
        json.dumps(fp, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    log.debug("Cached segment %d for node %s (%d frames)", idx + 1, node_id, tensor.shape[0])


def load_segment_cache(
    node_id: str | None,
    seg: SegmentPlan,
    plan: DirectorPlan,
) -> torch.Tensor | None:
    if not node_id:
        return None
    root = _cache_root(node_id)
    idx = seg.index
    meta_path = root / f"seg_{idx:04d}.meta.json"
    tensor_path = root / f"seg_{idx:04d}.pt"
    if not meta_path.is_file() or not tensor_path.is_file():
        return None
    try:
        stored = json.loads(meta_path.read_text(encoding="utf-8"))
        expected = segment_cache_fingerprint(seg, plan)
        if stored != expected:
            log.info(
                "Segment %d cache stale (timeline changed); re-run this segment to refresh.",
                idx + 1,
            )
            return None
        return torch.load(tensor_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        log.warning("Failed to load segment %d cache: %s", idx + 1, exc)
        return None
