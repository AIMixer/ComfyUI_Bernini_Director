"""Exact frame-count alignment for Director merge / cache / preview outputs."""

from __future__ import annotations

import torch


def pad_or_trim_frames(frames: torch.Tensor, target_len: int) -> torch.Tensor:
    """Trim to at most target_len frames. Does not fabricate last-frame duplicates."""
    target_len = max(0, int(target_len))
    if target_len <= 0:
        return frames[:0]
    if int(frames.shape[0]) > target_len:
        return frames[:target_len]
    return frames
