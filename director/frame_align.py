"""Exact frame-count alignment for Director merge / cache / preview outputs."""

from __future__ import annotations

import torch


def pad_or_trim_frames(frames: torch.Tensor, target_len: int) -> torch.Tensor:
    """Return exactly target_len frames, padding with the last frame when needed."""
    target_len = max(0, int(target_len))
    if target_len <= 0:
        return frames[:0]
    if int(frames.shape[0]) > target_len:
        return frames[:target_len]
    if int(frames.shape[0]) < target_len and int(frames.shape[0]) > 0:
        pad = frames[-1:].repeat(target_len - int(frames.shape[0]), 1, 1, 1)
        return torch.cat([frames, pad], dim=0)
    return frames
