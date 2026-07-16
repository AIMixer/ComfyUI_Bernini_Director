"""ComfyUI core CLIP text encoding for the official Bernini director path."""

from __future__ import annotations

from ..lib.task_prompts import apply_task_system_prompt


def encode_core_conditioning(
    clip,
    *,
    task_type: str,
    positive_prompt: str,
    negative_prompt: str,
):
    from nodes import CLIPTextEncode

    positive_text = apply_task_system_prompt(task_type, positive_prompt)
    encoder = CLIPTextEncode()
    positive, = encoder.encode(clip, positive_text)
    negative, = encoder.encode(clip, negative_prompt or "")
    return positive, negative
