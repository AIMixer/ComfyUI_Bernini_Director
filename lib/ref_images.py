"""Reference-image slot helpers (Bernini official: image0 … image4)."""

from __future__ import annotations

import torch

MAX_REFERENCE_IMAGES = 5
REF_IMAGE_KEY_PREFIX = "reference_image_"


def reference_image_input_types() -> dict:
    """ComfyUI optional inputs: reference_image_0 … reference_image_4 only."""
    return {
        f"{REF_IMAGE_KEY_PREFIX}{index}": (
            "IMAGE",
            {
                "tooltip": (
                    f"Reference image for prompt token image{index}. "
                    f"Long edge ≤ ref_max_size, native aspect kept."
                ),
            },
        )
        for index in range(MAX_REFERENCE_IMAGES)
    }


def flatten_reference_kwargs(kwargs: dict) -> dict[str, torch.Tensor | None]:
    refs: dict[str, torch.Tensor | None] = {}
    for key, value in kwargs.items():
        if not key.startswith(REF_IMAGE_KEY_PREFIX):
            continue
        index = int(key.removeprefix(REF_IMAGE_KEY_PREFIX))
        if index < 0 or index >= MAX_REFERENCE_IMAGES:
            raise ValueError(
                f"Invalid reference image slot {key!r}; "
                f"use reference_image_0 … reference_image_{MAX_REFERENCE_IMAGES - 1} only."
            )
        refs[key] = value
    return refs


def _slot_number(key: str) -> int:
    return int(key.removeprefix(REF_IMAGE_KEY_PREFIX))


def sorted_reference_items(
    extra_refs: dict[str, torch.Tensor | None],
) -> list[tuple[str, torch.Tensor]]:
    """Return connected reference images in image0, image1, … order."""
    items: list[tuple[str, torch.Tensor]] = []
    for key, value in extra_refs.items():
        if value is not None and value.shape[0] > 0:
            items.append((key, value))
    return sorted(items, key=lambda kv: _slot_number(kv[0]))


def collect_reference_batches(
    reference_images: torch.Tensor | None,
    extra_refs: dict[str, torch.Tensor | None],
) -> list[torch.Tensor]:
    batches: list[torch.Tensor] = []
    if reference_images is not None and reference_images.shape[0] > 0:
        batches.append(reference_images)

    for _, slot in sorted_reference_items(extra_refs):
        batches.append(slot)

    return batches
