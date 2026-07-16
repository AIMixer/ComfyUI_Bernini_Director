"""ComfyUI core dual-stage sampling for official Bernini (KSamplerAdvanced)."""

from __future__ import annotations

import logging

log = logging.getLogger("ComfyUI-Bernini-Director.director.core_sampling")


def sample_dual_stage(
    *,
    model_high,
    model_low,
    positive,
    negative,
    latent,
    high_seed: int,
    low_seed: int,
    high_cfg: float,
    low_cfg: float,
    steps: int,
    split_step: int,
    sampler_name: str,
    scheduler: str,
):
    from nodes import KSamplerAdvanced

    sampler = KSamplerAdvanced()
    split_step = max(1, min(int(split_step), int(steps) - 1))

    latent_high, = sampler.sample(
        model_high,
        "enable",
        int(high_seed),
        int(steps),
        float(high_cfg),
        sampler_name,
        scheduler,
        positive,
        negative,
        latent,
        0,
        split_step,
        "enable",
    )

    latent_low, = sampler.sample(
        model_low,
        "disable",
        int(low_seed),
        int(steps),
        float(low_cfg),
        sampler_name,
        scheduler,
        positive,
        negative,
        latent_high,
        split_step,
        int(steps),
        "disable",
    )
    return latent_low
