"""ComfyUI core dual-stage sampling for official Bernini (KSamplerAdvanced)."""

from __future__ import annotations

import logging
from typing import Callable

log = logging.getLogger("ComfyUI-Bernini-Director.director.core_sampling")

PhaseCallback = Callable[[str, float], None]


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
    on_phase: PhaseCallback | None = None,
):
    from nodes import KSamplerAdvanced

    def notify(phase: str, value: float) -> None:
        if on_phase:
            on_phase(phase, value)

    sampler = KSamplerAdvanced()
    split_step = max(1, min(int(split_step), int(steps) - 1))

    notify("high_noise", 0)
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
    notify("high_noise", 1)

    # Prefix lock only needed on the high-noise stage. Carrying noise_mask into the
    # low stage (leftover-noise handoff) can leave salt-pepper / snowflake artifacts
    # on Bernini dual-stage Wan sampling.
    latent_low_in = latent_high
    if isinstance(latent_high, dict) and "noise_mask" in latent_high:
        latent_low_in = dict(latent_high)
        latent_low_in.pop("noise_mask", None)

    notify("low_noise", 0)
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
        latent_low_in,
        split_step,
        int(steps),
        "disable",
    )
    notify("low_noise", 1)
    return latent_low
