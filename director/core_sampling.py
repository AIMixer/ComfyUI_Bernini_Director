"""ComfyUI core dual-stage sampling for official Bernini (KSamplerAdvanced)."""

from __future__ import annotations

import logging
from typing import Any, Callable

import torch

log = logging.getLogger("ComfyUI-Bernini-Director.director.core_sampling")

PhaseCallback = Callable[[str, float], None]


def _carry_scail_lock_into_low_stage(
    latent_high: dict[str, Any],
    latent_locked: dict[str, Any],
) -> dict[str, Any]:
    """Keep SCAIL prefix frozen through the low-noise stage.

    Historical bug (efc6811): ``noise_mask`` was stripped before low sampling so
    the locked prev-tail was freely rewritten — continuity looked identical to OFF.
    Official SCAIL-style two-phase samplers keep an anchor mask in phase 2 as well.

    Also re-bake mask==0 regions from the pre-high locked latent so any high-stage
    leakage cannot drift the handoff into low.
    """
    if not isinstance(latent_high, dict) or not isinstance(latent_locked, dict):
        return latent_high
    mask = latent_locked.get("noise_mask")
    src = latent_locked.get("samples")
    if mask is None or src is None:
        return latent_high

    out = dict(latent_high)
    samples = latent_high["samples"]
    if samples.ndim == 4:
        samples = samples.unsqueeze(0)
    src_samples = src
    if src_samples.ndim == 4:
        src_samples = src_samples.unsqueeze(0)

    m = mask.to(device=samples.device, dtype=samples.dtype)
    # Support (1,1,T,H,W) video masks and older spatial-only masks.
    while m.ndim < samples.ndim:
        m = m.unsqueeze(0)
    if m.shape[-2:] != samples.shape[-2:]:
        m = torch.nn.functional.interpolate(
            m.reshape(-1, 1, m.shape[-2], m.shape[-1]),
            size=samples.shape[-2:],
            mode="nearest",
        ).reshape(*m.shape[:-2], samples.shape[-2], samples.shape[-1])
    if m.shape[2] != samples.shape[2] and m.ndim == 5:
        # Temporal mismatch — fall back to carrying mask as-is on overlapping prefix.
        t = min(int(m.shape[2]), int(samples.shape[2]))
        m_use = torch.ones(
            (1, 1, samples.shape[2], samples.shape[3], samples.shape[4]),
            dtype=samples.dtype,
            device=samples.device,
        )
        m_use[:, :, :t] = m[:, :, :t]
        m = m_use

    src_samples = src_samples.to(device=samples.device, dtype=samples.dtype)
    if src_samples.shape != samples.shape:
        # Copy overlapping prefix only.
        t = min(int(src_samples.shape[2]), int(samples.shape[2]))
        h = min(int(src_samples.shape[3]), int(samples.shape[3]))
        w = min(int(src_samples.shape[4]), int(samples.shape[4]))
        patched = samples.clone()
        locked = (m[:, :, :t, :h, :w] < 0.5).to(dtype=samples.dtype)
        patched[:, :, :t, :h, :w] = (
            patched[:, :, :t, :h, :w] * (1.0 - locked)
            + src_samples[:, :, :t, :h, :w] * locked
        )
        out["samples"] = patched
    else:
        locked = (m < 0.5).to(dtype=samples.dtype)
        out["samples"] = samples * (1.0 - locked) + src_samples * locked

    out["noise_mask"] = m
    log.info("Segment continuity: carrying SCAIL noise_mask into low-noise stage")
    return out


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

    # Keep SCAIL prefix lock through low stage (do NOT strip noise_mask).
    latent_low_in = _carry_scail_lock_into_low_stage(latent_high, latent)

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
