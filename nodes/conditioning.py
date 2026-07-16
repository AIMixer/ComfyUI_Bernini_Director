"""Bernini conditioning — calls ComfyUI official BerniniConditioning only.

Source of truth: comfy_extras/nodes_bernini.py (Comfy-Org/ComfyUI).
No local reimplementation.
"""

from __future__ import annotations

from ..lib.ref_images import MAX_REFERENCE_IMAGES, REF_IMAGE_KEY_PREFIX, flatten_reference_kwargs
from ..lib.task_modes import TASK_DESCRIPTIONS, infer_task


def _shared_optional_inputs() -> dict:
    return {
        "source_video": (
            "IMAGE",
            {
                "tooltip": (
                    "Source video to edit or restyle (v2v, rv2v). "
                    "Resized to width/height and trimmed to length."
                ),
            },
        ),
        "reference_video": (
            "IMAGE",
            {
                "tooltip": "Video to insert into the source video (ads2v).",
            },
        ),
        **{
            f"{REF_IMAGE_KEY_PREFIX}{index}": (
                "IMAGE",
                {
                    "tooltip": (
                        "Reference image injected as an in-context token (r2v, rv2v). "
                        "Native aspect; long edge capped at ref_max_size."
                    ),
                },
            )
            for index in range(MAX_REFERENCE_IMAGES)
        },
        "ref_max_size": (
            "INT",
            {
                "default": 848,
                "min": 16,
                "max": 8192,
                "step": 16,
                "tooltip": (
                    "Max size for the long edge of reference_video and reference_images. "
                    "Resized with preserved aspect ratio and snapped to 16px."
                ),
            },
        ),
    }


def _load_bernini_conditioning_cls():
    try:
        from comfy_extras.nodes_bernini import BerniniConditioning
    except ImportError as exc:
        raise RuntimeError(
            "ComfyBerniniDirector 需要 ComfyUI 官方 BerniniConditioning "
            "(comfy_extras.nodes_bernini)。请升级到已合入 Bernini 的 ComfyUI。"
        ) from exc
    return BerniniConditioning


def _reference_images_dict_from_kwargs(kwargs: dict) -> dict | None:
    """Build official-style {reference_image_N: IMAGE} for BerniniConditioning."""
    nested = kwargs.get("reference_images")
    if isinstance(nested, dict) and nested:
        out = {k: v for k, v in nested.items() if v is not None}
        return out or None

    refs = flatten_reference_kwargs(kwargs)
    out = {k: v for k, v in refs.items() if v is not None}
    return out or None


def _unpack_bernini_output(out):
    if hasattr(out, "args"):
        args = out.args
        if len(args) >= 3:
            return args[0], args[1], args[2]
    if isinstance(out, (tuple, list)) and len(out) >= 3:
        return out[0], out[1], out[2]
    raise RuntimeError(
        f"Official BerniniConditioning returned unexpected output: {type(out)!r}"
    )


def _task_hint(source_video, reference_video, reference_images) -> str:
    ref_image_count = 0
    if reference_images:
        for imgs in reference_images.values():
            if imgs is not None:
                ref_image_count += int(imgs.shape[0])
    mode = infer_task(
        source_video is not None,
        reference_video is not None,
        ref_image_count,
    )
    stream_guess = int(source_video is not None) + int(reference_video is not None) + ref_image_count
    hint = f"{mode.value} — {TASK_DESCRIPTIONS[mode]} (native BerniniConditioning)"
    if stream_guess:
        hint += f" (~{stream_guess} stream(s))"
    return hint


def _run_conditioning(
    positive,
    negative,
    vae,
    width,
    height,
    length,
    batch_size,
    source_video=None,
    reference_video=None,
    ref_max_size=848,
    **kwargs,
):
    """Delegate entirely to ComfyUI ``BerniniConditioning.execute``."""
    BerniniConditioning = _load_bernini_conditioning_cls()
    reference_images = _reference_images_dict_from_kwargs(kwargs)

    out = BerniniConditioning.execute(
        positive,
        negative,
        vae,
        width,
        height,
        length,
        batch_size,
        source_video=source_video,
        reference_video=reference_video,
        reference_images=reference_images,
        ref_max_size=ref_max_size,
    )
    positive, negative, latent = _unpack_bernini_output(out)
    return (
        positive,
        negative,
        latent,
        _task_hint(source_video, reference_video, reference_images),
    )


class BerniniDirectorConditioning:
    """Thin wrapper around ComfyUI official BerniniConditioning (3 outputs)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "width": ("INT", {"default": 832, "min": 16, "max": 8192, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": 8192, "step": 16}),
                "length": ("INT", {"default": 81, "min": 1, "max": 8192, "step": 4}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
            },
            "optional": _shared_optional_inputs(),
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "apply"
    CATEGORY = "conditioning/video_models"

    def apply(self, *args, **kwargs):
        positive, negative, latent, _ = _run_conditioning(*args, **kwargs)
        return positive, negative, latent


class BerniniDirectorPlannerConditioning:
    """Official BerniniConditioning plus a task_mode string for planning UIs."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "width": ("INT", {"default": 832, "min": 16, "max": 8192, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": 8192, "step": 16}),
                "length": ("INT", {"default": 81, "min": 1, "max": 8192, "step": 4}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
            },
            "optional": _shared_optional_inputs(),
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT", "STRING")
    RETURN_NAMES = ("positive", "negative", "latent", "task_mode")
    FUNCTION = "apply"
    CATEGORY = "Bernini"

    def apply(self, *args, **kwargs):
        return _run_conditioning(*args, **kwargs)
