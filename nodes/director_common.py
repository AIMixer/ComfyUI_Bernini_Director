"""Shared helpers for the Bernini Director timeline node."""

from __future__ import annotations

import json
import logging

import torch

from ..director.audio_export import (
    build_director_audio_outputs,
    source_audio_report_note,
)
from ..director.frame_align import pad_or_trim_frames
from ..director.gen_timeline import is_prompt_batch_timeline, is_video_batch_task_key
from ..director.plan import build_director_plan, count_all_timeline_segments, count_timeline_segments, plan_summary
from ..director.progress import report_director_planning
from ..lib.image_prep import fit_canvas, fit_video_long_edge
from ..lib.video_io import load_timeline_segment
from ..lib.task_prompts import task_type_combo_options

log = logging.getLogger("ComfyUI-Bernini-Director")


def timeline_required_inputs() -> dict:
    """Timeline + prompt widgets shared by Director nodes."""
    combo_options, combo_meta = task_type_combo_options()
    return {
        "task_type": (combo_options, combo_meta),
        "global_prompt": (
            "STRING",
            {
                "default": "",
                "multiline": True,
                "tooltip": "Synced from in-node UI (global mode).",
            },
        ),
        "negative_prompt": (
            "STRING",
            {
                "default": "bad video",
                "multiline": True,
                "tooltip": "Synced from in-node UI — shared negative prompt for all segments.",
            },
        ),
        "bd_grp_high": ("BDGROUP", {"default": "高噪采样设置"}),
        "high_noise_cfg": (
            "FLOAT",
            {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.01, "tooltip": "CFG for high-noise sampler pass."},
        ),
        "high_noise_seed": (
            "INT",
            {
                "default": 0,
                "min": 0,
                "max": 0xFFFFFFFFFFFFFFFF,
                "control_after_generate": True,
                "tooltip": "Seed for high-noise sampler pass.",
            },
        ),
        "bd_grp_low": ("BDGROUP", {"default": "低噪采样设置"}),
        "low_noise_cfg": (
            "FLOAT",
            {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.01, "tooltip": "CFG for low-noise sampler pass."},
        ),
        "low_noise_seed": (
            "INT",
            {
                "default": 0,
                "min": 0,
                "max": 0xFFFFFFFFFFFFFFFF,
                "control_after_generate": True,
                "tooltip": "Seed for low-noise sampler pass.",
            },
        ),
        "frame_rate": (
            "FLOAT",
            {"default": 24.0, "min": 1.0, "max": 240.0, "step": 0.01, "tooltip": "Timeline / output FPS."},
        ),
        "width": ("INT", {"default": 832, "min": 16, "max": 8192, "step": 16}),
        "height": ("INT", {"default": 480, "min": 16, "max": 8192, "step": 16}),
        "ref_max_size": ("INT", {"default": 848, "min": 16, "max": 8192, "step": 16}),
        "total_frames": (
            "INT",
            {"default": 81, "min": 1, "max": 8192, "tooltip": "Synced from uploaded video / timeline UI."},
        ),
        "timeline_data": (
            "STRING",
            {"default": "", "multiline": True, "tooltip": "Internal — video, segments, refs (populated by UI)."},
        ),
    }


def director_llm_enhance_inputs() -> dict:
    """LLM prompt enhancement widgets (Bernini official templates + LLM APIs)."""
    return {
        "bd_grp_pe": ("BDGROUP", {"default": "提示词增强 LLM Prompt Enhancer"}),
        "llm_auto_enhance": (
            "BOOLEAN",
            {
                "default": False,
                "tooltip": (
                    "每次 Queue 时自动用 LLM 扩写当前片段/全局正向提示词（使用 Bernini 官方 task 模板）。"
                    "Auto-enhance 会在服务端附带源视频帧与参考图（若已上传）。"
                ),
            },
        ),
        "llm_api_format": (
            ["Ollama", "智谱 GLM", "OpenAI Compatible"],
            {
                "default": "Ollama",
                "tooltip": (
                    "Ollama 原生：/api/chat；"
                    "智谱 GLM：https://open.bigmodel.cn/api/paas/v4/chat/completions；"
                    "OpenAI Compatible：/v1/chat/completions（llama.cpp / vLLM / llama-swap 等）。"
                ),
            },
        ),
        "llm_openai_compat_mode": (
            ["标准", "llama-swap"],
            {
                "default": "标准",
                "tooltip": (
                    "仅在 OpenAI Compatible 格式下生效。选择 llama-swap 后可使用其模型卸载接口："
                    "POST /api/models/unload/{model_id}。"
                ),
            },
        ),
        "llm_url": (
            "STRING",
            {
                "default": "http://127.0.0.1:11434/v1",
                "tooltip": (
                    "LLM 服务地址。Ollama：http://127.0.0.1:11434/v1；"
                    "智谱：https://open.bigmodel.cn/api/paas/v4；"
                    "OpenAI Compatible：http://127.0.0.1:8080/v1。"
                ),
            },
        ),
        "llm_api_key": (
            "STRING",
            {
                "default": "",
                "tooltip": "智谱 API Key（Bearer）。OpenAI Compatible / llama-swap 可选。也可设环境变量 ZHIPU_API_KEY / BERNINI_PE_API_KEY。",
            },
        ),
        "llm_model": (
            "STRING",
            {
                "default": "qwen3.5",
                "tooltip": (
                    "模型名称。Ollama 默认 qwen3.5。"
                    "智谱默认 glm-4.6v-flash（支持参考图/视频帧 Base64）；"
                    "纯文本可选用 glm-4-flash-250414；"
                    "OpenAI Compatible 填 /v1/models 返回的模型 id。"
                ),
            },
        ),
        "llm_output_language": (
            ["English", "中文"],
            {
                "default": "中文",
                "tooltip": (
                    "LLM 扩写结果的输出语言。官方 Bernini 示例与 T5 系统提示词为英文；"
                    "选「中文」时扩写内容为简体中文，送入模型时仍会前置英文 task 系统提示词。"
                ),
            },
        ),
        "llm_character_feature_enhance": (
            "BOOLEAN",
            {
                "default": False,
                "tooltip": (
                    "角色特征增强（rv2v/r2v/r2i 等含参考图任务，默认关闭）。"
                    "未勾选时按 Bernini 官方提示词工程扩写；"
                    "勾选后追加详尽角色外观指令（总汉字≥300，充分描述参考图人物特征）。"
                ),
            },
        ),
        "llm_unload_after": (
            "BOOLEAN",
            {
                "default": False,
                "tooltip": "增强完成后立即卸载 Ollama 模型（keep_alive=0，仅 Ollama）。",
            },
        ),
        "llm_custom_template": (
            "STRING",
            {
                "default": "",
                "multiline": True,
                "tooltip": "可选：覆盖当前 task_type 的 LLM 扩写模板（留空则用官方模板）。",
            },
        ),
    }


def director_perf_inputs() -> dict:
    """Performance widgets shared by Bernini Director nodes."""
    return {
        "bd_grp_perf": ("BDGROUP", {"default": "性能 Performance"}),
        "clear_vram_between_segments": (
            "BOOLEAN",
            {
                "default": True,
                "tooltip": (
                    "段间清理显存：每段结束后卸载已加载模型并清空 CUDA 缓存，"
                    "降低多段峰值显存（段间略慢），从而降低爆显存风险"
                ),
            },
        ),
        "export_source_images": (
            "BOOLEAN",
            {
                "default": False,
                "tooltip": (
                    "输出 source_images（时间轴原片帧，用于与 images 并排对比）。"
                    "默认关：跳过二次解码，节省时间与内存；需要对比预览时再开。"
                ),
            },
        ),
    }


def default_timeline_json(
    *,
    task_type: str,
    global_prompt: str,
    total_frames: int,
    frame_rate: float,
    width: int,
    height: int,
    ref_max_size: int,
) -> str:
    return json.dumps(
        {
            "version": 4,
            "editMode": "global",
            "totalFrames": total_frames,
            "frameRate": frame_rate,
            "width": width,
            "height": height,
            "refMaxSize": ref_max_size,
            "output": {
                "mode": "long_edge",
                "longEdge": ref_max_size,
                "width": width,
                "height": height,
                "maxExportFrames": 0,
                "exportMode": "all",
            },
            "videoClips": [],
            "video": {
                "fileName": "",
                "videoFile": "",
                "subfolder": "",
                "type": "input",
                "frames": [],
                "frameMap": [],
            },
            "global": {"taskType": task_type, "prompt": global_prompt, "refs": [], "referenceVideo": {}, "continuousReference": False},
            "segments": [
                {
                    "id": "s0",
                    "start": 0,
                    "length": total_frames,
                    "prompt": "",
                    "taskType": "",
                    "refs": [],
                    "referenceVideo": {},
                }
            ],
        },
        ensure_ascii=False,
    )


def prepare_director_plan(
    *,
    timeline_data: str,
    task_type: str,
    global_prompt: str,
    total_frames: int,
    frame_rate: float,
    width: int,
    height: int,
    ref_max_size: int,
    unique_id: str | None,
):
    if not timeline_data or not timeline_data.strip():
        timeline_data = default_timeline_json(
            task_type=task_type,
            global_prompt=global_prompt,
            total_frames=total_frames,
            frame_rate=frame_rate,
            width=width,
            height=height,
            ref_max_size=ref_max_size,
        )

    report_director_planning(
        unique_id,
        count_timeline_segments(timeline_data),
        timeline_segment_total=count_all_timeline_segments(timeline_data),
    )

    plan = build_director_plan(
        timeline_data,
        global_task_type=task_type,
        global_prompt=global_prompt,
        total_frames=total_frames,
        frame_rate=frame_rate,
        width=width,
        height=height,
        ref_max_size=ref_max_size,
    )
    log.info(plan_summary(plan).replace("\n", " | "))
    return plan


def _fit_source_clip_to_plan(plan, raw_clip: torch.Tensor) -> torch.Tensor:
    """Match Director output dimensions for side-by-side preview."""
    if plan.output_mode == "fixed":
        return fit_canvas(raw_clip, plan.width, plan.height)
    return fit_video_long_edge(raw_clip, plan.ref_max_size)


def build_source_images_output(
    plan,
    images_out: list[torch.Tensor],
    *,
    split_outputs: bool,
) -> list[torch.Tensor]:
    """Expose source frames from the Director timeline, aligned with generated image outputs."""
    if split_outputs:
        chunks: list[torch.Tensor] = []
        for seg, generated in zip(plan.segments, images_out):
            target_len = int(generated.shape[0])
            raw = load_timeline_segment(plan.raw, seg.start_frame, seg.end_frame)
            fitted = _fit_source_clip_to_plan(plan, raw)
            chunks.append(pad_or_trim_frames(fitted, target_len).cpu().float())
        return chunks

    target_len = int(images_out[0].shape[0]) if images_out else int(plan.total_frames or 0)
    raw = load_timeline_segment(plan.raw, 0, target_len)
    fitted = _fit_source_clip_to_plan(plan, raw)
    return [pad_or_trim_frames(fitted, target_len).cpu().float()]


def _empty_source_images_for(images_out: list[torch.Tensor]) -> list[torch.Tensor]:
    """Neutral 1-frame placeholders when source export is off (OUTPUT_IS_LIST + downstream nodes need batch>=1)."""
    if not images_out:
        return [torch.full((1, 1, 1, 3), 0.5)]
    placeholders: list[torch.Tensor] = []
    for img in images_out:
        if isinstance(img, torch.Tensor) and img.ndim == 4:
            h, w, c = int(img.shape[1]), int(img.shape[2]), int(img.shape[3])
        else:
            h, w, c = 1, 1, 3
        placeholders.append(torch.full((1, h, w, c), 0.5))
    return placeholders


def _ensure_nonempty_image_batches(images_out: list[torch.Tensor], *, label: str) -> list[torch.Tensor]:
    """Drop or pad zero-frame tensors so IMAGE list outputs never break downstream nodes."""
    fixed: list[torch.Tensor] = []
    for i, img in enumerate(images_out):
        if not isinstance(img, torch.Tensor) or img.ndim != 4:
            raise ValueError(f"Director {label}[{i}] is not a valid IMAGE tensor.")
        if int(img.shape[0]) <= 0:
            h, w, c = int(img.shape[1]), int(img.shape[2]), int(img.shape[3])
            log.warning("Director %s[%d] has 0 frames; emitting 1-frame placeholder.", label, i)
            fixed.append(torch.full((1, max(1, h), max(1, w), max(1, c)), 0.5))
        else:
            fixed.append(img)
    return fixed


def finalize_director_outputs(
    plan,
    combined,
    segment_outputs,
    report,
    *,
    export_source_images: bool = False,
):
    is_batch = is_prompt_batch_timeline(plan.raw, plan.global_task_key)
    export_segments = plan.export_mode == "segments"
    video_batch = is_video_batch_task_key(plan.global_task_key)

    if export_segments or (is_batch and not video_batch):
        images_out = segment_outputs
        frame_count = sum(int(s.shape[0]) for s in segment_outputs)
        if export_segments and len(segment_outputs) > 1:
            report = (
                report
                + f"\n\nExport mode: segments — {len(segment_outputs)} clip(s) on images output "
                "(one MP4 per segment when connected to Video Combine / PreviewImage)."
            )
        if plan.run_indices is not None:
            report = (
                report
                + f"\n\nPartial run: output contains {len(segment_outputs)} re-generated "
                f"{'group(s)' if is_batch else 'segment clip(s)'} only."
            )
    else:
        combined = pad_or_trim_frames(combined, plan.total_frames).cpu().float()
        images_out = [combined]
        frame_count = int(combined.shape[0])
        if video_batch and is_batch and len(segment_outputs) > 1:
            report = (
                report
                + f"\n\nExport mode: all — merged {frame_count} frame(s) on images output "
                "(single clip when connected to Video Combine / PreviewImage)."
            )
        if plan.run_indices is not None and video_batch:
            report = (
                report
                + f"\n\nPartial run: re-generated {len(segment_outputs)} video group(s); "
                "skipped groups merged from cache or source when available."
            )

    split_for_audio = export_segments or (is_batch and not video_batch)
    audio_frame_end = frame_count if not split_for_audio else None
    audio_out = build_director_audio_outputs(
        plan,
        images_out,
        export_segments=split_for_audio,
        output_frame_end=audio_frame_end,
    )
    report = report + source_audio_report_note(
        plan,
        audio_out,
        export_segments=split_for_audio,
        output_frame_end=audio_frame_end,
    )

    split_source_outputs = export_segments or (is_batch and not video_batch)
    if export_source_images:
        try:
            source_images_out = build_source_images_output(
                plan,
                images_out,
                split_outputs=split_source_outputs,
            )
        except Exception as exc:
            log.warning("Source images output failed: %s", exc)
            source_images_out = images_out
            report = report + f"\n\nSource images: fallback to generated output ({exc})."
    else:
        source_images_out = _empty_source_images_for(images_out)

    images_out = _ensure_nonempty_image_batches(images_out, label="images")
    source_images_out = _ensure_nonempty_image_batches(source_images_out, label="source_images")

    report = (
        report
        + "\n\n有问题联系作者：AI搅拌手  QQ交流群：551482703"
    )

    fps_out = float(plan.frame_rate or 24.0)
    return images_out, audio_out, fps_out, frame_count, source_images_out, report
