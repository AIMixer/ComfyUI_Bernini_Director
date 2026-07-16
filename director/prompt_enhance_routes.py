"""HTTP routes for Bernini Director LLM prompt enhancement."""

from __future__ import annotations

import logging

from aiohttp import web

from ..lib.prompt_enhance_templates import (
    DETAILED_MIN_TOTAL_HAN,
    OUTPUT_LANGUAGE_EN,
    build_character_detail_directive,
    count_han_chars,
    get_enhance_template,
    is_character_feature_enhance_enabled,
)
from ..lib.prompt_enhancer import (
    API_FORMAT_OLLAMA,
    API_FORMAT_OPENAI_COMPAT,
    DEFAULT_API_FORMAT,
    DEFAULT_OPENAI_COMPAT_MODE,
    OPENAI_COMPAT_MODE_LLAMA_SWAP,
    coerce_llm_model,
    coerce_llm_url,
    default_url_for_format,
    enhance_prompt_sync,
    infer_api_format,
    list_llm_models,
    llama_swap_unload_endpoint,
    llm_headers,
    llm_unload_endpoint,
    normalize_openai_compat_mode,
)
from ..lib.task_prompts import resolve_task_key
from .prompt_enhance_media import extract_input_video_frames_b64, load_input_image_b64

log = logging.getLogger("ComfyUI-Bernini-Director.director")


async def director_enhance_models(request):
    try:
        data = await request.json()
    except Exception:
        data = {}
    raw_url = data.get("llm_url") or data.get("ollama_url")
    api_format = infer_api_format(raw_url or "", data.get("api_format") or data.get("llm_api_format") or DEFAULT_API_FORMAT)
    url = coerce_llm_url(raw_url, default=default_url_for_format(api_format))
    api_key = (data.get("api_key") or data.get("llm_api_key") or "").strip()
    models, err = await list_llm_models(url, api_format, api_key=api_key)
    if err:
        return web.json_response({"error": err}, status=502)
    return web.json_response({"models": models})


async def director_get_template(request):
    try:
        data = await request.json()
    except Exception:
        data = {}
    task = resolve_task_key(data.get("task_type") or "default")
    output_language = data.get("output_language") or data.get("llm_output_language") or OUTPUT_LANGUAGE_EN
    return web.json_response({
        "template": get_enhance_template(task, output_language=output_language),
        "task_type": task,
    })


async def director_enhance_prompt(request):
    try:
        data = await request.json()
    except Exception as exc:
        return web.json_response({"error": f"Invalid JSON: {exc}"}, status=400)

    model = coerce_llm_model((data.get("model") or data.get("llm_model") or "").strip())
    if not model:
        return web.json_response({"error": "No model selected"}, status=400)

    user_prompt = (data.get("prompt") or "").strip()
    if not user_prompt:
        return web.json_response({"error": "Empty prompt"}, status=400)

    task_type = data.get("task_type") or "default"
    raw_url = data.get("llm_url") or data.get("ollama_url")
    api_format = infer_api_format(raw_url or "", data.get("api_format") or data.get("llm_api_format") or DEFAULT_API_FORMAT)
    url = coerce_llm_url(raw_url, default=default_url_for_format(api_format))
    openai_compat_mode = normalize_openai_compat_mode(
        data.get("openai_compat_mode")
        or data.get("llm_openai_compat_mode")
        or DEFAULT_OPENAI_COMPAT_MODE
    )
    api_key = (data.get("api_key") or data.get("llm_api_key") or "").strip()
    images = data.get("images") or []
    image_num = int(data.get("image_num") or max(1, len(images)))
    ref_slots_raw = data.get("ref_slots") or data.get("ref_slot_indices") or []
    ref_slots = [int(s) for s in ref_slots_raw if s is not None and str(s).strip() != ""]
    source_count = data.get("source_count", data.get("vision_source_count"))
    if source_count is not None:
        source_count = int(source_count)
    ref_video_count = int(data.get("ref_video_count") or data.get("vision_ref_video_count") or 0)
    custom_template = (data.get("custom_template") or "").strip()
    output_language = data.get("output_language") or data.get("llm_output_language") or OUTPUT_LANGUAGE_EN
    character_feature_enhance = data.get("character_feature_enhance")
    if character_feature_enhance is None:
        character_feature_enhance = data.get("llm_character_feature_enhance")
    if character_feature_enhance is None:
        character_feature_enhance = False
    unload_after = bool(data.get("unload_ollama") or data.get("llm_unload_after"))

    try:
        text, err = enhance_prompt_sync(
            task_type=task_type,
            user_prompt=user_prompt,
            url=url,
            model=model,
            api_format=api_format,
            openai_compat_mode=openai_compat_mode,
            api_key=api_key,
            images_b64=images if images else None,
            image_num=image_num,
            custom_template=custom_template,
            output_language=output_language,
            character_feature_enhance=character_feature_enhance,
            vision_source_count=source_count,
            ref_slots=ref_slots or None,
            vision_ref_video_count=ref_video_count,
            unload_after=unload_after,
        )
    except Exception as exc:
        log.exception("Director enhance route failed")
        return web.json_response({"error": f"{type(exc).__name__}: {exc}"}, status=502)
    if err:
        return web.json_response({"error": err}, status=502)
    if not text:
        return web.json_response({"error": "Enhancement returned empty"}, status=502)
    task_key = resolve_task_key(task_type)
    feature_enhance = is_character_feature_enhance_enabled(character_feature_enhance)
    detail_directive = build_character_detail_directive(
        feature_enhance,
        output_language=output_language,
        task_key=task_key,
    )
    detailed_mode = feature_enhance and bool(detail_directive)
    han_count = count_han_chars(text)
    return web.json_response({
        "response": text,
        "han_count": han_count,
        "detailed_mode": detailed_mode,
        "character_feature_enhance": feature_enhance,
        "detail_target_han": DETAILED_MIN_TOTAL_HAN if detailed_mode else None,
    })


async def director_extract_frames(request):
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    filename = data.get("filename") or data.get("videoFile") or ""
    subfolder = data.get("subfolder") or ""
    num_frames = min(int(data.get("num_frames") or 3), 5)
    frames, err = extract_input_video_frames_b64(filename, subfolder=subfolder, num_frames=num_frames)
    if err:
        status = 404 if "not found" in err.lower() else 500
        return web.json_response({"error": err}, status=status)
    return web.json_response({"frames": frames})


async def director_image_b64(request):
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    filename = data.get("filename") or data.get("imageFile") or ""
    b64, err = load_input_image_b64(filename)
    if err:
        status = 404 if "not found" in err.lower() else 400
        return web.json_response({"error": err}, status=status)
    return web.json_response({"image": b64})


async def director_unload_model(request):
    import aiohttp

    try:
        data = await request.json()
    except Exception:
        data = {}
    raw_url = data.get("llm_url") or data.get("ollama_url")
    api_format = infer_api_format(raw_url or "", data.get("api_format") or data.get("llm_api_format") or API_FORMAT_OLLAMA)
    url = coerce_llm_url(raw_url, default=default_url_for_format(api_format))
    model = coerce_llm_model((data.get("model") or data.get("llm_model") or "").strip())
    if not model:
        return web.json_response({"error": "No model selected"}, status=400)
    openai_compat_mode = normalize_openai_compat_mode(
        data.get("openai_compat_mode")
        or data.get("llm_openai_compat_mode")
        or DEFAULT_OPENAI_COMPAT_MODE
    )
    api_key = (data.get("api_key") or data.get("llm_api_key") or "").strip()

    if api_format == API_FORMAT_OLLAMA:
        endpoint = llm_unload_endpoint(url)
        payload = {"model": model, "keep_alive": 0}
        headers = None
        success_label = "Ollama"
    elif api_format == API_FORMAT_OPENAI_COMPAT and openai_compat_mode == OPENAI_COMPAT_MODE_LLAMA_SWAP:
        endpoint = llama_swap_unload_endpoint(url, model)
        payload = None
        headers = llm_headers(api_format, include_json=False, api_key=api_key)
        success_label = "llama-swap"
    else:
        return web.json_response({"error": "Current API format does not support model unload"}, status=400)

    try:
        async with aiohttp.ClientSession() as session:
            kwargs = {
                "timeout": aiohttp.ClientTimeout(total=10),
            }
            if payload is not None:
                kwargs["json"] = payload
            if headers:
                kwargs["headers"] = headers
            async with session.post(endpoint, **kwargs) as resp:
                if not (200 <= resp.status < 300):
                    text = await resp.text()
                    return web.json_response({"error": f"{success_label} HTTP {resp.status}: {text[:200]}"}, status=502)
                await resp.read()
        return web.json_response({"status": "unloaded", "model": model, "provider": success_label})
    except Exception as exc:
        return web.json_response({"error": f"{type(exc).__name__}: {exc}"}, status=502)


async def director_unload_ollama(request):
    return await director_unload_model(request)


def register_prompt_enhance_routes(routes, register_route) -> None:
    register_route(routes, "POST", "/bernini/director/enhance_models", director_enhance_models)
    register_route(routes, "POST", "/bernini/director/get_template", director_get_template)
    register_route(routes, "POST", "/bernini/director/enhance", director_enhance_prompt)
    register_route(routes, "POST", "/bernini/director/extract_frames", director_extract_frames)
    register_route(routes, "POST", "/bernini/director/image_b64", director_image_b64)
    register_route(routes, "POST", "/bernini/director/unload_model", director_unload_model)
    register_route(routes, "POST", "/bernini/director/unload_ollama", director_unload_ollama)
    log.info("Bernini Director prompt-enhance HTTP routes registered")
