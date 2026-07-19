import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import {
    defaultFrameCount,
    genLayoutHint,
    getDirectorMode,
    imageBatchRequiresFixedOutput,
    isPromptBatchTask,
    isVideoBatchTask,
    MAX_GEN_FRAMES,
    minFrameCount,
    newBatchSegment,
    resolveTaskKey,
    sumFrameCounts,
    taskUsesReferenceImages,
    taskUsesReferenceVideo,
} from "./bernini_gen_timeline.js";
import {
    IMAGE_BATCH_STYLES,
    bindImageBatchEvents,
    ensureImageBatchTimeline,
    getImageBatchUiHeight,
    mountImageBatchPanel,
    normalizeImageBatchSegments,
    renderImageBatchGroups,
    setImageBatchPreview,
    setToolbarDisabledForBatch,
    wireBatchRunSelectControls,
} from "./bernini_image_batch.js";
import {
    getPromptEnhancerPanelHeight,
    mountPromptEnhancerPanel,
    registerDirectorPromptEnhancerEvents,
} from "./bernini_prompt_enhancer.js";
import { mountPromptImageMentions } from "./bernini_prompt_mentions.js";

const RULER_H = 24;
const SEG_LABEL_H = 20;
const TRACK_H = 160;
const TRACK_Y = RULER_H + SEG_LABEL_H;
const STAGE_PREVIEW_H = 220;
const MIN_SEG = 4;
const HANDLE_PX = 14;
/** Canvas-drawn run-select checkbox (not a DOM control). */
const RUN_CHECK_SIZE = 14;
const RUN_CHECK_HIT_PAD_X = 8;
const RUN_CHECK_HIT_PAD_Y = 4;
const THUMB_MAX_W = 168;
const THUMB_JPEG_Q = 0.55;
const TIMELINE_SYNC_DEBOUNCE_MS = 500;
const MAX_THUMBS_PER_SEGMENT = 20;
const THUMB_PREFETCH_BATCH = 6;
const DIRECTOR_MIN_WIDTH = 900;
const COMFY_UPLOAD_SOFT_LIMIT = 95 * 1024 * 1024;
const BERNINI_CHUNK_SIZE = 8 * 1024 * 1024;

/** Segment continuity is opt-in; default off unless explicitly true in output. */
function isContinuityEnabled(output) {
    if (!output) return false;
    return output.continuityEnabled === true || output.continuity_enabled === true;
}

function normalizeOutputContinuity(output = {}) {
    const rawOverlap = output.continuityOverlapFrames ?? output.continuity_overlap_frames ?? 9;
    return {
        ...output,
        continuityEnabled: isContinuityEnabled(output),
        continuityOverlapFrames: Math.max(1, Math.min(81, parseInt(rawOverlap, 10) || 9)),
    };
}

function stripTimelineContinuityRootFields(timeline) {
    if (!timeline || typeof timeline !== "object") return;
    delete timeline.continuityEnabled;
    delete timeline.continuity_enabled;
    delete timeline.continuityOverlapFrames;
    delete timeline.continuity_overlap_frames;
}

/** Drop ephemeral UI-only fields so they never persist in timeline_data. */
function stripTimelineEphemeralFields(timeline) {
    if (!timeline || typeof timeline !== "object") return;
    delete timeline.videoWorkspace;
}

const HIDDEN_WIDGETS = [
    "timeline_data", "total_frames", "width", "height", "ref_max_size",
    "task_type", "global_prompt", "frame_rate", "negative_prompt",
    "bd_grp_pe", "llm_auto_enhance", "llm_api_format", "llm_openai_compat_mode", "llm_url", "llm_api_key", "llm_model",
    "llm_unload_after", "llm_output_language", "llm_character_feature_enhance", "llm_custom_template",
];

const DIRECTOR_WIDGET_LABELS = {
    clear_vram_between_segments: "段间清理显存",
    export_source_images: "输出原片对比 source_images",
};

function applyDirectorWidgetLabels(node) {
    for (const w of node.widgets || []) {
        const label = DIRECTOR_WIDGET_LABELS[w.name];
        if (label) w.label = label;
    }
}

function drawGroupHeader(ctx, node, widget_width, y, H, label) {
    const margin = 10;
    const barH = Math.max(18, H - 4);
    ctx.fillStyle = "#2e2e2e";
    ctx.strokeStyle = "#555";
    ctx.lineWidth = 1;
    ctx.beginPath();
    if (ctx.roundRect) {
        ctx.roundRect(margin, y + 2, widget_width - margin * 2, barH, 4);
    } else {
        ctx.rect(margin, y + 2, widget_width - margin * 2, barH);
    }
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = "#d8dce8";
    ctx.font = "600 11px ui-sans-serif, system-ui, sans-serif";
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    ctx.fillText(label, margin + 10, y + 2 + barH / 2);
}

function makeGroupHeaderWidget(inputName, inputData) {
    const opts = inputData?.[1] || {};
    const label = opts.default || opts.label || inputName;
    const el = document.createElement("div");
    el.className = "bd-widget-group";
    el.textContent = label;
    el.style.cssText = [
        "width:100%;box-sizing:border-box;margin:8px 0 4px;padding:6px 10px",
        "border:1px solid #555;border-left:3px solid #7a9cff;border-radius:4px",
        "color:#d8dce8;font-size:11px;font-weight:600;letter-spacing:.02em",
        "background:linear-gradient(180deg,#2e2e2e 0%,#242424 100%)",
        "pointer-events:none;user-select:none",
    ].join(";");
    return {
        name: inputName,
        type: "BDGROUP",
        value: label,
        label: "",
        element: el,
        options: opts,
        _bdGroupHeader: true,
        draw(ctx, node, widget_width, y, H) {
            drawGroupHeader(ctx, node, widget_width, y, H, label);
        },
        computeSize(width) {
            return [width, 26];
        },
        mouse() {
            return false;
        },
    };
}

const STYLES = `
.bd-host{width:100%;box-sizing:border-box;display:block}
.bd-wrap{font-family:ui-sans-serif,system-ui,-apple-system,sans-serif;color:#e0e0e0;font-size:11px;display:flex;flex-direction:column;gap:6px;width:100%;box-sizing:border-box;position:relative;min-height:var(--comfy-widget-min-height,0px)}
.bd-main{flex:1 1 auto;min-height:0;display:flex;flex-direction:column;gap:6px;width:100%}
.bd-modal-overlay{position:absolute;inset:0;z-index:200;background:rgba(0,0,0,.72);display:flex;align-items:center;justify-content:center;padding:10px;box-sizing:border-box;border-radius:6px}
.bd-modal{background:#1e1e1e;border:1px solid #333;border-radius:6px;padding:12px;width:100%;max-width:460px;max-height:calc(100% - 8px);display:flex;flex-direction:column;gap:10px;box-shadow:0 10px 28px rgba(0,0,0,.5)}
.bd-modal-title{color:#e0e0e0;font-size:12px;font-weight:600;line-height:1.35}
.bd-modal-body{color:#aaa;font-size:11px;line-height:1.5;white-space:pre-wrap}
.bd-modal-body.hidden{display:none}
.bd-modal-list{flex:1;min-height:140px;max-height:240px;overflow:auto;background:#181818;border:1px solid #333;border-radius:6px;padding:4px;display:flex;flex-direction:column;gap:2px}
.bd-modal-list.hidden{display:none}
.bd-modal-item{padding:7px 8px;border-radius:4px;cursor:pointer;color:#ccc;font-size:11px;line-height:1.35;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;border:1px solid transparent}
.bd-modal-item:hover{background:#252525;color:#eee}
.bd-modal-item.selected{background:#2a2a2a;border-color:#4fff8f;color:#fff}
.bd-modal-actions{display:flex;gap:8px;justify-content:flex-end;flex-shrink:0}
.bd-toolbar{display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:6px;width:100%}
.bd-actions{display:flex;gap:6px;flex-wrap:wrap;align-items:center;flex:1;min-width:0}
.bd-stage{width:100%;box-sizing:border-box;background:#0c0c0c;border:1px solid #222;border-bottom:none;border-radius:6px 6px 0 0;overflow:hidden;position:relative;min-height:120px;max-height:280px;aspect-ratio:16/9;display:flex;align-items:center;justify-content:center}
.bd-stage.hidden{display:none!important}
.bd-stage-video,.bd-stage-img{width:100%;height:100%;max-height:280px;object-fit:contain;background:#000;display:block}
.bd-stage-img.hidden,.bd-stage-video.hidden{display:none!important}
.bd-stage-empty{color:#555;font-size:11px;pointer-events:none}
.bd-stage-badge{position:absolute;left:8px;bottom:8px;padding:2px 7px;border-radius:3px;background:rgba(0,0,0,.65);color:#ccc;font-size:10px;line-height:1.4;cursor:pointer;user-select:none}
.bd-stage-badge:hover{color:#fff;background:rgba(0,0,0,.8)}
.bd-frame-jump{display:inline-flex;align-items:center;gap:4px;color:#ddd;font-size:11px;white-space:nowrap;font-variant-numeric:tabular-nums}
.bd-frame-jump .bd-frame-input{width:64px;background:#181818;border:1px solid #444;border-radius:4px;color:#eee;padding:4px 4px;font-size:11px;text-align:center;-moz-appearance:textfield}
.bd-frame-jump .bd-frame-input:focus{border-color:#4fff8f;outline:none}
.bd-frame-jump .bd-frame-input::-webkit-outer-spin-button,.bd-frame-jump .bd-frame-input::-webkit-inner-spin-button{-webkit-appearance:none;margin:0}
.bd-frame-jump .bd-frame-total{color:#888;min-width:2.5em}
.bd-controls{width:100%;box-sizing:border-box;background:#151515;border:1px solid #222;border-radius:0 0 6px 6px;padding:8px 10px;margin-top:0}
.bd-stage.hidden+.bd-controls{border-radius:6px;border-color:#333;background:#1e1e1e}
.bd-viewport{width:100%;min-width:100%;overflow-x:auto;border-radius:6px;border:1px solid #111;background:#2a2a2a;box-sizing:border-box}
.bd-canvas{display:block;width:100%;min-width:100%;cursor:pointer;box-sizing:border-box}
.bd-canvas.bd-grab{cursor:grab}
.bd-canvas.bd-grabbing{cursor:grabbing}
.bd-output{width:100%;box-sizing:border-box;display:flex;align-items:center;gap:6px;flex-wrap:wrap;padding:6px 8px;background:#1e1e1e;border:1px solid #333;border-radius:6px}
.bd-split{display:block;width:100%;box-sizing:border-box}
.bd-player{display:flex;align-items:center;gap:10px;flex-wrap:wrap;width:100%}
.bd-btn{background:#222;color:#e0e0e0;border:1px solid #111;border-radius:4px;padding:6px 12px;font-size:11px;cursor:pointer}
.bd-btn:hover{background:#333;border-color:#555}
.bd-btn-danger:hover{background:#4a1515;border-color:#c44;color:#faa}
.bd-btn-sm{padding:3px 8px;font-size:10px}
.bd-btn-run-select.active{background:#1a3a2a;color:#4fff8f;border-color:#4fff8f}
.bd-run-select-bar{display:flex;align-items:center;gap:6px;flex-wrap:wrap;font-size:10px;color:#aaa}
.bd-run-select-all-wrap{display:inline-flex;align-items:center;gap:4px;font-size:11px;color:#aaa;cursor:pointer;user-select:none;margin-left:2px}
.bd-run-select-all-wrap.hidden{display:none!important}
.bd-run-select-all-wrap input{width:14px;height:14px;margin:0;cursor:pointer;accent-color:#4fff8f}
.bd-run-select-bar.hidden{display:none!important}
.bd-batch-run-check{margin-right:6px;width:14px;height:14px;cursor:pointer;accent-color:#4fff8f;flex-shrink:0}
.bd-btn-primary{background:#1a3a2a;border-color:#4fff8f;color:#4fff8f}
.bd-mode{display:flex;border:1px solid #333;border-radius:4px;overflow:hidden}
.bd-mode button{border:none;background:#222;color:#aaa;padding:6px 12px;font-size:11px;cursor:pointer}
.bd-mode button.active{background:#333;color:#fff}
.bd-right{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.bd-bounds,.bd-timecode{color:#aaa;font-size:11px}
.bd-timecode{color:#fff;font-weight:600;font-variant-numeric:tabular-nums;white-space:nowrap}
.bd-player .bd-timecode{min-width:88px;font-size:11px;color:#ddd}
.bd-icon-btn{background:#2a2a2a;border:1px solid #444;color:#eee;cursor:pointer;padding:6px 10px;border-radius:4px}
.bd-icon-btn.active{background:#1a3a2a;color:#4fff8f;border-color:#4fff8f;box-shadow:0 0 0 1px rgba(79,255,143,.35)}
.bd-seek{flex:1;min-width:120px;height:6px}
.bd-panel{width:100%;box-sizing:border-box;background:#222;border:1px solid #111;border-radius:6px;padding:8px;display:flex;flex-direction:column;gap:6px}
.bd-pe-host{width:100%;box-sizing:border-box;margin-top:8px;flex-shrink:0}
.bd-prompt-layout{display:grid;grid-template-columns:minmax(0,1fr) minmax(110px,38%);gap:8px;align-items:stretch}
.bd-prompt-col{display:flex;flex-direction:column;gap:5px;min-width:0}
.bd-prompt-col .bd-label,.bd-refs-col .bd-label{color:#888;font-size:10px;line-height:1.2;flex-shrink:0}
.bd-prompt{width:100%;min-height:48px;background:#181818;border:1px solid #333;border-radius:6px;color:#eee;padding:8px;resize:vertical;font-size:12px;box-sizing:border-box;font-family:inherit;line-height:1.35}
.bd-prompt-negative{min-height:40px;flex:0 0 auto}
.bd-refs-col{display:flex;flex-direction:column;gap:4px;min-width:0;height:100%}
.bd-refs{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:4px;width:100%;flex:1;align-content:start}
.bd-ref{position:relative;width:100%;aspect-ratio:1;min-width:0;max-height:76px;border:1px dashed #555;border-radius:4px;background:#111;display:flex;align-items:center;justify-content:center;cursor:pointer;overflow:hidden;font-size:9px;color:#666;transition:border-color .15s,background .15s}
.bd-ref:hover{border-color:#7a9cff;background:#1a1a1a}
.bd-ref .bd-ref-tag{position:absolute;inset:auto 0 3px 0;text-align:center;font-size:9px;color:#777;pointer-events:none;line-height:1}
.bd-ref.has-img .bd-ref-tag{display:none}
.bd-select{background:#181818;border:1px solid #333;border-radius:4px;color:#eee;padding:4px 6px;font-size:11px;max-width:240px}
.bd-ref img{width:100%;height:100%;object-fit:cover}
.bd-ref .x{position:absolute;top:1px;right:3px;color:#f88;font-size:12px;line-height:1;display:none}
.bd-ref:hover .x{display:block}
.bd-row{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.bd-meta{color:#888;font-size:10px}
.bd-video-tag{color:#4fff8f;font-size:10px}
.bd-num{width:42px;background:#181818;border:1px solid #333;border-radius:4px;color:#eee;padding:5px 4px;font-size:11px;text-align:center;-moz-appearance:textfield}
.bd-num::-webkit-outer-spin-button,.bd-num::-webkit-inner-spin-button{-webkit-appearance:none;margin:0}
.bd-output label{color:#888;font-size:10px;white-space:nowrap}
.bd-output .bd-out-fixed{display:flex;gap:4px;align-items:center}
.bd-output .bd-out-fixed.hidden{display:none}
.bd-run-status{width:100%;box-sizing:border-box;padding:8px 10px;background:#151515;border:1px solid #333;border-radius:6px;display:flex;flex-direction:column;gap:5px;margin-top:auto;flex-shrink:0}
.bd-run-status.idle .bd-run-title{color:#888}
.bd-run-status.active .bd-run-title{color:#4fff8f}
.bd-run-status.done .bd-run-title{color:#7a9cff}
.bd-run-status.error .bd-run-title{color:#f88}
.bd-run-title{font-size:11px;font-weight:600;line-height:1.35}
.bd-run-detail{color:#999;font-size:10px;line-height:1.4}
.bd-run-bars{display:flex;flex-direction:column;gap:3px}
.bd-run-bar{height:5px;background:#2a2a2a;border-radius:3px;overflow:hidden}
.bd-run-bar-fill{height:100%;background:linear-gradient(90deg,#2a6b4a,#4fff8f);border-radius:3px;transition:width .15s ease}
.bd-run-bar-sub .bd-run-bar-fill{background:linear-gradient(90deg,#3a5080,#7a9cff)}
.hidden{display:none!important}
.bd-controls.hidden{display:none!important}
.bd-gen-src{width:100%;min-height:72px;max-height:100px;border:1px dashed #555;border-radius:4px;background:#111;display:flex;align-items:center;justify-content:center;cursor:pointer;overflow:hidden;color:#666;font-size:10px;margin-top:4px;position:relative;box-sizing:border-box}
.bd-gen-src.has-img{border-style:solid;border-color:#444}
.bd-gen-src img{width:100%;height:100%;object-fit:contain;background:#000}
.bd-gen-src .x{position:absolute;top:1px;right:3px;color:#f88;font-size:12px;line-height:1;display:none;cursor:pointer;z-index:2}
.bd-gen-src.has-img:hover .x{display:block}
.bd-gen-src.has-video{padding:0;cursor:default;align-items:stretch;justify-content:flex-start;flex-direction:column}
.bd-gen-src.has-video .bd-ref-video-preview{width:100%;flex:1;min-height:100px;max-height:220px;object-fit:contain;background:#000;display:block;border-radius:3px}
.bd-gen-src .bd-ref-replace{position:absolute;bottom:4px;left:4px;z-index:3;background:rgba(0,0,0,.72);color:#ccc;border:1px solid #555;border-radius:3px;padding:2px 7px;font-size:9px;cursor:pointer;line-height:1.4}
.bd-gen-src .bd-ref-replace:hover{color:#fff;border-color:#888}
.bd-gen-src.has-video .x{display:block;z-index:3}
.bd-ref-video-col{display:flex;flex-direction:column;gap:4px;min-width:0;width:100%;flex:1}
.bd-ref-video-col .bd-gen-src{min-height:140px;max-height:none;flex:1}
.bd-ref-video-name{word-break:break-all;line-height:1.3}
.bd-continuous-ref{display:flex;align-items:center;gap:6px;font-size:10px;color:#aaa;user-select:none;margin-left:8px}
.bd-continuous-ref label{display:flex;align-items:center;gap:4px;cursor:pointer}
.bd-continuous-ref input[type="checkbox"]{width:14px;height:14px;margin:0;cursor:pointer;accent-color:#4fff8f}
.bd-gen-fc-row{display:flex;align-items:center;gap:6px;margin-top:6px}
${IMAGE_BATCH_STYLES}
@media(max-width:480px){
.bd-prompt-layout{grid-template-columns:1fr}
.bd-ref{max-height:64px}
}
`;

function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
function uid() { return Date.now().toString(36) + Math.random().toString(36).slice(2, 7); }

function snapDim(v, stride = 16) {
    return Math.max(stride, Math.round(v / stride) * stride);
}

function resolveOutputDimensions(sourceW, sourceH, output, fallback = {}) {
    const mode = String(output?.mode || "long_edge").toLowerCase();
    const stride = 16;
    if (mode === "fixed") {
        const w = snapDim(+(output?.width ?? fallback.width ?? 832), stride);
        const h = snapDim(+(output?.height ?? fallback.height ?? 480), stride);
        return { mode: "fixed", width: w, height: h, refMaxSize: Math.max(w, h) };
    }
    const longEdge = Math.max(stride, +(output?.longEdge ?? output?.long_edge ?? fallback.refMaxSize ?? 848));
    const sw = sourceW || 0;
    const sh = sourceH || 0;
    if (!sw || !sh) {
        const w = snapDim(+(fallback.width ?? 832), stride);
        const h = snapDim(+(fallback.height ?? 480), stride);
        return { mode: "long_edge", width: w, height: h, refMaxSize: longEdge };
    }
    if (Math.max(sw, sh) <= longEdge) {
        return { mode: "long_edge", width: snapDim(sw, stride), height: snapDim(sh, stride), refMaxSize: longEdge };
    }
    const scale = longEdge / Math.max(sw, sh);
    return {
        mode: "long_edge",
        width: snapDim(Math.round(sw * scale), stride),
        height: snapDim(Math.round(sh * scale), stride),
        refMaxSize: longEdge,
    };
}

/** Upload a file to ComfyUI input/ (videos use the same endpoint as images). */
function isUploadSizeError(err) {
    const msg = String(err?.message || err);
    return /body size|413|max_upload|too large|104857600/i.test(msg);
}

function formatUploadError(err) {
    const msg = String(err?.message || err);
    if (isUploadSizeError(err)) {
        return "文件超过 ComfyUI 默认上传限制（100MB）。已尝试分块上传；若仍失败，请手动复制视频到 ComfyUI/input/ 后刷新，或启动时加参数 --max-upload-size 2048";
    }
    return msg;
}

function formatProbeFps(value) {
    const fps = Math.round(Number(value) * 100) / 100;
    if (Number.isInteger(fps)) return String(fps);
    return fps.toFixed(2).replace(/0+$/, "").replace(/\.$/, "");
}

function coerceTimelineFps(value, fallback = 24) {
    const fps = Number(value);
    if (!Number.isFinite(fps) || fps <= 0) return coerceTimelineFps(fallback, 24);
    return Math.round(clamp(fps, 1, 240) * 100) / 100;
}

async function uploadToInput(file) {
    const body = new FormData();
    body.append("image", file);
    body.append("type", "input");
    body.append("overwrite", "true");
    const resp = await api.fetchApi("/upload/image", { method: "POST", body });
    if (!resp.ok) {
        const text = await resp.text();
        throw new Error(text || `Upload failed (${resp.status})`);
    }
    return resp.json();
}

async function uploadVideoChunked(file, onProgress) {
    const uploadId = crypto.randomUUID();
    const totalChunks = Math.ceil(file.size / BERNINI_CHUNK_SIZE);
    for (let i = 0; i < totalChunks; i++) {
        const start = i * BERNINI_CHUNK_SIZE;
        const end = Math.min(start + BERNINI_CHUNK_SIZE, file.size);
        const body = new FormData();
        body.append("upload_id", uploadId);
        body.append("chunk_index", String(i));
        body.append("total_chunks", String(totalChunks));
        body.append("filename", file.name);
        body.append("chunk", file.slice(start, end), `${file.name}.part`);
        const resp = await api.fetchApi("/bernini/director/upload_chunk", { method: "POST", body });
        if (!resp.ok) {
            const text = await resp.text();
            throw new Error(text || `分块上传失败 (${resp.status})`);
        }
        onProgress?.((i + 1) / totalChunks, i + 1, totalChunks);
        const data = await resp.json();
        if (data.name) return data;
    }
    throw new Error("分块上传未完成");
}

async function uploadToInputSmart(file, onProgress) {
    if (file.size <= COMFY_UPLOAD_SOFT_LIMIT) {
        try {
            return await uploadToInput(file);
        } catch (err) {
            if (!isUploadSizeError(err)) throw err;
        }
    }
    return uploadVideoChunked(file, onProgress);
}

function videoRelativePath(upload) {
    const name = upload.name || upload.filename;
    const sub = (upload.subfolder || "").replace(/\\/g, "/").replace(/\/$/, "");
    return sub ? `${sub}/${name}` : name;
}

function inputViewUrl(relativePath, type = "input") {
    const norm = String(relativePath || "").replace(/\\/g, "/");
    const slash = norm.lastIndexOf("/");
    const filename = slash >= 0 ? norm.slice(slash + 1) : norm;
    const subfolder = slash >= 0 ? norm.slice(0, slash) : "";
    const params = new URLSearchParams({ filename, type });
    if (subfolder) params.set("subfolder", subfolder);
    return api.apiURL(`/view?${params.toString()}`);
}

function refViewUrl(imageFile) {
    return inputViewUrl(imageFile, "input");
}

function deletedSourceRanges(video) {
    return video?.deletedSourceRanges || video?.deleted_source_ranges || [];
}

function logicalToSourceFrame(logical, video) {
    const map = video?.frameMap;
    if (map?.length) {
        return normalizeFrameMapEntry(map[clamp(logical, 0, map.length - 1)]).frame;
    }
    let src = logical;
    for (const [start, end] of [...deletedSourceRanges(video)].sort((a, b) => a[0] - b[0])) {
        if (src >= start) src += end - start;
        else break;
    }
    return src;
}

/** Inverse of logicalToSourceFrame for sparse deletes; -1 if source is in a deleted gap. */
function sourceToLogicalFrame(srcFrame, video) {
    const map = video?.frameMap;
    if (map?.length) {
        let best = -1;
        for (let i = 0; i < map.length; i++) {
            const e = normalizeFrameMapEntry(map[i]);
            if (e.frame === srcFrame) return i;
            if (e.frame < srcFrame) best = i;
            else if (best < 0) return -1; // before first kept
        }
        return best;
    }
    let logical = srcFrame;
    for (const [start, end] of [...deletedSourceRanges(video)].sort((a, b) => a[0] - b[0])) {
        if (srcFrame >= end) logical -= (end - start);
        else if (srcFrame >= start) return -1;
        else break;
    }
    return Math.max(0, logical);
}

function buildIdentityFrameMap(count) {
    return Array.from({ length: count }, (_, i) => i);
}

function normalizeFrameMapEntry(entry, defaultClip = 0) {
    if (entry == null) return { clip: defaultClip, frame: 0 };
    if (typeof entry === "number") return { clip: defaultClip, frame: entry };
    return {
        clip: entry.clip ?? entry.videoClip ?? defaultClip,
        frame: entry.frame ?? 0,
    };
}

function buildClipFrameMap(clipIndex, count) {
    return Array.from({ length: count }, (_, i) => ({ clip: clipIndex, frame: i }));
}

const CLIP_SEGMENT_COLORS = ["rgba(255,200,50,0.9)", "rgba(102,170,255,0.9)", "rgba(79,255,143,0.9)", "rgba(255,102,170,0.9)"];

function getDirectorUiHeight(editor) {
    const peH = getPromptEnhancerPanelHeight(editor);
    if (editor?.getDirectorMode?.() === "prompt_batch") {
        return getImageBatchUiHeight(editor) + 140 + peH;
    }
    let h = (editor?.canvasHeight || RULER_H + SEG_LABEL_H + TRACK_H) + 370 + 52 + peH;
    if (
        editor?.hasVideo?.()
        && !editor?.isImageBatch?.()
        && !editor?.isGenMode?.()
    ) {
        h += STAGE_PREVIEW_H + 10;
    }
    return h;
}

function hookTaskTypeWidget(node) {
    const tw = node.widgets?.find((w) => w.name === "task_type");
    if (!tw || tw._berniniTaskHooked) return;
    tw._berniniTaskHooked = true;
    const orig = tw.callback;
    tw.callback = function (...args) {
        const r = orig?.apply(this, args);
        const ed = node._berniniEditor;
        if (ed?.globalTask) ed.globalTask.value = tw.value;
        ed?.onTaskTypeChanged?.(tw.value);
        return r;
    };
}

function syncDirectorNodeSize(node, editor) {
    if (editor?.isPlaying) return;
    if (!node?.computeSize) return;
    if (editor) editor.updateDomWidgetHeight?.();
    const sz = node.computeSize();
    node.setSize([node.size[0], sz[1]]);
    node.setDirtyCanvas?.(true, true);
}

function ensureDirectorDomWidgetWidth(node) {
    const widget = node?._directorDomWidget;
    const fullW = node?.size?.[0];
    if (!widget || !fullW) return false;
    if (widget.width === fullW) return false;
    widget.width = fullW;
    return true;
}

function moveDirectorDomWidgetToEnd(node) {
    const widget = node?._directorDomWidget;
    if (!widget || !node?.widgets?.length) return;
    const idx = node.widgets.indexOf(widget);
    if (idx === -1 || idx === node.widgets.length - 1) return;
    node.widgets.splice(idx, 1);
    node.widgets.push(widget);
}

const PERF_WIDGET_ORDER = ["bd_grp_perf", "clear_vram_between_segments", "export_source_images"];

function moveDirectorPerfWidgetsBeforeTimeline(node) {
    const dom = node?._directorDomWidget;
    if (!node?.widgets?.length) return;

    const perfWidgets = PERF_WIDGET_ORDER
        .map((name) => node.widgets.find((w) => w.name === name))
        .filter(Boolean);
    if (!perfWidgets.length) return;

    for (const w of perfWidgets) {
        const idx = node.widgets.indexOf(w);
        if (idx !== -1) node.widgets.splice(idx, 1);
    }

    const insertAt = dom ? node.widgets.indexOf(dom) : -1;
    const at = insertAt === -1 ? node.widgets.length : insertAt;
    node.widgets.splice(at, 0, ...perfWidgets);
}

function finalizeDirectorWidgetOrder(node) {
    moveDirectorPerfWidgetsBeforeTimeline(node);
    moveDirectorDomWidgetToEnd(node);
}

function bindDirectorDomWidgetSizing(node, widget, getEditor) {
    const minHeight = () => getDirectorUiHeight(getEditor?.());
    widget.computeSize = (width) => [width, minHeight()];
    widget.computeLayoutSize = () => ({
        minHeight: minHeight(),
        minWidth: DIRECTOR_MIN_WIDTH,
    });
    if (widget.options) {
        widget.options.getMinHeight = minHeight;
    }
    const el = widget.element;
    if (el) el.style.minHeight = `${minHeight()}px`;
}

function initDirectorEditor(node) {
    if (node._berniniEditor) return node._berniniEditor;
    const container = node._directorDomWidget?.element;
    if (!container) return null;
    try {
        hookTaskTypeWidget(node);
        node._berniniEditor = new BerniniDirectorEditor(node, container, node._directorDomWidget);
        ensureDirectorDomWidgetWidth(node);
        bindDirectorDomWidgetSizing(node, node._directorDomWidget, () => node._berniniEditor);
        syncDirectorNodeSize(node, node._berniniEditor);
        return node._berniniEditor;
    } catch (err) {
        console.error("[BerniniDirector] UI init failed:", err);
        return null;
    }
}

function patchDirectorDomWidgetLayout() {
    const canvas = app.canvas;
    if (!canvas || canvas._berniniDirectorLayoutPatch) return;
    canvas._berniniDirectorLayoutPatch = true;
    const prev = canvas.onDrawForeground;
    canvas.onDrawForeground = function (ctx) {
        const graph = app.graph ?? canvas.graph;
        for (const node of graph?._nodes ?? graph?.nodes ?? []) {
            if (node._berniniEditor?.isPlaying) continue;
            ensureDirectorDomWidgetWidth(node);
        }
        return prev?.apply(this, arguments);
    };
}

function stopDomEvent(e) {
    e.stopPropagation();
}

function hideWidget(w) {
    if (!w) return;
    // Group headers in HIDDEN_WIDGETS (e.g. bd_grp_pe) duplicate timeline panel sections — hide them too.
    if (w._bdGroupHeader && !HIDDEN_WIDGETS.includes(w.name)) return;
    w.hidden = true;
    if (!w.options) w.options = {};
    w.options.hidden = true;
    w.computeSize = () => [0, 0];
    if (w.element) w.element.style.display = "none";
}

function parseTimeline(raw, totalFrames, fps) {
    const total = totalFrames || 81;
    const base = {
        version: 4,
        editMode: "global",
        totalFrames: total,
        frameRate: coerceTimelineFps(fps || 24),
        video: {
            fileName: "",
            videoFile: "",
            subfolder: "",
            type: "input",
            frames: [],
            frameMap: [],
        },
        videoClips: [],
        global: { taskType: "", prompt: "", refs: [], referenceVideo: {}, continuousReference: false },
        output: { mode: "long_edge", longEdge: 848, width: 832, height: 480, maxExportFrames: 0, exportMode: "all", continuityEnabled: false, continuityOverlapFrames: 9 },
        runSelectEnabled: false,
        runSelection: [],
        segments: [{ id: uid(), start: 0, length: total, prompt: "", taskType: "", refs: [], referenceVideo: {} }],
    };
    if (!raw?.trim()) return base;
    try {
        const data = JSON.parse(raw);
        data.version = data.version || 4;
        data.editMode = data.editMode || "global";
        data.frameRate = coerceTimelineFps(data.frameRate ?? fps ?? 24);
        data.video = data.video || { fileName: "", frames: [] };
        if (!data.video.videoFile && data.video.fileName) {
            data.video.videoFile = data.video.fileName;
        }
        data.video.type = data.video.type || "input";
        data.video.subfolder = data.video.subfolder || "";
        data.video.frames = data.video.frames || [];
        data.global = data.global || { refs: [], referenceVideo: {}, continuousReference: false };
        data.global.referenceVideo = data.global.referenceVideo || data.global.reference_video || {};
        data.global.continuousReference = !!data.global.continuousReference || !!data.global.continuous_reference;
        const legacyRef = data.referenceVideo || data.reference_video;
        if (legacyRef && (legacyRef.videoFile || legacyRef.fileName)
            && !(data.global.referenceVideo.videoFile || data.global.referenceVideo.fileName)) {
            data.global.referenceVideo = { ...legacyRef };
        }
        delete data.referenceVideo;
        delete data.reference_video;
        data.output = normalizeOutputContinuity({
            mode: data.output?.mode || "long_edge",
            longEdge: data.output?.longEdge ?? data.output?.long_edge ?? data.refMaxSize ?? 848,
            width: data.output?.width ?? data.width ?? 832,
            height: data.output?.height ?? data.height ?? 480,
            maxExportFrames: data.output?.maxExportFrames ?? data.output?.max_export_frames ?? 0,
            exportMode: data.output?.exportMode ?? data.output?.export_mode ?? "all",
            continuityEnabled: data.output?.continuityEnabled ?? data.output?.continuity_enabled,
            continuityOverlapFrames: data.output?.continuityOverlapFrames ?? data.output?.continuity_overlap_frames,
        });
        stripTimelineContinuityRootFields(data);
        stripTimelineEphemeralFields(data);
        const legacyFrames = data.video.frames?.length || 0;
        if (!data.video.frameMap?.length) {
            const n = data.totalFrames || data.video.sourceFrameCount || legacyFrames || total;
            data.totalFrames = n;
            data.video.sourceFrameCount = data.video.sourceFrameCount || n;
            data.video.deletedSourceRanges = data.video.deletedSourceRanges || [];
            data.video.frameMap = [];
        }
        if (!data.segments?.length) {
            const n = data.totalFrames || data.video.sourceFrameCount || legacyFrames || total;
            data.segments = [{ id: uid(), start: 0, length: Math.max(MIN_SEG, n), prompt: "", taskType: "", refs: [], referenceVideo: {} }];
        }
        for (const seg of data.segments) {
            if (!seg.id) seg.id = uid();
            if (seg.length == null && seg.end != null) seg.length = seg.end - seg.start;
            if (seg.frameCount == null && seg.length != null) seg.frameCount = seg.length;
            seg.refs = seg.refs || [];
            seg.referenceVideo = seg.referenceVideo || seg.reference_video || {};
            seg.genImage = seg.genImage || { imageFile: seg.imageFile || "" };
            seg.negativePrompt = seg.negativePrompt ?? "";
        }
        data.gen = data.gen || { defaultFrameCount: 81 };
        if (data.global) {
            data.global.genImage = data.global.genImage || { imageFile: data.global.imageFile || "" };
        }
        data.runSelectEnabled = !!data.runSelectEnabled;
        data.runSelection = Array.isArray(data.runSelection) ? data.runSelection.map((i) => parseInt(i, 10)).filter((i) => i >= 0) : [];
        if (data.timelineMode === "image_batch" || data.timelineMode === "prompt_batch") {
            data.timelineMode = "prompt_batch";
            data.editMode = "segment";
            data.totalFrames = sumFrameCounts(data.segments) || data.totalFrames || total;
            return data;
        }
        if (data.timelineMode === "gen_blank" || data.timelineMode === "gen_image") {
            const gkey = resolveTaskKey(data.global?.taskType || "");
            if (isPromptBatchTask(gkey)) {
                data.timelineMode = "prompt_batch";
                data.editMode = "segment";
            }
            data.totalFrames = sumFrameCounts(data.segments) || data.totalFrames || total;
            return data;
        }
        if (!data.videoClips?.length && data.video?.videoFile) {
            data.videoClips = [{
                id: data.video.id || uid(),
                fileName: data.video.fileName || "",
                videoFile: data.video.videoFile || data.video.fileName || "",
                subfolder: data.video.subfolder || "",
                type: data.video.type || "input",
                width: data.video.width || 0,
                height: data.video.height || 0,
                duration: data.video.duration || 0,
                nativeFps: data.video.nativeFps || data.video.native_fps || 0,
                nativeFrameCount: data.video.nativeFrameCount || data.video.native_frame_count || 0,
                sourceFrameCount: data.video.sourceFrameCount || data.video.frameMap?.length || 0,
                storageWidth: data.video.storageWidth,
                storageHeight: data.video.storageHeight,
            }];
        }
        data.videoClips = data.videoClips || [];
        data.totalFrames = data.totalFrames || data.video.sourceFrameCount || data.video.frameMap?.length || total;
        return data;
    } catch {
        return base;
    }
}

class BerniniDirectorEditor {
    constructor(node, container, domWidget) {
        this.node = node;
        this.container = container;
        this.domWidget = domWidget;
        this.zoom = 1;
        this.selectedIndex = 0;
        this.currentFrame = 0;
        this.isPlaying = false;
        this.isLooping = false;
        this._playRaf = null;
        this._drag = null;
        this._previewSegments = null;
        this._edgeSnapshot = null;
        this._isHovering = false;
        this._thumbCache = new Map();
        this._thumbPending = new Set();
        this._seekChain = Promise.resolve();
        this._legacyFrames = [];
        this._storageWidth = 0;
        this._storageHeight = 0;
        this._previewVideo = null;
        this._previewVideos = new Map();
        this._thumbCanvas = null;
        this._syncTimer = null;
        this._resizeRaf = null;
        this._renderPending = false;
        this._lastSeekUiMs = 0;
        this._playCanvasWidth = 0;
        this._pauseSettling = false;
        this._runHighlightSeg = -1;
        this._modalEl = null;
        this._modalKeyHandler = null;
        this._drawWidth = 0;
        this._reorderDropRank = -1;
        this._reorderFromRank = -1;
        this.canvasHeight = RULER_H + SEG_LABEL_H + TRACK_H;
        this._stageClipIndex = -1;
        this._stageSyncMs = 0;
        this._playHandoff = false;

        for (const w of node.widgets || []) {
            if (HIDDEN_WIDGETS.includes(w.name)) hideWidget(w);
        }

        this.timelineWidget = this.widget("timeline_data");
        this.totalFramesWidget = this.widget("total_frames");
        this.frameRateWidget = this.widget("frame_rate");
        this.taskTypeWidget = this.widget("task_type");
        this.globalPromptWidget = this.widget("global_prompt");
        this.negativePromptWidget = this.widget("negative_prompt");
        this.widthWidget = this.widget("width");
        this.heightWidget = this.widget("height");
        this.refMaxWidget = this.widget("ref_max_size");

        const initTotal = Math.max(0, parseInt(this.totalFramesWidget?.value || 81, 10));
        const initFps = coerceTimelineFps(this.frameRateWidget?.value || 24);
        this.timeline = parseTimeline(this.timelineWidget?.value, initTotal, initFps);
        this.buildDOM();
        this.bindEvents();
        this._directorMode = getDirectorMode(this.taskTypeWidget?.value);
        if (this._directorMode === "video") {
            this.restoreVideoFromTimeline();
        } else if (this._directorMode === "prompt_batch" || this._directorMode === "image_batch") {
            ensureImageBatchTimeline(this);
        } else {
            this.ensureGenTimeline();
        }
        this.applyTaskLayout(this._directorMode);

        this.updateDomWidgetHeight();
        this.applyZoomWidth();
        this.syncFromWidgets();
        this.updateModeUI();
        this.updateSelectionUI();
        this.commit(true, { syncTimeline: false });
        this._observeViewportResize();
        this.scheduleRender();
    }

    _observeViewportResize() {
        if (!this.viewport || typeof ResizeObserver === "undefined") return;
        this._resizeObserver?.disconnect();
        this._resizeObserver = new ResizeObserver(() => {
            if (this.isPlaying) return;
            this.scheduleRender();
        });
        this._resizeObserver.observe(this.viewport);
    }

    _capturePlayCanvasWidth() {
        const w = this.viewport?.clientWidth
            || this.container?.offsetWidth
            || this.node?.size?.[0]
            || DIRECTOR_MIN_WIDTH;
        if (w > 0) this._playCanvasWidth = w;
        return this._playCanvasWidth;
    }

    _lockPlayLayout() {
        this._capturePlayCanvasWidth();
    }

    _resetLayoutStyles() {
        if (this.isPlaying) return;
        for (const el of [this.container, this.root, this.viewport]) {
            if (!el) continue;
            el.style.removeProperty("width");
            el.style.removeProperty("min-width");
            el.style.removeProperty("max-width");
        }
        this._playCanvasWidth = 0;
        this.applyZoomWidth();
    }

    _releasePlayLayoutLock() {
        this._resetLayoutStyles();
    }

    updateDomWidgetHeight() {
        const h = getDirectorUiHeight(this);
        this.container?.style.setProperty("--comfy-widget-min-height", String(h));
        if (this.container) this.container.style.minHeight = `${h}px`;
        if (this.domWidget) {
            this.domWidget.computeSize = (width) => [width, h];
            if (this.domWidget.options) {
                this.domWidget.options.getMinHeight = () => getDirectorUiHeight(this);
            }
        }
    }

    scheduleRender() {
        if (this._renderPending) return;
        this._renderPending = true;
        this._resizeRaf = requestAnimationFrame(() => {
            this._renderPending = false;
            if (this.isPlaying) this.renderTimelineOnly();
            else this.render();
        });
    }

    buildTimelinePayload() {
        if (this.isImageBatch()) {
            const taskKey = this.getTaskKey();
            const i2iSrc = (taskKey === "i2i" || taskKey === "i2v") ? this.getI2iSourceDimensions() : null;
            const outMode = imageBatchRequiresFixedOutput(taskKey)
                ? "fixed"
                : (this.timeline.output?.mode || "long_edge");
            const output = normalizeOutputContinuity({
                ...this.timeline.output,
                mode: outMode,
            });
            if (!isVideoBatchTask(taskKey)) {
                output.exportMode = "all";
            }
            if (i2iSrc?.width > 0 && i2iSrc?.height > 0) {
                output.sourceWidth = i2iSrc.width;
                output.sourceHeight = i2iSrc.height;
            }
            const batchBody = { ...this.timeline };
            stripTimelineContinuityRootFields(batchBody);
            stripTimelineEphemeralFields(batchBody);
            return {
                ...batchBody,
                version: 5,
                timelineMode: "prompt_batch",
                editMode: "segment",
                totalFrames: sumFrameCounts(this.timeline.segments),
                frameRate: this.getFrameRate(),
                width: this.timeline.output?.width,
                height: this.timeline.output?.height,
                global: {
                    ...this.timeline.global,
                    taskType: this.globalTask?.value || this.taskTypeWidget?.value || "",
                    prompt: this.timeline.global?.prompt || "",
                    ...(i2iSrc?.width > 0 ? { sourceWidth: i2iSrc.width, sourceHeight: i2iSrc.height } : {}),
                },
                output,
                segments: this.timeline.segments.map((s) => ({
                    id: s.id,
                    start: s.start,
                    length: s.frameCount ?? s.length ?? 1,
                    frameCount: s.frameCount ?? s.length ?? 1,
                    prompt: s.prompt || "",
                    negativePrompt: s.negativePrompt || "",
                    taskType: s.taskType || "",
                    refs: s.refs || [],
                    genImage: s.genImage || { imageFile: "" },
                })),
                ...this._runSelectionPayload(),
            };
        }
        if (this.isGenMode()) {
            const mode = this.getDirectorMode();
            const genBody = { ...this.timeline };
            stripTimelineContinuityRootFields(genBody);
            stripTimelineEphemeralFields(genBody);
            return {
                ...genBody,
                version: 5,
                timelineMode: mode,
                totalFrames: sumFrameCounts(this.timeline.segments),
                frameRate: this.getFrameRate(),
                width: this.timeline.output?.width,
                height: this.timeline.output?.height,
                refMaxSize: this.timeline.output?.longEdge,
                global: {
                    ...this.timeline.global,
                    taskType: this.globalTask?.value || this.taskTypeWidget?.value || "",
                    prompt: this.timeline.global?.prompt || "",
                },
                output: normalizeOutputContinuity({ ...this.timeline.output }),
                segments: this.timeline.segments.map((s) => ({
                    ...s,
                    frameCount: s.frameCount ?? s.length,
                })),
                ...this._runSelectionPayload(),
            };
        }
        const video = { ...(this.timeline.video || {}) };
        const frameMap = video.frameMap?.length ? video.frameMap : [];
        const src = this.getSourceDimensions();
        const resolved = resolveOutputDimensions(src.width, src.height, this.timeline.output || {}, {
            refMaxSize: this.refMaxWidget?.value,
        });
        const storageW = resolved.width || video.storageWidth || this._storageWidth;
        const storageH = resolved.height || video.storageHeight || this._storageHeight;
        const clips = this.getVideoClips().map((c) => ({
            ...c,
            storageWidth: storageW,
            storageHeight: storageH,
        }));
        const { referenceVideo: _legacyRefVideo, reference_video: _legacyRefVideo2, ...timelineBody } = this.timeline;
        stripTimelineContinuityRootFields(timelineBody);
        stripTimelineEphemeralFields(timelineBody);
        const clipSourceTotal = clips.reduce(
            (s, c) => s + (parseInt(c.sourceFrameCount, 10) || 0),
            0,
        );
        const sourceFrameCount = parseInt(video.sourceFrameCount, 10)
            || clipSourceTotal
            || (frameMap.length ? 0 : this.getTotalFrames());
        return {
            ...timelineBody,
            version: 4,
            timelineMode: "video",
            totalFrames: this.getTotalFrames(),
            frameRate: this.getFrameRate(),
            videoClips: clips,
            global: {
                ...(this.timeline.global || {}),
                taskType: this.globalTask?.value || this.taskTypeWidget?.value || "",
                prompt: this.timeline.global?.prompt || "",
                referenceVideo: this.timeline.global?.referenceVideo || {},
                continuousReference: !!this.timeline.global?.continuousReference,
            },
            segments: (this.timeline.segments || []).map((s) => ({
                ...s,
                referenceVideo: s.referenceVideo || {},
            })),
            video: {
                ...video,
                frameMap,
                sourceFrameCount,
                deletedSourceRanges: frameMap.length ? [] : (video.deletedSourceRanges || []),
                frames: this._legacyFrames.length ? this._legacyFrames : [],
                storageWidth: storageW,
                storageHeight: storageH,
            },
            output: normalizeOutputContinuity({ ...this.timeline.output }),
            ...this._runSelectionPayload(),
        };
    }

    flushTimelineSync() {
        clearTimeout(this._syncTimer);
        this._syncTimer = null;
        this._writeTimelineWidget();
    }

    scheduleTimelineSync() {
        clearTimeout(this._syncTimer);
        this._syncTimer = setTimeout(() => this._writeTimelineWidget(), TIMELINE_SYNC_DEBOUNCE_MS);
    }

    _writeTimelineWidget() {
        if (!this.timelineWidget) return;
        this.syncFromWidgets();
        this.timelineWidget.value = JSON.stringify(this.buildTimelinePayload());
        this.node.setDirtyCanvas(true, false);
    }

    _markNodeDirtyLight() {
        this.node.setDirtyCanvas(true, false);
    }

    buildDOM() {
        this.root = document.createElement("div");
        this.root.className = "bd-wrap";
        this.root.innerHTML = `<style>${STYLES}</style>`;

        const toolbar = document.createElement("div");
        toolbar.className = "bd-toolbar";
        toolbar.innerHTML = `
            <div class="bd-actions">
                <button type="button" class="bd-btn bd-btn-primary" data-a="video">上传视频</button>
                <button type="button" class="bd-btn" data-a="video-append" title="上传并追加到时间轴末尾，作为独立片段">追加视频</button>
                <button type="button" class="bd-btn" data-a="split">+ 分割</button>
                <input type="number" class="bd-num" data-r="equal-n" min="2" max="64" value="2" title="均分段数">
                <button type="button" class="bd-btn" data-a="equal">均分</button>
                <button type="button" class="bd-btn" data-a="run-select-toggle" title="开启后可勾选要运行的片段/提示词组；关闭时运行全部">选择运行</button>
                <label class="bd-run-select-all-wrap hidden" data-r="run-select-all-wrap" title="勾选=全选，取消=全部不选；仍可在各片段上单独勾选">
                    <input type="checkbox" data-r="run-select-all-cb">
                    <span>全选</span>
                </label>
                <button type="button" class="bd-btn bd-btn-danger" data-a="del" title="删除选中片段并裁剪视频，时间轴自动衔接">删除片段</button>
                <div class="bd-mode">
                    <button type="button" data-a="mode-global" class="active">全局模式</button>
                    <button type="button" data-a="mode-segment">分段模式</button>
                </div>
                <select class="bd-select" data-r="global-task" title="task_type"></select>
                <span class="bd-video-tag" data-r="video-name">未上传视频</span>
            </div>
            <div class="bd-right">
                <div class="bd-bounds" data-r="bounds">Start: 0.00 | End: -</div>
                <div class="bd-timecode" data-r="timecode">0.00s</div>
            </div>`;
        this.root.appendChild(toolbar);

        this.mainBody = document.createElement("div");
        this.mainBody.className = "bd-main";
        this.root.appendChild(this.mainBody);

        const stage = document.createElement("div");
        stage.className = "bd-stage hidden";
        stage.setAttribute("data-r", "video-stage");
        stage.innerHTML = `
            <video class="bd-stage-video hidden" data-r="stage-video" muted playsinline preload="auto"></video>
            <img class="bd-stage-img hidden" data-r="stage-img" alt="">
            <div class="bd-stage-empty" data-r="stage-empty">上传视频后可在此预览播放</div>
            <div class="bd-stage-badge hidden" data-r="stage-badge"></div>`;
        this.mainBody.appendChild(stage);

        // Playback bar sits between video stage and timeline edit area.
        const controls = document.createElement("div");
        controls.className = "bd-controls";
        controls.innerHTML = `
            <div class="bd-player">
                <button type="button" class="bd-icon-btn" data-a="play" title="播放 / 暂停">▶</button>
                <button type="button" class="bd-icon-btn" data-a="loop" title="循环播放：开启后预览播放到末尾会自动从头开始">⟳</button>
                <button type="button" class="bd-icon-btn" data-a="frame-prev" title="上一帧 (←)">‹</button>
                <button type="button" class="bd-icon-btn" data-a="frame-next" title="下一帧 (→)">›</button>
                <span class="bd-frame-jump" title="输入帧号后回车定位（从 1 开始）">
                    <span>帧</span>
                    <input type="number" class="bd-frame-input" data-r="frame-input" min="1" step="1" value="1">
                    <span>/</span>
                    <span class="bd-frame-total" data-r="frame-total">0</span>
                </span>
                <div class="bd-timecode" data-r="player-timecode">0.00 / 0.00</div>
                <input type="range" class="bd-seek" data-r="seek" min="0" value="0" step="1">
                <div class="bd-zoom bd-row">
                    <button type="button" class="bd-icon-btn" data-a="zoom-out">−</button>
                    <input type="range" data-r="zoom" min="1" max="10" step="0.25" value="1" style="width:80px">
                    <button type="button" class="bd-icon-btn" data-a="zoom-in">+</button>
                </div>
            </div>`;
        this.mainBody.appendChild(controls);

        this.viewport = document.createElement("div");
        this.viewport.className = "bd-viewport";
        this.canvas = document.createElement("canvas");
        this.canvas.className = "bd-canvas";
        this.viewport.appendChild(this.canvas);
        this.mainBody.appendChild(this.viewport);
        this.ctx = this.canvas.getContext("2d");

        const outputBar = document.createElement("div");
        outputBar.className = "bd-output";
        outputBar.innerHTML = `
            <label>输出分辨率</label>
            <select class="bd-select" data-r="out-mode" title="输出缩放模式">
                <option value="long_edge">最长边缩放</option>
                <option value="fixed">固定宽高</option>
            </select>
            <span class="bd-out-long" data-r="out-long-wrap">
                <label>最长边</label>
                <input type="number" class="bd-num" data-r="out-long" min="16" max="8192" step="16" value="848" style="width:56px">
            </span>
            <span class="bd-out-fixed hidden" data-r="out-fixed-wrap">
                <label>宽</label>
                <input type="number" class="bd-num" data-r="out-w" min="16" max="8192" step="16" value="832" style="width:56px">
                <label>高</label>
                <input type="number" class="bd-num" data-r="out-h" min="16" max="8192" step="16" value="480" style="width:56px">
            </span>
            <label title="上传后默认跟源视频 FPS；修改时会保持真实时长不变并重算帧数（例：30→24fps 时 275 帧→约 220 帧，时长仍约 9.2s）">FPS</label>
            <input type="number" class="bd-num" data-r="timeline-fps" min="1" max="240" step="0.01" value="24" style="width:64px" title="时间线/导出 FPS">
            <span class="bd-meta" data-r="out-preview">—</span>
            <span class="bd-meta hidden" data-r="out-hint"></span>
            <label title="全部导出：合并为一个视频；分段导出：每段时间轴片段单独输出（images 输出为列表，Video Combine 会生成多个 MP4）">导出方式</label>
            <select class="bd-select" data-r="out-export-mode" title="输出方式">
                <option value="all">全部导出</option>
                <option value="segments">分段导出</option>
            </select>
            <label title="0 = 导出全部帧；大于 0 时仅处理前 N 帧">最大帧数</label>
            <input type="number" class="bd-num" data-r="out-max-frames" min="0" max="999999" step="1" value="0" style="width:64px" title="0 = 全部导出">
            <span class="bd-continuous-ref hidden" data-r="segment-continuity-wrap" title="多段衔接：用上一段尾帧做 SCAIL 引导；短段+高重叠时易在接缝附近粘帧，已自动裁掉回声帧">
                <label><input type="checkbox" data-r="segment-continuity-cb">段间引导</label>
                <span class="bd-meta">参考帧数</span>
                <input type="number" class="bd-num" data-r="segment-continuity-overlap" min="1" max="81" step="4" value="9" style="width:48px" title="SCAIL 锁定帧数（建议 9 或 13）；过短分段时建议关闭段间引导">
            </span>`;
        this.mainBody.appendChild(outputBar);

        const bottom = document.createElement("div");
        bottom.className = "bd-split";
        bottom.innerHTML = `
            <div class="bd-panel" data-r="global-panel">
                <b>全局提示词 & 参考图 (image0–4)</b>
                <div class="bd-prompt-layout">
                    <div class="bd-prompt-col">
                        <span class="bd-label">正向提示词</span>
                        <textarea class="bd-prompt" data-r="global-prompt" placeholder="全局提示词 — 输入 @ 选择参考图"></textarea>
                        <span class="bd-label">反向提示词</span>
                        <textarea class="bd-prompt bd-prompt-negative" data-r="global-negative" placeholder="反向提示词 — 所有片段共用"></textarea>
                    </div>
                    <div class="bd-refs-col" data-r="global-refs-col">
                        <div data-r="global-refs-images-wrap">
                            <span class="bd-label" data-r="global-refs-label">参考图 (image0–4)</span>
                            <div class="bd-refs" data-r="global-refs"></div>
                        </div>
                        <div class="bd-ref-video-col hidden" data-r="global-ref-video-col">
                            <span class="bd-label">参考视频（植入内容）</span>
                            <div class="bd-gen-src" data-r="global-ref-video" title="上传要植入的参考视频">点击上传参考视频</div>
                            <span class="bd-meta bd-ref-video-name" data-r="global-ref-video-name"></span>
                            <label class="bd-continuous-ref hidden" data-r="continuous-ref-wrap" title="勾选后，各片段的参考视频从与源片段时间轴相同的帧位置开始（如第2段从第30帧起，参考视频也从第30帧起）；未勾选时每段均从参考视频第1帧开始">
                                <input type="checkbox" data-r="continuous-ref-cb">
                                <span>连续参考</span>
                            </label>
                        </div>
                        <div class="bd-gen-src hidden" data-r="gen-global-img" title="上传源图片">点击上传源图片</div>
                    </div>
                </div>
                <div class="bd-gen-fc-row hidden" data-r="gen-global-fc-row">
                    <span class="bd-label">默认片段帧数</span>
                    <input type="number" class="bd-num" data-r="gen-default-fc" min="1" max="${MAX_GEN_FRAMES}" value="81" style="width:72px">
                </div>
            </div>
            <div class="bd-panel" data-r="segment-panel" style="display:none">
                <b data-r="seg-label">片段 1</b>
                <div class="bd-meta" data-r="seg-info"></div>
                <div class="bd-prompt-layout">
                    <div class="bd-prompt-col">
                        <span class="bd-label">正向提示词</span>
                        <textarea class="bd-prompt" data-r="seg-prompt" placeholder="该片段的提示词 — 输入 @ 选择参考图"></textarea>
                        <span class="bd-label">反向提示词</span>
                        <textarea class="bd-prompt bd-prompt-negative" data-r="seg-negative" placeholder="反向提示词 — 所有片段共用"></textarea>
                    </div>
                    <div class="bd-refs-col" data-r="seg-refs-col">
                        <div data-r="seg-refs-images-wrap">
                            <span class="bd-label" data-r="seg-refs-label">片段参考图 (image0–4)</span>
                            <div class="bd-refs" data-r="seg-refs"></div>
                        </div>
                        <div class="bd-ref-video-col hidden" data-r="seg-ref-video-col">
                            <span class="bd-label">片段参考视频（植入内容）</span>
                            <div class="bd-gen-src" data-r="seg-ref-video" title="上传要植入的参考视频">点击上传参考视频</div>
                            <span class="bd-meta bd-ref-video-name" data-r="seg-ref-video-name"></span>
                        </div>
                        <div class="bd-gen-src hidden" data-r="gen-seg-img" title="上传片段源图片">点击上传源图片</div>
                    </div>
                </div>
                <div class="bd-gen-fc-row hidden" data-r="gen-seg-fc-row">
                    <span class="bd-label">片段帧数</span>
                    <input type="number" class="bd-num" data-r="gen-seg-fc" min="1" max="${MAX_GEN_FRAMES}" value="81" style="width:72px">
                </div>
            </div>`;
        this.mainBody.appendChild(bottom);

        const batchUi = mountImageBatchPanel(this.mainBody);
        this.batchPanel = batchUi.panel;
        this.batchList = batchUi.list;
        this.batchHint = batchUi.hint;
        this.batchI2vNotice = batchUi.i2vNotice;
        this.batchAddBtn = batchUi.addBtn;
        wireBatchRunSelectControls(this, batchUi);

        // PE 始终紧跟在所有提示词区域之后（视频/生成 .bd-split，批量 t2i/i2i/i2v 等 .bd-batch）
        const peHost = document.createElement("div");
        peHost.className = "bd-pe-host";
        this.mainBody.appendChild(peHost);
        this.peHost = peHost;
        mountPromptEnhancerPanel(this, peHost);

        const runStatus = document.createElement("div");
        runStatus.className = "bd-run-status idle";
        runStatus.dataset.r = "run-status";
        runStatus.innerHTML = `
            <div class="bd-run-title" data-r="run-title">运行状态：待命</div>
            <div class="bd-run-detail" data-r="run-detail">队列执行时将显示当前片段与阶段进度</div>
            <div class="bd-run-select-bar hidden" data-r="run-select-bar">
                <span data-r="run-select-summary">将运行全部片段</span>
            </div>
            <div class="bd-run-bars">
                <div class="bd-run-bar" title="整体进度"><div class="bd-run-bar-fill" data-r="run-overall" style="width:0%"></div></div>
                <div class="bd-run-bar bd-run-bar-sub" title="当前阶段"><div class="bd-run-bar-fill" data-r="run-phase" style="width:0%"></div></div>
            </div>`;
        this.root.appendChild(runStatus);

        this.container.appendChild(this.root);

        this._previewVideo = document.createElement("video");
        this._previewVideo.crossOrigin = "anonymous";
        this._previewVideo.muted = true;
        this._previewVideo.playsInline = true;
        this._previewVideo.preload = "auto";
        this._previewVideo.style.cssText = "position:fixed;width:0;height:0;opacity:0;pointer-events:none";
        document.body.appendChild(this._previewVideo);

        this._thumbCanvas = document.createElement("canvas");
        this._thumbCtx = this._thumbCanvas.getContext("2d", { alpha: false });

        this.videoNameEl = this.root.querySelector('[data-r="video-name"]');
        this.equalCountInput = this.root.querySelector('[data-r="equal-n"]');
        this.boundsEl = this.root.querySelector('[data-r="bounds"]');
        this.timecodeEl = this.root.querySelector('[data-r="timecode"]');
        this.playerTimecodeEl = this.root.querySelector('[data-r="player-timecode"]');
        this.frameInputEl = this.root.querySelector('[data-r="frame-input"]');
        this.frameTotalEl = this.root.querySelector('[data-r="frame-total"]');
        this.seekBar = this.root.querySelector('[data-r="seek"]');
        this.zoomSlider = this.root.querySelector('[data-r="zoom"]');
        this.stageEl = this.root.querySelector('[data-r="video-stage"]');
        this.stageVideo = this.root.querySelector('[data-r="stage-video"]');
        this.stageImg = this.root.querySelector('[data-r="stage-img"]');
        this.stageEmpty = this.root.querySelector('[data-r="stage-empty"]');
        this.stageBadge = this.root.querySelector('[data-r="stage-badge"]');
        if (this.stageVideo) {
            this.stageVideo.crossOrigin = "anonymous";
            this.stageVideo.muted = true;
            this.stageVideo.playsInline = true;
        }
        this.globalTask = this.root.querySelector('[data-r="global-task"]');
        this.globalPanel = this.root.querySelector('[data-r="global-panel"]');
        this.globalPanelTitle = this.globalPanel?.querySelector("b");
        this.segmentPanel = this.root.querySelector('[data-r="segment-panel"]');
        this.globalPrompt = this.root.querySelector('[data-r="global-prompt"]');
        this.globalNegative = this.root.querySelector('[data-r="global-negative"]');
        this.globalRefsBox = this.root.querySelector('[data-r="global-refs"]');
        this.globalRefsImagesWrap = this.root.querySelector('[data-r="global-refs-images-wrap"]');
        this.segRefsImagesWrap = this.root.querySelector('[data-r="seg-refs-images-wrap"]');
        this.segLabel = this.root.querySelector('[data-r="seg-label"]');
        this.segInfo = this.root.querySelector('[data-r="seg-info"]');
        this.segPrompt = this.root.querySelector('[data-r="seg-prompt"]');
        this.segNegative = this.root.querySelector('[data-r="seg-negative"]');
        this.segRefsBox = this.root.querySelector('[data-r="seg-refs"]');
        this.globalRefsCol = this.root.querySelector('[data-r="global-refs-col"]');
        this.segRefsCol = this.root.querySelector('[data-r="seg-refs-col"]');
        this.globalRefVideoCol = this.root.querySelector('[data-r="global-ref-video-col"]');
        this.globalRefVideo = this.root.querySelector('[data-r="global-ref-video"]');
        this.globalRefVideoNameEl = this.root.querySelector('[data-r="global-ref-video-name"]');
        this.segRefVideoCol = this.root.querySelector('[data-r="seg-ref-video-col"]');
        this.segRefVideo = this.root.querySelector('[data-r="seg-ref-video"]');
        this.segRefVideoNameEl = this.root.querySelector('[data-r="seg-ref-video-name"]');
        this.continuousRefWrap = this.root.querySelector('[data-r="continuous-ref-wrap"]');
        this.continuousRefCb = this.root.querySelector('[data-r="continuous-ref-cb"]');
        this.genGlobalImg = this.root.querySelector('[data-r="gen-global-img"]');
        this.genSegImg = this.root.querySelector('[data-r="gen-seg-img"]');
        this.genGlobalFcRow = this.root.querySelector('[data-r="gen-global-fc-row"]');
        this.genSegFcRow = this.root.querySelector('[data-r="gen-seg-fc-row"]');
        this.genDefaultFc = this.root.querySelector('[data-r="gen-default-fc"]');
        this.genSegFc = this.root.querySelector('[data-r="gen-seg-fc"]');
        this.controlsBar = this.root.querySelector(".bd-controls");
        this.btnVideo = this.root.querySelector('[data-a="video"]');
        this.btnVideoAppend = this.root.querySelector('[data-a="video-append"]');
        this.outHint = this.root.querySelector('[data-r="out-hint"]');
        this.outMode = this.root.querySelector('[data-r="out-mode"]');
        this.outLongWrap = this.root.querySelector('[data-r="out-long-wrap"]');
        this.outFixedWrap = this.root.querySelector('[data-r="out-fixed-wrap"]');
        this.outLong = this.root.querySelector('[data-r="out-long"]');
        this.outW = this.root.querySelector('[data-r="out-w"]');
        this.outH = this.root.querySelector('[data-r="out-h"]');
        this.fpsInput = this.root.querySelector('[data-r="timeline-fps"]');
        this.outMaxFrames = this.root.querySelector('[data-r="out-max-frames"]');
        this.outExportMode = this.root.querySelector('[data-r="out-export-mode"]');
        this.segmentContinuityWrap = this.root.querySelector('[data-r="segment-continuity-wrap"]');
        this.segmentContinuityCb = this.root.querySelector('[data-r="segment-continuity-cb"]');
        this.segmentContinuityOverlap = this.root.querySelector('[data-r="segment-continuity-overlap"]');
        this.outPreview = this.root.querySelector('[data-r="out-preview"]');
        this.runStatusEl = this.root.querySelector('[data-r="run-status"]');
        this.runTitleEl = this.root.querySelector('[data-r="run-title"]');
        this.runDetailEl = this.root.querySelector('[data-r="run-detail"]');
        this.runOverallEl = this.root.querySelector('[data-r="run-overall"]');
        this.runPhaseEl = this.root.querySelector('[data-r="run-phase"]');
        this.runSelectBar = this.root.querySelector('[data-r="run-select-bar"]');
        this.runSelectSummary = this.root.querySelector('[data-r="run-select-summary"]');
        this.btnRunSelectToggle = this.root.querySelector('[data-a="run-select-toggle"]');
        this.runSelectAllWrap = this.root.querySelector('[data-r="run-select-all-wrap"]');
        this.runSelectAllCb = this.root.querySelector('[data-r="run-select-all-cb"]');

        this.populateTaskSelect(this.globalTask, this.taskTypeWidget?.value);
        this.syncNegativeFromWidget();
        this.syncOutputUIFromTimeline();
        bindImageBatchEvents(this);
    }

    renderImageBatchGroups() {
        renderImageBatchGroups(this);
    }

    normalizeImageBatchSegments() {
        normalizeImageBatchSegments(this);
    }

    syncNegativeFromWidget() {
        const v = this.negativePromptWidget?.value ?? "";
        if (this.globalNegative) this.globalNegative.value = v;
        if (this.segNegative) this.segNegative.value = v;
    }

    bindEvents() {
        const bind = (sel, fn) => {
            const el = this.root.querySelector(sel);
            if (!el) return;
            el.onclick = (e) => { stopDomEvent(e); fn(); };
        };
        bind('[data-a="video"]', () => this.pickVideoFile());
        bind('[data-a="video-append"]', () => this.pickAppendVideoFile());
        bind('[data-a="split"]', () => this.splitAtFrame(this.currentFrame));
        bind('[data-a="equal"]', () => this.equalSplit());
        bind('[data-a="run-select-toggle"]', () => this.toggleRunSelectMode());
        bind('[data-a="del"]', () => this.deleteSelectedSegment());
        bind('[data-a="mode-global"]', () => this.setEditMode("global"));
        bind('[data-a="mode-segment"]', () => this.setEditMode("segment"));
        bind('[data-a="play"]', () => this.togglePlay());
        bind('[data-a="loop"]', () => this.toggleLoop());
        bind('[data-a="frame-prev"]', () => this.stepFrame(-1));
        bind('[data-a="frame-next"]', () => this.stepFrame(1));
        bind('[data-a="zoom-in"]', () => this.adjustZoom(0.5));
        bind('[data-a="zoom-out"]', () => this.adjustZoom(-0.5));

        this.seekBar.oninput = () => {
            this.seekToFrame(+this.seekBar.value, { fromUi: true });
        };
        if (this.frameInputEl) {
            const applyFrameInput = () => {
                const total = this.getTotalFrames();
                if (total < 1) return;
                const raw = parseInt(this.frameInputEl.value, 10);
                if (!Number.isFinite(raw)) {
                    this.frameInputEl.value = String(this.currentFrame + 1);
                    return;
                }
                // UI is 1-based; internal currentFrame is 0-based.
                this.seekToFrame(raw - 1, { fromUi: true });
            };
            this.frameInputEl.addEventListener("keydown", (e) => {
                e.stopPropagation();
                if (e.key === "Enter") {
                    e.preventDefault();
                    applyFrameInput();
                    this.frameInputEl.blur();
                } else if (e.key === "Escape") {
                    e.preventDefault();
                    this.frameInputEl.value = String(this.currentFrame + 1);
                    this.frameInputEl.blur();
                }
            });
            this.frameInputEl.addEventListener("change", applyFrameInput);
            this.frameInputEl.addEventListener("focus", () => {
                if (this.isPlaying) this._stopPlay();
                this.frameInputEl.select();
            });
        }
        if (this.stageBadge) {
            this.stageBadge.title = "点击输入精确帧号";
            this.stageBadge.addEventListener("click", (e) => {
                e.stopPropagation();
                if (this.isPlaying) this._stopPlay();
                this.frameInputEl?.focus();
                this.frameInputEl?.select();
            });
        }
        this.zoomSlider.oninput = () => { this.zoom = +this.zoomSlider.value; this.applyZoomWidth(); this.scheduleRender(); };
        if (this.runSelectAllCb) {
            this.runSelectAllCb.onchange = (e) => {
                stopDomEvent(e);
                if (!this.isRunSelectEnabled()) return;
                this.setRunSelectionAll(this.runSelectAllCb.checked);
            };
        }
        this.globalTask.onchange = () => this.onGlobalField("taskType", this.globalTask.value);
        this.globalPrompt.oninput = () => this.onGlobalField("prompt", this.globalPrompt.value);
        if (this.continuousRefCb) {
            this.continuousRefCb.onchange = () => {
                this.timeline.global = this.timeline.global || { refs: [], referenceVideo: {} };
                this.timeline.global.continuousReference = !!this.continuousRefCb.checked;
                this.scheduleTimelineSync();
            };
        }
        this.segPrompt.oninput = () => this.onSegField("prompt", this.segPrompt.value);
        this.globalNegative.oninput = () => this.onNegativePrompt(this.globalNegative.value);
        this.segNegative.oninput = () => this.onNegativePrompt(this.segNegative.value);

        mountPromptImageMentions(this);

        this.outMode.onchange = () => this.onOutputField("mode", this.outMode.value);
        this.outLong.onchange = () => this.onOutputField("longEdge", +this.outLong.value);
        this.outW.onchange = () => this.onOutputField("width", +this.outW.value);
        this.outH.onchange = () => this.onOutputField("height", +this.outH.value);
        this.fpsInput.onchange = () => this.onFrameRateChanged(this.fpsInput.value);
        this.fpsInput.oninput = () => {
            clearTimeout(this._fpsInputTimer);
            this._fpsInputTimer = setTimeout(() => this.onFrameRateChanged(this.fpsInput.value), 350);
        };
        this.outMaxFrames.onchange = () => this.onOutputField("maxExportFrames", +this.outMaxFrames.value);
        this.outExportMode.onchange = () => this.onOutputField("exportMode", this.outExportMode.value);
        if (this.segmentContinuityCb) {
            this.segmentContinuityCb.onchange = () => {
                this.onOutputField("continuityEnabled", this.segmentContinuityCb.checked);
                this.updateSegmentContinuityUI();
            };
        }
        if (this.segmentContinuityOverlap) {
            const applyOverlap = () => this.onOutputField("continuityOverlapFrames", +this.segmentContinuityOverlap.value);
            this.segmentContinuityOverlap.onchange = applyOverlap;
            this.segmentContinuityOverlap.oninput = applyOverlap;
            this.segmentContinuityOverlap.addEventListener("keydown", (e) => e.stopPropagation());
            this.segmentContinuityOverlap.addEventListener("keyup", (e) => e.stopPropagation());
        }

        this.genGlobalImg?.addEventListener("click", (e) => { stopDomEvent(e); this.pickGenSrcImage(true); });
        this.genSegImg?.addEventListener("click", (e) => { stopDomEvent(e); this.pickGenSrcImage(false); });
        this.genDefaultFc?.addEventListener("change", () => this.onGenDefaultFcChange());
        this.genSegFc?.addEventListener("change", () => this.onGenSegFcChange());

        this.canvas.addEventListener("mousedown", (e) => this.onMouseDown(e));
        this.canvas.addEventListener("dblclick", (e) => this.addSplitAtMouse(e));
        this.canvas.addEventListener("contextmenu", (e) => { e.preventDefault(); this.addSplitAtMouse(e); });
        this._onMouseMove = (e) => this.onMouseMove(e);
        this._onMouseUp = () => this.onMouseUp();
        this._onCanvasHover = (e) => {
            if (this._drag || this.isPlaying) return;
            const { x, y } = this.getMousePos(e);
            const hit = this.hitTest(x, y);
            this.canvas.classList.remove("bd-grab");
            if (hit?.type === "run-check") {
                this.canvas.style.cursor = "pointer";
            } else if (hit?.type === "segment" && this.timeline.segments.length >= 2) {
                this.canvas.classList.add("bd-grab");
                this.canvas.style.cursor = "";
            } else {
                this.canvas.style.cursor = "";
            }
        };
        window.addEventListener("mousemove", this._onMouseMove);
        window.addEventListener("mouseup", this._onMouseUp);
        this.canvas.addEventListener("mousemove", this._onCanvasHover);
        this.canvas.addEventListener("mouseleave", () => {
            this.canvas.classList.remove("bd-grab");
            this.canvas.style.cursor = "";
        });

        this.root.addEventListener("mouseenter", () => { this._isHovering = true; });
        this.root.addEventListener("mouseleave", () => { this._isHovering = false; });
        this._onKeyDown = (e) => {
            if (!this._isHovering) return;
            const tag = document.activeElement?.tagName;
            if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
            if ((e.key === "Delete" || e.key === "Backspace") && this.timeline.segments.length >= 1) {
                this.deleteSelectedSegment(); e.preventDefault();
            } else if (e.code === "Space") {
                this.togglePlay(); e.preventDefault();
            } else if (e.key === "ArrowLeft") {
                this.stepFrame(e.shiftKey ? -10 : -1);
                e.preventDefault();
            } else if (e.key === "ArrowRight") {
                this.stepFrame(e.shiftKey ? 10 : 1);
                e.preventDefault();
            }
        };
        window.addEventListener("keydown", this._onKeyDown, true);

        this.root.addEventListener("dragover", (e) => e.preventDefault());
        this.root.addEventListener("drop", (e) => {
            e.preventDefault();
            const f = e.dataTransfer.files?.[0];
            if (f?.type.startsWith("video/")) this.loadVideoFile(f);
            else if (f?.type.startsWith("image/")) {
                if (this.isImageBatch?.() && e.target.closest?.(".bd-batch-ref")) return;
                if (this.isImageBatch?.()) return;
                this.addRefFromFile(f, this.getRefTarget());
            }
        });
    }

    destroy() {
        clearTimeout(this._syncTimer);
        cancelAnimationFrame(this._resizeRaf);
        cancelAnimationFrame(this._playRaf);
        this._resizeObserver?.disconnect();
        this._closeBdModal();
        this._previewVideo?.remove();
        this._previewVideo = null;
        window.removeEventListener("mousemove", this._onMouseMove);
        window.removeEventListener("mouseup", this._onMouseUp);
        this.canvas?.removeEventListener("mousemove", this._onCanvasHover);
        this.canvas?.classList.remove("bd-grab", "bd-grabbing");
        window.removeEventListener("keydown", this._onKeyDown, true);
    }

    widget(name) { return this.node.widgets?.find((w) => w.name === name); }

    hasVideo() {
        const v = this.timeline?.video || {};
        return !!(this.getVideoClips().length || v.videoFile || this._legacyFrames.length || v.frames?.length);
    }

    getVideoClips() {
        if (this.timeline.videoClips?.length) return this.timeline.videoClips;
        const v = this.timeline?.video || {};
        if (v.videoFile || v.fileName) {
            return [{
                id: v.id || "c0",
                fileName: v.fileName || "",
                videoFile: v.videoFile || v.fileName || "",
                subfolder: v.subfolder || "",
                type: v.type || "input",
                width: v.width || 0,
                height: v.height || 0,
                duration: v.duration || 0,
                nativeFps: v.nativeFps || v.native_fps || 0,
                nativeFrameCount: v.nativeFrameCount || v.native_frame_count || 0,
                sourceFrameCount: v.sourceFrameCount || this.getFrameMap().length,
                storageWidth: v.storageWidth,
                storageHeight: v.storageHeight,
            }];
        }
        return [];
    }

    _ensureVideoClipsArray() {
        if (!this.timeline.videoClips?.length) {
            const v = this.timeline?.video || {};
            if (v.videoFile || v.fileName) {
                this.timeline.videoClips = [{
                    id: v.id || uid(),
                    fileName: v.fileName || "",
                    videoFile: v.videoFile || v.fileName || "",
                    subfolder: v.subfolder || "",
                    type: v.type || "input",
                    width: v.width || 0,
                    height: v.height || 0,
                    duration: v.duration || 0,
                    nativeFps: v.nativeFps || v.native_fps || 0,
                    nativeFrameCount: v.nativeFrameCount || v.native_frame_count || 0,
                    sourceFrameCount: v.sourceFrameCount || this.getFrameMap().length,
                    storageWidth: v.storageWidth,
                    storageHeight: v.storageHeight,
                }];
            } else {
                this.timeline.videoClips = [];
            }
        }
    }

    getClipViewUrl(clipIndex) {
        const clip = this.getVideoClips()[clipIndex];
        if (!clip?.videoFile) return "";
        return inputViewUrl(clip.videoFile, clip.type || "input");
    }

    getRefVideoTarget() {
        if (this.isGlobalMode()) {
            this.timeline.global = this.timeline.global || { refs: [], referenceVideo: {} };
            if (!this.timeline.global.referenceVideo) this.timeline.global.referenceVideo = {};
            return this.timeline.global;
        }
        const seg = this.timeline.segments[this.selectedIndex];
        if (seg) {
            if (!seg.referenceVideo) seg.referenceVideo = {};
            return seg;
        }
        this.timeline.global = this.timeline.global || { refs: [], referenceVideo: {} };
        return this.timeline.global;
    }

    getReferenceVideoViewUrl(ref) {
        const block = ref || {};
        const file = block.videoFile || block.fileName;
        if (!file) return "";
        return inputViewUrl(file, block.type || "input");
    }

    _stopRefVideoPreviews(onlyEls = null) {
        const targets = onlyEls || [this.globalRefVideo, this.segRefVideo];
        for (const el of targets) {
            const v = el?.querySelector("video");
            if (v) {
                v.pause();
                v.removeAttribute("src");
                v.load();
            }
        }
    }

    getTaskKey() {
        return resolveTaskKey(
            this.globalTask?.value
            || this.timeline.global?.taskType
            || this.taskTypeWidget?.value,
        );
    }

    getRunnableSegmentCount() {
        return this.timeline.segments?.length || 0;
    }

    supportsRunSelect() {
        const n = this.getRunnableSegmentCount();
        if (n < 2) return false;
        const mode = this.getDirectorMode();
        if (mode === "video") return true;
        if (this.isImageBatch()) return isPromptBatchTask(this.getTaskKey());
        return false;
    }

    getRunProgressSegmentTotal() {
        const n = this.getRunnableSegmentCount();
        if (!this.isRunSelectEnabled() || n < 2) return Math.max(n, 1);
        const count = (this.timeline.runSelection || []).length;
        return count > 0 ? count : Math.max(n, 1);
    }

    isRunSelectEnabled() {
        return !!this.timeline.runSelectEnabled;
    }

    normalizeRunSelection() {
        const n = this.getRunnableSegmentCount();
        if (!this.isRunSelectEnabled() || n < 1) return;
        this.timeline.runSelection = [...new Set(
            (this.timeline.runSelection || []).filter((i) => i >= 0 && i < n),
        )].sort((a, b) => a - b);
    }

    isSegmentRunEnabled(index) {
        if (!this.isRunSelectEnabled()) return true;
        return (this.timeline.runSelection || []).includes(index);
    }

    toggleSegmentRun(index) {
        if (!this.isRunSelectEnabled()) return;
        const n = this.getRunnableSegmentCount();
        if (index < 0 || index >= n) return;
        const sel = new Set(this.timeline.runSelection || []);
        if (sel.has(index)) sel.delete(index);
        else sel.add(index);
        this.timeline.runSelection = [...sel].sort((a, b) => a - b);
        this.updateRunSelectUI();
        this.commit(false, { syncTimeline: true });
        if (this.isImageBatch()) this.renderImageBatchGroups();
        else this.scheduleRender();
    }

    toggleRunSelectMode() {
        if (!this.supportsRunSelect()) return;
        const n = this.getRunnableSegmentCount();
        this.timeline.runSelectEnabled = !this.timeline.runSelectEnabled;
        if (this.timeline.runSelectEnabled) {
            if (!(this.timeline.runSelection || []).length) {
                this.timeline.runSelection = Array.from({ length: n }, (_, i) => i);
            } else {
                this.normalizeRunSelection();
            }
        }
        this.updateRunSelectUI();
        this.commit(false, { syncTimeline: true });
        if (this.isImageBatch()) this.renderImageBatchGroups();
        else this.scheduleRender();
    }

    setRunSelectionAll(on) {
        if (!this.isRunSelectEnabled()) return;
        const n = this.getRunnableSegmentCount();
        this.timeline.runSelection = on ? Array.from({ length: n }, (_, i) => i) : [];
        this.updateRunSelectUI();
        this.commit(false, { syncTimeline: true });
        if (this.isImageBatch()) this.renderImageBatchGroups();
        else this.scheduleRender();
    }

    updateRunSelectUI() {
        const n = this.getRunnableSegmentCount();
        const canRunSelect = this.supportsRunSelect();
        const enabled = this.isRunSelectEnabled() && canRunSelect;
        const useBatchBar = this.isImageBatch() && canRunSelect;
        this.btnRunSelectToggle?.classList.toggle("active", enabled);
        this.btnRunSelectToggle?.classList.toggle("bd-btn-run-select", true);
        this.btnRunSelectToggle?.classList.toggle("hidden", !canRunSelect || useBatchBar);
        this.batchRunSelectBtn?.classList.toggle("active", enabled);
        this.batchRunSelectBtn?.classList.toggle("hidden", !useBatchBar);
        this.runSelectAllWrap?.classList.toggle("hidden", !enabled || useBatchBar);
        this.batchRunSelectAllWrap?.classList.toggle("hidden", !enabled || !useBatchBar);
        // Keep the chip hidden while a run is active — otherwise commit/sync
        // re-shows it on top of the green progress title.
        const running = !!this.runStatusEl?.classList.contains("active");
        this.runSelectBar?.classList.toggle("hidden", !enabled || running);
        if (!canRunSelect) return;
        this.normalizeRunSelection();
        const count = (this.timeline.runSelection || []).length;
        const syncAllCb = (cb) => {
            if (!cb) return;
            cb.checked = count >= n && n > 0;
            cb.indeterminate = count > 0 && count < n;
        };
        syncAllCb(this.runSelectAllCb);
        syncAllCb(this.batchRunSelectAllCb);
        const label = this.isImageBatch() ? "组" : "段";
        if (!this.runSelectSummary) return;
        if (!count) {
            this.runSelectSummary.textContent = `未勾选任何${label}（无法运行）`;
            this.runSelectSummary.style.color = "#f88";
        } else if (count >= n) {
            this.runSelectSummary.textContent = `将运行全部 ${n} ${label}`;
            this.runSelectSummary.style.color = "#aaa";
        } else {
            const nums = (this.timeline.runSelection || []).map((i) => i + 1).join(", ");
            this.runSelectSummary.textContent = count === 1
                ? `将运行 1 ${label}（#${nums}）`
                : `将运行 ${count} ${label}（#${nums}）`;
            this.runSelectSummary.style.color = "#4fff8f";
        }
    }

    _runSelectionPayload() {
        if (!this.timeline.runSelectEnabled) {
            return { runSelectEnabled: false, runSelection: [] };
        }
        this.normalizeRunSelection();
        return {
            runSelectEnabled: true,
            runSelection: [...(this.timeline.runSelection || [])],
        };
    }

    getDirectorMode() {
        return getDirectorMode(this.globalTask?.value || this.taskTypeWidget?.value);
    }

    isGenMode() {
        const mode = this.getDirectorMode();
        return mode !== "video" && mode !== "prompt_batch";
    }

    isImageBatch() {
        const mode = this.getDirectorMode();
        return mode === "prompt_batch" || mode === "image_batch";
    }

    isGenBlank() {
        return this.getDirectorMode() === "gen_blank";
    }

    isGenImage() {
        return this.getDirectorMode() === "gen_image";
    }

    onTaskTypeChanged(value) {
        this.onGlobalField("taskType", value);
        this._promptEnhancer?.onTaskTypeChanged?.();
    }

    /** Snapshot v2v/rv2v workspace before switching to t2i / batch / gen modes. */
    _stashVideoWorkspace() {
        const video = this.timeline.video || {};
        const clips = this.timeline.videoClips || [];
        const hasVid = !!(
            clips.length
            || video.videoFile
            || video.fileName
            || this._legacyFrames?.length
            || video.frames?.length
        );
        const segs = this.timeline.segments || [];
        if (!hasVid && !segs.length) return;

        this.timeline.videoWorkspace = {
            segments: JSON.parse(JSON.stringify(segs)),
            selectedIndex: this.selectedIndex,
            currentFrame: this.currentFrame,
            editMode: this.timeline.editMode || "global",
            runSelectEnabled: !!this.timeline.runSelectEnabled,
            runSelection: Array.isArray(this.timeline.runSelection)
                ? [...this.timeline.runSelection]
                : undefined,
            video: JSON.parse(JSON.stringify(video)),
            videoClips: JSON.parse(JSON.stringify(clips)),
            totalFrames: this.timeline.totalFrames ?? this.getTotalFrames(),
            frameRate: this.timeline.frameRate ?? this.getFrameRate(),
            legacyFrames: this._legacyFrames?.length ? [...this._legacyFrames] : [],
            storageWidth: this._storageWidth || 0,
            storageHeight: this._storageHeight || 0,
        };
    }

    /** Restore v2v/rv2v workspace after returning from t2i / batch / gen. */
    _restoreVideoWorkspace() {
        const ws = this.timeline.videoWorkspace;
        if (!ws || typeof ws !== "object") {
            this.normalizeSegments();
            this.restoreVideoFromTimeline();
            this.updateStageVisibility();
            return false;
        }

        if (ws.video && typeof ws.video === "object") {
            this.timeline.video = JSON.parse(JSON.stringify(ws.video));
        }
        if (Array.isArray(ws.videoClips)) {
            this.timeline.videoClips = JSON.parse(JSON.stringify(ws.videoClips));
        }
        if (Array.isArray(ws.segments) && ws.segments.length) {
            this.timeline.segments = JSON.parse(JSON.stringify(ws.segments));
        }
        if (ws.totalFrames != null) this.timeline.totalFrames = ws.totalFrames;
        if (ws.frameRate != null) this.timeline.frameRate = ws.frameRate;
        if (ws.editMode) this.timeline.editMode = ws.editMode;
        if (ws.runSelectEnabled != null) this.timeline.runSelectEnabled = !!ws.runSelectEnabled;
        if (Array.isArray(ws.runSelection)) this.timeline.runSelection = [...ws.runSelection];

        this.selectedIndex = clamp(
            ws.selectedIndex ?? 0,
            0,
            Math.max(0, (this.timeline.segments?.length || 1) - 1),
        );
        this.currentFrame = Math.max(0, ws.currentFrame ?? 0);
        if (Array.isArray(ws.legacyFrames) && ws.legacyFrames.length) {
            this._legacyFrames = [...ws.legacyFrames];
        }
        if (ws.storageWidth) this._storageWidth = ws.storageWidth;
        if (ws.storageHeight) this._storageHeight = ws.storageHeight;

        this.normalizeSegments();
        this.restoreVideoFromTimeline();
        const total = this.getTotalFrames();
        this.currentFrame = clamp(this.currentFrame, 0, Math.max(0, total - 1));
        if (this.seekBar) {
            this.seekBar.max = Math.max(0, total - 1);
            this.seekBar.value = this.currentFrame;
        }
        if (this.totalFramesWidget) this.totalFramesWidget.value = total;
        this.updateVideoNameLabel();
        this.updateStageVisibility();
        // Live state is now in timeline.*; drop the snapshot so later edits
        // cannot be overwritten by a stale workspace on the next mode switch.
        this.timeline.videoWorkspace = null;
        return true;
    }

    ensureGenTimeline() {
        const key = this.getTaskKey();
        this.timeline.gen = this.timeline.gen || {};
        const defFc = defaultFrameCount(key);
        if (!this.timeline.segments?.length || !sumFrameCounts(this.timeline.segments)) {
            this.timeline.segments = [{
                id: uid(), start: 0, length: defFc, frameCount: defFc,
                prompt: "", taskType: "", refs: [], genImage: { imageFile: "" },
            }];
        }
        for (const seg of this.timeline.segments) {
            if (seg.frameCount == null) seg.frameCount = seg.length ?? defFc;
            seg.genImage = seg.genImage || { imageFile: seg.imageFile || "" };
        }
        this.timeline.global = this.timeline.global || { refs: [] };
        this.timeline.global.genImage = this.timeline.global.genImage || { imageFile: "" };
        if (this.isGenBlank()) {
            this.timeline.output = this.timeline.output || {};
            this.timeline.output.mode = "fixed";
        }
        this.normalizeGenSegments();
    }

    normalizeGenSegments() {
        const key = this.getTaskKey();
        const minFc = minFrameCount(key);
        let start = 0;
        const fixed = [];
        for (const seg of [...this.timeline.segments]) {
            let fc = clamp(parseInt(seg.frameCount ?? seg.length, 10) || defaultFrameCount(key), minFc, MAX_GEN_FRAMES);
            fixed.push({
                ...seg,
                start,
                length: fc,
                frameCount: fc,
                refs: seg.refs || [],
                genImage: seg.genImage || { imageFile: "" },
            });
            start += fc;
        }
        if (!fixed.length) {
            const fc = defaultFrameCount(key);
            fixed.push({
                id: uid(), start: 0, length: fc, frameCount: fc,
                prompt: "", taskType: "", refs: [], genImage: { imageFile: "" },
            });
        }
        this.timeline.segments = fixed;
        this.timeline.totalFrames = start || fixed[0].frameCount;
        this.selectedIndex = clamp(this.selectedIndex, 0, fixed.length - 1);
    }

    updateReferenceImageVisibility({ hideTimeline = false, seg = null } = {}) {
        const globalKey = this.getTaskKey();
        const showGlobalRefs = !hideTimeline && taskUsesReferenceImages(globalKey);
        const showGlobalRefVideo = !hideTimeline && taskUsesReferenceVideo(globalKey);

        this.globalRefsCol?.classList.toggle("hidden", !showGlobalRefs && !showGlobalRefVideo);
        this.globalRefsImagesWrap?.classList.toggle("hidden", !showGlobalRefs);
        this.globalRefVideoCol?.classList.toggle("hidden", !showGlobalRefVideo);
        if (this.globalPanelTitle) {
            if (showGlobalRefVideo) {
                this.globalPanelTitle.textContent = "全局提示词 & 参考视频";
            } else if (showGlobalRefs) {
                this.globalPanelTitle.textContent = "全局提示词 & 参考图 (image0–4)";
            } else {
                this.globalPanelTitle.textContent = "全局提示词";
            }
        }

        const segKey = resolveTaskKey(
            seg?.taskType || this.timeline.global?.taskType || this.globalTask?.value || globalKey,
        );
        const showSegRefs = !hideTimeline && taskUsesReferenceImages(segKey);
        const showSegRefVideo = !hideTimeline && taskUsesReferenceVideo(segKey);
        this.segRefsCol?.classList.toggle("hidden", !showSegRefs && !showSegRefVideo);
        this.segRefsImagesWrap?.classList.toggle("hidden", !showSegRefs);
        this.segRefVideoCol?.classList.toggle("hidden", !showSegRefVideo);
        const showContinuousRef = !hideTimeline
            && this.isGlobalMode()
            && showGlobalRefVideo
            && globalKey === "ads2v";
        this.continuousRefWrap?.classList.toggle("hidden", !showContinuousRef);
        if (this.continuousRefCb) {
            this.continuousRefCb.checked = !!this.timeline.global?.continuousReference;
        }
        if (showGlobalRefVideo || showSegRefVideo) this.renderRefVideoSlot();
    }

    applyTaskLayout(prevMode) {
        const mode = this.getDirectorMode();
        const prev = prevMode || "video";
        const wasBatch = prev === "prompt_batch" || prev === "image_batch";
        const isBatch = mode === "prompt_batch";
        const wasGen = prev !== "video" && prev !== "prompt_batch" && prev !== "image_batch";
        const isGen = mode !== "video" && mode !== "prompt_batch";

        if (this.isPlaying) this._stopPlay();

        if (isBatch) {
            if (!wasBatch) {
                // Keep v2v/rv2v video + segments so switching back can restore them.
                if (prev === "video") this._stashVideoWorkspace();
                const keep = this.timeline.global?.prompt
                    || this.timeline.segments?.[0]?.prompt
                    || "";
                this.timeline.segments = [newBatchSegment({
                    prompt: keep,
                    negativePrompt: this.negativePromptWidget?.value || "bad video",
                })];
            }
            ensureImageBatchTimeline(this);
        } else if (isGen) {
            if (!wasGen && !wasBatch) {
                if (prev === "video") this._stashVideoWorkspace();
                const key = this.getTaskKey();
                const defFc = defaultFrameCount(key);
                const keepPrompt = this.timeline.global?.prompt || "";
                this.timeline.segments = [{
                    id: uid(),
                    start: 0,
                    length: defFc,
                    frameCount: defFc,
                    prompt: keepPrompt,
                    taskType: "",
                    refs: [],
                    genImage: { imageFile: "" },
                }];
            }
            this.ensureGenTimeline();
        } else if (prev !== "video") {
            this.timeline.timelineMode = "video";
            // Prefer restoring the stashed v2v/rv2v session (segments + thumbs source).
            if (!this._restoreVideoWorkspace()) {
                this.normalizeSegments();
            }
        }
        this.timeline.timelineMode = mode;
        this._directorMode = mode;

        const hideTimeline = isBatch || isGen;
        const taskKey = this.getTaskKey();
        const showBatchExport = isBatch && isVideoBatchTask(taskKey);
        this.btnVideo?.classList.toggle("hidden", hideTimeline);
        this.btnVideoAppend?.classList.toggle("hidden", hideTimeline);
        this.controlsBar?.classList.toggle("hidden", hideTimeline || isBatch);
        this.boundsEl?.classList.toggle("hidden", hideTimeline || isBatch);
        this.timecodeEl?.classList.toggle("hidden", hideTimeline || isBatch);
        this.viewport?.classList.toggle("hidden", isBatch);
        this.updateStageVisibility();
        this.root.querySelector(".bd-split")?.classList.toggle("hidden", isBatch);
        this.batchPanel?.classList.toggle("hidden", !isBatch);
        setToolbarDisabledForBatch(this, isBatch);

        this.updateReferenceImageVisibility({ hideTimeline });

        const showGenImg = mode === "gen_image";
        this.genGlobalImg?.classList.toggle("hidden", !showGenImg || !this.isGlobalMode());
        this.genSegImg?.classList.toggle("hidden", !showGenImg || this.isGlobalMode());
        this.genGlobalFcRow?.classList.toggle("hidden", !isGen || !this.isGlobalMode());
        this.genSegFcRow?.classList.toggle("hidden", !isGen || this.isGlobalMode());

        if (mode === "gen_blank" || (isBatch && imageBatchRequiresFixedOutput(taskKey))) {
            this.timeline.output = this.timeline.output || {};
            this.timeline.output.mode = "fixed";
            if (isBatch && !isVideoBatchTask(taskKey)) this.timeline.output.exportMode = "all";
            if (this.outMode) {
                this.outMode.value = "fixed";
                this.outMode.disabled = true;
            }
            this.outLongWrap && (this.outLongWrap.style.display = "none");
            this.outFixedWrap?.classList.remove("hidden");
        } else if (isBatch && taskKey === "i2i") {
            this.timeline.output = this.timeline.output || {};
            const modeVal = String(this.timeline.output.mode || "long_edge").toLowerCase();
            this.timeline.output.mode = modeVal === "fixed" ? "fixed" : "long_edge";
            this.timeline.output.exportMode = "all";
            if (this.outMode) {
                this.outMode.disabled = false;
                this.outMode.value = this.timeline.output.mode;
            }
            this.updateOutputModeUI();
        } else if (isBatch && isVideoBatchTask(taskKey)) {
            this.timeline.output = this.timeline.output || {};
            if (this.outMode) {
                this.outMode.disabled = false;
                this.updateOutputModeUI();
            }
        } else if (this.outMode) {
            this.outMode.disabled = false;
            this.updateOutputModeUI();
        }

        if (this.outHint) {
            this.outHint.classList.toggle("hidden", !isGen && !isBatch);
            this.outHint.textContent = (isGen || isBatch) ? genLayoutHint(this.getTaskKey()) : "";
        }
        if (this.outExportMode) {
            this.outExportMode.disabled = isBatch && !showBatchExport;
            this.outExportMode.classList.toggle("hidden", isBatch && !showBatchExport);
            this.outExportMode.previousElementSibling?.classList.toggle("hidden", isBatch && !showBatchExport);
        }
        if (this.outMaxFrames) {
            this.outMaxFrames.disabled = isBatch && !showBatchExport;
            this.outMaxFrames.classList.toggle("hidden", isBatch && !showBatchExport);
            this.outMaxFrames.previousElementSibling?.classList.toggle("hidden", isBatch && !showBatchExport);
        }

        if ((isGen || isBatch) && prev === "video") {
            this.currentFrame = 0;
        }
        this.updateVideoNameLabel();
        if (isBatch) {
            this.timeline.editMode = "segment";
            this.renderImageBatchGroups();
        } else {
            this.updateModeUI();
            this.updateSelectionUI();
        }
        this.updateDomWidgetHeight();
        this.syncOutputUIFromTimeline();
        this.seekBar.max = Math.max(0, this.getTotalFrames() - 1);
        if (!isBatch) this.scheduleRender();
        this.scheduleTimelineSync();
        this.updateRunSelectUI();
    }

    renderGenSrcSlot(el, imageFile, label) {
        if (!el) return;
        el.classList.toggle("has-img", !!imageFile);
        if (imageFile) {
            el.innerHTML = `<img src="${refViewUrl(imageFile)}" alt="">`;
        } else {
            el.textContent = label;
        }
    }

    _paintRefVideoSlot(el, nameEl, refBlock) {
        if (!el) return;
        const ref = refBlock || {};
        const has = !!(ref.videoFile || ref.fileName);
        el.classList.toggle("has-img", false);
        el.classList.toggle("has-video", has);
        if (nameEl) {
            if (has) {
                const dur = ref.duration > 0 ? ` · ${ref.duration.toFixed(2)}s` : "";
                const fps = ref.nativeFps > 0 ? ` · ${Math.round(ref.nativeFps)}fps` : "";
                const dim = ref.width && ref.height ? ` · ${ref.width}×${ref.height}` : "";
                nameEl.textContent = `${ref.fileName || ref.videoFile || ""}${dim}${dur}${fps}`;
            } else {
                nameEl.textContent = "";
            }
        }
        if (!has) {
            el.innerHTML = "";
            el.textContent = "点击上传参考视频";
            el.onclick = () => this.pickReferenceVideoFile();
            return;
        }
        const viewUrl = this.getReferenceVideoViewUrl(ref);
        el.innerHTML = `
            <video class="bd-ref-video-preview" muted playsinline preload="metadata" controls></video>
            <button type="button" class="bd-ref-replace" title="更换参考视频">更换</button>
            <span class="x" title="移除参考视频">×</span>`;
        el.onclick = null;
        const video = el.querySelector("video");
        if (video && viewUrl) {
            video.src = viewUrl;
            video.addEventListener("click", (e) => e.stopPropagation());
            video.addEventListener("dblclick", (e) => {
                e.stopPropagation();
                if (video.paused) video.play().catch(() => {});
                else video.pause();
            });
        }
        const replaceBtn = el.querySelector(".bd-ref-replace");
        if (replaceBtn) {
            replaceBtn.onclick = (e) => {
                e.stopPropagation();
                this.pickReferenceVideoFile();
            };
        }
        const removeBtn = el.querySelector(".x");
        if (removeBtn) {
            removeBtn.onclick = (e) => {
                e.stopPropagation();
                this.clearReferenceVideo();
            };
        }
    }

    renderRefVideoSlot() {
        if (this.isGlobalMode()) {
            this._stopRefVideoPreviews([this.segRefVideo]);
            this._paintRefVideoSlot(
                this.globalRefVideo,
                this.globalRefVideoNameEl,
                this.timeline.global?.referenceVideo || {},
            );
        } else {
            this._stopRefVideoPreviews([this.globalRefVideo]);
            const seg = this.timeline.segments[this.selectedIndex];
            this._paintRefVideoSlot(this.segRefVideo, this.segRefVideoNameEl, seg?.referenceVideo || {});
        }
    }

    _activeRefVideoTaskKey() {
        if (this.isGlobalMode()) return this.getTaskKey();
        const seg = this.timeline.segments[this.selectedIndex];
        return resolveTaskKey(seg?.taskType || this.timeline.global?.taskType || this.getTaskKey());
    }

    pickReferenceVideoFile() {
        if (!taskUsesReferenceVideo(this._activeRefVideoTaskKey())) return;
        const input = document.createElement("input");
        input.type = "file";
        input.accept = "video/*";
        input.onchange = () => {
            if (input.files?.[0]) this.loadReferenceVideoFile(input.files[0]);
        };
        input.click();
    }

    clearReferenceVideo() {
        const target = this.getRefVideoTarget();
        this._stopRefVideoPreviews();
        target.referenceVideo = {};
        this.renderRefVideoSlot();
        this.commit();
    }

    async loadReferenceVideoFile(file) {
        const slotEl = this.isGlobalMode() ? this.globalRefVideo : this.segRefVideo;
        const nameEl = this.isGlobalMode() ? this.globalRefVideoNameEl : this.segRefVideoNameEl;
        const status = `上传中: ${file.name}…`;
        if (slotEl) {
            slotEl.classList.remove("has-img", "has-video");
            slotEl.textContent = status;
        }
        if (nameEl) nameEl.textContent = status;
        try {
            const uploaded = await uploadToInputSmart(file, (frac, cur, total) => {
                const pct = Math.round(frac * 100);
                const mode = file.size > COMFY_UPLOAD_SOFT_LIMIT ? "分块" : "上传";
                if (nameEl) nameEl.textContent = `${mode}参考视频: ${file.name} (${cur}/${total}, ${pct}%)…`;
            });
            const relPath = videoRelativePath(uploaded);
            const prep = await this._prepareVideoFrames({
                fileName: file.name,
                relPath,
                subfolder: uploaded.subfolder || "",
                type: uploaded.type || "input",
                statusPrefix: "解析参考视频",
                syncNativeFps: false,
            });
            this.getRefVideoTarget().referenceVideo = this._buildClipRecord(prep);
            this.renderRefVideoSlot();
            this.commit(false, { syncTimeline: true });
        } catch (err) {
            console.error("[BerniniDirector] reference video load failed:", err);
            if (nameEl) nameEl.textContent = `参考视频加载失败: ${formatUploadError(err)}`;
            this.renderRefVideoSlot();
        }
    }

    pickGenSrcImage(isGlobal) {
        if (!this.isGenImage()) return;
        const input = document.createElement("input");
        input.type = "file";
        input.accept = "image/*";
        input.onchange = async () => {
            const file = input.files?.[0];
            if (!file) return;
            try {
                const uploaded = await uploadToInput(file);
                const relPath = videoRelativePath(uploaded);
                if (isGlobal) {
                    this.timeline.global = this.timeline.global || { refs: [] };
                    this.timeline.global.genImage = { imageFile: relPath };
                } else {
                    const seg = this.timeline.segments[this.selectedIndex];
                    if (seg) {
                        seg.genImage = { imageFile: relPath };
                        seg.imageFile = relPath;
                    }
                }
                this.commit();
            } catch (err) {
                console.error("[BerniniDirector] gen image upload failed:", err);
            }
        };
        input.click();
    }

    onGenDefaultFcChange() {
        const fc = clamp(parseInt(this.genDefaultFc?.value, 10) || 1, minFrameCount(this.getTaskKey()), MAX_GEN_FRAMES);
        if (this.genDefaultFc) this.genDefaultFc.value = fc;
        this.timeline.gen = this.timeline.gen || {};
        this.timeline.gen.defaultFrameCount = fc;
        if (this.timeline.segments.length === 1) {
            this.timeline.segments[0].frameCount = fc;
            this.timeline.segments[0].length = fc;
        }
        this.commit();
    }

    onGenSegFcChange() {
        const seg = this.timeline.segments[this.selectedIndex];
        if (!seg) return;
        const minFc = minFrameCount(this.getTaskKey());
        seg.frameCount = clamp(parseInt(this.genSegFc?.value, 10) || minFc, minFc, MAX_GEN_FRAMES);
        if (this.genSegFc) this.genSegFc.value = seg.frameCount;
        this.commit();
    }

    genSplitAtFrame(frame) {
        const total = this.getTotalFrames();
        const minFc = minFrameCount(this.getTaskKey());
        if (frame <= minFc || frame >= total - minFc) return;
        const newSegs = [];
        let cursor = 0;
        for (const seg of this.timeline.segments) {
            const fc = seg.frameCount ?? seg.length;
            const end = cursor + fc;
            if (frame > cursor && frame < end) {
                const left = frame - cursor;
                const right = end - frame;
                newSegs.push({ ...seg, frameCount: left, length: left });
                newSegs.push({
                    id: uid(), start: frame, frameCount: right, length: right,
                    prompt: "", taskType: "", refs: [], genImage: { imageFile: "" },
                });
            } else {
                newSegs.push({ ...seg });
            }
            cursor = end;
        }
        this.timeline.segments = newSegs;
        this.commit();
    }

    genEqualSplit() {
        const n = parseInt(this.equalCountInput?.value || "2", 10);
        if (!n || n < 2) return;
        const total = this.getTotalFrames();
        const minFc = minFrameCount(this.getTaskKey());
        const count = clamp(n, 2, Math.max(2, Math.floor(total / minFc)));
        const base = Math.floor(total / count);
        let rem = total - base * count;
        this.timeline.segments = Array.from({ length: count }, () => {
            const fc = base + (rem > 0 ? 1 : 0);
            if (rem > 0) rem -= 1;
            return {
                id: uid(), frameCount: fc, length: fc, prompt: "", taskType: "", refs: [],
                genImage: { imageFile: "" },
            };
        });
        this.commit();
    }

    genDeleteSelectedSegment() {
        if (this.timeline.segments.length <= 1) return;
        this.timeline.segments.splice(this.selectedIndex, 1);
        this.selectedIndex = clamp(this.selectedIndex, 0, this.timeline.segments.length - 1);
        this.commit();
    }

    updateVideoNameLabel() {
        if (this.isImageBatch()) {
            const n = this.timeline.segments?.length || 0;
            const key = this.getTaskKey();
            if (isVideoBatchTask(key)) {
                const total = this.getTotalFrames();
                const exp = key === "i2v" ? " · 实验性" : "";
                this.videoNameEl.textContent = total
                    ? `${key} · ${n} 组提示词 · ${total}f 视频${exp}`
                    : `${key} · ${n} 组提示词 · 视频输出${exp}`;
            } else {
                this.videoNameEl.textContent = `${key} · ${n} 组提示词 · 单帧输出`;
            }
            return;
        }
        if (this.isGenMode()) {
            const total = this.getTotalFrames();
            const key = this.getTaskKey();
            if (this.isGenBlank()) {
                this.videoNameEl.textContent = total ? `空白画布 · ${total}f` : "空白画布 · 请设置片段帧数";
            } else {
                this.videoNameEl.textContent = total ? `${key} · ${total}f` : `${key} · 请上传源图片`;
            }
            return;
        }
        const clips = this.getVideoClips();
        const total = this.getTotalFrames();
        if (!clips.length || !total) {
            this.videoNameEl.textContent = "未上传视频";
            return;
        }
        if (clips.length === 1) {
            const c = clips[0];
            const dim = c.storageWidth && c.storageHeight
                ? ` · ${c.storageWidth}×${c.storageHeight}`
                : (this._storageWidth && this._storageHeight ? ` · ${this._storageWidth}×${this._storageHeight}` : "");
            const nativeHint = c.nativeFps > 0 ? ` · 源${formatProbeFps(c.nativeFps)}fps` : "";
            const tlFps = this.getFrameRate();
            const dur = this.getTimelineDurationSec().toFixed(2);
            this.videoNameEl.textContent = `${c.fileName || c.videoFile} (${total}f · 时间轴${formatProbeFps(tlFps)}fps · ${dur}s${nativeHint}${dim})`;
            return;
        }
        const tlFps = this.getFrameRate();
        const dur = this.getTimelineDurationSec().toFixed(2);
        this.videoNameEl.textContent = `${clips.length} 段视频 · 共 ${total} 帧 · 时间轴${formatProbeFps(tlFps)}fps · ${dur}s`;
    }

    getFrameMapEntry(logicalFrame) {
        const map = this.getFrameMap();
        if (map.length) return normalizeFrameMapEntry(map[clamp(logicalFrame, 0, map.length - 1)]);
        return { clip: 0, frame: logicalToSourceFrame(logicalFrame, this.timeline.video || {}) };
    }

    getSegmentClipIndex(seg) {
        return this.getFrameMapEntry(seg.start).clip;
    }

    getClipBoundaries() {
        const map = this.getFrameMap();
        const boundaries = [];
        for (let i = 1; i < map.length; i++) {
            const a = normalizeFrameMapEntry(map[i - 1]);
            const b = normalizeFrameMapEntry(map[i]);
            if (b.clip !== a.clip) boundaries.push(i);
        }
        return boundaries;
    }

    _segmentMetaAtFrame(frame) {
        const segs = [...this.timeline.segments].sort((a, b) => a.start - b.start);
        for (const seg of segs) {
            if (frame >= seg.start && frame < seg.start + seg.length) {
                return {
                    prompt: seg.prompt || "",
                    taskType: seg.taskType || "",
                    refs: seg.refs ? JSON.parse(JSON.stringify(seg.refs)) : [],
                };
            }
        }
        const last = segs[segs.length - 1];
        if (last) {
            return {
                prompt: last.prompt || "",
                taskType: last.taskType || "",
                refs: last.refs ? JSON.parse(JSON.stringify(last.refs)) : [],
            };
        }
        return { prompt: "", taskType: "", refs: [] };
    }

    _buildSegmentsFromSplitPoints(points, forcedPoints = null) {
        const forced = new Set(forcedPoints || []);
        forced.add(0);
        const sorted = [...new Set(points)].sort((a, b) => a - b);
        forced.add(sorted[sorted.length - 1]);
        const newSegs = [];
        for (let i = 0; i < sorted.length - 1; i++) {
            const start = sorted[i];
            const length = sorted[i + 1] - start;
            const endsForced = forced.has(sorted[i + 1]);
            const startsForced = forced.has(start);
            if (length < MIN_SEG && !endsForced && !startsForced) continue;
            if (length < 1) continue;
            const meta = this._segmentMetaAtFrame(start);
            newSegs.push({
                id: uid(),
                start,
                length,
                prompt: meta.prompt,
                taskType: meta.taskType,
                refs: meta.refs,
            });
        }
        if (!newSegs.length) return null;
        let cursor = 0;
        return newSegs.map((seg) => {
            const s = { ...seg, start: cursor, length: seg.length };
            cursor += s.length;
            return s;
        });
    }

    _getReorderInsertFrame(dropRank, fromRank) {
        const ordered = [...this.timeline.segments].sort((a, b) => a.start - b.start);
        const lengths = ordered.map((s) => s.length);
        const without = lengths.filter((_, i) => i !== fromRank);
        let frame = 0;
        for (let i = 0; i < dropRank && i < without.length; i++) frame += without[i];
        return frame;
    }

    _orderedSegmentsWithRank() {
        return [...this.timeline.segments]
            .map((seg, arrayIndex) => ({ seg, arrayIndex }))
            .sort((a, b) => a.seg.start - b.seg.start)
            .map((item, visualRank) => ({ ...item, visualRank }));
    }

    _visualRankFromArrayIndex(arrayIndex) {
        const ordered = this._orderedSegmentsWithRank();
        return ordered.find((o) => o.arrayIndex === arrayIndex)?.visualRank ?? arrayIndex;
    }

    _computeReorderDropRank(frame, fromRank) {
        const ordered = this._orderedSegmentsWithRank();
        for (const item of ordered) {
            if (item.visualRank === fromRank) continue;
            const mid = item.seg.start + item.seg.length / 2;
            if (frame < mid) return item.visualRank;
        }
        return ordered.length - 1;
    }

    reorderSegmentsByRank(fromRank, toRank) {
        const ordered = [...this.timeline.segments]
            .map((seg) => ({ seg }))
            .sort((a, b) => a.seg.start - b.seg.start);
        if (fromRank < 0 || fromRank >= ordered.length) return;
        if (toRank < 0 || toRank >= ordered.length) return;
        if (fromRank === toRank) return;

        if (!this.getFrameMap().length && this.getTotalFrames() > 0) {
            this.materializeFrameMap();
        }
        const map = [...this.getFrameMap()];
        const slices = ordered.map((o) => map.slice(o.seg.start, o.seg.start + o.seg.length));
        const metas = ordered.map((o) => ({
            ...o.seg,
            refs: o.seg.refs ? JSON.parse(JSON.stringify(o.seg.refs)) : [],
        }));

        const [mSlice] = slices.splice(fromRank, 1);
        const [mMeta] = metas.splice(fromRank, 1);
        let insertRank = toRank;
        if (insertRank > fromRank) insertRank -= 1;
        slices.splice(insertRank, 0, mSlice);
        metas.splice(insertRank, 0, mMeta);

        const newMap = slices.flat();
        let start = 0;
        const newSegs = metas.map((seg, idx) => {
            const s = { ...seg, start, length: slices[idx].length };
            start += s.length;
            return s;
        });

        this.setFrameMap(newMap);
        this.timeline.segments = newSegs;
        this._syncPrimaryVideoFromClips(newMap);
        this._thumbCache.clear();
        this._thumbPending.clear();
        this.selectedIndex = insertRank;
        this._prefetchSegmentThumbs(0, Math.min(newMap.length, THUMB_PREFETCH_BATCH * 4));
    }

    materializeFrameMap() {
        const total = this.getTotalFrames();
        const video = this.timeline.video || {};
        if (video.frameMap?.length === total) return;
        const map = [];
        for (let i = 0; i < total; i++) map.push(this.getFrameMapEntry(i));
        video.frameMap = map;
        video.deletedSourceRanges = [];
        this.timeline.video = video;
        this.timeline.totalFrames = total;
    }

    getFrameMap() {
        const v = this.timeline?.video || {};
        if (v.frameMap?.length) return v.frameMap;
        if (this._legacyFrames.length) return buildIdentityFrameMap(this._legacyFrames.length);
        if (v.frames?.length) return buildIdentityFrameMap(v.frames.length);
        return [];
    }

    setFrameMap(map) {
        this.timeline.video = this.timeline.video || {};
        this.timeline.video.frameMap = map;
        if (map.length) {
            this.timeline.totalFrames = map.length;
            this.timeline.video.deletedSourceRanges = [];
        }
    }

    setSparseVideoFrames(totalFrames) {
        this.timeline.video = this.timeline.video || {};
        this.timeline.video.frameMap = [];
        this.timeline.video.sourceFrameCount = totalFrames;
        this.timeline.video.deletedSourceRanges = [];
        this.timeline.totalFrames = totalFrames;
    }

    logicalToSourceFrame(logical) {
        return logicalToSourceFrame(logical, this.timeline.video || {});
    }

    getTotalFrames() {
        if (this.isImageBatch() || this.isGenMode()) return sumFrameCounts(this.timeline.segments);
        const mapLen = this.timeline?.video?.frameMap?.length || 0;
        if (mapLen > 0) return mapLen;
        // Sparse deletes: sourceFrameCount − ranges beats a stale totalFrames.
        const src = parseInt(this.timeline?.video?.sourceFrameCount || 0, 10);
        if (src > 0) {
            const removed = deletedSourceRanges(this.timeline.video).reduce((s, [a, b]) => s + (b - a), 0);
            return Math.max(0, src - removed);
        }
        const total = Math.max(0, parseInt(this.timeline?.totalFrames || this.totalFramesWidget?.value || 0, 10));
        if (total > 0) return total;
        if (!this.hasVideo()) return 0;
        return 0;
    }

    getMaxExportFrames() {
        const n = parseInt(this.timeline.output?.maxExportFrames ?? 0, 10);
        return Number.isFinite(n) && n > 0 ? n : 0;
    }

    getExportFrameTotal() {
        const total = this.getTotalFrames();
        const cap = this.getMaxExportFrames();
        return cap > 0 ? Math.min(total, cap) : total;
    }

    getFrameRate() {
        return coerceTimelineFps(this.fpsInput?.value ?? this.frameRateWidget?.value ?? this.timeline.frameRate ?? 24);
    }

    syncFrameRateUI(value = null) {
        const fps = coerceTimelineFps(value ?? this.fpsInput?.value ?? this.frameRateWidget?.value ?? this.timeline.frameRate ?? 24);
        this.timeline.frameRate = fps;
        if (this.frameRateWidget) this.frameRateWidget.value = fps;
        if (this.fpsInput) this.fpsInput.value = fps;
        return fps;
    }

    _clipFrameCountAtFps(clip, fps, fallback = 0) {
        const nativeFps = Number(clip?.nativeFps || 0);
        const nativeCount = Number(clip?.nativeFrameCount || 0);
        if (nativeFps > 0 && nativeCount > 0) {
            return Math.max(1, Math.round((nativeCount / nativeFps) * fps));
        }
        const duration = Number(clip?.duration || 0);
        if (duration > 0) return Math.max(1, Math.round(duration * fps));
        return Math.max(1, Math.round(fallback || Number(clip?.sourceFrameCount || 0) || 1));
    }

    _timelineFrameCountAtFps(fps, oldFps = null, oldTotal = null) {
        const nextFps = coerceTimelineFps(fps);
        const prevTotal = Number(oldTotal ?? this.getTotalFrames() ?? 0);
        const prevFps = coerceTimelineFps(oldFps ?? this.timeline.frameRate ?? this.frameRateWidget?.value ?? 24);
        // When user changes timeline FPS, preserve wall-clock duration: T = N/fps → N' = T * fps'.
        if (prevTotal > 0 && oldFps != null && Math.abs(prevFps - nextFps) >= 0.001) {
            return Math.max(1, Math.round(prevTotal * nextFps / prevFps));
        }
        const clips = this.getVideoClips();
        if (clips.length && clips.some((c) => Number(c.duration || 0) > 0 || Number(c.nativeFrameCount || 0) > 0)) {
            return clips.reduce((sum, clip) => sum + this._clipFrameCountAtFps(clip, nextFps), 0);
        }
        if (prevTotal > 0) {
            return Math.max(1, Math.round(prevTotal * nextFps / Math.max(prevFps, 0.001)));
        }
        return 1;
    }

    _rescaleSegmentsForTotal(oldTotal, newTotal) {
        if (!oldTotal || !newTotal || !this.timeline.segments?.length) {
            this._setSingleSegment(newTotal);
            return;
        }
        const ordered = [...this.timeline.segments].sort((a, b) => a.start - b.start);
        let cursor = 0;
        this.timeline.segments = ordered.map((seg, idx) => {
            const rawStart = idx === 0 ? 0 : Math.round((seg.start / oldTotal) * newTotal);
            const rawEnd = idx === ordered.length - 1
                ? newTotal
                : Math.round(((seg.start + seg.length) / oldTotal) * newTotal);
            const start = clamp(rawStart, cursor, newTotal);
            const end = clamp(rawEnd, start + 1, newTotal);
            cursor = end;
            return {
                ...seg,
                start,
                length: Math.max(1, end - start),
                frameCount: Math.max(1, end - start),
            };
        });
    }

    _syncClipFrameCountsForFps(fps, oldFps = null) {
        const clips = this.getVideoClips();
        if (!clips.length) return;
        const prevFps = coerceTimelineFps(oldFps ?? this.timeline.frameRate ?? 24);
        this.timeline.videoClips = clips.map((clip) => {
            const fallback = Number(clip.sourceFrameCount || 0) * fps / Math.max(prevFps, 0.001);
            return { ...clip, sourceFrameCount: this._clipFrameCountAtFps(clip, fps, fallback) };
        });
    }

    _resampleFrameMapForFps(oldFps, newFps, newTotal) {
        const oldTotal = this.getTotalFrames();
        if (!oldTotal || !newTotal) return [];
        const oldEntries = Array.from({ length: oldTotal }, (_, i) => this.getFrameMapEntry(i));
        const clips = this.getVideoClips();
        const map = [];
        for (let i = 0; i < newTotal; i++) {
            const oldLogical = clamp(Math.round((i / newFps) * oldFps), 0, oldTotal - 1);
            const entry = normalizeFrameMapEntry(oldEntries[oldLogical]);
            const clip = clips[entry.clip] || clips[0] || {};
            const maxFrame = this._clipFrameCountAtFps(clip, newFps) - 1;
            const sourceTime = Number(entry.frame || 0) / Math.max(oldFps, 0.001);
            map.push({
                clip: entry.clip,
                frame: clamp(Math.round(sourceTime * newFps), 0, Math.max(0, maxFrame)),
            });
        }
        return map;
    }

    _resampleTimelineForFrameRate(oldFps, newFps) {
        if (this.isImageBatch() || this.isGenMode() || !this.hasVideo()) return;
        const oldTotal = this.getTotalFrames();
        const newTotal = this._timelineFrameCountAtFps(newFps, oldFps, oldTotal);
        const hasExplicitMap = this.getFrameMap().length > 0;
        const hasSparseDeletes = deletedSourceRanges(this.timeline.video || {}).length > 0;

        if (hasExplicitMap || hasSparseDeletes || this.getVideoClips().length > 1) {
            const newMap = this._resampleFrameMapForFps(oldFps, newFps, newTotal);
            this.setFrameMap(newMap);
            this._syncClipFrameCountsForFps(newFps, oldFps);
            this._syncPrimaryVideoFromClips(newMap);
        } else {
            this._syncClipFrameCountsForFps(newFps, oldFps);
            this.setSparseVideoFrames(newTotal);
            this._syncPrimaryVideoFromClips([]);
        }

        this._rescaleSegmentsForTotal(oldTotal, newTotal);
        this.currentFrame = clamp(Math.round((this.currentFrame / Math.max(oldTotal, 1)) * newTotal), 0, Math.max(0, newTotal - 1));
        if (this.totalFramesWidget) this.totalFramesWidget.value = newTotal;
        if (this.seekBar) {
            this.seekBar.max = Math.max(0, newTotal - 1);
            this.seekBar.value = this.currentFrame;
        }
        this._thumbCache.clear();
        this._thumbPending.clear();
    }

    onFrameRateChanged(value) {
        const oldFps = coerceTimelineFps(this.timeline.frameRate ?? this.frameRateWidget?.value ?? 24);
        const newFps = this.syncFrameRateUI(value);
        if (Math.abs(oldFps - newFps) < 0.001) {
            this.commit(false, { syncTimeline: true });
            return;
        }
        this._resampleTimelineForFrameRate(oldFps, newFps);
        this.updateVideoNameLabel();
        this.updateOutputPreview();
        this.scheduleRender();
        this.commit(false, { syncTimeline: true });
    }

    getTimelineDurationSec() {
        const total = this.getTotalFrames();
        const fps = this.getFrameRate();
        return total / Math.max(fps, 0.001);
    }

    isGlobalMode() { return (this.timeline.editMode || "global") === "global"; }

    setEditMode(mode) {
        this.timeline.editMode = mode;
        this.root.querySelector('[data-a="mode-global"]').classList.toggle("active", mode === "global");
        this.root.querySelector('[data-a="mode-segment"]').classList.toggle("active", mode === "segment");
        this.updateModeUI();
        this.commit();
    }

    updateModeUI() {
        const global = this.isGlobalMode();
        this.globalPanel.style.display = global ? "flex" : "none";
        this.segmentPanel.style.display = global ? "none" : "flex";
        this.updateReferenceImageVisibility({
            hideTimeline: this.isImageBatch() || this.isGenMode(),
            seg: global ? null : this.timeline.segments[this.selectedIndex],
        });
        if (!global) this.updateSelectionUI();
        else if (taskUsesReferenceVideo(this.getTaskKey())) this.renderRefVideoSlot();
    }

    getRefTarget() {
        if (this.isGlobalMode()) return this.timeline.global;
        const seg = this.timeline.segments[this.selectedIndex];
        return seg || this.timeline.global;
    }

    getDisplayPrompt(seg) {
        if (this.isGlobalMode()) return this.timeline.global?.prompt || "";
        return seg?.prompt || "";
    }

    populateTaskSelect(el, selected) {
        if (!el) return;
        const opts = this.taskTypeWidget?.options?.values || [];
        el.innerHTML = "";
        for (const v of opts) {
            const o = document.createElement("option");
            o.value = v; o.textContent = v;
            el.appendChild(o);
        }
        if (selected) el.value = selected;
    }

    getI2iSourceDimensions() {
        for (const seg of this.timeline.segments || []) {
            const gi = seg.genImage || {};
            const w = +(gi.width || 0);
            const h = +(gi.height || 0);
            if (w > 0 && h > 0) return { width: w, height: h };
        }
        const out = this.timeline.output || {};
        if (+(out.sourceWidth || 0) > 0 && +(out.sourceHeight || 0) > 0) {
            return { width: +out.sourceWidth, height: +out.sourceHeight };
        }
        return { width: 0, height: 0 };
    }

    getSourceDimensions() {
        const clips = this.getVideoClips?.() || [];
        const video = clips[0] || this.timeline.video || {};
        if (+(video.width || 0) > 0 && +(video.height || 0) > 0) {
            return { width: +video.width, height: +video.height };
        }
        return {
            width: this.timeline.width || this.widthWidget?.value || 832,
            height: this.timeline.height || this.heightWidget?.value || 480,
        };
    }

    _refreshVideoStorageDimensions(resolved) {
        if (!resolved?.width || !resolved?.height) return;
        this._storageWidth = resolved.width;
        this._storageHeight = resolved.height;
        if (this.timeline.video) {
            this.timeline.video.storageWidth = resolved.width;
            this.timeline.video.storageHeight = resolved.height;
        }
        for (const clip of this.getVideoClips()) {
            clip.storageWidth = resolved.width;
            clip.storageHeight = resolved.height;
        }
    }

    syncOutputUIFromTimeline() {
        const out = this.timeline.output || {
            mode: "long_edge", longEdge: 848, width: 832, height: 480,
            maxExportFrames: 0, exportMode: "all",
            continuityEnabled: false, continuityOverlapFrames: 9,
        };
        if (this.outMode) this.outMode.value = out.mode || "long_edge";
        if (this.outLong) this.outLong.value = String(out.longEdge ?? 848);
        if (this.outW) this.outW.value = String(out.width ?? 832);
        if (this.outH) this.outH.value = String(out.height ?? 480);
        if (this.outMaxFrames) this.outMaxFrames.value = String(out.maxExportFrames ?? 0);
        if (this.outExportMode) this.outExportMode.value = out.exportMode === "segments" ? "segments" : "all";
        if (this.segmentContinuityCb) this.segmentContinuityCb.checked = isContinuityEnabled(out);
        if (this.segmentContinuityOverlap) {
            this.segmentContinuityOverlap.value = String(out.continuityOverlapFrames ?? 9);
        }
        this.syncFrameRateUI(this.timeline.frameRate);
        this.updateOutputModeUI();
        this.updateSegmentContinuityUI();
        this.updateOutputPreview();
    }

    updateSegmentContinuityUI() {
        if (!this.segmentContinuityWrap) return;
        const show = !this.isGenMode() && !this.isImageBatch() && (this.timeline.segments?.length ?? 0) >= 2;
        this.segmentContinuityWrap.classList.toggle("hidden", !show);
    }

    updateOutputModeUI() {
        const mode = this.timeline.output?.mode || "long_edge";
        const isFixed = mode === "fixed";
        if (this.outLongWrap) this.outLongWrap.style.display = isFixed ? "none" : "";
        if (this.outFixedWrap) this.outFixedWrap.classList.toggle("hidden", !isFixed);
    }

    updateOutputPreview() {
        if (!this.outPreview) return;
        if (this.isImageBatch() && (this.getTaskKey() === "i2i" || this.getTaskKey() === "i2v")) {
            const out = this.timeline.output || {};
            if ((out.mode || "long_edge") === "long_edge") {
                const src = this.getI2iSourceDimensions();
                const resolved = resolveOutputDimensions(src.width, src.height, out, {
                    refMaxSize: this.refMaxWidget?.value,
                });
                const note = src.width > 0 ? "" : " · 上传源图后按最长边计算";
                this.outPreview.textContent = `→ ${resolved.width}×${resolved.height}${note}${this._exportPreviewSuffix()}`;
            } else {
                const w = snapDim(+(out.width ?? this.outW?.value ?? 832));
                const h = snapDim(+(out.height ?? this.outH?.value ?? 480));
                this.outPreview.textContent = `→ ${w}×${h}${this._exportPreviewSuffix()}`;
            }
            return;
        }
        if (this.isGenBlank() || this.isImageBatch()) {
            const out = this.timeline.output || {};
            const w = snapDim(+(out.width ?? this.outW?.value ?? 832));
            const h = snapDim(+(out.height ?? this.outH?.value ?? 480));
            this.outPreview.textContent = `→ ${w}×${h}${this._exportPreviewSuffix()}`;
            return;
        }
        const src = this.getSourceDimensions();
        const resolved = resolveOutputDimensions(src.width, src.height, this.timeline.output, {
            width: this.widthWidget?.value,
            height: this.heightWidget?.value,
            refMaxSize: this.refMaxWidget?.value,
        });
        this.outPreview.textContent = `→ ${resolved.width}×${resolved.height}${this._exportPreviewSuffix()}`;
    }

    _exportPreviewSuffix() {
        const cap = this.getMaxExportFrames();
        const exportMode = this.timeline.output?.exportMode === "segments" ? " · 分段导出" : "";
        const dur = this.getTimelineDurationSec().toFixed(2);
        const fps = formatProbeFps(this.getFrameRate());
        const timeHint = ` · ${dur}s @ ${fps}fps`;
        if (cap <= 0) return `${timeHint}${exportMode}`;
        const total = this.getTotalFrames();
        const exportTotal = this.getExportFrameTotal();
        if (exportTotal >= total) return `${timeHint} · 导出 ${exportTotal} 帧${exportMode}`;
        return `${timeHint} · 导出 ${exportTotal}/${total} 帧${exportMode}`;
    }

    onOutputField(key, value) {
        this.timeline.output = this.timeline.output || {
            mode: "long_edge", longEdge: 848, width: 832, height: 480,
            maxExportFrames: 0, exportMode: "all",
            continuityEnabled: false, continuityOverlapFrames: 9,
        };
        if (key === "mode") {
            this.timeline.output.mode = value;
        } else if (key === "longEdge") {
            this.timeline.output.longEdge = snapDim(value || 848);
        } else if (key === "width") {
            this.timeline.output.width = snapDim(value || 832);
        } else if (key === "height") {
            this.timeline.output.height = snapDim(value || 480);
        } else if (key === "maxExportFrames") {
            const n = parseInt(value, 10);
            this.timeline.output.maxExportFrames = Number.isFinite(n) && n > 0 ? n : 0;
        } else if (key === "exportMode") {
            this.timeline.output.exportMode = value === "segments" ? "segments" : "all";
        } else if (key === "continuityEnabled") {
            this.timeline.output.continuityEnabled = !!value;
        } else if (key === "continuityOverlapFrames") {
            const n = parseInt(value, 10);
            this.timeline.output.continuityOverlapFrames = Number.isFinite(n)
                ? Math.max(1, Math.min(81, n))
                : 9;
        }
        this.syncOutputUIFromTimeline();
        this.commit();
        this.flushTimelineSync();
    }

    syncOutputToWidgets() {
        if (this.isImageBatch() && (this.getTaskKey() === "i2i" || this.getTaskKey() === "i2v")) {
            const out = this.timeline.output || {};
            const mode = (out.mode || "long_edge").toLowerCase();
            if (mode === "long_edge") {
                const src = this.getI2iSourceDimensions();
                const resolved = resolveOutputDimensions(src.width, src.height, out, {
                    width: this.widthWidget?.value,
                    height: this.heightWidget?.value,
                    refMaxSize: this.refMaxWidget?.value,
                });
                this.timeline.output = {
                    ...out,
                    mode: "long_edge",
                    longEdge: out.longEdge ?? resolved.refMaxSize,
                    width: resolved.width,
                    height: resolved.height,
                };
                if (this.widthWidget) this.widthWidget.value = resolved.width;
                if (this.heightWidget) this.heightWidget.value = resolved.height;
                if (this.refMaxWidget) this.refMaxWidget.value = resolved.refMaxSize;
                this.timeline.width = resolved.width;
                this.timeline.height = resolved.height;
                this.timeline.refMaxSize = resolved.refMaxSize;
            } else {
                const w = snapDim(+(out.width ?? this.widthWidget?.value ?? 832));
                const h = snapDim(+(out.height ?? this.heightWidget?.value ?? 480));
                this.timeline.output = { ...out, mode: "fixed", width: w, height: h };
                if (this.widthWidget) this.widthWidget.value = w;
                if (this.heightWidget) this.heightWidget.value = h;
                this.timeline.width = w;
                this.timeline.height = h;
            }
            this.updateOutputPreview();
            return;
        }
        if (this.isGenBlank() || this.isImageBatch()) {
            const out = this.timeline.output || {};
            const w = snapDim(+(out.width ?? this.widthWidget?.value ?? 832));
            const h = snapDim(+(out.height ?? this.heightWidget?.value ?? 480));
            this.timeline.output = { ...out, mode: "fixed", width: w, height: h };
            if (this.widthWidget) this.widthWidget.value = w;
            if (this.heightWidget) this.heightWidget.value = h;
            this.timeline.width = w;
            this.timeline.height = h;
            this.updateOutputPreview();
            return;
        }
        const src = this.getSourceDimensions();
        const resolved = resolveOutputDimensions(src.width, src.height, this.timeline.output, {
            width: this.timeline.width,
            height: this.timeline.height,
            refMaxSize: this.timeline.refMaxSize,
        });
        this.timeline.output = {
            mode: resolved.mode,
            longEdge: this.timeline.output?.longEdge ?? resolved.refMaxSize,
            width: resolved.width,
            height: resolved.height,
            maxExportFrames: this.timeline.output?.maxExportFrames ?? 0,
            exportMode: this.timeline.output?.exportMode ?? "all",
            continuityEnabled: isContinuityEnabled(this.timeline.output),
            continuityOverlapFrames: Math.max(1, Math.min(81,
                parseInt(this.timeline.output?.continuityOverlapFrames ?? 9, 10) || 9)),
        };
        if (this.widthWidget) this.widthWidget.value = resolved.width;
        if (this.heightWidget) this.heightWidget.value = resolved.height;
        if (this.refMaxWidget) this.refMaxWidget.value = resolved.refMaxSize;
        this.timeline.width = resolved.width;
        this.timeline.height = resolved.height;
        this.timeline.refMaxSize = resolved.refMaxSize;
        this._refreshVideoStorageDimensions(resolved);
        this.updateOutputPreview();
    }

    syncFromWidgets() {
        this.timeline.global = this.timeline.global || { refs: [], referenceVideo: {}, continuousReference: false };
        this.timeline.global.taskType = this.globalTask?.value || this.taskTypeWidget?.value || "";
        this.timeline.global.prompt = this.globalPrompt?.value ?? this.globalPromptWidget?.value ?? "";
        if (this.continuousRefCb) {
            this.timeline.global.continuousReference = !!this.continuousRefCb.checked;
        }
        this.timeline.totalFrames = this.getTotalFrames();
        this.timeline.frameRate = this.getFrameRate();
        this.timeline.output = this.timeline.output || {
            mode: "long_edge", longEdge: 848, width: 832, height: 480,
            maxExportFrames: 0, exportMode: "all",
            continuityEnabled: false, continuityOverlapFrames: 9,
        };
        if (this.segmentContinuityCb) {
            this.timeline.output.continuityEnabled = !!this.segmentContinuityCb.checked;
        }
        if (this.segmentContinuityOverlap) {
            const n = parseInt(this.segmentContinuityOverlap.value, 10);
            this.timeline.output.continuityOverlapFrames = Number.isFinite(n)
                ? Math.max(1, Math.min(81, n))
                : (this.timeline.output.continuityOverlapFrames ?? 9);
        }
        this.syncOutputToWidgets();
    }

    commit(skipRender = false, { syncTimeline = true } = {}) {
        this._promptEnhancer?.syncToWidgets?.();
        this.syncFromWidgets();
        this.normalizeSegments();
        if (this.isRunSelectEnabled()) this.normalizeRunSelection();
        this.updateRunSelectUI();
        if (this.taskTypeWidget) this.taskTypeWidget.value = this.timeline.global.taskType;
        if (this.globalPromptWidget) this.globalPromptWidget.value = this.timeline.global.prompt;
        if (this.negativePromptWidget) {
            const neg = this.globalNegative?.value ?? this.segNegative?.value ?? this.negativePromptWidget.value ?? "";
            this.negativePromptWidget.value = neg;
        }
        if (this.totalFramesWidget) this.totalFramesWidget.value = Math.max(0, this.getTotalFrames());
        this.seekBar.max = Math.max(0, this.getTotalFrames() - 1);
        if (syncTimeline) this.scheduleTimelineSync();
        if (!skipRender) this.scheduleRender();
        if (this.isGlobalMode() && taskUsesReferenceImages(this.getTaskKey())) {
            this.renderRefSlots(this.timeline.global.refs, this.globalRefsBox, true);
        } else if (this.isImageBatch()) this.renderImageBatchGroups();
        else this.updateSelectionUI();
    }

    normalizeSegments() {
        if (this.isImageBatch()) {
            this.normalizeImageBatchSegments();
            return;
        }
        if (this.isGenMode()) {
            this.normalizeGenSegments();
            return;
        }
        const total = this.getTotalFrames();
        let segs = [...this.timeline.segments].sort((a, b) => a.start - b.start);
        if (!total) {
            this.timeline.segments = [];
            this.timeline.totalFrames = 0;
            return;
        }
        if (!segs.length) segs = [{ id: uid(), start: 0, length: total, prompt: "", taskType: "", refs: [], referenceVideo: {} }];
        const fixed = [];
        let cursor = 0;
        for (const seg of segs) {
            const start = clamp(seg.start, cursor, total);
            let length = Math.max(MIN_SEG, seg.length ?? (total - start));
            if (start + length > total) length = total - start;
            if (length < MIN_SEG) continue;
            fixed.push({ ...seg, start, length, refs: seg.refs || [] });
            cursor = start + length;
        }
        if (fixed.length && cursor < total) fixed[fixed.length - 1].length += total - cursor;
        this.timeline.segments = fixed;
        this.timeline.totalFrames = total;
        this.selectedIndex = clamp(this.selectedIndex, 0, Math.max(0, fixed.length - 1));
        this.updateSegmentContinuityUI();
    }

    getVideoViewUrl() {
        return this.getClipViewUrl(0);
    }

    getSourceFrameIndex(logicalFrame) {
        return this.getFrameMapEntry(logicalFrame).frame;
    }

    _getPreviewVideoForClip(clipIndex) {
        const url = this.getClipViewUrl(clipIndex);
        if (!this._previewVideos) this._previewVideos = new Map();
        if (clipIndex === 0 && this._previewVideo && !this._previewVideos.has(0)) {
            if (url) this._previewVideo.src = url;
            this._previewVideos.set(0, this._previewVideo);
        }
        if (!url) return this._previewVideos.get(clipIndex) || (clipIndex === 0 ? this._previewVideo : null);
        let v = this._previewVideos.get(clipIndex);
        if (!v) {
            v = document.createElement("video");
            v.crossOrigin = "anonymous";
            v.muted = true;
            v.playsInline = true;
            v.preload = "auto";
            v.style.cssText = "position:fixed;left:-9999px;width:1px;height:1px;opacity:0;pointer-events:none";
            document.body.appendChild(v);
            v.src = url;
            this._previewVideos.set(clipIndex, v);
        } else if (url && v.src !== url && !String(v.src).includes(encodeURIComponent(url.split("/").pop()?.split("?")[0] || ""))) {
            v.src = url;
        }
        return v;
    }

    _restorePreviewVideos() {
        const clips = this.getVideoClips();
        if (!clips.length) return;
        for (let i = 0; i < clips.length; i++) this._getPreviewVideoForClip(i);
        this._previewVideo = this._previewVideos.get(0) || this._previewVideo;
    }

    _clearPreviewVideos(removeExtra = true) {
        if (!this._previewVideos) return;
        for (const [idx, v] of this._previewVideos.entries()) {
            v.pause();
            if (idx === 0 && v === this._previewVideo) {
                v.removeAttribute("src");
                v.load();
                continue;
            }
            if (removeExtra) {
                v.removeAttribute("src");
                v.load();
                v.remove();
            }
        }
        const keep = this._previewVideo;
        this._previewVideos.clear();
        if (keep) this._previewVideos.set(0, keep);
    }

    async _seekPreviewVideo(timeSec, clipIndex = 0) {
        this._seekChain = this._seekChain.then(() => new Promise((resolve) => {
            const v = this._getPreviewVideoForClip(clipIndex);
            if (!v?.src) { resolve(); return; }
            const target = Math.max(0, timeSec);
            const onSeeked = () => {
                v.removeEventListener("seeked", onSeeked);
                resolve();
            };
            v.addEventListener("seeked", onSeeked);
            try {
                v.currentTime = target;
            } catch {
                onSeeked();
                return;
            }
            if (Math.abs(v.currentTime - target) < 0.02 && v.readyState >= 2) {
                onSeeked();
            }
        }));
        return this._seekChain;
    }

    updateStageVisibility() {
        if (!this.stageEl) return;
        const show = this.hasVideo()
            && !this.isImageBatch()
            && !this.isGenMode();
        this.stageEl.classList.toggle("hidden", !show);
        if (!show) {
            if (this.stageVideo) {
                this.stageVideo.pause();
                this.stageVideo.classList.add("hidden");
            }
            this.stageImg?.classList.add("hidden");
            this.stageEmpty?.classList.remove("hidden");
            this.stageBadge?.classList.add("hidden");
            this._stageClipIndex = -1;
        } else {
            this._syncStagePreview(this.currentFrame, { force: true });
        }
        this.updateDomWidgetHeight();
        syncDirectorNodeSize(this.node, this);
    }

    _updateStageBadge(logicalFrame) {
        if (!this.stageBadge) return;
        const total = this.getTotalFrames();
        const frame = clamp(logicalFrame | 0, 0, Math.max(0, total - 1));
        const clips = this.getVideoClips();
        const entry = this.getFrameMapEntry(frame);
        const clipHint = clips.length > 1 ? ` · 片${entry.clip + 1}` : "";
        this.stageBadge.textContent = `帧 ${frame + 1}/${total}${clipHint}`;
        this.stageBadge.classList.remove("hidden");
    }

    _logicalRangeForClip(clipIndex) {
        const map = this.getFrameMap();
        let start = -1;
        let end = -1;
        for (let i = 0; i < map.length; i++) {
            const e = normalizeFrameMapEntry(map[i]);
            if (e.clip !== clipIndex) {
                if (start >= 0) break;
                continue;
            }
            if (start < 0) start = i;
            end = i + 1;
        }
        if (start < 0) return { start: 0, end: this.getTotalFrames() };
        return { start, end };
    }

    _logicalFromStageTime(clipIndex, timeSec) {
        const fps = Math.max(0.001, this.getFrameRate());
        const srcFrame = Math.max(0, Math.round(Number(timeSec) * fps));
        const map = this.getFrameMap();
        if (!map.length) {
            const logical = sourceToLogicalFrame(srcFrame, this.timeline.video || {});
            if (logical < 0) return -1; // source lands in a deleted gap
            return clamp(logical, 0, Math.max(0, this.getTotalFrames() - 1));
        }
        let first = -1;
        let best = -1;
        for (let i = 0; i < map.length; i++) {
            const e = normalizeFrameMapEntry(map[i]);
            if (e.clip !== clipIndex) continue;
            if (first < 0) first = i;
            if (e.frame === srcFrame) return i;
            if (e.frame <= srcFrame) best = i;
        }
        if (best >= 0) return best;
        if (first >= 0) return first;
        return 0;
    }

    /** Next logical index whose source frame is strictly after srcFrame (same clip). */
    _nextLogicalAfterSourceFrame(clipIndex, srcFrame) {
        const map = this.getFrameMap();
        if (!map.length) {
            // Sparse: walk forward until source maps to a kept logical frame.
            const total = this.getTotalFrames();
            const startLogical = sourceToLogicalFrame(srcFrame, this.timeline.video || {});
            const from = startLogical < 0 ? 0 : startLogical;
            for (let i = from; i < total; i++) {
                if (this.logicalToSourceFrame(i) > srcFrame) return i;
            }
            return -1;
        }
        for (let i = 0; i < map.length; i++) {
            const e = normalizeFrameMapEntry(map[i]);
            if (e.clip === clipIndex && e.frame > srcFrame) return i;
        }
        return -1;
    }

    _syncStagePreview(logicalFrame, { force = false } = {}) {
        if (!this.stageEl || this.stageEl.classList.contains("hidden")) return;
        if (!this.hasVideo()) {
            this.stageEmpty?.classList.remove("hidden");
            this.stageVideo?.classList.add("hidden");
            this.stageImg?.classList.add("hidden");
            return;
        }

        // During native playback, do not seek every tick (that causes stutter).
        // Only refresh the badge; playhead is driven from video.currentTime.
        if (this.isPlaying && !force && !this._legacyFrames.length) {
            this._updateStageBadge(logicalFrame);
            return;
        }

        const frame = clamp(logicalFrame | 0, 0, Math.max(0, this.getTotalFrames() - 1));
        const fps = Math.max(0.001, this.getFrameRate());

        if (this._legacyFrames.length) {
            const dataUrl = this._legacyFrames[frame];
            if (this.stageVideo) {
                this.stageVideo.pause();
                this.stageVideo.classList.add("hidden");
            }
            if (this.stageImg && dataUrl) {
                this.stageImg.src = dataUrl;
                this.stageImg.classList.remove("hidden");
                this.stageEmpty?.classList.add("hidden");
            }
            this._updateStageBadge(frame);
            return;
        }

        const entry = this.getFrameMapEntry(frame);
        const url = this.getClipViewUrl(entry.clip);
        const v = this.stageVideo;
        if (!v || !url) {
            this.stageEmpty?.classList.remove("hidden");
            return;
        }

        this.stageImg?.classList.add("hidden");
        this.stageEmpty?.classList.add("hidden");
        v.classList.remove("hidden");

        let sameSrc = false;
        if (v.src && url) {
            try {
                sameSrc = new URL(v.src, location.href).href === new URL(url, location.href).href;
            } catch {
                sameSrc = v.src === url;
            }
        }
        // Must reload when the file changes even if clip index stays 0 (replace upload).
        if (this._stageClipIndex !== entry.clip || !sameSrc) {
            this._stageClipIndex = entry.clip;
            if (!sameSrc) {
                v.pause();
                v.src = url;
                v.load();
            }
        }

        const target = Math.max(0, entry.frame / fps);
        if (force || Math.abs(v.currentTime - target) > 0.035) {
            try {
                v.currentTime = target;
            } catch {
                /* ignore seek races while loading */
            }
        }
        if (this.isPlaying && force) {
            v.play().catch(() => {});
        }
        this._updateStageBadge(frame);
    }

    async _ensureStageReadyForFrame(logicalFrame) {
        this._syncStagePreview(logicalFrame, { force: true });
        const v = this.stageVideo;
        if (!v || this._legacyFrames.length) return false;
        if (v.readyState >= 2) return true;
        await new Promise((resolve) => {
            const done = () => {
                v.removeEventListener("loadeddata", done);
                v.removeEventListener("canplay", done);
                resolve();
            };
            v.addEventListener("loadeddata", done);
            v.addEventListener("canplay", done);
            setTimeout(done, 800);
        });
        return true;
    }

    _queueThumbPrefetch(logicalFrame) {
        if (this.isPlaying) return;
        if (this._thumbCache.has(logicalFrame) || this._thumbPending.has(logicalFrame)) return;
        if (!this.hasVideo() && !this._legacyFrames.length) return;
        this._thumbPending.add(logicalFrame);
        this._fetchThumb(logicalFrame).then((img) => {
            this._thumbPending.delete(logicalFrame);
            if (img) this._thumbCache.set(logicalFrame, img);
            this.scheduleRender();
        });
    }

    async _fetchThumb(logicalFrame) {
        if (this._legacyFrames.length) {
            const dataUrl = this._legacyFrames[logicalFrame];
            if (!dataUrl) return null;
            return this._decodeThumb(dataUrl);
        }
        const entry = this.getFrameMapEntry(logicalFrame);
        const v = this._getPreviewVideoForClip(entry.clip);
        if (!v?.src || !v.videoWidth) return null;
        const t = Math.max(0, entry.frame / this.getFrameRate());
        await this._seekPreviewVideo(t, entry.clip);
        const ratio = v.videoWidth > THUMB_MAX_W ? THUMB_MAX_W / v.videoWidth : 1;
        const tw = Math.max(1, Math.round(v.videoWidth * ratio));
        const th = Math.max(1, Math.round(v.videoHeight * ratio));
        this._thumbCanvas.width = tw;
        this._thumbCanvas.height = th;
        this._thumbCtx.drawImage(v, 0, 0, tw, th);
        return new Promise((resolve) => {
            const img = new Image();
            img.onload = () => resolve(img);
            img.onerror = () => resolve(null);
            img.src = this._thumbCanvas.toDataURL("image/jpeg", THUMB_JPEG_Q);
        });
    }

    _clearVideoState() {
        this._thumbCache.clear();
        this._thumbPending.clear();
        this._legacyFrames = [];
        this.timeline.videoClips = [];
        this.timeline.videoWorkspace = null;
        // Wipe video identity BEFORE visibility sync — otherwise hasVideo() stays
        // true via the old videoFile and stage reloads the previous clip.
        this.timeline.video = {
            fileName: "",
            videoFile: "",
            subfolder: "",
            type: "input",
            frames: [],
            frameMap: [],
            deletedSourceRanges: [],
            sourceFrameCount: 0,
            width: 0,
            height: 0,
        };
        this.timeline.totalFrames = 0;
        this._storageWidth = 0;
        this._storageHeight = 0;
        this._clearPreviewVideos(true);
        if (this._previewVideo) {
            this._previewVideo.pause();
            this._previewVideo.removeAttribute("src");
            this._previewVideo.load();
        }
        if (this.stageVideo) {
            this.stageVideo.pause();
            this.stageVideo.removeAttribute("src");
            this.stageVideo.load();
            this.stageVideo.classList.add("hidden");
        }
        this.stageImg?.classList.add("hidden");
        if (this.stageImg) this.stageImg.removeAttribute("src");
        this.stageEmpty?.classList.remove("hidden");
        this.stageBadge?.classList.add("hidden");
        this._stageClipIndex = -1;
        this.updateStageVisibility();
    }

    _resetTimelineForReplaceUpload() {
        this._clearVideoState();
        this.timeline.segments = [];
        this.selectedIndex = 0;
        this.currentFrame = 0;
        if (this.seekBar) {
            this.seekBar.value = 0;
            this.seekBar.max = 0;
        }
    }

    _setSingleSegment(totalFrames) {
        const total = Math.max(0, totalFrames);
        this.timeline.segments = total > 0
            ? [{ id: uid(), start: 0, length: total, prompt: "", taskType: "", refs: [], referenceVideo: {} }]
            : [];
        this.selectedIndex = 0;
        this.currentFrame = 0;
        if (this.seekBar) {
            this.seekBar.max = Math.max(0, total - 1);
            this.seekBar.value = 0;
        }
    }

    restoreVideoFromTimeline() {
        const video = this.timeline.video || {};
        this._storageWidth = video.storageWidth || 0;
        this._storageHeight = video.storageHeight || 0;

        const legacy = video.frames || [];
        if (legacy.length && !video.videoFile) {
            this._legacyFrames = legacy;
            this.setFrameMap(buildIdentityFrameMap(legacy.length));
            this.videoNameEl.textContent = `${video.fileName || "视频"} (${legacy.length}f · 旧版内嵌)`;
            this._prefetchSegmentThumbs(0, legacy.length);
            this.updateStageVisibility();
            return;
        }

        if (!video.videoFile) {
            this._clearVideoState();
            return;
        }

        this._restorePreviewVideos();
        const n = this.getTotalFrames();
        this._prefetchSegmentThumbs(0, Math.min(n, THUMB_PREFETCH_BATCH * 4));
        this.updateVideoNameLabel();
        if (taskUsesReferenceVideo(this.getTaskKey()) && this.getReferenceVideoViewUrl(this.timeline.global?.referenceVideo)) {
            this.renderRefVideoSlot();
        }
        this.updateStageVisibility();
    }

    _prefetchSegmentThumbs(from, to) {
        for (let f = from; f < to; f++) this._queueThumbPrefetch(f);
    }

    _decodeThumb(dataUrl) {
        return new Promise((resolve) => {
            const img = new Image();
            img.onload = () => {
                if (!img.naturalWidth || img.naturalWidth <= THUMB_MAX_W) {
                    resolve(img);
                    return;
                }
                const ratio = THUMB_MAX_W / img.naturalWidth;
                const w = THUMB_MAX_W;
                const h = Math.max(1, Math.round(img.naturalHeight * ratio));
                const c = document.createElement("canvas");
                c.width = w;
                c.height = h;
                c.getContext("2d").drawImage(img, 0, 0, w, h);
                const thumb = new Image();
                thumb.onload = () => resolve(thumb);
                thumb.onerror = () => resolve(img);
                thumb.src = c.toDataURL("image/jpeg", THUMB_JPEG_Q);
            };
            img.onerror = () => resolve(null);
            img.src = dataUrl.startsWith("data:") ? dataUrl : `data:image/jpeg;base64,${dataUrl}`;
        });
    }

    pickVideoFile() {
        const input = document.createElement("input");
        input.type = "file"; input.accept = "video/*";
        input.onchange = () => { if (input.files?.[0]) this.loadVideoFile(input.files[0]); };
        input.click();
    }

    pickAppendVideoFile() {
        if (!this.hasVideo()) {
            this.showBdMessage(
                "追加视频",
                "请先上传第一个视频，再使用「追加视频」。"
            );
            return;
        }
        const input = document.createElement("input");
        input.type = "file"; input.accept = "video/*";
        input.onchange = () => { if (input.files?.[0]) this.appendVideoFile(input.files[0]); };
        input.click();
    }

    async appendVideoFile(file) {
        const btn = this.root.querySelector('[data-a="video-append"]');
        if (btn) { btn.disabled = true; btn.textContent = "上传中…"; }
        this.videoNameEl.textContent = `追加中: ${file.name}…`;
        try {
            const uploaded = await uploadToInputSmart(file, (frac, cur, total) => {
                const pct = Math.round(frac * 100);
                const mode = file.size > COMFY_UPLOAD_SOFT_LIMIT ? "分块" : "上传";
                this.videoNameEl.textContent = `追加${mode}: ${file.name} (${cur}/${total}, ${pct}%)…`;
            });
            const relPath = videoRelativePath(uploaded);
            await this._applyAppendedVideo({
                fileName: file.name,
                relPath,
                subfolder: uploaded.subfolder || "",
                type: uploaded.type || "input",
                statusPrefix: "解析",
            });
        } catch (err) {
            console.error("[BerniniDirector] append video failed:", err);
            this.videoNameEl.textContent = `追加失败: ${formatUploadError(err)}`;
            this.updateVideoNameLabel();
        } finally {
            if (btn) { btn.disabled = false; btn.textContent = "追加视频"; }
        }
    }

    async loadVideoFile(file) {
        const btn = this.root.querySelector('[data-a="video"]');
        if (btn) { btn.disabled = true; btn.textContent = "上传中…"; }
        this.videoNameEl.textContent = `上传中: ${file.name}…`;
        try {
            this._resetTimelineForReplaceUpload();
            const uploaded = await uploadToInputSmart(file, (frac, cur, total) => {
                const pct = Math.round(frac * 100);
                const mode = file.size > COMFY_UPLOAD_SOFT_LIMIT ? "分块" : "上传";
                this.videoNameEl.textContent = `${mode}中: ${file.name} (${cur}/${total}, ${pct}%)…`;
            });
            const relPath = videoRelativePath(uploaded);
            await this._applyLoadedVideo({
                fileName: file.name,
                relPath,
                subfolder: uploaded.subfolder || "",
                type: uploaded.type || "input",
                statusPrefix: "解析",
            });
        } catch (err) {
            console.error("[BerniniDirector] video load failed:", err);
            this.videoNameEl.textContent = `加载失败: ${formatUploadError(err)}`;
            this._resetTimelineForReplaceUpload();
        } finally {
            if (btn) { btn.disabled = false; btn.textContent = "上传视频"; }
        }
    }

    _closeBdModal() {
        if (this._modalKeyHandler) {
            window.removeEventListener("keydown", this._modalKeyHandler, true);
            this._modalKeyHandler = null;
        }
        if (this._modalEl) {
            this._modalEl.remove();
            this._modalEl = null;
        }
    }

    showBdMessage(title, message) {
        return this.showBdDialog({ title, message, confirmText: "确定", cancelText: null });
    }

    showBdDialog({ title, message, items, confirmText = "确定", cancelText = "取消" }) {
        return new Promise((resolve) => {
            this._closeBdModal();

            const overlay = document.createElement("div");
            overlay.className = "bd-modal-overlay";
            const panel = document.createElement("div");
            panel.className = "bd-modal";
            panel.innerHTML = `
                <div class="bd-modal-title"></div>
                <div class="bd-modal-body hidden"></div>
                <div class="bd-modal-list hidden"></div>
                <div class="bd-modal-actions"></div>`;

            panel.querySelector(".bd-modal-title").textContent = title || "";

            const bodyEl = panel.querySelector(".bd-modal-body");
            const listEl = panel.querySelector(".bd-modal-list");
            const actionsEl = panel.querySelector(".bd-modal-actions");

            let selectedValue = items?.length ? items[0].value : null;

            const finish = (val) => {
                this._closeBdModal();
                resolve(val);
            };

            if (message) {
                bodyEl.textContent = message;
                bodyEl.classList.remove("hidden");
            }

            if (items?.length) {
                listEl.classList.remove("hidden");
                for (const item of items) {
                    const row = document.createElement("div");
                    row.className = "bd-modal-item";
                    row.textContent = item.label ?? item.value;
                    row.title = item.label ?? item.value;
                    row.dataset.value = item.value;
                    if (item.value === selectedValue) row.classList.add("selected");
                    row.onclick = () => {
                        selectedValue = item.value;
                        for (const el of listEl.querySelectorAll(".bd-modal-item")) {
                            el.classList.toggle("selected", el === row);
                        }
                    };
                    row.ondblclick = () => finish(item.value);
                    listEl.appendChild(row);
                }
            }

            if (cancelText) {
                const cancelBtn = document.createElement("button");
                cancelBtn.type = "button";
                cancelBtn.className = "bd-btn";
                cancelBtn.textContent = cancelText;
                cancelBtn.onclick = () => finish(null);
                actionsEl.appendChild(cancelBtn);
            }

            const okBtn = document.createElement("button");
            okBtn.type = "button";
            okBtn.className = "bd-btn bd-btn-primary";
            okBtn.textContent = confirmText;
            okBtn.onclick = () => finish(items?.length ? selectedValue : true);
            actionsEl.appendChild(okBtn);

            overlay.onclick = (e) => {
                if (e.target === overlay && cancelText) finish(null);
            };
            panel.onclick = (e) => e.stopPropagation();

            this._modalKeyHandler = (e) => {
                if (e.key === "Escape") {
                    e.preventDefault();
                    e.stopPropagation();
                    finish(cancelText ? null : true);
                } else if (e.key === "Enter" && items?.length) {
                    e.preventDefault();
                    finish(selectedValue);
                }
            };
            window.addEventListener("keydown", this._modalKeyHandler, true);

            overlay.appendChild(panel);
            this.root.appendChild(overlay);
            this._modalEl = overlay;
            okBtn.focus();
        });
    }

    async _prepareVideoFrames({ fileName, relPath, subfolder, type, statusPrefix, syncNativeFps = true }) {
        this.videoNameEl.textContent = `${statusPrefix}: ${fileName}…`;
        const viewUrl = inputViewUrl(relPath, type || "input");

        let serverProbe = null;
        try {
            serverProbe = await this.probeVideoFile(relPath, subfolder, type);
        } catch (err) {
            console.warn("[BerniniDirector] video probe failed, using browser estimate:", err);
        }
        const browserMeta = await this.probeVideoMetadata(viewUrl);
        const nativeFps = Number(serverProbe?.native_fps || 0);
        const nativeFrameCount = Number(serverProbe?.frame_count || 0);
        const meta = {
            width: Number(serverProbe?.width || browserMeta.width || 0),
            height: Number(serverProbe?.height || browserMeta.height || 0),
            duration: Number(serverProbe?.duration ?? browserMeta.duration ?? 0),
            nativeFps,
            nativeFrameCount,
            probeMethod: serverProbe?.probe_method || "browser_estimate",
        };

        if (syncNativeFps && nativeFps > 0) {
            this.syncFrameRateUI(nativeFps);
        }

        const fps = this.getFrameRate();
        const totalFrames = Math.max(
            1,
            Math.round(meta.duration * fps) || nativeFrameCount,
        );

        const store = resolveOutputDimensions(meta.width, meta.height, this.timeline.output || { mode: "long_edge", longEdge: 848 }, {
            refMaxSize: this.refMaxWidget?.value,
        });

        return { fileName, relPath, subfolder, type, meta, totalFrames, store, viewUrl };
    }

    async probeVideoFile(relPath, subfolder = "", type = "input") {
        const resp = await api.fetchApi("/bernini/director/probe_video", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ videoFile: relPath, subfolder, type: type || "input" }),
        });
        if (!resp.ok) {
            throw new Error(await resp.text());
        }
        return resp.json();
    }

    _buildClipRecord({ fileName, relPath, subfolder, type, meta, totalFrames, store }) {
        return {
            id: uid(),
            fileName,
            videoFile: relPath,
            subfolder: subfolder || "",
            type: type || "input",
            width: meta.width,
            height: meta.height,
            duration: meta.duration,
            nativeFps: meta.nativeFps || null,
            nativeFrameCount: meta.nativeFrameCount || null,
            sourceFrameCount: totalFrames,
            storageWidth: store.width,
            storageHeight: store.height,
        };
    }

    _syncPrimaryVideoFromClips(frameMap) {
        const clips = this.getVideoClips();
        const primary = clips[0] || {};
        const prev = this.timeline.video || {};
        const map = Array.isArray(frameMap) ? frameMap : (prev.frameMap || []);
        this.timeline.video = {
            ...prev,
            ...primary,
            // Keep path/type from the clip record, but never drop timeline edits.
            fileName: primary.fileName || prev.fileName || "",
            videoFile: primary.videoFile || prev.videoFile || "",
            subfolder: primary.subfolder ?? prev.subfolder ?? "",
            type: primary.type || prev.type || "input",
            frames: prev.frames || [],
            frameMap: map,
            // Explicit map already encodes deletes; sparse mode keeps ranges.
            deletedSourceRanges: map.length ? [] : (prev.deletedSourceRanges || []),
            sourceFrameCount: prev.sourceFrameCount || primary.sourceFrameCount || map.length || 0,
        };
        if (map.length) this.timeline.totalFrames = map.length;
    }

    async _applyLoadedVideo({ fileName, relPath, subfolder, type, statusPrefix }) {
        const prep = await this._prepareVideoFrames({ fileName, relPath, subfolder, type, statusPrefix });
        const { totalFrames, store, viewUrl } = prep;

        this._storageWidth = store.width;
        this._storageHeight = store.height;
        const clip = this._buildClipRecord(prep);

        this.timeline.videoClips = [clip];
        this.setSparseVideoFrames(totalFrames);
        this._syncPrimaryVideoFromClips([]);
        this._setSingleSegment(totalFrames);

        this._clearPreviewVideos(true);
        this._previewVideo = this._getPreviewVideoForClip(0);
        if (this._previewVideo && viewUrl) this._previewVideo.src = viewUrl;

        // Force stage to drop any previous media before binding the new clip.
        this._stageClipIndex = -1;
        if (this.stageVideo) {
            this.stageVideo.pause();
            this.stageVideo.removeAttribute("src");
            this.stageVideo.load();
        }
        this.currentFrame = 0;

        if (this.totalFramesWidget) this.totalFramesWidget.value = totalFrames;
        this.syncOutputUIFromTimeline();
        this.updateVideoNameLabel();
        this._prefetchSegmentThumbs(0, Math.min(totalFrames, THUMB_PREFETCH_BATCH * 4));
        this.updateStageVisibility();
        this._syncStagePreview(0, { force: true });
        this.commit(false, { syncTimeline: true });
    }

    async _applyAppendedVideo({ fileName, relPath, subfolder, type, statusPrefix }) {
        const prep = await this._prepareVideoFrames({
            fileName, relPath, subfolder, type, statusPrefix,
            syncNativeFps: false,
        });
        const { totalFrames, store } = prep;

        this._ensureVideoClipsArray();
        const clipIndex = this.timeline.videoClips.length;
        const clip = this._buildClipRecord(prep);
        this.timeline.videoClips.push(clip);

        const prevTotal = this.getTotalFrames();
        if (!this.getFrameMap().length && prevTotal > 0) {
            this.materializeFrameMap();
        }
        const newEntries = buildClipFrameMap(clipIndex, totalFrames);
        const map = [...this.getFrameMap(), ...newEntries];
        this.setFrameMap(map);
        this.timeline.totalFrames = map.length;
        this._syncPrimaryVideoFromClips(map);

        this._getPreviewVideoForClip(clipIndex);

        this.timeline.segments.push({
            id: uid(),
            start: prevTotal,
            length: totalFrames,
            prompt: "",
            taskType: "",
            refs: [],
            referenceVideo: {},
            videoClipId: clip.id,
        });

        if (this.totalFramesWidget) this.totalFramesWidget.value = map.length;
        this.selectedIndex = this.timeline.segments.length - 1;
        this.currentFrame = prevTotal;
        if (this.seekBar) {
            this.seekBar.max = Math.max(0, map.length - 1);
            this.seekBar.value = this.currentFrame;
        }

        this.normalizeSegments();
        this.syncOutputUIFromTimeline();
        this.updateVideoNameLabel();
        this._prefetchSegmentThumbs(prevTotal, Math.min(prevTotal + totalFrames, prevTotal + THUMB_PREFETCH_BATCH * 4));
        this.updateStageVisibility();
        this.commit(false, { syncTimeline: true });
    }

    async probeVideoMetadata(url) {
        const video = document.createElement("video");
        video.src = url;
        video.muted = true;
        video.playsInline = true;
        video.preload = "metadata";
        await new Promise((res, rej) => {
            video.onloadedmetadata = () => res();
            video.onerror = () => rej(new Error("无法读取视频元数据"));
        });
        return {
            width: video.videoWidth || 0,
            height: video.videoHeight || 0,
            duration: video.duration || 0,
        };
    }

    onNodeResize() {
        if (this.isPlaying || this._pauseSettling) return;
        this._resetLayoutStyles();
        this.applyZoomWidth();
        this.scheduleRender();
    }

    applyZoomWidth() {
        if (!this.canvas) return;
        if (this.zoom <= 1) {
            this.canvas.style.width = "100%";
            return;
        }
        const base = this.viewport?.clientWidth || 960;
        this.canvas.style.width = `${Math.max(base, base * this.zoom)}px`;
    }

    adjustZoom(delta) {
        this.zoom = clamp(this.zoom + delta, 1, 10);
        this.zoomSlider.value = this.zoom;
        this.applyZoomWidth();
        this.scheduleRender();
    }

    frameToX(frame, width) { return (frame / Math.max(1, this.getTotalFrames())) * width; }
    xToFrame(x, width) { return clamp(Math.round((x / width) * this.getTotalFrames()), 0, this.getTotalFrames()); }

    getLayoutWidth() {
        return this._drawWidth
            || this.canvas?.getBoundingClientRect().width
            || this.canvas?.offsetWidth
            || this.viewport?.clientWidth
            || 0;
    }

    getMousePos(e) {
        const rect = this.canvas.getBoundingClientRect();
        const layoutW = this.getLayoutWidth();
        const layoutH = this.canvasHeight || (RULER_H + SEG_LABEL_H + TRACK_H);
        const scaleX = rect.width > 0 ? layoutW / rect.width : 1;
        const scaleY = rect.height > 0 ? layoutH / rect.height : 1;
        return {
            x: (e.clientX - rect.left) * scaleX,
            y: (e.clientY - rect.top) * scaleY,
        };
    }

    /** Shared draw + hit geometry for per-segment run checkboxes (segment top-left). */
    _runCheckGeometry(seg, width) {
        const x0 = this.frameToX(seg.start, width);
        const size = RUN_CHECK_SIZE;
        const boxX = x0 + 5;
        const boxY = TRACK_Y + 5;
        return {
            boxX,
            boxY,
            size,
            hitX0: boxX - RUN_CHECK_HIT_PAD_X,
            hitY0: boxY - RUN_CHECK_HIT_PAD_Y,
            hitX1: boxX + size + RUN_CHECK_HIT_PAD_X,
            hitY1: boxY + size + RUN_CHECK_HIT_PAD_Y,
        };
    }

    hitTest(x, y) {
        const width = this.getLayoutWidth();
        if (!width) return null;
        const segs = this._previewSegments || this.timeline.segments;
        const phx = this.frameToX(this.currentFrame, width);

        if (y <= RULER_H) {
            if (Math.abs(x - phx) <= HANDLE_PX) return { type: "playhead" };
            return { type: "ruler" };
        }

        // Label band is range text only (not a drag / toggle surface).
        if (y < TRACK_Y) return null;

        // Checkbox corner wins over generic segment hit (same toggle action either way
        // in run-select mode; keeps hit type accurate for cursor / future hooks).
        if (this.isRunSelectEnabled() && segs.length >= 2) {
            for (let i = segs.length - 1; i >= 0; i--) {
                const g = this._runCheckGeometry(segs[i], width);
                if (x >= g.hitX0 && x <= g.hitX1 && y >= g.hitY0 && y <= g.hitY1) {
                    return { type: "run-check", index: i };
                }
            }
        }

        for (let i = segs.length - 1; i >= 0; i--) {
            const seg = segs[i];
            const x0 = this.frameToX(seg.start, width);
            const x1 = this.frameToX(seg.start + seg.length, width);
            const isLast = i === segs.length - 1;
            const insideX = isLast ? (x >= x0 && x <= x1) : (x >= x0 && x < x1);
            if (insideX && y >= TRACK_Y && y <= TRACK_Y + TRACK_H) {
                return { type: "segment", index: i };
            }
        }

        for (let i = 0; i < segs.length; i++) {
            const seg = segs[i];
            const x0 = this.frameToX(seg.start, width);
            const x1 = this.frameToX(seg.start + seg.length, width);
            if (y < TRACK_Y || y > TRACK_Y + TRACK_H) continue;
            if (Math.abs(x - x0) <= HANDLE_PX) return { type: "edge", index: i, edge: "left" };
            if (Math.abs(x - x1) <= HANDLE_PX) return { type: "edge", index: i, edge: "right" };
        }

        if (Math.abs(x - phx) <= HANDLE_PX) return { type: "playhead" };
        return null;
    }

    onMouseDown(e) {
        if (e.button !== 0) return;
        // Keep LiteGraph / node drag from eating timeline clicks.
        stopDomEvent(e);
        e.preventDefault();
        const { x, y } = this.getMousePos(e);
        const hit = this.hitTest(x, y);
        if (!hit) return;
        const width = this.getLayoutWidth();
        if (hit.type === "playhead" || hit.type === "ruler") {
            this.currentFrame = this.xToFrame(x, width);
            this._drag = { kind: "playhead" };
        } else if (hit.type === "run-check") {
            this.toggleSegmentRun(hit.index);
            this._drag = null;
        } else if (hit.type === "segment") {
            this.selectedIndex = hit.index;
            this.updateSelectionUI();
            if (this.timeline.segments.length >= 2) {
                this._drag = {
                    kind: "segment-pending",
                    index: hit.index,
                    x0: x,
                    y0: y,
                    fromRank: this._visualRankFromArrayIndex(hit.index),
                };
            } else {
                this._drag = { kind: "segment" };
            }
        } else if (hit.type === "edge") {
            this.selectedIndex = hit.index;
            this.updateSelectionUI();
            this._drag = { kind: "edge", index: hit.index, edge: hit.edge };
            this._edgeSnapshot = JSON.parse(JSON.stringify(this.timeline.segments));
        }
        this.scheduleRender();
    }

    onMouseMove(e) {
        if (!this._drag) return;
        const { x, y } = this.getMousePos(e);
        const width = this.getLayoutWidth();
        const frame = this.xToFrame(x, width);

        if (this._drag.kind === "segment-pending") {
            if (Math.hypot(x - this._drag.x0, y - this._drag.y0) > 6) {
                this._drag = {
                    kind: "reorder",
                    fromRank: this._drag.fromRank,
                    index: this._drag.index,
                };
                this._reorderFromRank = this._drag.fromRank;
                this._reorderDropRank = this._drag.fromRank;
                this.canvas.classList.add("bd-grabbing");
            }
            return;
        }

        if (this._drag.kind === "playhead") {
            this.currentFrame = frame;
        } else if (this._drag.kind === "reorder") {
            this._reorderDropRank = this._computeReorderDropRank(frame, this._drag.fromRank);
            this.scheduleRender();
            return;
        } else if (this._drag.kind === "edge") {
            const segs = this._edgeSnapshot.map((s) => ({ ...s }));
            const i = this._drag.index;
            const seg = segs[i];
            if (this._drag.edge === "left") {
                const prev = segs[i - 1];
                const minStart = prev ? prev.start + MIN_SEG : 0;
                const maxStart = seg.start + seg.length - MIN_SEG;
                seg.start = clamp(frame, minStart, maxStart);
                seg.length = (this._edgeSnapshot[i].start + this._edgeSnapshot[i].length) - seg.start;
                if (prev) prev.length = seg.start - prev.start;
            } else {
                const next = segs[i + 1];
                const minEnd = seg.start + MIN_SEG;
                const maxEnd = next ? next.start + next.length : this.getTotalFrames();
                const end = clamp(frame, minEnd, maxEnd);
                seg.length = end - seg.start;
                if (next) {
                    next.start = end;
                    next.length = (this._edgeSnapshot[i + 1].start + this._edgeSnapshot[i + 1].length) - end;
                }
            }
            this._previewSegments = segs;
        }
        this.scheduleRender();
    }

    onMouseUp() {
        if (this._drag?.kind === "edge" && this._previewSegments) {
            this.timeline.segments = this._previewSegments;
            this._previewSegments = null;
            this.commit();
        } else if (this._drag?.kind === "reorder") {
            const toRank = this._reorderDropRank;
            if (toRank >= 0 && toRank !== this._drag.fromRank) {
                this.reorderSegmentsByRank(this._drag.fromRank, toRank);
                this.commit(false, { syncTimeline: true });
            }
            this._reorderDropRank = -1;
            this._reorderFromRank = -1;
            this.canvas.classList.remove("bd-grabbing");
        } else if (this._drag) {
            this.seekBar.value = this.currentFrame;
            this.scheduleRender();
        }
        this._drag = null;
        this._edgeSnapshot = null;
    }

    addSplitAtMouse(e) {
        const { x } = this.getMousePos(e);
        this.splitAtFrame(this.xToFrame(x, this.getLayoutWidth()));
    }

    splitAtFrame(frame) {
        if (this.isGenMode()) {
            this.genSplitAtFrame(frame);
            return;
        }
        const total = this.getTotalFrames();
        if (frame <= MIN_SEG || frame >= total - MIN_SEG) return;
        const newSegs = [];
        for (const seg of [...this.timeline.segments].sort((a, b) => a.start - b.start)) {
            const end = seg.start + seg.length;
            if (frame > seg.start && frame < end) {
                newSegs.push({ ...seg, length: frame - seg.start });
                newSegs.push({ id: uid(), start: frame, length: end - frame, prompt: "", taskType: "", refs: [], referenceVideo: {} });
            } else newSegs.push({ ...seg });
        }
        this.timeline.segments = newSegs;
        this.commit();
    }

    equalSplit() {
        if (this.isGenMode()) {
            this.genEqualSplit();
            return;
        }
        const n = parseInt(this.equalCountInput?.value || "2", 10);
        if (!n || n < 2) return;
        const total = this.getTotalFrames();
        if (total < MIN_SEG * 2) return;
        const maxSeg = Math.floor(total / MIN_SEG);
        const count = clamp(n, 2, Math.max(2, maxSeg || 2));
        if (this.equalCountInput) this.equalCountInput.value = String(count);

        const points = new Set([0, total]);
        const clipBounds = this.getClipBoundaries();
        for (const b of clipBounds) {
            if (b > 0 && b < total) points.add(b);
        }
        for (let i = 1; i < count; i++) {
            const p = Math.round((i * total) / count);
            if (p > 0 && p < total) points.add(p);
        }

        const forced = new Set([0, total, ...clipBounds]);
        const newSegs = this._buildSegmentsFromSplitPoints([...points], forced);
        if (!newSegs?.length) return;
        this.timeline.segments = newSegs;
        this.commit();
    }

    deleteSelectedSegment() {
        if (this.isGenMode()) {
            this.genDeleteSelectedSegment();
            return;
        }
        if (this.isImageBatch()) {
            this.genDeleteSelectedSegment();
            return;
        }
        const idx = this.selectedIndex;
        const seg = this.timeline.segments[idx];
        if (!seg) return;

        const start = Math.max(0, parseInt(seg.start, 10) || 0);
        const len = Math.max(0, parseInt(seg.length, 10) || 0);

        // Remove segment UI entry first, then cut matching frames from the
        // logical timeline so preview / export no longer include that range.
        this.timeline.segments.splice(idx, 1);

        let total = this.getTotalFrames();
        let map = [];
        if (len > 0 && total > 0) {
            // Sparse uploads start with an empty frameMap; materialize so we can
            // splice out the deleted range from the source-frame mapping.
            if (!this.getFrameMap().length) this.materializeFrameMap();
            map = [...this.getFrameMap()];
            if (map.length) {
                const from = clamp(start, 0, map.length);
                const count = clamp(len, 0, map.length - from);
                if (count > 0) map.splice(from, count);
                this.setFrameMap(map);
                this._syncPrimaryVideoFromClips(map);
                total = map.length;
            } else {
                // Fallback: record deleted source ranges (kept across sync).
                const video = this.timeline.video || {};
                video.deletedSourceRanges = video.deletedSourceRanges || [];
                const srcStart = this.logicalToSourceFrame(start);
                video.deletedSourceRanges.push([srcStart, srcStart + len]);
                video.deletedSourceRanges.sort((a, b) => a[0] - b[0]);
                this.timeline.video = video;
                total = this.getTotalFrames();
                this.timeline.totalFrames = total;
                this._syncPrimaryVideoFromClips([]);
            }
        }

        this._thumbCache.clear();
        this._thumbPending.clear();
        // Invalidate stashed workspace — it still contains the deleted range.
        this.timeline.videoWorkspace = null;

        if (this.totalFramesWidget) this.totalFramesWidget.value = total;

        this.compactSegmentsAfterDelete();

        this.selectedIndex = clamp(idx, 0, Math.max(0, this.timeline.segments.length - 1));
        this.currentFrame = clamp(this.currentFrame, 0, Math.max(0, total - 1));
        if (this.seekBar) {
            this.seekBar.max = Math.max(0, total - 1);
            this.seekBar.value = this.currentFrame;
        }

        if (!total) {
            this.videoNameEl.textContent = "未上传视频";
            this.timeline.videoClips = [];
            this.timeline.video = {
                fileName: "",
                videoFile: "",
                subfolder: "",
                type: "input",
                frames: [],
                frameMap: [],
                width: 0,
                height: 0,
            };
            this._clearVideoState();
        } else {
            this.updateVideoNameLabel();
            this._prefetchSegmentThumbs(0, Math.min(total, THUMB_PREFETCH_BATCH * 4));
            this._syncStagePreview(this.currentFrame, { force: true });
            this.updateStageVisibility();
        }

        this.commit(false, { syncTimeline: true });
    }

    compactSegmentsAfterDelete() {
        const total = this.getTotalFrames();
        if (total <= 0) {
            this.timeline.segments = [];
            return;
        }
        const segs = [...this.timeline.segments].sort((a, b) => a.start - b.start);
        if (!segs.length) {
            this.timeline.segments = [{ id: uid(), start: 0, length: total, prompt: "", taskType: "", refs: [], referenceVideo: {} }];
            return;
        }
        let cursor = 0;
        const fixed = [];
        for (const seg of segs) {
            let length = seg.length ?? MIN_SEG;
            if (cursor + length > total) length = total - cursor;
            if (length < MIN_SEG) {
                if (fixed.length) fixed[fixed.length - 1].length += length;
                cursor += length;
                continue;
            }
            fixed.push({ ...seg, start: cursor, length, refs: seg.refs || [] });
            cursor += length;
        }
        if (!fixed.length) {
            this.timeline.segments = [{ id: uid(), start: 0, length: total, prompt: "", taskType: "", refs: [], referenceVideo: {} }];
        } else if (cursor < total) {
            fixed[fixed.length - 1].length += total - cursor;
        }
        this.timeline.segments = fixed;
    }

    getFrameImage(frameIndex) {
        return this._thumbCache.get(frameIndex) || null;
    }

    drawSegmentThumbnails(ctx, seg, startX, pxWidth, y0, h) {
        ctx.save();
        ctx.beginPath();
        ctx.rect(startX, y0 + 1, pxWidth, h - 2);
        ctx.clip();

        if (this.isGenBlank()) {
            ctx.fillStyle = "#0d0d0d";
            ctx.fillRect(startX, y0 + 1, pxWidth, h - 2);
            ctx.strokeStyle = "#333";
            ctx.lineWidth = 1;
            ctx.setLineDash([4, 4]);
            ctx.strokeRect(startX + 2, y0 + 4, pxWidth - 4, h - 8);
            ctx.setLineDash([]);
            ctx.fillStyle = "#888";
            ctx.font = "11px sans-serif";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            const fc = seg.frameCount ?? seg.length;
            ctx.fillText(`${fc}f`, startX + pxWidth / 2, y0 + h / 2 - 6);
            ctx.fillStyle = "#555";
            ctx.font = "10px sans-serif";
            ctx.fillText("空白画布", startX + pxWidth / 2, y0 + h / 2 + 8);
            ctx.restore();
            return;
        }

        if (this.isGenImage()) {
            const imgFile = this.isGlobalMode()
                ? this.timeline.global?.genImage?.imageFile
                : (seg.genImage?.imageFile || "");
            ctx.fillStyle = "#111";
            ctx.fillRect(startX, y0 + 1, pxWidth, h - 2);
            if (imgFile) {
                const cacheKey = `gen:${imgFile}`;
                let img = this._thumbCache.get(cacheKey);
                if (img?.naturalWidth) {
                    const ratio = img.naturalWidth / img.naturalHeight;
                    let dw = pxWidth - 4, dh = dw / ratio;
                    if (dh > h - 4) { dh = h - 4; dw = dh * ratio; }
                    ctx.drawImage(img, startX + (pxWidth - dw) / 2, y0 + (h - dh) / 2, dw, dh);
                } else if (!this._thumbPending.has(cacheKey)) {
                    this._thumbPending.add(cacheKey);
                    const el = new Image();
                    el.crossOrigin = "anonymous";
                    el.onload = () => {
                        this._thumbCache.set(cacheKey, el);
                        this._thumbPending.delete(cacheKey);
                        this.scheduleRender();
                    };
                    el.onerror = () => this._thumbPending.delete(cacheKey);
                    el.src = refViewUrl(imgFile);
                }
            } else {
                ctx.fillStyle = "#666";
                ctx.font = "12px sans-serif";
                ctx.textAlign = "center";
                ctx.textBaseline = "middle";
                ctx.fillText("点击上传源图片", startX + pxWidth / 2, y0 + h / 2);
            }
            ctx.restore();
            return;
        }

        ctx.fillStyle = "#000";
        ctx.fillRect(startX, y0 + 1, pxWidth, h - 2);
        if (!this.hasVideo()) {
            ctx.fillStyle = "#666";
            ctx.font = "12px sans-serif";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText("点击「上传视频」", startX + pxWidth / 2, y0 + h / 2);
            ctx.restore();
            return;
        }
        const thumbW = Math.max(32, pxWidth / Math.max(1, Math.min(MAX_THUMBS_PER_SEGMENT, Math.ceil(seg.length / 4))));
        const step = Math.max(1, Math.floor(seg.length / Math.max(1, Math.ceil(pxWidth / thumbW))));
        let drawn = 0;
        for (let f = seg.start; f < seg.start + seg.length && drawn < MAX_THUMBS_PER_SEGMENT; f += step, drawn++) {
            this._queueThumbPrefetch(f);
            const img = this.getFrameImage(f);
            const tx = startX + ((f - seg.start) / seg.length) * pxWidth;
            if (img?.naturalWidth) {
                const ratio = img.naturalWidth / img.naturalHeight;
                let dw = thumbW, dh = thumbW / ratio;
                if (dh > h - 2) { dh = h - 2; dw = dh * ratio; }
                ctx.drawImage(img, tx, y0 + (h - dh) / 2, dw, dh);
            } else {
                ctx.fillStyle = "#333";
                ctx.fillRect(tx, y0 + 2, Math.max(8, thumbW * 0.6), h - 4);
            }
        }
        ctx.restore();
    }

    _drawSegmentRunCheck(x, y, enabled) {
        const ctx = this.ctx;
        const s = RUN_CHECK_SIZE;
        ctx.save();
        // Opaque plate so the control never blends into timeline chrome.
        ctx.fillStyle = "#0e0e0e";
        ctx.fillRect(x - 1, y - 1, s + 2, s + 2);
        ctx.fillStyle = enabled ? "#1a3a2a" : "#1c1c1c";
        ctx.strokeStyle = enabled ? "#4fff8f" : "#888";
        ctx.lineWidth = 1;
        ctx.fillRect(x, y, s, s);
        ctx.strokeRect(x + 0.5, y + 0.5, s - 1, s - 1);
        if (enabled) {
            ctx.fillStyle = "#4fff8f";
            ctx.font = "11px sans-serif";
            ctx.textAlign = "left";
            ctx.textBaseline = "alphabetic";
            ctx.fillText("✓", x + 2, y + 11);
        }
        ctx.restore();
    }

    drawPromptOverlay(ctx, seg, startX, pxWidth, y0, h) {
        const prompt = this.getDisplayPrompt(seg);
        if (!prompt || pxWidth < 24) return;
        const overlayH = Math.round(h * 0.22);
        const overlayY = y0 + h - overlayH;
        ctx.save();
        ctx.beginPath();
        ctx.rect(startX, overlayY, pxWidth, overlayH);
        ctx.clip();
        ctx.fillStyle = "rgba(0,0,0,0.65)";
        ctx.fillRect(startX, overlayY, pxWidth, overlayH);
        ctx.font = `${Math.min(11, overlayH * 0.55)}px sans-serif`;
        ctx.fillStyle = "#e0e3ed";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        let label = prompt;
        const maxW = pxWidth - 10;
        if (ctx.measureText(label).width > maxW) {
            while (label.length > 0 && ctx.measureText(label + "…").width > maxW) label = label.slice(0, -1);
            label += "…";
        }
        ctx.fillText(label, startX + pxWidth / 2, overlayY + overlayH / 2);
        ctx.restore();
    }

    render() {
        if (this.isPlaying) {
            this.renderTimelineOnly();
            return;
        }
        const width = this.canvas?.getBoundingClientRect().width || this.canvas?.offsetWidth || 0;
        if (!width) return;
        this._drawWidth = width;
        this._drawTimelineCanvas(width);
        this._updateTimelineDom();
        this._syncStagePreview(this.currentFrame);
    }

    renderTimelineOnly() {
        const width = this._playCanvasWidth
            || this.viewport?.clientWidth
            || this.canvas?.getBoundingClientRect().width
            || this.canvas?.offsetWidth
            || this.node?.size?.[0]
            || 0;
        if (!width) return;
        this._drawWidth = width;
        this._drawTimelineCanvas(width);
        this._syncStagePreview(this.currentFrame);
    }

    _drawTimelineCanvas(width) {
        const height = this.canvasHeight;
        const dpr = window.devicePixelRatio || 1;
        const bw = Math.round(width * dpr);
        const bh = Math.round(height * dpr);
        if (this.canvas.width !== bw || this.canvas.height !== bh) {
            this.canvas.width = bw;
            this.canvas.height = bh;
            this.canvas.style.height = `${height}px`;
            this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        }
        this.ctx.clearRect(0, 0, width, height);

        const total = this.getTotalFrames();
        const fps = this.getFrameRate();
        const segs = this._previewSegments || this.timeline.segments;

        this.ctx.fillStyle = "#252525";
        this.ctx.fillRect(0, 0, width, RULER_H);
        this.ctx.fillStyle = "#888";
        this.ctx.font = "10px sans-serif";
        const durationSec = total / Math.max(fps, 0.001);
        const stepSec = Math.max(1, durationSec / 10);
        for (let s = 0; s < durationSec; s += stepSec) {
            const f = Math.min(total - 1, Math.round(s * fps));
            const x = this.frameToX(f, width);
            this.ctx.fillRect(x, RULER_H - 6, 1, 6);
            const label = Math.abs(s - Math.round(s)) < 0.01 ? `${Math.round(s)}.00` : s.toFixed(1);
            this.ctx.fillText(label, x + 2, 11);
        }
        if (total > 0) {
            const endF = total - 1;
            const endX = this.frameToX(endF, width);
            this.ctx.fillRect(endX, RULER_H - 6, 1, 6);
            const endLabel = durationSec.toFixed(2);
            const textW = this.ctx.measureText(endLabel).width;
            this.ctx.fillText(endLabel, Math.max(2, endX - textW - 2), 11);
        }

        // Frame-range labels above each segment (1-based inclusive, e.g. 1-10).
        this.ctx.fillStyle = "#1a1a1a";
        this.ctx.fillRect(0, RULER_H, width, SEG_LABEL_H);
        this.ctx.font = "10px sans-serif";
        this.ctx.textBaseline = "middle";
        for (let i = 0; i < segs.length; i++) {
            const seg = segs[i];
            const x0 = this.frameToX(seg.start, width);
            const x1 = this.frameToX(seg.start + seg.length, width);
            const pxW = Math.max(0, x1 - x0);
            if (pxW < 8 || seg.length <= 0) continue;
            const a = seg.start + 1;
            const b = seg.start + seg.length;
            const rangeText = `${a}-${b}`;
            this.ctx.fillStyle = i === this.selectedIndex ? "#eee" : "#9a9a9a";
            let draw = rangeText;
            if (this.ctx.measureText(draw).width > pxW - 6) {
                while (draw.length > 1 && this.ctx.measureText(`${draw}…`).width > pxW - 6) {
                    draw = draw.slice(0, -1);
                }
                draw = draw.length < rangeText.length ? `${draw}…` : draw;
            }
            this.ctx.fillText(draw, x0 + 4, RULER_H + SEG_LABEL_H / 2);
        }

        this.ctx.fillStyle = "#111";
        this.ctx.fillRect(0, TRACK_Y, width, TRACK_H);

        const clipBounds = this.getClipBoundaries();
        if (clipBounds.length) {
            this.ctx.strokeStyle = "rgba(102,170,255,0.55)";
            this.ctx.lineWidth = 2;
            this.ctx.setLineDash([5, 4]);
            for (const b of clipBounds) {
                const bx = this.frameToX(b, width);
                this.ctx.beginPath();
                this.ctx.moveTo(bx, TRACK_Y);
                this.ctx.lineTo(bx, TRACK_Y + TRACK_H);
                this.ctx.stroke();
            }
            this.ctx.setLineDash([]);
        }

        const reordering = this._drag?.kind === "reorder";
        const dragFromRank = reordering ? this._drag.fromRank : -1;

        for (let i = 0; i < segs.length; i++) {
            const seg = segs[i];
            const x0 = this.frameToX(seg.start, width);
            const x1 = this.frameToX(seg.start + seg.length, width);
            const pxW = x1 - x0;
            const sel = i === this.selectedIndex;
            const running = i === this._runHighlightSeg;
            const runOn = this.isSegmentRunEnabled(i);
            if (this.isRunSelectEnabled() && segs.length >= 2 && !runOn) {
                this.ctx.globalAlpha = 0.32;
            } else if (reordering && this._visualRankFromArrayIndex(i) === dragFromRank) {
                this.ctx.globalAlpha = 0.38;
            }
            this.drawSegmentThumbnails(this.ctx, seg, x0, pxW, TRACK_Y, TRACK_H);
            this.drawPromptOverlay(this.ctx, seg, x0, pxW, TRACK_Y, TRACK_H);
            const clipIdx = this.getSegmentClipIndex(seg);
            const clipColor = CLIP_SEGMENT_COLORS[clipIdx % CLIP_SEGMENT_COLORS.length];
            this.ctx.strokeStyle = running ? "#4fff8f" : sel ? "#fff" : clipColor;
            this.ctx.lineWidth = running ? 3 : sel ? 2.5 : 1.5;
            this.ctx.strokeRect(x0 + 0.5, TRACK_Y + 0.5, pxW - 1, TRACK_H - 1);
            this.ctx.fillStyle = "#ffcc00";
            this.ctx.fillRect(x0 - 2, TRACK_Y + TRACK_H / 2 - 12, 4, 24);
            this.ctx.fillRect(x1 - 2, TRACK_Y + TRACK_H / 2 - 12, 4, 24);
            this.ctx.globalAlpha = 1;
            // Checkbox on top-left; drawn last so it stays clear on dimmed segments.
            if (this.isRunSelectEnabled() && segs.length >= 2 && pxW >= RUN_CHECK_SIZE + 8) {
                const g = this._runCheckGeometry(seg, width);
                this._drawSegmentRunCheck(g.boxX, g.boxY, runOn);
            }
        }

        if (reordering && this._reorderDropRank >= 0) {
            const insertFrame = this._getReorderInsertFrame(this._reorderDropRank, dragFromRank);
            const ix = this.frameToX(insertFrame, width);
            this.ctx.strokeStyle = "#4fff8f";
            this.ctx.lineWidth = 3;
            this.ctx.beginPath();
            this.ctx.moveTo(ix, TRACK_Y);
            this.ctx.lineTo(ix, TRACK_Y + TRACK_H);
            this.ctx.stroke();
        }

        const phx = this.frameToX(this.currentFrame, width);
        this.ctx.strokeStyle = "#ff4444";
        this.ctx.lineWidth = 2;
        this.ctx.beginPath();
        this.ctx.moveTo(phx, 0);
        this.ctx.lineTo(phx, height);
        this.ctx.stroke();

        const exportCap = this.getMaxExportFrames();
        const exportTotal = this.getExportFrameTotal();
        if (exportCap > 0 && exportTotal < total) {
            const capX = this.frameToX(exportTotal, width);
            this.ctx.fillStyle = "rgba(0,0,0,0.35)";
            this.ctx.fillRect(capX, TRACK_Y, width - capX, TRACK_H);
            this.ctx.strokeStyle = "#66aaff";
            this.ctx.lineWidth = 2;
            this.ctx.setLineDash([4, 3]);
            this.ctx.beginPath();
            this.ctx.moveTo(capX, 0);
            this.ctx.lineTo(capX, height);
            this.ctx.stroke();
            this.ctx.setLineDash([]);
            this.ctx.fillStyle = "#66aaff";
            this.ctx.font = "10px sans-serif";
            this.ctx.fillText(`导出 ${exportTotal}`, capX + 4, TRACK_Y + 12);
        }
    }

    _updateTimelineDom({ skipSeek = false } = {}) {
        const segs = this._previewSegments || this.timeline.segments;
        const totalFrames = Math.max(0, this.getTotalFrames());
        const cur = this.formatTime(this.currentFrame);
        const total = this.formatTime(totalFrames);
        if (this.timecodeEl) this.timecodeEl.textContent = `${cur}s`;
        if (this.playerTimecodeEl) this.playerTimecodeEl.textContent = `${cur} / ${total}`;
        if (this.frameTotalEl) this.frameTotalEl.textContent = String(totalFrames);
        if (this.frameInputEl) {
            this.frameInputEl.max = String(Math.max(1, totalFrames));
            // Don't overwrite while the user is typing a target frame.
            if (document.activeElement !== this.frameInputEl) {
                this.frameInputEl.value = String(totalFrames > 0 ? this.currentFrame + 1 : 1);
            }
        }
        if (!skipSeek && this.seekBar && +this.seekBar.value !== this.currentFrame) {
            this.seekBar.value = this.currentFrame;
        }
        if (this.seekBar) this.seekBar.max = Math.max(0, totalFrames - 1);
        const seg = segs[this.selectedIndex];
        if (seg) this.boundsEl.textContent = `Start: ${this.formatTime(seg.start)} | End: ${this.formatTime(seg.start + seg.length)}`;
    }

    /** Jump to an exact 0-based logical frame; syncs seek bar, preview, playhead. */
    seekToFrame(frame, { fromUi = false } = {}) {
        const total = this.getTotalFrames();
        if (total < 1) return;
        if (this.isPlaying) this._stopPlay();
        const next = clamp(Math.round(Number(frame) || 0), 0, total - 1);
        this.currentFrame = next;
        if (this.seekBar) {
            this.seekBar.max = Math.max(0, total - 1);
            this.seekBar.value = next;
        }
        this._syncStagePreview(next, { force: true });
        this._updateTimelineDom({ skipSeek: true });
        // Select the segment that contains this frame for editing context.
        const segs = this.timeline.segments || [];
        for (let i = 0; i < segs.length; i++) {
            const s = segs[i];
            if (next >= s.start && next < s.start + s.length) {
                if (this.selectedIndex !== i) {
                    this.selectedIndex = i;
                    this.updateSelectionUI();
                }
                break;
            }
        }
        this.scheduleRender();
        if (fromUi) this._queueThumbPrefetch?.(next);
    }

    stepFrame(delta) {
        const total = this.getTotalFrames();
        if (total < 1) return;
        this.seekToFrame(this.currentFrame + (Number(delta) || 0), { fromUi: true });
    }

    formatTime(frames) { return (frames / this.getFrameRate()).toFixed(2); }

    updateSelectionUI() {
        this.timeline.global = this.timeline.global || { taskType: "", prompt: "", refs: [] };
        if (this.globalTask) this.globalTask.value = this.timeline.global.taskType || "";
        if (this.globalPrompt) this.globalPrompt.value = this.timeline.global.prompt || "";
        this.syncNegativeFromWidget();

        const hideTimeline = this.isImageBatch() || this.isGenMode();
        const seg = this.isGlobalMode() ? null : this.timeline.segments[this.selectedIndex];
        this.updateReferenceImageVisibility({ hideTimeline, seg: seg || null });

        if (this.isGlobalMode() && taskUsesReferenceImages(this.getTaskKey())) {
            this.renderRefSlots(this.timeline.global.refs, this.globalRefsBox, true);
        }
        const refVideoKey = this.isGlobalMode()
            ? this.getTaskKey()
            : resolveTaskKey(seg?.taskType || this.timeline.global?.taskType || this.getTaskKey());
        if (taskUsesReferenceVideo(refVideoKey)) {
            this.renderRefVideoSlot();
        }
        if (this.isGenImage() && this.isGlobalMode()) {
            this.renderGenSrcSlot(
                this.genGlobalImg,
                this.timeline.global?.genImage?.imageFile,
                "点击上传源图片",
            );
        }
        if (this.isGenMode() && this.isGlobalMode()) {
            const defFc = this.timeline.gen?.defaultFrameCount ?? defaultFrameCount(this.getTaskKey());
            if (this.genDefaultFc) this.genDefaultFc.value = defFc;
        }

        if (this.isGlobalMode()) return;

        if (!seg) return;
        const fps = this.getFrameRate();
        const segKey = resolveTaskKey(seg.taskType || this.timeline.global?.taskType || this.getTaskKey());
        this.segLabel.textContent = `片段 ${this.selectedIndex + 1}`;
        let info;
        if (this.isGenMode()) {
            const fc = seg.frameCount ?? seg.length;
            info = `${fc} 帧`;
            if (this.isGenImage()) info += seg.genImage?.imageFile ? " · 已上传图片" : " · 未上传图片";
        } else {
            info = `帧 ${seg.start}–${seg.start + seg.length} (${seg.length}f) · ${(seg.length / fps).toFixed(2)}s`;
            const clips = this.getVideoClips();
            if (clips.length > 1) {
                const clip = clips[this.getSegmentClipIndex(seg)];
                const clipName = clip?.fileName || clip?.videoFile || `视频 ${this.getSegmentClipIndex(seg) + 1}`;
                info += ` · ${clipName}`;
            }
            if (taskUsesReferenceVideo(segKey)) {
                info += seg.referenceVideo?.videoFile || seg.referenceVideo?.fileName
                    ? " · 已上传参考视频"
                    : " · 未上传参考视频";
            }
        }
        this.segInfo.textContent = info;
        this.segPrompt.value = seg.prompt || "";
        if (taskUsesReferenceImages(segKey)) {
            this.renderRefSlots(seg.refs, this.segRefsBox, false);
        }
        if (this.isGenImage() && !this.isGlobalMode()) {
            this.renderGenSrcSlot(this.genSegImg, seg.genImage?.imageFile, "点击上传片段源图片");
        }
        if (this.isGenMode() && !this.isGlobalMode()) {
            const fc = seg.frameCount ?? seg.length ?? defaultFrameCount(this.getTaskKey());
            if (this.genSegFc) this.genSegFc.value = fc;
        }
    }

    renderRefSlots(refs, box, isGlobal) {
        box.innerHTML = "";
        for (let i = 0; i < 5; i++) {
            const el = document.createElement("div");
            el.className = "bd-ref";
            el.title = `image${i} — 点击上传`;
            const ref = (refs || []).find((r) => Number(r.index ?? r.slot) === i);
            const tag = document.createElement("span");
            tag.className = "bd-ref-tag";
            tag.textContent = `image${i}`;
            el.appendChild(tag);
            if (ref?.imageFile) {
                el.classList.add("has-img");
                const img = document.createElement("img");
                img.src = refViewUrl(ref.imageFile);
                el.appendChild(img);
                const x = document.createElement("span");
                x.className = "x";
                x.textContent = "×";
                x.onclick = (e) => {
                    e.stopPropagation();
                    this.removeRef(isGlobal ? this.timeline.global : this.timeline.segments[this.selectedIndex], i);
                };
                el.appendChild(x);
            } else if (ref?.imageB64) {
                el.classList.add("has-img");
                const img = document.createElement("img");
                img.src = ref.imageB64.startsWith("data:") ? ref.imageB64 : `data:image/png;base64,${ref.imageB64}`;
                el.appendChild(img);
                const x = document.createElement("span");
                x.className = "x";
                x.textContent = "×";
                x.onclick = (e) => {
                    e.stopPropagation();
                    this.removeRef(isGlobal ? this.timeline.global : this.timeline.segments[this.selectedIndex], i);
                };
                el.appendChild(x);
            }
            el.onclick = () => this.pickRef(isGlobal ? this.timeline.global : this.timeline.segments[this.selectedIndex], i, isGlobal);
            box.appendChild(el);
        }
    }

    removeRef(target, index) {
        target.refs = (target.refs || []).filter((r) => Number(r.index ?? r.slot) !== index);
        this.commit();
    }

    pickRef(target, index, isGlobal) {
        const input = document.createElement("input");
        input.type = "file"; input.accept = "image/*";
        input.onchange = () => {
            const file = input.files?.[0];
            if (file) this.addRefFromFile(file, target, index, isGlobal);
        };
        input.click();
    }

    async addRefFromFile(file, target, slotIndex = null, isGlobal = null) {
        target.refs = target.refs || [];
        let index = slotIndex;
        if (index == null) {
            index = [0, 1, 2, 3, 4].find((i) => !target.refs.some((r) => Number(r.index ?? r.slot) === i));
            if (index == null) return;
        }
        try {
            const uploaded = await uploadToInput(file);
            const relPath = videoRelativePath(uploaded);
            target.refs = target.refs.filter((r) => Number(r.index ?? r.slot) !== index);
            target.refs.push({ index, imageFile: relPath, imageB64: "" });
            if (isGlobal) this.timeline.global = target;
            this.commit();
        } catch (err) {
            console.error("[BerniniDirector] ref upload failed:", err);
        }
    }

    onGlobalField(field, value) {
        this.timeline.global = this.timeline.global || { refs: [] };
        if (field === "taskType") {
            const prevTaskKey = resolveTaskKey(
                this.timeline.global?.taskType || this.globalTask?.value || this.taskTypeWidget?.value || "",
            );
            this.timeline.global[field] = value;
            const prevMode = this._directorMode || "video";
            if (this.globalTask && this.globalTask.value !== value) this.globalTask.value = value;
            if (this.taskTypeWidget) this.taskTypeWidget.value = value;
            if (prevTaskKey === "ads2v" && resolveTaskKey(value) !== "ads2v") {
                this._stopRefVideoPreviews();
            }
            this.applyTaskLayout(prevMode);
        } else {
            this.timeline.global[field] = value;
        }
        if (field === "prompt" && this.globalPromptWidget) this.globalPromptWidget.value = value;
        this.scheduleTimelineSync();
        this.scheduleRender();
    }

    onSegField(field, value) {
        const seg = this.timeline.segments[this.selectedIndex];
        if (!seg) return;
        seg[field] = value;
        this.scheduleTimelineSync();
        this.scheduleRender();
    }

    onNegativePrompt(value) {
        if (this.negativePromptWidget) this.negativePromptWidget.value = value;
        if (this.globalNegative && this.globalNegative.value !== value) this.globalNegative.value = value;
        if (this.segNegative && this.segNegative.value !== value) this.segNegative.value = value;
        this._markNodeDirtyLight();
    }

    toggleLoop() {
        this.isLooping = !this.isLooping;
        const btn = this.root.querySelector('[data-a="loop"]');
        btn.classList.toggle("active", this.isLooping);
        btn.title = this.isLooping
            ? "循环播放：已开启（播放到末尾后从头开始）"
            : "循环播放：已关闭（播放到末尾后停止）";
    }

    setRunProgress(detail) {
        if (!this.runStatusEl) return;
        const timelineTotal = this.timeline?.segments?.length || 0;
        const runTotal = Math.max(detail.segment_total || this.getRunProgressSegmentTotal(), 1);
        const runSeg = Math.max(1, detail.segment || 1);
        const timelineSeg = detail.timeline_segment ?? runSeg;
        const partialRun = !!detail.partial_run
            || (this.isRunSelectEnabled?.() && runTotal < timelineTotal);
        const phaseLabel = detail.phase_label || detail.phase || "运行中";
        const overallPct = detail.overall_max > 0
            ? Math.round((100 * detail.overall_value) / detail.overall_max)
            : 0;
        const phasePct = detail.phase_max > 0
            ? Math.round((100 * detail.phase_value) / detail.phase_max)
            : 0;
        const remain = Math.max(0, runTotal - runSeg);

        if (detail.phase === "finish") {
            this.runStatusEl.className = "bd-run-status done";
            this.runTitleEl.textContent = "运行状态：全部完成";
            this.runDetailEl.textContent = runTotal
                ? (this.isImageBatch()
                    ? (isVideoBatchTask(this.getTaskKey())
                        ? `共生成 ${runTotal} 组视频`
                        : `共生成 ${runTotal} 张图片`)
                    : (partialRun
                        ? `共处理 ${runTotal} 个选中片段`
                        : `共处理 ${runTotal} 个片段`))
                : "处理完成";
            this.runOverallEl.style.width = "100%";
            this.runPhaseEl.style.width = "100%";
            this._runHighlightSeg = -1;
            this.updateRunSelectUI();
            if (this.isImageBatch()) this.renderImageBatchGroups();
            else this.scheduleRender();
            return;
        }

        this.runStatusEl.className = "bd-run-status active";
        // Hide the pre-run "将运行 N 段" chip while progress is live — it sits
        // under the title in the same green accent and reads as a layout glitch.
        this.runSelectBar?.classList.add("hidden");
        this._runHighlightSeg = timelineSeg - 1;
        let title;
        if (detail.phase === "plan") {
            title = runTotal > 1 ? `共 ${runTotal} 段 · ${phaseLabel}` : phaseLabel;
        } else if (this.isImageBatch()) {
            title = `第 ${runSeg}/${runTotal} 组 · ${phaseLabel}`;
        } else if (partialRun) {
            title = `段 #${timelineSeg}（${runSeg}/${runTotal}）· ${phaseLabel}`;
        } else {
            title = `段 ${runSeg}/${runTotal} · ${phaseLabel}`;
        }
        if (phasePct > 0 && detail.phase !== "plan") {
            title += ` · ${phasePct}%`;
        }
        this.runTitleEl.textContent = title;
        const parts = [];
        if (detail.frames_label) parts.push(detail.frames_label);
        if (detail.task_key) parts.push(detail.task_key);
        parts.push(`整体 ${overallPct}%`);
        if (runTotal > 1) {
            parts.push(this.isImageBatch() ? `还剩 ${remain} 组` : `还剩 ${remain} 段`);
        }
        if (partialRun && timelineTotal > runTotal) {
            parts.push(`时间轴共 ${timelineTotal} 段`);
        }
        this.runDetailEl.textContent = parts.join(" · ");
        this.runOverallEl.style.width = `${overallPct}%`;
        this.runPhaseEl.style.width = `${phasePct}%`;
        if (this.isImageBatch()) this.renderImageBatchGroups();
        else this.scheduleRender();
    }

    clearRunProgress(title, detail) {
        if (!this.runStatusEl) return;
        this.runStatusEl.className = "bd-run-status idle";
        this.runTitleEl.textContent = title || "运行状态：待命";
        this.runDetailEl.textContent = detail || "队列执行时将显示当前片段与阶段进度";
        this.runOverallEl.style.width = "0%";
        this.runPhaseEl.style.width = "0%";
        this._runHighlightSeg = -1;
        this.updateRunSelectUI();
        if (this.isImageBatch()) this.renderImageBatchGroups();
        else this.scheduleRender();
    }

    setRunError(message) {
        if (!this.runStatusEl) return;
        this.runStatusEl.className = "bd-run-status error";
        this.runTitleEl.textContent = "运行状态：出错";
        this.runDetailEl.textContent = message || "执行中断，请查看终端日志";
        if (this.runOverallEl) this.runOverallEl.style.width = "0%";
        if (this.runPhaseEl) this.runPhaseEl.style.width = "0%";
        this._runHighlightSeg = -1;
        this.updateRunSelectUI();
        this.scheduleRender();
    }

    _stopPlay() {
        this.isPlaying = false;
        this._playHandoff = false;
        this._nativePlayFailed = false;
        this._pauseSettling = true;
        cancelAnimationFrame(this._playRaf);
        this._playRaf = null;
        this.stageVideo?.pause();
        this.root.querySelector('[data-a="play"]').textContent = "▶";
        this._resizeObserver?.disconnect();

        const w = this._playCanvasWidth;
        this._releasePlayLayoutLock();

        if (w) this._drawTimelineCanvas(w);
        this._updateTimelineDom({ skipSeek: true });
        this._syncStagePreview(this.currentFrame, { force: true });

        requestAnimationFrame(() => {
            requestAnimationFrame(() => {
                if (+this.seekBar.value !== this.currentFrame) {
                    this.seekBar.value = this.currentFrame;
                }
                this._observeViewportResize();
                const drawW = this.viewport?.clientWidth || w;
                if (drawW) this._drawTimelineCanvas(drawW);
                this._syncStagePreview(this.currentFrame, { force: true });
                this._pauseSettling = false;
            });
        });
    }

    async _beginNativePlay() {
        const total = this.getTotalFrames();
        if (total < 1) return;
        if (this.currentFrame >= total) this.currentFrame = 0;
        await this._ensureStageReadyForFrame(this.currentFrame);
        if (!this.isPlaying) return;
        const v = this.stageVideo;
        if (!v) return;
        try {
            await v.play();
        } catch {
            // Native play blocked/failed — keep isPlaying but drive via frame clock.
            this._nativePlayFailed = true;
        }
    }

    async _advanceNativePlayToNextClipOrEnd() {
        if (this._playHandoff || !this.isPlaying) return;
        this._playHandoff = true;
        try {
            const total = this.getTotalFrames();
            const range = this._logicalRangeForClip(this._stageClipIndex);
            const next = range.end < total ? range.end : -1;
            if (next >= 0) {
                this.currentFrame = next;
                await this._beginNativePlay();
                return;
            }
            if (this.isLooping) {
                this.currentFrame = 0;
                await this._beginNativePlay();
                return;
            }
            this.currentFrame = Math.max(0, total - 1);
            this._stopPlay();
        } finally {
            this._playHandoff = false;
        }
    }

    togglePlay() {
        if (this.isPlaying) {
            this._stopPlay();
            return;
        }
        const total = this.getTotalFrames();
        if (total < 1) return;

        this.isPlaying = true;
        this._nativePlayFailed = false;
        this.root.querySelector('[data-a="play"]').textContent = "⏸";
        this._lockPlayLayout();
        this._resizeObserver?.disconnect();

        if (this.currentFrame >= total) this.currentFrame = 0;
        this.renderTimelineOnly();
        this._updateTimelineDom();

        const useNative = !this._legacyFrames.length && !!this.stageVideo;
        if (useNative) {
            this._beginNativePlay();
        } else {
            this._syncStagePreview(this.currentFrame, { force: true });
        }

        const tick = () => {
            if (!this.isPlaying) return;
            const fps = Math.max(0.001, this.getFrameRate());

            if (useNative && this.stageVideo && !this._nativePlayFailed) {
                const v = this.stageVideo;
                const clipIndex = this._stageClipIndex >= 0 ? this._stageClipIndex : 0;
                const range = this._logicalRangeForClip(clipIndex);
                const lastLogical = Math.max(range.start, range.end - 1);
                const lastTime = this.getFrameMapEntry(lastLogical).frame / fps;
                const atMappedEnd = v.currentTime >= Math.max(0, lastTime - 0.04);
                const hasTimelineEdits = !!(
                    this.getFrameMap().length
                    || deletedSourceRanges(this.timeline.video || {}).length
                );
                // With deletes, file duration still includes removed tails — trust mapped end.
                const atMediaEnd = !hasTimelineEdits && (
                    v.ended || (v.duration > 0 && v.currentTime >= v.duration - 0.04)
                );

                if ((atMappedEnd || atMediaEnd) && !v.seeking && !this._playHandoff) {
                    this.currentFrame = lastLogical;
                    this.renderTimelineOnly();
                    this._updateTimelineDom();
                    this._advanceNativePlayToNextClipOrEnd();
                    if (this.isPlaying) this._playRaf = requestAnimationFrame(tick);
                    return;
                }

                if (!v.paused) {
                    const srcFrame = Math.max(0, Math.round(v.currentTime * fps));
                    let logical = this._logicalFromStageTime(clipIndex, v.currentTime);
                    const jumpToKept = () => {
                        const nextLogical = this._nextLogicalAfterSourceFrame(clipIndex, srcFrame);
                        if (nextLogical >= 0) {
                            const nextSrc = this.getFrameMapEntry(nextLogical).frame;
                            try { v.currentTime = nextSrc / fps; } catch { /* seek race */ }
                            return nextLogical;
                        }
                        return -1;
                    };
                    // Sparse deleted gap, or mid/leading gap vs mapped source.
                    if (logical < 0) {
                        const next = jumpToKept();
                        if (next < 0) {
                            this.currentFrame = lastLogical;
                            this.renderTimelineOnly();
                            this._updateTimelineDom();
                            this._advanceNativePlayToNextClipOrEnd();
                            if (this.isPlaying) this._playRaf = requestAnimationFrame(tick);
                            return;
                        }
                        logical = next;
                    } else {
                        const mapped = this.getFrameMapEntry(logical);
                        if (mapped.clip === clipIndex && mapped.frame !== srcFrame) {
                            // leading gap (mapped > src) or mid gap (mapped < src)
                            if (mapped.frame > srcFrame || mapped.frame < srcFrame) {
                                const next = mapped.frame > srcFrame ? logical : jumpToKept();
                                if (next < 0) {
                                    this.currentFrame = clamp(logical, 0, total - 1);
                                    this.renderTimelineOnly();
                                    this._updateTimelineDom();
                                    this._advanceNativePlayToNextClipOrEnd();
                                    if (this.isPlaying) this._playRaf = requestAnimationFrame(tick);
                                    return;
                                }
                                if (mapped.frame > srcFrame) {
                                    try { v.currentTime = mapped.frame / fps; } catch { /* seek race */ }
                                }
                                logical = next;
                            }
                        }
                    }
                    this.currentFrame = clamp(logical, 0, total - 1);
                    this.renderTimelineOnly();
                    const now = performance.now();
                    if (now - this._lastSeekUiMs > 66) {
                        this._updateTimelineDom();
                        this._lastSeekUiMs = now;
                    }
                }
            } else {
                // Legacy embedded frames (or native play unavailable): step by logical frame.
                this.currentFrame += 1;
                if (this.currentFrame >= total) {
                    if (this.isLooping) this.currentFrame = 0;
                    else {
                        this.currentFrame = total - 1;
                        this._stopPlay();
                        return;
                    }
                }
                this.renderTimelineOnly();
                this._syncStagePreview(this.currentFrame, { force: true });
                const now = performance.now();
                if (now - this._lastSeekUiMs > 80) {
                    this._updateTimelineDom();
                    this._lastSeekUiMs = now;
                }
            }
            this._playRaf = requestAnimationFrame(tick);
        };
        this._playRaf = requestAnimationFrame(tick);
    }
}

function findDirectorNode(nodeId) {
    const id = String(nodeId);
    const graph = app.graph ?? app.canvas?.graph;
    for (const node of graph?._nodes ?? graph?.nodes ?? []) {
        if (String(node.id) === id) return node;
    }
    return null;
}

function clearAllDirectorRunStatus() {
    const graph = app.graph ?? app.canvas?.graph;
    for (const node of graph?._nodes ?? graph?.nodes ?? []) {
        node._berniniEditor?.clearRunProgress?.();
    }
}

/** Old workflows may still list removed output slots (e.g. segment_images). */
function isBerniniDirectorNode(node) {
    const cls = node?.comfyClass || node?.type || "";
    return cls === "ComfyBerniniDirector";
}

function isDirectorNodeDef(nodeType, nodeData) {
    const cls = nodeType?.comfyClass || nodeData?.name || "";
    return cls === "ComfyBerniniDirector";
}

function stripDeprecatedDirectorOutputs(node) {
    if (!isBerniniDirectorNode(node) || !node.outputs?.length) return;
    const stale = new Set(["segment_images"]);
    for (let i = node.outputs.length - 1; i >= 0; i--) {
        if (stale.has(node.outputs[i]?.name)) {
            node.removeOutput(i);
        }
    }
}

/** Reorder legacy output links after slot layout changes. */
function migrateDirectorOutputLinks(node) {
    if (!isBerniniDirectorNode(node)) return;
    const graph = app.graph ?? app.canvas?.graph;
    const links = graph?.links;
    if (!links?.length) return;
    const outputs = node.outputs || [];
    const byName = Object.fromEntries(
        outputs.map((o, i) => [o?.name, i]).filter(([n]) => !!n)
    );

    for (const link of links) {
        if (!link || String(link.origin_id) !== String(node.id)) continue;
        const target = graph.getNodeById?.(link.target_id);
        const input = target?.inputs?.[link.target_slot];
        const inputType = (input?.type || "").toUpperCase();

        // Old layouts had report at slot 1 or 3 as STRING.
        if (inputType === "STRING" && byName.report != null && link.origin_slot !== byName.report) {
            link.origin_slot = byName.report;
            continue;
        }
        // Old layouts had fps last (slot 5) as FLOAT.
        if (inputType === "FLOAT" && byName.fps != null && link.origin_slot !== byName.fps) {
            link.origin_slot = byName.fps;
            continue;
        }
        // Old layouts had frame_count at slot 2 as INT.
        if (inputType === "INT" && byName.frame_count != null && link.origin_slot !== byName.frame_count) {
            link.origin_slot = byName.frame_count;
        }
    }
}

function normalizeDirectorOutputs(node) {
    stripDeprecatedDirectorOutputs(node);
    migrateDirectorOutputLinks(node);
}

app.registerExtension({
    name: "ComfyUI.BerniniDirectorPlugin",
    async setup() {
        const flushDirectors = () => {
            const graph = app.graph ?? app.canvas?.graph;
            for (const node of graph?._nodes ?? graph?.nodes ?? []) {
                node._berniniEditor?._promptEnhancer?.syncToWidgets?.();
                node._berniniEditor?.flushTimelineSync?.();
            }
        };
        if (app.queuePrompt && !app.queuePrompt._berniniPatched) {
            const orig = app.queuePrompt.bind(app);
            app.queuePrompt = function (...args) {
                flushDirectors();
                clearAllDirectorRunStatus();
                return orig(...args);
            };
            app.queuePrompt._berniniPatched = true;
        }

        api.addEventListener("bernini_director_progress", ({ detail }) => {
            findDirectorNode(detail?.node_id)?._berniniEditor?.setRunProgress?.(detail);
        });

        api.addEventListener("bernini_director_preview", ({ detail }) => {
            const editor = findDirectorNode(detail?.node_id)?._berniniEditor;
            if (!editor?.isImageBatch?.()) return;
            setImageBatchPreview(
                editor,
                detail?.segment_index ?? 0,
                detail?.image_b64 || "",
                { frames: detail?.frames, fps: detail?.fps },
            );
            editor.renderImageBatchGroups?.();
        });

        api.addEventListener("executing", ({ detail }) => {
            if (detail == null) return;
            const node = findDirectorNode(detail);
            const editor = node?._berniniEditor;
            if (!editor) return;
            editor.flushTimelineSync?.();
            if (editor.isImageBatch?.()) {
                for (const seg of editor.timeline.segments || []) {
                    seg.previewB64 = "";
                    seg.previewFrames = [];
                }
                editor.renderImageBatchGroups?.();
            }
            const segTotal = editor.getRunProgressSegmentTotal?.() ?? (editor.timeline?.segments?.length || 1);
            const timelineTotal = editor.timeline?.segments?.length || segTotal;
            editor.setRunProgress({
                node_id: detail,
                segment: 1,
                segment_total: segTotal,
                timeline_segment: 1,
                timeline_segment_total: timelineTotal,
                partial_run: editor.isRunSelectEnabled?.() && segTotal < timelineTotal,
                phase: "plan",
                phase_label: "解析时间轴 / 加载视频",
                phase_value: 0,
                phase_max: 1,
                overall_value: 0,
                overall_max: Math.max(1, segTotal * 6),
                remaining_segments: Math.max(0, segTotal - 1),
            });
        });

        api.addEventListener("execution_error", ({ detail }) => {
            const node = findDirectorNode(detail?.node_id);
            if (node?._berniniEditor) {
                node._berniniEditor.setRunError(detail?.exception_message || "执行出错");
            }
        });

        registerDirectorPromptEnhancerEvents(findDirectorNode);

        patchDirectorDomWidgetLayout();
        setTimeout(patchDirectorDomWidgetLayout, 500);
    },
    async loadedGraphNode(node) {
        if (isBerniniDirectorNode(node)) normalizeDirectorOutputs(node);
        if (node._directorDomWidget) {
            finalizeDirectorWidgetOrder(node);
            ensureDirectorDomWidgetWidth(node);
            bindDirectorDomWidgetSizing(node, node._directorDomWidget, () => node._berniniEditor);
            initDirectorEditor(node);
            node._berniniEditor?.scheduleRender?.();
            setTimeout(() => {
                node._berniniEditor?._promptEnhancer?.syncFromWidgets?.();
                node._berniniEditor?._promptEnhancer?.fetchTemplate?.(true);
            }, 120);
        }
    },
    async getCustomWidgets() {
        return {
            BDGROUP(node, inputName, inputData) {
                const w = makeGroupHeaderWidget(inputName, inputData);
                if (!node.widgets) node.widgets = [];
                node.widgets.push(w);
                return w;
            },
        };
    },
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!isDirectorNodeDef(nodeType, nodeData)) return;

        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onCreated?.apply(this, arguments);
            normalizeDirectorOutputs(this);
            applyDirectorWidgetLabels(this);
            this.size = [1000, 680];

            const container = document.createElement("div");
            container.className = "bd-host";
            container.style.minHeight = `${getDirectorUiHeight(null)}px`;
            container.style.setProperty("--comfy-widget-min-height", String(getDirectorUiHeight(null)));
            const self = this;
            const widget = this.addDOMWidget("bernini_director_ui", "director", container, {
                getValue: () => "",
                setValue: () => {},
                getMinHeight: () => getDirectorUiHeight(self._berniniEditor),
                hideOnZoom: false,
                onDraw() {
                    if (self._berniniEditor?.isPlaying) return;
                    ensureDirectorDomWidgetWidth(self);
                },
                afterResize: () => {
                    if (self._berniniEditor?.isPlaying || self._berniniEditor?._pauseSettling) return;
                    ensureDirectorDomWidgetWidth(self);
                    self._berniniEditor?.onNodeResize?.();
                },
            });
            bindDirectorDomWidgetSizing(self, widget, () => self._berniniEditor);
            widget.element = container;
            ensureDirectorDomWidgetWidth(self);
            self._directorDomWidget = widget;
            finalizeDirectorWidgetOrder(self);

            setTimeout(() => {
                finalizeDirectorWidgetOrder(self);
                initDirectorEditor(self);
            }, 0);
            return r;
        };

        const onResize = nodeType.prototype.onResize;
        nodeType.prototype.onResize = function (size) {
            ensureDirectorDomWidgetWidth(this);
            const out = onResize?.apply(this, arguments);
            if (!this._berniniEditor?.isPlaying && !this._berniniEditor?._pauseSettling) {
                this._berniniEditor?.onNodeResize?.(size);
            }
            return out;
        };

        const onSelected = nodeType.prototype.onSelected;
        nodeType.prototype.onSelected = function () {
            ensureDirectorDomWidgetWidth(this);
            const out = onSelected?.apply(this, arguments);
            this._berniniEditor?.scheduleRender?.();
            return out;
        };

        const onDeselected = nodeType.prototype.onDeselected;
        nodeType.prototype.onDeselected = function () {
            const out = onDeselected?.apply(this, arguments);
            if (this._berniniEditor?.isPlaying) this._berniniEditor._stopPlay();
            return out;
        };

        const onRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            this._berniniEditor?.destroy();
            return onRemoved?.apply(this, arguments);
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            normalizeDirectorOutputs(this);
            const out = onConfigure?.apply(this, arguments);
            setTimeout(() => {
                finalizeDirectorWidgetOrder(this);
                const ed = initDirectorEditor(this) || this._berniniEditor;
                if (!ed) return;
                const initTotal = Math.max(0, parseInt(ed.totalFramesWidget?.value || 81, 10));
                const initFps = coerceTimelineFps(ed.frameRateWidget?.value || 24);
                ed.timeline = parseTimeline(ed.timelineWidget?.value, initTotal, initFps);
                ed.syncFrameRateUI(ed.timeline.frameRate);
                ed._directorMode = ed.getDirectorMode();
                if (ed._directorMode === "video") {
                    ed.restoreVideoFromTimeline();
                } else if (ed._directorMode === "prompt_batch" || ed._directorMode === "image_batch") {
                    ensureImageBatchTimeline(ed);
                } else {
                    ed.ensureGenTimeline();
                }
                ed.applyTaskLayout(ed._directorMode);
                ed.populateTaskSelect(ed.globalTask, ed.taskTypeWidget?.value);
                ed.setEditMode(ed.timeline.editMode || "global");
                ed.selectedIndex = 0;
                ed.updateSelectionUI();
                ed.commit(true, { syncTimeline: false });
                ed._promptEnhancer?.syncFromWidgets?.();
                ed._promptEnhancer?.fetchTemplate?.(true);
            }, 80);
            return out;
        };
    },
});
