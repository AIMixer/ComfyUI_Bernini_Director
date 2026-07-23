/** Multi prompt-group UI for t2i / i2i / r2i / t2v / i2v / r2v (prompt batch mode). */

import { api } from "../../scripts/api.js";
import {
    defaultFrameCount,
    I2V_EXPERIMENTAL_NOTICE,
    imageBatchRequiresFixedOutput,
    imageBatchVariant,
    isVideoBatchTask,
    MAX_GEN_FRAMES,
    minFrameCount,
    newBatchSegment,
    resolveTaskKey,
} from "./bernini_gen_timeline.js";

const _players = new WeakMap();

function clamp(n, lo, hi) {
    return Math.max(lo, Math.min(hi, n));
}

function formatPreviewFps(value) {
    const fps = Math.round(Number(value) * 100) / 100;
    if (Number.isInteger(fps)) return String(fps);
    return fps.toFixed(2).replace(/0+$/, "").replace(/\.$/, "");
}

function stopPlayer(el) {
    const st = _players.get(el);
    if (!st) return;
    st.playing = false;
    if (st.timer) {
        clearInterval(st.timer);
        st.timer = null;
    }
}

function stopAllPlayers(root) {
    root?.querySelectorAll(".bd-batch-vpreview")?.forEach((wrap) => stopPlayer(wrap));
}

export const IMAGE_BATCH_STYLES = `
.bd-btn.bd-disabled,.bd-btn:disabled{opacity:.38;cursor:not-allowed;pointer-events:none}
.bd-mode button.bd-disabled,.bd-mode button:disabled{opacity:.38;cursor:not-allowed;pointer-events:none}
.bd-batch{width:100%;box-sizing:border-box;display:flex;flex-direction:column;gap:8px}
.bd-batch-i2v-notice{display:none;color:#ffb74d;background:#3a2a12;border:1px solid #a67c00;border-radius:6px;padding:8px 10px;font-size:11px;line-height:1.5}
.bd-batch-i2v-notice.visible{display:block}
.bd-batch-toolbar{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.bd-batch-run-select.active{background:#1a3a2a;color:#4fff8f;border-color:#4fff8f}
.bd-batch-run-all{display:inline-flex;align-items:center;gap:4px;font-size:11px;color:#aaa;cursor:pointer;user-select:none}
.bd-batch-run-all.hidden{display:none!important}
.bd-batch-run-all input{width:14px;height:14px;margin:0;cursor:pointer;accent-color:#4fff8f}
.bd-batch-list{display:flex;flex-direction:column;gap:8px;width:100%;max-height:560px;overflow-y:auto;padding-right:2px}
.bd-batch-card{background:#1a1a1a;border:1px solid #333;border-radius:6px;padding:8px;display:grid;grid-template-columns:auto minmax(0,1fr) minmax(120px,30%);gap:8px;align-items:stretch}
.bd-batch-card.running{border-color:#4fff8f;box-shadow:0 0 0 1px rgba(79,255,143,.25)}
.bd-batch-card.done{border-color:#3a5080}
.bd-batch-card.run-skipped{opacity:.42}
.bd-batch-head{grid-column:1/-1;display:flex;align-items:center;justify-content:space-between;gap:8px;flex-wrap:wrap}
.bd-batch-head b{color:#ccc;font-size:11px}
.bd-batch-head-meta{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.bd-batch-fc{display:flex;align-items:center;gap:4px;color:#888;font-size:10px}
.bd-batch-fc input{width:52px;background:#181818;border:1px solid #444;border-radius:4px;color:#eee;padding:3px 5px;font-size:11px}
.bd-batch-del{background:transparent;border:1px solid #553;color:#f88;border-radius:4px;padding:3px 8px;font-size:10px;cursor:pointer}
.bd-batch-del:hover{background:#3a1515}
.bd-batch-media{display:flex;flex-direction:column;gap:4px;min-width:88px;max-width:120px}
.bd-batch-src{width:88px;height:88px;border:1px dashed #555;border-radius:4px;background:#111;display:flex;align-items:center;justify-content:center;cursor:pointer;overflow:hidden;color:#666;font-size:9px;text-align:center;padding:4px;box-sizing:border-box}
.bd-batch-src.has-img{border-style:solid;border-color:#444}
.bd-batch-src img{width:100%;height:100%;object-fit:contain;background:#000}
.bd-batch-refs{display:grid;grid-template-columns:repeat(3,1fr);gap:3px;width:96px}
.bd-batch-ref{position:relative;aspect-ratio:1;border:1px dashed #555;border-radius:3px;background:#111;display:flex;align-items:center;justify-content:center;cursor:pointer;overflow:hidden;font-size:8px;color:#666}
.bd-batch-ref.has-img{border-style:solid}
.bd-batch-ref img{width:100%;height:100%;object-fit:cover}
.bd-batch-ref .x{position:absolute;top:0;right:2px;color:#f88;font-size:10px;display:none;line-height:1}
.bd-batch-ref:hover .x{display:block}
.bd-batch-prompts{display:flex;flex-direction:column;gap:4px;min-width:0}
.bd-batch-prompts .bd-label{color:#888;font-size:10px}
.bd-batch-prompts textarea{width:100%;min-height:44px;background:#181818;border:1px solid #333;border-radius:4px;color:#eee;padding:6px;resize:vertical;font-size:11px;box-sizing:border-box;font-family:inherit;line-height:1.35}
.bd-batch-preview{background:#0d0d0d;border:1px solid #333;border-radius:4px;min-height:100px;display:flex;flex-direction:column;align-items:center;justify-content:center;overflow:hidden;color:#555;font-size:10px;text-align:center;padding:4px;box-sizing:border-box}
.bd-batch-preview img{max-width:100%;max-height:160px;object-fit:contain;display:block}
.bd-batch-vpreview{width:100%;display:flex;flex-direction:column;align-items:stretch;gap:4px}
.bd-batch-vpreview canvas{width:100%;max-height:160px;background:#000;border-radius:3px;display:block}
.bd-batch-vpreview-ctrl{display:flex;align-items:center;justify-content:center;gap:6px}
.bd-batch-vpreview-ctrl button{font-size:10px;padding:2px 8px}
.bd-batch-vpreview-meta{color:#666;font-size:9px;text-align:center}
@media(max-width:720px){
.bd-batch-card{grid-template-columns:1fr}
.bd-batch-preview{min-height:80px}
}
`;

async function uploadImage(file) {
    const body = new FormData();
    body.append("image", file);
    body.append("type", "input");
    body.append("overwrite", "true");
    const resp = await api.fetchApi("/upload/image", { method: "POST", body });
    if (!resp.ok) throw new Error(await resp.text() || `Upload failed (${resp.status})`);
    return resp.json();
}

function relPath(upload) {
    const name = upload.name || upload.filename;
    const sub = (upload.subfolder || "").replace(/\\/g, "/").replace(/\/$/, "");
    return sub ? `${sub}/${name}` : name;
}

function viewUrl(imageFile) {
    const norm = String(imageFile || "").replace(/\\/g, "/");
    const slash = norm.lastIndexOf("/");
    const filename = slash >= 0 ? norm.slice(slash + 1) : norm;
    const subfolder = slash >= 0 ? norm.slice(0, slash) : "";
    const params = new URLSearchParams({ filename, type: "input" });
    if (subfolder) params.set("subfolder", subfolder);
    return api.apiURL(`/view?${params.toString()}`);
}

export function mountImageBatchPanel(root) {
    const panel = document.createElement("div");
    panel.className = "bd-batch hidden";
    panel.dataset.r = "batch-panel";
    panel.innerHTML = `
        <div class="bd-batch-toolbar">
            <button type="button" class="bd-btn bd-btn-primary" data-a="batch-add">+ 添加提示词组</button>
            <button type="button" class="bd-btn bd-batch-run-select hidden" data-a="batch-run-select" title="开启后可勾选要运行的提示词组">选择运行</button>
            <label class="bd-batch-run-all hidden" data-r="batch-run-all-wrap" title="勾选=全选，取消=全部不选">
                <input type="checkbox" data-r="batch-run-all-cb">
                <span>全选</span>
            </label>
            <span class="bd-meta" data-r="batch-hint">每组生成 1 张图片</span>
        </div>
        <div class="bd-batch-i2v-notice" data-r="batch-i2v-notice"></div>
        <div class="bd-batch-list" data-r="batch-list"></div>`;
    root.appendChild(panel);
    return {
        panel,
        list: panel.querySelector('[data-r="batch-list"]'),
        hint: panel.querySelector('[data-r="batch-hint"]'),
        i2vNotice: panel.querySelector('[data-r="batch-i2v-notice"]'),
        addBtn: panel.querySelector('[data-a="batch-add"]'),
        runSelectBtn: panel.querySelector('[data-a="batch-run-select"]'),
        runSelectAllWrap: panel.querySelector('[data-r="batch-run-all-wrap"]'),
        runSelectAllCb: panel.querySelector('[data-r="batch-run-all-cb"]'),
    };
}

export function wireBatchRunSelectControls(editor, batchUi) {
    editor.batchRunSelectBtn = batchUi.runSelectBtn;
    editor.batchRunSelectAllWrap = batchUi.runSelectAllWrap;
    editor.batchRunSelectAllCb = batchUi.runSelectAllCb;
    batchUi.runSelectBtn?.addEventListener("click", (e) => {
        e.stopPropagation();
        editor.toggleRunSelectMode?.();
    });
    batchUi.runSelectAllCb?.addEventListener("change", (e) => {
        e.stopPropagation();
        if (!editor.isRunSelectEnabled?.()) return;
        editor.setRunSelectionAll?.(batchUi.runSelectAllCb.checked);
    });
}

function cloneRefs(refs) {
    if (!Array.isArray(refs) || !refs.length) return [];
    try {
        return JSON.parse(JSON.stringify(refs));
    } catch {
        return refs.map((r) => ({ ...r }));
    }
}

/** Copy global.refs into batch segments that have no refs (r2i / r2v). */
export function migrateGlobalRefsIntoBatchSegments(editor, taskKey) {
    const key = resolveTaskKey(taskKey || editor.getTaskKey?.() || "");
    if (key !== "r2i" && key !== "r2v") return false;
    const globalRefs = editor.timeline?.global?.refs;
    if (!Array.isArray(globalRefs) || !globalRefs.length) return false;
    let moved = false;
    for (const seg of editor.timeline.segments || []) {
        if ((seg.refs || []).length) continue;
        seg.refs = cloneRefs(globalRefs);
        moved = true;
    }
    return moved;
}

export function ensureImageBatchTimeline(editor) {
    editor.timeline.editMode = "segment";
    editor.timeline.output = editor.timeline.output || {};
    const taskKey = resolveTaskKey(editor.getTaskKey?.() || editor.taskTypeWidget?.value);
    if (imageBatchRequiresFixedOutput(taskKey)) {
        editor.timeline.output.mode = "fixed";
    } else if (taskKey === "i2i" || taskKey === "i2v") {
        const mode = String(editor.timeline.output.mode || "long_edge").toLowerCase();
        editor.timeline.output.mode = mode === "fixed" ? "fixed" : "long_edge";
    }
    if (!isVideoBatchTask(taskKey)) {
        editor.timeline.output.exportMode = "all";
    }
    const defFc = defaultFrameCount(taskKey);
    if (taskKey === "i2v") {
        editor.timeline.video = {
            fileName: "",
            videoFile: "",
            subfolder: "",
            type: "input",
            frames: [],
            frameMap: [],
        };
        editor.timeline.videoClips = [];
    }
    if (!editor.timeline.segments?.length) {
        editor.timeline.segments = [newBatchSegment({ frameCount: defFc, length: defFc })];
    }
    // r2i/r2v need per-group refs. If the user came from rv2v (global refs) or left
    // refs only on global, copy them into empty batch groups so generation actually
    // receives reference_image_* — otherwise it silently behaves like t2v/t2i.
    migrateGlobalRefsIntoBatchSegments(editor, taskKey);
    for (const seg of editor.timeline.segments) {
        if (isVideoBatchTask(taskKey)) {
            // Prefer frame count remembered before a t2i/r2i detour (which forces 1f).
            const preferred = seg._videoFrameCount ?? seg.frameCount ?? seg.length;
            const fc = clamp(parseInt(preferred, 10) || defFc, minFrameCount(taskKey), MAX_GEN_FRAMES);
            seg.frameCount = fc;
            seg.length = fc;
            seg._videoFrameCount = fc;
        } else {
            // Keep prior video-batch length so r2v → t2i → r2v can restore it.
            const prevFc = parseInt(seg.frameCount ?? seg.length, 10) || 0;
            if (prevFc > 1) seg._videoFrameCount = prevFc;
            else if (seg._videoFrameCount == null && defFc > 1) {
                /* keep existing _videoFrameCount if any */
            }
            seg.frameCount = 1;
            seg.length = 1;
        }
        seg.negativePrompt = seg.negativePrompt ?? "";
        seg.genImage = seg.genImage || { imageFile: seg.imageFile || "" };
        // Do NOT clear refs for i2v — backend ignores them, but wiping here breaks
        // r2v → i2v → r2v (user loses uploaded reference images).
        seg.refs = seg.refs || [];
        seg.previewB64 = seg.previewB64 || "";
        seg.previewFrames = seg.previewFrames || [];
        seg.previewFps = seg.previewFps || parseFloat(editor.frameRateWidget?.value || 24);
        if (!seg.id) seg.id = newBatchSegment().id;
    }
    normalizeImageBatchSegments(editor);
}

export function normalizeImageBatchSegments(editor) {
    const taskKey = resolveTaskKey(editor.getTaskKey?.() || editor.taskTypeWidget?.value);
    const isVideo = isVideoBatchTask(taskKey);
    const defFc = defaultFrameCount(taskKey);
    const minFc = minFrameCount(taskKey);
    let start = 0;
    const fixed = [];
    for (const seg of editor.timeline.segments) {
        const fc = isVideo
            ? clamp(parseInt(seg.frameCount ?? seg.length, 10) || defFc, minFc, MAX_GEN_FRAMES)
            : 1;
        fixed.push({
            ...seg,
            start,
            length: fc,
            frameCount: fc,
            negativePrompt: seg.negativePrompt ?? "",
            genImage: seg.genImage || { imageFile: "" },
            refs: seg.refs || [],
            _videoFrameCount: seg._videoFrameCount,
            previewB64: seg.previewB64 || "",
            previewFrames: seg.previewFrames || [],
            previewFps: seg.previewFps || parseFloat(editor.frameRateWidget?.value || 24),
        });
        start += fc;
    }
    if (!fixed.length) fixed.push(newBatchSegment({ frameCount: defFc, length: defFc }));
    editor.timeline.segments = fixed;
    editor.timeline.totalFrames = start || fixed[0].frameCount;
}

export function addImageBatchGroup(editor) {
    const taskKey = resolveTaskKey(editor.getTaskKey?.() || editor.taskTypeWidget?.value);
    const defFc = defaultFrameCount(taskKey);
    editor.timeline.segments.push(newBatchSegment({
        frameCount: defFc,
        length: defFc,
        negativePrompt: editor.negativePromptWidget?.value || "bad video",
    }));
    normalizeImageBatchSegments(editor);
    editor.renderImageBatchGroups();
    editor.commit();
}

export function deleteImageBatchGroup(editor, index) {
    if (editor.timeline.segments.length <= 1) return;
    editor.timeline.segments.splice(index, 1);
    normalizeImageBatchSegments(editor);
    editor.renderImageBatchGroups();
    editor.commit();
}

function pickFile(accept, onFile) {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = accept;
    input.onchange = () => {
        const file = input.files?.[0];
        if (file) onFile(file);
    };
    input.click();
}

async function uploadSegSource(editor, index) {
    pickFile("image/*", async (file) => {
        try {
            const uploaded = await uploadImage(file);
            const seg = editor.timeline.segments[index];
            if (!seg) return;
            const imageFile = relPath(uploaded);
            const dims = await readImageDimensions(file);
            seg.genImage = { imageFile, width: dims.width, height: dims.height };
            seg.imageFile = imageFile;
            editor.renderImageBatchGroups();
            editor.updateOutputPreview?.();
            editor.commit();
        } catch (err) {
            console.error("[BerniniDirector] batch source upload failed:", err);
        }
    });
}

function readImageDimensions(file) {
    return new Promise((resolve, reject) => {
        const url = URL.createObjectURL(file);
        const img = new Image();
        img.onload = () => {
            URL.revokeObjectURL(url);
            resolve({ width: img.naturalWidth, height: img.naturalHeight });
        };
        img.onerror = () => {
            URL.revokeObjectURL(url);
            reject(new Error("Failed to read image dimensions"));
        };
        img.src = url;
    });
}

async function assignSegRefFromFile(editor, index, slot, file) {
    if (!file?.type?.startsWith("image/")) return;
    try {
        const uploaded = await uploadImage(file);
        const seg = editor.timeline.segments[index];
        if (!seg) return;
        seg.refs = (seg.refs || []).filter((r) => Number(r.index ?? r.slot) !== slot);
        seg.refs.push({ index: slot, imageFile: relPath(uploaded), imageB64: "" });
        editor.renderImageBatchGroups();
        editor.commit();
    } catch (err) {
        console.error("[BerniniDirector] batch ref upload failed:", err);
    }
}

async function uploadSegRef(editor, index, slot) {
    pickFile("image/*", (file) => assignSegRefFromFile(editor, index, slot, file));
}

function moveBatchRefSlot(editor, segIndex, fromSlot, toSlot) {
    if (fromSlot === toSlot) return;
    const seg = editor.timeline.segments[segIndex];
    if (!seg) return;
    const refs = [...(seg.refs || [])];
    const fromRef = refs.find((r) => Number(r.index ?? r.slot) === fromSlot);
    if (!fromRef) return;
    const toRef = refs.find((r) => Number(r.index ?? r.slot) === toSlot);
    seg.refs = refs.filter((r) => {
        const idx = Number(r.index ?? r.slot);
        return idx !== fromSlot && idx !== toSlot;
    });
    seg.refs.push({ ...fromRef, index: toSlot, slot: undefined });
    if (toRef) {
        seg.refs.push({ ...toRef, index: fromSlot, slot: undefined });
    }
    editor.renderImageBatchGroups();
    editor.commit();
}

function bindBatchRefDrop(slot, editor, index, slotIndex) {
    const hasImg = slot.classList.contains("has-img");
    slot.draggable = hasImg;
    slot.addEventListener("dragstart", (e) => {
        if (!hasImg) {
            e.preventDefault();
            return;
        }
        editor._batchRefDragMoved = false;
        const payload = JSON.stringify({ segIndex: index, from: slotIndex });
        e.dataTransfer.setData("application/x-bernini-ref-slot", payload);
        e.dataTransfer.setData("text/plain", payload);
        e.dataTransfer.effectAllowed = "move";
    });
    slot.addEventListener("dragend", () => {
        setTimeout(() => { editor._batchRefDragMoved = false; }, 0);
    });
    slot.addEventListener("dragover", (e) => {
        e.preventDefault();
        e.stopPropagation();
        const types = [...(e.dataTransfer?.types || [])];
        e.dataTransfer.dropEffect = types.includes("application/x-bernini-ref-slot")
            ? "move"
            : "copy";
    });
    slot.addEventListener("drop", (e) => {
        e.preventDefault();
        e.stopPropagation();
        const raw = e.dataTransfer.getData("application/x-bernini-ref-slot")
            || e.dataTransfer.getData("text/plain");
        if (raw) {
            try {
                const data = JSON.parse(raw);
                if (Number(data.segIndex) !== index) return;
                editor._batchRefDragMoved = true;
                moveBatchRefSlot(editor, index, Number(data.from), slotIndex);
                return;
            } catch (_) { /* fall through */ }
        }
        const f = e.dataTransfer.files?.[0];
        if (f) assignSegRefFromFile(editor, index, slotIndex, f);
    });
}

function removeSegRef(editor, index, slot) {
    const seg = editor.timeline.segments[index];
    if (!seg) return;
    seg.refs = (seg.refs || []).filter((r) => Number(r.index ?? r.slot) !== slot);
    editor.renderImageBatchGroups();
    editor.commit();
}

function renderSourceSlot(el, imageFile) {
    el.classList.toggle("has-img", !!imageFile);
    if (imageFile) {
        el.innerHTML = `<img src="${viewUrl(imageFile)}" alt="">`;
    } else {
        el.textContent = "上传源图";
    }
}

function renderRefSlot(el, ref, slot, index, editor) {
    el.classList.toggle("has-img", !!ref?.imageFile);
    el.innerHTML = "";
    el.title = `img${slot} — 点击上传；拖到其他格可移动`;
    if (ref?.imageFile) {
        const img = document.createElement("img");
        img.src = viewUrl(ref.imageFile);
        img.draggable = false;
        el.appendChild(img);
        const x = document.createElement("span");
        x.className = "x";
        x.textContent = "×";
        x.onclick = (e) => { e.stopPropagation(); removeSegRef(editor, index, slot); };
        el.appendChild(x);
    } else {
        el.textContent = `img${slot}`;
    }
}

function frameSrc(b64) {
    if (!b64) return "";
    return b64.startsWith("data:") ? b64 : `data:image/jpeg;base64,${b64}`;
}

function loadFrameImages(frames) {
    return Promise.all(frames.map((b64) => new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = reject;
        img.src = frameSrc(b64);
    })));
}

function drawFrame(canvas, img) {
    const ctx = canvas.getContext("2d");
    if (!ctx || !img) return;
    const cw = canvas.clientWidth || 160;
    const ch = canvas.clientHeight || 90;
    if (canvas.width !== cw) canvas.width = cw;
    if (canvas.height !== ch) canvas.height = ch;
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, cw, ch);
    const scale = Math.min(cw / img.naturalWidth, ch / img.naturalHeight);
    const dw = img.naturalWidth * scale;
    const dh = img.naturalHeight * scale;
    ctx.drawImage(img, (cw - dw) / 2, (ch - dh) / 2, dw, dh);
}

function mountVideoPreview(el, seg, running, fps) {
    stopPlayer(el);
    el.innerHTML = "";
    if (running) {
        el.textContent = "生成中…";
        return;
    }
    const frames = (seg.previewFrames?.length ? seg.previewFrames : null)
        || (seg.previewB64 ? [seg.previewB64] : null);
    if (!frames?.length) {
        el.textContent = "运行后在此预览视频";
        return;
    }
    const wrap = document.createElement("div");
    wrap.className = "bd-batch-vpreview";
    const canvas = document.createElement("canvas");
    canvas.height = 90;
    const ctrl = document.createElement("div");
    ctrl.className = "bd-batch-vpreview-ctrl";
    const playBtn = document.createElement("button");
    playBtn.type = "button";
    playBtn.className = "bd-btn";
    playBtn.textContent = "▶ 播放";
    const meta = document.createElement("div");
    meta.className = "bd-batch-vpreview-meta";
    meta.textContent = `${frames.length}帧 · ${formatPreviewFps(fps)}fps(预览)`;
    ctrl.appendChild(playBtn);
    wrap.appendChild(canvas);
    wrap.appendChild(ctrl);
    wrap.appendChild(meta);
    el.appendChild(wrap);

    const state = { playing: false, timer: null, idx: 0, images: null };
    _players.set(wrap, state);

    loadFrameImages(frames).then((images) => {
        state.images = images;
        drawFrame(canvas, images[0]);
    }).catch(() => {
        meta.textContent = "预览加载失败";
    });

    playBtn.onclick = (e) => {
        e.stopPropagation();
        if (!state.images?.length) return;
        if (state.playing) {
            state.playing = false;
            if (state.timer) clearInterval(state.timer);
            state.timer = null;
            playBtn.textContent = "▶ 播放";
            return;
        }
        state.playing = true;
        playBtn.textContent = "⏸ 暂停";
        const interval = Math.max(20, 1000 / Math.max(1, fps));
        state.timer = setInterval(() => {
            if (!state.images?.length) return;
            state.idx = (state.idx + 1) % state.images.length;
            drawFrame(canvas, state.images[state.idx]);
        }, interval);
    };
}

function renderImagePreview(el, seg, running) {
    stopPlayer(el);
    el.innerHTML = "";
    if (running) {
        el.textContent = "生成中…";
        return;
    }
    if (seg.previewB64) {
        const img = document.createElement("img");
        img.src = frameSrc(seg.previewB64);
        img.alt = "preview";
        el.appendChild(img);
        return;
    }
    el.textContent = "运行后在此预览";
}

function renderPreview(el, seg, running, isVideo, fps) {
    if (isVideo) mountVideoPreview(el, seg, running, fps);
    else renderImagePreview(el, seg, running);
}

export function renderImageBatchGroups(editor) {
    const list = editor.batchList;
    if (!list) return;
    stopAllPlayers(list);
    const key = resolveTaskKey(editor.getTaskKey?.() || editor.taskTypeWidget?.value);
    const variant = imageBatchVariant(key);
    const isVideo = isVideoBatchTask(key);
    const runningIdx = editor._runHighlightSeg;
    const fps = parseFloat(editor.frameRateWidget?.value || editor.timeline?.frameRate || 24);

    if (editor.batchHint) {
        const hints = {
            t2i: "文生图 · 开启「选择运行」可只跑勾选的组",
            i2i: "图生图 · 每组需上传源图 · 开启「选择运行」可只跑勾选的组",
            r2i: "参考主体生图 · 请在每组卡片上传参考图（img0–img4）",
            t2v: "文生视频 · 每组可设帧数 · 开启「选择运行」可只跑勾选的组",
            r2v: "参考主体生视频 · 请在每组卡片上传参考图（img0–img4），否则会像纯文生视频",
            i2v: "图生视频 · 实验性功能 · 开启「选择运行」可只跑勾选的组",
        };
        editor.batchHint.textContent = hints[key] || (isVideo ? "每组生成一段视频" : "每组生成 1 张图片");
    }
    if (editor.batchI2vNotice) {
        const needsRefs = key === "r2i" || key === "r2v";
        const hasAnyRefs = (editor.timeline.segments || []).some((s) => (s.refs || []).length > 0);
        if (key === "i2v") {
            editor.batchI2vNotice.textContent = I2V_EXPERIMENTAL_NOTICE;
            editor.batchI2vNotice.classList.add("visible");
        } else if (needsRefs && !hasAnyRefs) {
            editor.batchI2vNotice.textContent = key === "r2v"
                ? "当前没有参考图：请在提示词组卡片中上传 img0–img4。未上传时生成会退化成文生视频（t2v），无法参考主体。"
                : "当前没有参考图：请在提示词组卡片中上传 img0–img4。未上传时生成会退化成文生图（t2i）。";
            editor.batchI2vNotice.classList.add("visible");
        } else {
            editor.batchI2vNotice.classList.remove("visible");
            editor.batchI2vNotice.textContent = "";
        }
    }

    list.innerHTML = "";
    editor.timeline.segments.forEach((seg, index) => {
        const card = document.createElement("div");
        card.className = "bd-batch-card";
        if (index === runningIdx) card.classList.add("running");
        if (editor.isRunSelectEnabled?.() && editor.supportsRunSelect?.() && !editor.isSegmentRunEnabled(index)) {
            card.classList.add("run-skipped");
        }
        const hasPreview = isVideo
            ? (seg.previewFrames?.length > 0 || seg.previewB64)
            : !!seg.previewB64;
        if (hasPreview && index !== runningIdx) card.classList.add("done");

        const head = document.createElement("div");
        head.className = "bd-batch-head";
        if (editor.isRunSelectEnabled?.() && editor.supportsRunSelect?.()) {
            const runCb = document.createElement("input");
            runCb.type = "checkbox";
            runCb.className = "bd-batch-run-check";
            runCb.checked = editor.isSegmentRunEnabled(index);
            runCb.title = "勾选后参与本次运行";
            runCb.onclick = (e) => {
                e.stopPropagation();
                editor.toggleSegmentRun(index);
            };
            head.appendChild(runCb);
        }
        const title = document.createElement("b");
        title.textContent = `提示词组 ${index + 1}`;
        head.appendChild(title);
        const meta = document.createElement("div");
        meta.className = "bd-batch-head-meta";
        if (isVideo) {
            const fcRow = document.createElement("label");
            fcRow.className = "bd-batch-fc";
            fcRow.innerHTML = `帧数 <input type="number" min="${minFrameCount(key)}" max="${MAX_GEN_FRAMES}" step="1" value="${seg.frameCount ?? seg.length ?? 81}">`;
            const fcInput = fcRow.querySelector("input");
            fcInput.onchange = () => {
                const v = clamp(parseInt(fcInput.value, 10) || defaultFrameCount(key), minFrameCount(key), MAX_GEN_FRAMES);
                fcInput.value = String(v);
                seg.frameCount = v;
                seg.length = v;
                normalizeImageBatchSegments(editor);
                editor.scheduleTimelineSync();
            };
            meta.appendChild(fcRow);
        }
        const del = document.createElement("button");
        del.type = "button";
        del.className = "bd-batch-del";
        del.textContent = "删除";
        del.disabled = editor.timeline.segments.length <= 1;
        del.onclick = (e) => { e.stopPropagation(); deleteImageBatchGroup(editor, index); };
        meta.appendChild(del);
        head.appendChild(meta);
        card.appendChild(head);

        const media = document.createElement("div");
        media.className = "bd-batch-media";
        if (variant === "source") {
            const src = document.createElement("div");
            src.className = "bd-batch-src";
            renderSourceSlot(src, seg.genImage?.imageFile);
            src.onclick = () => uploadSegSource(editor, index);
            media.appendChild(src);
        } else if (variant === "refs") {
            const refs = document.createElement("div");
            refs.className = "bd-batch-refs";
            for (let i = 0; i < 5; i++) {
                const ref = (seg.refs || []).find((r) => Number(r.index ?? r.slot) === i);
                const slot = document.createElement("div");
                slot.className = "bd-batch-ref";
                renderRefSlot(slot, ref, i, index, editor);
                slot.onclick = () => {
                    if (editor._batchRefDragMoved) {
                        editor._batchRefDragMoved = false;
                        return;
                    }
                    uploadSegRef(editor, index, i);
                };
                bindBatchRefDrop(slot, editor, index, i);
                refs.appendChild(slot);
            }
            media.appendChild(refs);
        }
        card.appendChild(media);

        const prompts = document.createElement("div");
        prompts.className = "bd-batch-prompts";
        prompts.innerHTML = `
            <span class="bd-label">正向提示词</span>
            <textarea data-f="prompt" placeholder="描述要生成的内容">${seg.prompt || ""}</textarea>
            <span class="bd-label">反向提示词</span>
            <textarea data-f="negative" placeholder="不希望出现的内容">${seg.negativePrompt ?? ""}</textarea>`;
        prompts.querySelector('[data-f="prompt"]').oninput = (e) => {
            seg.prompt = e.target.value;
            editor.scheduleTimelineSync();
        };
        prompts.querySelector('[data-f="negative"]').oninput = (e) => {
            seg.negativePrompt = e.target.value;
            editor.scheduleTimelineSync();
        };
        card.appendChild(prompts);

        const preview = document.createElement("div");
        preview.className = "bd-batch-preview";
        renderPreview(preview, seg, index === runningIdx, isVideo, seg.previewFps || fps);
        card.appendChild(preview);

        list.appendChild(card);
    });
}

export function setImageBatchPreview(editor, segmentIndex, imageB64, extra = {}) {
    const seg = editor.timeline.segments[segmentIndex];
    if (!seg) return;
    seg.previewB64 = imageB64 || "";
    if (Array.isArray(extra.frames) && extra.frames.length) {
        seg.previewFrames = extra.frames;
        seg.previewFps = extra.fps || seg.previewFps || 24;
    } else if (imageB64) {
        seg.previewFrames = [imageB64];
    }
    editor.renderImageBatchGroups();
}

export function bindImageBatchEvents(editor) {
    editor.batchAddBtn?.addEventListener("click", (e) => {
        e.stopPropagation();
        addImageBatchGroup(editor);
    });
}

export function getImageBatchUiHeight(editor) {
    const n = Math.max(1, editor?.timeline?.segments?.length || 1);
    const key = resolveTaskKey(editor?.getTaskKey?.() || editor?.taskTypeWidget?.value);
    const rowH = isVideoBatchTask(key) ? 155 : 130;
    return 200 + Math.min(n, 6) * rowH + 60;
}

export function setToolbarDisabledForBatch(editor, disabled) {
    const btns = [
        editor.btnVideo,
        editor.btnVideoAppend,
        editor.root?.querySelector('[data-a="split"]'),
        editor.root?.querySelector('[data-a="smart-split"]'),
        editor.root?.querySelector('[data-a="equal"]'),
        editor.root?.querySelector('[data-a="del"]'),
        editor.root?.querySelector('[data-a="mode-global"]'),
        editor.root?.querySelector('[data-a="mode-segment"]'),
    ];
    for (const btn of btns) {
        if (!btn) continue;
        btn.disabled = disabled;
        btn.classList.toggle("bd-disabled", disabled);
    }
    if (editor.equalCountInput) {
        editor.equalCountInput.disabled = disabled;
        editor.equalCountInput.classList.toggle("bd-disabled", disabled);
    }
}
