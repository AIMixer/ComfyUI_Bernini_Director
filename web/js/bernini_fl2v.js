/** First/last-frame (fl2v) — fixed total timeline, LTX-style edge drag, start/end flags. */

import { api } from "../../scripts/api.js";
import { defaultFrameCount, MAX_GEN_FRAMES, minFrameCount, resolveTaskKey } from "./bernini_gen_timeline.js";

export const FL2V_STYLES = `
.bd-fl2v-detail{width:100%;box-sizing:border-box;display:flex;flex-direction:column;gap:8px;background:#1a1a1a;border:1px solid #333;border-radius:6px;padding:10px}
.bd-fl2v-detail.hidden{display:none!important}
.bd-fl2v-detail-head{display:flex;align-items:center;justify-content:space-between;gap:8px;flex-wrap:wrap}
.bd-fl2v-detail-head b{color:#ccc;font-size:12px}
.bd-fl2v-detail-meta{color:#888;font-size:11px}
.bd-fl2v-flags{display:flex;flex-wrap:wrap;gap:12px 16px;align-items:center}
.bd-fl2v-flag{display:inline-flex;align-items:center;gap:6px;color:#ddd;font-size:12px;cursor:pointer;user-select:none}
.bd-fl2v-flag input{width:14px;height:14px;margin:0;accent-color:#4fff8f;cursor:pointer}
.bd-fl2v-detail .bd-label{color:#888;font-size:10px;margin-top:2px}
.bd-fl2v-detail textarea{width:100%;min-height:64px;background:#141414;border:1px solid #333;border-radius:4px;color:#eee;padding:6px;resize:vertical;font-size:11px;box-sizing:border-box;font-family:inherit;line-height:1.35}
.bd-fl2v-detail textarea[data-r="fl2v-negative"]{min-height:44px}
.bd-fl2v-detail textarea:disabled{opacity:.45;cursor:not-allowed}
.bd-fl2v-hint{color:#aaa;font-size:11px;line-height:1.45;background:#181818;border:1px solid #333;border-radius:6px;padding:8px 10px}
.bd-fl2v-hint b{color:#4fff8f;font-weight:600}
.bd-fl2v-total-wrap{display:inline-flex;align-items:center;gap:6px}
.bd-fl2v-total-wrap.hidden{display:none!important}
`;

const DEFAULT_TOTAL = 240;
/** Same default as the node ``negative_prompt`` widget. */
export const DEFAULT_FL2V_NEGATIVE = "bad video";

function uid() {
    return Date.now().toString(36) + Math.random().toString(36).slice(2, 7);
}

function clamp(n, lo, hi) {
    return Math.max(lo, Math.min(hi, n));
}

export function fl2vViewUrl(imageFile) {
    if (!imageFile) return "";
    const norm = String(imageFile).replace(/\\/g, "/");
    const slash = norm.lastIndexOf("/");
    const filename = slash >= 0 ? norm.slice(slash + 1) : norm;
    const subfolder = slash >= 0 ? norm.slice(0, slash) : "";
    const params = new URLSearchParams({ filename, type: "input" });
    if (subfolder) params.set("subfolder", subfolder);
    return api.apiURL(`/view?${params.toString()}`);
}

async function uploadImage(file) {
    const body = new FormData();
    body.append("image", file);
    body.append("type", "input");
    body.append("overwrite", "true");
    const resp = await api.fetchApi("/upload/image", { method: "POST", body });
    if (!resp.ok) throw new Error((await resp.text()) || `Upload failed (${resp.status})`);
    return resp.json();
}

function imageDims(file) {
    return new Promise((resolve) => {
        const url = URL.createObjectURL(file);
        const img = new Image();
        img.onload = () => {
            resolve({ width: img.naturalWidth || 0, height: img.naturalHeight || 0 });
            URL.revokeObjectURL(url);
        };
        img.onerror = () => {
            resolve({ width: 0, height: 0 });
            URL.revokeObjectURL(url);
        };
        img.src = url;
    });
}

export function getFl2vTotalFrames(editor) {
    const t = parseInt(editor?.timeline?.totalFrames, 10);
    if (Number.isFinite(t) && t > 0) return clamp(t, minFrameCount("fl2v"), MAX_GEN_FRAMES);
    const w = parseInt(editor?.totalFramesWidget?.value, 10);
    if (Number.isFinite(w) && w > 0) return clamp(w, minFrameCount("fl2v"), MAX_GEN_FRAMES);
    return DEFAULT_TOTAL;
}

export function setFl2vTotalFrames(editor, value, { fitSegments = true } = {}) {
    const total = clamp(parseInt(value, 10) || DEFAULT_TOTAL, minFrameCount("fl2v"), MAX_GEN_FRAMES);
    editor.timeline.totalFrames = total;
    if (editor.totalFramesWidget) editor.totalFramesWidget.value = total;
    if (fitSegments) normalizeFl2vSegments(editor);
    return total;
}

export function newFl2vSegment(overrides = {}) {
    const minFc = minFrameCount("fl2v");
    const fc = clamp(
        parseInt(overrides.frameCount ?? overrides.length ?? defaultFrameCount("fl2v"), 10) || 81,
        minFc,
        MAX_GEN_FRAMES,
    );
    const imageFile = overrides.imageFile
        || overrides.genImage?.imageFile
        || "";
    const isStartFrame = overrides.isStartFrame !== undefined
        ? !!overrides.isStartFrame
        : (overrides.breakBefore !== undefined
            ? !!overrides.breakBefore || overrides.isEndFrame === false
            : true);
    // Legacy: breakBefore=false meant continuous end-of-previous.
    let isEndFrame;
    if (overrides.isEndFrame !== undefined) isEndFrame = !!overrides.isEndFrame;
    else if (overrides.breakBefore !== undefined) isEndFrame = !overrides.breakBefore;
    else isEndFrame = false;

    return {
        id: overrides.id || uid(),
        start: Math.max(0, parseInt(overrides.start, 10) || 0),
        length: fc,
        frameCount: fc,
        prompt: overrides.prompt || "",
        negativePrompt: overrides.negativePrompt || DEFAULT_FL2V_NEGATIVE,
        taskType: "",
        refs: [],
        isStartFrame,
        isEndFrame,
        genImage: {
            imageFile,
            width: overrides.width || overrides.genImage?.width || 0,
            height: overrides.height || overrides.genImage?.height || 0,
        },
        imageFile,
    };
}

/** Sync keyframes mirror; keep free placement within total (no forced pack). */
export function normalizeFl2vSegments(editor) {
    const total = getFl2vTotalFrames(editor);
    const minFc = minFrameCount("fl2v");
    let segs = [...(editor.timeline.segments || [])].map((s, i) => {
        const imageFile = s.genImage?.imageFile || s.imageFile || "";
        const len = clamp(parseInt(s.length ?? s.frameCount, 10) || minFc, minFc, total);
        const start = clamp(parseInt(s.start, 10) || 0, 0, Math.max(0, total - len));
        let isStartFrame = !!s.isStartFrame;
        if (s.isStartFrame === undefined) {
            // Keep legacy: only the final continuous end-frame was non-start.
            const n = (editor.timeline.segments || []).length;
            const endOnlyLast = i > 0 && i === n - 1 && !s.breakBefore && s.isEndFrame !== false;
            isStartFrame = !endOnlyLast;
        }
        return {
            ...s,
            start,
            length: len,
            frameCount: len,
            isStartFrame,
            isEndFrame: !!s.isEndFrame,
            imageFile,
            genImage: {
                imageFile,
                width: s.genImage?.width || s.width || 0,
                height: s.genImage?.height || s.height || 0,
            },
        };
    });

    segs.sort((a, b) => a.start - b.start || a.id.localeCompare(b.id));

    // Push overlaps forward (stable, keep lengths when possible).
    let cursor = 0;
    for (const s of segs) {
        if (s.start < cursor) s.start = cursor;
        if (s.start + s.length > total) {
            s.start = Math.max(0, total - s.length);
            if (s.start < cursor) {
                s.start = cursor;
                s.length = Math.max(minFc, total - s.start);
                s.frameCount = s.length;
            }
        }
        cursor = s.start + s.length;
    }
    // If still overflowing after push, shrink from the end.
    if (cursor > total && segs.length) {
        const last = segs[segs.length - 1];
        last.length = Math.max(minFc, total - last.start);
        last.frameCount = last.length;
    }

    editor.timeline.segments = segs;
    editor.timeline.totalFrames = total;
    editor.timeline.keyframes = segs.map((s) => ({
        id: s.id,
        imageFile: s.genImage?.imageFile || "",
        width: s.genImage?.width || 0,
        height: s.genImage?.height || 0,
        start: s.start,
        length: s.length,
        frameCount: s.length,
        prompt: s.prompt || "",
        negativePrompt: s.negativePrompt || DEFAULT_FL2V_NEGATIVE,
        isStartFrame: !!s.isStartFrame,
        isEndFrame: !!s.isEndFrame,
    }));
    // Keep segment field in sync when legacy clips had an empty negative.
    for (const s of segs) {
        if (!s.negativePrompt) s.negativePrompt = DEFAULT_FL2V_NEGATIVE;
    }
    if (editor.totalFramesWidget) editor.totalFramesWidget.value = total;
    if (editor.fl2vUi?.totalInput && editor.fl2vUi.totalInput !== document.activeElement) {
        editor.fl2vUi.totalInput.value = String(total);
    }
    return segs;
}

/** @deprecated use normalizeFl2vSegments */
export function packFl2vSegments(editor) {
    return normalizeFl2vSegments(editor);
}

export function ensureFl2vTimeline(editor) {
    const t = editor.timeline;
    t.timelineMode = "fl2v";
    t.editMode = "segment";
    t.video = t.video || {};
    t.video.videoFile = "";
    t.video.fileName = "";
    t.video.frameMap = [];
    t.videoClips = [];

    if (!t.totalFrames || t.totalFrames < minFrameCount("fl2v")) {
        t.totalFrames = Math.max(
            DEFAULT_TOTAL,
            parseInt(editor.totalFramesWidget?.value, 10) || 0,
        );
    }

    const segs = t.segments || [];
    const imgSegs = segs.filter((s) => s.genImage?.imageFile || s.imageFile);
    const migrateFlags = (s, i, n) => {
        const endOnlyLast = i > 0 && i === n - 1 && !s.breakBefore && s.isEndFrame !== false
            && s.isStartFrame === undefined;
        return {
            isStartFrame: s.isStartFrame !== undefined ? !!s.isStartFrame : !endOnlyLast,
            isEndFrame: s.isEndFrame !== undefined
                ? !!s.isEndFrame
                : (i > 0 && !s.breakBefore),
        };
    };
    if (imgSegs.length) {
        const n = imgSegs.length;
        t.segments = imgSegs.map((s, i) => newFl2vSegment({
            ...s,
            ...migrateFlags(s, i, n),
        }));
    } else if (Array.isArray(t.keyframes) && t.keyframes.length) {
        const n = t.keyframes.length;
        t.segments = t.keyframes.map((k, i) => newFl2vSegment({
            id: k.id,
            imageFile: k.imageFile,
            width: k.width,
            height: k.height,
            start: k.start,
            frameCount: k.frameCount ?? k.length ?? 81,
            prompt: k.prompt || "",
            negativePrompt: k.negativePrompt || "",
            ...migrateFlags(k, i, n),
        }));
    } else {
        t.segments = [];
    }
    normalizeFl2vSegments(editor);
    return t;
}

export function fl2vStartIndices(editor) {
    return (editor.timeline.segments || [])
        .map((s, i) => (s.isStartFrame ? i : -1))
        .filter((i) => i >= 0);
}

/** Effective sample frame count for a start clip (absorbs following pure-end span). */
export function fl2vSampleFrameCount(editor, segIndex) {
    const segs = [...(editor.timeline.segments || [])].sort((a, b) => a.start - b.start);
    const seg = (editor.timeline.segments || [])[segIndex];
    if (!seg?.isStartFrame) return 0;
    const orderedIdx = segs.findIndex((s) => s.id === seg.id);
    if (orderedIdx < 0) return seg.length || 0;
    const startSeg = segs[orderedIdx];
    let endSeg = null;
    for (let j = orderedIdx + 1; j < segs.length; j++) {
        if (segs[j].isEndFrame) {
            endSeg = segs[j];
            break;
        }
    }
    const base = Math.max(minFrameCount("fl2v"), parseInt(startSeg.length, 10) || 0);
    if (!endSeg || endSeg.isStartFrame) return base;
    const endT = (parseInt(endSeg.start, 10) || 0) + (parseInt(endSeg.length, 10) || 0);
    const startT = parseInt(startSeg.start, 10) || 0;
    return Math.max(base, endT - startT);
}

export function mountFl2vPanel(parent) {
    const wrap = document.createElement("div");
    wrap.className = "bd-fl2v-detail-wrap";
    wrap.innerHTML = `
        <div class="bd-fl2v-hint">
            <b>温馨提示</b>：在上方设<strong>总帧数</strong>，再在轨道上放置图片；拖左右缘调整范围（相接处：上半黄调整前段，下半蓝调整后段）。
            勾选<strong>首帧</strong>才会采样；勾选<strong>尾帧</strong>为固定终点（本身不采样）。
            选中后点工具栏<strong>替换图片</strong>或<strong>双击</strong>换图；<strong>拖动</strong>图片可交换位置。
        </div>
        <div class="bd-fl2v-detail hidden" data-r="fl2v-detail">
            <div class="bd-fl2v-detail-head">
                <b data-r="fl2v-detail-title">关键帧 #1</b>
                <span class="bd-fl2v-detail-meta" data-r="fl2v-detail-meta"></span>
            </div>
            <div class="bd-fl2v-flags">
                <label class="bd-fl2v-flag">
                    <input type="checkbox" data-r="fl2v-start">
                    首帧（单独采样）
                </label>
                <label class="bd-fl2v-flag">
                    <input type="checkbox" data-r="fl2v-end">
                    尾帧（作为尾帧参考，不单独采样）
                </label>
            </div>
            <span class="bd-label">本镜提示词（仅首帧）</span>
            <textarea data-r="fl2v-prompt" placeholder="描述这一镜的运动/变化（可选）"></textarea>
            <span class="bd-label">反向提示词（仅首帧）</span>
            <textarea data-r="fl2v-negative" placeholder="${DEFAULT_FL2V_NEGATIVE}"></textarea>
        </div>
        <input type="file" accept="image/*" multiple hidden data-r="fl2v-file">
    `;
    parent.appendChild(wrap);
    return {
        root: wrap,
        hint: wrap.querySelector(".bd-fl2v-hint"),
        detail: wrap.querySelector('[data-r="fl2v-detail"]'),
        title: wrap.querySelector('[data-r="fl2v-detail-title"]'),
        meta: wrap.querySelector('[data-r="fl2v-detail-meta"]'),
        startCb: wrap.querySelector('[data-r="fl2v-start"]'),
        endCb: wrap.querySelector('[data-r="fl2v-end"]'),
        prompt: wrap.querySelector('[data-r="fl2v-prompt"]'),
        negative: wrap.querySelector('[data-r="fl2v-negative"]'),
        totalInput: null, // wired from output bar in bernini_timeline.js
        fileInput: wrap.querySelector('[data-r="fl2v-file"]'),
    };
}

/** Strip hard-lock wraps that PE / older builds may embed; UI stores motion body only. */
export function stripFl2vPromptBody(text) {
    let out = String(text || "").trim();
    if (!out) return "";
    const wraps = [
        "完全保持首尾帧。",
        "完全保持首帧。",
        "视频开始完全按照image0的画面，不修改，视频结束完全保持image1的画面。",
        "视频开始完全按照image0的画面，不修改，视频结束完全保持image1。",
        "视频开始完全按照image0的构图，不修改，视频结束完全保持image1。",
        "视频开始完全按照image0的画面，不修改。",
        "视频开始完全按照image0的构图，不修改。",
        "视频结束完全保持image1的画面。",
        "视频结束完全保持image1。",
        "完全保持首尾帧：开头必须是image0，结尾必须是image1。",
        "完全保持首帧：开头必须是image0。",
        "再次强调：开头锁定image0，结尾锁定image1。",
        "再次强调：开头锁定image0。",
        "中间过程：",
    ];
    let changed = true;
    while (changed && out) {
        changed = false;
        for (const w of wraps) {
            if (out.startsWith(w)) {
                out = out.slice(w.length).trim();
                changed = true;
            }
            if (out.endsWith(w)) {
                out = out.slice(0, -w.length).trim();
                changed = true;
            }
        }
    }
    return out
        .replace(/image0的构图/g, "image0的画面")
        .replace(/image1的构图/g, "image1的画面")
        .trim();
}

/** Persist textarea edits to the segment they belong to (not necessarily selectedIndex). */
export function flushFl2vPromptDraft(editor) {
    const ui = editor?.fl2vUi;
    if (!ui?.prompt && !ui?.negative) return;
    const segs = editor.timeline?.segments || [];
    const idx = editor._fl2vPromptSegIndex;
    if (!Number.isFinite(idx) || idx < 0 || idx >= segs.length) return;
    const seg = segs[idx];
    if (!seg?.isStartFrame) return;
    if (ui.prompt) seg.prompt = ui.prompt.value || "";
    if (ui.negative) seg.negativePrompt = ui.negative.value || "";
}

export function updateFl2vDetailUI(editor) {
    const ui = editor.fl2vUi;
    if (!ui?.detail) return;
    if (!editor.isFl2vMode?.()) {
        ui.detail.classList.add("hidden");
        return;
    }
    if (ui.totalInput && ui.totalInput !== document.activeElement) {
        ui.totalInput.value = String(getFl2vTotalFrames(editor));
    }
    const segs = editor.timeline.segments || [];
    const idx = editor.selectedIndex;
    const seg = segs[idx];
    if (!seg) {
        flushFl2vPromptDraft(editor);
        editor._fl2vPromptSegIndex = null;
        ui.detail.classList.add("hidden");
        return;
    }
    ui.detail.classList.remove("hidden");
    const imageFile = seg.genImage?.imageFile || seg.imageFile || "";
    const roles = [];
    if (seg.isStartFrame) roles.push("首帧");
    if (seg.isEndFrame) roles.push("尾帧");
    if (ui.title) {
        ui.title.textContent = `#${idx + 1}${imageFile ? "" : "（未上传图）"}${roles.length ? ` · ${roles.join("+")}` : ""}`;
    }
    if (ui.meta) {
        const clipLo = seg.start;
        const clipHi = seg.start + seg.length - 1;
        if (seg.isStartFrame) {
            const sampleN = fl2vSampleFrameCount(editor, idx);
            const sampleHi = seg.start + sampleN - 1;
            if (sampleN > seg.length) {
                ui.meta.textContent = `片段 ${seg.length} 帧 · ${clipLo}–${clipHi} → 采样 ${sampleN} 帧 · ${seg.start}–${sampleHi}`;
            } else {
                ui.meta.textContent = `采样 ${sampleN} 帧 · ${clipLo}–${clipHi}`;
            }
        } else {
            ui.meta.textContent = `占位 ${seg.length} 帧 · ${clipLo}–${clipHi}（并入上一镜采样）`;
        }
    }
    if (ui.startCb) ui.startCb.checked = !!seg.isStartFrame;
    if (ui.endCb) ui.endCb.checked = !!seg.isEndFrame;
    const canPrompt = !!seg.isStartFrame;
    const prevIdx = editor._fl2vPromptSegIndex;
    const selectionChanged = prevIdx !== idx;
    // Switching clips while a textarea is focused used to skip the value
    // update, then blur/change wrote the old text into the newly selected clip.
    if (selectionChanged) {
        flushFl2vPromptDraft(editor);
        editor._fl2vPromptSegIndex = idx;
    } else {
        editor._fl2vPromptSegIndex = idx;
    }
    if (ui.prompt) {
        ui.prompt.disabled = !canPrompt;
        if (selectionChanged || ui.prompt !== document.activeElement) {
            ui.prompt.value = canPrompt ? (seg.prompt || "") : "";
        }
        ui.prompt.placeholder = canPrompt
            ? "描述这一镜的运动/变化（可选）"
            : "非首帧不采样，无需提示词";
    }
    if (ui.negative) {
        ui.negative.disabled = !canPrompt;
        if (selectionChanged || ui.negative !== document.activeElement) {
            ui.negative.value = canPrompt
                ? (seg.negativePrompt || DEFAULT_FL2V_NEGATIVE)
                : "";
        }
        ui.negative.placeholder = canPrompt
            ? DEFAULT_FL2V_NEGATIVE
            : "非首帧不采样，无需反向提示词";
    }
}

export function bindFl2vEvents(editor) {
    const ui = editor.fl2vUi;
    if (!ui) return;

    const applyTotal = () => {
        setFl2vTotalFrames(editor, ui.totalInput?.value);
        editor.selectedIndex = clamp(
            editor.selectedIndex,
            0,
            Math.max(0, (editor.timeline.segments?.length || 1) - 1),
        );
        editor.currentFrame = clamp(editor.currentFrame, 0, Math.max(0, getFl2vTotalFrames(editor) - 1));
        if (editor.seekBar) {
            editor.seekBar.max = Math.max(0, getFl2vTotalFrames(editor) - 1);
            editor.seekBar.value = editor.currentFrame;
        }
        editor.commit(false, { syncTimeline: true });
        updateFl2vDetailUI(editor);
        editor.updateVideoNameLabel?.();
        editor.scheduleRender();
        editor.updateDomWidgetHeight?.();
    };
    ui.totalInput?.addEventListener("change", applyTotal);
    ui.totalInput?.addEventListener("keydown", (e) => e.stopPropagation());

    const promptTargetSeg = () => {
        const segs = editor.timeline.segments || [];
        const idx = Number.isFinite(editor._fl2vPromptSegIndex)
            ? editor._fl2vPromptSegIndex
            : editor.selectedIndex;
        return segs[idx] || null;
    };
    const bindPromptField = (el, field) => {
        if (!el) return;
        el.addEventListener("change", () => {
            const seg = promptTargetSeg();
            if (!seg?.isStartFrame) return;
            seg[field] = el.value || "";
            editor.commit(false, { syncTimeline: true });
            editor.scheduleRender();
        });
        el.addEventListener("input", () => {
            const seg = promptTargetSeg();
            if (!seg?.isStartFrame) return;
            seg[field] = el.value || "";
            editor.scheduleRender();
        });
        el.addEventListener("focus", () => {
            if (!Number.isFinite(editor._fl2vPromptSegIndex)) {
                editor._fl2vPromptSegIndex = editor.selectedIndex;
            }
        });
    };
    bindPromptField(ui.prompt, "prompt");
    bindPromptField(ui.negative, "negativePrompt");

    ui.startCb?.addEventListener("change", () => {
        const seg = editor.timeline.segments?.[editor.selectedIndex];
        if (!seg) return;
        seg.isStartFrame = !!ui.startCb.checked;
        if (!seg.isStartFrame) {
            seg.prompt = seg.prompt || "";
            seg.negativePrompt = seg.negativePrompt || "";
        }
        normalizeFl2vSegments(editor);
        editor.commit(false, { syncTimeline: true });
        updateFl2vDetailUI(editor);
        editor.updateVideoNameLabel?.();
        editor.scheduleRender();
    });

    ui.endCb?.addEventListener("change", () => {
        const seg = editor.timeline.segments?.[editor.selectedIndex];
        if (!seg) return;
        seg.isEndFrame = !!ui.endCb.checked;
        normalizeFl2vSegments(editor);
        editor.commit(false, { syncTimeline: true });
        updateFl2vDetailUI(editor);
        editor.updateVideoNameLabel?.();
        editor.scheduleRender();
    });

    ui.fileInput?.addEventListener("change", async () => {
        const files = [...(ui.fileInput.files || [])];
        const mode = editor._fl2vUploadMode || "append";
        const replaceIndex = editor._fl2vReplaceIndex;
        editor._fl2vUploadMode = "append";
        editor._fl2vReplaceIndex = null;
        if (ui.fileInput) ui.fileInput.multiple = true;
        if (!files.length) return;
        ensureFl2vTimeline(editor);
        try {
            if (mode === "replace") {
                await replaceFl2vSegmentImage(editor, replaceIndex ?? editor.selectedIndex, files[0]);
            } else {
                let total = getFl2vTotalFrames(editor);
                const minFc = minFrameCount("fl2v");
                const defFc = defaultFrameCount("fl2v");
                for (const file of files) {
                    const up = await uploadImage(file);
                    const dims = await imageDims(file);
                    const name = up.name || up.filename;
                    const sub = (up.subfolder || "").replace(/\\/g, "/").replace(/\/$/, "");
                    const path = sub ? `${sub}/${name}` : name;
                    const segs = editor.timeline.segments || [];
                    const lastEnd = segs.reduce((m, s) => Math.max(m, (s.start || 0) + (s.length || 0)), 0);
                    let start = lastEnd;
                    let length = Math.min(defFc, Math.max(minFc, total - start));
                    if (length < minFc) {
                        // Grow total so the new clip fits.
                        total = setFl2vTotalFrames(editor, lastEnd + defFc, { fitSegments: false });
                        start = lastEnd;
                        length = defFc;
                    }
                    const isFirst = segs.length === 0;
                    editor.timeline.segments.push(newFl2vSegment({
                        imageFile: path,
                        width: dims.width,
                        height: dims.height,
                        start,
                        frameCount: length,
                        isStartFrame: isFirst,
                        isEndFrame: !isFirst,
                    }));
                }
                normalizeFl2vSegments(editor);
                editor.selectedIndex = Math.max(0, editor.timeline.segments.length - 1);
            }
            editor.commit(false, { syncTimeline: true });
            updateFl2vDetailUI(editor);
            editor.updateVideoNameLabel?.();
            editor.scheduleRender();
            editor.updateDomWidgetHeight?.();
        } catch (err) {
            console.error("[Bernini fl2v] upload failed", err);
            alert(`上传失败：${err?.message || err}`);
        } finally {
            ui.fileInput.value = "";
        }
    });
}

async function replaceFl2vSegmentImage(editor, index, file) {
    const segs = editor.timeline.segments || [];
    const idx = clamp(parseInt(index, 10) || 0, 0, Math.max(0, segs.length - 1));
    const seg = segs[idx];
    if (!seg || !file) throw new Error("请先选中要替换的图片片段");
    const prevKey = `fl2v:${seg.genImage?.imageFile || seg.imageFile || ""}`;
    const up = await uploadImage(file);
    const dims = await imageDims(file);
    const name = up.name || up.filename;
    const sub = (up.subfolder || "").replace(/\\/g, "/").replace(/\/$/, "");
    const path = sub ? `${sub}/${name}` : name;
    seg.imageFile = path;
    seg.genImage = {
        ...(seg.genImage || {}),
        imageFile: path,
        width: dims.width || seg.genImage?.width || 0,
        height: dims.height || seg.genImage?.height || 0,
    };
    if (prevKey && prevKey !== `fl2v:${path}`) {
        editor._thumbCache?.delete(prevKey);
        editor._thumbPending?.delete(prevKey);
    }
    editor._thumbCache?.delete(`fl2v:${path}`);
    editor.selectedIndex = idx;
    normalizeFl2vSegments(editor);
}

/**
 * CSS-like background-repeat-x: height fills the track, width keeps aspect,
 * tiles from the left. 2× → 2 copies; 2.5× → 2 full + 0.5 cropped. Never stretch.
 */
export function drawFl2vSegmentThumbnails(editor, ctx, seg, startX, pxWidth, y0, h) {
    ctx.save();
    ctx.beginPath();
    ctx.rect(startX, y0 + 1, pxWidth, h - 2);
    ctx.clip();
    ctx.fillStyle = "#0d0d0d";
    ctx.fillRect(startX, y0 + 1, pxWidth, h - 2);

    const imageFile = seg.genImage?.imageFile || seg.imageFile || "";
    if (!imageFile) {
        ctx.fillStyle = "#666";
        ctx.font = "12px sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText("点击「上传图片」", startX + pxWidth / 2, y0 + h / 2);
        ctx.restore();
        return;
    }

    const cacheKey = `fl2v:${imageFile}`;
    let img = editor._thumbCache.get(cacheKey);
    if (!img?.naturalWidth) {
        if (!editor._thumbPending.has(cacheKey)) {
            editor._thumbPending.add(cacheKey);
            const el = new Image();
            el.crossOrigin = "anonymous";
            el.onload = () => {
                editor._thumbCache.set(cacheKey, el);
                editor._thumbPending.delete(cacheKey);
                editor.scheduleRender();
            };
            el.onerror = () => editor._thumbPending.delete(cacheKey);
            el.src = fl2vViewUrl(imageFile);
        }
        ctx.restore();
        return;
    }

    // background-size: auto 100%; background-repeat: repeat-x (never stretch)
    const natW = img.naturalWidth;
    const natH = Math.max(1, img.naturalHeight);
    const metaW = +(seg.genImage?.width || seg.width || 0);
    const metaH = +(seg.genImage?.height || seg.height || 0);
    const aspect = (natW > 0 && natH > 0)
        ? (natW / natH)
        : (metaW > 0 && metaH > 0 ? metaW / metaH : 1);
    const trackH = Math.max(1, h - 2);
    const trackW = Math.max(0, pxWidth);
    const tileH = trackH;
    const tileW = tileH * aspect;
    const drawY = y0 + 1;
    const endX = startX + trackW;

    if (tileW > 0.5 && trackW > 0.5) {
        for (let x = startX; x < endX - 0.5; x += tileW) {
            const remain = endX - x;
            if (remain >= tileW - 0.5) {
                ctx.drawImage(img, 0, 0, natW, natH, x, drawY, tileW, tileH);
            } else {
                // Partial tile: crop source — never squash into remaining width.
                const srcW = Math.max(1, (remain / tileW) * natW);
                ctx.drawImage(img, 0, 0, srcW, natH, x, drawY, remain, tileH);
            }
        }
    }

    // Role badges
    let badgeX = startX + 4;
    const badgeY = y0 + 6;
    if (seg.isStartFrame) {
        ctx.fillStyle = "rgba(79,255,143,0.92)";
        ctx.fillRect(badgeX, badgeY, 38, 14);
        ctx.fillStyle = "#111";
        ctx.font = "bold 9px sans-serif";
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        ctx.fillText("START", badgeX + 4, badgeY + 7);
        badgeX += 42;
    }
    if (seg.isEndFrame) {
        ctx.fillStyle = "rgba(240,160,48,0.92)";
        ctx.fillRect(badgeX, badgeY, 30, 14);
        ctx.fillStyle = "#111";
        ctx.font = "bold 9px sans-serif";
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        ctx.fillText("END", badgeX + 5, badgeY + 7);
    }

    ctx.restore();
}

export function getFl2vUiHeight(editor) {
    const n = editor.timeline?.segments?.length || 0;
    const peBase = 120;
    // Detail panel includes positive + negative textareas when a clip is selected.
    return 520 + Math.min(80, n * 4) + peBase;
}

export function buildFl2vPayloadFields(editor) {
    ensureFl2vTimeline(editor);
    normalizeFl2vSegments(editor);
    const total = getFl2vTotalFrames(editor);
    const keyframes = (editor.timeline.keyframes || []).map((k) => ({
        id: k.id,
        imageFile: k.imageFile || "",
        width: k.width || 0,
        height: k.height || 0,
        start: parseInt(k.start, 10) || 0,
        length: clamp(parseInt(k.length ?? k.frameCount, 10) || 81, minFrameCount("fl2v"), MAX_GEN_FRAMES),
        frameCount: clamp(parseInt(k.frameCount ?? k.length, 10) || 81, minFrameCount("fl2v"), MAX_GEN_FRAMES),
        prompt: k.prompt || "",
        negativePrompt: k.negativePrompt || DEFAULT_FL2V_NEGATIVE,
        isStartFrame: !!k.isStartFrame,
        isEndFrame: !!k.isEndFrame,
    }));
    return {
        timelineMode: "fl2v",
        editMode: "segment",
        keyframes,
        segments: (editor.timeline.segments || []).map((s) => ({
            id: s.id,
            start: s.start,
            length: s.length,
            frameCount: s.length,
            prompt: s.prompt || "",
            negativePrompt: s.negativePrompt || DEFAULT_FL2V_NEGATIVE,
            isStartFrame: !!s.isStartFrame,
            isEndFrame: !!s.isEndFrame,
            genImage: {
                imageFile: s.genImage?.imageFile || s.imageFile || "",
                width: s.genImage?.width || 0,
                height: s.genImage?.height || 0,
            },
            taskType: "",
            refs: [],
        })),
        totalFrames: total,
    };
}

export function isFl2vTaskValue(taskTypeValue) {
    return resolveTaskKey(taskTypeValue) === "fl2v";
}

export function openFl2vUpload(editor) {
    editor._fl2vUploadMode = "append";
    editor._fl2vReplaceIndex = null;
    const input = editor.fl2vUi?.fileInput;
    if (!input) return;
    input.multiple = true;
    input.click();
}

/** Replace image on the selected (or given) fl2v segment. */
export function openFl2vReplace(editor, index) {
    const segs = editor?.timeline?.segments || [];
    const idx = index != null ? index : editor.selectedIndex;
    if (!Number.isFinite(idx) || idx < 0 || idx >= segs.length) {
        alert("请先在时间轴上选中要替换的图片");
        return;
    }
    editor.selectedIndex = idx;
    editor._fl2vUploadMode = "replace";
    editor._fl2vReplaceIndex = idx;
    const input = editor.fl2vUi?.fileInput;
    if (!input) return;
    input.multiple = false;
    input.click();
}

export function setFl2vToolbar(editor, enabled) {
    const disable = [
        editor.btnVideoAppend,
        editor.root?.querySelector('[data-a="split"]'),
        editor.root?.querySelector('[data-a="smart-split"]'),
        editor.root?.querySelector('[data-a="equal"]'),
        editor.root?.querySelector('[data-a="mode-global"]'),
        editor.root?.querySelector('[data-a="mode-segment"]'),
    ];
    for (const btn of disable) {
        if (!btn) continue;
        btn.disabled = enabled;
        btn.classList.toggle("bd-disabled", enabled);
        btn.classList.toggle("hidden", enabled);
    }
    if (editor.equalCountInput) {
        editor.equalCountInput.disabled = enabled;
        editor.equalCountInput.classList.toggle("hidden", enabled);
    }
    if (editor.btnVideo) {
        editor.btnVideo.disabled = false;
        editor.btnVideo.classList.remove("bd-disabled", "hidden");
        editor.btnVideo.textContent = enabled ? "上传图片" : "上传视频";
    }
    const del = editor.root?.querySelector('[data-a="del"]');
    if (del) {
        del.disabled = false;
        del.classList.remove("bd-disabled", "hidden");
        del.textContent = enabled ? "删除图片" : "删除片段";
    }
    updateFl2vReplaceBtn(editor);
}

/** Toolbar「替换图片」：仅 fl2v 且选中某个图片片段时显示。 */
export function updateFl2vReplaceBtn(editor) {
    const btn = editor?.btnFl2vReplace || editor?.root?.querySelector?.('[data-a="fl2v-replace"]');
    if (!btn) return;
    const segs = editor?.timeline?.segments || [];
    const idx = editor?.selectedIndex;
    const show = !!editor?.isFl2vMode?.()
        && Number.isFinite(idx)
        && idx >= 0
        && idx < segs.length
        && !!(segs[idx]?.genImage?.imageFile || segs[idx]?.imageFile);
    btn.classList.toggle("hidden", !show);
    btn.disabled = !show;
}
