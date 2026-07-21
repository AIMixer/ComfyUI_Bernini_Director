/** Shared helpers for Bernini Director generation tasks. */

export const IMAGE_BATCH_TASKS = new Set(["t2i", "i2i", "r2i"]);
export const VIDEO_BATCH_TASKS = new Set(["t2v", "i2v", "r2v"]);
export const PROMPT_BATCH_TASKS = new Set([...IMAGE_BATCH_TASKS, ...VIDEO_BATCH_TASKS]);

/** Shown when task_type is i2v — Bernini upstream has no dedicated i2v testcase/demo. */
export const I2V_EXPERIMENTAL_NOTICE =
    "⚠ 实验性功能：Bernini 官方未提供 i2v 专用示例与效果演示；当前将源图作为单帧视频作为上下文生成，效果可能与预期有差异。";

export function resolveTaskKey(taskTypeValue) {
    let value = String(taskTypeValue || "").split(",[object Object]", 1)[0].trim();
    if (value.includes(" · ")) value = value.split(" · ", 1)[0].trim();
    for (const sep of [" — ", " —— ", " - ", " – "]) {
        if (value.includes(sep)) return value.split(sep, 1)[0].trim();
    }
    return value || "rv2v";
}

export function isGenTaskType(taskTypeValue) {
    const key = resolveTaskKey(taskTypeValue);
    return PROMPT_BATCH_TASKS.has(key);
}

export function isVideoBatchTask(taskKey) {
    return VIDEO_BATCH_TASKS.has(taskKey);
}

export function isImageBatchTask(taskKey) {
    return IMAGE_BATCH_TASKS.has(taskKey);
}

export function isPromptBatchTask(taskKey) {
    return PROMPT_BATCH_TASKS.has(taskKey);
}

export function getDirectorMode(taskTypeValue) {
    const key = resolveTaskKey(taskTypeValue);
    if (PROMPT_BATCH_TASKS.has(key)) return "prompt_batch";
    return "video";
}

/** t2i/t2v=plain, i2i/i2v=source image, r2i/r2v=5 reference images */
export function imageBatchVariant(taskKey) {
    if (taskKey === "i2i" || taskKey === "i2v") return "source";
    if (taskKey === "r2i" || taskKey === "r2v") return "refs";
    return "plain";
}

/** t2i/r2i/t2v/r2v need fixed canvas; i2i/i2v may use long_edge. */
export function imageBatchRequiresFixedOutput(taskKey) {
    return taskKey === "t2i" || taskKey === "r2i" || taskKey === "t2v" || taskKey === "r2v";
}

/** Maximum frames per diffusion segment (model / VRAM practical limit). */
export const MAX_GEN_FRAMES = 512;

/** v2v / mv2v / ads2v — no reference image slots (image0–4); ads2v uses reference video. */
const NO_REF_IMAGE_TASKS = new Set(["v2v", "mv2v", "ads2v"]);

export function taskUsesReferenceImages(taskKey) {
    return !NO_REF_IMAGE_TASKS.has(taskKey);
}

export function taskUsesReferenceVideo(taskKey) {
    return taskKey === "ads2v";
}

export function defaultFrameCount(taskKey) {
    if (isImageBatchTask(taskKey)) return 1;
    if (isVideoBatchTask(taskKey)) return 81;
    return 81;
}

export function minFrameCount(taskKey) {
    if (isImageBatchTask(taskKey)) return 1;
    if (isVideoBatchTask(taskKey)) return 4;
    return 4;
}

export function sumFrameCounts(segments) {
    return (segments || []).reduce(
        (s, seg) => s + Math.max(0, parseInt(seg.frameCount ?? seg.length, 10) || 0),
        0,
    );
}

export function genLayoutHint(taskKey) {
    switch (taskKey) {
        case "t2i": return "文生图 · 多组提示词 · 固定宽高 · 全部导出至 images · 选择运行";
        case "i2i": return "图生图 · 每组上传源图 · 全部导出至 images · 选择运行";
        case "r2i": return "参考主体生图 · 每组最多 5 张参考图（img0–img4）· 全部导出至 images · 选择运行";
        case "t2v": return "文生视频 · 多组提示词 · 每组可设帧数 · 固定宽高 · 支持全部/分段导出 · 选择运行";
        case "r2v": return "参考主体生视频 · 每组最多 5 张参考图（img0–img4）· 每组可设帧数 · 支持全部/分段导出 · 选择运行";
        case "i2v": return "图生视频 · 实验性功能 · 单帧视频输入 · 每组可设帧数 · 支持全部/分段导出 · 选择运行";
        default: return "";
    }
}

export function newBatchSegment(overrides = {}) {
    const taskKey = resolveTaskKey(overrides.taskType || "");
    const fc = isVideoBatchTask(taskKey) ? defaultFrameCount(taskKey) : 1;
    return {
        id: Date.now().toString(36) + Math.random().toString(36).slice(2, 7),
        start: 0,
        length: fc,
        frameCount: fc,
        prompt: "",
        negativePrompt: "",
        taskType: "",
        refs: [],
        genImage: { imageFile: "" },
        previewB64: "",
        previewFrames: [],
        previewFps: 24,
        ...overrides,
    };
}
