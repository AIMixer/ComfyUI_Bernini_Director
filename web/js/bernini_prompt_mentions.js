/** @-mention picker for Bernini director reference images (image0–image4). */

import { api } from "../../scripts/api.js";

const MENTION_STYLES = `
.bd-mention-menu{position:fixed;z-index:10050;min-width:210px;max-width:300px;max-height:240px;overflow:auto;background:#252525;border:1px solid #444;border-radius:8px;box-shadow:0 8px 24px rgba(0,0,0,.45);padding:4px 0}
.bd-mention-menu.hidden{display:none!important}
.bd-mention-title{padding:6px 10px 4px;font-size:10px;color:#888;user-select:none}
.bd-mention-item{display:flex;align-items:center;gap:8px;padding:6px 10px;cursor:pointer;font-size:11px;color:#ddd}
.bd-mention-item:hover,.bd-mention-item.active{background:#333;color:#fff}
.bd-mention-item img{width:36px;height:36px;object-fit:cover;border-radius:4px;flex-shrink:0;background:#111;border:1px solid #333}
.bd-mention-item .bd-mention-label{font-weight:600;color:#4fff8f}
.bd-mention-empty{padding:10px 12px;font-size:11px;color:#888;text-align:center;line-height:1.4}
`;

let stylesInjected = false;

function injectStyles() {
    if (stylesInjected) return;
    stylesInjected = true;
    const el = document.createElement("style");
    el.textContent = MENTION_STYLES;
    document.head.appendChild(el);
}

function inputViewUrl(filename, type = "input") {
    const subfolder = filename.includes("/") ? filename.slice(0, filename.lastIndexOf("/")) : "";
    const base = subfolder ? filename.slice(subfolder.length + 1) : filename;
    const params = new URLSearchParams({ filename: base, type });
    if (subfolder) params.set("subfolder", subfolder);
    return api.apiURL(`/view?${params.toString()}`);
}

function refThumbUrl(ref) {
    if (ref?.imageFile) return inputViewUrl(ref.imageFile, "input");
    if (ref?.imageB64) {
        return ref.imageB64.startsWith("data:") ? ref.imageB64 : `data:image/png;base64,${ref.imageB64}`;
    }
    return "";
}

function listAvailableRefs(refs) {
    return [...(refs || [])]
        .filter((r) => r?.imageFile || r?.imageB64)
        .sort((a, b) => Number(a.index ?? a.slot ?? 0) - Number(b.index ?? b.slot ?? 0))
        .map((r) => {
            const index = Number(r.index ?? r.slot ?? 0);
            return {
                index,
                label: `image${index}`,
                tag: `@image${index}`,
                thumb: refThumbUrl(r),
            };
        });
}

function positionMenu(menu, textarea) {
    const rect = textarea.getBoundingClientRect();
    menu.style.left = `${Math.max(8, rect.left)}px`;
    menu.style.top = `${Math.min(window.innerHeight - 16, rect.bottom + 4)}px`;
    menu.style.maxWidth = `${Math.max(210, rect.width)}px`;
}

/**
 * Wire @-mention dropdown on a prompt textarea.
 * Typing `@` lists uploaded reference images; pick one to insert `@imageN`.
 */
export function wirePromptImageMentions(editor, textarea, getRefs) {
    if (!textarea || textarea.dataset.mentionWired) return;
    textarea.dataset.mentionWired = "1";
    injectStyles();

    let menu = null;
    let mentionStart = -1;
    let activeIndex = 0;
    let filtered = [];

    const ensureMenu = () => {
        if (menu) return menu;
        menu = document.createElement("div");
        menu.className = "bd-mention-menu hidden";
        menu.setAttribute("role", "listbox");
        document.body.appendChild(menu);
        return menu;
    };

    const closeMenu = () => {
        mentionStart = -1;
        filtered = [];
        activeIndex = 0;
        if (menu) menu.classList.add("hidden");
    };

    const renderMenu = (query) => {
        const m = ensureMenu();
        const all = listAvailableRefs(getRefs());
        const q = (query || "").toLowerCase();
        filtered = all.filter((item) => !q || item.label.includes(q) || item.tag.includes(q));
        m.innerHTML = "";
        const title = document.createElement("div");
        title.className = "bd-mention-title";
        title.textContent = "选择参考图";
        m.appendChild(title);

        if (!filtered.length) {
            const empty = document.createElement("div");
            empty.className = "bd-mention-empty";
            empty.textContent = all.length ? "无匹配参考图" : "请先在右侧上传参考图";
            m.appendChild(empty);
        } else {
            filtered.forEach((item, i) => {
                const row = document.createElement("div");
                row.className = `bd-mention-item${i === activeIndex ? " active" : ""}`;
                row.dataset.index = String(i);
                if (item.thumb) {
                    const img = document.createElement("img");
                    img.src = item.thumb;
                    img.alt = item.label;
                    row.appendChild(img);
                }
                const label = document.createElement("span");
                label.innerHTML = `<span class="bd-mention-label">${item.tag}</span>`;
                row.appendChild(label);
                row.onmousedown = (e) => {
                    e.preventDefault();
                    insertMention(item.tag);
                };
                m.appendChild(row);
            });
        }
        positionMenu(m, textarea);
        m.classList.remove("hidden");
    };

    const insertMention = (tag) => {
        const text = textarea.value;
        const cursor = textarea.selectionStart;
        const before = text.slice(0, mentionStart);
        const after = text.slice(cursor);
        const next = `${before}${tag} ${after}`;
        textarea.value = next;
        const pos = before.length + tag.length + 1;
        textarea.setSelectionRange(pos, pos);
        closeMenu();
        textarea.dispatchEvent(new Event("input", { bubbles: true }));
        textarea.focus();
    };

    const openIfMention = () => {
        const cursor = textarea.selectionStart;
        const before = textarea.value.slice(0, cursor);
        const match = before.match(/@([a-zA-Z0-9]*)$/);
        if (!match) {
            closeMenu();
            return;
        }
        mentionStart = cursor - match[0].length;
        activeIndex = 0;
        renderMenu(match[1]);
    };

    textarea.addEventListener("input", openIfMention);
    textarea.addEventListener("click", openIfMention);
    textarea.addEventListener("keyup", (e) => {
        if (["ArrowLeft", "ArrowRight", "Home", "End"].includes(e.key)) openIfMention();
    });

    textarea.addEventListener("keydown", (e) => {
        if (menu?.classList.contains("hidden") || !filtered.length) return;
        if (e.key === "ArrowDown") {
            e.preventDefault();
            activeIndex = (activeIndex + 1) % filtered.length;
            renderMenu(textarea.value.slice(mentionStart + 1, textarea.selectionStart));
        } else if (e.key === "ArrowUp") {
            e.preventDefault();
            activeIndex = (activeIndex - 1 + filtered.length) % filtered.length;
            renderMenu(textarea.value.slice(mentionStart + 1, textarea.selectionStart));
        } else if (e.key === "Enter" || e.key === "Tab") {
            e.preventDefault();
            insertMention(filtered[activeIndex].tag);
        } else if (e.key === "Escape") {
            e.preventDefault();
            closeMenu();
        }
    });

    document.addEventListener("mousedown", (e) => {
        if (!menu || menu.classList.contains("hidden")) return;
        if (e.target === textarea || menu.contains(e.target)) return;
        closeMenu();
    });

    window.addEventListener("scroll", closeMenu, true);
    window.addEventListener("resize", closeMenu);
}

/** Attach @-mention to global + segment positive prompt fields. */
export function mountPromptImageMentions(editor) {
    if (!editor) return;
    wirePromptImageMentions(editor, editor.globalPrompt, () => editor.timeline?.global?.refs || []);
    wirePromptImageMentions(editor, editor.segPrompt, () => {
        const seg = editor.timeline?.segments?.[editor.selectedIndex];
        return seg?.refs || [];
    });
}
