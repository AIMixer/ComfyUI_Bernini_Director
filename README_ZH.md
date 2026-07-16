# ComfyUI Bernini Director

基于 **ComfyUI 官方 Bernini-R** 的多段视频导演台插件。仓库地址：[AIMixer/ComfyUI_Bernini_Director](https://github.com/AIMixer/ComfyUI_Bernini_Director)

**English** → [README.md](README.md)

![ComfyBerniniDirector 工作流截图](docs/screenshot.png)

## 功能介绍

**ComfyBerniniDirector** 是面向长视频、多段生成的 Bernini 导演台节点，把分段计划、条件编码、双阶段采样和导出整合在一个节点里。

### 核心能力

| 功能 | 说明 |
|------|------|
| **多段时间轴** | 节点内上传视频，支持切分、均分、追加；可视化时间轴预览每段范围 |
| **多任务模式** | 支持 v2v、rv2v、r2v、t2v 等 Bernini 官方任务（由输入自动推断） |
| **参考图引导** | 最多 5 张参考图（image0–image4），支持 `@imageN` 提示词引用 |
| **双阶段采样** | HIGH / LOW 双 UNET，独立 CFG、seed、步数与 split |
| **段间引导** | 默认关闭=官方 Studio 单段逻辑；勾选后才走跨段注入分支 |
| **LLM 提示词增强** | 内置 Ollama / 智谱等接口，按 Bernini 官方模板扩写提示词 |
| **音频导出** | 源视频含音轨时可从 `audio` 口输出，直连 VHS 合成 |
| **运行报告** | `report` 口输出分段计划、连贯设置、每段任务摘要 |

### 输入 / 输出

**输入：** `model_high` → `model_low` → `vae` → `clip`

**输出：** `images` → `audio` → `fps` → `frame_count` → `source_images` → `report`

> `model_shift` 请在工作流中用外部 **采样算法 (SD3)** 节点接到 `model_high` / `model_low`，不要与导演台内部重复设置。

## 依赖

请将 **ComfyUI** 升级到支持官方 Bernini 的版本及以上（[PR #14216](https://github.com/Comfy-Org/ComfyUI/pull/14216)）。

## 安装

### 方法一：手动安装（标准方式）

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/AIMixer/ComfyUI_Bernini_Director.git
```

可选依赖（非必须）：

```bash
pip install -r ComfyUI_Bernini_Director/requirements.txt
```

重启 ComfyUI。

### 方法二：ComfyUI Manager

1. 打开 **ComfyUI Manager**
2. 选择 **Install via Git URL**
3. 填入 `https://github.com/AIMixer/ComfyUI_Bernini_Director.git` 并安装
4. 重启 ComfyUI

## 模型与工作流下载

完整资源包（**Bernini 模型权重** + **示例 JSON 工作流**）见：

**[Comfyit 搅拌站 · 文章 489：视频编辑 Bernini 模型和工作流](https://comfyit.cn/article/489)**

下载后将 `models/` 合并到 `ComfyUI/models/`，JSON 工作流拖入 ComfyUI 即可。

本仓库也自带示例：`example_workflows/`

| 工作流 | task_type | 说明 |
|--------|-----------|------|
| `bernini_director_core_rv2v.json` | rv2v | 源视频 + 参考图 |

### 量化模型（GGUF / FP8）

**下载地址：** [comfyit.cn/article/489](https://comfyit.cn/article/489)（含 GGUF、scaled FP8、VAE、T5 及示例工作流）。

本插件走 **ComfyUI 官方链路**，推荐用 **UNETLoader** 加载 **FP8 safetensors**：

- `Wan22_Bernini_HIGH_fp8_e4m3fn_scaled.safetensors` → `models/diffusion_models/`
- `Wan22_Bernini_LOW_fp8_e4m3fn_scaled.safetensors` → `models/diffusion_models/`
- `wan_2.1_vae.safetensors` → `models/vae/`
- `umt5_xxl_fp8_e4m3fn_scaled.safetensors` → `models/text_encoders/`

## 快速开始

1. 确认 ComfyUI 版本已支持官方 Bernini（[PR #14216](https://github.com/Comfy-Org/ComfyUI/pull/14216)）
2. 从 [文章 489](https://comfyit.cn/article/489) 或本仓库 `example_workflows/` 加载示例
3. 连接 VAE / UNET×2 / CLIP，在导演台节点 UI 内上传源视频 / 参考图，编辑提示词后 Queue

## 配套生态 · [Comfyit 搅拌站](https://comfyit.cn/)

[Comfyit](https://comfyit.cn/) 提供环境、模型、工作流与教程配套：

| 栏目 | 链接 |
|------|------|
| 模型 / 工作流包 | [comfyit.cn/article/489](https://comfyit.cn/article/489) |
| 产品中心 | [comfyit.cn/products](https://comfyit.cn/products) |
| 插件广场 | [comfyit.cn/plugins](https://comfyit.cn/plugins) |
| 模型广场 | [comfyit.cn/resources/models](https://comfyit.cn/resources/models) |
| 工作流广场 | [comfyit.cn/workflows](https://comfyit.cn/workflows) |

## 作者与交流

| | |
|---|---|
| **维护者** | [AI搅拌手 / AIMixer](https://github.com/AIMixer) |
| **本仓库** | [github.com/AIMixer/ComfyUI_Bernini_Director](https://github.com/AIMixer/ComfyUI_Bernini_Director) |
| **作者 QQ** | **3697688140** |
| **B 站** | [space.bilibili.com/1997403556](https://space.bilibili.com/1997403556) |
| **QQ 交流群** | **551482703** · **425064221** · **559826331** |
| **Comfyit 搅拌站** | [comfyit.cn](https://comfyit.cn/) |

## 致谢

- [Comfy-Org / ComfyUI](https://github.com/Comfy-Org/ComfyUI) — 官方 Bernini 支持
- [bytedance/Bernini](https://github.com/bytedance/Bernini) — Bernini-R 模型与文档

## 许可证

Apache-2.0
