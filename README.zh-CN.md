# ComfyUI Llama Workbench

[English](README.md) | 中文

用于本地文本生成、多模态提示词、交互聊天和可复用提示词 Skill 的 ComfyUI 自定义节点包。
支持连接已有的 OpenAI 兼容服务器、启动本地 `llama-server`，或使用可选的
`llama-cpp-python` 嵌入式后端。

主要功能：

- 启动和停止一个由本包明确拥有的外部 `llama-server` 进程，支持指定二进制路径和
  shell 安全的自定义参数，具体参数由所选可执行文件支持。
- 连接另一个本地 OpenAI 兼容的 `llama-server`，不会假定拥有该进程的生命周期。
- 通过任一后端生成文本或图像反推提示词结果。
- 通过同一个后端运行 Qwen-Image-2.1 Prompt Enhancer，使用官方任务采样 profile，并
  输出经过严格校验的结构化字段、原生 64 通道 Qwen-Image-2.1 latent，以及有序的
  PE-I2I 图片引用。
- 可选地通过 `llama-cpp-python` 加载 Qwen/Gemma 风格的本地模型，包括常见的多模态
  聊天处理器、KV q8_0 缓存选项，以及在已安装绑定支持时使用 Qwen MoE CPU 选项。
- 通过带鉴权的 API 将文本请求与图片工作流共同排队，支持持久请求 ID、结构化结果和显存调度。
- 使用原生 ComfyUI 消息字段保存图结构中的聊天历史。
- 从本包自己的 `skills/` 目录加载纯数据 Skill。Skill 可以请求已声明的参考资料、
  提供阶段和选项，但不是工具，不能执行 shell、网络或 ComfyUI 操作。

## 安装

将本文件夹直接放到 `ComfyUI/custom_nodes/` 下：

```text
ComfyUI/custom_nodes/ComfyUI-Llama-Workbench/
```

使用启动 ComfyUI 的同一个 Python 安装基础依赖：

```powershell
python -m pip install -r ComfyUI-Llama-Workbench/requirements.txt
```

外部服务器后端需要可用的 `llama-server` 二进制文件。可以在节点中设置二进制路径，
设置 `LWB_LLAMA_SERVER_BINARY`，或者将其加入 `PATH`。

嵌入式后端需要安装匹配 GPU 后端的 llama-cpp-python 构建：

```powershell
python -m pip install "llama-cpp-python>=0.3.37"
```

重启 ComfyUI 后，节点会出现在 **Llama Workbench** 分类下。

**Llama Workbench Start Server** 的 `wait_seconds` 是启动就绪等待上限，不是模型生成上限。首次加载大型模型，尤其是带
mmproj 的 Qwen/Gemma VLM 时，建议使用 600–1800 秒。如果等待超时，Workbench 会终止
它所拥有的 llama-server，避免 GPU 显存、内存和端口阻塞后续工作流。调整等待时间后再
重试即可。队列 API 使用独立的超时策略，未知启动结果会保留 GPU 租约，详见
[队列 API 文档](docs/queue-api.md)。

对于 **Llama Workbench Start Server**，`timeout_seconds` 是服务器就绪后每次 Chat 或
Prompt 请求的独立等待上限，默认 120 秒，与 **Llama Workbench Connection** 相同。长
推理或不限 token 的响应建议使用 600 秒或更高值。

## 节点目录

| 节点 | 用途 |
|---|---|
| Llama Workbench Start Server | 使用指定的 llama-server 兼容可执行文件启动一个模型。服务器启动前，`release_comfy_models` 会释放 ComfyUI 管理的 GPU 模型；`cleanup_previous_server` 仅保留输入兼容，不再按可执行文件或端口清理外部进程。 |
| Llama Workbench Connection | 连接已有的 HTTP 服务器，不接管其生命周期。 |
| Llama Workbench Stop Owned Server | 只停止由本包启动的进程。 |
| Llama Workbench Server Status | 显示进程身份、租约状态和有限长度的启动日志。 |
| Llama Workbench H3 Auto Resolution Selector | 将输入图像匹配到最接近的支持比例，并根据目标百万像素计算宽高。 |
| Llama Workbench Embedded VL Model | 可选的 `llama-cpp-python` 直接加载器，适用于 Qwen/Gemma 风格模型。 |
| Llama Workbench Release Embedded Model | 显式关闭嵌入式模型。 |
| Llama Workbench Prompt / Image2Prompt | 通过统一后端 socket 进行文本提示或图像反推提示词。图像 socket 会随着连接从 `image` 变为 `image1`、`image2` 等，最多 10 个；还提供 `seed`、`max_images`、`max_image_edge`、`auto_unload` 以及默认关闭的 `thinking` 控制。 |
| Llama Workbench Qwen Image 2.1 Prompt Enhancer | 通过 Workbench backend 原生执行结构化提示词改写。使用官方 T2I/edit 采样 profile，校验 `rewritten_prompt`、`wh_ratio`、`ratio_follow` 及编辑图片引用；节点提供默认开启的 `thinking` 开关。格式错误或生成被截断时，会关闭 thinking 并最多重试一次。任务专用 System Prompt 可粘贴、从文件加载或从本地模型旁自动发现。节点提供 `debug` 复选框，可打印限长原始 answer 和校验错误。 |
| Llama Workbench Qwen Image 2.1 PE Canvas | 解析 `wh_ratio`、`ratio_follow=<imageN>`、可选手动比例覆盖和 `follow_input_size`，输出宽、高、`ratio_source` 以及原生 `[1,64,H/16,W/16]` Qwen-Image-2.1 `LATENT`。 |
| Llama Workbench Qwen Image 2.1 PE Resolution | 为已有工作流保留的仅尺寸兼容节点。新的 Qwen-Image-2.1 工作流应使用 PE Canvas，确保 KSampler 收到正确的 64 通道 latent。 |
| Llama Workbench Image Info | 不复制像素地读取 `IMAGE` 批次，输出宽、高、最长/最短边、总像素、百万像素、宽高比、方向、批次数、通道数及完整张量元数据。 |
| Llama Workbench Image Info Display | 将完整图片信息格式化为详细、紧凑或 JSON 文本，支持中英文，并在可滚动画布面板中直接显示。 |
| Llama Workbench Pad Image to Multiple | 使用原生颜色选择器、可选 Alpha 和九宫格位置，把图片宽高补到指定倍数，并输出可精确还原的 padding 元数据。 |
| Llama Workbench Restore Image from Padding | 使用 padding 元数据，将处理后的补边图片裁剪回原始内容区域和尺寸。 |
| Llama Workbench Chat | 交互式本地聊天：输入文本后点击节点上的 **发送** 按钮，只会将此 Chat 节点及其上游依赖加入队列，不需要点击 Queue Prompt。节点有实用的初始尺寸，可以自由调整大小，长历史记录会在可滚动的画布区域中显示。`clear_context_before_run` 默认开启，每次排队的工作流都会从新上下文开始；关闭后可继续多轮对话。`use_cache` 默认开启，会独立于 `seed` 重用未变化的完整请求；关闭后可强制新的模型请求。缓存开启时会跳过 `release_comfy_cache_after_run`，以便继续复用响应。`release_owned_server_after_run` 默认开启，Chat 响应后会立即停止自有 llama-server，再让下游图片或视频节点分配显存。节点包含 **清空上下文** / **清空输入** 操作和文本 token 上下文计量器。Start Server 会自动提供计量器所需的 `context_size`；外部 Connection 需要设置 `context_size` 才能显示百分比。还支持图持久化历史、直接的 `max_tokens` / `seed` / `thinking` / `auto_unload` 控制，以及带 `max_image_edge` 的动态图像 socket（最多 10 张）。 |
| Llama Workbench Chat Output Display | 仅画布终端查看器，分别预览 Chat 节点的 `thinking` 和 `assistant_message` 输出。 |
| Llama Workbench Chat Settings | 系统提示词、采样、上下文历史和图像尺寸控制。 |
| Llama Workbench Skill Loader | 加载一个包内 Skill、自动选择 Skill，或进行普通聊天。 |
| Llama Workbench Seed | 输出一个整数 seed，并提供一键固定随机、每次随机、递增、递减和固定控制。 |

## 可导入工作流

[`examples/`](examples/README.md) 中包含七个可直接导入的工作流 JSON：自有服务器聊天、
连接已有服务器进行图像反推提示词、Skill 聊天，以及嵌入式 Qwen/Gemma VLM 图像反推
提示词，以及完整的 Qwen Image 2.1 PE-T2I、双图 PE-I2I GGUF → 原生图像生成流程和
10 图 PE-I2I 传输 smoke。运行前请将其中的通用模型和二进制占位路径替换为 ComfyUI
主机可访问的路径。

Prompt / Image2Prompt 节点的 `max_tokens` 默认值为 `-1`，llama-server 会将其解释为
不限生成长度。生成仍会在 EOS 或模型上下文处理结束时停止；如果需要限制延迟或输出
大小，请使用正数上限。

Prompt / Image2Prompt 最多支持 10 个图像参考。连接第一个图像到 `image` 后，它会重命名
为 `image1`，并出现新的 `image2` socket。每增加一个连接，就会暴露下一个 socket，未使用
的尾部 socket 会被移除。每个 socket 也接受 `IMAGE` batch。`max_images` 会限制所有已连接
socket 及其 batch 的图像总数。

`max_image_edge=0` 是默认值，表示本包不会在发送到 llama-server 前缩小图像。设置正数
边长限制会降低分辨率，通常也会减少 VLM 的视觉 token 使用量。具体视觉模型或 projector
仍可能自行调整大小或进行 token 化。

`auto_unload` 默认关闭。启用后，Prompt 或 Chat 会在生成后停止 **Llama Workbench Start
Server** 启动的精确 `llama-server` 进程，或释放 Workbench 的嵌入式 `llama-cpp-python`
模型。**Llama Workbench Connection** 是外部连接，不会被自动停止或卸载。Start Server
和 Embedded Model 会在每次排队运行时重新检查运行时状态，因此自动卸载的自有模型会在
下一次工作流中重新加载。

**Llama Workbench Chat** 也支持同样的动态图像输入。连接其 `image` socket 后会创建
`image2`；每增加一个连接，就会暴露下一个 socket，最多 10 张图像。`max_images` 是所有
已连接 socket 和 `IMAGE` batch 的合计上限。

连接图像的具体解释由选中的 Skill 和用户请求决定。

## Qwen-Image-2.1 Prompt Enhancer

使用兼容的 `llama-server` 和本地 GGUF 配置 **Llama Workbench Start Server**，再将
backend 连接到 **Llama Workbench Qwen Image 2.1 Prompt Enhancer**。第一阶段重点测试
[`pottokao/Qwen-Image-2.1-PE-T2I-Heretic-GGUF`](https://huggingface.co/pottokao/Qwen-Image-2.1-PE-T2I-Heretic-GGUF)。
选择 `t2i`、不连接图片，并提供与模型匹配的 Qwen PE System Prompt。节点会根据
`task` 自动选择对应提示词：

- `t2i_system_prompt` / `edit_system_prompt`：按任务手动粘贴官方提示词。
- `t2i_system_prompt_path` / `edit_system_prompt_path`：按任务指定提示词文件或模型目录。
- `system_prompt` / `system_prompt_path`：兼容旧工作流的当前任务通用覆盖输入。
- `auto_load_system_prompt`：默认开启；没有手动覆盖时，会从 Start Server/Embedded 模型
  文件旁边或模型目录寻找 `system_prompt_t2i.txt`、`system_prompt_edit.txt`，并回退到
  `system_prompt.txt`。

按任务专用输入优先于旧的通用输入。切换 `task` 会同步切换提示词文件名、采样 profile
和输出校验规则，但不会切换 backend 已加载的模型；`t2i` 必须连接 PE-T2I，`edit` 必须
连接 PE-I2I 及匹配的 mmproj。

仓库有意不包含 Qwen 官方 System Prompt 和模型权重。它们受上游许可证约束，不属于
本项目 MIT 许可的资产。

节点内部固定 PE 请求契约，不需要用户在通用 Prompt 节点里手工抄写采样参数。`t2i`
profile 会发送 `temperature=1.0`、`top_p=0.95`、`top_k=20`、`min_p=0`、
`presence_penalty=1.5`、`max_tokens=16256`；节点 `thinking=on` 时发送
`enable_thinking=true`，改为 `off` 时关闭推理阶段以降低延迟和 token 用量，但可能降低改写质量。
解析器只接受单个
JSON 对象（允许一个完整的外层 JSON Markdown 代码围栏），不接受对象外文本、修复后的 JSON、字段别名或未知字段；
`rewritten_prompt`、`wh_ratio` 和规范化为空字符串的 `ratio_follow` 会作为独立 socket
输出，同时提供 `result_json`。第一次格式错误或服务端截断生成时会触发一次纠正重试；
重试会关闭 thinking，并只要求最终 JSON。第二次仍失败就停止工作流，不会把不可靠文本
静默传给下游。

如需临时查看 Prompt Enhancer 的原始输出，可勾选节点上的 `debug`，或在启动 ComfyUI
前设置环境变量 `LWB_PROMPT_REWRITE_DEBUG=1`。ComfyUI 后台会打印每次尝试的限长
answer（包括返回的 `wh_ratio`）、finish reason 和校验错误。两者默认关闭；修改环境变量后
需要重启 ComfyUI。

将 `wh_ratio` 和 `ratio_follow` 同时连接到 **Llama Workbench Qwen Image 2.1 PE
Canvas**。该节点会输出取整后的 `width`、`height`、诊断用 `ratio_source`，以及原生
Qwen-Image-2.1 `[1,64,H/16,W/16]` latent。`aspect_ratio_override` 可强制使用指定比例，
但不会改写提示词。在 Edit 工作流中，`ratio_follow=<imageN>` 会选择对应顺序的参考图；
`follow_input_size=true` 保留其取整后的输入尺寸，`false` 则仅保留其宽高比，并按指定
百万像素预算计算画布。PE 也接受 `9:19.5` 这类有限小数比例，并在计算尺寸前规范化为
最简整数比例（`6:13`）。完整示例 `05_qwen-image-2.1-pe-t2i-gguf.json` 直接使用该 latent，
不再使用通用四通道 `EmptyLatentImage`；同时将 `rewritten_prompt` 接入 ComfyUI 原生
Qwen-Image-2.1 生成链，在扩散采样前自动卸载 PE 服务，并保存生成图像。

`edit` profile 支持有序的 `image1`…`image10` 传输，并使用官方编辑采样差异
（`presence_penalty=0`、`max_tokens=24000`）。发送给 PE-I2I 的图片优先使用无损 PNG，
默认限制为 1,048,576 像素和 4096 最大边。`max_image_pixels` 可配置上限为
4,194,304 像素（4MP），设为 2,097,152 即使用 2MP。提高预算可能增加视觉 tokens、
prefill 延迟、RAM/VRAM 占用和上下文压力，不保证提高结果质量。
节点会在请求时按实际图片数量追加动态规则，
不会内置官方 System Prompt，并校验返回的 `<image1>`…`<imageN>` 引用。多图输出必须
引用每张输入图。新增可选的 `image_reference_policy` 控制单图引用约定：

- `compatible`（默认）：单图允许使用 “the reference image” 等自然语言或显式
  `<image1>`；保留标签，不自动替换。适合自定义 System Prompt、通用多模态 LLM/VLM、
  Character Sheet、自定义提示词改写和实验工作流。
- `official_strict`：保留官方 PE-I2I 约定，单图 rewritten_prompt 必须自然语言引用，
  正文中出现 `<image1>` 仍会触发校验失败。

两种策略均允许单图 `ratio_follow="<image1>"`，多图均要求引用每张输入图，
均拒绝格式错误或越界标签。严格 JSON/schema 校验、比例字段互斥、T2I 校验、图片顺序
和 Canvas 尺寸语义保持不变。动态规则与一次纠正重试同步采用所选策略；debug 输出增加
policy 和图片数量。新 widget 追加在现有 optional widgets 末尾，旧工作流缺少该字段时
使用 `compatible`；需要原来的单图风格限制时请选择 `official_strict`。官方 System Prompt
继续外置。早于 `thinking` 的旧导出会按保存的 widget 名称迁移，避免位置错位。

`06_qwen-image-2.1-pe-edit-2-images-gguf.json` 展示了双图 Edit 完整图。请配置匹配的
PE-I2I GGUF、BF16 mmproj、外置 System Prompt 和原生 Qwen-Image-2.1 生成模型。示例使用
49152 context 和 `--jinja --reasoning-format none --parallel 1 --image-min-tokens 1024`。
两张参考图以相同顺序进入 Prompt Enhancer、PE Canvas 和 `TextEncodeQwenImage21`；
TextEncode 的 positive/negative 进入 KSampler，PE Canvas 的 latent 进入 KSampler。
仓库不会打包或自动下载模型、mmproj 或官方 prompt。

`07_qwen-image-2.1-pe-edit-10-images-smoke-gguf.json` 是专注于 PE-I2I 传输和结构化
输出的 smoke。它严格按 `image1`…`image10` 顺序连接 10 个 Load Image 节点，使用相同的
49152 context 与 server 参数，并由解析器强制改写提示词引用全部 10 个标签。该示例有意
停在 Prompt Enhancer；完整扩散出图请使用 06。

## 图像自动分辨率

**Llama Workbench H3 Auto Resolution Selector** 接受一个 `IMAGE`、目标 `megapixels`
值和取整 `multiple`（默认 8）。在默认的 `Auto (nearest input image)` 模式下，它会从
支持的比例中选择最接近输入图像的一个：1:1、2:3、3:2、3:4、
4:3、9:16、16:9 或 21:9。随后根据目标像素数计算尺寸，并按指定倍数取整，输出 `width`
和 `height`，可连接到 `Empty Latent Image` 等节点。也可以手动选择列表中的比例覆盖
自动选择；图像输入仍可作为工作流中的视觉参考。

当 Chat 的 `thinking` 开启时，支持推理的服务器可能会单独返回思考轨迹和最终答案。
Chat 会将思考轨迹放到 `thinking` 输出，只将最终答案放到 `assistant_message`，会话历史
也只保存最终答案。可以将两个输出连接到 **Llama Workbench Chat Output Display**，在画布
上分别预览。如果推理模板只返回思考文本而没有最终答案，Chat 会在关闭 thinking 后
恰好重试一次，以恢复最终答案；原始思考内容仍保留在 `thinking` 输出中。Chat Settings
会在每次排队时重新构建，因此清空系统提示词会在下一次运行生效。

## 服务器配置与显存管理

如果文本推理和图像生成共用一块 GPU，请保持 Start Server 的
`release_comfy_models` 开启（默认值）。启动 llama-server 前，它会调用 ComfyUI 的受管
模型卸载和 CUDA 缓存释放，效果相当于 **Free Model and Node Cache** 中的模型内存部分。
对于 35B 模型，建议将 `wait_seconds` 设置为至少 `600`；旧工作流可能仍保存着之前的
60 或 300 秒值。`cleanup_previous_server` 只保留旧工作流输入兼容，
不会扫描或终止与可执行文件、端口匹配的进程。不要用 Start Server 管理手动启动的服务器；这种
情况应使用 **Llama Workbench Connection**。

Start Server 执行的是 argv 数组，不是 shell 命令。`extra_args` 使用跨平台的双引号参数
语法，因此包含空格的路径必须加引号。本包会先用 `--version` 询问二进制文件，然后使用
输入的模型、主机、端口、上下文、GPU 层数和可选 mmproj 参数启动，并原样传递额外参数。

它要求端点兼容 `/v1/chat/completions` 或 `/chat/completions`。如果自定义构建只有
`llama-cli` 可执行文件，请创建 HTTP 适配器，或使用可选的嵌入式后端。请根据确切的
`--help` 输出和真实模型验证具体分支；本项目不声称兼容所有 llama.cpp 衍生版本。

## Skill 和安全模型

Skill 层会将包内的 `SKILL.md` 文件视为提示词指令。它可以自动选择包内 Skill，在聊天
状态中携带阶段和选项，并重新加载一轮已声明的参考资料。Skill **不会**赋予模型任意
工具权限。在 ComfyUI 图中，提示词 Skill 只能生成计划或提示词，实际生成由工作流完成。

将每个自定义 Skill 放到下面目录的独立子文件夹中：

```text
ComfyUI/custom_nodes/ComfyUI-Llama-Workbench/skills/
└── my-skill/
    ├── SKILL.md
    ├── references/
    │   └── optional-guide.md
    └── runtime.json
```

只有 `SKILL.md` 是必需文件。文件夹名以字母或数字开头，可使用字母、数字、连字符、下划线和点，例如
`my-skill`。在 `SKILL.md` 中填写名称、描述和提示词指令：

```markdown
---
name: 文本润色
description: 在保留原意的基础上，让文字更清晰、简洁。
---
润色用户提供的文字，保留原意，只输出修改后的正文。
```

参考文件放入 `references/`，支持 `.md`、`.txt`、`.json`、`.yaml` 和 `.yml`。
添加后刷新 ComfyUI 页面以重新获取节点定义，在 **Llama Workbench Skill Loader** 中
选择文件夹名，并将 `skill` 输出连接到 **Llama Workbench Chat**。
选择 `Auto` 可自动选择 Skill，选择 `None` 则进行普通聊天。
导入已有 Skill 时，将包含 `SKILL.md` 和参考文件的完整文件夹复制到同一个 `skills/` 目录即可。

`runtime.json` 是可选的纯数据声明。Skill 可以在用户消息中声明普通的 `Field: value`
选择器，将路由值映射到自身已声明的参考文件，并向模型提供输出格式指导。Chat 会在推理
前预加载选中的参考资料，但始终接受模型的最终文本；不会根据标题、时间戳、标签或其他
推断格式校验、改写、重试或拒绝响应。没有 `runtime.json` 的 Skill 使用模型请求加载参考资料的流程。

## 进程生命周期

Workbench 跟踪自己创建的确切子进程，停止操作要求租约匹配且进程空闲。
通过 Connection 连接的服务由其外部管理者负责。嵌入式模型可以通过
**Llama Workbench Release Embedded Model** 显式释放。

队列 API 将完整文本请求与 ComfyUI 原生图片执行统一调度。调度、空闲释放、取消、鉴权和
未知结果恢复规则见 [队列 API 文档](docs/queue-api.md)。

## 开发

```powershell
python -m pytest -q
python -m compileall -q .
```

测试覆盖服务器命令构建和所有权规则、响应解析、Skill 发现与路径包含检查，以及
Skill 状态解析。测试不需要 ComfyUI、模型或 GPU。

## 许可证

[MIT](LICENSE)。

## 队列 API 接入

桌面应用、批处理脚本、Web 服务和其他工作流工具均可通过队列 API 接入 ComfyUI，
让文本与图片任务共享显存调度，并获取持久请求 ID、结构化结果和定向取消能力。
详见 [API 文档](docs/queue-api.md) 和可导入的 [Postman 集合](examples/api/queue-api.postman_collection.json)。
文本推理需要通过此 API 提交，才能参与完整推理周期的 GPU 租约。
