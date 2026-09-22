# ComfyUI Llama Workbench

[English](README.md) | 中文

一个用于本地 llama.cpp 工作流的独立 ComfyUI 自定义节点包。本项目可以与
`comfyui-llamacpp` 和 `comfyUI-llama-TE` 共存，使用独立的节点 ID
(`LlamaWorkbench_*`)、socket 类型 (`LLAMA_WORKBENCH_*`)、前端扩展名
(`LlamaWorkbench.SafeChat`)、环境变量 `LWB_LLAMA_SERVER_BINARY` 以及自有的
进程注册表。

它整合了以下功能，同时不会导入或替换其他节点包：

- 启动和停止一个由本包明确拥有的外部 `llama-server` 进程，支持指定二进制路径和
  shell 安全的自定义参数。只要兼容分支保留 `llama-server` 风格的 HTTP 聊天端点，
  TurboQuant 等分支也可以使用。
- 连接另一个本地 OpenAI 兼容的 `llama-server`，不会假定拥有该进程的生命周期。
- 通过任一后端生成文本或图像反推提示词结果。
- 可选地通过 `llama-cpp-python` 加载 Qwen/Gemma 风格的本地模型，包括常见的多模态
  聊天处理器、KV q8_0 缓存选项，以及在已安装绑定支持时使用 Qwen MoE CPU 选项。
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

`wait_seconds` 是启动就绪等待上限，不是模型生成上限。首次加载大型模型，尤其是带
mmproj 的 Qwen/Gemma VLM 时，建议使用 600–1800 秒。如果等待超时，Workbench 会终止
它所拥有的 llama-server，避免 GPU 显存、内存和端口阻塞后续工作流。调整等待时间后再
重试即可。

对于 **Llama Workbench Start Server**，`timeout_seconds` 是服务器就绪后每次 Chat 或
Prompt 请求的独立等待上限，默认 120 秒，与 **Llama Workbench Connection** 相同。长
推理或不限 token 的响应建议使用 600 秒或更高值。

## 节点目录

| 节点 | 用途 |
|---|---|
| Llama Workbench Start Server | 使用指定的 llama-server 兼容可执行文件启动一个模型。服务器启动前，`release_comfy_models` 会释放 ComfyUI 管理的 GPU 模型；默认开启的 `cleanup_previous_server` 会清理使用相同可执行文件和端口的陈旧服务器。两者适合在同一块 GPU 上交替使用 MiniMax H3。 |
| Llama Workbench Connection | 连接已有的 HTTP 服务器，不接管其生命周期。 |
| Llama Workbench Stop Owned Server | 只停止由本包启动的进程。 |
| Llama Workbench Server Status | 显示进程身份、命令和有限长度的日志尾部。 |
| Llama Workbench H3 Auto Resolution Selector | 将输入图像匹配到最接近的 MiniMax H3 支持比例，并根据目标百万像素计算兼容的宽高。 |
| Llama Workbench Embedded VL Model | 可选的 `llama-cpp-python` 直接加载器，适用于 Qwen/Gemma 风格模型。 |
| Llama Workbench Release Embedded Model | 显式关闭嵌入式模型。 |
| Llama Workbench Prompt / Image2Prompt | 通过统一后端 socket 进行文本提示或图像反推提示词。图像 socket 会随着连接从 `image` 变为 `image1`、`image2` 等，最多 8 个；还提供 `seed`、`max_images`、`max_image_edge`、`auto_unload` 以及默认关闭的 `thinking` 控制。 |
| Llama Workbench Chat | 交互式本地聊天：输入文本后点击节点上的 **发送** 按钮，只会将此 Chat 节点及其上游依赖加入队列，不需要点击 Queue Prompt。节点有实用的初始尺寸，可以自由调整大小，长历史记录会在可滚动的画布区域中显示。`clear_context_before_run` 默认开启，每次排队的工作流都会从新上下文开始；关闭后可继续多轮对话。`use_cache` 默认开启，会独立于 `seed` 重用未变化的完整请求；关闭后可强制新的模型请求。缓存开启时会跳过 `release_comfy_cache_after_run`，以便继续复用响应。`release_owned_server_after_run` 默认开启，Chat 响应后会立即停止自有 llama-server，再让下游 H3/视频节点分配显存。节点包含 **清空上下文** / **清空输入** 操作和文本 token 上下文计量器。Start Server 会自动提供计量器所需的 `context_size`；外部 Connection 需要设置 `context_size` 才能显示百分比。还支持图持久化历史、直接的 `max_tokens` / `seed` / `thinking` / `auto_unload` 控制，以及带 `max_image_edge` 的动态图像 socket（最多 8 张）。 |
| Llama Workbench Chat Output Display | 仅画布终端查看器，分别预览 Chat 节点的 `thinking` 和 `assistant_message` 输出。 |
| Llama Workbench Chat Settings | 系统提示词、采样、上下文历史和图像尺寸控制。 |
| Llama Workbench Skill Loader | 加载一个包内 Skill、自动选择 Skill，或进行普通聊天。 |

## 可导入工作流

[`examples/`](examples/README.md) 中包含四个可直接导入的工作流 JSON：自有服务器聊天、
连接已有服务器进行图像反推提示词、Skill 聊天，以及嵌入式 Qwen/Gemma VLM 图像反推
提示词。运行前请将其中的通用模型和二进制占位路径替换为 ComfyUI 主机可访问的路径。

Prompt / Image2Prompt 节点的 `max_tokens` 默认值为 `-1`，llama-server 会将其解释为
不限生成长度。生成仍会在 EOS 或模型上下文处理结束时停止；如果需要限制延迟或输出
大小，请使用正数上限。

Prompt / Image2Prompt 最多支持 8 个图像参考。连接第一个图像到 `image` 后，它会重命名
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
`image2`；每增加一个连接，就会暴露下一个 socket，最多 8 张图像。`max_images` 是所有
已连接 socket 和 `IMAGE` batch 的合计上限。

Chat 不规定 H3 专用的图像 socket 角色。连接图像的具体解释由选中的 Skill 和用户请求
决定。

## H3 兼容的自动分辨率

**Llama Workbench H3 Auto Resolution Selector** 接受一个 `IMAGE`、目标 `megapixels`
值和取整 `multiple`（默认 8）。在默认的 `Auto (nearest input image)` 模式下，它会从
MiniMax H3 Resolution Selector 的比例中选择最接近输入图像的一个：1:1、2:3、3:2、3:4、
4:3、9:16、16:9 或 21:9。随后使用相同的目标像素计算和最近倍数取整方式输出 `width`
和 `height`，可连接到 `Empty Latent Image` 等节点。也可以手动选择列表中的比例覆盖
自动选择；图像输入仍可作为工作流中的视觉参考。

当 Chat 的 `thinking` 开启时，支持推理的服务器可能会单独返回思考轨迹和最终答案。
Chat 会将思考轨迹放到 `thinking` 输出，只将最终答案放到 `assistant_message`，会话历史
也只保存最终答案。可以将两个输出连接到 **Llama Workbench Chat Output Display**，在画布
上分别预览。如果推理模板只返回思考文本而没有最终答案，Chat 会在关闭 thinking 后
恰好重试一次，以恢复最终答案；原始思考内容仍保留在 `thinking` 输出中。Chat Settings
会在每次排队时重新构建，因此清空系统提示词会在下一次运行生效。

## 自定义服务器命令和 TurboQuant 风格构建

如果在同一块 GPU 上交替运行 MiniMax H3 和 llama.cpp，请保持 Start Server 的
`release_comfy_models` 开启（默认值）。启动 llama-server 前，它会调用 ComfyUI 的受管
模型卸载和 CUDA 缓存释放，效果相当于 **Free Model and Node Cache** 中的模型内存部分。
对于 35B 模型，建议将 `wait_seconds` 设置为至少 `600`；旧工作流可能仍保存着之前的
60 或 300 秒值。同时保持 `cleanup_previous_server` 开启，它会在下一次启动前停止与可执行
文件和端口匹配的旧 Workbench 服务器。不要用 Start Server 管理手动启动的服务器；这种
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

添加 Skill 时可以创建：

```text
skills/my-skill/SKILL.md
skills/my-skill/references/optional-guide.md
skills/my-skill/runtime.json   # 可选的声明式路由和校验
```

使用简单的 YAML frontmatter 设置 `name` 和 `description`。参考资料仅限于该 Skill 目录
内的 `.md`、`.txt`、`.json`、`.yaml` 和 `.yml` 文件。

`runtime.json` 是可选的纯数据声明。Skill 可以在用户消息中声明普通的 `Field: value`
选择器，将路由值映射到自身已声明的参考文件，并向模型提供输出格式指导。Chat 会在推理
前预加载选中的参考资料，但始终接受模型的最终文本；不会根据标题、时间戳、标签或其他
推断格式校验、改写、重试或拒绝响应。核心节点没有 H3 专用分支；没有 runtime 文件的
Skill 会保留模型请求参考资料的原始流程。

本仓库不内置 MiniMax H3 prompt-writing Skill 或其参考指南。如果你使用的环境支持
Agent Skills，请按照上游说明安装官方 Skill：

```text
npx skills add https://github.com/MiniMax-AI/MiniMax-H3 --skill h3-prompt-writing
```

可在[官方 MiniMax H3 Skill](https://github.com/MiniMax-AI/MiniMax-H3/tree/main/.agents/skills/h3-prompt-writing)
中查看提示词示例和参考指南。上述命令会将 Skill 安装到本仓库之外；Llama Workbench
Skill Loader 只会发现本包自己的 `skills/` 目录中的 Skill。

本包也不包含 MiniMax H3 模型权重；使用这些模型时仍需遵守上游模型许可证。

## 共存和生命周期保证

本包不会 monkey-patch `comfy.model_management`、注册共享的 `LLM` 模型目录、使用其他
包的节点 ID，也不会按进程名杀进程。停止节点只会定位并停止它自己通过
`subprocess.Popen` 创建的进程。

外部连接永远不会被本包停止。嵌入式模型的清理是显式执行的，不会拦截 ComfyUI 的全局
卸载操作。

## 开发

```powershell
python -m pytest -q
python -m compileall -q .
```

测试覆盖服务器命令构建和所有权规则、响应解析、Skill 发现与路径包含检查，以及
Skill 状态解析。测试不需要 ComfyUI、模型或 GPU。

## 许可证

MIT。本项目为独立实现，不包含从相邻两个自定义节点目录复制的源代码或捆绑资源。
