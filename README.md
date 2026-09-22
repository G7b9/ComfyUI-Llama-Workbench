# ComfyUI Llama Workbench

[中文说明](README.zh-CN.md) | English

An independent ComfyUI custom node package for local llama.cpp work. It is
designed to coexist with `comfyui-llamacpp` and `comfyUI-llama-TE`: it uses its
own node IDs (`LlamaWorkbench_*`), socket types (`LLAMA_WORKBENCH_*`), frontend
extension (`LlamaWorkbench.SafeChat`), environment variable
`LWB_LLAMA_SERVER_BINARY`, and owned-process registry.

It combines the useful product capabilities without importing or replacing the
other packages:

- Start and stop one positively-owned external `llama-server` process, with an
  explicit binary path and shell-safe custom arguments. Compatible forks such
  as a TurboQuant build work when they retain a `llama-server`-style HTTP chat
  endpoint.
- Attach to another local OpenAI-compatible `llama-server` without assuming
  ownership of it.
- Generate text or image-to-prompt results through either backend.
- Run Qwen-Image-2.1 Prompt Enhancer checkpoints through the same backend,
  with official task sampling profiles and strict structured outputs.
- Optionally load Qwen/Gemma-style local models through `llama-cpp-python`,
  including common multimodal chat handlers, KV q8_0 cache choices, and Qwen
  MoE CPU options when the installed binding supports them.
- Keep a graph-serialized chat history with one native ComfyUI message field.
- Load data-only Skills from this package's own `skills/` directory. Skills can
  request declared references, present stages and options, but are not tools
  and cannot execute shell, network, or ComfyUI actions.

## Install

Place this folder directly under `ComfyUI/custom_nodes/`:

```text
ComfyUI/custom_nodes/ComfyUI-Llama-Workbench/
```

Install the base dependencies using the same Python that starts ComfyUI:

```powershell
python -m pip install -r ComfyUI-Llama-Workbench/requirements.txt
```

The external-server backend needs a current `llama-server` binary. Set the
binary path on the node, set `LWB_LLAMA_SERVER_BINARY`, or add it to `PATH`.

For the embedded backend, install a llama-cpp-python build matching your GPU
backend:

```powershell
python -m pip install "llama-cpp-python>=0.3.37"
```

Restart ComfyUI. The nodes appear under **Llama Workbench**.

`wait_seconds` is the startup-readiness limit, not a model-generation limit.
For a first load of a large model, especially a Qwen/Gemma VLM with an mmproj,
use 600–1800 seconds. If the limit expires, Workbench terminates its owned
llama-server so its GPU memory, RAM, and port cannot block the next workflow.
Increase the limit before retrying.

For **Llama Workbench Start Server**, `timeout_seconds` is the separate limit
for each Chat or Prompt request after the server is ready. It defaults to 120
seconds, matching **Llama Workbench Connection**; use 600 seconds or more for
long reasoning or unlimited-token responses.

## Node catalog

| Node | Use |
|---|---|
| Llama Workbench Start Server | Start one model with a chosen llama-server-compatible executable. `release_comfy_models` releases ComfyUI-managed GPU models before the server starts; `cleanup_previous_server` (on by default) clears a stale server using the same executable and port. Both are useful when alternating with MiniMax H3 on one GPU. |
| Llama Workbench Connection | Attach to an existing HTTP server without lifecycle ownership. |
| Llama Workbench Stop Owned Server | Stops only the process launched by this package. |
| Llama Workbench Server Status | Shows process identity, command, and bounded log tail. |
| Llama Workbench H3 Auto Resolution Selector | Matches an input image to the nearest MiniMax H3 supported aspect ratio, then calculates compatible width and height at a target megapixel count. |
| Llama Workbench Embedded VL Model | Optional direct `llama-cpp-python` loader for Qwen/Gemma-style models. |
| Llama Workbench Release Embedded Model | Closes an embedded model explicitly. |
| Llama Workbench Prompt / Image2Prompt | Text prompting and image-to-prompt from a unified backend socket. Its image socket grows from `image` to `image1`, `image2`, and so on as connections are added (up to 10). It also has `seed`, `max_images`, `max_image_edge`, `auto_unload`, and a `thinking` control that defaults to `off`. |
| Llama Workbench Qwen Image 2.1 Prompt Enhancer | Native structured prompt rewriting over a Workbench backend. It uses the official T2I/edit sampling profiles, forces thinking on, strictly returns `rewritten_prompt`, `wh_ratio`, and `ratio_follow`, and retries one malformed response once. The matching System Prompt must be pasted or loaded from a local file. |
| Llama Workbench Chat | Interactive local chat: enter text and click its on-node **发送** button to queue only this Chat node and its upstream dependencies (no Queue Prompt click); it starts at a practical default size, remains freely resizable, and keeps long history in a scrollable canvas viewport. `clear_context_before_run` defaults to on, so every queued workflow starts fresh; turn it off for a continuing multi-turn conversation. `use_cache` defaults to on and reuses an unchanged complete request independently of `seed`; turn it off to force a fresh model request. While it is on, `release_comfy_cache_after_run` is skipped so the response remains reusable. `release_owned_server_after_run` defaults to on and stops an owned llama-server immediately after Chat responds, before downstream H3/video nodes allocate VRAM. **清空上下文** / **清空输入** actions and a text-token context meter are included. Start Server supplies the meter's `context_size` automatically; set `context_size` on an external Connection to obtain a percentage. Graph-persisted history, direct `max_tokens` / `seed` / `thinking` / `auto_unload` controls, and dynamic image sockets (up to 10) with `max_image_edge` are also included. |
| Llama Workbench Chat Output Display | Canvas-only terminal viewer that separately previews a Chat node's `thinking` and `assistant_message` outputs. |
| Llama Workbench Chat Settings | System prompt, sampling, context-history, and image-size controls. |
| Llama Workbench Skill Loader | Loads one package-local Skill, Auto selection, or normal chat. |

## Importable workflows

Five ready-to-import workflow JSON files are included in
[`examples/`](examples/README.md): owned-server chat, attached-server
image-to-prompt, Skill chat, embedded Qwen/Gemma VLM image-to-prompt, and Qwen
Image 2.1 PE-T2I GGUF prompt rewriting.
Replace their generic model and binary placeholders with paths that are visible
to the host running ComfyUI.

The Prompt / Image2Prompt node defaults to `max_tokens=-1`, which llama-server
interprets as unlimited generation. It still stops on EOS or when the model's
context handling ends the request; use a positive limit when you need bounded
latency or output size.

Prompt / Image2Prompt supports up to ten image references. Connect the first
image to `image`; it is renamed to `image1` and a new `image2` socket appears.
Each further connection exposes the next socket, while unused trailing sockets
are removed. An `IMAGE` batch is also accepted at every socket. `max_images`
limits the total images sent across all connected sockets and batches.

`max_image_edge=0` is the default and means this package does not downscale the
image before it is sent to llama-server. A positive edge limit reduces image
resolution and normally reduces VLM visual-token use. The selected vision model
or its projector can still apply its own internal resize/tokenization.

`auto_unload` defaults to `false`. When enabled on Prompt or Chat, it stops the
exact `llama-server` process started by **Llama Workbench Start Server** after
generation, or releases the Workbench embedded `llama-cpp-python` model.
**Llama Workbench Connection** is an attached endpoint and is never stopped or
unloaded automatically. Start Server and Embedded Model recheck their runtime
on every queued run, so an auto-unloaded owned model is loaded again next time.

The same dynamic image inputs are available on **Llama Workbench Chat**. Connect
its `image` socket to create `image2`; each additional connection exposes the
next socket, up to 10 images. `max_images` is the combined cap for all connected
sockets and IMAGE batches.

Chat does not impose H3-specific image-socket roles. The selected Skill and
the user's request determine how connected images are interpreted.

## Qwen-Image-2.1 Prompt Enhancer

Use **Llama Workbench Start Server** with a compatible `llama-server` and local
GGUF, then connect its backend to **Llama Workbench Qwen Image 2.1 Prompt
Enhancer**. Phase-one testing targets
[`pottokao/Qwen-Image-2.1-PE-T2I-Heretic-GGUF`](https://huggingface.co/pottokao/Qwen-Image-2.1-PE-T2I-Heretic-GGUF).
Select `t2i`, leave all image sockets disconnected, and provide the matching
Qwen PE System Prompt through exactly one of:

- `system_prompt`: paste the prompt into the node.
- `system_prompt_path`: point to a local UTF-8 `system_prompt.txt`, or to the
  local model directory that contains that file.

The repository intentionally includes neither Qwen's official System Prompt
nor the model weights. They remain governed by their upstream license and are
not MIT assets from this project.

The node owns the PE request contract rather than asking users to copy sampling
values into a generic Prompt node. Its `t2i` profile sends `temperature=1.0`,
`top_p=0.95`, `top_k=20`, `min_p=0`, `presence_penalty=1.5`,
`max_tokens=16256`, and `enable_thinking=true`. It parses exactly one JSON
object—no Markdown fences, prose, repaired JSON, aliases, or unknown fields—and
exposes `rewritten_prompt`, `wh_ratio`, and the normalized empty
`ratio_follow` as separate sockets plus `result_json`. A formatting failure
gets one corrective retry; a second failure stops the workflow with an error
instead of silently passing unreliable text downstream.

The `edit` profile and ordered `image1`…`image10` transport are already exposed
for the later PE-I2I path. It uses the official edit sampling differences
(`presence_penalty=0`, `max_tokens=24000`) and requires at least one image. Use
it only with the matching PE-I2I checkpoint, System Prompt, and multimodal
projector supplied through Start Server's `mmproj_path`; those assets are not
bundled or auto-downloaded.

## H3-compatible automatic resolution

**Llama Workbench H3 Auto Resolution Selector** accepts an `IMAGE`, a target
`megapixels` value, and a rounding `multiple` (default 8). In its default
`Auto (nearest input image)` mode it selects the closest one of the MiniMax H3
Resolution Selector ratios: 1:1, 2:3, 3:2, 3:4, 4:3, 9:16, 16:9, or 21:9. It
then uses the same target-pixel calculation and nearest-multiple rounding as
the supplied H3 selector, outputting `width` and `height` for nodes such as
Empty Latent Image. Select a listed ratio manually to override Auto; the image
input remains useful as a visual reference for the graph.

When Chat `thinking` is `on`, reasoning-capable servers may return the thought
trace separately from the final answer. Chat now keeps that trace on its
`thinking` output and keeps `assistant_message` for the final answer only; the
conversation history also contains final answers only. Connect both outputs to
**Llama Workbench Chat Output Display** for separate on-canvas previews. The
preview expands with the node and the full strings remain available from its
two outputs. If a reasoning template returns only thought text and no final
answer, Chat retries exactly once with thinking disabled to recover the final
answer; the original thought remains on the `thinking` output. Chat Settings
are rebuilt on every queue, so clearing its system prompt takes effect on the
next run.

## Custom server commands and TurboQuant-style builds

When using one GPU alternately for MiniMax H3 and llama.cpp, keep Start Server's
`release_comfy_models` enabled (the default). Before starting llama-server it
calls ComfyUI's managed-model unload and CUDA-cache release, equivalent to the
model-memory portion of **Free Model and Node Cache**. For a 35B model, also
set `wait_seconds` to at least `600`; older saved workflows may still contain
the prior 60- or 300-second value. Keep `cleanup_previous_server` enabled as well: it
stops a previous timed-out Workbench server matching this executable and port
before the next launch. Do not use Start Server to manage a manually launched
server; use **Llama Workbench Connection** for that case.

Start Server executes an argv array, never a shell command. `extra_args` uses
portable double-quote argument syntax, so paths containing spaces must be quoted.
The package asks the binary for `--version`, launches it with the typed model,
host, port, context, GPU-layer and optional mmproj arguments, then passes the
extra arguments unchanged.

It requires an HTTP endpoint compatible with `/v1/chat/completions` or
`/chat/completions`. If a custom build is a `llama-cli` executable only, create
an HTTP adapter or use the optional embedded backend instead. Verify a specific
fork with its exact `--help` output and a real model; this repository does not
claim compatibility with every llama.cpp derivative.

## Skills and safety model

The Skill layer intentionally treats package-local `SKILL.md` files as prompt instructions.
It can auto-select a package-local Skill, carry stage/options in the chat
state, and reload exactly one declared reference round. It does **not** grant a
model arbitrary tools. That keeps a prompt-writing Skill truthful in a ComfyUI
graph: it can produce a generation plan or prompt, and the graph performs any
actual generation.

To add a Skill, create:

```text
skills/my-skill/SKILL.md
skills/my-skill/references/optional-guide.md
skills/my-skill/runtime.json   # optional declarative routing and validation
```

Use simple YAML frontmatter for `name` and `description`. References are
limited to `.md`, `.txt`, `.json`, `.yaml`, and `.yml` files inside that Skill.

`runtime.json` is optional and data-only. A Skill can declare a plain
`Field: value` selector in the user message, map route values to its own
declared reference files, and provide output-format guidance to the model.
Chat preloads the selected references before inference, but always accepts the
model's final text: it does not validate, rewrite, retry, or reject a response
based on headings, timestamps, labels, or any other inferred format. The core
node has no H3-specific branch; Skills without a package-local runtime file
retain the original model-requested reference flow.

This repository does not bundle the MiniMax H3 prompt-writing Skill or its
reference guides. To install the official upstream Skill in an environment that
supports Agent Skills, follow the upstream instructions:

```text
npx skills add https://github.com/MiniMax-AI/MiniMax-H3 --skill h3-prompt-writing
```

See the [official MiniMax H3 Skill](https://github.com/MiniMax-AI/MiniMax-H3/tree/main/.agents/skills/h3-prompt-writing)
for its prompt examples and reference guides. The command above installs that
Skill outside this repository; the Llama Workbench Skill Loader only discovers
Skills placed under this package's own `skills/` directory.

This package also does not bundle MiniMax H3 model weights; use of those models
remains subject to the upstream model license.

## Coexistence and lifecycle guarantees

This package does not monkey-patch `comfy.model_management`, register the
shared `LLM` model folder, use another package's node IDs, or kill processes by
name. Its stop node only targets the `subprocess.Popen` instance it created.
An attached connection is never stopped by this package. Embedded model cleanup
is explicit; it does not intercept ComfyUI's global unload action.

## Development

```powershell
python -m pytest -q
python -m compileall -q .
```

The tests cover server command construction and ownership rules, response
parsing, Skill discovery/path containment, and Skill state parsing. They do not
need ComfyUI, a model, or a GPU.

## License

MIT. This project was independently implemented. It does not include source
code or bundled assets copied from the two neighbouring custom-node folders.
