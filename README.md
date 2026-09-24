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
  with official task sampling profiles, strict structured outputs, native
  64-channel Qwen-Image-2.1 latents, and ordered PE-I2I image references.
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
| Llama Workbench Qwen Image 2.1 Prompt Enhancer | Native structured prompt rewriting over a Workbench backend. It uses the official T2I/edit sampling profiles, validates `rewritten_prompt`, `wh_ratio`, `ratio_follow`, and edit image references, and exposes a default-on `thinking` switch. A malformed or truncated response gets one retry with thinking disabled. Task-specific System Prompts can be pasted, loaded from a file, or discovered next to a local model. A `debug` checkbox can print bounded raw answers and validation errors. |
| Llama Workbench Qwen Image 2.1 PE Canvas | Resolves `wh_ratio`, `ratio_follow=<imageN>`, an optional manual ratio override, and `follow_input_size`, then emits width, height, `ratio_source`, and a native `[1,64,H/16,W/16]` Qwen-Image-2.1 `LATENT`. |
| Llama Workbench Qwen Image 2.1 PE Resolution | Compatibility dimensions-only helper for existing workflows. New Qwen-Image-2.1 workflows should use PE Canvas so KSampler receives the correct 64-channel latent. |
| Llama Workbench Chat | Interactive local chat: enter text and click its on-node **发送** button to queue only this Chat node and its upstream dependencies (no Queue Prompt click); it starts at a practical default size, remains freely resizable, and keeps long history in a scrollable canvas viewport. `clear_context_before_run` defaults to on, so every queued workflow starts fresh; turn it off for a continuing multi-turn conversation. `use_cache` defaults to on and reuses an unchanged complete request independently of `seed`; turn it off to force a fresh model request. While it is on, `release_comfy_cache_after_run` is skipped so the response remains reusable. `release_owned_server_after_run` defaults to on and stops an owned llama-server immediately after Chat responds, before downstream H3/video nodes allocate VRAM. **清空上下文** / **清空输入** actions and a text-token context meter are included. Start Server supplies the meter's `context_size` automatically; set `context_size` on an external Connection to obtain a percentage. Graph-persisted history, direct `max_tokens` / `seed` / `thinking` / `auto_unload` controls, and dynamic image sockets (up to 10) with `max_image_edge` are also included. |
| Llama Workbench Chat Output Display | Canvas-only terminal viewer that separately previews a Chat node's `thinking` and `assistant_message` outputs. |
| Llama Workbench Chat Settings | System prompt, sampling, context-history, and image-size controls. |
| Llama Workbench Skill Loader | Loads one package-local Skill, Auto selection, or normal chat. |
| Llama Workbench Seed | Outputs an integer seed using ComfyUI's native fixed, random, increment, and decrement controls. |

## Importable workflows

Seven ready-to-import workflow JSON files are included in
[`examples/`](examples/README.md): owned-server chat, attached-server
image-to-prompt, Skill chat, embedded Qwen/Gemma VLM image-to-prompt, and Qwen
Image 2.1 PE-T2I, two-image PE-I2I generation, and a ten-image PE-I2I transport
smoke test.
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
Qwen PE System Prompt. The node selects the task-specific prompt automatically:

- `t2i_system_prompt` / `edit_system_prompt`: paste an official prompt for each task.
- `t2i_system_prompt_path` / `edit_system_prompt_path`: point each task at a prompt file or model directory.
- `system_prompt` / `system_prompt_path`: legacy current-task override inputs retained for old workflows.
- `auto_load_system_prompt`: enabled by default; with no manual override, search the Start Server or
  Embedded model directory for `system_prompt_t2i.txt`, `system_prompt_edit.txt`, then fall back to
  `system_prompt.txt`.

Task-specific inputs take precedence over the legacy generic inputs. Switching `task` also switches
the prompt filename, sampling profile, and output validation, but it does not switch the model already
loaded by the backend: `t2i` must use PE-T2I and `edit` must use PE-I2I with its matching mmproj.

The repository intentionally includes neither Qwen's official System Prompt
nor the model weights. They remain governed by their upstream license and are
not MIT assets from this project.

The node owns the PE request contract rather than asking users to copy sampling
values into a generic Prompt node. Its `t2i` profile sends `temperature=1.0`,
`top_p=0.95`, `top_k=20`, `min_p=0`, `presence_penalty=1.5`,
`max_tokens=16256`, and `enable_thinking=true` when the node's `thinking` is `on`.
Set `thinking` to `off` to request the final JSON without the reasoning pass, which
reduces latency and token use but may reduce rewrite quality. It parses exactly one JSON
object—no Markdown fences, prose, repaired JSON, aliases, or unknown fields—and
exposes `rewritten_prompt`, `wh_ratio`, and the normalized empty
`ratio_follow` as separate sockets plus `result_json`. A formatting failure or
a generation truncated by the server gets one corrective retry. That retry
turns thinking off and asks only for the final JSON; a second failure stops the
workflow instead of silently passing unreliable text downstream.

For temporary Prompt Enhancer output diagnostics, either enable the node's
`debug` checkbox or set the environment variable `LWB_PROMPT_REWRITE_DEBUG=1`
before starting ComfyUI. The ComfyUI console then prints each bounded answer
(including the returned `wh_ratio`), finish reason, and validation error. Both
are off by default; restart ComfyUI after changing its environment.

Connect both `wh_ratio` and `ratio_follow` to **Llama Workbench Qwen Image 2.1
PE Canvas**. It emits rounded `width` and `height`, a diagnostic
`ratio_source`, and the native Qwen-Image-2.1 latent shape
`[1,64,H/16,W/16]`. `aspect_ratio_override` can force a ratio without changing
the rewritten prompt. In edit workflows, `ratio_follow=<imageN>` selects that
ordered reference image; `follow_input_size=true` preserves its rounded input
size, while `false` preserves only its aspect ratio at the selected megapixel
budget. The enhancer accepts finite-decimal aspect ratios such as `9:19.5` and
normalizes them to the simplest integer form (`6:13`) before sizing. The complete
`05_qwen-image-2.1-pe-t2i-gguf.json` example uses this
latent directly rather than the generic four-channel `EmptyLatentImage`, wires
`rewritten_prompt` into ComfyUI's native Qwen-Image-2.1 generation chain,
auto-unloads the PE server before diffusion sampling, and saves the image.

The `edit` profile supports ordered `image1`…`image10` transport and uses the
official edit sampling differences (`presence_penalty=0`,
`max_tokens=24000`). Images sent to PE-I2I are lossless PNGs constrained by
default to 1,048,576 pixels and a 4096-pixel maximum edge. At request time the
node adds a dynamic rule for the exact image count, without embedding the
official System Prompt, and validates the returned `<image1>`…`<imageN>`
references. Multi-image output must reference every input image, while a
single-image rewritten prompt must not include an image tag.

`06_qwen-image-2.1-pe-edit-2-images-gguf.json` demonstrates the two-image edit
graph. Configure a matching PE-I2I GGUF, BF16 mmproj, external System Prompt,
and the native Qwen-Image-2.1 generation models. It uses context 49152 and
`--jinja --reasoning-format none --parallel 1 --image-min-tokens 1024`. The two
reference images enter Prompt Enhancer, PE Canvas, and
`TextEncodeQwenImage21` in the same order; TextEncode supplies KSampler's
positive and negative conditioning, while PE Canvas supplies its latent. No
model, mmproj, or official prompt is bundled or downloaded automatically.

`07_qwen-image-2.1-pe-edit-10-images-smoke-gguf.json` is a focused PE-I2I
transport and structured-output smoke test. It connects ten Load Image nodes
in strict `image1`…`image10` order, uses the same 49152 context and server
arguments, and relies on the parser to require all ten tags in the rewritten
prompt. It intentionally stops after Prompt Enhancer; use 06 for the complete
diffusion-generation graph.

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
