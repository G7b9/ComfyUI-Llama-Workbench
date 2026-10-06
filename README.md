# ComfyUI Llama Workbench

[中文说明](README.zh-CN.md) | English

A ComfyUI custom node package for local text generation, multimodal prompting,
interactive chat, and reusable prompt Skills. Connect to an existing
OpenAI-compatible server, start a local `llama-server`, or use the optional
`llama-cpp-python` backend.

Features:

- Start and stop one positively-owned external `llama-server` process, with an
  explicit binary path and shell-safe custom arguments supported by that executable.
- Attach to another local OpenAI-compatible `llama-server` without assuming
  ownership of it.
- Generate text or image-to-prompt results through either backend.
- Run Qwen-Image-2.1 Prompt Enhancer checkpoints through the same backend,
  with official task sampling profiles, strict structured outputs, native
  64-channel Qwen-Image-2.1 latents, and ordered PE-I2I image references.
- Optionally load Qwen/Gemma-style local models through `llama-cpp-python`,
  including common multimodal chat handlers, KV q8_0 cache choices, and Qwen
  MoE CPU options when the installed binding supports them.
- Queue text requests alongside image workflows through an authenticated API,
  with persistent request IDs, structured results, and coordinated GPU use.
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

On **Llama Workbench Start Server**, `wait_seconds` is the startup-readiness limit,
not a model-generation limit.
For a first load of a large model, especially a Qwen/Gemma VLM with an mmproj,
use 600–1800 seconds. If the limit expires, Workbench terminates its owned
llama-server so its GPU memory, RAM, and port cannot block the next workflow.
Increase the limit before retrying. The queue API has a separate timeout policy:
unknown startup outcomes retain their GPU lease; see [Queue API](docs/queue-api.md).

For **Llama Workbench Start Server**, `timeout_seconds` is the separate limit
for each Chat or Prompt request after the server is ready. It defaults to 120
seconds, matching **Llama Workbench Connection**; use 600 seconds or more for
long reasoning or unlimited-token responses.

## Node catalog

| Node | Use |
|---|---|
| Llama Workbench Start Server | Start one model with a chosen llama-server-compatible executable. `release_comfy_models` releases ComfyUI-managed GPU models before the server starts; `cleanup_previous_server` is now a compatibility-only input; external processes are never discovered or killed by executable/port. |
| Llama Workbench Connection | Attach to an existing HTTP server without lifecycle ownership. |
| Llama Workbench Stop Owned Server | Stops only the process launched by this package. |
| Llama Workbench Server Status | Shows process identity, lease state, and a bounded startup log tail. |
| Llama Workbench H3 Auto Resolution Selector | Matches an input image to a supported aspect ratio and calculates width and height at a target megapixel count. |
| Llama Workbench Embedded VL Model | Optional direct `llama-cpp-python` loader for Qwen/Gemma-style models. |
| Llama Workbench Release Embedded Model | Closes an embedded model explicitly. |
| Llama Workbench Prompt / Image2Prompt | Text prompting and image-to-prompt from a unified backend socket. Its image socket grows from `image` to `image1`, `image2`, and so on as connections are added (up to 10). It also has `seed`, `max_images`, `max_image_edge`, `auto_unload`, and a `thinking` control that defaults to `off`. |
| Llama Workbench Qwen Image 2.1 Prompt Enhancer | Native structured prompt rewriting over a Workbench backend. It uses the official T2I/edit sampling profiles, validates `rewritten_prompt`, `wh_ratio`, `ratio_follow`, and edit image references, and exposes a default-on `thinking` switch. A malformed or truncated response gets one retry with thinking disabled. Task-specific System Prompts can be pasted, loaded from a file, or discovered next to a local model. A `debug` checkbox can print bounded raw answers and validation errors. |
| Llama Workbench Qwen Image 2.1 PE Canvas | Resolves `wh_ratio`, `ratio_follow=<imageN>`, an optional manual ratio override, and `follow_input_size`, then emits width, height, `ratio_source`, and a native `[1,64,H/16,W/16]` Qwen-Image-2.1 `LATENT`. |
| Llama Workbench Qwen Image 2.1 PE Resolution | Compatibility dimensions-only helper for existing workflows. New Qwen-Image-2.1 workflows should use PE Canvas so KSampler receives the correct 64-channel latent. |
| Llama Workbench Image Info | Reads an `IMAGE` batch without copying pixels and outputs width, height, longest/shortest side, pixels, megapixels, aspect ratio, orientation, batch size, channels, and complete tensor metadata. |
| Llama Workbench Image Info Display | Formats complete Image Info metadata as detailed, compact, or JSON text in Chinese or English and shows it in a scrollable canvas panel. |
| Llama Workbench Pad Image to Multiple | Pads image width and height up to a selected multiple with a native color picker, optional alpha, and nine placement choices; outputs exact reversible padding metadata. |
| Llama Workbench Restore Image from Padding | Uses the padding metadata to crop a processed padded image back to the original content rectangle and dimensions. |
| Llama Workbench Chat | Interactive local chat: enter text and click its on-node **发送** button to queue only this Chat node and its upstream dependencies (no Queue Prompt click); it starts at a practical default size, remains freely resizable, and keeps long history in a scrollable canvas viewport. `clear_context_before_run` defaults to on, so every queued workflow starts fresh; turn it off for a continuing multi-turn conversation. `use_cache` defaults to on and reuses an unchanged complete request independently of `seed`; turn it off to force a fresh model request. While it is on, `release_comfy_cache_after_run` is skipped so the response remains reusable. `release_owned_server_after_run` defaults to on and stops an owned llama-server immediately after Chat responds, before downstream image or video nodes allocate VRAM. **清空上下文** / **清空输入** actions and a text-token context meter are included. Start Server supplies the meter's `context_size` automatically; set `context_size` on an external Connection to obtain a percentage. Graph-persisted history, direct `max_tokens` / `seed` / `thinking` / `auto_unload` controls, and dynamic image sockets (up to 10) with `max_image_edge` are also included. |
| Llama Workbench Chat Output Display | Canvas-only terminal viewer that separately previews a Chat node's `thinking` and `assistant_message` outputs. |
| Llama Workbench Chat Settings | System prompt, sampling, context-history, and image-size controls. |
| Llama Workbench Skill Loader | Loads one package-local Skill, Auto selection, or normal chat. |
| Llama Workbench Seed | Outputs an integer seed with one-click fixed random, randomize-each-run, increment, decrement, and fixed controls. |

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

The selected Skill and the user's request determine how connected images are interpreted.

Chat keeps its message input after completion, so changing `thinking` or other
request parameters and queuing again uses the same message. Use **清空输入** to
clear it explicitly. Changed inputs invalidate the corresponding cached result;
editing a widget alone does not queue a new run or alter a request already running.

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
object (one complete outer JSON Markdown fence is tolerated)—no prose, repaired JSON, aliases, or unknown fields—and
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
default to 1,048,576 pixels and a 4096-pixel maximum edge. `max_image_pixels`
has a configurable maximum of 4,194,304 pixels (4MP); 2,097,152 selects 2MP.
Higher budgets may increase visual tokens, prefill latency, RAM/VRAM usage,
and context pressure; they do not guarantee better results. At request time the
node adds a dynamic rule for the exact image count, without embedding the
official System Prompt, and validates the returned `<image1>`…`<imageN>`
references. The optional `image_reference_policy` selects the edit convention:

- `compatible` (default): a single-image rewrite may use natural language such
  as "the reference image" or the explicit `<image1>` tag. Tags are preserved,
  not replaced. Suitable for custom System Prompts, generic multimodal LLM/VLMs,
  Character Sheet, custom prompt rewriting, and experimental workflows.
- `official_strict`: retains the official PE-I2I convention: a single-image
  rewrite must refer naturally to the image, without `<image1>` in its text.

Both policies allow single-image `ratio_follow="<image1>"`, require every
input tag for multiple images, and reject malformed or out-of-range tags.
Strict JSON/schema checks, exclusive `wh_ratio`/`ratio_follow`, T2I validation,
image ordering, and Canvas sizing are unchanged. Runtime mapping and the one
corrective retry use the selected policy. Debug output includes the policy and
image count. The widget is appended after existing optional widgets; old workflows
without it use `compatible`. Select `official_strict` to retain the previous
single-image style rejection. Older exports that predate `thinking` are restored
by their saved widget names to avoid positional shifts. Official System Prompts remain external.

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

## Automatic image resolution

**Llama Workbench H3 Auto Resolution Selector** accepts an `IMAGE`, a target
`megapixels` value, and a rounding `multiple` (default 8). In its default
`Auto (nearest input image)` mode it selects the closest supported aspect ratio:
1:1, 2:3, 3:2, 3:4, 4:3, 9:16, 16:9, or 21:9. It calculates dimensions from
the target pixel count, rounds them to the selected multiple, and outputs
`width` and `height` for nodes such as
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

## Native workflow access

By default, Workbench uses ComfyUI's own access control for native workflows.
**Every caller allowed to submit a ComfyUI workflow can use Start Server and
Stop Server**, including users opening the ComfyUI web UI from another LAN
machine. Existing workflows keep their executable path, model path, port and
launch arguments; no additional Workbench token or profile migration is required.
This policy applies equally to `/prompt` and `/api/prompt`.

The server-only environment variable `LWB_RESTRICT_LIFECYCLE` defaults to `false`.
To restore the previous restrictive behavior on an existing deployment, set it
in ComfyUI's startup environment:

```sh
export LWB_RESTRICT_LIFECYCLE=true
```

Strict mode preserves the previous loopback/Origin checks and rejects remote
Start/Stop submissions even with a valid Workbench token. When API tokens are
configured, strict mode also requires them for local lifecycle submissions.
Accepted values are `true/false`, `1/0`, `yes/no`, and `on/off`, ignoring case and
surrounding whitespace. Empty or other values produce a startup configuration
error. The setting is read when the plugin initializes; workflow inputs,
request bodies, headers and query parameters cannot override it. Apply changes
at the next planned ComfyUI restart.

In both modes, direct submission of the internal `QueuedText` node is forbidden,
and authentication for `/lwb/v1/*` management APIs remains independent. Process
ownership checks, model reuse, execution leases and GPU scheduling are unchanged.

## Server configuration and GPU memory

When sharing a GPU between image generation and text inference, keep Start Server's
`release_comfy_models` enabled (the default). Before starting llama-server it
calls ComfyUI's managed-model unload and CUDA-cache release, equivalent to the
model-memory portion of **Free Model and Node Cache**. For a 35B model, also
set `wait_seconds` to at least `600`; older saved workflows may still contain
the prior 60- or 300-second value. `cleanup_previous_server` is retained only for workflow compatibility; it never
scans or terminates processes by executable or port. Do not use Start Server to manage a manually launched
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

Place each custom Skill in its own subdirectory under:

```text
ComfyUI/custom_nodes/ComfyUI-Llama-Workbench/skills/
└── my-skill/
    ├── SKILL.md
    ├── references/
    │   └── optional-guide.md
    └── runtime.json
```

Only `SKILL.md` is required. Use a directory name such as `my-skill`, starting
with a letter or digit and containing letters, digits, hyphens, underscores, or dots. Start `SKILL.md` with a name,
description, and your instructions:

```markdown
---
name: Text editor
description: Rewrite text clearly while preserving its meaning.
---
Rewrite the user's text in concise language. Return only the revised text.
```

Put supporting files in `references/`; supported types are `.md`, `.txt`,
`.json`, `.yaml`, and `.yml`. After adding a Skill, refresh the ComfyUI page to
reload the node definitions, then select its directory name in **Llama Workbench
Skill Loader** and connect its `skill` output to **Llama Workbench Chat**.
Select `Auto` for automatic selection or `None` for regular chat. For an
existing Skill, copy its whole directory, including any referenced files,
into this same `skills/` location.

`runtime.json` is optional and data-only. A Skill can declare a plain
`Field: value` selector in the user message, map route values to its own
declared reference files, and provide output-format guidance to the model.
Chat preloads the selected references before inference, but always accepts the
model's final text: it does not validate, rewrite, retry, or reject a response
based on headings, timestamps, labels, or any other inferred format. Skills
without `runtime.json` use model-requested reference loading.

## Process lifecycle

Workbench tracks the exact child process it starts. Stop requests require the
matching lease and an idle process; services attached through Connection remain
under their external owner's control. Embedded models can be released explicitly
with **Llama Workbench Release Embedded Model**.

The queue API coordinates complete text requests with native ComfyUI image
execution. See [Queue API](docs/queue-api.md) for scheduling, idle release,
cancellation, authentication, and recovery after uncertain results.

## Development

```powershell
python -m pytest -q
python -m compileall -q .
```

The tests cover server command construction and ownership rules, response
parsing, Skill discovery/path containment, and Skill state parsing. They do not
need ComfyUI, a model, or a GPU.

## License

[MIT](LICENSE).

## Queue API integration

Use the queue API to integrate desktop applications, batch scripts, web services,
or other workflow tools with ComfyUI. Text and image jobs share GPU scheduling;
requests have persistent IDs, structured results, and targeted cancellation.
See [API documentation](docs/queue-api.md) and the importable
[Postman collection](examples/api/queue-api.postman_collection.json).
Submit text through this API to participate in the GPU lease.
