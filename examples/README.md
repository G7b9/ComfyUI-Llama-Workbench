# Importable example workflows

All four `.json` files in this directory are ComfyUI workflow exports. Import
one through **Workflow → Open** (or drag it onto the canvas), then replace the
placeholder model paths before queuing.

| File | Covers | Required setup |
|---|---|---|
| `01_start-server-chat.json` | Owned external server plus the native chat widget | Set `model_path` and `binary_path` before running. It waits up to 600 seconds for a large first model load. Chat can accept up to 8 dynamically added images; `max_image_edge=0` keeps each input at its original resolution. Set `auto_unload=true` to stop this owned server after its response. |
| `02_existing-server-image-to-prompt.json` | Attach to an existing VLM server and reverse an image into a prompt | Enter the endpoint and connect images one by one (the next socket appears automatically) or use IMAGE batches. Set `max_images` (1–8); `max_image_edge=0` keeps the original resolution. |
| `03_skill-chat.json` | Multi-turn chat with the bundled `prompt-refiner` Skill and the separate Chat output display | Connect to a running local server, enter a Message, then queue the prompt. The display receives `thinking` and `assistant_message` independently. |
| `04_embedded-qwen-gemma-vlm.json` | Embedded Qwen/Gemma VLM image-to-prompt | Install `llama-cpp-python`; set matching model and mmproj paths. |

The sample model and binary paths are generic placeholders, not bundled files.
The server and ComfyUI process must both be able to access the paths you enter.

The Connection examples do not start or stop the target endpoint. Use
`model_name` only when attaching to a router and set it to that router's exact
model ID. The Start Server example owns only the process it launches.
