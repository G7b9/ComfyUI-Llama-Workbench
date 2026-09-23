# Importable example workflows

All seven `.json` files in this directory are ComfyUI workflow exports. Import
one through **Workflow → Open** (or drag it onto the canvas), then replace the
placeholder model paths before queuing.

| File | Covers | Required setup |
|---|---|---|
| `01_start-server-chat.json` | Owned external server plus the native chat widget | Set `model_path` and `binary_path` before running. It waits up to 600 seconds for a large first model load. Chat can accept up to 10 dynamically added images; `max_image_edge=0` keeps each input at its original resolution. Set `auto_unload=true` to stop this owned server after its response. |
| `02_existing-server-image-to-prompt.json` | Attach to an existing VLM server and reverse an image into a prompt | Enter the endpoint and connect images one by one (the next socket appears automatically) or use IMAGE batches. Set `max_images` (1–10); `max_image_edge=0` keeps the original resolution. |
| `03_skill-chat.json` | Multi-turn chat with the bundled `prompt-refiner` Skill and the separate Chat output display | Connect to a running local server, enter a Message, then queue the prompt. The display receives `thinking` and `assistant_message` independently. |
| `04_embedded-qwen-gemma-vlm.json` | Embedded Qwen/Gemma VLM image-to-prompt | Install `llama-cpp-python`; set matching model and mmproj paths. |
| `05_qwen-image-2.1-pe-t2i-gguf.json` | End-to-end PE-T2I GGUF → native Qwen-Image-2.1 generation → saved image | Set the PE GGUF, `llama-server`, and local matching `system_prompt.txt` paths. Install the diffusion model, text encoder, and VAE selected in the loader nodes. PE Canvas uses the model ratio or a manual override and emits the native 64-channel latent. The PE server auto-unloads before sampling. The workflow includes no model or official prompt asset. |
| `06_qwen-image-2.1-pe-edit-2-images-gguf.json` | Two ordered reference images → PE-I2I GGUF + BF16 mmproj → native Qwen-Image-2.1 edit → saved image | Set matching PE-I2I GGUF, BF16 mmproj, `llama-server`, external `system_prompt.txt`, diffusion model, text encoder, and VAE paths/selections. Replace both Load Image placeholders. Keep both references in the same order across Prompt Enhancer, PE Canvas, and `TextEncodeQwenImage21`. The workflow uses context 49152, lossless budgeted PE images, native conditioning, and a Canvas-generated 64-channel latent. |
| `07_qwen-image-2.1-pe-edit-10-images-smoke-gguf.json` | Ten ordered references → PE-I2I structured-output transport smoke | Set matching PE-I2I GGUF, BF16 mmproj, `llama-server`, and external `system_prompt.txt`, then populate all ten Load Image nodes. It uses context 49152 and requires the returned rewrite to reference every `<image1>`…`<image10>` tag. It intentionally stops after Prompt Enhancer; use 06 for end-to-end diffusion generation. |

The sample model and binary paths are generic placeholders, not bundled files.
The server and ComfyUI process must both be able to access the paths you enter.

The Connection examples do not start or stop the target endpoint. Use
`model_name` only when attaching to a router and set it to that router's exact
model ID. The Start Server example owns only the process it launches.
