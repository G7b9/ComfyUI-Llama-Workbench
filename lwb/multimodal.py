"""Bounded local image inputs for managed Chat, not the text requests API."""

import base64
from io import BytesIO
import warnings

from PIL import Image


def has_media(messages):
    return any(
        isinstance(m.get("content"), list) and any(p.get("type") != "text" for p in m["content"])
        for m in messages
    )


def ninfer_count_body(payload):
    """Translate the same turns/kwargs to NInfer's real media-aware count API.

    No remote URLs, filesystem URLs, tools, video or audio. Nodes already encode
    pixels as data URIs; enforce limits here as well before host/GPU decoding.
    """
    turns, images, total_bytes = [], 0, 0
    for message in payload["messages"]:
        if set(message) - {"role", "content"} or message.get("role") not in {
            "system",
            "user",
            "assistant",
        }:
            raise ValueError("unsupported managed multimodal message")
        content = message["content"]
        parts = [{"type": "text", "text": content}] if isinstance(content, str) else content
        translated = []
        for part in parts:
            if part.get("type") == "text" and set(part) == {"type", "text"}:
                translated.append({"type": "input_text", "text": part["text"]})
                continue
            if part.get("type") != "image_url" or set(part) != {"type", "image_url"}:
                raise ValueError("managed Chat accepts only text and image_url parts")
            image = part["image_url"]
            if not isinstance(image, dict) or set(image) - {"url", "detail"}:
                raise ValueError("invalid image_url")
            uri = image.get("url", "")
            if len(uri) > 24 * 1024 * 1024 or not uri.startswith(
                ("data:image/png;base64,", "data:image/jpeg;base64,", "data:image/webp;base64,")
            ):
                raise ValueError("managed images require bounded PNG/JPEG/WebP base64 data URIs")
            raw = base64.b64decode(uri.split(",", 1)[1], validate=True)
            total_bytes += len(raw)
            images += 1
            if total_bytes > 32 * 1024 * 1024 or images > 10:
                raise ValueError("managed image count/byte budget exceeded")
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(BytesIO(raw)) as decoded:
                    width, height = decoded.size
                    if width * height > 4194304 or max(width, height) > 4096:
                        raise ValueError("managed image exceeds 4MP/4096 edge budget")
                    decoded.verify()
            translated.append(
                {"type": "input_image", "image_url": uri, "detail": image.get("detail", "auto")}
            )
        turns.append({"role": message["role"], "content": translated})
    body = {"input": turns, "truncation": "disabled"}
    if "chat_template_kwargs" in payload:
        body["chat_template_kwargs"] = payload["chat_template_kwargs"]
    if "response_format" in payload:
        fmt = payload["response_format"]
        body["text"] = {
            "format": (
                {"type": "json_schema", **fmt["json_schema"]}
                if fmt.get("type") == "json_schema"
                else fmt
            )
        }
    return body
