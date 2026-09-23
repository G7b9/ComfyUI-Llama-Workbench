"""Comfy image tensor conversion without retaining a dependency on ComfyUI internals."""

from __future__ import annotations

import base64
import io
import math
from typing import Any


def image_tensor_to_data_urls(
    image: Any,
    max_images: int = 10,
    max_edge: int = 0,
    *,
    max_pixels: int = 0,
    image_format: str = "jpeg",
) -> list[str]:
    """Encode a ComfyUI IMAGE tensor batch as JPEG or lossless PNG data URLs.

    ``IMAGE`` is normally a float tensor shaped ``[B, H, W, C]`` in the 0..1
    range. Positive edge/pixel limits downscale before encoding, which usually
    reduces VLM visual-token use. General Prompt/Chat calls retain compact JPEG
    defaults; PE-I2I opts into a bounded lossless PNG path.
    Imports are intentionally local, so text-only installations do not load
    Pillow or NumPy until an image socket is actually used.
    """

    if image is None:
        return []
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - depends on host installation
        raise RuntimeError("Image prompting requires numpy and Pillow") from exc

    try:
        batch_size = min(int(image.shape[0]), int(max_images))
    except (AttributeError, IndexError, TypeError) as exc:
        raise ValueError("IMAGE input must be a ComfyUI image batch") from exc

    encoding = str(image_format or "jpeg").strip().lower()
    if encoding not in {"jpeg", "jpg", "png"}:
        raise ValueError("image_format must be jpeg or png")
    edge_limit = max(0, int(max_edge or 0))
    pixel_limit = max(0, int(max_pixels or 0))

    encoded: list[str] = []
    for index in range(batch_size):
        values = image[index].detach().cpu().numpy() if hasattr(image[index], "detach") else image[index]
        array = np.clip(values * 255.0, 0, 255).astype(np.uint8)
        pil = Image.fromarray(array)
        if pil.mode != "RGB":
            if pil.mode == "RGBA":
                background = Image.new("RGBA", pil.size, (255, 255, 255, 255))
                background.alpha_composite(pil)
                pil = background.convert("RGB")
            else:
                pil = pil.convert("RGB")
        scales = [1.0]
        if edge_limit > 0:
            scales.append(edge_limit / max(pil.size))
        if pixel_limit > 0:
            scales.append(math.sqrt(pixel_limit / (pil.width * pil.height)))
        scale = min(scales)
        if scale < 1.0:
            target_size = (max(1, int(pil.width * scale)), max(1, int(pil.height * scale)))
            pil = pil.resize(target_size, Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        if encoding == "png":
            pil.save(buffer, format="PNG", optimize=True)
            media_type = "image/png"
        else:
            pil.save(buffer, format="JPEG", quality=90, optimize=True)
            media_type = "image/jpeg"
        encoded.append(f"data:{media_type};base64," + base64.b64encode(buffer.getvalue()).decode("ascii"))
    return encoded
