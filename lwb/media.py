"""Comfy image tensor conversion without retaining a dependency on ComfyUI internals."""

from __future__ import annotations

import base64
import io
from typing import Any


def image_tensor_to_data_urls(image: Any, max_images: int = 10, max_edge: int = 0) -> list[str]:
    """Encode a ComfyUI IMAGE tensor batch as compact JPEG data URLs.

    ``IMAGE`` is normally a float tensor shaped ``[B, H, W, C]`` in the 0..1
    range. ``max_edge=0`` preserves input resolution; a positive value
    downscales before encoding, which usually reduces VLM visual-token use.
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
        largest = max(pil.size)
        if max_edge > 0 and largest > max_edge:
            scale = max_edge / largest
            pil = pil.resize((round(pil.width * scale), round(pil.height * scale)), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        pil.save(buffer, format="JPEG", quality=90, optimize=True)
        encoded.append("data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii"))
    return encoded
