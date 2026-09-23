from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image

from lwb.media import image_tensor_to_data_urls


def _decode_data_url(value: str) -> Image.Image:
    _, encoded = value.split(",", 1)
    return Image.open(io.BytesIO(base64.b64decode(encoded)))


def test_pe_image_transport_uses_lossless_png_and_pixel_budget():
    image = np.zeros((1, 800, 2000, 3), dtype=np.float32)

    encoded = image_tensor_to_data_urls(
        image,
        max_images=1,
        max_edge=4096,
        max_pixels=1048576,
        image_format="png",
    )

    assert encoded[0].startswith("data:image/png;base64,")
    decoded = _decode_data_url(encoded[0])
    assert decoded.width * decoded.height <= 1048576
    assert max(decoded.size) <= 4096
    assert decoded.format == "PNG"


def test_image_transport_applies_edge_cap_and_keeps_jpeg_default():
    image = np.zeros((1, 8, 5000, 3), dtype=np.float32)

    encoded = image_tensor_to_data_urls(image, max_images=1, max_edge=4096)

    assert encoded[0].startswith("data:image/jpeg;base64,")
    decoded = _decode_data_url(encoded[0])
    assert max(decoded.size) <= 4096
    assert decoded.format == "JPEG"
