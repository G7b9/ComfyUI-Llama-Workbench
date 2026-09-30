"""ComfyUI entry point for Llama Workbench.

All node IDs and frontend extension names are deliberately prefixed with
``LlamaWorkbench`` so this package can be installed beside other llama.cpp
custom nodes.
"""

from .lwb.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

# Outside ComfyUI (unit tests/library imports) no prompt worker exists.
try:
    import server as _comfy_server
except ImportError:
    _comfy_server = None
if _comfy_server is not None and hasattr(_comfy_server, "PromptServer"):
    from .lwb.scheduler import install
    install()
