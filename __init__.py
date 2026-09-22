"""ComfyUI entry point for Llama Workbench.

All node IDs and frontend extension names are deliberately prefixed with
``LlamaWorkbench`` so this package can be installed beside other llama.cpp
custom nodes.
"""

from .lwb.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
