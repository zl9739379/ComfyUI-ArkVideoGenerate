from .ark_video_node import ArkVideoGenerate

NODE_CLASS_MAPPINGS = {
    "ArkVideoGenerate": ArkVideoGenerate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArkVideoGenerate": "Ark Video Generate (Volces)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
