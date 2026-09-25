"""ComfyUI node pack for HyperFlow (Video Rebirth's 8-step LoRA for MiniMax-H3).

Reference: github.com/Video-Rebirth/hyperflow (Apache-2.0 loader; the LoRA
weights are a Model Derivative of MiniMax-H3 under the MiniMax H3 Community
License, huggingface.co/videorebirth/hyperflow).

Converted .safetensors weights resolve from models/hyperflow/ and its
subdirectories, including additional registered model folders.
"""

import os, sys
_PKG = os.path.dirname(__file__)
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

import folder_paths


def _register_folder():
    bases = [folder_paths.models_dir]
    bases.extend(os.path.dirname(p) for p in folder_paths.get_folder_paths("loras"))
    for base in dict.fromkeys(bases):
        folder_paths.add_model_folder_path("hyperflow", os.path.join(base, "hyperflow"))
    paths, extensions = folder_paths.folder_names_and_paths["hyperflow"]
    folder_paths.folder_names_and_paths["hyperflow"] = (paths, set(extensions) | {".safetensors"})


_register_folder()

from hyperflow_h3.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
