"""ComfyUI Bernini Director — official ComfyUI Bernini timeline plugin.

Based on ComfyUI official Bernini support (PR #14216 / docs).
Thanks to Comfy-Org and bytedance/Bernini.

Licensed under the Apache License, Version 2.0. See LICENSE.
"""

from .nodes.conditioning import (
    BerniniDirectorConditioning,
    BerniniDirectorPlannerConditioning,
)
from .nodes.director import ComfyBerniniDirector

NODE_CLASS_MAPPINGS = {
    "ComfyBerniniDirector": ComfyBerniniDirector,
    "BerniniDirectorConditioning": BerniniDirectorConditioning,
    "BerniniDirectorPlannerConditioning": BerniniDirectorPlannerConditioning,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyBerniniDirector": "ComfyBerniniDirector",
    "BerniniDirectorConditioning": "Bernini Director Conditioning",
    "BerniniDirectorPlannerConditioning": "Bernini Director Planner Conditioning",
}

WEB_DIRECTORY = "./web/js"

import logging

_log = logging.getLogger("ComfyUI-Bernini-Director")

try:
    from .director.http_routes import register_routes as _register_director_routes

    if not _register_director_routes():
        _log.warning(
            "Bernini Director HTTP routes deferred (PromptServer not ready). "
            "Restart ComfyUI if /bernini/director/* returns 404."
        )
except Exception as _director_routes_exc:
    _log.warning("Bernini Director HTTP routes failed to load: %s", _director_routes_exc)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
