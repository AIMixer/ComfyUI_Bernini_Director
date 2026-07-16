"""Bernini Director orchestration (ComfyUI official Bernini path).

Based on ComfyUI official Bernini support (PR #14216 / docs).
Thanks to Comfy-Org and bytedance/Bernini.
"""

from .executor_core import execute_director_plan_core

__all__ = ["execute_director_plan_core"]
