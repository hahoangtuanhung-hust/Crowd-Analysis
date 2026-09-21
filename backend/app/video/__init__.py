"""Video package exports without eagerly loading detector-dependent pipeline code."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "FrameRenderer",
    "OpenCVVideoSource",
    "OverlayOptions",
    "PipelineStats",
    "TrackingPipeline",
    "VideoMetadata",
]

_EXPORT_MODULES = {
    "PipelineStats": ".pipeline",
    "TrackingPipeline": ".pipeline",
    "FrameRenderer": ".renderer",
    "OverlayOptions": ".renderer",
    "OpenCVVideoSource": ".source",
    "VideoMetadata": ".source",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(name)
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value
