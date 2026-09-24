"""Scene Graph Explorer backend (FastAPI transport layer).

This package contains ONLY HTTP orchestration:

    HTTP request -> uploaded image -> SceneGraphPipeline -> JSON response

All ML logic (detection, features, relationship model, post-processing)
lives in the ``scene_graph`` package; the backend must never duplicate it.
"""

from .main import app

__all__ = ["app"]