"""FastAPI application for the Scene Graph Explorer.

Thin transport layer around the already-working ML pipeline:

    request -> uploaded image -> SceneGraphPipeline.analyze()
            -> SceneGraphResult.to_dict() -> JSON response

Design rules:
- The ML pipeline is created ONCE at application startup (lifespan) and
  reused by every request; requests never reload YOLO / CLIP / checkpoint.
- The backend contains NO ML logic; all of it lives in scene_graph/.
- Only clean scene-graph JSON is returned; never tensors, logits,
  embeddings, YOLO results or model internals.

Run from the repository root:

    .venv\\Scripts\\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000

Swagger UI: http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import io
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

# All model weights used by the pipeline are local / cached (verified:
# CLIP ViT-B/32 safetensors live in ~/.cache/huggingface/hub).  Forcing
# offline mode avoids the per-start HTTP HEAD round-trip to huggingface.co
# (the same setting the test scripts use), which would otherwise stall
# startup on slow or blocked networks.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# Repository root on sys.path so ``scene_graph`` and the notebook module
# packages import regardless of the working directory uvicorn is started
# from.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scene_graph.pipeline import SceneGraphPipeline  # noqa: E402

# ----------------------------------------------------------------------
# Configuration (repository-relative; mirrors the Phase A/B test paths)
# ----------------------------------------------------------------------

DATASET_ROOT = REPO_ROOT / "datasets" / "visual_genome"
MODEL_DIR = REPO_ROOT / "notebook" / "models"
CHECKPOINT_PATH = MODEL_DIR / "best_full_fusion_30epoch.pt"
YOLO_MODEL_PATH = MODEL_DIR / "yolov8s-world.pt"

# Development CORS: the React dev server origin.  Extend this list later
# when the frontend is deployed; nothing else about CORS is hardened yet.
CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

# Upload validation (basic only, by file signature check after decode)
ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB safety cap

logger = logging.getLogger("scene_graph_backend")
logging.basicConfig(level=logging.INFO)


# ----------------------------------------------------------------------
# Pipeline singleton (loaded once per server process)
# ----------------------------------------------------------------------

_pipeline: SceneGraphPipeline | None = None


def get_pipeline() -> SceneGraphPipeline:
    """Return the process-wide pipeline instance.

    Created lazily on first use, but normally already built during app
    startup (lifespan).  Requests NEVER construct a pipeline themselves.
    """
    global _pipeline
    if _pipeline is None:
        logger.info("Loading SceneGraphPipeline (one-time)...")
        _pipeline = SceneGraphPipeline(
            checkpoint_path=CHECKPOINT_PATH,
            yolo_model_path=YOLO_MODEL_PATH,
            dataset_root=DATASET_ROOT,
            image_root=DATASET_ROOT,
        )
        logger.info("SceneGraphPipeline ready.")
    return _pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the ML stack once when the server starts."""
    get_pipeline()
    yield
    # Nothing to release explicitly; process teardown handles the models.


app = FastAPI(
    title="Scene Graph Explorer API",
    version="0.1.0",
    description="Image -> YOLO-World + relationship model -> scene graph JSON.",
    lifespan=lifespan,
)

# ----------------------------------------------------------------------
# CORS (development only)
# ----------------------------------------------------------------------
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------------------------------------------------
# Endpoints
# ----------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    """Liveness probe.  Does NOT run inference."""
    return {"status": "healthy"}


def _decode_upload(data: bytes, content_type: str | None):
    """Validate and decode the uploaded bytes into an RGB PIL image.

    Raises HTTPException(400) for missing/oversized/unsupported/invalid
    images.  Only basic validation - no elaborate framework.
    """
    if not data:
        raise HTTPException(status_code=400, detail="Empty upload.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="Image too large.")
    if content_type is not None and content_type.lower() not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Use JPEG, PNG or WEBP.",
        )

    from PIL import Image, UnidentifiedImageError

    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(status_code=400, detail="Invalid image data.")
    return image.convert("RGB")


@app.post("/analyze")
def analyze(file: UploadFile = File(...)) -> dict:
    """Analyze an uploaded image and return the scene graph JSON.

    The response is exactly ``SceneGraphResult.to_dict()`` - no model
    internals are exposed.
    """
    try:
        data = file.file.read()
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read upload.")

    image = _decode_upload(data, file.content_type)

    try:
        result = get_pipeline().analyze(image)
    except HTTPException:
        raise
    except Exception:
        # Server-side detail only; the client gets a clean message.
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail="Inference failed.")

    return result.to_dict()