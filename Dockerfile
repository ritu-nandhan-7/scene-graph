# ---------------------------------------------------------------------------
# Scene Graph Explorer - FastAPI inference backend (CPU-only, ONE worker)
#
# Deployment target: Hugging Face Docker Space (listens on port 7860) or any
# Linux CPU host. Image contents:
#   * production source (backend/, scene_graph/, notebook/models/relationship/)
#   * the 5 tracked inference artifacts (2 checkpoints + 3 VG metadata files)
#   * the two public CLIP weight files, fetched at BUILD time only
#   * requirements-deploy.txt (CPU torch, no CUDA/TensorRT/OpenVINO/ONNX)
#
# The 43.8 GB Visual Genome IMAGE dataset is never part of the build context
# (see .dockerignore) and nothing is downloaded at runtime: the backend keeps
# HF_HUB_OFFLINE=1 / TRANSFORMERS_OFFLINE=1 and every artifact is pre-baked.
# ---------------------------------------------------------------------------
FROM python:3.12-slim

# `git` is required ONLY to install the pinned `clip` git dependency below.
# It is intentionally kept in the image (purging it via apt autoremove can also
# drop ca-certificates); the runtime guarantee comes from `clip` being
# PRE-INSTALLED, so ultralytics' lazy requirement check passes and never runs
# pip. No GPU/accelerator packages are installed.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      git ca-certificates \
      libgl1 libglib2.0-0 libxcb1 \
 && rm -rf /var/lib/apt/lists/*

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/matplotlib

# ---------------------------------------------------------------- deps ------
COPY requirements-deploy.txt /tmp/requirements-deploy.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements-deploy.txt \
 && python -m pip check \
 && python -c "import clip, open_clip, ultralytics, torch, fastapi, h5py, matplotlib, PIL, numpy; print('dependency imports OK; torch', torch.__version__)"

WORKDIR /app

# --------------------------------------------- application + VG artifacts ----
COPY docker/ ./docker/
COPY backend/ ./backend/
COPY scene_graph/ ./scene_graph/
COPY notebook/__init__.py ./notebook/__init__.py
COPY notebook/models/__init__.py ./notebook/models/__init__.py
COPY notebook/models/yolov8s-world.pt ./notebook/models/yolov8s-world.pt
COPY notebook/models/best_full_fusion_30epoch.pt ./notebook/models/best_full_fusion_30epoch.pt
COPY notebook/models/relationship/ ./notebook/models/relationship/
COPY datasets/visual_genome/__init__.py ./datasets/visual_genome/__init__.py
COPY datasets/visual_genome/models.py ./datasets/visual_genome/models.py
COPY datasets/visual_genome/loader.py ./datasets/visual_genome/loader.py
COPY datasets/visual_genome/visualisations.py ./datasets/visual_genome/visualisations.py
COPY datasets/visual_genome/VG-SGG.h5 ./datasets/visual_genome/VG-SGG.h5
COPY datasets/visual_genome/VG-SGG-dicts.json ./datasets/visual_genome/VG-SGG-dicts.json
COPY datasets/visual_genome/image_data.json ./datasets/visual_genome/image_data.json

# --------------------------------------------- CLIP artifacts (build time) ---
# PATH A: ultralytics/CLIP text tower -> /opt/sg/weights/clip/ViT-B-32.pt
#         (official OpenAI file, SHA-256 verified in the fetch script)
# PATH B: open_clip visual encoder  -> $HF_HOME/hub/models--timm--
#         vit_base_patch32_clip_224.openai (prefetched with the SAME
#         create_model_and_transforms call the runtime uses)
# Then: pin ultralytics weights_dir to the ABSOLUTE path above (Phase 4
# discovered the default is CWD-relative, which is not container-safe).
ENV YOLO_CONFIG_DIR=/opt/sg/ultralytics \
    HF_HOME=/opt/sg/hf \
    SG_WEIGHTS_DIR=/opt/sg/weights
RUN python /app/docker/fetch_clip_text_weights.py \
 && python /app/docker/prefetch_open_clip.py \
 && python /app/docker/seed_ultralytics_settings.py

# ------------------------------------------------------------ runtime user ---
# Hugging Face Spaces expects a non-root user (UID 1000).
RUN useradd -m -u 1000 -s /bin/bash appuser \
 && chown -R 1000:1000 /opt/sg /app /home/appuser
USER 1000

# ---------------------------------------------------------- runtime config ---
# HOST/PORT stay environment-driven (defaults: all interfaces, HF port 7860).
# CORS_ORIGINS is deliberately NOT hardcoded here - set it per environment
# (e.g. the future Vercel origin); the backend default remains the localhost
# dev origins. Offline flags are part of the image contract.
ENV HOME=/home/appuser \
    HOST=0.0.0.0 \
    PORT=7860 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

EXPOSE 7860

# Exactly ONE uvicorn worker: the ML stack is built once in the app lifespan
# (each extra worker would duplicate ~2.5 GB RAM). No supervisor/gunicorn.
CMD ["sh", "-c", "exec uvicorn backend.main:app --host \"${HOST:-0.0.0.0}\" --port \"${PORT:-7860}\" --workers 1"]
