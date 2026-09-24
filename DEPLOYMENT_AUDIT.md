# DEPLOYMENT AUDIT - Scene Graph Explorer

**Audit-only report. Nothing was deployed, moved, deleted, or modified.**
Architecture, ML pipeline, API contract, requirements.txt and React code are untouched.
Date: 2026-09-24. Measured on: Windows, Python 3.11 (.venv), Node v22.14.0 / npm 11.5.2.
All numbers were measured on the live filesystem/processes during this audit unless
attributed to a prior documented run (post_training_full_stack.md / tests/_profile_*.txt).

---

## 1. Current Architecture

Two-process local application:

| Process | Command (current) | Port | Role |
|---|---|---|---|
| FastAPI backend | `.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000` | 8000 | ML inference: `GET /health`, `POST /analyze` |
| React frontend | `npm run dev` (Vite) | 5173 | UI (dev only; production output = static `frontend/dist`) |

- `backend/main.py`: lifespan loads `SceneGraphPipeline` ONCE per process; every request
  reuses it. Uploads decoded in memory (BytesIO), 20 MB cap, JPEG/PNG/WEBP allowlist,
  no temp files, no database, no authentication.
- `scene_graph/pipeline.py`: YOLO-World detect -> ordered pairs -> batched open_clip ViT-B/32
  union-crop features (512-D) + 8-D geometry -> `CachedRelationshipModel` (51 classes)
  -> post-processing -> clean JSON `{image, objects, relationships}`.
- Frontend: React 18 + @xyflow/react + @dagrejs/dagre + html-to-image; exactly one API
  call (`POST /analyze`) from `frontend/src/services/api.js`.

Measured repository layout:

| Area | Size | Contents |
|---|---|---|
| `datasets/` | 43,845 MB | VG_100K 9,352 MB (64,346 files), VG_100K_2 5,265 MB (43,903 files), zips 14,504 MB, VG-SGG.h5 67.6 MB, image_data.json 16.8 MB, VG-SGG-dicts.json ~10 KB, VG150_curated.zip 22.5 MB |
| `notebook/` | 1,753 MB | preprocessed_cache 1,708 MB (87 shards), yolov8s-world.pt 25.91 MB, best_full_fusion_30epoch.pt 2.36 MB, 5 .ipynb ~17 MB |
| `.venv/` | 1,458 MB | site-packages 1,448 MB; torch 494 MB; NO nvidia packages (CPU build) |
| `frontend/` | 72.7 MB | src (10 files), dist 434 KB (3 files), node_modules, verifier scripts |
| `backend/` | ~0.02 MB | 2 source files |
| `scene_graph/` | ~0.1 MB | 7 source files |
| `tests/` / `scripts/` | ~0.1 MB | 13 + 3 files (tests include scratch outputs) |
| **Total excl. .venv/node_modules/.git** | **45,608 MB** | dataset images/zips dominate |

External caches used at runtime (outside the repository):

| Cache | Size | Content |
|---|---|---|
| `~/.cache/huggingface/hub` | 577 MB | `models--timm--vit_base_patch32_clip_224.openai` -> `open_clip_model.safetensors` (open_clip visual encoder) |
| `C:\Users\psytr\weights\clip` | 337.6 MB | `ViT-B-32.pt` (ultralytics/CLIP fork text tower, per ultralytics settings `weights_dir`) |
| `~/.cache/ultralytics`, `~/.cache/clip`, `~/.cache/torch` | absent | not used |

Git: history exists (5+ commits, all titled "bug fixes"), remote
`origin https://github.com/ritu-nandhan-7/scene-graph.git`. Tracked today: docs,
notebooks, VG metadata (h5/json), VG150 zip. **All application code (backend/,
scene_graph/, frontend/src, scripts/, tests/, post_training_full_stack.md) is
currently UNTRACKED** (51 files) plus `requirements.txt` modified - see section 11.

---

## 2. Runtime Dependency Graph (exactly what `POST /analyze` needs)

```
backend/main.py
  sets HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1   (before importing the pipeline)
  REPO_ROOT = Path(main.py).parents[1] -> sys.path
  |
  +-> scene_graph/pipeline.py  -> SceneGraphPipeline(checkpoint, yolo, dataset_root)
        |
        +-> datasets/visual_genome VisualGenomeLoader.__init__  [ALL 3 files opened at STARTUP]
        |     +-> VG-SGG.h5         h5py.File        [REQUIRED TO START; never read per-request]
        |     +-> VG-SGG-dicts.json json.load        [REQUIRED AND USED: 150 labels + 50 predicates;
        |     |                                        YOLO vocabulary + predicate decode tables]
        |     +-> image_data.json   json.load        [REQUIRED TO START; never read per-request]
        |     +-> visualisations.py -> import matplotlib   [IMPORT-TIME runtime dependency!]
        |     (VG_100K / VG_100K_2 images are read ONLY by loader.load_scene(), which the
        |      pipeline never calls - analyze() uses the uploaded image exclusively)
        |
        +-> notebook/models/relationship/encoder.py
        |     open_clip.create_model_and_transforms("ViT-B-32", "openai")
        |     loads from HF cache: .../open_clip_model.safetensors (577 MB)  [confirmed in uvicorn log]
        +-> notebook/models/relationship/model.py -> torch only (CachedRelationshipModel)
        +-> scene_graph/detector.py
        |     ultralytics YOLOWorld(notebook/models/yolov8s-world.pt)   [25.91 MB local file]
        |     .set_classes(149 VG names) -> ultralytics.nn.text_model.CLIP
        |        -> clip.load("ViT-B/32", download_root=WEIGHTS_DIR/"clip")
        |        = C:\Users\psytr\weights\clip\ViT-B-32.pt (337.6 MB)
        |          (downloaded ONCE from openaipublic.azureedge.net on first-ever run)
        +-> scene_graph/{features,mapping,postprocessing,predictor}.py  [pure code]
        +-> torch.load(notebook/models/best_full_fusion_30epoch.pt)     [2.36 MB]
```

Everything above was verified by: source tracing from `backend/main.py` and
`scene_graph/pipeline.py`, the live uvicorn startup log (which names the exact
safetensors path it opened), and filesystem measurement.

### Per-artifact table

| FILE | SIZE | REQUIRED AT RUNTIME? | WHY | DL DURING DEPLOY? | IN GIT? | RECOMMENDED STORAGE |
|---|---|---|---|---|---|---|
| `notebook/models/yolov8s-world.pt` | 25.91 MB | **YES** | `YOLOWorld(model_path)` direct load | possible (Ultralytics asset) but adds 3rd-party URL dependency | **ignored** by `*.pt` rule | repo per-file exception, or release artifact / image layer |
| `notebook/models/best_full_fusion_30epoch.pt` | 2.36 MB | **YES** | `torch.load` relationship checkpoint | **NO** - trained here, must be transported | **ignored** by `*.pt` rule | **must ship**: repo exception (small) or artifact store |
| `C:\Users\psytr\weights\clip\ViT-B-32.pt` | 337.6 MB | **YES (startup only)** | CLIP text embeddings for `set_classes(149)` | YES - auto from `openaipublic.azureedge.net` on first run | outside repo | allow first-run download **or** bake into image |
| `~/.cache/huggingface/hub/.../open_clip_model.safetensors` | 577 MB | **YES** | open_clip ViT-B/32 "openai" visual encoder | YES from huggingface.co **unless** `HF_HUB_OFFLINE=1` stays set (it does) | no (cache) | bake into image / pre-seed cache; or relax offline flag for first boot |
| `datasets/visual_genome/VG-SGG-dicts.json` | ~10 KB | **YES** | vocabulary + predicate index maps | ships with repo | yes (tracked) | keep in repo |
| `datasets/visual_genome/VG-SGG.h5` | 67.6 MB | **YES to start** (unused per-request) | unconditional `h5py.File` in loader `__init__` | ships with repo | yes (tracked) | keep in repo (slimming the loader = later code change) |
| `datasets/visual_genome/image_data.json` | 16.8 MB | **YES to start** (unused per-request) | unconditional `json.load` in loader `__init__` | ships with repo | yes (tracked) | keep in repo |
| VG images `VG_100K`, `VG_100K_2` | 14.6 GB | **NO** | only `load_scene()` reads them; inference never does | N/A | **never** (ignored) | local only; keep 1 sample for demos |
| `VG_100K*.zip`, `VG150_curated.zip` | 14.5 GB / 22.5 MB | **NO** | source archives | N/A | zips ignored (VG150 already tracked) | local only |
| `notebook/models/preprocessed_cache/` | 1,708 MB | **NO** | training feature cache | N/A | ignored | local only |
| HuggingFace / ultralytics CLIP caches | 914.6 MB total | **YES** | two separate CLIP copies (text tower + visual tower) | yes / pre-seed | never (outside repo) | image bake or cache volume |

Notes:
- There are **two distinct CLIP weight copies**: the ultralytics/CLIP fork text tower
  (337.6 MB, `weights_dir/clip`) used only during startup `set_classes`, and the
  open_clip visual encoder (577 MB HF safetensors) used per request for pair features.
- Ultralytics settings live in `%APPDATA%\Ultralytics\settings.json` and contain
  **empty** `api_key` / `openai_api_key` fields (no secrets; machine-local config).

---

## 3. Training-Only vs Inference-Time Files

**TRAINING-ONLY - not needed to deploy:**
- `datasets/visual_genome/VG_100K/` (9,352 MB), `VG_100K_2/` (5,265 MB), both zips (14,504 MB)
- `notebook/models/preprocessed_cache/` (1,708 MB, 87 shards)
- `datasets/original/VG150_curated.zip` (22.5 MB), `zeroshot_triplet.pytorch` (~0.07 MB)
- `notebook/*.ipynb` (5 notebooks, ~17 MB)
- training-side modules `notebook/models/relationship/{dataset,trainer,transforms,predictor}.py`
  (`encoder.py`, `model.py`, `geometry.py`, `__init__.py` ARE runtime imports)
- `tests/` scratch outputs (`_profile_*.txt`, `_scene.json`, `_sweep2.txt`, `_fetch_scene.py`,
  `_profile_pipeline.py`), `frontend/verify_*.mjs` verification harnesses

**INFERENCE-TIME - must deploy:**
- Code: `backend/`, `scene_graph/`, `datasets/visual_genome/*.py` (13.2 KB total),
  `notebook/__init__.py`, `notebook/models/__init__.py`,
  `notebook/models/relationship/{__init__,encoder,model,geometry}.py`
- Metadata: `VG-SGG.h5` + `VG-SGG-dicts.json` + `image_data.json` (~85 MB)
- Weights: `yolov8s-world.pt` (25.91 MB), `best_full_fusion_30epoch.pt` (2.36 MB),
  ultralytics `ViT-B-32.pt` (337.6 MB), HF open_clip safetensors (577 MB)
- Frontend: `frontend/dist/` build output (434 KB)

**DEMO-ONLY (optional):** one VG sample image (e.g. `VG_100K_2/1.jpg`) for manual E2E tests.

---

## 4. Model Artifacts

| Model | Exact path | Size | Load behavior |
|---|---|---|---|
| YOLO-World | `notebook/models/yolov8s-world.pt` | 25.91 MB | **A** - local file; re-headed via `set_classes` (open-vocabulary) |
| CLIP text tower (startup) | `C:\Users\psytr\weights\clip\ViT-B-32.pt` | 337.6 MB | **D->E then A** - ultralytics/CLIP fork downloads once from OpenAI's CDN into ultralytics `weights_dir/clip`, then loads locally; location governed by `%APPDATA%\Ultralytics\settings.json` (outside repo) |
| CLIP visual encoder | `~/.cache/huggingface/hub/models--timm--vit_base_patch32_clip_224.openai/.../open_clip_model.safetensors` | 577 MB | **C** - HuggingFace cache via `open_clip(pretrained="openai")`; backend forces offline mode, so a fresh host fails unless the cache is pre-seeded or offline flags are relaxed |
| Relationship checkpoint | `notebook/models/best_full_fusion_30epoch.pt` | 2.36 MB | **A** - `torch.load(map_location, weights_only=False)` + `load_state_dict` |

No other model artifacts exist or are referenced anywhere in code. The
`preprocessed_cache/` shards are training data, not models. `zeroshot_triplet.pytorch`
is referenced only by roadmap documentation.

---
## 5. Dataset Requirements

Training data is NOT assumed to be inference data - traced in code:

| Dataset item | Needed for `POST /analyze`? | Evidence |
|---|---|---|
| `VG_100K` / `VG_100K_2` images | **NO** | only `loader.load_scene()` opens image files; `pipeline.analyze()` analyzes the uploaded image; no other reader exists |
| `VG-SGG.h5` | **start-up YES / per-request NO** | `h5py.File(...)` unconditional in `VisualGenomeLoader.__init__`; zero `self.h5[...]` accesses in the inference path |
| `VG-SGG-dicts.json` | **YES** | `build_label_name_to_vg_id`, `build_idx_to_predicate`, `build_yolo_vocabulary` consume it |
| `image_data.json` | **start-up YES / per-request NO** | `json.load` unconditional in `__init__`; only `_get_image_info()` (i.e. `load_scene()`) reads it |
| Image zips | **NO** | archives of the image dirs |
| `zeroshot_triplet.pytorch` | **NO** | no code imports it |

Bottom line: inference needs **~85 MB of Visual Genome metadata**, not the 43.8 GB dataset.
The h5/json presence is a code-shape constraint of the loader constructor, not a data
dependency of the inference math. Demo scripts use `VG_100K_2/1.jpg` only as a sample input.

---

## 6. Python Dependency Audit (REPORT ONLY - requirements.txt NOT modified)

Facts measured:
- **82 pinned lines**, file encoded **UTF-16LE with BOM** (`FF FE`) - typical of PowerShell
  `>` redirection. pip 26.2.1 CAN parse it (`req_file.py` BOM table includes UTF-16), but
  other tooling (poetry, some CI linters, non-Windows editors) may not.
- **1 git dependency**: `clip @ git+https://github.com/ultralytics/CLIP.git@c4b6ea09...`
  (commit-pinned, good) - requires `git` present in any build image.
- **torch==2.13.0** with NO CPU index pin. Local venv reports `torch 2.13.0+cpu`,
  `cuda: False`, 8 threads, no `nvidia/*` packages installed. On Linux, plain PyPI
  resolution of `torch==2.13.0` pulls the CUDA-variant wheel set (multi-GB) unless
  `--index-url https://download.pytorch.org/whl/cpu` (or a local +cpu pin) is used.
  **CPU-only pinning is therefore NOT correctly encoded for deployment.**
- **`python-multipart==0.0.20` present** - FastAPI `File(...)` uploads will work.

Categories (by import tracing of production code vs notebook/training code): PRODUCTION-DIRECT (imported at runtime by backend path):
`fastapi`, `uvicorn`, `python-multipart`, `torch(+cpu)`, `torchvision` (ultralytics dep),
`ultralytics`, `open_clip_torch`, `clip` (git), `h5py`, `pillow`, `numpy`, `matplotlib`
(imported by `datasets/visual_genome/visualisations.py` at package import!),
`opencv-python` (ultralytics), `PyYAML`, `requests`, `tqdm`, `huggingface_hub`, `safetensors`,
`filelock`, `regex`, `ftfy`, `fsspec`, `httpx/httpcore/h11` (transitive), `psutil`, `scipy` (transitive of ultralytics/torch stacks).

TRAINING / NOTEBOOK / DEV-ONLY (not needed to serve):
`ipykernel`, `ipython`, `ipython_pygments_lexers`, `jedi`, `parso`, `prompt_toolkit`,
`stack-data`, `executing`, `pure_eval`, `debugpy`, `comm`, `matplotlib-inline`,
`jupyter_client`, `jupyter_core`, `traitlets`, `pyzmq`, `tornado`, `wcwidth`,
`pandas`, `polars`, `polars-runtime-32`, `contourpy`, `cycler`, `fonttools`,
`kiwisolver`, `pyparsing`, `python-dateutil`, `six`, `pyDeprecate`, `nest-asyncio2`,
`sympy`, `mpmath`, `tzdata`, `nvidia-ml-py` (profiling), `supervision` (**imported NOWHERE**).

Problematic / noteworthy for cloud deployment (findings only):
1. git+https dep needs `git` in the build environment.
2. torch CPU wheel index not pinned -> accidental multi-GB CUDA install on Linux.
3. ~30 notebook/Jupyter packages inflate image size for no runtime benefit.
4. `opencv-python` (GUI-capable) instead of `opencv-python-headless` (server convention).
5. Ultralytics and the CLIP fork are **AGPL-3.0** licensed - distribution/service
   obligations must be reviewed; the repo itself has **no LICENSE file**.
6. UTF-16 encoding: works with pip, but fragile across tools (recommend UTF-8 later).
7. `datasets` package name shadows the popular HuggingFace `datasets` library - if it
   is ever installed, imports in this repo would resolve to the wrong package.

---

## 7. Frontend Dependency Audit

- Build command: `cd frontend; npx vite build` (or `node node_modules/vite/bin/vite.js build`).
  **Known local quirk:** `npm run build` exits 1 with no output on this machine
  (npm 11.5.2 / Node 22) - documented in Entry 10; CI must use the direct invocation.
- Output directory: `frontend/dist` - measured **3 files, 434 KB** (index.html, hashed JS
  ~408 kB, hashed CSS ~27 kB). Default Vite output; not configured explicitly.
- Runtime API URL: **hardcoded** `const API_BASE_URL = 'http://127.0.0.1:8000'` in
  `frontend/src/services/api.js:1`. No `import.meta.env` / `VITE_*` usage anywhere.
  No Vite dev proxy. This is the single source of backend addressing.
- Hardcoded localhost references: only `api.js:1` in application source (plus docs,
  run scripts and test scripts - dev tooling).
- CORS assumptions: frontend never configures CORS; it relies entirely on the backend's
  allow-list (`http://localhost:5173`, `http://127.0.0.1:5173`).
- Environment-variable requirements: **none** today (nothing is read at build or runtime).
- Dependencies (runtime): `@dagrejs/dagre`, `@xyflow/react`, `html-to-image`, `react`,
  `react-dom`. Dev: `vite ^6.0.7`, `@vitejs/plugin-react`.
- `index.html`: standard Vite entry, `/vite.svg` favicon, title "Scene Graph Explorer".

## 8. Backend Audit

- Production startup command today: `.venv\Scripts\python.exe -m uvicorn backend.main:app
  --host 127.0.0.1 --port 8000` (or `scripts/run_backend.ps1`). Host/port come **only
  from CLI flags**; nothing reads env vars for bind configuration.
- CORS: hardcoded dev allow-list in `backend/main.py` (`http://localhost:5173`,
  `http://127.0.0.1:5173`), `allow_credentials=True`, all methods/headers. A deployed
  frontend origin will be rejected until this list changes.
- Model loading: load-once via FastAPI `lifespan` -> `get_pipeline()` (module singleton
  with lazy fallback). Verified live: "Loading SceneGraphPipeline (one-time)..." once
  per process, requests reuse it. **Load-once survives production** with two caveats:
  (1) every worker process pays the full RAM+startup cost - run exactly 1 model worker
  (scale by adding machines, not workers); (2) the process must not be recycled per request.
- Health: `GET /health` -> `{"status":"healthy"}`, does NOT run inference. Because the
  lifespan blocks until models are loaded, a listening server is inherently "ready".
- Analyze: `POST /analyze` sync def (FastAPI runs it in the threadpool - the event loop
  stays responsive), full body read into memory, 20 MB cap, content-type allowlist,
  PIL decode validation, generic 500 on inference failure (details server-side only).
- Temp files: **none** - upload is decoded from BytesIO; nothing is written to disk.
- Environment variables: only `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` (setdefault,
  before pipeline import). No secrets, no config env vars.
- Path assumptions: **repository-relative** via `REPO_ROOT = Path(__file__).parents[1]`
  and explicit `sys.path` insertion. Requires deploying with the same layout
  (`backend/`, `scene_graph/`, `datasets/visual_genome/`, `notebook/models/`) under one
  root. No absolute machine paths in backend code. (Absolute dev path exists only in
  `tests/test_connection.py`, a dev script.)

---

## 9. Cold-Start Analysis

Measured and documented values:

| Scenario | Time |
|---|---|
| **This audit: fresh process -> "Application startup complete"** (page cache warm, machine idle) | **45.2 s** (start 11:40:55 -> log end 11:41:40) |
| Pipeline construction alone (tests/_profile_final.txt) | 27.47 s |
| Documented ranges (post_training entries / scripts README) | 20-37 s best case; 50-120 s; 60-90 s under load; "1-2 minutes" in run scripts |
| Warm `POST /analyze` (this audit, live server, 800x600 / 9 obj / 68 pairs) | **8.32 s** |
| Warm reference (profile, same image) | 7.01 s / 7.50 s second request |
| Documented warm range | ~5.5-9 s |

Cold-start cost drivers (in order): Python + torch + ultralytics imports; VisualGenomeLoader
(h5 open + two JSON parses); YOLO-World load + `set_classes` (loads the 337.6 MB CLIP text
tower); open_clip construction (577 MB safetensors read from disk); checkpoint load.
HF offline flags remove network round-trips; all remaining cost is local disk + CPU.

Warm-request cost drivers: batched CLIP forward over all union crops dominates
(~6.7-7 s of the 8.3 s; batch-size 8 is FINAL per project decision - no further
optimization in scope); YOLO detect ~0.3-1.1 s; relationship model ~3 ms batched.

Unavoidable costs: model file loads (cold), CLIP forward per request (warm),
20 MB image decode. Deployment-specific costs: image pull / dependency install,
weight & cache provisioning, platform health-check gating, TLS/proxy hop.

---

## 10. RAM / CPU / Disk Requirements (measured)

Live warm backend (pre-existing dev server, PID 14132, models loaded, serving traffic):

| Metric | Value |
|---|---|
| Working Set | 455 MB (much of the weight memory is file-backed / paged out) |
| **Private memory (committed)** | **2,536 MB** - the number that matters for cgroup/container limits |
| Peak Working Set | 1,847 MB |
| Peak paged memory | ~3,538 MB (upper bound seen) |
| Threads / handles | 33 / 625 |
| Torch threads | 8 (CPU) |

- **RAM**: budget **>= 4 GB** for one worker (2.5 GB committed + OS + page-cache headroom;
  2 GB is borderline/risky, 1 GB will OOM). Each extra uvicorn worker adds ~2.5 GB - keep 1.
- **CPU**: CPU-only inference is proven (entire session: `cuda: False`). Realistic on any
  modern 2-4 vCPU tier; request latency scales inversely with cores (CLIP batch forward).
- **Disk (fresh deployment estimate)**:
  code < 5 MB + VG metadata 84.5 MB + repo weights 28.3 MB + CLIP caches 914.6 MB
  + Python env ~1.2-1.5 GB (CPU torch) + frontend dist 0.4 MB  =>  **~2.5-3 GB** total.
  (Deploying the current dev venv verbatim would be 1.46 GB for site-packages alone.)
- **CPU-only free/low-cost tiers**: realistic on 2 GB+ plans ONLY if RAM budget above is
  respected; free 512 MB-1 GB serverless/container tiers are NOT realistic (commit alone
  is 2.5 GB). Cold start 45-120 s exceeds many free-tier health-check windows.
- **Persistent storage**: no runtime writes - app is stateless (no DB, no uploads kept).
  Only read-only weight/cache provisioning is needed (bake into image or mount volume).

---

## 11. Git Hygiene

Current tracked set: docs (`README.md`, roadmap, workflow), 5 notebooks, VG metadata
(h5/json/dicts), `VG150_curated.zip`, dataset Python SDK, relationship model modules,
`requirements.txt` (modified). **Untracked: all 51 application files** (backend,
scene_graph, frontend/src + configs + verifiers, scripts, tests, post_training log,
`notebook/__init__.py`, `notebook/models/__init__.py`).

`git check-ignore` verified coverage:

| Item | Ignored? | By |
|---|---|---|
| `node_modules/` | YES | frontend/.gitignore |
| `frontend/dist` | YES | frontend/.gitignore `dist/` |
| `__pycache__/`, `*.pyc` | YES | root .gitignore |
| `.venv/` | YES | root .gitignore |
| HuggingFace / ultralytics / clip caches | N/A (live outside repo - never committed) | location |
| `frontend/_browser_shots/` + scratch logs | YES | frontend/.gitignore |
| `*.pt` (BOTH model checkpoints) | YES - note: also blocks committing them | root .gitignore |
| `VG_100K/`, `VG_100K_2/`, their zips, `visual_genome_images/` | YES | root .gitignore |
| `notebook/models/preprocessed_cache/` | YES | root .gitignore |
| `*.log` | YES | root .gitignore |
| **`tests/_profile_*.txt`, `_scene.json`, `_sweep2.txt`, `_fetch_scene.py`, `_profile_pipeline.py`** | **NO - would be committed** | gap |
| `.env*` files | NO rule (none exist today) | gap |
| `zeroshot_triplet.pytorch` (already tracked) | NO rule | benign (0.07 MB) |

If `git add -A` ran now: all application source + tests scratch files would be staged;
both `.pt` checkpoints would silently stay out. No secrets, datasets, node_modules,
dist, screenshots or caches would be committed. Recommended (NOT done in this phase):
ignore the tests scratch pattern, decide an explicit weights policy (per-file
`!` exceptions for the two checkpoints, or an artifact store), add a `.env*` rule.
---

## 12. Deployment Architecture Options (no winner chosen)

### Option A - Single always-on server (VM or one container)
| Aspect | Detail |
|---|---|
| Frontend hosting | FastAPI serves `frontend/dist` as static files (one origin) - or nginx in front |
| Backend hosting | uvicorn (1 worker) under systemd / Docker on a small VPS or always-on container |
| Model storage | weights + both CLIP caches baked into image / VM disk (~950 MB) |
| Database | none |
| GPU | none (CPU-only proven) |
| CPU / RAM | 2-4 vCPU, **4 GB RAM min** |
| Cold start | only on boot/redeploy: 45-120 s, invisible to users afterward |
| Operational complexity | low: one box, one deploy pipeline, SSH/systemd or compose |
| Cost category | small VPS / entry VM: roughly $5-20/mo class |
| Advantages | matches load-once design perfectly; no CORS anywhere; simplest mental model; predictable 6-9 s inference |
| Disadvantages | single point of failure; manual TLS/renewal; vertical scaling only; you manage updates |

### Option B - Split: static frontend CDN + backend PaaS/container platform
| Aspect | Detail |
|---|---|
| Frontend hosting | static `dist/` on GitHub Pages / Netlify / Vercel / CloudFront |
| Backend hosting | Render / Fly.io / Railway / Cloud Run container with start cmd `uvicorn backend.main:app --host 0.0.0.0 --port $PORT` |
| Model storage | image layers (~1 GB) or volume for HF + ultralytics caches |
| Database | none |
| GPU | none |
| CPU / RAM | 2-4 vCPU, 4 GB+ RAM plan |
| Cold start | platform-dependent: cold repo boots 45-120 s (needs start-time budget / warm instances); scale-to-zero hosts re-pay it every idle period |
| Operational complexity | medium: two deploy targets, env/config per side, CORS management |
| Cost category | free tiers exist but are hostile to this cold start; realistic ~$7-25/mo for comfortable tiers |
| Advantages | CDN-fast frontend, independent deploys, platform-managed TLS/rollbacks |
| Disadvantages | requires the API-URL + CORS config changes (currently blockers); cold starts punish scale-to-zero; two pipelines to maintain |

### Option C - Serverless (Lambda / Cloud Functions + static frontend)
| Aspect | Detail |
|---|---|
| Frontend hosting | static CDN (as B) |
| Backend hosting | AWS Lambda / GCP Cloud Run scale-to-zero with strict defaults |
| Model storage | package layers hit size limits (~950 MB of weights+caches + torch runtime) |
| Database | none |
| GPU | none |
| CPU / RAM | memory settings up to 4-8 GB class; init must fit timeout windows |
| Cold start | **hostile**: 45-120 s init vs typical 10-60 s limits; needs provisioned/warm instances (cost) or snapstart-style mitigations |
| Operational complexity | high: artifact packaging, warm-pool management, 8 s sync invocations vs 6-9 s CPU inference leave no margin |
| Cost category | Seemingly cheap per call, but provisioned concurrency to be usable is NOT cheap |
| Advantages | zero idle cost in theory, managed scaling |
| Disadvantages | fundamental mismatch with load-once + heavy-CPU design; highest complexity; rework pressure on architecture (out of scope by rule) |

---

## 13. Single-Server Possibility (React + FastAPI + YOLO + CLIP + relationship model)

**Yes - technically fully possible and arguably the natural fit for this codebase.**

- How: build `frontend/dist`, then mount it in FastAPI via `StaticFiles` (e.g. `/`) plus a
  catch-all to `index.html`; the API keeps its `/analyze` and `/health` routes. FastAPI
  serving static files is a first-class pattern (no new architecture; no ML changes).
- Effect on deployment simplicity: **major win** - one process, one port, one deploy
  artifact, one origin, one health check. The current Windows run scripts already
  demonstrate the two commands; single-server merges serving, not logic.
- Architecture clarity: pros - one repo == one service, trivial local/production parity;
  cons - static-serving and API concerns share a process (minor; standard practice).
- Scaling: vertical only (add RAM/CPU); horizontal = replicas behind a balancer, each
  replica paying 2.5 GB + 45 s boot. Same constraint as Option A anyway.
- Cold start: unchanged (models still load once per process) - but only one service has
  a cold start at all; the frontend no longer depends on a separate CDN deploy order.
- Portfolio presentation: clean "I shipped a complete ML app as one deployable unit"
  story; CORS discussion disappears entirely.
- Required enabler (later implementation phase): frontend must use a **relative** API
  base (e.g. `import.meta.env.VITE_API_BASE ?? ''` -> `fetch('/analyze')`), because the
  hardcoded `http://127.0.0.1:8000` cannot point at a remote origin. This is a config
  change, not an architecture change.

## 14. Separate Frontend + Backend Analysis

- Do NOT assume separate hosting is better. It is only better if you specifically want
  CDN edge caching for the ~434 KB bundle, independent deploy cadences for UI vs model,
  or platform-managed frontend previews.
- Costs of separation: CORS allow-list maintenance (currently dev-only), a second deploy
  pipeline, environment-specific API URLs (build-time env var), cross-origin latency,
  and dual rollback/debug surface.
- With a ~434 KB static bundle and a single heavy API, the classic reasons for splitting
  are weak here; separation mainly buys frontend hosting convenience.
- Either choice requires the same small config work (API base URL + CORS or same-origin).

---

## 15. Proposed Production File Set (report only - nothing moved)

```
deployment/                     (logical grouping; files stay where they are today)
├── backend-code/               ~0.2 MB
│   ├── backend/                main.py, __init__.py
│   ├── scene_graph/            7 modules
│   ├── datasets/visual_genome/ *.py only (13.2 KB)
│   ├── notebook/__init__.py
│   ├── notebook/models/__init__.py
│   └── notebook/models/relationship/  __init__, encoder, model, geometry
├── runtime-data/               ~85 MB
│   ├── VG-SGG.h5
│   ├── VG-SGG-dicts.json
│   └── image_data.json
├── model-artifacts/            ~943 MB (repo part) + 914.6 MB caches
│   ├── yolov8s-world.pt                  25.91 MB
│   ├── best_full_fusion_30epoch.pt        2.36 MB
│   ├── clip/ViT-B-32.pt                337.6 MB   (from weights_dir)
│   └── hf-cache/ (open_clip safetensors) 577 MB
├── frontend-build/
│   └── dist/                   434 KB (3 files)
├── requirements.txt            (to be split/normalized in a later phase)
└── run: uvicorn backend.main:app --host 0.0.0.0 --port $PORT   (1 worker)

EXCLUDED from deployment:
  VG_100K/, VG_100K_2/, *.zip, preprocessed_cache/, notebooks,
  datasets/original/, tests scratch, frontend verifiers (optional),
  .venv/, node_modules/, _browser_shots/, __pycache__/, .git/
```
Total deployment footprint: **~2.5-3 GB** including a fresh CPU Python environment.
---

## 16. Security / Configuration Findings

| FILE | LINE / LOCATION | TYPE | VALUE |
|---|---|---|---|
| (none) | - | **hardcoded secrets** | **No API keys, passwords, tokens, private keys or credentials found in any source file** |
| `frontend/src/services/api.js` | 1 | hardcoded localhost URL | `http://127.0.0.1:8000` [redacted-as-config] |
| `backend/main.py` | 61-64 | dev-only CORS allow-list | `localhost:5173` origins |
| `tests/test_connection.py` | 40 | absolute local machine path | `C:\Users\...\VG_100K_2\1.jpg` [redacted] |
| `scripts/run_backend.ps1`, `run_frontend.ps1`, `tests/*.py`, `frontend/verify_*.mjs` | various | localhost dev assumptions | 127.0.0.1 ports 8000/5173 |
| `%APPDATA%\Ultralytics\settings.json` (outside repo) | api_key / openai_api_key | credential slots | **empty strings** (no secret present) |
| `tests/_profile_final.txt`, `_sweep2.txt` | HF warning lines | mentions `HF_TOKEN` env name only | no token value |
| `.env` files | - | none exist anywhere in the repo | - |

Additional notes: `allow_credentials=True` broader than needed; public deployments should
consider disabling `/docs` and adding rate limiting later (optional hardening, no code
change in this phase). Licensing: ultralytics + ultralytics/CLIP are AGPL-3.0 and the
repository ships **no LICENSE file** - resolve before public distribution.

---

## 17. Deployment Blockers

**CRITICAL BLOCKERS**
1. Hardcoded API base URL (`frontend/src/services/api.js:1`) - a deployed frontend would
   call the visitor's own `127.0.0.1:8000`. Blocks ANY remote deployment (A, B, or C).
2. Dev-only CORS allow-list (`backend/main.py`) - deployed origins rejected by browsers.
3. No artifact-transport story: both `.pt` checkpoints are git-ignored; the 914.6 MB of
   CLIP caches live OUTSIDE the repo; `HF_HUB_OFFLINE=1` makes a fresh host fail at
   startup unless caches are pre-seeded or the flag is relaxed in a controlled way.
4. `requirements.txt` not deployment-ready: git+https dep (needs git), no CPU wheel index
   (Linux default pulls CUDA wheels), UTF-16 encoding, training stack included.
   (Report-only: file was NOT modified.)
5. All application code is **untracked** - the GitHub remote does not contain the app;
   there is nothing CI/CD could deploy today.
6. Load-once cold start 45-120 s + 6-9 s synchronous inference - incompatible with
   default serverless/scale-to-zero windows; needs an always-on or warm-instance host.

**HIGH PRIORITY**
1. Host/port/CORS not environment-configurable (config lives in code + CLI flags only).
2. No production process definition for Linux (workers=1 must be enforced; each extra
   worker costs ~2.5 GB RAM).
3. Tests scratch files (`tests/_*.txt`, `_scene.json`, ...) are not git-ignored.
4. No LICENSE file vs AGPL dependencies.
5. No rate limiting/auth on `/analyze` before going public (CPU cost abuse vector).
6. `npm run build` quirk on the dev machine - CI must call vite directly.

**MEDIUM PRIORITY**
1. `VG-SGG.h5` + `image_data.json` loaded but unused at inference (~85 MB; slimming
   requires a loader change - out of audit scope).
2. `opencv-python` instead of `-headless`; ~30 training/Jupyter packages in requirements.
3. `datasets` package name collides with HuggingFace `datasets` if ever installed.
4. README status section is stale (Deployment/Frontend checkboxes).
5. `/docs` and broad CORS credentials flag exposed by default.

**OPTIONAL**
1. `.gitignore`: add tests scratch patterns and `.env*`.
2. Split requirements into base / train / dev files (later phase - do not touch yet).
3. Health endpoint verbosity / readiness detail (schema change - avoid).
4. Gzip/CDN tuning for the static bundle; container health-check wrappers.

---

## 18. Recommended Next Implementation Steps

Ordered, each a separate phase; NOTHING in this list was executed now:

1. **Commit the application** - add ignore rules for tests scratch + `.env*`, then stage
   and commit backend/scene_graph/frontend/scripts/tests/log so the remote matches reality.
2. **Configuration phase** (matches the "next phase" already reserved in Entry 10):
   build-time API base URL with same-origin default (`import.meta.env`), env-driven CORS
   origins, env-driven bind host/port. API schema and endpoints stay untouched.
3. **Normalize requirements**: re-save as UTF-8, split prod vs train, pin the CPU torch
   index for Linux, decide how the git-pinned CLIP dep is installed in CI (ensure git).
4. **Weight/cache provisioning decision**: bake HF cache + `weights/clip/ViT-B-32.pt`
   into the deploy image (recommended for determinism with offline flags intact) OR
   allow one controlled first-boot download; explicitly un-ignore / artifact-store the
   two `.pt` checkpoints so they can ship.
5. **Choose an architecture** from section 12 (owner's decision) - the single-server
   variant (section 13) is the lowest-friction match for the current design.
6. **Containerize / compose + deploy** (first actual deployment activity), with 1 worker,
   4 GB RAM, ~3 GB disk, health check on `/health`, and a start-time budget >= 150 s.
7. **Post-deploy verification**: `/health`, one real `/analyze`, frontend round-trip,
   then re-run the existing verifiers against production URLs.

================================================================================
END OF DEPLOYMENT AUDIT (inspection, measurement and reporting only)
================================================================================
