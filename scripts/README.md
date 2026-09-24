# Running Scene Graph Explorer locally (Windows / PowerShell)

The app has two processes: the **FastAPI backend** (ML inference) and the
**React frontend** (UI). Run them in two terminals.

## 0. One-time setup

From the repository root:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt

cd frontend
npm install
```

## 1. Terminal 1 - backend

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_backend.ps1
```

The script needs no manual venv activation - it calls the venv interpreter
directly.  Equivalent manual command:

```powershell
.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Startup loads YOLO-World + CLIP + the relationship checkpoint **once** and
takes roughly 1-2 minutes on CPU.  Wait for `Application startup complete.`
before analyzing an image.  Models are reused by every later request.

## 2. Terminal 2 - frontend

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_frontend.ps1
```

## 3. URLs

| What | URL |
|---|---|
| **Frontend (open this)** | http://127.0.0.1:5173 |
| Backend health | http://127.0.0.1:8000/health |
| Swagger UI | http://127.0.0.1:8000/docs |

## 4. Verify the backend is up

```powershell
.venv\Scripts\python.exe -c "import httpx; print(httpx.get('http://127.0.0.1:8000/health').json())"
```

Expected output: `{'status': 'healthy'}`

## 5. Using the app end to end

1. Open http://127.0.0.1:5173
2. **Choose Image** and pick a photo (e.g.
   `datasets\visual_genome\VG_100K_2\1.jpg`)
3. **Analyze** and wait for the "Analyzing image..." state to finish
4. The explorer appears: image + bounding boxes on the left, scene graph on the right
5. **Image -> graph**: click a bounding box - the matching node highlights and the inspector opens
6. **Graph -> image**: click a node - the matching bounding box highlights
7. Click an edge - the relationship, both endpoint objects and both boxes highlight
8. Click empty graph space - the selection clears and the inspector disappears
9. **Drag nodes** - the graph is unlocked by default, so the layout can be
   rearranged by hand; **Lock** freezes the nodes (pan and zoom keep working)
10. **Show object detections** - the switch sets the DEFAULT overlay
    visibility: off + nothing selected shows the raw image, while off + a
    selection still shows only the selected object / the relationship's two
    endpoints
11. **Search objects** (e.g. `person`) - highlights every match in both views,
    including numbered duplicates (`person 1`, `person 2`)
12. **Download** - saves `scene-graph.png` containing only the graph
13. **Reset** - clears file, preview, scene, search and selection

## 6. Stopping

`Ctrl+C` in each terminal.

## 7. Tests

```powershell
# frontend build
cd frontend; npm run build

# frontend logic verifiers (run from frontend/, need node_modules)
node verify_layout.mjs      # dagre TB layout + component wiring
node verify_d4_logic.mjs    # search + inspector resolution
node verify_phase_e.mjs     # toggle / inspector visibility / inverse pairs
node verify_interaction.mjs # detection visibility, numbering, lock, curved edges

# backend tests
.venv\Scripts\python.exe tests\test_postprocessing_local.py
.venv\Scripts\python.exe tests\test_pipeline_local.py
.venv\Scripts\python.exe tests\test_batching_local.py   # batched == per-pair

# purge + backend tests (backend running for test_connection.py)
.venv\Scripts\python.exe tests\test_api_local.py
.venv\Scripts\python.exe tests\test_connection.py
```

`verify_layout.mjs`, `verify_d4_logic.mjs`, `verify_phase_e.mjs` and
`verify_interaction.mjs` read `tests\_scene.json`, a real `/analyze` response
(the browser verifier does not - it drives the live app).
Regenerate it with the backend running:

```powershell
.venv\Scripts\python.exe tests\_fetch_scene.py
```

### Inference profiling (optional - needs a quiet machine)

`tests\_profile_pipeline.py` reports where an `/analyze` request spends its
time (image load, YOLO, pair enumeration, CLIP features, relationship model,
post-processing) and compares the batched path with the per-pair reference path
on the same instance.  Stop the backend first so nothing competes for the CPU:

```powershell
.venv\Scripts\python.exe tests\_profile_pipeline.py
```

Both test scripts load the models once each (roughly 30 s on CPU), so a single
run takes a few minutes.

### Browser verification (optional - needs both servers running)

`frontend\verify_browser.mjs` drives the real UI in a headless browser and
checks what a build cannot: bounding boxes actually drawn over the image, the
detection toggle (including the toggle-off + selection cases), selection
synchronisation in both directions, edge/node/empty-canvas clicks, node
dragging with the lock on and off, search highlighting, the numbered duplicate
labels, equal panel heights, curved edge paths, the exported PNG, Reset, and a
second image run.  Playwright is deliberately **not** a project dependency -
install it once:

```powershell
cd frontend
npm install --no-save playwright
npx playwright install chromium     # skip if you have Edge/Chrome installed
```

Then, with the backend (8000) and frontend (5173) running:

```powershell
cd frontend
node verify_browser.mjs
```

It prints one PASS/FAIL line per check and writes screenshots - including the
real exported `scene-graph-download.png` - to `frontend\_browser_shots\`.
Chromium, Edge and Chrome are all tried automatically.