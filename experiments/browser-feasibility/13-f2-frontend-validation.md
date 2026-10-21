# Phase F2 — Full React Scene Graph Explorer Validation Before Vercel

**Date:** October 21, 2026
**Final classification:** **NOT READY**

## 1. Scope

F2 validated the React Scene Graph Explorer integrated with the browser-only E1 pipeline. This phase did not redesign the UI, modify the Python backend or ONNX artifacts, change inference thresholds/object limits/mappings, deploy to Vercel, or restart browser/Python numerical-parity work. Tests used real headless Chrome driven by Playwright; model execution occurred in browser page JavaScript.

## 2. Environment and fixtures

| Item | Value |
|---|---|
| Development application | Vite at `http://127.0.0.1:5173/` |
| Built application | `vite build`, then `vite preview` at `http://127.0.0.1:4173/` |
| Runtime | `onnxruntime-web@1.24.2`; YOLO + relationship MLP WASM, OpenCLIP WebGPU-preferred/WASM fallback |
| Fixtures | Existing E1 fixture JPEGs under `experiments/browser-feasibility/browser-yolo-test/public/e1/images/` |

| Fixture | Coverage |
|---|---|
| `few-100.jpg` | Few objects |
| `fixture-1.jpg` | Original fixture |
| `moderate-107899.jpg` | Moderate objects / high candidate-pair workload |
| `many-107900.jpg` | Existing many-candidate-pair fixture |
| `overlap-107901.jpg` | Overlapping objects |

## 3. React UI fixture matrix — development server

All five fixtures were uploaded and analyzed in one no-reload React session. Counts are actual browser results. `Elapsed` is wall-clock from clicking **Analyze** until the results section appeared; it includes browser/UI/session behavior and is not model-only timing.

| Fixture | Preview | Objects / overlays | Relationships / edge labels | React Flow nodes | Elapsed |
|---|---:|---:|---:|---:|---:|
| `few-100` | Yes | 2 | 1 | 2 | 48.53 s |
| `fixture-1` | Yes | 10 | 15 | 10 | 36.98 s |
| `moderate-107899` | Yes | 14 | 23 | 14 | 39.90 s |
| `many-107900` | Yes | 3 | 3 | 3 | 34.24 s |
| `overlap-107901` | Yes | 7 | 12 | 7 | 34.14 s |

Every run completed without a visible application error or page exception. For every successful run, overlay count equaled object count and React Flow node count. Every fixture produced one or more relationships, so the real empty-relationship rendering path is **untested** and not claimed as passed.

## 4. Repeated uploads and state replacement

After the five-fixture sequence, F2 selected `few-100.jpg` without a page reload and analyzed it again.

| Check | Result |
|---|---|
| New file selection clears prior explorer state | Passed: prior overlap scene (7 boxes, 7 nodes, 12 edges) became 0 boxes/nodes/edges; selection and inspector cleared. |
| Repeated `few-100` analysis | Passed: 2 objects, 1 relationship, 2 nodes, graph rendered; 33.46 s observed wall time. |
| Prior graph/overlay leakage | Not observed. |
| Session/request reuse in shared page | Passed: exactly 7 model/runtime/metadata URLs occurred across the five-fixture plus repeat session—one per required asset URL, not one per upload. |

The repeated `few-100` result was consistent within this run (2 objects, 1 relationship). This is an integration consistency observation, not a Python/provider parity claim.

## 5. Image ↔ graph ↔ inspector interactions

Interaction tests used the final overlap scene.

| Interaction | Result |
|---|---|
| Click image bounding box | Passed. Clicking `obj_0` selected the box and showed Object Details with ID, label, confidence, bounding box, and relationships. |
| Image selection highlights graph object | Passed. Built preview showed `obj_0` with its intended selected border: `3px solid rgb(239, 68, 68)`. React Flow's own `.selected` class is click-origin-specific and is not the state contract for image-origin selection. |
| Click graph node | Passed. Clicking `obj_0` selected the matching image box and correct Object Details inspector. |
| Click relationship label | Passed. Clicking `rel_0` (`behind`) showed Relationship Details: subject `car 1`, predicate `behind`, object `surfer`, confidence `53.85%`; selected-edge styling appeared. |
| Switch image after selection | Passed. New file selection cleared selections and stale inspector before new inference. |
| Empty-relationship UI | **Untested**: no real F2 fixture produced zero relationships; no synthetic scene was used. |

No interaction-related code change was required.

## 6. Model loading, fallback, and error behavior

### WebGPU-preferred

Normal dev and preview contexts exposed `navigator.gpu` and completed the WebGPU-preferred path. Successful runs had zero page errors, zero failed requests, and zero `/api/analyze` or `:8000/analyze` requests. Chrome emitted known non-fatal ONNX Runtime CPU-vendor/node-assignment warnings only.

### Forced WASM fallback

A fresh context suppressed `navigator.gpu` before page scripts. The real fallback completed `few-100`:

| Check | Result |
|---|---|
| `navigator.gpu` | `false` |
| Preview | Yes |
| Objects / relationships / nodes | 2 / 1 / 2 |
| End-to-end wall time | 45.21 s |
| Page errors / failed requests / FastAPI calls | 0 / 0 / 0 |

No provider-specific inference behavior was altered.

### Empty-cache asset behavior

Fresh contexts requested vocabulary JSON, mapping JSON, YOLO ONNX, relationship ONNX, CLIP ONNX, and ORT `.mjs`/`.wasm` assets under `/browser-models/scene-graph-v1/`. The three ONNX files total **230,981,989 bytes** (~220 MiB): YOLO 54,078,927; CLIP 176,081,784; relationship MLP 821,278 bytes.

### Failure paths

| Case | Result |
|---|---|
| Aborted YOLO model request | Meaningful UI error: `Could not load a required browser model asset. Failed to fetch`; no graph, no page exception, no FastAPI call. |
| Text file selected | `Please select an image file (JPEG, PNG or WEBP).` |
| Corrupt JPEG submitted | `This image could not be decoded. Choose a valid JPEG, PNG, or WEBP image.` |

No fake percentage progress was introduced.

## 7. Responsiveness and timing

Observed end-to-end development times were 33.46–48.53 s across the five-fixture/repeat session. The fresh built-preview `few-100` run took 15.42 s; fresh forced-WASM `few-100` took 45.21 s. These are browser/UI wall-time measurements, not E1's separately instrumented model-stage timings.

A fresh real `few-100` responsiveness probe completed in 15.45 s and recorded 309 animation frames but a **6,111.5 ms maximum frame gap**. The page recovered and did not crash, but this is material main-thread jank and a deployment/product limitation. Numeric browser-memory growth was not available in this Chrome environment and is **untested**.

## 8. Production build and local preview

`vite build` passed: 208 modules, 7.50 s; output was `index.html`, 28.23 kB CSS, and 782.79 kB minified JS (235.37 kB gzip). No ONNX models or duplicate ORT WASM were emitted to `dist/`. Vite emitted the expected large-chunk warning.

A fresh real Chrome `vite preview` run passed:

| Check | Result |
|---|---|
| Upload/preview/inference | Passed: 2 objects, 1 relationship, 2 overlays/nodes |
| Image→graph highlight and Object Details | Passed (`obj_0`, red selected node border) |
| Model/metadata/ORT paths | All resolved below `/browser-models/scene-graph-v1/` |
| MIME types | `.onnx` `application/octet-stream`; `.mjs` `text/javascript`; `.wasm` `application/wasm` |
| Failed requests / page errors / FastAPI analysis calls | 0 / 0 / 0 |
| Cold preview wall time | 15.42 s |

The observed UI transitions included **Loading AI models** and **Extracting visual features**. Other named synchronous stages were too brief for the mutation observer; they are genuine stages, not fabricated percentages.

## 9. Issues found and fixes

No new F2 application integration defect was found, so no application source was changed. F2 verified F1's `.mjs` local-asset MIME fix: the ORT dynamic module is now served as `text/javascript` and production preview inference completes.

## 10. Vercel Hobby readiness and blockers

### Blocking deployment issue

**Production model hosting is absent.** Local Vite dev/preview middleware bridges `/browser-models/scene-graph-v1/` to experiment assets. It does not put the roughly 220 MiB model payload into `dist/`, and this middleware will not exist in a Vercel static deployment. Without immutable external model hosting and build-time `VITE_BROWSER_MODEL_BASE_URL`, the deployed default base-relative model URLs will fail.

### Required configuration

Set `VITE_BROWSER_MODEL_BASE_URL` at Vercel build time to a stable HTTPS, versioned asset prefix containing:

```text
model/
references/vocabulary.json
e1/references/mappings.json
ort/
```

Do not use localhost, filesystem paths, or FastAPI endpoints.

If this host is cross-origin, it must allow browser GET/HEAD from the Vercel frontend and serve correct ONNX/JSON/`.mjs`/`.wasm` MIME types. The static host should use immutable long-lived caching; the current local bridge's `no-cache` policy is not suitable for ~220 MiB first-use downloads. ORT `.mjs` and `.wasm` must remain at the configured versioned prefix. Current app routing is only `/`; no SPA deep-link rewrite was needed for F2, but one is needed if routes are added later.

The frontend bundle intentionally remains small enough to avoid duplicating models, but Vercel cannot serve the model tree until a distinct asset strategy is implemented. Do not simply copy the ONNX artifacts into the frontend deployment without assessing Hobby deployment, bandwidth, and cache constraints.

## 11. Remaining limitations

1. Publish immutable E1 assets at a durable HTTPS host, configure `VITE_BROWSER_MODEL_BASE_URL`, then verify an actual Vercel deployment against CORS/cache/ORT runtime paths.
2. Make explicit product decisions on cold download disclosure, retry/cancellation, supported devices, and 6.11-second observed frame jank.
3. Test Chrome/Edge hardware diversity, mobile/low-memory behavior, repeated long-session memory growth, and a real zero-relationship image.
4. Retain E2's documented browser/Python preprocessing and provider non-parity limitation; F2 makes no Python-equivalence claim.

## 12. Final classification

**NOT READY.**

The integrated React application passed the real-browser five-fixture matrix, interaction tests, repeated upload/session-reuse check, forced-WASM fallback, model/image failure states, and real built-preview test with no FastAPI inference calls. It is not deployable to Vercel yet because production model hosting and `VITE_BROWSER_MODEL_BASE_URL` configuration do not exist, local Vite middleware will not be available there, and large cold-start/jank behavior requires deployment/product acceptance. No Vercel deployment was performed.
