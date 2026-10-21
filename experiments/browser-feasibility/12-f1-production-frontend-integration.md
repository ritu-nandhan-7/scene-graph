# Phase F1 — Browser-Only Production Frontend Integration

**Date:** October 21, 2026
**Final classification:** **FRONTEND INTEGRATION COMPLETE WITH LIMITATIONS**

## 1. Scope and decision

F1 connects the existing React Scene Graph Explorer UI directly to the verified
E1 browser inference path. The frontend no longer sends an image to FastAPI for
analysis. The existing React Flow scene graph, image overlays, inspector,
selection synchronization, visibility controls, layout, and public scene JSON
contract remain unchanged.

This phase intentionally proceeds with the E1 implementation as directed. It
does **not** resume E2 parity investigation, change any model, retrain or
re-export models, change thresholds, cap pairs, alter mapping order, modify the
Python backend, delete feasibility experiments, or deploy to Vercel.

The E2 conclusion still applies: browser preprocessing and browser ONNX outputs
are not numerically identical to the Python/production references on the full
fixture set. That limitation is accepted for this portfolio frontend route and
must remain visible in future product documentation.

## 2. Files changed

| File | Change |
|---|---|
| `frontend/package.json` / `frontend/package-lock.json` | Added `onnxruntime-web@^1.24.2`, matching the validated E1 experiment runtime. |
| `frontend/src/services/browserInference.js` | New reusable browser-only inference service. |
| `frontend/src/services/api.js` | Preserved the existing import boundary, but re-exports the browser implementation rather than posting to `/analyze`. |
| `frontend/src/App.jsx` | Uses the existing `analyzeImage` call with real stage-progress updates; restricts accepted files to JPEG, PNG, and WEBP. |
| `frontend/src/App.css` | Adds minimal styling for the loading-stage detail text. |
| `frontend/vite.config.js` | Serves the existing E1 assets locally at a versioned static prefix without copying them into `frontend/`; configures ONNX Runtime's external-WASM export path. |

No existing image/graph/inspector component was redesigned or replaced.

## 3. Browser inference module

`frontend/src/services/browserInference.js` exports:

```js
analyzeImage(file, onProgress) => Promise<{ image, objects, relationships }>
```

Its result is the pre-existing frontend contract. Object IDs remain `obj_N` in
detection order; relationship IDs remain `rel_N` after filtering/deduplication.
No diagnostic/timing fields are added to the public scene object consumed by
React Flow, overlays, or the inspector.

The port retains the E1 chain and semantics:

1. static-square VG150 YOLO-World ONNX on WASM;
2. exact E1 VG150 vocabulary order and VG label mapping;
3. confidence `0.25`, class-aware NMS IoU `0.70`, and `max_det=300`;
4. all directed unequal-label object pairs in detector order, with no pair cap;
5. `20%` expanded union regions, ties-to-even bounds rounding, and original
   8-D geometry feature ordering;
6. identical pixel-crop deduplication before CLIP;
7. repaired FP16 OpenCLIP visual ONNX with WebGPU first and WASM fallback;
8. raw unnormalized CLIP features for the original relationship MLP ONNX on
   WASM;
9. stable softmax, no-relationship class removal, `>= 0.30` confidence filter,
   and first-occurrence exact directed-triple deduplication.

Sessions and metadata are cached across analyses. A module-level in-flight guard
rejects concurrent analyses, preventing double clicks from initiating duplicate
downloads or model executions. Temporary image bitmaps are closed when an
analysis completes or fails.

## 4. Model asset strategy

The selected immutable artifacts are not copied under `frontend/public` and are
not imported into JavaScript source:

| Artifact | Size |
|---|---:|
| Static VG150 YOLO-World ONNX | 54,078,927 bytes |
| Repaired FP16 OpenCLIP visual ONNX | 176,081,784 bytes |
| Original relationship MLP ONNX | 821,278 bytes |

At runtime, `VITE_BROWSER_MODEL_BASE_URL` may point to a trusted versioned static
prefix such as `https://<static-host>/browser-models/scene-graph-v1/`. When it
is unset, the service resolves a Vite-base-relative default:

```text
<Vite base>/browser-models/scene-graph-v1/
```

The prefix contains the three immutable ONNX artifacts, the E1 vocabulary and
mapping JSON, and the matching ONNX Runtime WASM runtime files. There are no
localhost URLs, Windows paths, FastAPI URLs, Python requirements, or third-party
runtime model downloads in the browser service.

For local development and `vite preview`, `frontend/vite.config.js` exposes the
existing immutable tree at:

```text
/browser-models/scene-graph-v1/
```

from `experiments/browser-feasibility/browser-yolo-test/public/`. This avoids a
second roughly 231 MB copy of the selected ONNX models in the repository.
Production hosting must publish that same versioned tree at the configured
trusted static prefix before Vercel deployment is attempted.

Vite is configured with ONNX Runtime's `onnxruntime-web-use-extern-wasm` export
condition. The verified production build emits JavaScript and CSS only; it does
not bundle a duplicate ~24.9 MB ONNX Runtime WASM binary.

## 5. React integration and user states

`App.jsx` remains the sole owner of upload, preview, scene, selection,
visibility, and reset state. The only inference-boundary change is that its
existing `analyzeImage` import now runs locally in the browser.

The loading card reports real stage transitions rather than synthetic percent
progress:

- Loading AI models
- Detecting objects
- Preparing object relationships
- Extracting visual features
- Predicting relationships
- Building scene graph
- Complete

The browser service gives useful errors for unsupported formats, decode failures,
model fetch/session errors, absent WebGPU, WebGPU initialization failure, and
concurrent requests. No-object and no-valid-pair cases return a valid empty
relationship graph instead of leaving the UI loading indefinitely.

OpenCLIP uses WebGPU when it is available and session creation works. It reports
and uses WASM otherwise. YOLO and relationship inference remain WASM by design.

## 6. Local validation results

### Executed

| Check | Result |
|---|---|
| `npm ci --ignore-scripts --no-audit --no-fund` | Passed; 157 packages installed from the lockfile. |
| JavaScript syntax checks for changed service/config modules | Passed. |
| `git diff --check` | Passed. |
| Vite production build | Passed: 208 modules transformed; build completed in 9.01 seconds. |
| Build output inspection | Passed: `dist/assets` contained only JS/CSS; no ONNX model or duplicate ORT WASM artifact was bundled. |
| Local static-asset HEAD checks | Passed for app HTML, vocabulary JSON, mappings JSON, YOLO ONNX, and the ORT dynamic-loader module. |
| Real Chrome smoke — upload/preview | Passed using E1 `few-100.jpg`. |
| Real Chrome smoke — local end-to-end analysis | Passed: 2 objects, 1 relationship, React Flow rendered, and 2 image boxes rendered. |
| Real Chrome smoke — network boundary | Passed: zero `/api/analyze` or `:8000/analyze` requests. All inference assets came from `/browser-models/scene-graph-v1/`. |
| Real Chrome smoke — errors | Passed: no page exceptions and no visible application error. Chrome logged only known non-fatal ONNX Runtime CPU-vendor/node-assignment warnings. |

The first smoke attempt correctly exposed a local middleware MIME defect for
ONNX Runtime's dynamically imported `.mjs` loader. F1 fixed the local bridge to
serve it as `text/javascript`; the repeat completed successfully.

### Pending / not claimed

- The complete multi-image acceptance matrix (original E1 fixture, many-pair
  fixture, and a second upload without reload) was not run in this phase after
  the successful `few-100` real-Chrome smoke.
- Existing long Playwright interaction suites were not rerun end-to-end against
  full browser inference because they were written for the former backend flow
  and their multi-analysis workload is materially longer with cold local models.
- No Vercel deployment, cache/CDN configuration test, device matrix, or mobile
  memory/performance test was performed.

## 7. Known limitations

1. E2 numerical parity was not achieved. Canvas/browser preprocessing and
   ONNX Runtime provider behavior can change detector candidates, CLIP inputs,
   and final graph edges relative to Python on some fixtures.
2. Browser inference has substantial cold-start and download cost. E1 measured
   approximately 9.1 seconds for a cached 170-pair WebGPU case and approximately
   38.3 seconds for its WASM-only fallback on the measured hardware; first use
   additionally initializes/downloads model sessions.
3. WebGPU behavior and FP16 outcomes require broader browser/device acceptance
   testing. The fallback is functional but can be slow.
4. The frontend bundle has a Vite large-chunk warning (about 783 kB minified JS
   after adding ONNX Runtime). The model files themselves remain external. Code
   splitting the inference service is a deployment optimization, not required
   for this F1 integration.

## 8. Remaining work before Vercel deployment

1. Publish the immutable E1 `model/`, `ort/`, `references/vocabulary.json`, and
   `e1/references/mappings.json` artifacts under a durable trusted
   `scene-graph-v1` static prefix.
2. Set `VITE_BROWSER_MODEL_BASE_URL` to that static prefix during the Vercel
   build and verify Vite base-path behavior under the production deployment URL.
3. Run real Chrome/Edge acceptance tests covering the original E1 fixture, a
   many-pair fixture, WebGPU and forced WASM fallback, second upload without
   reload, selection synchronization, and error paths.
4. Make an explicit product decision about long analyses, cancellation, cache
   headers, first-use disclosure, supported-device messaging, and browser memory
   limits.
5. Keep the documented E2 preprocessing/non-parity limitation visible; do not
   claim Python-equivalent scene graphs without a separate parity acceptance
   phase.

## 9. Final classification

**FRONTEND INTEGRATION COMPLETE WITH LIMITATIONS.**

The existing React frontend now performs the validated E1 three-model pipeline
locally in Chrome, updates the existing scene graph/overlay/inspector UI through
the original JSON contract, and makes no FastAPI analysis request. Remaining
work is production asset publishing, broader browser acceptance testing, and
explicit product handling of the already-documented parity and performance
limitations—not a missing frontend integration path.