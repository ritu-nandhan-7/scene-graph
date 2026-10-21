# Phase F3A — Production Hosting Preparation and UI Responsiveness

**Date:** October 21, 2026
**Final classification:** **NOT READY**

## 1. Scope and result

F3A prepared the browser-only React path for externally hosted immutable model
assets and tested a production build against a separate-origin local static
host. It did not deploy to Vercel, upload assets, provision hosting, change the
Python backend, alter ONNX artifacts, change thresholds, cap objects/pairs, or
change the public scene graph JSON contract.

The external-host simulation and five-fixture production regression passed.
The remaining deployment blocker is operational: no durable HTTPS host has yet
been selected/published and no Vercel build has been configured with its final
`VITE_BROWSER_MODEL_BASE_URL`. In addition, browser analysis still has severe
main-thread jank; the safe CPU worker extraction did **not** improve the
maximum animation-frame gap in this environment.

## 2. Asset audit and immutable manifest

The checked-in deployment manifest is:

`experiments/browser-feasibility/scene-graph-v1-asset-manifest.json`

It is generated from the exact existing E1 public asset tree and records every
relative path, byte count, SHA-256, required MIME type, and immutable cache
policy. It contains 13 files totaling **312,280,662 bytes**. The three ONNX
models total **230,981,989 bytes** (~220 MiB):

| Required path | Bytes | SHA-256 | MIME |
|---|---:|---|---|
| `model/yolov8s-world-vg150-640-static-adaptive-max.onnx` | 54,078,927 | `123c8f2c7f614e3c2449c27114ff29b0f83b016f37882d21dc766bbf082d76d5` | `application/octet-stream` |
| `model/d1-fp16-clip-vit-b-32-openai-visual-fp16-io-fp32.onnx` | 176,081,784 | `e3af45d11ec5f50e842118fbed4d78225f75875de486bf053bc679c1d920c004` | `application/octet-stream` |
| `model/d1-relationship-mlp-full-fusion-fp32-dynamic.onnx` | 821,278 | `01128b3d3e07fc5440f4c886e8307f3e4732f882e3136d7d0884d222ec259fbc` | `application/octet-stream` |
| `references/vocabulary.json` | 1,331 | `4f848a0066c644b7c39901813526ca18d7a97af6045c34760bf4075902ef8795` | `application/json` |
| `e1/references/mappings.json` | 3,921 | `08cce38767c87a1744f002b7d8fcbb948f921ac51f2fa6c105a32a6dd2228086` | `application/json` |

The ORT subtree must be published **unchanged** at `ort/`. It contains these
eight files: `ort-wasm-simd-threaded.{mjs,wasm}`, plus the
`asyncify`, `jsep`, and `jspi` `.mjs`/`.wasm` pairs. The manifest has each
individual hash and size; the largest are `asyncify.wasm` (27,099,554 bytes),
`jsep.wasm` (24,925,138), `jspi.wasm` (16,800,717), and base `.wasm`
(12,303,130).

## 3. URL resolution and deployment hosting contract

`frontend/src/services/browserInference.js` resolves every asset by joining a
relative path to one normalized trailing-slash prefix:

```text
VITE_BROWSER_MODEL_BASE_URL
  or, only for local Vite development/preview,
<Vite base>/browser-models/scene-graph-v1/
```

No browser JavaScript contains a Windows path, model-storage credential, or
FastAPI analysis endpoint. Model fetches explicitly use CORS mode and the
browser cache (`force-cache`). Sessions and metadata remain module-cached for
the lifetime of the page. Failed fetch/session promises are cleared from cache,
so a subsequent user retry is not trapped by a rejected promise.

### Required future host configuration

After selecting a provider, publish the exact manifest tree below one immutable
HTTPS prefix, for example:

```text
https://<chosen-static-host>/browser-models/scene-graph-v1/
```

Configure the host for every manifest file:

```text
Access-Control-Allow-Origin: https://<vercel-project-domain>
# `*` is also acceptable because these assets use no cookies or credentials.
Access-Control-Allow-Methods: GET, HEAD, OPTIONS
Cache-Control: public, max-age=31536000, immutable
```

Serve ONNX as `application/octet-stream`, JSON as `application/json`, ORT
`.mjs` as `text/javascript`, and ORT `.wasm` as `application/wasm`. Keep the
exact relative `ort/` layout: ONNX Runtime dynamically imports its `.mjs` and
then loads the related WASM binary.

Set this **at Vercel build time**, with no secret required or permitted:

```text
VITE_BROWSER_MODEL_BASE_URL=https://<chosen-static-host>/browser-models/scene-graph-v1/
```

The commented non-secret setting was added to `.env.example`. Do not copy the
220 MiB models under `frontend/public` or into the Vercel deployment bundle.

### Provider-neutral upload procedure (not performed)

1. Download/copy only the manifest-listed files from
   `experiments/browser-feasibility/browser-yolo-test/public/` while preserving
   the listed relative paths.
2. Verify each uploaded object byte count and SHA-256 against the manifest.
3. Set MIME, CORS, and immutable-cache headers above at the object/CDN layer.
4. Verify HTTPS `GET` and `HEAD` for every manifest path.
5. Configure the versioned public prefix in a Vercel Preview build and run the
   cross-origin acceptance test before production deployment.

## 4. Responsiveness instrumentation and safe worker extraction

F3A added `scene-graph-inference-timing` browser events. They are diagnostics
only and are not added to `{ image, objects, relationships }`.

Measured stages are:

1. image decode plus YOLO tensor preprocessing;
2. metadata/model session initialization;
3. YOLO execution;
4. YOLO decoding/NMS, pair generation, union-crop extraction, and CLIP tensor
   preprocessing;
5. OpenCLIP execution;
6. relationship MLP execution;
7. softmax/filter/dedup and scene construction.

`frontend/src/services/inferenceWorker.js` now owns only CPU/canvas work:
image decode, YOLO input creation, YOLO decode/NMS, exact E1 pair/crop creation,
CLIP crop tensor construction, and relationship scene postprocessing.
Transferable `ImageBitmap` and `ArrayBuffer` values avoid copying the large
pixel/tensor buffers back to the UI thread.

All ONNX Runtime session initialization and inference remains on the verified
main-thread path. In particular, OpenCLIP remains WebGPU-preferred and keeps
the existing WASM fallback. Moving ORT/WebGPU into a worker was intentionally
not attempted because it would be a new unverified provider path.

An initial worker implementation accidentally changed crop semantics to pad the
combined union instead of expanding each box by 20% before forming the union.
That changed two fixture relationship counts. The issue was detected by the
required regression matrix, corrected to the exact E1 algorithm, rebuilt, and
the final full matrix passed.

## 5. First-load experience

The existing real-stage loading card was retained and clarified:

- it warns that first-time setup can be slow;
- it reports actual model/session, detection, relationship preparation, visual
  feature, relationship prediction, and graph-build stages;
- it reports WebGPU initialization and explicitly reports the slower WASM
  fallback;
- it does not fabricate percentage progress;
- models are still loaded only after the user starts analysis.

The three sessions and metadata are reused after their first successful load.
Browser HTTP caching is requested, while durable cache behavior is a required
responsibility of the future static host.

## 6. Production build and cross-origin static-host simulation

Built with:

```text
VITE_BROWSER_MODEL_BASE_URL=http://127.0.0.1:4184/browser-models/scene-graph-v1/
node node_modules/vite/bin/vite.js build
```

The production build passed in **6.74 s**. `dist/` contains the normal app
files plus a separate **5.46 kB** worker chunk; it contains no ONNX or ORT
assets. Vite still warns that the main JavaScript chunk is 779.99 kB minified
(234.19 kB gzip), an existing code-splitting limitation.

The built app was served at `http://127.0.0.1:4173/`; a separate custom static
server served the model prefix at `http://127.0.0.1:4184/`. The simulator set
`Access-Control-Allow-Origin: *`, `public, max-age=31536000, immutable`, and
the required ONNX/JSON/`.mjs`/`.wasm` MIME types. Chrome completed all
inference using those cross-origin URLs. The required ORT dynamic `jsep.mjs`
and `jsep.wasm` requests succeeded. No `/api/analyze` or `:8000/analyze`
request occurred.

Reusable simulation tools retained for future preview acceptance are:

- `experiments/browser-feasibility/f3a-static-asset-server.mjs`
- `frontend/verify_f3a.mjs`

## 7. Final production-build regression matrix

All results below are real Chrome runs of the built preview using the separate
asset origin. Counts match F2 exactly after the crop correction.

| Fixture | Objects / overlays / nodes | Relationships / edge labels | Wall time |
|---|---:|---:|---:|
| `few-100.jpg` | 2 / 2 / 2 | 1 | 16.748 s |
| `fixture-1.jpg` | 10 / 10 / 10 | 15 | 9.081 s |
| `moderate-107899.jpg` | 14 / 14 / 14 | 23 | 10.995 s |
| `many-107900.jpg` | 3 / 3 / 3 | 3 | 2.948 s |
| `overlap-107901.jpg` | 7 / 7 / 7 | 12 | 7.505 s |

The final run had zero page errors and zero FastAPI analysis requests. It
validated upload/preview, overlay and React Flow rendering, repeated in-page
model/session reuse across fixtures, and the production static-host boundary.
The broader F2 image/graph/relationship inspector interaction suite, forced
WASM fallback, malformed file/image, and model-download failure tests were not
re-executed after this internal worker refactor; F2 remains the last evidence
for those behaviors. A real zero-relationship image remains unavailable.

## 8. Measured performance and responsiveness

F2's comparable fresh-preview `few-100` measurement was 15.42 s, with a
maximum animation-frame gap of **6.11 s**. F3A's separate-origin fresh
`few-100` wall time was **16.748 s**; it includes local cross-origin serving
and differs in test setup, so it is not a reliable total-time regression
comparison.

The instrumented first run recorded:

| Stage | Time |
|---|---:|
| Decode + YOLO preprocessing | 107.8 ms |
| Model/metadata initialization | 8.271 s |
| YOLO inference | 2.304 s |
| YOLO postprocess + pair/crop preprocessing | 69.8 ms |
| OpenCLIP inference | 1.083 s |
| Relationship MLP inference | 0.9 ms |
| Relationship postprocess + graph scene construction | 0.9 ms |

Warm-session runs showed initialization overhead near 21–150 ms, but model
execution varied by fixture/provider workload. The final full-matrix frame
probe measured a **6.128 s** maximum gap. This is effectively unchanged (and
slightly worse) than F2's 6.11 s; F3A therefore makes **no responsiveness
improvement claim**. The worker removes the synchronous canvas/NMS/crop work
from the UI thread, but main-thread ORT/session/provider work remains dominant
on this machine.

## 9. Remaining blockers and required next work

1. Choose and publish to a durable HTTPS static host, verify its uploaded
   objects against the manifest, set the Vercel build variable, and test a real
   Vercel Preview deployment.
2. Make a product decision about 220 MiB first use, cold-start messaging,
   cancellation/retry, and the unresolved multi-second UI freeze.
3. Investigate an ORT worker/provider architecture only as a separately
   validated effort; do not claim WebGPU worker compatibility without testing.
4. Re-run F2 interaction, forced-WASM, and error-path acceptance after F3A;
   test a true zero-relationship image, Chrome/Edge hardware variation,
   mobile/low-memory behavior, and repeated-session memory growth.
5. Preserve the existing E2 browser/Python non-parity limitation; F3A does not
   make a Python-equivalence claim.

## 10. Final classification

**NOT READY.**

The frontend can now consume an immutable versioned external asset prefix and
has passed an actual separate-origin production-build simulation with correct
CORS, MIME, caching, dynamic ORT imports, no FastAPI traffic, and unchanged
five-fixture outputs. It is not ready for Vercel deployment because the real
HTTPS asset host and Vercel build configuration do not yet exist, and the
measured approximately six-second maximum frame stall remains unacceptable
without an explicit product acceptance decision.