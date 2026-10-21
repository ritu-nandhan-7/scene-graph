# Phase F3B — Vercel Blob Publication and Frontend Redeployment Preparation

**Date:** October 22, 2026
**Classification:** **BLOB ASSET PUBLICATION COMPLETE; FRONTEND DEPLOYMENT UNVERIFIED**

## Scope

F3B publishes the immutable browser model/runtime tree used by the existing
browser-only React/Vite application. It does not modify ONNX artifacts,
inference semantics, thresholds, model architecture, Python/browser parity, or
the existing Scene Graph Explorer UI.

## Completed Blob publication

The 13 manifest assets were published below this public, immutable prefix:

```text
https://wzbcnqif6ddnztyy.public.blob.vercel-storage.com/scene-graph-v1/
```

The source-of-truth manifest remains:

```text
experiments/browser-feasibility/scene-graph-v1-asset-manifest.json
```

It specifies the exact pathname, byte count, SHA-256, MIME type, and cache
policy for all three ONNX models, metadata, and ONNX Runtime assets. The
published assets retain their fixed `scene-graph-v1/` pathnames, public access,
and one-year immutable cache policy. The large OpenCLIP ONNX model was uploaded
through multipart upload.

The uploader is retained as reproducible operational tooling at:

```text
experiments/browser-feasibility/upload_vercel_blob.py
```

It validates every local manifest SHA-256 before network changes, refuses to
overwrite mismatched immutable objects, and defaults to public-content SHA-256
verification. Its explicit metadata-only verification option exists solely for
transient Windows network resets that interrupted long public Blob reads; it
still validates local hashes and exact remote object sizes. `BLOB_READ_WRITE_TOKEN`
is never stored in this repository or exposed to browser code.

## Frontend configuration

`frontend/src/services/browserInference.js` reads the Vite build-time value
`VITE_BROWSER_MODEL_BASE_URL`, normalizes it to one trailing slash, and resolves
all model, metadata, and ONNX Runtime URLs beneath it. The required Vercel value
for **both Production and Preview** is:

```text
VITE_BROWSER_MODEL_BASE_URL=https://wzbcnqif6ddnztyy.public.blob.vercel-storage.com/scene-graph-v1/
```

This value is public model-hosting configuration, not a secret. Do **not** put a
Blob read/write token, API key, private key, or any other credential in a
`VITE_*` variable: Vite embeds these variables in browser JavaScript at build
time.

The local Vite bridge at `/browser-models/scene-graph-v1/` remains for local
development/preview only. It is not shipped as a Vercel runtime dependency and
does not copy the model tree into `frontend/dist`.

## Vercel project settings

Configure the linked Vercel project in the dashboard as follows:

| Setting | Value |
|---|---|
| Root Directory | `frontend` |
| Framework | Vite |
| Build Command | `npm run build` |
| Output Directory | `dist` |
| `VITE_BROWSER_MODEL_BASE_URL` | The exact public prefix above, for Preview and Production |

The repository does not contain local `.vercel/project.json` metadata, so the
dashboard Root Directory must be verified manually. `frontend/vercel.json`
contains only Vite build/output configuration and deliberately has **no**
`rootDirectory` setting: Root Directory is a Vercel project-dashboard setting,
not a nested `vercel.json` setting.

## Git-triggered redeploy

Once this frontend commit is pushed to the branch connected to the Vercel
project, Vercel should create a deployment automatically according to that
project's Git integration. The deployment will only use the Blob prefix if the
build-time environment variable above is saved for the relevant Vercel target
before the build runs. Redeploy after adding or changing it.

## Required public acceptance test — not yet performed

Publishing the assets and pushing the frontend do **not** prove the public
application works. Before classifying the deployment as working, open the actual
Vercel HTTPS deployment in Chrome and run:

```text
experiments/browser-feasibility/browser-yolo-test/public/e1/images/few-100.jpg
```

Verify all of the following:

1. The React application loads via HTTPS.
2. All 13 Blob URLs resolve under the configured `scene-graph-v1/` prefix with
   expected MIME/cache headers and browser CORS access.
3. `few-100.jpg` produces 2 object overlays/nodes and 1 relationship edge.
4. Object/relationship selection synchronization and inspectors function.
5. A second upload/run succeeds without a page reload.
6. Network activity contains no `/api/analyze`, `:8000/analyze`, FastAPI, GPU,
   or other server-side inference request.
7. At minimum, confirm the dynamically loaded ONNX Runtime paths
   `ort/ort-wasm-simd-threaded.jsep.mjs` and
   `ort/ort-wasm-simd-threaded.jsep.wasm` load correctly.

Record the final Vercel URL, browser/version, response headers, counts, page
errors, and any failed requests. The existing F1/F2/F3A browser evidence applies
to local/separate-origin validation; it is not a substitute for this real public
Vercel acceptance test.

## Known limitations retained

- The external asset tree is approximately 312 MB; first use can require a large
  download and significant initialization time.
- Browser execution can cause multi-second UI stalls on the measured hardware.
- WebGPU may fall back to slower WASM.
- Browser results are not claimed to be numerically identical to the Python
  pipeline on all fixtures.

## Final classification

**Blob asset publication is complete. The public frontend deployment and its
browser acceptance test remain unverified.**