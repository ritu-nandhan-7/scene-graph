# Phase C1 — Real Browser YOLO-World ONNX Inference Feasibility

**Date:** October 20, 2026
**Final classification:** **BROWSER INFERENCE VERIFIED WITH LIMITATIONS**
**Scope:** Isolated browser-runtime experiment only. No production frontend/backend, FastAPI route, detector checkpoint, ONNX artifact, vocabulary, training data, or deployment configuration was modified.

## 1. Decision

The exact Phase B1.1 static VG150 YOLO-World ONNX detector completed actual inference inside a real Chromium browser through ONNX Runtime Web. This is not a model-load-only claim: browser-page `InferenceSession.create()` and `session.run()` completed and produced the required FP32 `[1,153,8400]` output tensor.

Both providers were exercised in installed Chrome:

- **WASM:** completed on all three B1 images.
- **WebGPU:** completed an independent session creation and inference on `VG_100K_2/1.jpg`.

The qualification is important: the browser Canvas implementation of the required static square letterbox route is not bit-equivalent to Python/OpenCV `INTER_LINEAR`. Therefore browser-image results cannot yet claim the B1.2 exact static-input parity result. It produced the expected output shape and substantially corresponding detections, but one near-threshold detection differs on the first image.

## 2. Test environment

| Item | Observed value |
|---|---|
| Host OS | Windows (local experiment machine) |
| Browser | Google Chrome `155.0.8059.80` (headless real browser process driven by Playwright) |
| ONNX Runtime Web | `1.24.2` |
| Build host/tooling | Node `22.14.0`, Vite `6.0.7`, TypeScript `5.9.3` |
| Browser CPU visibility | `navigator.hardwareConcurrency = 16` |
| Browser memory hint | `navigator.deviceMemory = 8 GiB` |
| WebGPU | `navigator.gpu` present; actual ORT WebGPU run completed |
| Cross-origin isolation | `false`; WASM was deliberately configured with one thread |

Chrome reported one ONNX Runtime warning (`Unknown CPU vendor`) and no page exceptions. The browser exposed an empty `performance.memory` object, so no browser memory amount is reported or invented.

## 3. Browser implementation

Created isolated test app:

```text
experiments/browser-feasibility/browser-yolo-test/
  public/model/yolov8s-world-vg150-640-static-adaptive-max.onnx
  public/images/{1,100,107899}.jpg
  public/references/{raw-pytorch-*.npy,raw-onnxruntime-*.npy,vocabulary.json}
  public/ort/ort-wasm-simd-threaded.*
  src/main.ts
  verify-browser.mjs
```

The page presents runtime/WebGPU information, model/session metadata, provider selection, image selection/upload, model loading state, durations, tensor dimensions, class labels/scores, detection table, box overlay, browser/runtime log, and a machine-readable result record.

The exact immutable model was copied as a static experiment asset. It was downloaded by the browser as **54,078,927 bytes**. The model/session reported one input and one output, and every successful run validated output dimensions `[1,153,8400]` before postprocessing.

Node was used only for Vite and Playwright orchestration. No FastAPI endpoint, Python ONNX Runtime call, Docker process, Node ONNX package, or cloud inference service participated in browser inference.

## 4. Input and postprocessing behavior

### Fixed-square preprocessing implemented

The page implements the Phase B fixed graph input route:

1. Browser image decoding applies browser orientation handling where supported by the browser decoder.
2. Compute `r = min(640 / width, 640 / height)`.
3. Resize to `round(width*r) × round(height*r)`.
4. Center-pad to 640×640 with RGB `(114,114,114)`; placement uses the Ultralytics `round(dw - 0.1)` / `round(dh - 0.1)` convention.
5. Read RGB, normalize to float32 `[0,1]`, and write NCHW `[1,3,640,640]`.

The source inspected for this behavior was Ultralytics `LetterBox`: `auto=False`, `center=True`, `padding_value=114`, `cv2.INTER_LINEAR`; predictor preprocessing then converts BGR→RGB, HWC→CHW, and divides by 255.

### Known preprocessing limitation

Browser Canvas high-quality interpolation is not specified to be OpenCV `INTER_LINEAR`. Browser-vs-Python input tensor differences measured in the successful Chrome run were:

| Image | Max abs input delta | Mean abs input delta |
|---|---:|---:|
| `VG_100K_2/1.jpg` | 0.003922 | 0.001117 |
| `VG_100K_2/100.jpg` | 0.050980 | 0.001363 |
| `VG_100K/107899.jpg` | 0.066667 | 0.005372 |

This is an image-preprocessing difference, not evidence that ONNX Runtime Web changed detector semantics. An optional saved-Python-tensor route exists in the app for pure runtime isolation when fixed tensor assets are supplied; this recorded C1 run used browser-decoded images.

### Browser postprocessing

The original raw tensor is preserved. A copy is decoded as pixel-space `xywh`, converted to `xyxy`, confidence-filtered at `0.25`, class-aware NMS-filtered at IoU `0.7`, capped at 300 detections, mapped with the exact 149-label VG vocabulary, and scaled back from the fixed letterbox image to original pixel coordinates. No raw tensor is mutated for parity metrics.

## 5. Actual browser runtime results

### WASM

| Image | Session init | First inference | Total image analysis | Output | Browser detections | B1 static reference |
|---|---:|---:|---:|---|---:|---:|
| `VG_100K_2/1.jpg` | 4,799.1 ms | 1,784.0 ms | 2,530.6 ms | `[1,153,8400]` | 10 | 11 |
| `VG_100K_2/100.jpg` | reused | 1,943.0 ms | 2,718.9 ms | `[1,153,8400]` | 2 | 2 |
| `VG_100K/107899.jpg` | reused | 2,456.0 ms | 3,182.5 ms | `[1,153,8400]` | 15 | 15 |

The first-image class sequence was the B1 reference sequence except the final near-threshold `jacket` detection was absent. The second and third image class sequences/counts matched the reference static-square sequences. This result is consistent with the separately measured browser-vs-OpenCV preprocessing difference.

### WebGPU

WebGPU was not inferred from `navigator.gpu`; ONNX Runtime Web created a WebGPU session and completed a page-context inference run.

| Image | Session init | First inference | Total image analysis | Output | Detections |
|---|---:|---:|---:|---|---:|
| `VG_100K_2/1.jpg` | 3,720.0 ms | 3,470.7 ms | 4,342.1 ms | `[1,153,8400]` | 10 |

On this host/browser configuration WebGPU was slower than WASM for the single tested image. This establishes functionality, not a performance recommendation for WebGPU.

### Warm/repeated stability

After the WASM session was initialized, a four-run sequence on `1.jpg` recorded an initial 1,666.2 ms inference followed by warm durations of **1,698.3 ms**, **1,788.9 ms**, and **1,792.0 ms**; median warm inference was **1,788.9 ms**. Every iteration completed with the same 10 browser-preprocessed detections. The browser remained responsive enough for the Playwright page flow to proceed through all image/provider selections.

## 6. Raw and decoded parity interpretation

Browser raw values must be interpreted in light of the input differences above:

| Image | Browser vs Python raw max abs | Mean abs | Class-score max abs |
|---|---:|---:|---:|
| `1.jpg` | 40.6946 | 0.008167 | 0.020680 |
| `100.jpg` | 49.5225 | 0.012377 | 0.024417 |
| `107899.jpg` | 101.0779 | 0.017267 | 0.168030 |

Those are **not** a valid ONNX Runtime Web-versus-Python inference-only measurement because the browser did not receive the identical Python float32 input tensor. They include browser decoding/resampling differences. They must not be compared to B1.2’s `<=0.02` raw criterion, which was explicitly an identical-input comparison.

The browser raw output was also compared to saved Python ONNX Runtime output and showed essentially the same deltas, supporting the conclusion that input preprocessing dominates this C1 measurement. Exact runtime-only raw parity remains to be completed with saved fixed Python float32 inputs or a browser implementation proven pixel-identical to OpenCV interpolation.

## 7. Compatibility and hosting observations

1. The self-contained graph loaded with no unsupported operator error in browser WASM or WebGPU execution.
2. ONNX Runtime Web external WASM/JSEP assets must be served alongside the app. The app configures `ort.env.wasm.wasmPaths = '/ort/'` and carries the exact `ort-wasm-simd-threaded.*` assets under `public/ort/`.
3. Vite development server module transformation interfered with ORT’s dynamic JSEP module import. The successful test used `vite preview` on the production build at `http://127.0.0.1:4175`.
4. The run used Chrome. Edge was installed but was not independently executed in this phase; Edge compatibility is therefore unverified.
5. Model caching/reload behavior was not quantified. The test deliberately fetched model bytes with `cache: 'no-store'` for transparent download-size/session measurements, so these timings are conservative for browser cache reuse.
6. The page exposes `performance.memory` when a browser provides numeric values, but Chrome’s observed value was empty. No numeric memory claim is made.

## 8. Reproducibility artifacts

- Browser app and instructions: `experiments/browser-feasibility/browser-yolo-test/`
- Real-browser orchestration: `browser-yolo-test/verify-browser.mjs`
- Full browser output (including raw arrays): `c1-yoloworld-browser-runtime/browser-results.json`
- Compact result summary: `c1-yoloworld-browser-runtime/summary.json`
- Browser screenshot: `c1-yoloworld-browser-runtime/chrome-wasm-overlay.png`
- Vite/browser logs: `c1-yoloworld-browser-runtime/*.log`

Run instructions are in `browser-yolo-test/README.md`. The browser test is intentionally isolated; it does not contact `localhost:8000` or any production route.

## 9. Final decision

**BROWSER INFERENCE VERIFIED WITH LIMITATIONS.**

The exact frozen-text, 149-class, static FP32 YOLO-World ONNX artifact executes successfully in a real browser under ONNX Runtime Web WASM and WebGPU, with actual model output and decoded detections. The remaining limitation is not model loading or ONNX operator support: it is pixel-level preprocessing equivalence to the B1.2 Python/OpenCV fixed-square route. Before claiming exact browser parity with B1.2, use saved Python float32 inputs to isolate runtime parity and/or implement a resampler demonstrated to reproduce OpenCV `INTER_LINEAR` exactly. Normal production rectangular-letterbox server behavior remains outside this static-model experiment.