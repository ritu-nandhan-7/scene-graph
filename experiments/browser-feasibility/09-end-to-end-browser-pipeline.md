# Phase E1 — End-to-End Browser-Only Scene Graph Pipeline Proof of Concept

**Date:** October 21, 2026
**Final classification:** **END-TO-END VERIFIED WITH LIMITATIONS**
**Scope:** An isolated experiment under `experiments/browser-feasibility/`. Production frontend, backend, API route, Python pipeline source, checkpoints, and source ONNX artifacts were not modified.

## 1. Decision

A complete scene-graph path executed in a real Chrome page:

```text
Browser image / user-selected local file
  -> VG150 YOLO-World ONNX (WASM)
  -> original-order detected objects
  -> all valid directed unequal-label pairs
  -> expanded union crop, exact crop-bound rounding, and 8-D geometry
  -> exact-pixel-crop deduplication
  -> repaired FP16 CLIP ONNX (WebGPU preferred; WASM fallback)
  -> original FP32 relationship MLP ONNX (WASM)
  -> stable softmax, class-50 removal, >= 0.30 filter, directed-triple dedupe
  -> { image, objects, relationships }
```

Chrome `155.0.8059.85` completed seven actual analyses: five distinct real images in WebGPU-preferred mode, a no-reload WebGPU repeat, and a WASM-only fallback. The harness recorded **zero page errors**. All ONNX `session.run()` calls occurred in browser page JavaScript; Node only served Vite assets and drove Playwright.

This is not ready for frontend integration yet because the current browser image-processing path is not pixel-identical to the production Python/OpenCV/Pillow paths. In particular, detector preprocessing changes detections on several images, and browser Canvas resize differs materially from Pillow/OpenCLIP bicubic preprocessing. Those effects change candidate sets and some final relationship edges. No threshold, candidate cap, model, mapping, or postprocessing rule was changed to hide that difference.

## 2. Preserved production contract

The E1 browser implementation was ported from the active `SceneGraphPipeline.analyze()` and helper modules.

| Contract item | Preserved behavior |
|---|---|
| Object IDs | `obj_{detection_index}`; browser detector order is retained. |
| Relationship IDs | `rel_{final_postprocessed_index}` after all filtering/deduplication. |
| Detection threshold | `0.25`; detector NMS IoU `0.70`; `max_det=300`. No object cap was added. |
| Bounding boxes | Original-image pixel coordinates, center form `{cx, cy, width, height}`. |
| VG mapping | Exact 149-item YOLO vocabulary and exact 150-entry VG `label_to_idx` mapping; MLP IDs are `VG ID - 1`. |
| Pair generation | Double loop in detection order; exclude self pairs and equal-label pairs; retain reverse directions. No pair cap was added. |
| Union region | Expand each input box by `20%` in width/height, then union them. |
| Crop pixels | Bounds separately clamp to image dimensions after Python-compatible ties-to-even rounding. |
| Geometry | Unclamped mathematical union, vector order `[sub cx, cy, w, h, obj cx, cy, w, h]`. |
| Visual features | Raw, unnormalized `[N,512]` FP32 CLIP output. |
| Predicate decode | Classes `0..49` map to VG predicate IDs `1..50`; class `50` is `no relationship`. Sentinels confirmed: class 35 `riding`, class 49 `worn by`, class 50 `no relationship`. |
| Relationship filtering | Stable softmax/argmax, remove class 50, retain confidence `>= 0.30`, then first-occurrence exact directed `(subject, predicate, object)` dedupe. |
| Public result | Exactly `{ "image": {}, "objects": [], "relationships": [] }`; no experimental/timing/tensor fields are added to the scene JSON. |

The page additionally accepts a **user-selected local JPEG, PNG, or WEBP file**. The file is decoded and processed locally through the same browser path; it is not uploaded to FastAPI or any cloud service.

## 3. Browser model configuration

| Stage | Immutable browser asset | Provider |
|---|---|---|
| YOLO-World | `yolov8s-world-vg150-640-static-adaptive-max.onnx`, SHA-256 `123C8F2C7F614E3C2449C27114FF29B0F83B016F37882D21DC766BBF082D76D5` | WASM |
| CLIP visual encoder | repaired FP16, FP32-I/O `d1-fp16-clip-vit-b-32-openai-visual-fp16-io-fp32.onnx`, SHA-256 `E3AF45D11EC5F50E842118FBED4D78225F75875DE486BF053BC679C1D920C004` | WebGPU when available; WASM fallback |
| Relationship MLP | `d1-relationship-mlp-full-fusion-fp32-dynamic.onnx`, SHA-256 `01128B3D3E07FC5440F4C886E8307F3E4732F882E3136D7D0884D222EC259FBC` | WASM |

Measured no-store model fetch sizes were YOLO `54,078,927` bytes, CLIP `176,081,784` bytes, and MLP `821,278` bytes. The first WebGPU-preferred session setup recorded YOLO WASM `7,103.6 ms`, CLIP WebGPU `14,962.3 ms`, and MLP WASM `186.2 ms`. These initialization values are reported separately and are not added to warm analysis time.

## 4. Preprocessing implementation and validation

### YOLO

The browser uses the previously validated fixed-square ONNX route: RGB, `[1,3,640,640]` FP32, 640-square centered letterbox with fill `(114,114,114)`, original-coordinate inverse mapping, confidence `0.25`, class-aware NMS `0.70`, and at most 300 detections.

The limitation from C1 remains: browser Canvas interpolation is not bit-equivalent to production Ultralytics/OpenCV preprocessing. It is the principal source of object/pair differences below.

### CLIP

For the Python-reference detections, all five images had **exactly equal browser/Python crop bounds**, including Python ties-to-even rounding. The browser then uses Canvas high-quality image smoothing for shortest-side resize, center crop, RGB conversion, and the exact CLIP mean/std. Canvas does not claim to implement Pillow/OpenCLIP bicubic/antialias pixel-equivalently.

| Image | Pair tensors checked | Crop bounds equal | Tensor max abs. difference | Mean abs. difference | Tensor cosine |
|---|---:|---:|---:|---:|---:|
| `fixture-1` | 68 | yes | 2.5513 | 0.04554 | 0.992799 |
| `few-100` | 2 | yes | 1.7956 | 0.04730 | 0.991477 |
| `moderate-107899` | 170 | yes | 2.1311 | 0.06587 | 0.996038 |
| `many-107900` | 6 | yes | 1.3082 | 0.10081 | 0.970491 |
| `overlap-107901` | 28 | yes | 2.2482 | 0.03831 | 0.996916 |

Therefore crop geometry/rounding is validated, but pixel/resampling parity is **not**. The measured downstream discrepancies cannot be assigned to FP16 alone: they include both different YOLO detections and Canvas-versus-Pillow preprocessing.

## 5. Five-image Python comparison

Python references were freshly generated by the unchanged production `SceneGraphPipeline` on the same local images. Object matching below is greedy, same-label matching by IoU; relationship comparison maps only matched endpoints.

| Image/category | Python objects / pairs / unique crops / rels | Browser objects / pairs / unique crops / rels | Matched objects | Mean IoU | Geometry mean / max abs. diff. | Common / Python-only / Browser-only rels |
|---|---:|---:|---:|---:|---:|---:|
| `fixture-1` — original 68-pair source | 9 / 68 / 25 / 16 | 10 / 84 / 33 / 15 | 8 | 0.9393 | 0.00908 / 0.15495 | 10 / 6 / 0 |
| `few-100` — few objects | 2 / 2 / 1 / 1 | 2 / 2 / 1 / 1 | 2 | 0.9969 | 0.00024 / 0.00104 | 1 / 0 / 0 |
| `moderate-107899` — moderate objects | 14 / 170 / 71 / 25 | 14 / 174 / 67 / 23 | 11 | 0.9681 | 0.00253 / 0.03240 | 14 / 11 / 4 |
| `many-107900` — selected many-object candidate | 3 / 6 / 2 / 3 | 3 / 6 / 2 / 3 | 3 | 0.9848 | 0.00134 / 0.01199 | 3 / 0 / 0 |
| `overlap-107901` — overlapping objects | 6 / 28 / 10 / 9 | 7 / 40 / 14 / 12 | 6 | 0.9610 | 0.00484 / 0.02831 | 8 / 1 / 1 |

For common, endpoint-mappable relationships, maximum browser/Python confidence differences ranged from `0.02686` (`few-100`) to `0.16518` (`fixture-1`). The browser did not alter the fixed `0.30` threshold to improve agreement.

Two images (`few-100`, `many-107900`) preserve complete final relationship sets under this comparison. The fixture, moderate, and overlap images do not, which blocks a frontend-ready parity claim.

## 6. End-to-end performance and stability

Timing measures the actual browser page workflow, not sums of old isolated benchmarks. It includes decode/fetch of the selected test image but excludes model download/session initialization after sessions have been cached.

| Run | Provider | YOLO | Pair generation | Crop/preprocess | CLIP + reconstruction | MLP | JSON | Total |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `fixture-1`, first WebGPU-preferred | WebGPU | 2,150.4 ms | 31.0 ms | 258.9 ms | 5,433.8 ms | 5.5 ms | 7.8 ms | 30,266.1 ms* |
| `few-100` | WebGPU | 2,484.0 ms | 19.7 ms | 4.9 ms | 432.9 ms | 1.0 ms | 0.1 ms | 2,986.5 ms |
| `moderate-107899` | WebGPU | 2,385.8 ms | 23.6 ms | 540.7 ms | 6,078.2 ms | 5.7 ms | 2.6 ms | 9,122.5 ms |
| `many-107900` | WebGPU | 2,555.1 ms | 33.8 ms | 12.2 ms | 892.6 ms | 4.1 ms | 1.0 ms | 3,701.1 ms |
| `overlap-107901` | WebGPU | 2,460.4 ms | 10.1 ms | 96.8 ms | 1,563.8 ms | 2.0 ms | 0.9 ms | 4,180.3 ms |
| `fixture-1`, repeated without reload | WebGPU | 1,698.1 ms | 6.2 ms | 137.7 ms | 3,507.8 ms | 4.0 ms | 0.8 ms | **5,394.0 ms** |
| `fixture-1`, WASM-only fallback | WASM | — | — | — | — | — | — | **38,265.8 ms** |

`*` The first fixture row is cold with respect to model sessions; its total includes session creation and must not be read as a warm pipeline time.

The repeated WebGPU fixture returned an exactly equal final JSON scene graph to its preceding WebGPU run. The WASM fallback completed end-to-end, but its final JSON was not byte-identical to WebGPU, reinforcing that FP16/provider behavior needs broader acceptance testing before default use.

The browser exposed `performance.memory` as an empty object, so no numeric browser-memory claim is made. Chrome logged the known non-fatal CPU-vendor warning and WebGPU node-assignment notices; there were no page exceptions.

## 7. Reproducible artifacts

| Artifact | Purpose |
|---|---|
| `browser-yolo-test/e1.html` / `src/e1-main.ts` | Isolated user-image/browser-only E1 page. |
| `browser-yolo-test/verify-e1-browser.mjs` | Playwright driver; no Node inference. |
| `run_e1_browser_pipeline_references.py` | Fresh, unchanged-production Python references plus static E1 assets. |
| `summarize_e1_browser_pipeline.py` | Reproducible Python/browser comparison summarizer. |
| `e1-browser-pipeline/browser-results.json` | Full real-Chrome output. |
| `e1-browser-pipeline/summary.json` | Compact machine-readable comparison and performance summary. |
| `e1-browser-pipeline/example-browser-scene-graph.json` | Example exact browser response shape. |
| `e1-browser-pipeline/chrome-e1.png` | Browser screenshot. |

## 8. Production integration recommendation

Do **not** modify the production React app in this phase.

Before opt-in frontend integration, complete the following:

1. Replace or validate browser resize operations against the exact production OpenCV YOLO letterbox and Pillow/OpenCLIP bicubic preprocessing paths. Demonstrate tensor and downstream scene-graph parity on a larger fixture set.
2. Validate FP16 WebGPU and FP16 WASM against FP32 on multiple devices, image distributions, and confidence cases near `0.30`; retain an FP32 fallback.
3. Decide explicit product behavior for long workloads: the measured 170-pair case is about `9.1 s` after sessions are cached, and WASM-only fallback is about `38.3 s` on the fixture.
4. Repeat on representative Chrome and Edge hardware, including devices without WebGPU.
5. Integrate only as an explicit browser mode with progress/cancel behavior, artifact caching, clear privacy wording, and the current server path retained as a compatibility fallback until parity acceptance criteria pass.

## 9. Final classification

**END-TO-END VERIFIED WITH LIMITATIONS.**

The browser can run the actual three-model scene-graph inference chain and generate the existing public JSON structure from a local image without a backend or cloud inference request. However, it does not yet reproduce the production Python scene graph reliably across all tested images because browser preprocessing changes detector candidates and CLIP inputs. The remaining work is parity and product-readiness validation, not proof that browser inference itself is possible.