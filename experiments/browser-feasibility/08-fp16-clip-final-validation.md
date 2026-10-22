# Phase D1.1 — Final FP16 CLIP Browser Validation

**Date:** October 21, 2026
**Final classification:** **VIABLE WITH LIMITATIONS**
**Measured environment:** Headless Chrome `155.0.8059.85` on the local Windows host, ONNX Runtime Web `1.24.2`, Vite production preview, Playwright driver only. WebGPU was available; WASM used one thread because the page was not cross-origin isolated.

## Decision

The repaired FP16 OpenCLIP visual encoder is a real, browser-executable alternative to the FP32 graph in this controlled workload. Chrome completed all **30/30** FP16 matrix runs—15 WASM and 15 WebGPU—with no provider error and no page error. It accepts the same external FP32 image tensors as FP32, returns raw FP32 `[N,512]` features suitable for the unchanged FP32 relationship MLP, and preserves all 68 predicate decisions, retention decisions, and `0.30` confidence-threshold decisions in the complete reference fixture.

It is **not recommended as the unconditional production default yet**. The strongest result is WebGPU-specific; FP16 WASM is slightly slower than the measured FP32 WASM baseline, WebGPU FP16 has greater raw feature/probability variance than WASM FP16 even though all fixture decisions agree, only one recorded real-image fixture was tested, and end-to-end browser crop/resize preprocessing parity remains open. Retain FP32 plus exact crop deduplication as the default compatibility path. FP16 is a viable WebGPU-preferred candidate once broader fixtures and preprocessing parity are validated.

No production frontend/backend code, checkpoint, FP32 source artifact, or repaired FP16 source artifact was modified. The browser-visible FP16 file is an experiment-only byte-for-byte copy.

## 1. Artifact validation

### Repaired graph identity

| Property | Result |
|---|---|
| Source artifact | `d1-optimization/model/clip-vit-b-32-openai-visual-fp16-io-fp32.onnx` |
| Browser experiment copy | `browser-yolo-test/public/model/d1-fp16-clip-vit-b-32-openai-visual-fp16-io-fp32.onnx` |
| Size | 176,081,784 bytes (168.0 MiB) |
| SHA-256, both paths | `E3AF45D11EC5F50E842118FBED4D78225F75875DE486BF053BC679C1D920C004` |
| FP32 comparison size | 351,780,080 bytes (335.5 MiB) |
| Size reduction | 49.9% |
| ONNX checker | pass |
| ONNX IR / opset | IR 8 / `ai.onnx` opset 17 |
| Nodes / initializers | 2,015 / 152 |
| Initializer storage | all 152 are `FLOAT16` |
| External tensors | none; one self-contained `.onnx` file |

The original conversion was invalid because a generated cast value was consumed before being produced. `try_fp16_repair.py` topologically sorted the converted graph; the final `no_shape_infer_plus_topo_sort` variant above is the checker-valid artifact tested here. Checker validity was treated only as a structural prerequisite, not as browser compatibility evidence.

### Interface and operator review

| Interface | Value |
|---|---|
| Input | `image`: `FLOAT`, `[batch, 3, 224, 224]` |
| Output | `image_features`: `FLOAT`, `[batch, 512]` |
| Dynamic batch evidence | Chrome successfully ran batches 1, 8, 26, and 68. |
| Browser input conversion | none — the existing saved FP32 CLIP tensors are passed unchanged. |
| Browser output conversion | none — the graph returns FP32 raw features directly to the FP32 relationship MLP. |
| Feature normalization | none — all metrics and MLP calls use raw features. |

The graph contains standard opset-17 operators only: arithmetic, `Cast`, `Conv`, `Gemm`, `MatMul`, `LayerNormalization`, `Softmax`, shape/indexing/reshape operations, and `Erf`. It has no custom domain operators and no external initializers.

An attempted standalone Python CPU ONNX Runtime smoke process terminated with Windows exit code `0xC0000005` before it could emit results. That failure is recorded as a provider-specific warning and was not used as proof of browser behavior. The Chrome results below are the runtime evidence for this phase.

## 2. Browser workload and method

The test reused the D1 deterministic fixture without changing pair construction:

- nine recorded YOLO-World detections from `VG_100K_2/1.jpg`;
- 68 ordered eligible pairs, excluding self-pairs and equal-label pairs;
- 26 exact unique preprocessed union-crop tensors;
- exact reconstruction of all original 68 directed feature rows after deduplicated CLIP execution;
- unchanged subject IDs, object IDs, geometry, predicate map, stable softmax, class-50 removal, and confidence threshold `>= 0.30`.

The page ran the repaired visual graph in Chrome JavaScript and fed its raw FP32 outputs into the original FP32 relationship MLP. The FP32 comparator was the saved raw FP32 CLIP feature matrix for these exact tensors, passed through the browser WASM MLP. Prior D1 evidence established the browser FP32 pipeline’s decision agreement for this fixture. This avoids a redundant 335.5 MiB FP32 visual-session rebuild while preserving identical model inputs and unchanged downstream logic.

The full matrix has three repetitions per configuration, with sessions cached per provider. Model fetch and session construction are not included in warm inference totals. The separate single-crop probe used a fresh session per provider, one unmeasured warm-up call, then three measured calls.

## 3. Real Chrome execution and stability

| Provider | FP16 session load, full matrix* | Successful full-matrix runs | Provider errors | Page errors |
|---|---:|---:|---:|---:|
| WASM | 13,278.1 ms | 15/15 | 0 | 0 |
| WebGPU | 10,902.5 ms | 15/15 | 0 | 0 |

\* Full-matrix `sessionLoadMs` includes local `cache: 'no-store'` fetch plus session creation. The single-crop probe separately records those components below.

WebGPU emitted ONNX Runtime’s non-fatal warning that some nodes were not assigned to the preferred execution provider (for example, shape-related CPU assignment). The graph nonetheless created and ran successfully. The browser console also contained a generic local 404 and an ORT CPU-vendor warning; neither caused a page error or a failed inference.

## 4. Warm performance

### Single exact union crop

| Provider | Local fetch | Session creation | Measured warm runs | Median warm inference |
|---|---:|---:|---:|---:|
| WASM | 12,964.7 ms | 4,189.4 ms | 549.3, 533.6, 493.4 ms | **533.6 ms** |
| WebGPU | 10,165.0 ms | 1,746.5 ms | 132.3, 132.5, 126.6 ms | **132.3 ms** |

The single-crop test uses the first exact saved unique crop, explicit warm-up before timing, and validates that the returned feature has 512 elements. Fetch uses local Vite preview with `cache: 'no-store'`; it is not a public-network download estimate.

### Complete 68-directed-pair reference workload

Medians below include CLIP, exact feature reconstruction where relevant, and the relationship MLP; they exclude session load. All rows completed three repetitions.

| Provider | Mode | CLIP chunk | CLIP median | Reconstruct | MLP | Total median |
|---|---|---:|---:|---:|---:|---:|
| WASM FP16 | directed | 1 | 27,792.5 ms | 0.0 ms | 3.2 ms | 27,795.7 ms |
| WASM FP16 | directed | 68 | 31,474.9 ms | 0.0 ms | 3.7 ms | 31,478.6 ms |
| WASM FP16 | deduplicated | 1 | **10,531.2 ms** | 0.4 ms | 1.6 ms | **10,535.9 ms** |
| WASM FP16 | deduplicated | 8 | 11,099.6 ms | 0.1 ms | 1.9 ms | 11,101.6 ms |
| WASM FP16 | deduplicated | 26 | 11,573.8 ms | 0.2 ms | 2.4 ms | 11,575.7 ms |
| WebGPU FP16 | directed | 1 | 5,692.9 ms | 0.0 ms | 4.1 ms | 5,695.9 ms |
| WebGPU FP16 | directed | 68 | 4,796.9 ms | 0.0 ms | 3.1 ms | 4,800.0 ms |
| WebGPU FP16 | deduplicated | 1 | 2,355.8 ms | 0.2 ms | 3.0 ms | 2,359.0 ms |
| WebGPU FP16 | deduplicated | 8 | 1,938.5 ms | 0.1 ms | 2.5 ms | 1,941.2 ms |
| WebGPU FP16 | deduplicated | 26 | **1,375.0 ms** | 0.2 ms | 3.4 ms | **1,378.5 ms** |

### Comparison with the existing FP32 D1 baseline

| Provider / best deduplicated configuration | FP32 total | FP16 total | Result |
|---|---:|---:|---|
| WASM: FP32 chunk 8 vs FP16 chunk 1 | 9,922.3 ms | 10,535.9 ms | FP16 is 6.2% slower. |
| WebGPU: FP32 chunk 34 vs FP16 chunk 26 | 3,938.1 ms | 1,378.5 ms | FP16 is 65.0% faster (2.86×). |

The best compatible chunk is provider-specific: 1 for WASM and 26 (the entire deduplicated set) for WebGPU. This does not establish device-independent behavior; it is a real measurement on one Chrome/Windows host.

## 5. Raw feature parity

Metrics compare the final 68-directed-pair FP16 run for each provider with the saved FP32 raw visual features. No L2 normalization was applied.

| Provider | Max abs. error | Mean abs. error | Cosine similarity | Output-norm difference | FP16 / FP32 global norms |
|---|---:|---:|---:|---:|---:|
| WASM FP16 | 0.0087690 | 0.00055757 | 0.9999989067 | 0.0128742 | 93.59036 / 93.57749 |
| WebGPU FP16 | 0.1223143 | 0.00763259 | 0.9997902166 | 0.1775128 | 93.75500 / 93.57749 |

WASM FP16 is much closer numerically to the FP32 feature matrix. WebGPU FP16 has visibly larger raw feature error, so feature cosine alone is not used as the parity decision.

## 6. Downstream relationship parity

The unchanged FP32 MLP consumed the FP16-derived raw feature rows together with the original 68 subject IDs, object IDs, and geometry rows. Postprocessing remained: stable softmax, argmax, remove class 50 (`no relationship`), then retain non-class-50 predictions only at confidence `>= 0.30`.

| Provider | Probability max abs. difference | Probability cosine | Predicate class changes / 68 | Retention changes / 68 | Threshold crossings | Maximum selected-confidence difference |
|---|---:|---:|---:|---:|---:|---:|
| WASM FP16 | 0.00111830 | 0.9999998257 | 0 | 0 | 0 | 0.00111830 |
| WebGPU FP16 | 0.02369559 | 0.9999649242 | 0 | 0 | 0 | 0.02369559 |

Thus, for the complete real 68-directed-pair fixture, FP16 preserved **68/68 predicate choices**, **68/68 retain/remove outcomes**, and **68/68 threshold-side outcomes** on both providers. This is a downstream decision result, not an inference from cosine similarity alone.

## 7. Recommendation and limitations

### Classification: VIABLE WITH LIMITATIONS

FP16 is technically viable in Chrome with ONNX Runtime Web `1.24.2`, including both WASM and WebGPU, and WebGPU FP16 is the strongest measured browser visual-encoder configuration in this study: half the artifact size, 1.38 s warm for 26 exact crops/68 reconstructed directed pairs, and no changed downstream decisions in the fixture.

However, retain **FP32 plus exact union-crop deduplication** as the default public-browser compatibility choice until all of the following are resolved:

1. Validate browser crop/resize/normalization parity against the Python/OpenCLIP preprocessing path; this study intentionally uses deterministic saved Python-preprocessed tensors.
2. Repeat downstream parity on more images, pair-label distributions, and near-threshold predictions. WebGPU FP16’s maximum probability difference is 0.0237 even though no current fixture row crosses the fixed `0.30` threshold.
3. Confirm behavior on representative user hardware and Chrome/Edge versions. The local no-store Vite fetch timing is not a public-network download estimate.
4. Preserve an FP32 fallback: FP16 WASM is not a performance win on this host, while the large WebGPU win must not be assumed for browsers/devices without suitable WebGPU behavior.

No frontend integration, deployment work, confidence-threshold adjustment, model substitution, or retraining was performed.

## Evidence artifacts

- Structural inspection: `d1-fp16-runtime/fp16-graph-inspection.json`
- Raw 30-run Chrome matrix: `d1-fp16-runtime/browser-results.json`
- Matrix medians/parity summary: `d1-fp16-runtime/summary.json`
- Single-crop Chrome timing: `d1-fp16-runtime/single-crop-results.json`
- Chrome screenshot: `d1-fp16-runtime/chrome-d1-fp16.png`
- Isolated benchmark page/harness: `browser-yolo-test/d1-fp16.html`, `src/d1-fp16-main.ts`, `verify-d1-fp16-browser.mjs`
