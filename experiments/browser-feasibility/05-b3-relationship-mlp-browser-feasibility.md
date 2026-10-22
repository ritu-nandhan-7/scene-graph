# Phase B3 — Relationship MLP Browser Feasibility

**Date:** October 20, 2026
**Final classification:** **BROWSER INFERENCE VERIFIED**
**Scope:** Isolated artifacts and browser page only. The trained checkpoint was read but never overwritten.

## 1. Exact model and mappings

Checkpoint: `notebook/models/best_full_fusion_30epoch.pt`.

| Check | Result |
|---|---|
| SHA-256 before export | `198C331FFACA78FF6CB1C377231332E837E6D0D95E381587D0B430D248317683` |
| SHA-256 after export/browser test | same |
| Evaluation behavior | `model.eval()`; dropout `p=0.30` inactive |
| Checkpoint state | 8 FP32 tensors, loaded from `checkpoint["model_state_dict"]` |

Architecture verified from `CachedRelationshipModel`:

```text
visual          float32 [N,512]  (raw CLIP features; no normalization)
subject_label   int64   [N]      (0-based VG embedding ID)
object_label    int64   [N]      (0-based VG embedding ID)
geometry        float32 [N,8]   [sub cx,cy,w,h,obj cx,cy,w,h]

Embedding(150,64) for subject and object
Linear(8,32) -> ReLU
concat: 512 + 64 + 64 + 32 = 672
Linear(672,256) -> ReLU -> Dropout(0.30, inactive in eval) -> Linear(256,51)
logits          float32 [N,51]
```

Object inputs are `SceneObject.label_index - 1`. Geometry is relative to the **unclamped mathematical expanded union**; crop pixel rounding is a separate CLIP-crop concern.

The exact predicate convention was retained:

- model classes `0..49` map to Visual Genome predicate IDs `1..50`;
- model class `50` is **`no relationship`**;
- class `49` is `worn by`; it must not be confused with class 50;
- the Visual Genome map was exported into static experiment assets and independently checked: class 35 = `riding`, class 49 = `worn by`, class 50 = `no relationship`.

Browser postprocessing calculated stable softmax, selected the argmax, removed class 50, then retained only confidences `>= 0.30`. The recorded deterministic test pairs correctly retained zero rows: class-27 rows had confidence about 0.213 (<0.30), while the remaining rows predicted class 50 and were removed.

## 2. ONNX export

The exact loaded evaluation-mode model was exported with dynamic batch axes and standard ONNX opset 17.

| ONNX property | Value |
|---|---:|
| File | `b2-b3-export-parity/model/relationship-mlp-full-fusion-fp32-dynamic.onnx` |
| Inputs | `visual`, `subject_label`, `object_label`, `geometry` |
| Output | `logits` |
| Size | 821,278 bytes |
| SHA-256 | `01128B3D3E07FC5440F4C886E8307F3E4732F882E3136D7D0884D222EC259FBC` |
| Nodes / initializers | 8 / 8 |
| ONNX checker | pass |

No retraining, weight conversion, normalization, or checkpoint mutation occurred.

## 3. Python numerical parity

Using the exact saved raw CLIP features, subject/object int64 IDs, and geometry vectors for 100 deterministic rows:

| Comparison | Max abs | Mean abs | Cosine |
|---|---:|---:|---:|
| PyTorch vs Python ORT logits | `1.5258789e-05` | `1.4165891e-06` | `0.9999999999999863` |
| PyTorch vs Python ORT probabilities | `1.1920929e-07` | `3.4965036e-09` | `0.9999999999999949` |

## 4. Real Chrome browser evidence

The relationship model ran in the same Chrome page immediately after browser-produced CLIP features. It received actual browser raw features, `BigInt64Array` labels, and float32 geometry—there was no Node, Python, server, or API fallback.

| Provider | Rows | Init | MLP run | Output | Logit max abs vs Python | Probability max abs vs Python |
|---|---:|---:|---:|---|---:|---:|
| WASM | 1 | 95.1 ms | 4.2 ms | `[1,51]` | `4.0054e-05` | `1.6913e-06` |
| WASM | 10 | reused | 2.9 ms | `[10,51]` | `4.0054e-05` | `1.6913e-06` |
| WASM | 30 | reused | 2.3 ms | `[30,51]` | `4.0054e-05` | `1.6913e-06` |
| WASM | 68 | reused | 3.5 ms | `[68,51]` | `4.0054e-05` | `1.6913e-06` |
| WASM | 100 | reused | 4.7 ms | `[100,51]` | `4.0054e-05` | `1.6913e-06` |
| WebGPU | 1 | 166.3 ms | 148.1 ms | `[1,51]` | `2.2888e-05` | `4.5449e-07` |

WASM was substantially faster than WebGPU for this tiny MLP on this host. Browser `performance.memory` was an empty object, so no numeric memory figure was fabricated. There were no page errors.

## 5. Conclusion

The exact relationship MLP is small, dynamic-batch capable, numerically close to the source model, and executed successfully in Chrome under both WASM and WebGPU. The relationship MLP is **not** a technical blocker for browser-only scene graph inference. Its end-to-end feasibility is constrained by the upstream CLIP visual encoder and preprocessing, not by the classifier.

## 6. Artifacts

- Export script: `run_b2_b3_clip_relationship_export.py`
- Browser harness: `browser-yolo-test/src/main.ts`, `verify-b2-b3-browser.mjs`
- Full browser evidence: `b2-b3-browser-runtime/browser-results.json`
