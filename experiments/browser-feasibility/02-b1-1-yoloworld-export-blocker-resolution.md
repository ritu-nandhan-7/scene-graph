# Phase B1.1 — YOLO-World Adaptive Max-Pooling ONNX Export Blocker Resolution

**Date:** October 20, 2026
**Final classification:** **EXPORT SUCCEEDED BUT PARITY FAILED**
**Scope:** Isolated experiment only. No production backend/frontend/model-wrapper source, model checkpoint, trained architecture/weights, Visual Genome vocabulary, or NMS configuration was modified.

## 1. Decision

The original configured YOLO-World detector can be exported to a valid, self-contained, FP32 static `640×640` ONNX graph after an export-only replacement for the unsupported adaptive max-pooling nodes. The replacement is mathematically exact for the three observed static feature-map shapes, and it is bit-exact against the original PyTorch pooling/model before export.

However, this phase does **not** establish raw detector parity against the Phase B1 pre-export tensors. After `YOLOWorld.export()` mutates export state on the in-memory model, its raw box channels differ substantially from the saved B1 pre-export raw tensors (maximum absolute errors `632.60`–`634.09`), while class-score channels remain exact and decoded detections remain effectively identical. ONNX Runtime closely matches that post-export PyTorch trace model, but the required saved-reference raw parity does not pass. Therefore this result is conservatively classified **EXPORT SUCCEEDED BUT PARITY FAILED** and browser execution must not be claimed.

## 2. Root cause and exact failing operation

### Original B1 error

The installed PyTorch 2.13 legacy ONNX exporter stopped before artifact creation with:

```text
Unsupported: ONNX export of operator adaptive_max_pool2d,
output size that are not factor of input size.
```

The target was `[3, 3]`.

### Instrumented operation identity

Forward hooks on the exact configured detector (`YOLOWorld(yolov8s-world.pt)` followed by the production VG150 `set_classes()`) located all live `AdaptiveMaxPool2d` calls. Results are recorded in `b1-1-yoloworld-export-resolution/adaptive-pool-trace.json`.

| Module instance | Owning module | Input tensor | Output tensor | Requested output |
|---|---|---:|---:|---:|
| `model.16.im_pools.0` | `ImagePoolingAttn` | `[1,256,80,80]` | `[1,256,3,3]` | `[3,3]` |
| `model.16.im_pools.1` | `ImagePoolingAttn` | `[1,256,40,40]` | `[1,256,3,3]` | `[3,3]` |
| `model.16.im_pools.2` | `ImagePoolingAttn` | `[1,256,20,20]` | `[1,256,3,3]` | `[3,3]` |

The source is `ultralytics.nn.modules.block.ImagePoolingAttn` (installed Ultralytics 8.4.115), which creates:

```python
self.im_pools = nn.ModuleList([nn.AdaptiveMaxPool2d((k, k)) for _ in range(nf)])
```

The configured checkpoint YAML places this module at model layer 16 in the YAML **head list**. It consumes layers 15, 12, and 9 and is in the detector neck/head path. It is **not** the backbone `SPP` or `SPPF` module; those use regular static `MaxPool2d` operations and were not changed.

For the required static, batch-one `640×640` detector input, these spatial shapes are fixed at `80×80`, `40×40`, and `20×20`. The replacement rejects any other spatial shape rather than silently using altered semantics.

## 3. Workarounds investigated

### A. Exact static regional-max replacement — selected

An experimental `StaticAdaptiveMaxPool2d3x3` module was added only to:

```text
experiments/browser-feasibility/run_b1_1_yoloworld_export_resolution.py
```

Immediately before export, it replaced only the three in-memory modules above. The `.pt` checkpoint was never written. Ultralytics subsequently deep-copies that model for export.

For an input dimension `L` and output bin `i ∈ {0,1,2}`, PyTorch adaptive max pooling uses the region:

```text
[floor(i·L/3), ceil((i+1)·L/3))
```

The replacement constructs precisely those overlapping regions with tensor slices, applies `amax` over each height/width region, and concatenates the resulting `3×3` grid. It does **not** substitute a fixed-kernel/stride `MaxPool`, which would be incorrect for the non-divisible `80`, `40`, and `20` dimensions.

The exact regions were:

| Input H/W | Bins |
|---:|---|
| 80 | `[0,27)`, `[26,54)`, `[53,80)` |
| 40 | `[0,14)`, `[13,27)`, `[26,40)` |
| 20 | `[0,7)`, `[6,14)`, `[13,20)` |

### B. Custom ONNX symbolic — not used

No custom symbolic registration or custom-runtime ONNX operator was used. The emitted graph uses standard ONNX `Slice`, `ReduceMax`, and `Concat` operations for the replacement, avoiding a browser-incompatible custom runtime dependency.

### C. Alternative exporter/toolchain — not pursued after A

The exact static replacement removed the original exporter block in the current isolated environment, so no broad PyTorch/Ultralytics upgrade or model substitution was justified. The remaining issue is raw-reference parity, not the original adaptive-pooling exporter failure.

## 4. Mathematical pooling-equivalence evidence

`pooling-equivalence.json` contains comparisons between the original `nn.AdaptiveMaxPool2d((3,3))` and the replacement.

- Tested relevant detector feature-map shapes: `80×80`, `40×40`, `20×20`.
- Tested additional non-square/boundary shapes: `5×7`, `3×3`, `2×4`.
- Tested each shape with deterministic signed arange, negative arange, seeded random, and constant negative inputs.
- Every case had identical output shape `[2,3,3,3]`, **exact tensor equality**, maximum absolute difference `0.0`, and mean absolute difference `0.0`.

The full detector check before export also compared the patched model to the saved B1 raw output for image `1.jpg` and was exact:

```text
shape: [1,153,8400]
max absolute error: 0.0
mean absolute error: 0.0
box channels: 0.0
class-score channels: 0.0
```

Thus the pooling replacement itself was proven exact for the fixed target input configuration.

## 5. Export result

Export configuration remained constrained to the requested detector setup:

```text
checkpoint: notebook/models/yolov8s-world.pt
vocabulary: production build_yolo_vocabulary(VG-SGG-dicts.json), 149 ordered classes
configuration: set_classes(vocabulary) before export
format: ONNX / FP32
input: [1,3,640,640], static
opset: 17
batch: 1
embedded NMS: no
quantization: no
```

Artifact:

```text
experiments/browser-feasibility/b1-1-yoloworld-export-resolution/
  yolov8s-world-vg150-640-static-adaptive-max.onnx
```

Size: **54,078,927 bytes** (about 54.1 MB decimal / 51.6 MiB).

The original checkpoint SHA-256 before and after was unchanged:

```text
095F5266BB9B654BD5AD9E21E9CDEDA78E0F2C8460F5D652EAF04BAB7EE251CF
```

## 6. ONNX graph validation and configured-text preservation

`onnx.checker.check_model()` passed.

| Item | Result |
|---|---|
| ONNX input | `images`, FP32 `[1,3,640,640]` |
| ONNX output | `output0`, FP32 `[1,153,8400]` |
| Detector channel structure | 4 box channels + 149 VG class-score channels |
| Graph nodes / initializers | 705 / 173 |
| External tensor data | None |
| Custom-domain operators | None |
| Adaptive-pool operators remaining | None |
| NMS embedded | No |

The graph contains standard ONNX operators only. The replacement is represented through standard `Slice` (46 total graph nodes), `ReduceMax` (31), and `Concat` (32), alongside normal detector operators. There are no custom domains.

The exact ordered 149 names in ONNX metadata equal the production VG vocabulary. Although the configured `WorldModel.txt_feats` is not a PyTorch state-dict buffer, the frozen feature tensor is embedded in the graph as an FP32 `Constant` attribute:

```text
onnx::Expand_174  shape [1,149,512]  FLOAT
```

There is no ONNX graph input for text features and no external text encoder/runtime dependency. This confirms the configured post-`set_classes()` text state was folded into this graph.

## 7. ONNX Runtime numerical parity

Python ONNX Runtime 1.31.0 ran the model with `CPUExecutionProvider` only. Identical fixed-square predictor preprocessing and the same saved reference images were used.

### Raw tensors

There are two raw comparisons that must be distinguished:

1. **Saved B1 pre-export PyTorch vs patched PyTorch after export:** class scores remain exact, but raw box channels differ materially. The exporter changes state on the supplied in-memory model while preparing export, so this is not caused by the static pooling replacement (which was exact immediately before export).
2. **Patched/export-state PyTorch vs ONNX Runtime:** ONNX Runtime closely reproduces the actual traced/export-state PyTorch model, but is not bit-exact.

| Image | Saved B1 → patched PyTorch max abs | Saved B1 class-score max abs | Saved B1 box max abs | Patched PyTorch → ORT max abs | Patched PyTorch → ORT mean abs |
|---|---:|---:|---:|---:|---:|
| `VG_100K_2/1.jpg` | 633.7784 | 0.0 | 633.7784 | 0.006409 | 0.000002016 |
| `VG_100K_2/100.jpg` | 634.0902 | 0.0 | 634.0902 | 0.007446 | 0.000001847 |
| `VG_100K/107899.jpg` | 632.6025 | 0.0 | 632.6025 | 0.012939 | 0.000002074 |

The B1.1 runner’s strict raw criterion (`max_abs_error <= 1e-4` versus saved B1 raw output) therefore failed. This is why `run-status.json` reports:

```json
{
  "status": "parity_failed",
  "raw_parity_pass": false,
  "decoded_parity_pass": true
}
```

No threshold was changed to improve the result.

### Decoded detector / NMS parity

The exact Ultralytics NMS configuration was retained: `conf=0.25`, IoU `0.7`, `max_det=300`, class-aware (non-agnostic) NMS. The same decode path used in B1 was applied to raw PyTorch and ORT outputs.

| Image | PyTorch detections | ORT detections | Missing / extra | Class sequence | Minimum matched IoU | Maximum confidence difference |
|---|---:|---:|---|---|---:|---:|
| `VG_100K_2/1.jpg` | 11 | 11 | 0 / 0 | identical | 0.999997712 | 0.000003427 |
| `VG_100K_2/100.jpg` | 2 | 2 | 0 / 0 | identical | 0.999999509 | 0.000001669 |
| `VG_100K/107899.jpg` | 15 | 15 | 0 / 0 | identical | 0.999996624 | 0.000004232 |

Decoded outputs thus agree for all test images, but decoded agreement cannot override failure of the explicit raw-reference requirement.

## 8. Browser compatibility precheck

This is a graph-only precheck, not a browser execution claim.

- The graph is self-contained (no external tensor files and no runtime text encoder).
- It uses no custom-domain operators or custom runtime ops.
- The replacement uses standard `Slice`, `ReduceMax`, and `Concat`; these are conventional ONNX operators rather than custom symbolic/runtime extensions.
- Other material operator families include `Conv`, `Einsum`, `LayerNormalization`, `Resize`, `Softmax`, `Gather`, `MaxPool`, and `ReduceL2`.
- The static FP32 download is approximately 54.1 MB before HTTP compression/caching.

This makes ONNX Runtime Web WASM/WebGPU investigation plausible *only after raw-parity reconciliation*. It does **not** prove ORT Web operator coverage, graph partitioning, WebGPU execution, WASM execution, browser numerical parity, download behavior, or browser memory/performance.

## 9. Remaining blockers

1. **Strict raw parity failure:** raw box tensor representation/state after export does not match Phase B1 saved pre-export raw outputs, even though class scores and decoded detections align.
2. **Exporter mutation must be isolated/explained:** the current Ultralytics export process sets export-related state on the supplied in-memory model. A subsequent constrained investigation would need to identify the precise mutation affecting raw boxes and compare a clean original model against an independently prepared export copy without changing detector behavior.
3. **No browser test:** per the phase rules, browser ONNX Runtime testing must wait until Python raw parity passes.

## 10. Files produced

```text
experiments/browser-feasibility/
  run_b1_1_yoloworld_export_resolution.py
  run_b1_1_yoloworld_export_resolution_task.cmd
  02-b1-1-yoloworld-export-blocker-resolution.md
  b1-1-yoloworld-export-resolution/
    pooling-equivalence.json
    adaptive-pool-trace.json
    export-only-modification.json
    yolov8s-world-vg150-640-static-adaptive-max.onnx
    onnx-graph-inspection.json
    raw-onnxruntime-1.npy
    raw-onnxruntime-100.npy
    raw-onnxruntime-107899.npy
    onnxruntime-parity.json
    run-status.json
    run.log
```

## 11. Final decision

**EXPORT SUCCEEDED BUT PARITY FAILED.**

The `adaptive_max_pool2d([3,3])` export blocker is resolved for static `640×640` inference through a proven exact export-only pooling implementation. A valid ONNX graph with frozen configured VG150 text features was emitted and runs in Python ONNX Runtime. Nevertheless, the requirement to establish raw numerical parity with the saved Phase B1 PyTorch references is not satisfied. Stop here; do not proceed to CLIP export, relationship MLP export, frontend migration, deployment, or browser execution on the basis of this phase.