# Phase B1 — YOLO-World VG150 ONNX Export and Numerical-Parity Attempt

**Date:** October 20, 2026
**Status:** **Blocked at ONNX export. No ONNX artifact was produced.**
**Scope:** Isolated experiment only. Production backend, frontend, detector wrapper, original checkpoint, and model vocabulary were not changed.

## 1. Question tested

Can the repository's **exact configured detector** be exported to ONNX through the installed official Ultralytics path, while retaining the Visual Genome 149-class configuration created by:

```python
model = YOLOWorld("notebook/models/yolov8s-world.pt")
model.set_classes(vg150_vocabulary)
```

The answer for the current environment is **not yet**. The official export reaches `torch.onnx.export` but fails during ONNX symbolic conversion, before an `.onnx` model is emitted. Therefore graph inspection, ONNX Runtime parity, and browser testing cannot be claimed for this phase.

## 2. Isolation and artifacts

All experiment source and generated artifacts are under:

```text
experiments/browser-feasibility/
```

Key files:

| File | Purpose |
|---|---|
| `run_b1_yoloworld_vg150.py` | Reproducible B1 runner. |
| `requirements-b1.txt` | Experiment-only additions: `onnx`, `onnxruntime`. Production manifests were not edited. |
| `b1-yoloworld-vg150/environment.json` | Exact runtime/package/export request evidence. |
| `b1-yoloworld-vg150/vocabulary.json` | Exact ordered 149-class VG vocabulary. |
| `b1-yoloworld-vg150/configured-model-state.json` | Post-`set_classes()` model state evidence. |
| `b1-yoloworld-vg150/pytorch-production-reference.json` | Decoded PyTorch baseline using production predictor defaults. |
| `b1-yoloworld-vg150/pytorch-raw-reference.json` | Fixed-shape raw-output baseline for future ONNX graph parity. |
| `b1-yoloworld-vg150/raw-pytorch-*.npy` | Raw float32 outputs for the three test images. |
| `b1-yoloworld-vg150/run-status.json` / `run.log` | Exact export failure and stack trace. |

The original checkpoint was not overwritten. Its SHA-256 before/after the experiment was:

```text
095F5266BB9B654BD5AD9E21E9CDEDA78E0F2C8460F5D652EAF04BAB7EE251CF
```

No `.onnx` file exists in `notebook/models/` or in the B1 output directory.

## 3. Environment findings

| Item | Observed value |
|---|---|
| OS | Windows 10.0.26300, AMD64 |
| Python | 3.11.5 |
| Ultralytics | 8.4.115 |
| PyTorch | 2.13.0+cpu |
| CUDA available | `False` (0 CUDA devices) |
| ONNX before B1 | not installed |
| ONNX installed for B1 | 1.23.2 |
| ONNX Runtime installed for B1 | 1.31.0 |
| ONNX Runtime providers | `AzureExecutionProvider`, `CPUExecutionProvider` |
| Official export API | `YOLOWorld.export(**kwargs)` |
| Official configuration API | `YOLOWorld.set_classes(classes)` |
| Export API supports ONNX | yes; exporter exposes ONNX `batch`, `dynamic`, `opset`, `simplify`, and `nms` controls |

The experiment installed only `onnx` and `onnxruntime` into the existing virtual environment. It did not upgrade or replace PyTorch, Ultralytics, the YOLO checkpoint, or production dependency files.

## 4. Exact configured detector state

The vocabulary was built with the production helper:

```python
from scene_graph.mapping import build_yolo_vocabulary
vg150_vocabulary = build_yolo_vocabulary(vg_dict)
```

That helper preserves `VG-SGG-dicts.json` label insertion order and excludes only `"background"`. The resulting list contains **149** names. `vocabulary.json` records every class in order.

After:

```python
detector = YOLOWorld("notebook/models/yolov8s-world.pt")
detector.set_classes(list(vg150_vocabulary))
```

the captured state shows:

| State | Observed value |
|---|---|
| Detector class | `ultralytics.nn.tasks.WorldModel` |
| Detection head | `ultralytics.nn.modules.head.WorldDetect` |
| Detection-head `nc` | `149` |
| `model.names` count | `149` |
| `model.names` ordering | exact equality with `vg150_vocabulary` |
| Configured text features | `WorldModel.txt_feats`, float32, CPU, shape `[1, 149, 512]`, `requires_grad=False` |
| Python-side CLIP text model remains present | yes: `ultralytics.nn.text_model.CLIP` |

Installed Ultralytics source establishes that original `WorldModel.set_classes()` computes and assigns these configured text embeddings as `self.txt_feats`, then assigns the head class count. Its `predict()` path uses `self.txt_feats` when no text tensor is supplied. Thus the VG150 configuration is frozen as a tensor in the configured PyTorch model for inference, even though it is not a registered `state_dict` buffer.

Whether this tensor would become a constant initializer in a successfully exported ONNX graph remains unverified, because export failed before graph generation.

## 5. Deterministic PyTorch reference

The runner set Python, NumPy, and PyTorch seeds to `20261020`, set the configured detector to evaluation mode, disabled gradients, and used CPU execution.

### 5.1 Production-behavior decoded reference

This route matches production detector invocation:

```python
detector.predict(image, conf=0.25, device="cpu", verbose=False)
```

No image size, NMS IoU, NMS mode, or max-detection override was passed. Captured effective Ultralytics settings were:

| Setting | Value |
|---|---:|
| confidence threshold | 0.25 |
| predictor image size setting | 640 |
| IoU NMS threshold | 0.7 |
| max detections | 300 |
| class filter | none |
| class-agnostic NMS | false |
| reported detector input resolution | 640×640 |

The production predictor uses its normal rectangular letterbox behavior where applicable. Full machine-readable class IDs, labels, confidence scores, and original-image `xyxy` boxes are in `pytorch-production-reference.json`.

| Image | Original dimensions | Detections |
|---|---:|---:|
| `datasets/visual_genome/VG_100K_2/1.jpg` | 800×600 | 9 |
| `datasets/visual_genome/VG_100K_2/100.jpg` | 350×466 | 2 |
| `datasets/visual_genome/VG_100K/107899.jpg` | 768×1024 | 14 |

For example, the 800×600 known baseline begins with `person` class ID 93 (0.813167), another `person` (0.770418), `car` ID 28 (0.645905), and `bike` ID 12 (0.563726). The complete baseline—not a manually abbreviated version—is retained in the JSON artifact.

### 5.2 Fixed-shape raw-model reference

The requested ONNX export is static `1×3×640×640`. For a later raw PyTorch-versus-ONNX comparison, the runner made a **separate** route using Ultralytics preprocessing with `imgsz=640` and `rect=False` so every source image feeds an identical 640×640 tensor shape to both runtimes.

This is intentionally not substituted for the production reference: it is an export-shape parity baseline only. It can change decoded detections relative to production rectangular preprocessing.

For all three images, the configured PyTorch model emitted:

```text
float32 [1, 153, 8400]
```

The 153 channels equal `4 box coordinates + 149 VG150 class scores`. This is direct numerical evidence that the configured raw detector output is VG150-sized—not its original generic/COCO class count.

| Image | Fixed raw input | Raw output | Decoded count after same Ultralytics NMS |
|---|---|---|---:|
| `VG_100K_2/1.jpg` | `[1,3,640,640]` | `[1,153,8400]` | 11 |
| `VG_100K_2/100.jpg` | `[1,3,640,640]` | `[1,153,8400]` | 2 |
| `VG_100K/107899.jpg` | `[1,3,640,640]` | `[1,153,8400]` | 15 |

Raw float32 outputs are retained in the three `raw-pytorch-*.npy` files for a future successful export attempt.

## 6. Export attempt

The export was attempted only **after** the exact VG150 `set_classes()` call completed.

```python
detector.export(
    format="onnx",
    imgsz=640,
    batch=1,
    dynamic=False,
    simplify=False,
    nms=False,
    opset=17,
    device="cpu",
)
```

Design choices:

- **FP32**: no quantization or precision conversion.
- **Static 640×640**, batch 1: an initial graph-parity target, not an assertion that production preprocessing is static.
- **`nms=False`**: NMS remains external, as in normal Ultralytics detection postprocessing. This avoids silently changing or embedding NMS behavior before it can be independently checked.
- **`simplify=False`**: no graph simplifier was installed or allowed to alter the failing graph.
- **Opset 17**: explicitly requested.

### Export result

Ultralytics invoked `torch.onnx.export`, but the legacy PyTorch ONNX symbolic path aborted with:

```text
torch.onnx.errors.SymbolicValueError:
Unsupported: ONNX export of operator adaptive_max_pool2d,
output size that are not factor of input size.
```

The failing constant corresponds to output size `[3, 3]`. The failure occurs in the TorchScript ONNX graph optimization/symbolic translation phase, before `onnx.load`, metadata writing, artifact copying, or ONNX Runtime session creation.

The installed Ultralytics exporter also emits a source-level warning that original YOLO-World export is not supported (distinguishing it from YOLO-World v2), but it does proceed far enough to try export. This B1 result is therefore an actual attempted export failure, not merely a refusal based on that warning.

## 7. ONNX graph inspection and ONNX Runtime result

| Requested B1 step | Result | Reason |
|---|---|---|
| ONNX graph artifact | Not produced | `adaptive_max_pool2d` symbolic export failure. |
| `onnx.checker.check_model()` | Not run | No ONNX file exists. |
| Inputs/outputs/operator inspection | Not run | No ONNX graph exists. |
| Verify frozen text features in graph | Not possible | No ONNX initializers/constant graph exists. |
| ONNX Runtime inference | Not run | No ONNX model exists. |
| Raw PyTorch/ORT numerical comparison | Not run | No ONNX output exists. |
| Reproduced decode/NMS comparison | Not run | No ONNX output exists. |
| Browser test | Not run | Python export itself is blocked; this phase does not claim browser compatibility. |

No placeholder model, altered architecture, substitute detector, model retraining, vocabulary reordering, or output transformation was used to bypass this failure.

## 8. Conclusion

### What B1 proved

1. The installed original YOLO-World model can be configured with the exact production VG150 label order.
2. `set_classes()` produces a frozen float32 `[1,149,512]` text-feature tensor and `WorldDetect.nc=149` in the live PyTorch model.
3. The configured raw PyTorch detector emits `[1,153,8400]`, consistent with 4 box channels plus 149 class-score channels.
4. A three-image production decoded baseline and a three-image fixed-shape raw baseline have been saved for future numerical parity testing.
5. The official Ultralytics ONNX export path in this pinned CPU environment fails before generating an ONNX model due to unsupported `adaptive_max_pool2d` symbolic conversion.

### What B1 did not prove

- It did **not** prove ONNX Runtime compatibility.
- It did **not** prove WebGPU, WASM, Chrome, Edge, or any browser compatibility.
- It did **not** prove that the frozen VG150 text features would be represented correctly in an ONNX graph.
- It did **not** validate postprocessing/NMS parity between PyTorch and ONNX.

### Recommended constrained next investigation

Do not proceed to browser integration, CLIP visual encoder conversion, or relationship MLP conversion on the basis of B1. First isolate this detector export blocker while preserving the current detector and vocabulary, for example by evaluating a deliberately documented alternative exporter path/version combination or a minimal source-level export-only patch for the unsupported pooling operation. Any such follow-up must:

1. continue to use `yolov8s-world.pt` plus the exact VG150 `set_classes()` output;
2. generate a real ONNX graph before claiming progress;
3. inspect that graph for 149-class output and frozen text-feature representation;
4. compare it against the saved raw tensors and decoded references; and
5. only then begin real-browser runtime testing.