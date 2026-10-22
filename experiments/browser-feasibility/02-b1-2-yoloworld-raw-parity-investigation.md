# Phase B1.2 — YOLO-World Raw Output Parity Investigation

**Date:** October 20, 2026
**Final classification:** **EXPORT VERIFIED WITH REPRESENTATION DIFFERENCE**
**Scope:** Isolated detector-export investigation only. No production backend, frontend, detector wrapper, checkpoint, vocabulary, trained weights, or deployment artifact was modified. This phase does not begin browser execution, CLIP export, relationship-model export, frontend integration, or deployment.

## 1. Decision and root cause

The B1.1 apparent `632.60–634.09` raw box-channel discrepancy was **not** caused by an ONNX export semantic change, a detector-head export flag, a change in coordinate normalization, or altered image preprocessing.

It was a **comparison-order bug in the B1.1 experiment runner**:

1. The saved B1 tensor was retained in raw detector representation: four decoded box channels in `xywh` followed by 149 class-score channels.
2. B1.1 passed its in-memory PyTorch raw array to Ultralytics `non_max_suppression()` for decoded comparison.
3. That NMS routine converts the first four channels from `xywh` to `xyxy` **in place**.
4. B1.1 then compared the unchanged saved raw tensor with the already mutated PyTorch tensor. Class scores were untouched and therefore stayed exact, while only box channels showed the large apparent difference.

The controlled B1.2 reproduction proves this exact transformation bit-for-bit:

| Control on `VG_100K_2/1.jpg` static input | Box max abs error | Class-score max abs error |
|---|---:|---:|
| Saved B1 raw vs fresh configured PyTorch raw, **before** NMS | `0.0` | `0.0` |
| Saved B1 raw transformed `xywh → xyxy` vs fresh array **after** NMS | `0.0` | `0.0` |
| One raw array before vs after NMS | `633.778381` | `0.0` |

Thus there was a real **representation difference between the pre-NMS raw output and the array after NMS mutates it**, but there was no representation difference between B1 raw PyTorch and export-state PyTorch/ONNX raw outputs when compared correctly. The valid static FP32 ONNX model is verified against a freshly loaded and production-configured PyTorch detector on identical static-square input tensors.

## 2. Relevant Ultralytics implementation paths

Installed environment: Ultralytics `8.4.115`, PyTorch `2.13.0+cpu`.

### Detector and box representation

- `C:\Users\psytr\Desktop\Projects\scene_graph\.venv\Lib\site-packages\ultralytics\nn\tasks.py:1112–1122` — `WorldModel.set_classes()` writes the configured `[1,149,512]` text features to `txt_feats` and sets `WorldDetect.nc=149`.
- `...\tasks.py:1146–1185` — `WorldModel.predict()` uses the configured text tensor, passes it to `ImagePoolingAttn` / `WorldDetect`, and expands text only when required by batch/export state.
- `...\nn\modules\head.py:187–195` — `Detect._get_decode_boxes()` creates anchors from feature maps and decodes DFL distances, multiplying by stride.
- `...\nn\modules\head.py:211–218` — `Detect.decode_bboxes()` calls `dist2bbox(..., xywh=xywh and not self.end2end and not self.xyxy)`.
- `...\utils\tal.py:392–418` — anchors are centered at `(grid + 0.5)` and `dist2bbox(..., xywh=True)` returns `[cx, cy, width, height]`; `xywh=False` returns `[x1, y1, x2, y2]`.
- `...\nn\modules\head.py` `WorldDetect.forward()` / `_inference()` — concatenates those decoded four box channels with 149 sigmoid class-score channels, yielding `[1,153,8400]`.

The raw detector representation is therefore **pixel-coordinate `xywh` in the letterboxed 640×640 input space**, not normalized coordinates, not grid-relative distances, and not `xyxy`.

### Export preparation

- `...\engine\exporter.py:828–834` — exporter deep-copies the supplied model, switches it to eval/FP32, and fuses it.
- `...\engine\exporter.py:859–872` — exporter sets `WorldDetect.dynamic`, `export=True`, `format="onnx"`, `max_det`, `agnostic_nms`, `xyxy`, and resets cached `shape`.
- With B1/B1.1 parameters (`nms=False`, ONNX), the exporter configuration is `xyxy=False`; therefore its decode branch remains `xywh`.
- `...\nn\tasks.py:236–268` — `BaseModel.fuse()` folds convolution/batch-norm modules on the exporter copy.

### The B1.1 measurement issue

- `C:\Users\psytr\Desktop\Projects\scene_graph\experiments\browser-feasibility\run_b1_1_yoloworld_export_resolution.py:370–377` decoded `torch_raw` using `non_max_suppression()` and then recorded `saved_b1_vs_patched_pytorch_raw` with that same mutable array.
- `C:\Users\psytr\Desktop\Projects\scene_graph\experiments\browser-feasibility\run_b1_yoloworld_vg150.py:231–245` shows the shared decode path. `non_max_suppression()` receives `torch.from_numpy(raw)` and mutates the first four channels during its `xywh → xyxy` conversion.

## 3. Controlled detector-state comparison

The B1.2 runner started a fresh process, loaded `yolov8s-world.pt`, applied the exact production Visual Genome 149-label vocabulary, and used the same static preprocessing tensor for every candidate. Detector state is retained in `b1-2-yoloworld-raw-parity/detector-state.json`.

| Candidate / state | Eval | `head.export` | `head.format` | `head.xyxy` | Strides | Output state |
|---|---:|---:|---|---:|---|---|
| Fresh configured original | yes | false | `null` | false | `[8,16,32]` | `[1,153,8400]` after first forward |
| Pooling-only copy | yes | false | `null` | false | `[8,16,32]` | same |
| Export-prepared deep copy | yes | true | `onnx` | false | `[8,16,32]` | same |
| Existing ONNX Runtime model | n/a | n/a | n/a | n/a | embedded static graph | `[1,153,8400]` |

Additional observed state:

- `WorldDetect.nc=149`, `reg_max=16`, `max_det=300`, `end2end=false`, and class-aware NMS (`agnostic_nms=false`) throughout.
- The configured text encoder object remained available in the Python model; the existing ONNX graph had already been verified in B1.1 to embed frozen configured text features and require no runtime text input.
- The head cached anchors/strides became `[2,8400]` / `[1,8400]` after the first 640×640 forward, as expected. Resetting cached `shape` in the export copy only caused anchor regeneration; it did not change outputs.
- B1.2 uses a deep copy for pooling and export-state tests. The fresh original model was never modified.

The explicit export-copy state did **not** change detector output: fresh original versus pooling-only versus fused/export-flag copy was bit-exact for all three input tensors.

## 4. Verified representation transformation

The exact transformation performed by NMS is:

```text
raw box = [cx, cy, w, h]
x1 = cx - w / 2
y1 = cy - h / 2
x2 = cx + w / 2
y2 = cy + h / 2
```

This is the `xywh → xyxy` formula independently reproduced in B1.2. It exactly equaled the raw array after the NMS call in the controlled reproduction. NMS does not change class-score channels before selection, explaining why B1.1 reported exact class scores alongside large box errors.

No normalization or stride scaling was added by export. Stride multiplication and DFL decoding occur before both PyTorch and ONNX raw output, in the common `WorldDetect` graph path.

## 5. Correct raw tensor comparison

All values below are evaluated before NMS mutates any candidate. Inputs are the same B1 static preprocessing tensors (`[1,3,640,640]`, `imgsz=640`, `rect=False`). Metrics separate four box channels from 149 score channels.

| Image | Fresh PyTorch vs saved B1 | Fresh vs pooling-only | Fresh vs export-state copy | Fresh vs ONNX: full max abs | ONNX box max abs | ONNX class-score max abs |
|---|---:|---:|---:|---:|---:|---:|
| `VG_100K_2/1.jpg` | exact | exact | exact | `0.009490967` | `0.009490967` | `0.000014275` |
| `VG_100K_2/100.jpg` | exact | exact | exact | `0.007324219` | `0.007324219` | `0.000004888` |
| `VG_100K/107899.jpg` | exact | exact | exact | `0.013244629` | `0.013244629` | `0.000011444` |

The PyTorch candidates are bit-exact to saved B1 tensors in both boxes and class scores. The small ONNX Runtime deltas are ordinary floating-point implementation differences and are bounded by `0.013244629` over the complete raw tensor; they are not a coordinate-format conversion or semantic head change.

## 6. Fresh production-configured PyTorch vs ONNX decoded parity

Decoded parity was performed from raw copies only, preserving the arrays used for numerical comparison. The comparison used identical static-square preprocessing and the unchanged B1 Ultralytics postprocessing configuration:

- confidence threshold `0.25`;
- IoU threshold `0.7`;
- maximum detections `300`;
- class-aware NMS (`agnostic_nms=false`);
- identical `scale_boxes()` coordinate rescaling to each original image.

| Image | Fresh PyTorch / ONNX detections | Class-ID and label order | Missing / extra | Minimum matched box IoU | Maximum confidence difference |
|---|---:|---|---|---:|---:|
| `VG_100K_2/1.jpg` | `11 / 11` | identical | `0 / 0` | `0.999997712` | `0.000003427` |
| `VG_100K_2/100.jpg` | `2 / 2` | identical | `0 / 0` | `0.999999509` | `0.000001669` |
| `VG_100K/107899.jpg` | `15 / 15` | identical | `0 / 0` | `0.999996624` | `0.000004232` |

This comparison is against a **fresh original production-configured PyTorch model**, not merely the exporter trace model. It proves no missing/extra detections and identical detection-label order for the exact tensors the static ONNX model accepts.

### Production-reference boundary

The existing B1 normal production references remain valid and were not overwritten: `9`, `2`, and `14` detections for the three images. They use the normal Ultralytics predictor call without a `rect=False` override, which applies rectangular letterboxing where applicable. The static ONNX graph accepts only `[1,3,640,640]`; B1’s fixed-shape graph-parity path intentionally uses square preprocessing and obtains `11`, `2`, and `15` detections.

Consequently, those two count sets must not be compared as though they came from identical tensors. The verified claim is narrower and correct: **for an identical static-square preprocessed tensor, ONNX matches a fresh production-configured detector semantically and numerically within the reported backend tolerance.** A future browser product would still need to implement this static-square preprocessing exactly, or a new dynamic/non-square export investigation would be required before claiming parity with the present server predictor’s normal rectangular route.

## 7. Artifacts and reproducibility

Created:

```text
experiments/browser-feasibility/
  run_b1_2_yoloworld_raw_parity_investigation.py
  02-b1-2-yoloworld-raw-parity-investigation.md
  b1-2-yoloworld-raw-parity/
    detector-state.json
    mutation-and-transformation-evidence.json
    raw-and-decoded-comparison.json
    run-status.json
    run.log
```

`run-status.json` reports `status: success`, raw parity pass at absolute tolerance `0.02`, decoded parity pass, and the original checkpoint SHA-256 after the run:

```text
095F5266BB9B654BD5AD9E21E9CDEDA78E0F2C8460F5D652EAF04BAB7EE251CF
```

The temporary scheduled task used only to accommodate the local command timeout was removed after completion.

## 8. Remaining risks

1. **Static-shape constraint:** the validated artifact is fixed at batch one and 640×640. Browser preprocessing must exactly reproduce B1’s `rect=False` square-letterbox tensor construction for this result to apply.
2. **Normal production rectangular preprocessing remains a separate behavior:** no claim is made that feeding the existing server’s variable rectangular inputs to a static graph is possible or equivalent.
3. **Browser runtime remains untested:** B1.2 verifies Python ONNX Runtime CPU only. It does not establish ONNX Runtime Web WASM/WebGPU operator support, browser numerical behavior, performance, memory usage, download behavior, or preprocessing implementation parity.
4. **Raw-array handling:** future experiments must decode copies (`raw.copy()`) or calculate raw metrics before NMS. `non_max_suppression()` must be treated as a mutable postprocessing operation.

## 9. Final classification

**EXPORT VERIFIED WITH REPRESENTATION DIFFERENCE.**

The previous raw-parity failure is fully explained by an in-place postprocessing `xywh → xyxy` representation conversion occurring before B1.1 calculated “raw” metrics. Under correctly controlled comparisons, the fresh configured PyTorch model exactly reproduces B1 raw references; the pooling-only and export-prepared models remain exact; and the existing ONNX graph achieves close raw numerical parity plus complete decoded parity on all three fixed-shape B1 inputs.

Per phase scope, stop after this report. No browser, CLIP, relationship-model, frontend, or deployment work was started.