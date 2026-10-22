# Browser-Only Scene Graph Explorer — Repository Audit

**Phase:** A — repository investigation
**Date:** October 20, 2026
**Scope:** Read-only audit of the currently implemented local pipeline. No production code, model artifact, Docker/deployment file, model weight, or frontend API integration was changed.

## 1. Audit result

The described architecture is substantially accurate, with two important implementation details that drive browser feasibility:

1. The detector is the local **Ultralytics YOLO-World small checkpoint** `yolov8s-world.pt`, but it is **not** used as a fixed closed-set detector. At Python startup, it is reconfigured with the 149-name Visual Genome vocabulary by `YOLOWorld.set_classes(...)`. That call builds/loads a separate CLIP text model to produce the vocabulary representation.
2. The relationship classifier does not invoke CLIP itself. It receives **raw, unnormalized 512-D `open_clip` ViT-B/32 image embeddings** from expanded subject/object union crops, then combines them with learned subject/object label embeddings and 8-D geometry.

The backend is a transport layer only. The actual inference implementation is in `scene_graph/` and `notebook/models/relationship/`.

## 2. Exact end-to-end request path

```text
frontend/src/App.jsx handleAnalyze()
  -> frontend/src/services/api.js analyzeImage(file)
  -> POST {API_BASE_URL}/analyze, multipart field "file"
  -> backend/main.py analyze(file)
     -> _decode_upload(): RGB Pillow image
     -> SceneGraphPipeline.analyze(image)
        -> YoloWorldDetector.detect(image)
        -> detection records -> Visual Genome SceneObject records
        -> ordered directed object pairs (excluding self pairs and same-label pairs)
        -> expanded union crop + 8-D geometry per pair
        -> batched open_clip ViT-B/32 image encoding (512-D)
        -> CachedRelationshipModel logits (51 classes)
        -> softmax / argmax
        -> remove class 50 ("no relationship")
        -> relationship confidence threshold >= 0.30
        -> exact directed-triple duplicate removal
        -> SceneGraphResult.to_dict()
  -> React state `scene`
  -> SceneImage bounding boxes + React Flow graph + Inspector
```

## 3. FastAPI ingress and response contract

### Source: `backend/main.py`

- `@app.post("/analyze")`, function `analyze(file: UploadFile = File(...)) -> dict` at lines 257–291.
- It reads the upload and calls `_decode_upload()` (lines 231–254):
  - rejects empty uploads;
  - maximum upload size is enforced by `MAX_UPLOAD_BYTES` (defined earlier in the module);
  - accepts JPEG, PNG, and WEBP MIME types;
  - decodes through Pillow and converts to RGB.
- An inference semaphore permits one concurrent inference. This server-only admission control is irrelevant to a browser-only port, but guards the present API.
- The endpoint calls `get_pipeline().analyze(image)` and returns `result.to_dict()` directly.
- `get_pipeline()` constructs a single long-lived `SceneGraphPipeline` using:
  - `notebook/models/yolov8s-world.pt`
  - `notebook/models/best_full_fusion_30epoch.pt`
  - `datasets/visual_genome/`

### Public JSON schema

Source: `scene_graph/pipeline.py`, `SceneGraphResult.to_dict()` at lines 91–140.

```json
{
  "image": { "width": 800, "height": 600 },
  "objects": [
    {
      "id": "obj_0",
      "label": "person",
      "confidence": 0.81,
      "bbox": { "cx": 100.0, "cy": 200.0, "width": 80.0, "height": 300.0 }
    }
  ],
  "relationships": [
    {
      "id": "rel_0",
      "subject_id": "obj_0",
      "predicate": "riding",
      "object_id": "obj_3",
      "confidence": 0.2179
    }
  ]
}
```

Important compatibility rules:

- Object IDs are `obj_{detection_index}`; the detection index is preserved even if a detector label could not be mapped.
- Relationship IDs are `rel_{final_postprocessed_index}`.
- Bounding boxes use original-image pixel coordinates in center format (`cx`, `cy`, `width`, `height`), not normalized coordinates and not `xyxy`.
- The endpoint intentionally returns no tensors, logits, model metadata, raw detections, or per-pair features.

## 4. Detector audit: exact YOLO-World behavior

### Implementation

| Item | Verified value |
|---|---|
| Wrapper | `scene_graph/detector.py`, `YoloWorldDetector` |
| Runtime package | `ultralytics` (`YOLOWorld`) |
| Checkpoint | `notebook/models/yolov8s-world.pt` |
| Checkpoint size | 27,169,314 bytes (25.91 MiB) |
| SHA-256 | `095F5266BB9B654BD5AD9E21E9CDEDA78E0F2C8460F5D652EAF04BAB7EE251CF` |
| Model variant indicated by checkpoint name | YOLOv8s-World (`yolov8s-world.pt`) |
| Vocabulary source | `datasets/visual_genome/VG-SGG-dicts.json` |
| Vocabulary builder | `scene_graph/mapping.py:build_yolo_vocabulary()` |
| Vocabulary content | `label_to_idx` iteration order, excluding exactly `"background"` |
| Vocabulary count | 149 names (VG dictionary has 150 labels including background) |
| Runtime re-heading | `self.model.set_classes(self.class_names)` |
| Detection confidence threshold | `0.25` (`SceneGraphPipeline.conf_threshold` default) |
| Input source | RGB Pillow image passed directly to Ultralytics |
| Explicit inference image size | None: relies on Ultralytics default predictor behavior |
| Explicit NMS IoU threshold / max detections | None: relies on Ultralytics default predictor behavior |
| Box source | `results[0].boxes.xyxy`, original-image pixels |
| Box conversion | `xyxy` -> center-format `BoundingBox` |

Relevant code:

- `scene_graph/pipeline.py:199–213` loads Visual Genome metadata and constructs the detector.
- `scene_graph/detector.py:36–48` calls `YOLOWorld(model_path)` then `set_classes(class_names)`.
- `scene_graph/detector.py:56–87` invokes `model.predict(image, conf=0.25, device=..., verbose=False)` and converts returned boxes.
- `scene_graph/mapping.py:104–110` creates the 149-name vocabulary.

### VG150 mapping behavior

The lookup table has 150 object labels, with `background` assigned Visual Genome ID 6. The YOLO vocabulary is the original JSON insertion order excluding that one name. The model does **not** use the YOLO class index as the relationship model label ID. It carries the detector label string forward, then resolves it via:

```python
vg_label_id = self.label_name_to_vg_id.get(det.label)
```

in `scene_graph/pipeline.py:280–293`. The resulting 1-based Visual Genome label ID is converted to the 0-based embedding ID only when relationship inference starts.

### Detector-specific browser risk

`YOLOWorld.set_classes()` is essential model behavior, not merely UI metadata. Deployment records and `requirements-deploy.txt:28–55` establish that Ultralytics lazily imports its `clip` package in this method and uses a separate `ViT-B-32.pt` text tower. The current implementation therefore depends on a model-specific text embedding initialization step before detection can reproduce the VG150 vocabulary.

**Audit verdict:** exporting only the loaded `yolov8s-world.pt` checkpoint is insufficient evidence of browser equivalence. Phase B must determine whether text features can be frozen after `set_classes`, whether they are embedded in an exported graph, and whether an exported graph produces equivalent decoded detections and NMS outcomes.

## 5. CLIP visual encoder audit

### Implementation

| Item | Verified value |
|---|---|
| Source | `notebook/models/relationship/encoder.py`, `ImageEncoder` |
| Library | `open_clip` / `open_clip_torch` |
| Model | `ViT-B-32` |
| Pretrained weights | `openai` |
| Call | `open_clip.create_model_and_transforms(model_name="ViT-B-32", pretrained="openai")` |
| Inference method | `self.model.encode_image(images)` |
| Output | `features.float()`, 512-D per crop |
| Evaluation mode | Yes (`model.eval()`) |
| Gradients | Disabled |
| Post-encoder L2 normalization | **No** |

The previously recorded local runtime preprocessing configuration is:

```text
size: (224, 224)
mode: RGB
mean: (0.48145466, 0.4578275, 0.40821073)
std:  (0.26862954, 0.26130258, 0.27577711)
interpolation: bicubic
resize_mode: shortest
fill_color: 0
```

Evidence: `tests/_api_local_err.txt`, emitted by the active `open_clip` loader; the actual transform is held in `ImageEncoder.preprocess`.

### Input region and feature semantics

CLIP runs **once per ordered object pair**, not once over the entire image and not once per individual object. It processes an expanded union crop:

```text
subject box + object box
  -> independently expand each box by 20% in width and height
  -> union of the expanded boxes
  -> clamp crop bounds to image dimensions after Python round()
  -> crop original RGB image
  -> CLIP preprocess to 224×224
  -> 512-D image embedding
```

Sources:

- `scene_graph/features.py:75–114` single-pair feature extraction.
- `scene_graph/features.py:117–174` batched feature extraction.
- `notebook/models/relationship/geometry.py:15–121` exact expansion, union, crop rounding, relative-box, and normalization logic.

The current default batches up to 8 crops through CLIP (`pair_batch_size=8`). This changes scheduling only; code comments and local tests state the features/predictions are identical to the one-pair reference path.

### CLIP artifact context

The active visual encoder is distinct from the CLIP text model used by YOLO-World initialization. Existing deployment audit records identify the local OpenCLIP visual weight cache as an approximately 577 MB safetensors artifact. This Phase A audit did not copy, redownload, or hash cache files because the recursive cache scan exceeded the command time budget. Exact artifact presence, source layout, and exported model size must be measured in Phase B.

## 6. Relationship model audit

### Checkpoint

| Item | Verified value |
|---|---|
| Checkpoint | `notebook/models/best_full_fusion_30epoch.pt` |
| Size | 2,470,857 bytes (2.36 MiB) |
| SHA-256 | `198C331FFACA78FF6CB1C377231332E837E6D0D95E381587D0B430D248317683` |
| Load code | `torch.load(..., weights_only=False)` then `checkpoint["model_state_dict"]` |
| Checkpoint top-level keys | `config`, `epoch`, `history`, `model_state_dict`, `optimizer_state_dict`, `val_accuracy`, `val_loss`, `val_macro_f1`, `val_top5_accuracy` |
| State tensors | 8 float32 tensors |

### Exact architecture

Source: `notebook/models/relationship/model.py`, `CachedRelationshipModel` at lines 94–230. Runtime construction is in `scene_graph/pipeline.py:227–243`.

```text
Inputs:
  visual                 float32 [N, 512]
  subject_label          int64   [N]     (0-based Visual Genome embedding ID)
  object_label           int64   [N]     (0-based Visual Genome embedding ID)
  geometry               float32 [N, 8]

subject_embedding        Embedding(150, 64)
object_embedding         Embedding(150, 64)
geometry_encoder         Linear(8, 32) -> ReLU
fusion                   concat [visual, subject, object, geometry]
                         512 + 64 + 64 + 32 = 672 dimensions
classifier               Linear(672, 256) -> ReLU -> Dropout(0.30) -> Linear(256, 51)
output                   float32 [N, 51] logits
```

Verified tensor shapes in the checkpoint:

```text
subject_embedding.weight      (150, 64)
object_embedding.weight       (150, 64)
geometry_encoder.0.weight     (32, 8)
geometry_encoder.0.bias       (32,)
classifier.0.weight           (256, 672)
classifier.0.bias             (256,)
classifier.3.weight           (51, 256)
classifier.3.bias             (51,)
```

At inference the model is in evaluation mode, so the stored dropout layer is inactive.

### Semantic and geometry features

Feature concatenation order is exactly:

```text
[raw CLIP visual (512), subject embedding (64), object embedding (64), geometry encoder output (32)]
```

The label inputs are derived as:

```text
subject_label = SceneObject.label_index - 1
object_label  = SceneObject.label_index - 1
```

where `SceneObject.label_index` is the 1-based Visual Genome label ID.

The 8-D geometry vector is:

```text
[subject_cx, subject_cy, subject_width, subject_height,
 object_cx,  object_cy,  object_width,  object_height]
```

Each coordinate is relative to the **unclamped mathematical union box**, then normalized by that union box’s width or height. The crop pixel bounds are separately clamped and rounded for extraction. A browser port must preserve this distinction, including JavaScript rounding differences from Python’s `round()` (Python uses ties-to-even; JavaScript `Math.round()` does not for positive `.5` values).

### Predicate mapping and prediction

Sources:

- `scene_graph/mapping.py:51–96`
- `scene_graph/predictor.py:51–127`

The 51 output classes mean:

```text
classes 0..49  -> Visual Genome predicate IDs 1..50
class 50       -> "no relationship"
```

The model applies `softmax(logits, dim=1)`, selects `argmax`, and reports the selected probability as relationship confidence.

The full Visual Genome predicate dictionary contains 50 real predicates. Verified examples: model class 35 decodes to `riding`; model class 49 decodes to `worn by`; model class 50 is the separate no-relationship class.

## 7. Pair generation and postprocessing

### Pair generation

Source: `scene_graph/pipeline.py:296–308`.

For detected/model-compatible objects in detector result order, the pipeline generates directed pairs:

```python
for i, subject in enumerate(objects):
    for j, obj in enumerate(objects):
        if i == j:
            continue
        if skip_same_label and subject.label == obj.label:
            continue
        pairs.append((subject, obj))
```

Defaults:

| Setting | Value |
|---|---:|
| self-pairs | excluded |
| same-label pairs | excluded (`skip_same_label=True`) |
| pairs are directed | yes |
| union-box expansion | `0.20` |
| CLIP crop batch size | `8` |
| relationship model batch size | `32` |

### Postprocessing

Source: `scene_graph/postprocessing.py:55–141`.

The order is semantically important:

1. Remove model class 50 (`"no relationship"`).
2. Retain confidence `>= 0.30`.
3. Remove exact duplicates keyed by `(subject object_index, predicate, object object_index)`, retaining the first pair-order occurrence.

Inverse edges are intentionally retained because direction is part of the key.

## 8. Frontend contract and browser-compatible pieces

### Existing client integration

Source: `frontend/src/services/api.js`.

- `analyzeImage(file)` POSTs multipart `file` to `${API_BASE_URL}/analyze`.
- API base uses `VITE_API_URL`, or `/api` for the combined production deployment, or `http://127.0.0.1:8000` in local development.
- `frontend/src/App.jsx:172–194` awaits that function and stores the JSON unchanged with `setScene(result)`.

### Consumers of the JSON

- `frontend/src/components/SceneImage.jsx`
  - uses `scene.image.width/height` plus center-format original-pixel boxes;
  - renders labels/confidences and selectable overlay boxes.
- `frontend/src/components/SceneGraph.jsx`
  - turns `objects` into React Flow nodes;
  - turns each relationship into a directed edge using `subject_id`, `object_id`, `predicate`, and `confidence`.
- `frontend/src/components/Inspector.jsx` and graph utility modules consume the same IDs and relationship endpoints.

This is already browser-compatible UI logic. A future browser inference adapter can preserve the production UI by producing the exact current JSON schema and introducing a separate experimental/local analyzer implementation behind an explicit feature boundary. It must not silently replace the existing API path during feasibility work.

## 9. Dependency map for browser migration

| Component / source | Current role | Browser classification | Migration implication / risk |
|---|---|---|---|
| `frontend/src/**` | Upload, React Flow, image/graph selection, rendering | Already browser-compatible | Keep unchanged initially; emit same JSON schema. |
| `frontend/src/services/api.js` | HTTP API transport | Browser-compatible but server-dependent | Do not modify during experiment. A future opt-in local adapter can coexist. |
| `scene_graph/mapping.py` | VG IDs, vocabulary order, predicate decoding | Pure Python logic | Straightforward JavaScript port; preserve JSON insertion order and IDs exactly. |
| `scene_graph/postprocessing.py` | class-50 removal, threshold, de-dup, stable IDs | Pure Python logic | Straightforward JavaScript port. |
| `notebook/models/relationship/geometry.py` | union/crop geometry | Image-processing logic | Port carefully, particularly Python banker’s rounding vs browser rounding. |
| `scene_graph/features.py` except CLIP call | crop scheduling and geometry assembly | Image-processing logic | Canvas/OffscreenCanvas/ImageBitmap equivalents needed; pixel/crop semantics need parity tests. |
| `scene_graph/detector.py` wrapper | YOLO-World invocation and conversion | Model inference + postprocessing | Must replace with ONNX/browser runtime plus exact decode/NMS. High risk. |
| Ultralytics `YOLOWorld.set_classes()` | Generate/configure VG150 text features | PyTorch/Ultralytics model behavior | Highest risk: browser runtime needs frozen exact text features or a compatible text encoder path. |
| `notebook/models/relationship/encoder.py` | open_clip visual encoding | PyTorch inference | Needs an exact-weight ONNX/Transformers.js-compatible image encoder and preprocessing parity. Large artifact risk. |
| `notebook/models/relationship/model.py:CachedRelationshipModel` | 51-class MLP | PyTorch inference | Low risk ONNX export; four tensor inputs including two int64 label arrays. |
| `scene_graph/predictor.py` | softmax/argmax/decode | Pure Python plus MLP call | JavaScript softmax/argmax is straightforward after ONNX output. |
| `datasets/visual_genome/VG-SGG-dicts.json` | object/predicate mappings | Static browser-compatible data | Ship a small derived JSON subset, preserving order/mappings. |
| `VG-SGG.h5`, `image_data.json`, image folders | Dataset/loader startup support | Not needed for browser inference after mapping extraction | Browser build should not distribute full dataset artifacts. The current inference runtime reads them through `VisualGenomeLoader`; feasibility work should isolate the minimal static mapping data needed. |

## 10. Existing measured Python baseline

Source: `tests/_profile_final.txt`; it is a prior local CPU measurement, **not browser performance**.

On `datasets/visual_genome/VG_100K_2/1.jpg` (800×600), with no CUDA:

| Measurement | Recorded result |
|---|---:|
| Pipeline construction / model load | 27.47 s |
| YOLO-World detection | 1.111 s (9 detections) in manual run; 0.31–0.34 s in warmed `analyze()` runs |
| Ordered pairs | 68 |
| Batched CLIP feature extraction | 6.985 s manually; 6.65–7.15 s in warmed `analyze()` runs |
| Batched relationship MLP | 0.003 s for 68 pairs |
| Warm full pipeline request | 7.011–7.500 s |
| Final relationships | 16 |
| Batched vs per-pair predicate parity | 68/68 |

This establishes CLIP union-crop encoding as the dominant current CPU cost. It does not predict WebGPU or browser WASM performance.

## 11. Initial browser feasibility risks, ranked

### R1 — Exact YOLO-World VG150 re-heading / text features (critical)

- **Evidence:** `scene_graph/detector.py:47–48` calls `YOLOWorld(...).set_classes(vg_classes)`; deployment requirements document the lazy Ultralytics CLIP dependency.
- **Why it matters:** browser deployment must preserve the actual learned detector plus this exact vocabulary behavior. Replacing it with COCO labels, a heuristic, or a different detector violates the project constraints.
- **Required experiment:** export the *post-`set_classes`* detector or freeze/extract its exact text features, inspect graph inputs/outputs, then compare PyTorch versus ONNX detections after equivalent decode/NMS.

### R2 — Browser-supported YOLO-World operator graph (critical)

- **Evidence:** exact ONNX graph is not yet available.
- **Required experiment:** minimal real model export, ONNX graph inspection, ONNX Runtime Python parity, then real Chrome/Edge ONNX Runtime Web execution. Node.js success is explicitly insufficient.

### R3 — Large OpenCLIP visual encoder distribution (high)

- **Evidence:** deployment audit records a roughly 577 MB OpenCLIP safetensors artifact for the exact `ViT-B-32/openai` visual encoder.
- **Why it matters:** every visitor must obtain browser-compatible model artifacts; static hosting removes compute cost but not transfer/caching constraints.
- **Required experiment:** export/reuse an exact equivalent encoder, record ONNX size, numerical parity, browser initialization memory/time, and WebGPU/WASM support.

### R4 — Pixel/crop/preprocess parity (high)

- **Evidence:** relationship prediction is trained on expanded union crops and the crop math mixes unclamped geometry with clamped/rounded crop coordinates.
- **Required experiment:** compare browser generated crops/preprocessed tensors to Python across edge cases and compare CLIP embeddings numerically.

### R5 — Pair-count scaling (high)

- **Evidence:** each directed valid pair requires one CLIP union crop. Pair count grows approximately quadratically in detected object count before same-label exclusions.
- **Required experiment:** measure browser batch scheduling, responsiveness, and memory with representative 5-image set and repeated runs.

### R6 — Relationship MLP input types (low)

- **Evidence:** MLP takes float32 visual/geometry tensors and int64 subject/object label IDs.
- **Required experiment:** export four-input ONNX graph and verify WebGPU/WASM support for `int64` embedding-index inputs; if necessary, preserving semantics may require one-hot/pre-gather handling only if parity is proven.

## 12. Phase A decision

**Phase A is complete.** The actual pipeline, model artifacts, model inputs, mappings, thresholds, postprocessing, JSON response, and UI contract have been identified from repository source and checkpoint inspection.

No browser-only feasibility claim is made at this stage. The next work must prioritize a minimal end-to-end **real-browser YOLO-World proof**, because it is the highest-risk required component. Before installing export/browser dependencies or generating ONNX artifacts, Phase B should first check the currently installed environment and model-export capability, reuse existing local checkpoints/caches, and retain all work under `experiments/browser-feasibility/`.