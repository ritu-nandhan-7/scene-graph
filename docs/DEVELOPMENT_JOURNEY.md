# Scene Graph Explorer: Development Journey

> **Production demo:** [https://scene-graph.vercel.app/](https://scene-graph.vercel.app/)
> **Current production topology:** a Vite/React application deployed by Vercel, with inference executed in the visitor's browser and versioned model/runtime assets fetched from Vercel Blob.
> **Documentation purpose:** preserve the engineering decisions, evidence, failures, and accepted limitations that led from a Visual Genome research pipeline to the public browser-only explorer.

## 1. Executive narrative

Scene Graph Explorer began as a scene-graph prediction project: given an image, identify objects and predict directed semantic relationships such as `person → riding → bike`. The original implementation was a Python research and application stack built around the Visual Genome (VG) scene-graph data, YOLO-World object detection, CLIP visual features, and a learned relationship classifier. It subsequently became a FastAPI-backed React application.

Deployment pressure changed the technical direction. The Python inference chain was valid, but it was large, CPU-intensive, slow to cold-start, and awkward to host economically as a public service. The project therefore investigated whether the **same essential three-model chain** could run locally in a browser. That investigation did not replace the model with a simpler detector or a heuristic relationship rule. It exported a Visual-Genome-configured YOLO-World detector, an OpenCLIP visual encoder, and the existing relationship MLP to ONNX; verified each component in real Chrome; measured performance and numerical behavior; and integrated the resulting pipeline into the existing React UI.

The final public implementation is a deliberately documented trade-off:

- It performs inference on the user's device rather than uploading the image to a FastAPI inference API.
- It preserves the browser scene-graph JSON contract and the original relationship-model inputs, thresholds, mappings, and postprocessing rules.
- It is **not claimed to be numerically identical to the Python pipeline for all images**. Browser image resizing/preprocessing and ONNX Runtime behavior changed candidates and/or downstream results on some fixtures.
- It can have a substantial first-use download and initialization cost, can fall back from WebGPU to slower WASM, and did not eliminate the measured multi-second UI stall.

This document is chronological. The companion [Architecture and Reproducibility Guide](ARCHITECTURE_AND_REPRODUCIBILITY.md) describes the current system as it is implemented and deployed.

## 2. Chronological experiment summary

| Era / phase | Objective | Outcome | Primary evidence |
|---|---|---|---|
| Visual Genome foundation | Represent annotated images as objects, boxes, and directed predicates | Custom loader and visualization foundation established | [`datasets/visual_genome/loader.py`](../datasets/visual_genome/loader.py), [`datasets/visual_genome/models.py`](../datasets/visual_genome/models.py) |
| Relationship training | Learn a predicate/no-relationship classifier using visual, label, and geometry features | 51-output full-fusion model checkpoint selected after 30 epochs | [`notebook/models/relationship/model.py`](../notebook/models/relationship/model.py), [`scene_graph/mapping.py`](../scene_graph/mapping.py) |
| Python pipeline | Turn notebook work into reusable image-level inference | `SceneGraphPipeline` combined detector, pairs, CLIP, MLP, and JSON postprocessing | [`scene_graph/pipeline.py`](../scene_graph/pipeline.py), retained local reference tests in [`tests/`](../tests/) |
| FastAPI + React | Expose the Python pipeline and visualize results interactively | Local two-process application was validated | [`backend/main.py`](../backend/main.py), [`frontend/src/`](../frontend/src/), retained API tests in [`tests/`](../tests/) |
| OCI/Docker planning | Host CPU inference on an Always Free ARM64 VM | Deployment topology and scripts were prepared, but real OCI validation was explicitly still required | Historical deployment work; not the current production path |
| B1 | Export exact VG150-configured YOLO-World | Initial official export failed on `adaptive_max_pool2d` | [`run_b1_yoloworld_vg150.py`](../experiments/browser-feasibility/run_b1_yoloworld_vg150.py) |
| B1/B1.1/B1.2 | Export the configured detector and establish an ONNX baseline | Initial exporter failed on adaptive max pooling; export-only static pooling replacement and corrected parity interpretation produced a valid static graph | [`run_b1_1_yoloworld_export_resolution.py`](../experiments/browser-feasibility/run_b1_1_yoloworld_export_resolution.py), [`run_b1_2_yoloworld_raw_parity_investigation.py`](../experiments/browser-feasibility/run_b1_2_yoloworld_raw_parity_investigation.py) |
| C1/B2/B3 | Prove browser execution of detector, visual encoder, and relationship model | All three ONNX components executed in Chrome; preprocessing and performance limitations remained | [`browser-yolo-test`](../experiments/browser-feasibility/browser-yolo-test/), [`run_b2_b3_clip_relationship_export.py`](../experiments/browser-feasibility/run_b2_b3_clip_relationship_export.py) |
| D1/D1.1 | Reduce browser CLIP cost without silently changing decisions | Exact crop deduplication helped; INT8 changed decisions; FP16 became a viable WebGPU-preferred candidate | [`browserInference.js`](../frontend/src/services/browserInference.js), [`scene-graph-v1-asset-manifest.json`](../experiments/browser-feasibility/scene-graph-v1-asset-manifest.json) |
| E1/E2 | Validate end-to-end browser graph and investigate parity | Entire chain ran locally; strict browser/Python parity was not achieved | [`inferenceWorker.js`](../frontend/src/services/inferenceWorker.js), [`browserInference.js`](../frontend/src/services/browserInference.js) |
| F1/F2/F3A | Integrate, regression-test, externalize assets, and investigate responsiveness | UI integration and regression passed; worker extraction did not remove the frame stall | [`verify_interaction.mjs`](../frontend/verify_interaction.mjs), [`vite.config.js`](../frontend/vite.config.js) |
| F3B and public deployment | Publish immutable assets and deploy the frontend | Vercel Blob asset publication established the final hosting contract; the supplied production URL is now the public demo | [`upload_vercel_blob.py`](../experiments/browser-feasibility/upload_vercel_blob.py), [`scene-graph-v1-asset-manifest.json`](../experiments/browser-feasibility/scene-graph-v1-asset-manifest.json) |
| F3B and public deployment | Publish immutable assets and deploy the frontend | Vercel Blob asset publication established the final hosting contract; the supplied production URL is now the public demo | [`upload_vercel_blob.py`](../experiments/browser-feasibility/upload_vercel_blob.py), [`scene-graph-v1-asset-manifest.json`](../experiments/browser-feasibility/scene-graph-v1-asset-manifest.json) |

## 3. Origins: Visual Genome and the scene-graph representation

### Motivation

A useful scene graph is more than a list of object classes. It represents an image as a directed graph whose nodes are detected objects and whose edges are semantic predicates. This makes it possible to distinguish a `person` merely near a `bike` from a `person → riding → bike`, and to inspect those predictions spatially and graphically.

### Dataset exploration and ground truth

The project used the Visual Genome Scene Graph Generation data. The retained workflow describes the core local data layout as `VG-SGG.h5`, `VG-SGG-dicts.json`, `image_data.json`, and Visual Genome image folders. The dataset contributes 150 object classes and 50 Visual Genome predicate classes. The custom loader creates typed scene structures for image metadata, objects, bounding boxes, and relationships, and supports scene visualization rather than treating labels as disconnected notebook arrays.

This exploration established an important discipline used throughout the project: several numeric namespaces coexist and cannot be interchanged.

| Namespace | Range / meaning |
|---|---|
| Visual Genome object label ID | 1–150; e.g. the dictionary's object label identifier |
| MLP object embedding index | 0–149; VG object ID minus one |
| Visual Genome predicate ID | 1–50 |
| Relationship model output class | 0–49 for predicates, 50 for `no relationship` |
| YOLO-World custom class index | 0–148 into the 149-item VG vocabulary with `background` removed |

The apparently small distinction between predicate ID 50 and model output class 50 later became critical: VG predicate ID 50 is `worn by`, whereas model class 50 means `no relationship`. The mapping module preserves that distinction explicitly.

### Ground-truth visualization as a development tool

Before detector-driven inference, scenes could be loaded from ground truth and drawn with their boxes and relationships. This allowed the project to validate the representation, inspect label mappings, and test feature construction incrementally: cached features first, then ground-truth objects, then ground-truth relationships, then detector outputs, and finally full image-level inference. The approach avoided debugging detection, crop construction, model indexing, and graph rendering all at once.

## 4. Detector and relationship model development

### 4.1 YOLO-World with the Visual Genome vocabulary

**Motivation.** Relationship prediction needs candidate objects. The project chose YOLO-World because it could be configured with the project's Visual Genome-compatible object vocabulary rather than limiting the interface to a generic closed label set.

**Implementation.** The Python wrapper loads `yolov8s-world.pt`, builds a vocabulary from `VG-SGG-dicts.json` in its stored insertion order, removes only `background`, and calls `YOLOWorld.set_classes(...)`. The resulting detector vocabulary contains 149 names. At Python inference time the wrapper used a detection confidence threshold of `0.25`; later browser postprocessing reproduced `0.25` confidence, class-aware NMS IoU `0.70`, and `max_det=300`.

**Decision.** The configured detector—not a replacement COCO detector—became a hard constraint for later browser work. Browser feasibility had to preserve the 149-class order and freeze the configured text features into the exported graph.

### 4.2 Relationship classifier: visual, semantic, and geometric fusion

**Motivation.** A relationship cannot be inferred reliably from labels alone. `person` and `bike` may be unrelated, adjacent, or linked by `riding`; visual context and relative geometry matter.

**Implementation.** The trained `CachedRelationshipModel` receives four inputs per directed pair:

1. a raw 512-dimensional OpenCLIP ViT-B/32 visual embedding from an expanded union crop;
2. a 0-based subject object-label embedding ID;
3. a 0-based object object-label embedding ID; and
4. an eight-value normalized geometry vector.

The classifier has separate 150×64 subject and object embeddings, an `8 → 32 → ReLU` geometry branch, concatenates `512 + 64 + 64 + 32 = 672` values, then applies `Linear(672,256) → ReLU → Dropout(0.30) → Linear(256,51)`. In evaluation mode dropout is inactive.

**Training context and limitation.** The retained workflow records a 30-epoch full-fusion checkpoint and emphasizes severe predicate imbalance. It also notes that a reported 68% accuracy did not adequately represent minority-predicate behavior, so macro F1 mattered. This consolidated history does not invent a final macro-F1 score because the retained sources inspected for this document do not provide one as a verified final number.

### 4.3 CLIP features and union geometry

For each directed subject/object pair, both boxes are enlarged by 20% in width and height, then combined into a union region. The crop is clamped and rounded to valid image pixels, while geometry is computed relative to the **unclamped mathematical union**. That distinction is intentional: pixel crop bounds determine which CLIP pixels are encoded, whereas the geometry vector retains the expanded coordinate relationship used by the trained model.

The visual branch uses `open_clip` ViT-B/32 with OpenAI weights and the visual encoder only. It produces raw float32 512-D embeddings; there is no post-encoder L2 normalization. The original preprocessing includes RGB conversion, resize to 224 with bicubic interpolation and antialiasing, center crop, and CLIP mean/std normalization. Those seemingly ordinary image operations later became the largest unresolved browser/Python parity boundary.

## 5. From notebooks to a Python/FastAPI application

### 5.1 Reusable scene-graph pipeline

**Objective.** Convert notebook-level work into a testable application component without duplicating model logic in an HTTP layer.

**Implementation.** `SceneGraphPipeline` loads the Visual Genome mappings, configures YOLO-World, initializes frozen CLIP and the relationship checkpoint, then executes:

```text
image → YOLO-World detections → model-compatible objects
      → ordered directed unequal-label pairs
      → expanded union crops + geometry + CLIP features
      → relationship MLP → softmax/argmax
      → remove no-relationship → confidence filter → directed-triple dedupe
      → { image, objects, relationships }
```

Pairs preserve detection order; self-pairs and pairs with equal labels are excluded, but reverse directions remain. The postprocessor removes class 50, retains confidence at least `0.30`, and keeps the first occurrence of an exact directed `(subject, predicate, object)` triple.

**Result.** The source log records local end-to-end tests and a representative 9-object, 16-relationship API result. In the measured CPU reference path, batched CLIP union-crop encoding dominated warm inference: approximately 6.65–7.15 seconds for 68 crops, while the batched relationship MLP took roughly 0.003 seconds for 68 pairs. This motivated later attention to browser visual-feature cost rather than optimization of the small MLP.

### 5.2 FastAPI service

**Motivation.** A web UI needed a stable application-facing response rather than direct notebook invocation.

**Implementation.** The historical FastAPI layer decodes JPEG, PNG, or WEBP uploads, enforces a 20 MB cap, converts the image to RGB, and returns `SceneGraphResult.to_dict()`. A process-wide pipeline is initialized once and reused. The server returned only public scene JSON—not tensors, embeddings, logits, or raw detector arrays.

**Result.** The post-training log records five local API tests as passing: health, successful analysis, missing file, invalid image, and unsupported type. It also records the intended server behavior: one loaded pipeline, not one model load per request.

### 5.3 React Flow explorer and synchronized image UI

The UI was designed around a single scene object shared by image overlays, graph nodes/edges, and an inspector. Object IDs such as `obj_0` are the bridge between views; labels alone are insufficient because a scene can contain two `person` detections.

The React application added:

- responsive percentage-based bounding-box overlays aligned to the rendered image;
- a React Flow graph laid out by Dagre from top to bottom;
- deterministic curved routing and label placement for parallel, fan-in/fan-out, and inverse-predicate edges;
- shared selection so image clicks highlight graph nodes and graph clicks highlight image boxes;
- object-label search, graph-only visibility controls, graph lock/unlock, and PNG download;
- inspector views for selected object or relationship details.

Hiding graph elements is deliberately presentation-only: it never deletes scene data, image detections, search entries, or API/browser results.

## 6. Why server hosting became difficult

### Original hosting idea

The initial deployment plan was a server-backed site: React frontend, FastAPI API, and CPU-loaded YOLO-World, CLIP, and relationship model. The retained deployment audit measured a local two-process system and documented a significant resource profile. It recorded a 45–120 second cold-start range and approximately 6–9 seconds of synchronous CPU inference, with the visual encoder dominating warm work.

### Docker, Cloud Run, and OCI evidence

The repository contains Docker and OCI materials, including a two-container Caddy/React/FastAPI Compose topology and an API-only OCI split-deployment design for a Vercel frontend. These are important historical engineering artifacts, but the retained OCI guide explicitly distinguishes local preparation from work that had to be performed on real ARM64 hardware. It should not be read as proof that a particular OCI VM or Cloud Run deployment was completed.

The deployment audit explains why generic serverless hosting was uncomfortable for this stack: cold starts, large runtime artifacts and caches, CPU-bound CLIP execution, and multi-gigabyte memory implications for multiple workers. The OCI route was designed as a low-cost always-on alternative, but required actual ARM64 build, memory, public networking, TLS, and end-to-end validation.

### Decision

Rather than claiming that the server path had been made effortless, the project pursued a different architecture experiment: download browser-compatible models once, execute locally, and remove the per-image inference API from the public frontend. This shifted compute and privacy characteristics to the client while introducing substantial model-transfer, browser compatibility, and preprocessing-parity risks.

## 7. Browser-only feasibility journey

### 7.1 Phase A: establish the non-negotiable contract

The initial browser audit established that the detector was configured dynamically with a 149-name Visual Genome vocabulary and that the relationship classifier consumed **raw** CLIP embeddings, labels, and geometry—not a generic CLIP similarity score. It also captured the expected JSON schema, thresholds, pair rules, and mapping constraints before export work began. This prevented a superficially successful browser demo from silently changing the project semantics.

### 7.2 B1: first YOLO-World export attempt failed

**Hypothesis.** The exact configured YOLO-World detector could be exported through the installed official Ultralytics route.

**Failure.** The exporter reached `torch.onnx.export` but failed before producing an ONNX file: the legacy PyTorch ONNX symbolic path did not support the detector's `adaptive_max_pool2d` with an output size of `[3,3]` that was not a factor of the input dimensions. This phase correctly made no ONNX Runtime or browser claim.

**Next step.** Isolate only the export blocker without changing weights, vocabulary, NMS policy, or production code.

### 7.3 B1.1/B1.2: static detector export and the raw-parity investigation

**Implementation.** The export experiment replaced only three in-memory `AdaptiveMaxPool2d((3,3))` operations with an exact static regional-max expression for the observed 80×80, 40×40, and 20×20 feature-map shapes. The replacement was export-only; the checkpoint was not changed. It produced a static batch-one FP32 640×640 model with one `[1,153,8400]` output: four `xywh` channels plus 149 class-score channels.

**Initial limitation.** B1.1 appeared to show enormous raw box discrepancies—maximum absolute error roughly 632.60–634.09—despite nearly identical decoded detections. It was conservatively classified as export succeeded but parity failed.

**Resolution.** B1.2 showed that the problem was a comparison-order bug. Ultralytics NMS converts the first four raw channels from `xywh` to `xyxy` **in place**. The experiment had compared saved pre-NMS data with an array after that mutation. With correctly controlled pre-NMS comparisons, the fresh configured PyTorch detector exactly reproduced the saved reference, and Python ONNX Runtime achieved raw parity within the documented tolerance and complete decoded parity on the three fixed-square inputs.

**Boundary preserved.** This was not proof of equivalence to the normal server predictor's potentially rectangular letterbox route. The exported graph accepts only `[1,3,640,640]`; its validity depends on static-square preprocessing.

### 7.4 C1: detector in real Chrome

**Objective.** Test actual browser model execution, not merely model loading or Node.js execution.

**Result.** In Chrome 155 with ONNX Runtime Web 1.24.2, the static detector completed WASM inference on all three fixtures and an independent WebGPU run on one fixture. The browser produced the required `[1,153,8400]` tensor. The 54,078,927-byte model loaded as a self-contained graph, and no unsupported-operator error occurred in those runs.

**Limitation.** Canvas-based browser resizing was not bit-identical to Python/OpenCV linear resizing. Therefore raw browser-versus-Python comparisons included input differences; one near-threshold detection differed for the first image. The result was correctly classified as **browser inference verified with limitations**, not exact detector parity.

### 7.5 B2: CLIP browser feasibility

**Objective.** Export the exact OpenCLIP visual behavior and determine if it was usable in a browser.

**Implementation.** A visual-only wrapper exported the ViT-B/32 visual tower to ONNX with dynamic batch dimensions: float32 `[N,3,224,224] → [N,512]`. The model preserves raw, unnormalized features. Export required disabling a PyTorch export-time multi-head-attention fast path so equivalent attention mathematics could lower to standard ONNX operators; this did not change production weights or preprocessing.

**Numerical result.** Python ONNX Runtime versus PyTorch for 100 features recorded maximum absolute error `1.5437603e-05`, mean absolute error `1.5139201e-06`, and cosine similarity `0.9999999999896285`.

**Browser result and failure mode.** Chrome WASM ran saved Python tensors accurately, but a 68-crop batch took 32,744.4 ms and 100 crops took 45,770.6 ms. The 335.48 MiB FP32 artifact and long WASM runs meant model compatibility was proven, but interactive performance was not.

### 7.6 B3: relationship MLP browser feasibility

**Objective.** Establish whether the existing four-input MLP—including int64 label inputs—was a browser blocker.

**Result.** The 821,278-byte dynamic ONNX model ran in Chrome. Python parity for 100 deterministic rows recorded maximum logit error `1.5258789e-05`; the browser WASM run for 68 rows took 3.5 ms after reuse. The MLP was therefore not the bottleneck. The model retained its `BigInt64Array`/int64 label semantics rather than substituting one-hot labels or a different architecture.

### 7.7 D1: optimize the real bottleneck without hiding decision drift

The D1 fixture contained 68 directed candidate pairs made from nine recorded detections, but reverse directions can share a union crop. Exact crop-bound deduplication reduced the visual inputs from 68 directed tensors to 26 unique preprocessed tensors. This was a semantic-preserving optimization: CLIP runs once per exactly matching crop bound and the resulting 512-D feature is copied back to each corresponding directed row.

The experiment compared formats and providers. FP32 preserved downstream decisions. INT8 was much smaller but was rejected as a transparent optimization because it changed six of 68 browser predicate and retention decisions in the recorded fixture. The repaired FP16 graph was 176,081,784 bytes—49.9% smaller than FP32—and later preserved all 68 predicate choices, retain/remove outcomes, and threshold-side outcomes in the tested fixture. It became the WebGPU-preferred production candidate, with WASM retained as fallback. It was not described as universally equivalent across devices or images.

### 7.8 E1: complete browser pipeline proof

**Implementation.** The E1 browser path executed image decoding, static YOLO inference, directed pair generation, 20% union expansion, Python-compatible ties-to-even crop-bound rounding, exact crop deduplication, FP16 CLIP with WebGPU preference/WASM fallback, FP32 relationship MLP, stable softmax, class-50 removal, `>=0.30` filtering, and directed-triple deduplication. It returned the existing `{image, objects, relationships}` JSON schema without diagnostic fields.

**Result.** Chrome completed seven actual analyses: five distinct images, a WebGPU repeat without reload, and a WASM fallback. A cached WebGPU repeat of the main fixture took 5,394.0 ms; the forced WASM fallback completed in 38,265.8 ms. A 170-pair fixture was measured at about 9.1 seconds after sessions were cached. This proved that all three real models could run locally in the browser.

**Limitation.** It did not prove that browser results reproduced Python scene graphs across fixtures. No candidate cap, threshold adjustment, model substitution, or mapping change was used to conceal differences.

### 7.9 E2: parity investigation did not pass

E2 replaced simple Canvas resizing with explicit browser CPU kernels and compared five fixtures against saved references. It improved control but did not establish strict parity:

- YOLO input exact fractions were 91.29%–91.88% for four fixtures and 100% for one, with each remaining difference at most one 8-bit channel level after normalization.
- Browser static detections matched counts for several fixtures but differed for `fixture-1` (Python 11 versus browser 10) and had label mismatches in the moderate fixture.
- Browser CLIP input MAE ranged from `0.029148` to `0.123045`, with maximum absolute differences from `1.500777` to `2.776437`.

The conclusion remains an accepted limitation of the deployed portfolio route: **browser/Python numerical parity was not achieved**. The browser system is a real local inference implementation, but not an assertion that every Python reference graph will recur exactly.

## 8. Production React integration, validation, and deployment

### F1: connect the existing UI to local inference

F1 retained the public import boundary `analyzeImage(file, onProgress)` but changed it from a FastAPI call to `browserInference.js`. Existing graph, overlay, selection, inspector, and scene-schema components were intentionally not replaced. The implementation caches metadata and sessions, prevents concurrent analyses with an in-flight guard, closes temporary image bitmaps, uses WebGPU for CLIP when available, and falls back to WASM if initialization fails.

### F2: full UI regression

The F2 real-Chrome matrix completed five image analyses in a no-reload session without page exceptions. For all successful cases, object overlays and React Flow node counts matched. The subsequent repeated upload cleared prior state and reused fetched model/runtime/metadata URLs rather than fetching them per image. Tests also exercised image-to-graph selection, inspector details, graph visibility, downloads, built-preview operation, forced-WASM behavior, and error paths as documented in the retained report.

F2 did not claim the empty-relationship rendering path was passed because every matrix fixture produced at least one relationship.

### F3A: external asset boundary and worker investigation

The production bundle deliberately excludes the roughly 220 MiB ONNX model payload. It resolves all assets from a versioned URL prefix. A separate-origin static-host simulation passed the five-fixture production regression with CORS, MIME types, dynamic ONNX Runtime imports, browser caching, and no FastAPI analysis traffic.

CPU-heavy image decode, preprocessing, NMS, crop construction, and scene building moved to a Web Worker using `OffscreenCanvas`. This reduced synchronous non-ORT work on the UI thread, but it did **not** eliminate the UI freeze: the maximum animation-frame gap was 6.128 seconds, essentially unchanged from F2's 6.11 seconds. Main-thread ONNX Runtime/session/provider work remained dominant. The final documentation therefore does not claim that responsiveness has been solved.

### F3B: immutable Vercel Blob assets and public frontend

The final asset publication contract placed 13 manifest-controlled files under the immutable `scene-graph-v1/` prefix in Vercel Blob. The manifest records byte counts, SHA-256 values, MIME types, and one-year immutable cache directives. The required Vite build-time base URL points to that public prefix; the browser contains no Blob write credential. The upload tool validates local hashes and refuses to overwrite an existing mismatched immutable object.

The current public application is served at [scene-graph.vercel.app](https://scene-graph.vercel.app/). It downloads application code from Vercel and model/runtime assets from Vercel Blob, then runs inference locally in the browser.

## 9. Final outcomes and limitations

### Outcomes

1. A Visual Genome-informed scene-graph pipeline progressed from data exploration and notebooks to reusable Python inference, an API/UI application, and a public browser-only frontend.
2. The final client preserves the core detector vocabulary, relationship model, directed-pair semantics, geometry inputs, predicate decoding, thresholds, JSON schema, and interactive visualization model.
3. A real browser executed the exported YOLO-World detector, CLIP encoder, and relationship MLP. The public frontend no longer uses FastAPI for image analysis.
4. Immutable Blob hosting separates large versioned models/runtime binaries from the Vite bundle and permits cacheable external assets.

### Accepted limitations

1. **No universal Python/browser numerical-parity claim.** E2 measured unresolved preprocessing and runtime/output differences.
2. **No claim that UI freezing is eliminated.** The worker-based CPU refactor left a measured approximately six-second worst animation-frame gap in its test environment.
3. **No claim of complete mobile verification.** The evidence focuses on local Chrome; Edge, mobile, low-memory devices, and broad hardware/provider coverage remain incomplete.
4. **First use is large and slow.** The manifest totals 312,280,662 bytes, including approximately 220 MiB for the three ONNX models, before considering runtime downloads/caching conditions.
5. **WASM is a functional fallback, not equivalent performance.** It can be substantially slower than a successful WebGPU path.
6. **The Python/FastAPI reference remains retained, while the Docker/OCI deployment route was retired on October 22, 2026.** The retained backend is useful for local research and regression reference; it is not the current production inference path.

## 10. Consolidated historical record

### Training and model-contract decisions

- The relationship model was trained against Visual Genome's 150 object categories and 50 predicate categories, with a 51-way output that includes a no-relationship class. The live checkpoint convention is authoritative: model classes `0..49` map to predicate IDs `1..50`, while class `50` means `no relationship`. This corrects an early workflow document that described class `0` as no relationship; [`scene_graph/mapping.py`](../scene_graph/mapping.py) enforces the verified convention.
- The selected full-fusion checkpoint followed 30-epoch training and combines raw 512-D CLIP visual features, two 64-D object-label embeddings, and 8-D geometry. The core lesson was that label-ID domains—YOLO order, Visual Genome IDs, embedding indices, predicate IDs, and model outputs—must not be conflated.
- The Python pipeline was deliberately validated incrementally: cached features, ground-truth objects, detector outputs, directed pairs, batched visual features, postprocessing, and finally scene JSON. The retained Python tests preserve those reference checks without requiring the retired deployment path.

### Browser export, parity, and optimization decisions

- The first official YOLO-World ONNX export failed on non-factor `adaptive_max_pool2d([3,3])`. An export-only static regional-max replacement for the three fixed `640×640` feature-map shapes unblocked a self-contained detector graph without changing the checkpoint, vocabulary, or runtime Python source.
- An apparent large raw-box discrepancy was later traced to Ultralytics NMS mutating raw box channels from `xywh` to `xyxy` in place. Under correctly ordered, identical-input comparison, the static ONNX detector matched the freshly configured export-state reference; this result does not prove equivalence to the Python server's normal rectangular-letterbox route.
- Chrome executed the detector, visual encoder, and relationship MLP with ONNX Runtime Web. Saved Python tensors established close component runtime parity, but browser image decode, resize, crop, and provider behavior prevented a universal Python/browser numerical-parity claim.
- Exact union-crop deduplication is retained because it materially reduces repeated CLIP work. INT8 visual CLIP changed predicate and retention decisions on the recorded fixture and was rejected as a transparent substitution. The repaired FP16 visual graph halved model bytes and preserved fixture decisions, but remains a WebGPU-preferred optimization with an FP32 fallback rather than an unconditional equivalence claim.

### Retired Docker/OCI deployment record

The former OCI design packaged the retained FastAPI/Python reference pipeline in a single-worker, CPU-only container. It selected architecture-specific CPU dependency pins, baked Visual Genome metadata/checkpoints and public CLIP artifacts into the image, forced offline cache use, and avoided multi-worker memory duplication. Measurements and the deployment audit showed why it was not adopted as the public path: model loading could take roughly 45–120 seconds, synchronous inference roughly 6–9 seconds, and each additional worker would duplicate a multi-gigabyte ML process. The public Vercel architecture eliminates that server dependency by moving inference to the visitor's browser.

On October 22, 2026, the tracked root Dockerfile, Docker ignore rules, deployment requirements, and Docker-only prefetch helpers were retired. This was a repository-simplification decision, not deletion of the Python reference backend or ML pipeline. Untracked OCI deployment work remains outside this change and requires separate approval before it is altered or revived.

## 11. Retained source trail

This history is the consolidated record for the retired reports. The following retained source, test, script, and manifest files provide implementation-level evidence:

- [Python reference pipeline](../scene_graph/pipeline.py)
- [Mapping safeguards](../scene_graph/mapping.py)
- [Relationship model definition](../notebook/models/relationship/model.py)
- [Retained Python reference tests](../tests/)
- [Browser inference service](../frontend/src/services/browserInference.js)
- [Browser preprocessing worker](../frontend/src/services/inferenceWorker.js)
- [Detector export/parity scripts](../experiments/browser-feasibility/)
- [Immutable asset manifest](../experiments/browser-feasibility/scene-graph-v1-asset-manifest.json)

For an implementation-oriented explanation of the final public system, see [Architecture and Reproducibility](ARCHITECTURE_AND_REPRODUCIBILITY.md).