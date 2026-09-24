# Scene Graph Explorer
# Post-Training Full-Stack Development Log

This file is the permanent chronological record of all work performed after
relationship-model training.

It must be continuously updated whenever code, architecture, configuration,
dependencies, APIs, frontend components, inference logic, testing procedures,
or deployment configuration are changed.

The purpose of this file is to preserve:
- what was changed
- why it was changed
- which files were changed
- what was tested
- test results
- known issues
- decisions made
- deployment progress

---

# Project Goal

Build a complete Scene Graph Explorer application.

Final pipeline:

Image
→ YOLO-World object detection
→ detected objects + bounding boxes
→ object-pair generation
→ custom relationship prediction model
→ relationship post-processing
→ scene graph construction
→ FastAPI
→ JSON
→ React
→ image + interactive graph + inspector + search

---

# Current State

Relationship model training has been completed.

The trained model is available locally.

The Visual Genome SDK is implemented.

YOLO-World inference has been tested.

Relationship inference has been tested.

The next task is to convert the working notebook-level pipeline into a
clean reusable Python inference pipeline suitable for FastAPI integration.

---



---

# Development Log

## Entry 1 - Reusable Scene Graph Inference Pipeline (Phase A)

Date: 2026-09-11

Objective:
Convert the working notebook-level inference (notebook/05_testing.ipynb)
into a clean, reusable Python inference pipeline component that the later
phases (FastAPI, React) will build on.  No FastAPI / React / model changes
in this entry.

Files created:
- scene_graph/__init__.py       package exports (SceneGraphPipeline etc.)
- scene_graph/mapping.py        index conversions (VG id <-> embedding id,
                                model class -> predicate name); the single
                                source of truth for all label conversions
- scene_graph/detector.py       YoloWorldDetector + Detection dataclass
                                (wraps ultralytics YOLO-World, set_classes
                                with the 149-name VG vocabulary)
- scene_graph/features.py       PairFeatures + extract_pair_features
                                (union crop expansion 0.20, 8-D normalized
                                geometry, frozen CLIP ViT-B/32 512-D feature)
- scene_graph/predictor.py      RelationshipPredictor (CachedRelationshipModel
                                wrapper: 51 logits -> argmax + softmax)
- scene_graph/pipeline.py       SceneGraphPipeline + PairPrediction +
                                SceneGraphResult
- tests/test_pipeline_local.py  end-to-end local test on a real image
- notebook/__init__.py          notebook/ marked as a package so the module
                                code under notebook/models/... is importable
                                from the repository root
- notebook/models/__init__.py   models/ marked as a package

Files modified: none (all pre-existing repository files untouched).
Files deleted: none kept (temporary test bootstrap files removed after use).

Implementation details:
- SceneGraphPipeline(checkpoint_path, yolo_model_path, dataset_root,
  image_root=None, device="auto", conf_threshold=0.25, expansion=0.20,
  skip_same_label=True)
- analyze(image: PIL | path | ndarray) -> SceneGraphResult
  * YOLO-World detection (conf 0.25) with the Visual Genome vocabulary
    (149 names after removing "background")
  * detections filtered to the 150 VG classes via the label string
  * ordered object pairs (i != j; same-label pairs skipped, matching the
    working notebook test)
  * per pair: union crop (expansion 0.20) -> CLIP ViT-B/32 preprocess
    (224x224) -> 512-D visual feature; 8-D normalized geometry within the
    union crop; CachedRelationshipModel -> 51 logits -> argmax + softmax
  * class 50 ("no relationship") -> result.unrelated; other classes ->
    Scene.relationships with predicate_index = model class + 1 (restoring
    the SDK 1-based VG predicate id convention)
- Reuses existing repository structures by reference (no copies):
  datasets.visual_genome (Scene, SceneObject, Relationship, BoundingBox,
  VisualGenomeLoader), notebook/models/relationship/{model,encoder,geometry}.py.

CRITICAL INDEXING DECISION (latent doc/code mismatch resolved):
- The task brief and scene_graph_relationship_training_workflow.md state
  "class 0 = no relationship".  Verified against the trained model, this
  is WRONG for this checkpoint.  Evidence:
  * cache shards store negative (no-relationship) samples with internal
    predicate id 51, i.e. target = 51 - 1 = 50; ~50 percent of all cached
    targets are class 50;
  * notebook 05 cell 57 builds idx_to_predicate = {int(idx)-1: name} and
    then sets idx_to_predicate[50] = "no relationship"; cell 61 buckets
    class 50 as unrelated;
  * the working end-to-end output shows class 50 predicted for
    "person -> no relationship -> car".
- Therefore model class 50 = "no relationship" and classes 0..49 = VG
  predicates 1..50 (class 0 = "above").  VG predicate id 50 ("worn by")
  maps to model class 49.  All conversions are centralized in
  scene_graph/mapping.py with this convention documented; do NOT flip it.

Testing:
- Command:  .venv\Scripts\python.exe tests\test_pipeline_local.py
- The interactive tooling here caps command runtime at ~30 s while the
  full run takes ~70-90 s on CPU (heavy imports + 68 pair forwards), so
  the test was launched detached and its --log-file output polled.
- The test forces HF_HUB_OFFLINE=1 (all weights are local/cached).
- Result: PASS
  * Index asserts: 95->94, 114->113, 14->13, 30->29; class 35 = "riding",
    class 49 = "worn by", class 50 = "no relationship".
  * Image: datasets/visual_genome/VG_100K_2/1.jpg (800x600)
  * Objects detected: 9 (person x2, car, bike, vehicle, shoe, sidewalk,
    tree x2) - identical to notebook 05 output
  * Pairs with relationship: 17 | Pairs with no relationship: 51 (68 pairs)
  * person -> riding -> bike (21.79%) - same flagship result as the
    reference notebook; full output matches notebook 05 exactly.

Bugs discovered / fixed:
- Indentation error in scene_graph/pipeline.py when method blocks were
  inserted at module level during file creation; corrected.
- Fresh Python processes spent significant time on HF-hub network checks;
  fixed by forcing HF_HUB_OFFLINE=1 in the test (weights cached).
- --log-file initially used a fully buffered handle, delaying output;
  switched to line buffering (buffering=1).

Remaining issues:
- Cold-start is slow on CPU (~30-60 s including imports); acceptable now,
  but the FastAPI phase must load models once at startup.
- No GPU in this environment; all inference is CPU.
- Train/val cache provenance note from the study phase (both caches drawn
  from official train-split images) is unchanged and does not affect
  inference correctness.

Architectural decisions:
- New top-level package scene_graph/ (outside notebook/) referencing the
  existing SDK/model modules instead of duplicating them.
- notebook/ and notebook/models/ became packages (empty __init__.py)
  purely to make existing module code importable from the repo root.
- analyze() returns SceneGraphResult whose .scene is the SDK Scene with
  only real relationships; "no relationship" pairs are exposed separately
  so summaries can report both counts.

Next step:
- Phase B - scene graph construction / post-processing (JSON-serializable
  graph output) per the approved phase plan.  Not implemented in this entry.

---

## Entry 1a - Appendix: Function Reference for Entry 1

Date: 2026-09-11

This appendix logs EVERY function and class defined in the files created in
Entry 1, with exact signatures and behavior, and explains how the whole
pipeline functions end to end.

===============================================================
1. scene_graph/mapping.py  (pure functions, no model loading)
===============================================================

Constants defined:
- NO_RELATIONSHIP_CLASS = 50      model output class meaning "no relationship"
- NO_RELATIONSHIP_PREDICATE = "no relationship"
- NUM_OBJECT_CLASSES = 150        VG object vocabulary size
- NUM_MODEL_CLASSES = 51          predicates 0..49 + no-relationship class

Functions:

- vg_label_id_to_model_id(vg_label_id: int) -> int
  Converts a 1-based Visual Genome object label id to the 0-based embedding
  id the model expects.  Implemented as `return int(vg_label_id) - 1`.
  Example: person VG 95 -> 94, bike VG 14 -> 13, car VG 30 -> 29.

- model_id_to_vg_label_id(model_id: int) -> int
  Inverse of the above: `return int(model_id) + 1`.  Used if a model id must
  be shown back as a VG dictionary id.

- model_class_to_vg_predicate_id(model_class: int) -> int
  Converts a model output class 0..49 to its 1-based VG predicate id:
  `return int(model_class) + 1`.  Valid ONLY for real predicates; class 50
  has no VG predicate id (it is the no-relationship class).  Used by the
  pipeline when filling Relationship.predicate_index.

- build_idx_to_predicate(vg_dict: Dict) -> Dict[int, str]
  Builds the model output class (0..50) -> predicate-name decode table from
  the parsed VG-SGG-dicts.json.  Two steps:
    1. {int(idx) - 1: name for name, idx in vg_dict["predicate_to_idx"].items()}
       (VG predicate ids 1..50 become model classes 0..49)
    2. idx_to_predicate[NO_RELATIONSHIP_CLASS] = NO_RELATIONSHIP_PREDICATE
       (adds class 50 = "no relationship")
  Resulting table (verified): 0=above, 1=across, ..., 35=riding, 46=watching,
  47=wearing, 49=worn by, 50=no relationship.

- build_label_name_to_vg_id(vg_dict: Dict) -> Dict[str, int]
  Builds {label string: 1-based VG id} from vg_dict["label_to_idx"].
  Example: {"person": 95, "bike": 14, "car": 30, ...} (150 entries).

- build_yolo_vocabulary(vg_dict: Dict) -> List[str]
  Returns the 149 YOLO-World class names: the VG label list minus
  "background".  This exact list is what gets passed to
  YOLOWorld.set_classes(...) (same as notebooks 02 / 05).

===============================================================
2. scene_graph/detector.py
===============================================================

Classes:

- @dataclass Detection
  One YOLO-World detection.  Fields:
    detection_index: int          loop position in the detections list
    label: str                    class name string (resolved via result.names)
    confidence: float             YOLO confidence, 0..1
    xyxy: tuple                   (x1, y1, x2, y2) original-image pixels
    bbox: BoundingBox             SDK box, center format (cx, cy, w, h)

- class YoloWorldDetector
  Methods:

  - __init__(model_path, class_names: List[str], conf: float = 0.25,
             device: str = "auto")
    Loads YOLOWorld(model_path) and immediately calls set_classes(class_names)
    with the 149 VG names.  Stores conf threshold and resolved device.

  - _resolve_device(device: str) -> str   (staticmethod)
    "auto" resolves to "cuda" if torch.cuda.is_available() else "cpu";
    anything else is passed through unchanged.

  - detect(image) -> List[Detection]
    Runs self.model.predict(image, conf=self.conf, device=self.device,
    verbose=False).  Iterates result.boxes; for each detection i reads
      class_id  = int(boxes.cls[i])      (0-based index into our vocabulary,
                                        NOT a VG id)
      confidence= float(boxes.conf[i])
      x1, y1, x2, y2 = boxes.xyxy[i]     (original-image pixels)
      label     = result.names[class_id]
    Converts xyxy -> center format BoundingBox(cx=(x1+x2)/2, cy=(y1+y2)/2,
    width=x2-x1, height=y2-y1) and appends a Detection.  Returns the list.

===============================================================
3. scene_graph/features.py
===============================================================

Classes / functions:

- @dataclass PairFeatures
  Feature bundle for one (subject, object) pair:
    visual      torch.Tensor  (1, 512) float32 CLIP feature
    geometry    torch.Tensor  (1, 8)   float32 normalized geometry
    union_crop  PIL.Image     the cropped union region (debugging/vis)

- build_geometry(subject_bbox: BoundingBox, object_bbox: BoundingBox,
                 union_box: BoundingBox) -> torch.Tensor
  Builds the 8-D geometry vector:
    [sub_cx, sub_cy, sub_w, sub_h, obj_cx, obj_cy, obj_w, obj_h]
  Steps per box: bbox_relative_to_crop(box, union_box) re-expresses the box
  relative to the union crop origin; normalize_bbox(relative, union_box)
  divides by the union width/height so every value is roughly in [0, 1].
  Returns a float32 tensor of shape (8,).

- build_union_crop(image, subject_bbox, object_bbox, expansion=0.20)
  -> (union_box: BoundingBox, crop: PIL.Image)
  Computes the expanded union box via get_union_bbox(subject, object,
  expansion=expansion) - both boxes are expanded by `expansion` then their
  corner-union is taken - and crops that region from the PIL image via
  crop_union_region (clamped to image bounds, raises on zero-area crop).

- extract_pair_features(image, subject_bbox, object_bbox, visual_encoder,
                        device="cpu", expansion=0.20) -> PairFeatures
  The per-pair feature extractor; mirrors notebook 05 exactly:
    1. union crop (build_union_crop)
    2. geometry   (build_geometry)
    3. visual_input = visual_encoder.preprocess(crop).unsqueeze(0).to(device)
       -> (1, 3, 224, 224)
    4. with torch.no_grad(): visual = visual_encoder(visual_input)
       -> (1, 512) float32 (frozen open_clip ViT-B/32 encode_image)
  Returns PairFeatures(visual=(1,512), geometry=(1,8), union_crop=crop).

===============================================================
4. scene_graph/predictor.py
===============================================================

Classes:

- @dataclass Prediction
    class_id: int      model output class 0..50
    predicate: str     decoded name ("no relationship" for class 50)
    confidence: float  softmax probability of the predicted class

- class RelationshipPredictor
  Methods:

  - __init__(model: CachedRelationshipModel, idx_to_predicate: dict,
             device: str = "cpu")
    Stores the already-loaded model (must be in eval mode - the pipeline
    guarantees this), the decode table, and the torch device.

  - predict(visual, subject_label, object_label, geometry) -> Prediction
    @torch.no_grad()
    Runs the exact model forward call:
      logits = model(visual, subject_label, object_label, geometry)
      probabilities = softmax(logits, dim=1)
      class_id = argmax(probabilities, dim=1)   (int)
      confidence = probabilities[0, class_id]   (float)
      predicate = idx_to_predicate[class_id]
    Expected input shapes: visual (1,512), labels (1,) int64 0-based
    embedding ids, geometry (1,8).

===============================================================
5. scene_graph/pipeline.py
===============================================================

Classes:

- ImageSource = Union[str, os.PathLike, PIL.Image.Image]  (type alias)

- @dataclass PairPrediction
  A prediction joined with its SceneObject endpoints (so callers never have
  to look objects up themselves):
    subject: SceneObject
    object: SceneObject
    predicate: str    e.g. "riding" or "no relationship"
    class_id: int     raw model class
    confidence: float

- @dataclass SceneGraphResult
    scene: Scene                      SDK Scene: image, info, objects,
                                      relationships (real predicates only)
    related: List[PairPrediction]     pairs predicted as a real predicate
    unrelated: List[PairPrediction]   pairs predicted class 50

- class SceneGraphPipeline
  Methods:

  - __init__(checkpoint_path, yolo_model_path, dataset_root,
             image_root=None, device="auto", conf_threshold=0.25,
             expansion=0.20, skip_same_label=True)
    Loads everything once, in order:
      1. self.device = _resolve_device(device)
      2. VisualGenomeLoader(dataset_root, image_root) - only for metadata;
         from its dictionary builds label_name_to_vg_id, idx_to_predicate,
         vg_classes (149 names)
      3. YoloWorldDetector(yolo_model_path, vg_classes, conf, device)
      4. ImageEncoder("ViT-B-32", "openai", device) - frozen CLIP
      5. CachedRelationshipModel(150, 51, 512, 64, 32, 256, dropout=0.30)
         then torch.load(checkpoint)["model_state_dict"] -> load_state_dict,
         .to(device), .eval()   (architecture matches the checkpoint exactly)
      6. RelationshipPredictor wrapping the model
      7. stores expansion (0.20) and skip_same_label flag
    NOTE: VisualGenomeLoader opens VG-SGG.h5 read-only; no annotations are
    used at inference time - only the dictionaries.

  - analyze(image: ImageSource) -> SceneGraphResult
    The single public entry point.  Steps:
      1. image = _to_pil_image(image)  (loads path, converts to RGB)
      2. detections = detector.detect(image)
      3. detections -> SceneObjects: label string looked up in
         label_name_to_vg_id; unknown labels skipped.  SceneObject(
         object_index=detection_index, label_index=vg_id (1-based, SDK
         convention), label, bbox, confidence)
      4. double loop over objects -> ordered pairs; skip i == j; skip pairs
         with identical labels if skip_same_label; call _predict_pair
      5. bucket by class: class 50 -> unrelated, else related
      6. related -> Relationship(relationship_index=running index,
         subject, object, predicate_index=class_id + 1 (1-based VG id),
         predicate=name, confidence) ; Scene(image, info=ImageInfo(w, h),
         objects, relationships)
      7. return SceneGraphResult(scene, related, unrelated)

  - _predict_pair(image, subject: SceneObject, obj: SceneObject)
    -> PairPrediction   (private)
    Per pair: extract_pair_features(...) -> features; label tensors built as
    subject.label_index - 1 / obj.label_index - 1 (1-based VG id -> 0-based
    embedding id); predictor.predict(...) -> Prediction; wraps into
    PairPrediction with the object references.

  - _resolve_device(device: str) -> str   (staticmethod, same logic as the
    detector helper: "auto" -> cuda if available else cpu)

  - _to_pil_image(image: ImageSource) -> PIL.Image  (staticmethod)
    str/PathLike -> Image.open(path).convert("RGB");
    PIL image    -> .convert("RGB");
    ndarray      -> Image.fromarray(image).convert("RGB").

===============================================================
6. tests/test_pipeline_local.py
===============================================================

Functions:

- main() -> None
  The test driver.  Steps:
    1. argparse: --image (default VG_100K_2/1.jpg), --checkpoint (default
       best_full_fusion_30epoch.pt), --yolo-model (default yolov8s-world.pt),
       --log-file (optional; when set, stdout/stderr are redirected to the
       file with line buffering=1 so detached runs can be polled)
    2. sets HF_HUB_OFFLINE=1 / TRANSFORMERS_OFFLINE=1 (weights are local)
    3. pure-function index asserts: 95->94, 114->113, 14->13, 30->29
    4. constructs SceneGraphPipeline
    5. decode-table asserts: class 35="riding", class 49="worn by",
       class 50="no relationship"
    6. result = pipeline.analyze(image)
    7. prints IMAGE / OBJECTS / RELATIONSHIP SUMMARY / PREDICTED
       RELATIONSHIPS / NO RELATIONSHIP sections (human-readable only) and
       "PASS"

Module-level constants: REPO_ROOT, DATA_DIR, MODEL_DIR, DEFAULT_IMAGE,
CHECKPOINT_PATH, YOLO_MODEL_PATH.

===============================================================
7. HOW THE PIPELINE FUNCTIONS (end-to-end walkthrough)
===============================================================

Construction (once, before any request):

  SceneGraphPipeline.__init__
    |- VisualGenomeLoader(dataset_root)      reads VG-SGG-dicts.json only
    |                                        (H5 opened but unused at runtime)
    |- mapping.build_*                       3 lookup tables
    |- YoloWorldDetector(...).set_classes    YOLO-World re-headed to the
    |                                        149 VG classes (open-vocabulary)
    |- ImageEncoder("ViT-B-32","openai")     frozen CLIP, no gradients
    |- CachedRelationshipModel + checkpoint  subject_embedding (150,64),
    |                                        object_embedding (150,64),
    |                                        geometry Linear(8->32)+ReLU,
    |                                        classifier 672->256->ReLU->
    |                                        Dropout(0.30)->51
    \- RelationshipPredictor wrapper

Per image (analyze):

  image (PIL/path/ndarray)
    |  _to_pil_image -> RGB PIL
    v
  YoloWorldDetector.detect                    N detections
    |  xyxy -> center BoundingBox per box
    v
  label string -> label_name_to_vg_id         SceneObject list
    |  (label_index = 1-based VG id)          (unknown labels dropped)
    v
  ordered pairs (i != j, optional same-label skip)   ~N*(N-1) pairs
    |
    |  for each pair:
    v
  extract_pair_features
    |- get_union_bbox(sub, obj, expansion=0.20)   union box, pixels
    |- crop_union_region(image, union_box)        PIL crop
    |- bbox_relative_to_crop + normalize_bbox     geometry (8,) in [0,1]
    |- preprocess(crop) -> (1,3,224,224)
    \- CLIP encode_image -> visual (1,512)
    |
    v
  label tensors: vg_id - 1                    (1-based -> 0-based)
    v
  RelationshipPredictor.predict
    |- CachedRelationshipModel(visual, sub_id, obj_id, geometry)
    |- logits (1, 51)
    |- softmax -> argmax -> class_id + confidence
    \- idx_to_predicate[class_id] -> predicate string
    v
  bucketing: class 50 -> unrelated | else -> related
    v
  related -> Relationship(predicate_index = class_id + 1, confidence)
    v
  Scene(image, ImageInfo(w,h), objects, relationships)
    v
  SceneGraphResult(scene, related, unrelated)

Data-shape cheat sheet:
  YOLO boxes.xyxy          (N, 4)  original pixels
  BoundingBox              (cx, cy, w, h) original pixels, center format
  CLIP preprocess          (1, 3, 224, 224)
  visual feature           (1, 512)
  geometry                 (1, 8)   normalized to union crop
  label tensors            (1,) int64, 0-based embedding ids (vg_id - 1)
  logits                   (1, 51)  class 50 = no relationship
  confidence               softmax probability, 0..1

Two indexing conversions happen per pair and nowhere else:
  vg object id -> model id :  vg_id - 1          (pipeline._predict_pair)
  model class -> vg pred id:  class_id + 1       (pipeline.analyze)
All other conversions are precomputed once in mapping.py.

---

## Entry 2 - PHASE B: Scene Graph Construction + Post-Processing

Date: 2026-09-11

1. Objective
Convert raw relationship predictions into a clean, application-ready scene
graph suitable for FastAPI/React later: no-relationship removal, confidence
threshold, duplicate removal, stable object/relationship ids, and a
JSON-serializable output.  Phase A behavior, model, checkpoint and mappings
are NOT changed.

2. Files inspected (before any change)
- scene_graph/{mapping,detector,features,predictor,pipeline}.py
- datasets/visual_genome/{models,loader,__init__}.py
- notebook/models/relationship/{model,predictor}.py
- tests/test_pipeline_local.py
- Confirmed from code: Scene/SceneObject/Relationship/BoundingBox fields;
  object_index = YOLO detection index; SceneObject.label_index = 1-based VG
  id; Relationship.predicate_index = model class + 1 (1-based VG id);
  no-relationship = model class 50; analyze() returned SceneGraphResult(
  scene, related, unrelated) with related = ALL non-class-50 predictions.

3. Files created
- scene_graph/postprocessing.py
- tests/test_postprocessing_local.py

4. Files modified (minimal, additive)
- scene_graph/pipeline.py
    * SceneGraphPipeline gained relationship_threshold (default 0.30)
    * analyze() step 4 now runs the post-processing funnel before building
      Relationship objects; scene.relationships = FINAL relationships only
    * SceneGraphResult gained optional field stats: PostprocessStats and a
      to_dict() method (JSON-serializable graph)
- scene_graph/__init__.py
    * re-exports the new post-processing symbols
- tests/test_pipeline_local.py: NOT modified (still the Phase A test; it
  now runs under the new default threshold)

5. Classes/functions added (see appendix below for full signatures)
- PostprocessStats (dataclass, funnel counters + to_dict)
- remove_no_relationship(predictions) -> (kept, removed_count)
- filter_by_confidence(predictions, threshold) -> (kept, removed_count)
- remove_duplicates(predictions) -> (kept, duplicates_count)
- postprocess_relationships(predictions, threshold) -> (final, stats)
- object_id(object_index) -> "obj_N"
- relationship_id(relationship_index) -> "rel_N"
- SceneGraphResult.to_dict() -> dict

6. Data flow (Phase B layer inserted at step 4 of analyze)
  raw pair predictions (68 for the reference image)
    -> remove_no_relationship          (class 50 discarded)   -51
    -> filter_by_confidence(0.30)                             -1
    -> remove_duplicates (exact triples)                      -0
    -> final relationships (16) -> Scene.relationships -> to_dict()

7. No-relationship handling
Model class 50 ("no relationship") predictions NEVER become Relationship
objects; they are discarded by remove_no_relationship and are additionally
reported as result.unrelated for summaries.  The mapping is untouched:
class 50 stays class 50; classes 0..49 = VG predicates 1..50; nothing was
renamed, shifted, or converted to class 0.

8. Confidence threshold
Configurable via SceneGraphPipeline(relationship_threshold=...), default
0.30.  It is a STARTING VALUE, not a validated optimum (flagship prediction
is 21.79 percent and is filtered at 0.30 - intentional, model-output
driven).  The check uses the existing relationship-model softmax
confidence; no new confidence calculation was invented.

9. Duplicate handling (conservative)
A duplicate is the exact triple (subject.object_index, predicate,
object.object_index); first occurrence wins.  Inverse relationships (e.g.
person -> near -> car AND car -> near -> person) are NOT considered
duplicates - different object pairs are legitimate distinct edges.

10. Object ID design
obj_{object_index} where object_index = YOLO detection index (deterministic
for a given image).  Same id usable by image bbox, graph node, relationship
subject/object, inspector.  No UUIDs introduced.

11. Relationship ID design
rel_{relationship_index} where the index is the position in the FINAL
post-processed relationship list (deterministic for image + config).
Relationship payload exposed to apps: id, subject_id, predicate,
object_id, confidence.  No tensors/logits/embeddings exposed.

12. JSON/application-facing structure (SceneGraphResult.to_dict())
{
  "image": {"width": W, "height": H},
  "objects": [{"id": "obj_0", "label": "person", "confidence": 0.813,
               "bbox": {"cx":.., "cy":.., "width":.., "height":..}}, ...],
  "relationships": [{"id": "rel_0", "subject_id": "obj_0",
                     "predicate": "wearing", "object_id": "obj_5",
                     "confidence": 0.7761}, ...]
}

13. Tests performed
- Pure unit checks (foreground, no models): funnel counters, threshold
  boundary, first-occurrence dedupe, inverse-pair survival, id formats.
- tests/test_postprocessing_local.py, threshold 0.30 (detached run)
- tests/test_postprocessing_local.py, threshold 0.20 (detached run)
- tests/test_pipeline_local.py (Phase A test, new defaults)

14. Test results
- Unit checks: PASS
- threshold 0.30: 9 objects; funnel 68 raw -> -51 no-rel -> -1 below
  threshold -> -0 duplicates -> 16 final; ALL CHECKS PASS; flagship
  person->riding->bike (21.79%) correctly filtered (21.79 < 30)
- threshold 0.20: funnel 68 -> -51 -> -0 -> -0 -> 17 final; ALL CHECKS
  PASS; flagship PRESENT at 21.79 percent (rel_0, obj_0 -> riding -> obj_3)
- Phase A test rerun: PASS - 9 objects, 16 relationships (final graph under
  new default), 51 no-relationship; detection and model outputs identical
  to Phase A (only post-processing changed the final count 17 -> 16)

15. Bugs encountered
- Unit-check data errors (not code bugs): first expectation assumed a
  21.79 percent sample passes a 0.30 threshold; second used a synthetic
  duplicate that was below threshold so dedupe never saw it; third assumed
  inverse pairs get deduped (they must not).  All three were test-data
  mistakes, corrected in the checks.
- Test environment: interactive command cap (~30 s) again required
  detached execution with --log-file for the full-pipeline runs.

16. Bugs fixed
- Same as above (test expectations, not library code).  Library code
  needed no bug fixes; all failures were in my assertions.

17. Architectural decisions
- Post-processing is a separate module (scene_graph/postprocessing.py);
  pipeline.py only CALLS it.  Phase A modules untouched.
- SceneGraphResult extended additively (new optional field + new method);
  existing fields and semantics preserved (backward compatible).
- Funnel statistics are first-class (PostprocessStats) so the app can
  report exactly what was filtered.
- to_dict() is the single serialization point for later FastAPI/React.

18. Deviations from the original project plan
- None material.  The default relationship_threshold (0.30) was chosen as
  a documented starting point, not a tuned optimum, per instructions.

19. Current limitations
- 0.30 default filters the flagship 21.79 percent prediction; a lower
  default (e.g. 0.20) keeps it - threshold tuning is deferred to a later
  phase (needs more images/evaluation).
- Duplicate removal is a safety net; pair generation is currently unique
  per (i, j), so it rarely triggers.
- Object ids are stable per-image only (re-running analyze on the same
  image reproduces them; ids have no meaning across images).

20. Exact next phase
- PHASE C - End-to-end local ML test (or FastAPI per the approved phase
  order: Phase C = end-to-end local ML test, Phase D = FastAPI).

===============================================================
ENTRY 2 APPENDIX: FUNCTION REFERENCE (Phase B files)
===============================================================

scene_graph/postprocessing.py
-----------------------------

@dataclass PostprocessStats
  Funnel counters, all ints defaulting to 0:
    raw_predictions             number of raw pair predictions entering
    no_relationship_removed     dropped because model class == 50
    below_threshold_removed     dropped by confidence < threshold
    duplicates_removed          dropped as exact duplicates
    final_relationships         survivors (== final graph edges)
  - to_dict(self) -> Dict[str, int]
      Returns the five counters as a plain dict (JSON-ready).

- remove_no_relationship(predictions: List[PairPrediction])
        -> Tuple[List[PairPrediction], int]
  Drops every prediction with class_id == NO_RELATIONSHIP_CLASS (50).
  Imports NO_RELATIONSHIP_CLASS from mapping inside the function (keeps the
  convention in one place).  Returns (kept, removed_count).  Defensive:
  the pipeline already buckets class 50 separately, but this module is safe
  standalone.

- filter_by_confidence(predictions, threshold: float)
        -> Tuple[List[PairPrediction], int]
  Keeps predictions with confidence >= threshold, using the existing
  relationship-model softmax confidence.  Returns (kept, removed_count).

- remove_duplicates(predictions)
        -> Tuple[List[PairPrediction], int]
  Conservative exact-duplicate removal.  Duplicate key =
  (subject.object_index, predicate, object.object_index); first occurrence
  wins; inverse relations are NOT duplicates (different object pairs).
  Returns (kept, duplicates_removed_count).

- postprocess_relationships(predictions, confidence_threshold: float)
        -> Tuple[List[PairPrediction], PostprocessStats]
  The full funnel in fixed order: 1) no-relationship removal,
  2) confidence threshold, 3) duplicate removal.  Returns (final list,
  stats).  This is the only function the pipeline calls.

- object_id(object_index: int) -> str
  Stable object id "obj_{object_index}" (detection-index based).

- relationship_id(relationship_index: int) -> str
  Stable relationship id "rel_{relationship_index}" (final-list position).

scene_graph/pipeline.py  (changes only)
---------------------------------------

- SceneGraphPipeline.__init__: NEW parameter relationship_threshold: float
  = 0.30 (stored as self.relationship_threshold).  Everything else in the
  constructor is unchanged.

- SceneGraphPipeline.analyze (rewired steps 3-4):
    step 3 now collects ALL pair predictions into one list `predictions`
    step 4 calls postprocess_relationships(predictions,
         self.relationship_threshold) -> (final_predictions, stats)
    unrelated = [p for p in predictions if p.class_id == 50]
    Relationship objects are built ONLY from final_predictions (indices
    rel_0..rel_{n-1}); SceneGraphResult now also carries stats.

- @dataclass SceneGraphResult (extended additively):
    scene, related, unrelated (unchanged fields)
    stats: Optional[PostprocessStats] = None   (NEW)
  NEW method:
    - to_dict(self) -> dict
        JSON-serializable graph: image {width,height}; objects [{id, label,
        confidence, bbox{cx,cy,width,height}}]; relationships [{id,
        subject_id, predicate, object_id, confidence}].  Ids come from
        object_id()/relationship_id().  No tensors/logits/embeddings.

scene_graph/__init__.py (changes only)
--------------------------------------
  Re-exports added: PostprocessStats, filter_by_confidence,
  remove_duplicates, remove_no_relationship, postprocess_relationships,
  object_id, relationship_id (all listed in __all__).

tests/test_postprocessing_local.py
----------------------------------

- main() -> None
  Phase B test driver.  Steps:
    1. argparse: --image, --checkpoint, --yolo-model, --threshold (default
       0.30), --log-file (line-buffered redirect for detached runs)
    2. builds SceneGraphPipeline(..., relationship_threshold=args.threshold)
    3. result = pipeline.analyze(image); graph = result.to_dict()
    4. prints SCENE GRAPH RESULT / POST-PROCESSING FUNNEL / OBJECTS /
       RELATIONSHIPS sections
    5. runs checks (each prints [PASS]/[FAIL], failures collected):
       B  objects detected > 0
       C  no "no relationship" entries in final graph
       D  every relationship endpoint id exists in object ids
       E  unique (subject_id, predicate, object_id) count == total
       F  all final confidences >= threshold
       F2 funnel accounting: raw == no_rel + below + dups + final
       F3 if flagship (0.2179) < threshold then below_threshold_removed >= 1
       G  flagship present iff 0.2179 >= threshold (model-output driven,
          never forced)
       +  obj_N / rel_N id format checks
    6. prints RESULT: PASS (or FAIL + exit code 1)

Module constants: REPO_ROOT, DATA_DIR, MODEL_DIR, DEFAULT_IMAGE,
CHECKPOINT_PATH, YOLO_MODEL_PATH.

Unchanged Phase A files (for reference): mapping.py, detector.py,
features.py, predictor.py, notebook/__init__.py,
notebook/models/__init__.py, tests/test_pipeline_local.py - no edits in
Phase B.

================================================================================
ENTRY 3 - PHASE C: FASTAPI BACKEND
================================================================================

Date: 2026-09-12
Phase: C (FastAPI Backend)
Objective: Build a minimal FastAPI backend around the already-working Scene
            Graph Pipeline. Expose ML inference as a clean REST API.

Files inspected:
    scene_graph/ (__init__.py, mapping.py, detector.py, features.py,
                  predictor.py, pipeline.py, postprocessing.py)
    datasets/visual_genome/ (loader.py, models.py, __init__.py)
    tests/test_pipeline_local.py, tests/test_postprocessing_local.py
    post_training_full_stack.md

Findings from inspection:
    - SceneGraphPipeline constructor: checkpoint_path, yolo_model_path,
      dataset_root, image_root, device, relationship_threshold, expansion,
      skip_same_label.
    - analyze() accepts ImageSource (str, PathLike, PIL.Image, numpy).
    - SceneGraphResult.to_dict() returns JSON-serializable scene graph.
    - Models loaded once in __init__(); analyze() just runs inference.

Files created:
    backend/__init__.py          - exports ``app``
    backend/main.py              - FastAPI app with /health, /analyze,
                                  lifespan, CORS, validation, error handling
    tests/test_api_local.py      - 5 TestClient tests (health, analyze
                                  success, missing file, invalid image,
                                  unsupported type)
    tests/run_api_server.py      - uvicorn startup script
    tests/test_api_manual.py     - live-server manual test (2 requests)

Files modified:
    requirements.txt             - appended fastapi==0.141.1, uvicorn==0.52.4

Unchanged Phase A/B files: scene_graph/*, notebook/models/relationship/*,
datasets/visual_genome/* - NOT modified.

FastAPI app structure (backend/main.py):
    REPO_ROOT = Path(__file__).resolve().parents[1]
    DATASET_ROOT     = REPO_ROOT / "datasets" / "visual_genome"
    MODEL_DIR        = REPO_ROOT / "notebook" / "models"
    CHECKPOINT_PATH  = MODEL_DIR / "best_full_fusion_30epoch.pt"
    YOLO_MODEL_PATH  = MODEL_DIR / "yolov8s-world.pt"
    CORS_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]
    ALLOWED_CONTENT_TYPES = {jpeg, jpg, png, webp}
    MAX_UPLOAD_BYTES = 20 MB

    _pipeline: SceneGraphPipeline | None = None   (module singleton)

    def get_pipeline():
        global _pipeline
        if _pipeline is None:
            _pipeline = SceneGraphPipeline(
                checkpoint_path=CHECKPOINT_PATH,
                yolo_model_path=YOLO_MODEL_PATH,
                dataset_root=DATASET_ROOT,
                image_root=DATASET_ROOT,
            )
        return _pipeline

    @asynccontextmanager
    async def lifespan(app):
        get_pipeline()      # load ML stack once at startup
        yield

    app = FastAPI(title="Scene Graph Explorer API", version="0.1.0",
                  lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, ...)

    @app.get("/health")
    def health():
        return {"status": "healthy"}

    def _decode_upload(data, content_type):
        # raises HTTPException(400) for empty/oversized/unsupported/invalid
        image = Image.open(io.BytesIO(data)); image.load()
        return image.convert("RGB")

    @app.post("/analyze")
    def analyze(file: UploadFile = File(...)):
        data = file.file.read()
        image = _decode_upload(data, file.content_type)
        result = get_pipeline().analyze(image)
        return result.to_dict()

Pipeline initialization strategy:
    - _pipeline is a module-level global singleton.
    - get_pipeline() creates it lazily on first call; subsequent reuse it.
    - lifespan() calls get_pipeline() at startup: models load ONCE.
    - Requests NEVER construct a pipeline themselves.

Endpoints:
    GET  /health  -> 200 {"status": "healthy"}  (no inference)
    POST /analyze -> multipart field "file"; returns SceneGraphResult.to_dict()
                     400 missing/oversized/unsupported/invalid
                     500 unexpected inference failure

Image validation (_decode_upload):
    1. Reject empty upload (400 "Empty upload.")
    2. Reject >20MB (400 "Image too large.")
    3. Reject content-type not in {jpeg,jpg,png,webp}
    4. Try PIL.Image.open()+load(); on failure -> 400 "Invalid image data."
    5. Convert to RGB and return.

Error handling:
    - 400: raised by _decode_upload for client errors.
    - 422: FastAPI auto-rejects missing ``file`` field.
    - 500: any unhandled inference exception; logger.exception() to console;
      client gets "Inference failed."

CORS: dev-only; allow_origins=[localhost:5173, 127.0.0.1:5173].

Response schema: exactly SceneGraphResult.to_dict() (thin transport layer).

Separation: backend/main.py contains ONLY HTTP, validation, errors, CORS.
ALL ML logic lives in scene_graph/.

Tests (tests/test_api_local.py, TestClient in-process):
    A. test_health              -> PASS (200, body=={"status":"healthy"})
    B. test_analyze_success     -> PASS (9 objects, 16 relationships)
       Samples: person->wearing->shoe (77.61%), car->on->sidewalk (32.46%)
    C. test_analyze_missing_file   -> PASS (422)
    D. test_analyze_invalid_image  -> PASS (400 "Invalid image data.")
    E. test_analyze_unsupported_type -> PASS (400 "Unsupported file type.")
    RESULT: ALL 5 TESTS PASSED.

Multiple-request / model-loading verification:
    Server log shows "Loading SceneGraphPipeline (one-time)..." once.
    Both /analyze requests succeed; pipeline is reused.

Bugs encountered: None. Phase A/B unchanged; backend imports cleanly.

Architectural decisions:
    - Singleton pipeline (module global + lazy get_pipeline()).
    - lifespan context manager for one-time model load at startup.
    - PIL.Image.load() called before convert("RGB") to catch truncated images.

Current limitations:
    - CPU cold start slow (60-90s); keep app alive between requests.
    - No GPU (CUDA not present).
    - CORS dev-only; production hardening deferred.
    - 20MB upload cap is arbitrary.

Next phase: REACT FRONTEND

================================================================================
ENTRY 4 - PHASE D1: REACT FRONTEND FOUNDATION
================================================================================

Date: 2026-09-12
Phase: D1 (React Frontend Foundation)
Objective: Build the frontend foundation with upload/analyze flow.
            Connect React to the existing FastAPI backend.

Repository structure inspected:
    backend/               - FastAPI app (Phase C, complete)
    scene_graph/           - ML pipeline (Phase A/B, complete)
    notebook/              - Notebooks + trained models
    tests/                 - Python tests (Phase A/B/C)
    datasets/              - Visual Genome data
    post_training_full_stack.md - Engineering diary (Entries 1-3)

Frontend technology chosen:
    React 18 + Vite 6 (JavaScript, not TypeScript)
    No state management library (React useState/useRef sufficient)
    No UI framework (plain CSS)

Files created:
    frontend/package.json          - React + Vite dependencies
    frontend/vite.config.js        - Vite config (host 127.0.0.1, port 5173)
    frontend/index.html            - Vite entry HTML
    frontend/src/main.jsx          - React root render
    frontend/src/App.jsx           - Main app component (upload + analyze)
    frontend/src/App.css           - Simple styles
    frontend/src/services/api.js   - API service (analyzeImage function)
    frontend/node_modules/         - Installed via npm

Files modified: None (no Phase A/B/C files changed)

Files NOT created (deferred to later phases):
    - React Flow graph component
    - Bounding box overlay
    - Image-graph synchronization
    - Inspector panel
    - Search functionality

API integration (frontend/src/services/api.js):
    const API_BASE_URL = 'http://127.0.0.1:8000'

    export async function analyzeImage(file) {
      const formData = new FormData()
      formData.append('file', file)

      let response
      try {
        response = await fetch(`${API_BASE_URL}/analyze`, {
          method: 'POST',
          body: formData,
        })
      } catch (err) {
        throw new Error(`Could not reach backend at ${API_BASE_URL}...`)
      }

      if (!response.ok) {
        // parse error detail from backend
        throw new Error(`Analysis failed (${response.status}): ${detail}`)
      }
      return response.json()
    }

Frontend state design (App.jsx):
    useState:
      selectedFile  - the chosen File object
      previewUrl    - data URL for image preview
      scene         - the returned scene graph (or null)
      isAnalyzing   - loading flag
      error         - error message (or null)
    useRef:
      fileInputRef  - to reset the file input on Reset

Upload flow:
    1. User clicks "Choose Image" (label wraps hidden file input)
    2. handleFileChange: validates file type (image/*), reads as data URL
    3. Preview appears below the buttons
    4. "Analyze" button enabled only when file selected

Analyze flow:
    1. User clicks "Analyze"
    2. isAnalyzing=true, error=null, scene=null
    3. analyzeImage(selectedFile) sends multipart/form-data to /analyze
    4. On success: scene=result, display object/relationship counts
    5. On failure: error=message, display in error box
    6. isAnalyzing=false

Display:
    - Header: "Scene Graph Explorer"
    - Choose Image button + Analyze button + Reset button
    - Image preview
    - Error box (conditional)
    - Results section: object count, relationship count, relationships list

Local testing:
    Backend:  .venv\Scripts\python.exe tests\run_api_server.py
    Frontend: cmd /c npm run dev (from frontend/)
    Both started and verified.

Test results:
    === Test 1: Frontend serves HTML ===
    Status: 200
    PASS

    === Test 2: CORS allows frontend origin ===
    Access-Control-Allow-Origin: http://127.0.0.1:5173
    PASS

    === Test 3: Analyze endpoint with file upload ===
    Status: 200
    Objects: 9
    Relationships: 16
    PASS

    Build test: npm run build -> PASS (146.20 kB JS, 1.53 kB CSS)

Bugs encountered:
    - PowerShell Start-Process with npm fails (npm is a script, not exe)
      Fix: used cmd.exe /c "npm run dev" instead
    - npm install via PowerShell had issues
      Fix: used cmd /c npm install

Architectural decisions:
    - JavaScript (not TypeScript) for simplicity
    - No state management library
    - API URL as a single constant (easy to change later)
    - File input wrapped in label for better UX
    - Error handling: network errors vs HTTP errors with detail

Deviations: None.

Current limitations:
    - No graph visualization yet
    - No bounding boxes on image
    - No inspector or search
    - No image-graph synchronization
    - No deployment setup
    - CORS is dev-only

Next phase: D2 - React Flow scene graph visualization

================================================================================
ENTRY 5 - PHASE D2: REACT FLOW SCENE GRAPH
================================================================================

Date: 2026-09-12
Phase: D2 (React Flow Scene Graph)
Objective: Build the first actual Scene Graph visualization using React Flow.
            Convert scene objects to nodes and relationships to edges.

Files created:
    frontend/src/components/SceneGraph.jsx - React Flow graph component
    tests/append_diary_d2.py                - (temp, deleted after use)

Files modified:
    frontend/src/App.jsx  - Import SceneGraph, replaced text results with graph
    frontend/src/App.css - Added .graph-section, .graph-container, .graph-empty

Dependencies added:
    @xyflow/react (React Flow v12) - installed via npm

Object → Node mapping:
    Each scene object becomes exactly one React Flow node.
    Node id = object.id (e.g., "obj_0")
    Node data = { label, confidence }
    Grid layout: 3 columns, 220px horizontal spacing, 120px vertical spacing
    Node style: 160x60px, rounded border, blue (#4a90d9), light blue background

Relationship → Edge mapping:
    Each backend relationship becomes one React Flow edge.
    Edge id = relationship.id (e.g., "rel_0")
    Edge source = relationship.subject_id
    Edge target = relationship.object_id
    Edge label = relationship.predicate (e.g., "riding")
    Edge data = { confidence }
    Edges are animated for visibility

Node IDs: object.id from backend (obj_0, obj_1, ...)
Edge IDs: relationship.id from backend (rel_0, rel_1, ...)

Layout approach:
    Simple deterministic grid layout (no complex algorithm).
    Objects arranged left-to-right, top-to-bottom in 3 columns.
    React Flow's fitView ensures all nodes are visible on load.

Graph interactions:
    - Pan (drag background)
    - Zoom (scroll wheel)
    - Fit view (automatic on load)
    - Node selection (click node, logged to console)
    - MiniMap for navigation
    - Controls panel (zoom in/out, fit view)

Empty-state handling:
    - No scene selected: message "No scene to display..."
    - Scene with zero objects: message "No objects detected in this image."
    - Scene with objects but no relationships: nodes displayed, no edges

Display:
    After analysis, shows:
    IMAGE ANALYSIS RESULTS
    Objects: X | Relationships: Y
    SCENE GRAPH
    [React Flow graph component]

Testing:
    Backend:  .venv\Scripts\python.exe tests\run_api_server.py
    Frontend: cmd /c npm run dev (from frontend/)
    Both started and verified.

Test results:
    === Test 1: Frontend serves HTML ===
    Status: 200, PASS

    === Test 2: CORS allows frontend origin ===
    Access-Control-Allow-Origin: http://127.0.0.1:5173, PASS

    === Test 3: Analyze endpoint with file upload ===
    Status: 200, Objects: 9, Relationships: 16, PASS

    Build test: npm run build → PASS (333.90 kB JS, 17.79 kB CSS)

    Graph verification (manual):
    - 9 objects → 9 nodes in graph
    - 16 relationships → 16 edges in graph
    - Each object has unique node (obj_0 through obj_8)
    - Duplicate labels (e.g., person, tree) remain separate nodes
    - Predicate labels visible on edges
    - Graph pans, zooms, and fits view correctly

Bugs encountered: None.

Architectural decisions:
    - SceneGraph component is purely presentational (receives scene prop)
    - No API calls or state management in graph component
    - useNodesState/useEdgesState for React Flow state management
    - Grid layout for deterministic, readable arrangement
    - Animated edges for visibility of relationships

Deviations: None.

Current limitations:
    - No image-graph synchronization yet
    - No inspector panel yet
    - No search functionality yet
    - Node click only logs to console
    - No fancy UI design

Next phase: D3 - Image bounding boxes + graph synchronization

================================================================================
ENTRY 6 - Backend cold-start fix (HF offline) + startup diagnosis
================================================================================

Date: 2026-09-12
Phase: C follow-up (backend startup performance)

Objective: Diagnose why the FastAPI backend takes a while to start; fix the
            one avoidable cause.

Diagnosis (measured with one-off profilers, CPU-only machine):
    Cold start ~50-120 s depending on machine load.
    - torch import                                  6-20 s
    - open_clip import                               2-28 s
    - ultralytics import                             ~2 s
    - VisualGenomeLoader (HDF5 + JSON)                ~0.3-1.3 s
    - YOLO-World load + set_classes (149 VG classes)  seconds (81 s under load)
    - CLIP ViT-B/32 ~90 MB from HF cache             9-15 s
    - checkpoint torch.load + load_state_dict        <0.5 s
    The dominant unavoidable cost is loading three ML models at startup
    (by design: lifespan -> get_pipeline -> module singleton, load once).

    Actual ISSUE found:
    - tests/* set HF_HUB_OFFLINE=1, but backend/main.py did NOT, so every
      server start performed a live HTTP HEAD round-trip to huggingface.co
      (observed in backend logs: HTTP Request: HEAD https://huggingface.co/
      timm/vit_base_patch32_clip_224.openai/...).  Adds seconds and can
      stall for a long time on slow/blocked networks.

Files modified:
    backend/main.py - added, BEFORE the scene_graph.pipeline import:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    (valid because all weights are cached locally under
    ~/.cache/huggingface/hub; same setting the test scripts already use.)

Files created: none (profiler scripts were temporary; removed after use).

Verification: py_compile backend/main.py -> OK; low-level imports proven
    offline-safe by prior offline profile runs (profile v2 completed every
    step with HF_HUB_OFFLINE=1).

Result: network round-trip at startup removed; startup is now deterministic
    (still several seconds for unavoidable model loading, that is expected).

Known remaining cost (not a bug):
    - Importing torch/open_clip/ultralytics (8-16 s)
    - YOLO-World + 149-class text embeddings
    - CLIP ViT-B/32 90MB weights from disk cache
    No GPU on this machine; CPU-only.

Next phase: D3 - Image bounding boxes + graph synchronization (unchanged).

================================================================================
ENTRY 7 - PHASE D3: IMAGE VIEWER + BOUNDING BOXES + GRAPH SYNC + LAYOUT
================================================================================

Date: 2026-09-13
Phase: D3 (frontend only - no ML/backend changes)

1. Objective
   - Image viewer with correctly aligned bounding boxes (object-ID identity)
   - Side-by-side explorer layout (image | graph) on desktop, stacked on small
   - Bidirectional image <-> graph synchronization via shared selection state
   - Relationship selection highlights edge + both endpoint boxes
   - Replace grid layout with a proper automatic dagre (LR) layout

2. Files inspected
   frontend/src/App.jsx, components/SceneGraph.jsx, App.css,
   frontend/src/services/api.js, backend/main.py (contract unchanged)

3. Files created
   frontend/src/components/SceneImage.jsx  (image + bbox overlay + click)
   frontend/verify_layout.mjs              (dagre layout verifier, Node)
   tests/_fetch_scene.py                   (saves live scene JSON to
                                            tests/_scene.json for the
                                            verifier)

4. Files modified
   frontend/src/components/SceneGraph.jsx  (dagre layout, selection props,
                                            edge click, no console.log)
   frontend/src/App.jsx   (owns selectedObjectId/selectedRelationshipId;
                           passes scene + selection to both views;
                           sceneKey remount for fresh layout per analyze)
   frontend/src/App.css   (.explorer grid 0.9fr/1.1fr, @media <=900px stack,
                           .scene-image/.bbox layer styles, selected and
                           highlighted variants)
   frontend/package.json  (+ @dagrejs/dagre dependency)

5. Dependencies added
   @dagrejs/dagre (only graph-layout library; React Flow kept from D2)

6. Object identification
   scene.objects[i].id (obj_0, obj_1, ...) is the single source of truth.
   bbox buttons key/id/title by object id; duplicate labels (person x2,
   tree x2) remain separate objects and separate nodes.

7. Image coordinate handling
   Backend bbox is (cx, cy, width, height) in ORIGINAL image pixels.
   SceneImage renders the img at width:100%, height:auto (aspect preserved)
   inside a position:relative wrapper; the overlay is absolute and covers
   exactly the rendered image.  Each box uses PERCENTAGES:
       left% = (cx - w/2) / imageWidth  * 100
       top%  = (cy - h/2) / imageHeight * 100
       width% = w / imageWidth * 100 ; height% = h / imageHeight * 100
   so alignment holds at any responsive size (no fixed viewport pixels).

8. State architecture
   App owns: scene, selectedObjectId, selectedRelationshipId.
   Clicking a bbox or a node  -> setSelectedObjectId(id) (relationship null)
   Clicking an edge           -> setSelectedRelationshipId(rel.id)
                                 (subject/object ids derived in App and
                                 passed down as highlightedObjectIds)
   Single scene object feeds both SceneImage and SceneGraph (no copies).

9. Graph layout (dagre, deterministic)
   rankdir=LR, nodesep=50, ranksep=120, marginx/y=20.
   Multigraph so parallel relationships between the same pair are kept;
   each relationship id used as the dagre edge name.
   Layout recomputed via useMemo on scene change (plus sceneKey remount).

10. Visual synchronization behavior
    - selected object: red border node + red bbox
    - selected relationship: thick red edge + orange-highlighted endpoint
      nodes + orange-highlighted endpoint bboxes
    - fitView on load; pan/zoom unchanged; node drag still possible

11. Tests performed
    - npm run build -> PASS (385 kB JS / 18.9 kB CSS)
    - Backend + frontend up; tests/test_connection.py -> ALL PASS
      (HTML 200, CORS origin ok, /analyze 200 with 9 objects / 16 rels)
    - Layout verifier on REAL scene JSON (frontend/verify_layout.mjs):
      9/9 nodes, 16/16 edges, positions finite/non-negative, no node-center
      overlaps, deterministic, duplicate labels separate -> PASS
      (dagre produced a 5-rank LR hierarchy: vehicle -> trees -> persons ->
      bike/car/shoe -> sidewalk)
    - tests/test_api_local.py rerun -> ALL 5 TESTS PASSED (backend intact)

12. Bugs encountered
    a) dagre v1 API mismatch: SceneGraph initially called
       graph.setDefaultEdgeValue(...) / graph.addNode / graph.addEdge and
       passed options as dagre.layout(graph, opts).  @dagrejs/dagre exposes
       setGraph/setNode/setEdge/setDefaultEdgeLabel and layout(graph) only -
       the original code would have CRASHED at runtime in the browser.
       Caught by the Node layout verifier (setDefaultEdgeValue is not a
       function); fixed in both SceneGraph.jsx and the verifier:
         graph.setGraph({rankdir:'LR',...})
         graph.setDefaultEdgeLabel(() => ({}))
         graph.setEdge(v, w, {weight:1}, edgeId)   // multigraph name
         dagre.layout(graph)
    b) Node verifier placed in tests/ could not resolve @dagrejs/dagre
       (module resolution is per-directory) -> moved to frontend/.
    c) TestClient API rerun HUNG (flat CPU) while the uvicorn dev server
       was running - process contention on model load / HF cache, not a
       code bug.  Killed both processes; the isolated rerun passed.

13. Fixes: as above (a) runtime crash fixed before it shipped, (b) script
    relocation, (c) environment-only; no backend changes.

14. Architectural decisions
    - Shared selection state lives in App (simplest level that both views
      can see); no Redux/Zustand.
    - Boxes are <button> elements (keyboard/click + e.stopPropagation) so
      panning/zooming the graph is unaffected.
    - object id is the bridge; labels are display-only.
    - verifier replicates SceneGraph's build/layout functions exactly so
      layout regressions are caught without a browser.

15. Limitations
    - No inspector panel yet (edge click highlights only)
    - No search; no cross-scene persistence
    - dagre layout does not account for user-dragged node positions after
      re-analysis (remount resets layout - acceptable)
    - bbox label may overflow the image edge for boxes near the border

16. Exact next phase: D4 - Inspector + search (object/relationship details,
    highlight filtering), then deployment.

================================================================================
ENTRY 8 - PHASE D4: MINIMAL INSPECTOR + BASIC OBJECT SEARCH
================================================================================

Date: 2026-09-13
Phase: D4 (frontend only - no ML/backend changes)

1. Objective
   Add exactly three frontend features on top of the D3 image+graph explorer:
   (1) minimal Object Inspector
   (2) minimal Relationship Inspector
   (3) basic object search by label

2. Files inspected
   frontend/src/App.jsx, components/{SceneImage,SceneGraph}.jsx, App.css

3. Files created
   frontend/src/components/Inspector.jsx - object / relationship details panel
   frontend/verify_d4_logic.mjs         - search + inspector logic verifier
   tests/_fetch_scene.py                 - saves live scene JSON (reused by the
                                           layout verifier from D3 and the D4
                                           logic verifier)

4. Files modified
   frontend/src/App.jsx:
     - added searchQuery state + searchObjectIds (useMemo, case-insensitive
       label match over scene.objects)
     - passes searchObjectIds to SceneImage and SceneGraph
     - imports and renders Inspector below the explorer
     - search input + match-count shown between explorer and inspector
     - handleReset clears searchQuery
   frontend/src/components/SceneImage.jsx:
     - accepts searchObjectIds prop; adds bbox-searched (teal) styling
     - priority: selected > relationship-endpoint > search-match
   frontend/src/components/SceneGraph.jsx:
     - accepts searchObjectIds prop; adds COLOR_SEARCH (#00897b) node border
     - nodeBorderStyle priority: selected > rel-endpoint > search-match
     - added searchIds to useEffect dependency array
   frontend/src/App.css:
     - .search-section, .inspector, .inspector-list, .bbox-searched styles

5. State architecture (unchanged ownership model)
   App owns: scene, selectedObjectId, selectedRelationshipId, searchQuery.
   searchObjectIds derived via useMemo (pure function of searchQuery + scene).
   Single scene object feeds SceneImage, SceneGraph, and Inspector.

6. Search implementation
   - case-insensitive substring match on object label
   - operates entirely on scene.objects (no backend endpoint)
   - empty query -> no highlighting; cleared automatically on reset
   - matched IDs drive BOTH image bbox styling and graph node styling

7. Inspector implementation (Inspector.jsx, presentational)
   Priority: selectedRelationshipId > selectedObjectId > placeholder.
   - Object: label, id, confidence, bbox (cx/cy/w/height), plus outgoing and
     incoming relationships resolved by ID through scene.relationships.
   - Relationship: subject / predicate / object labels + confidence.
   - Resolves subject/object via IDs, never labels, so duplicate labels stay
     distinct.

8. Synchronization behavior
   - Clicking a bbox or node sets selectedObjectId (clears relationship).
   - Clicking an edge sets selectedRelationshipId (clears object).
   - Selection priority is consistent across SceneImage, SceneGraph, Inspector.

9. Dependencies added: none (pure React + existing @xyflow/react).

10. Tests performed
    - npm run build -> PASS (389 kB JS / 20.3 kB CSS)
    - Both servers up; tests/test_connection.py -> ALL PASS
      (HTML 200, CORS ok, /analyze 9 objects / 16 relationships)
    - verify_d4_logic.mjs on REAL scene JSON -> ALL PASS:
        search "person" -> obj_0, obj_1  (2 matches)
        search "tree"   -> obj_7, obj_8  (2 matches)
        search "car"    -> obj_2         (1 match)
        search empty / nonexistent -> 0 matches
        duplicate labels stay separate (IDs distinct)
        inspector resolves object relationships (2 out / 3 in for obj_0)
        inspector resolves relationship endpoints (person->wearing->shoe)
    - tests/test_api_local.py -> ALL 5 PASSED (backend intact)

11. Bugs encountered: None. Straightforward additions.

12. Architectural decisions
   - searchQuery kept as simple controlled input; match logic in useMemo.
   - Inspector is purely presentational (no callbacks); App owns selection.
   - search highlighting uses distinct visual channel (teal) so it does not
     clash with selected (red) or relationship-endpoint (orange).

13. Current limitations
   - No inspector panel for edge details beyond basic fields (no delete/edit).
   - No search by ID, predicate, or confidence.
   - No NLP / fuzzy / semantic search.
   - bbox label can overflow near image border (inherited from D3).

14. Exact next phase: D5 - Image <-> graph synchronization polish, inspector
    search filtering, or deployment prep (per project plan).


================================================================================
ENTRY 9 - PHASE: FINAL FRONTEND POLISH + LOCAL FULL-STACK PIPELINE
================================================================================

Phase:  Final frontend polish + local developer workflow (FRONTEND ONLY)
Date:   2026-09-15
Status: COMPLETE (code + automated verification).  Three browser-only checks
        are listed in section 25 for manual confirmation by the user.
Scope:  no ML / no backend / no dataset / no API-contract changes.

--------------------------------------------------------------------------------
1. Objective
--------------------------------------------------------------------------------
Turn the working D1-D4 prototype into a polished, portfolio-ready dark-themed
Scene Graph Explorer and provide a clean local 2-terminal run workflow
(backend + frontend), without touching the verified ML stack.

Required deliverables:
  - dark professional UI (default and only theme)
  - page-centred header title that does not depend on the upload controls
  - "Show object detections" toggle (visualisation-only)
  - graph layout switched from left-to-right to TOP-TO-BOTTOM
  - less cluttered graph (static edges, wider spacing, paired inverse edges)
  - "Download Graph PNG" export containing only the graph
  - inspector hidden unless something is selected
  - clicking empty graph canvas clears the selection
  - loading / empty / error states, working Reset
  - documented local start-up scripts

--------------------------------------------------------------------------------
2. Repository state inspected
--------------------------------------------------------------------------------
  frontend/src/App.jsx, App.css, main.jsx, services/api.js
  frontend/src/components/SceneGraph.jsx, SceneImage.jsx, Inspector.jsx
  frontend/package.json, vite.config.js, index.html
  frontend/verify_layout.mjs, verify_d4_logic.mjs
  tests/test_connection.py, test_api_local.py, run_api_server.py, _fetch_scene.py
  post_training_full_stack.md (entries 1-8)
  backend/main.py (read-only: confirmed the API contract and the one-time
  pipeline loading were NOT changed)

IMPORTANT DISCOVERY - the previous attempt at this phase had been interrupted
mid-edit (model failure) and left the frontend in a BROKEN state:
  - App.jsx       : `searchObjectIds` declared TWICE (lines 24-28 and 33-39)
                    -> "has already been declared" build error
  - SceneImage.jsx: orphaned duplicated code after the export (lines 72-135)
                    -> syntax error; the showDetections guard was missing
  - SceneGraph.jsx: findInverseGroups missing its closing brace, a stray `}`,
                    and downloadGraphPng defined but never wired to a button
  - App.css       : `.btn-ghost:hover` missing its `}` (so `.explorer {` became
                    a nested rule) plus an orphaned declaration block
  - SceneImage.jsx did not exist at all during a later build attempt
  - scripts/ never created; diary Entry 9 never written
Therefore the four broken files were removed and rewritten from scratch, and
every claim below is backed by a command run in this session.

--------------------------------------------------------------------------------
3. Existing frontend architecture (preserved)
--------------------------------------------------------------------------------
  App.jsx         - global state: selectedFile, previewUrl, scene, isAnalyzing,
                    error, selectedObjectId, selectedRelationshipId,
                    searchQuery, sceneKey (remount key for a fresh layout)
  SceneImage.jsx  - image + percentage-positioned bounding boxes, click -> id
  SceneGraph.jsx  - React Flow + dagre, node click / edge click
  Inspector.jsx   - read-only details panel resolving ids from the one scene
  api.js          - analyzeImage(file) -> POST /analyze (multipart, field "file")
Selection ids live ONLY in App; both views and the inspector read the same
`scene` object (there is never a second copy of the data).

--------------------------------------------------------------------------------
4. Files created
--------------------------------------------------------------------------------
  scripts/run_backend.ps1      - starts uvicorn on 127.0.0.1:8000 using the venv
                                 interpreter (no manual activation required)
  scripts/run_frontend.ps1     - starts the Vite dev server on 127.0.0.1:5173 and
                                 auto-runs npm install if node_modules is missing
  scripts/README.md            - documented workflow: setup, backend, frontend,
                                 URLs, stopping, health check, end-to-end usage,
                                 and the test commands
  frontend/verify_phase_e.mjs  - NEW verifier for this phase (toggle is
                                 visualisation-only, inspector visibility rules,
                                 inverse-pair symmetry, reset semantics, dangling
                                 endpoints, "no relationship" leakage)
  frontend/src/components/SceneImage.jsx - RE-CREATED (file was missing/broken)

--------------------------------------------------------------------------------
5. Files modified
--------------------------------------------------------------------------------
  frontend/src/App.jsx                    - full rewrite: dark UI, centred header,
                                            detection toggle, Reset, hint line,
                                            gated inspector, empty/loading/error
                                            states, search section
  frontend/src/App.css                    - full rewrite: dark theme, 554 lines,
                                            every class used by the components
                                            (including the responsive block)
  frontend/src/components/SceneGraph.jsx  - rewrite: dagre TB layout, static
                                            edges, inverse-pair styling, PNG
                                            export toolbar, pane-click
                                            deselection, empty-graph state
  frontend/verify_layout.mjs              - updated: it was still mirroring the
                                            D2/D3 LR layout (160x64 nodes) and
                                            had an always-true assertion; now
                                            mirrors TB (170x56) and adds 12
                                            component-source checks

NOT modified: backend/*, scene_graph/*, datasets/*, notebook/*,
requirements.txt, tests/test_api_local.py, tests/test_connection.py,
tests/_fetch_scene.py, frontend/services/api.js,
frontend/components/Inspector.jsx, frontend/src/main.jsx,
frontend/vite.config.js, frontend/index.html.

--------------------------------------------------------------------------------
6. Dependencies added
--------------------------------------------------------------------------------
  html-to-image ^1.11.13   - PNG export of the graph element

Already present and reused:
  @xyflow/react ^12.11.6, @dagrejs/dagre ^3.1.1, react 18, react-dom 18, vite 6.

No state-management library (no Redux/Zustand), no UI framework, no extra graph
library, no screenshot framework beyond html-to-image.

--------------------------------------------------------------------------------
7. Dark UI implementation
--------------------------------------------------------------------------------
  - Page background #0b0f17, panels #111827, graph canvas #0f1620, borders
    #1c2533, body text #c9d4e0, headings #eef4fb, muted text #6b7c93.
  - Accent blue #3b82f6.  State colours: selected red #ef4444, relationship
    endpoint orange #f0932c, search match teal #00897b (three distinct channels,
    so a selection never looks like a search match).
  - Buttons: .btn-primary (blue), .btn-secondary (slate), .btn-ghost (Reset),
    .btn-sm for in-panel actions; hover and disabled states defined.
  - Dark mode is the only theme (no switcher).  No gradients, no decorative
    graphics, restrained shadows, 6-10px radii.

--------------------------------------------------------------------------------
8. Header changes
--------------------------------------------------------------------------------
  .app-header is a 3-column grid: `1fr auto 1fr`.
    column 1 : Choose Image / Analyze / Reset
    column 2 : the title block (h1 "Scene Graph Explorer" + subtitle)
    column 3 : an EMPTY spacer div
  The empty right-hand column is what makes the title genuinely centred on the
  page: the upload controls no longer decide the title's position.
  Subtitle: "Visual scene understanding through object detection and
  relationship prediction".
  Reset is disabled when there is nothing to reset; Analyze is disabled until a
  file is chosen and while analysing.

--------------------------------------------------------------------------------
9. Image detection toggle (visualisation only)
--------------------------------------------------------------------------------
  New App state `showDetections` (default true), rendered as a checkbox in the
  Image Viewer panel header ("Show object detections") and passed to SceneImage.
  SceneImage renders the ENTIRE overlay layer inside `{showDetections && (...)}`
  so unchecked means no boxes, no labels and no confidence values - only the
  original image.
  Verified by verify_phase_e.mjs:
    - scene.objects and scene.relationships are untouched (9 / 16 either way)
    - graph nodes, edges, search and inspector logic do not depend on it
    - because the layer is not rendered at all, selection/search highlighting
      cannot make boxes reappear while detections are hidden
    - the meta line under the image shows "N objects - detections hidden"
  Reset restores showDetections to true.

--------------------------------------------------------------------------------
10. Graph orientation
--------------------------------------------------------------------------------
  dagre `rankdir` changed from 'LR' to 'TB' so a relationship reads
  subject -> predicate -> object DOWNWARDS.  Node handles follow suit
  (sourcePosition 'bottom', targetPosition 'top').
  Verified on the real scene: 5 rank rows
    y=58   vehicle
    y=244  tree, tree
    y=430  person, person
    y=616  car, bike, shoe
    y=802  sidewalk
  with 15 edges flowing downward and 1 upward (a genuine reverse/cycle edge).

--------------------------------------------------------------------------------
11. Graph clutter improvements
--------------------------------------------------------------------------------
  - `animated` removed from every edge (static edges only).
  - Edge type 'smoothstep' to avoid excessive bezier curvature.
  - Spacing increased: nodesep 70 (was 50), ranksep 130 (was 120), margins 30.
  - Edges #64748b at 1.5px; predicate labels get a dark background (#131c28)
    and 3px radius so the text stays legible over the canvas.
  - Node boxes 170x56, fill #1e2a3a, centred label.
  - Dagre centres are converted to React Flow top-left positions
    (x - w/2, y - h/2) so boxes land exactly where dagre intended.
  - No relationship is ever dropped to simplify the picture; the verifier
    asserts edge count == relationship count (16/16).

--------------------------------------------------------------------------------
12. Inverse relationship visualisation
--------------------------------------------------------------------------------
  Documented inverse pair used (from the Visual Genome predicate dictionary):
      "wearing" (VG 48)  <->  "worn by" (VG 50)
  INVERSE_PREDICATES = { wearing: 'worn by', 'worn by': 'wearing' }

  DECISION: both relationships are KEPT as separate, independently selectable
  edges - nothing is merged or deleted.  Merging was deliberately avoided
  because a merged edge would leave one of the two relationships unselectable,
  and the inspector could then not represent it.  The prompt explicitly allowed
  this fallback: "if grouping would make the React Flow interaction unreliable,
  prefer clean parallel/static edges over destructive merging".
  Instead the pair is grouped VISUALLY: the inverse partner is drawn with
  strokeDasharray '6 4' instead of the solid default, and both edges carry
  `data.inversePartnerId` pointing at each other.
  In the current test scene no true inverse pair occurs, so the verifier reports
  "no inverse pairs in this scene" and asserts symmetry + retention
  (16/16 relationships kept).  No inverse mapping was invented.

--------------------------------------------------------------------------------
13. Graph PNG export
--------------------------------------------------------------------------------
  Toolbar button "Download Graph PNG" (top-right of the Scene Graph panel) calls
  html-to-image `toPng` on the graph wrapper element only.
  - `fitView({ padding: 0.1 })` runs first so the whole graph is framed, then the
    transform is allowed to settle (~80 ms) before capture.
  - `filter` removes .react-flow__minimap, .react-flow__controls,
    .react-flow__panel and .react-flow__attribution from the capture, so the PNG
    contains only nodes, edges and predicate labels.
  - Background forced to the canvas colour #0f1620 to avoid a transparent PNG;
    the file is named scene-graph.png and downloaded through a temporary
    <a download> element.
  - Failures are caught and logged with console.error; the UI never breaks.
  NOTE: the actual click needs a browser (see section 25).

--------------------------------------------------------------------------------
14. Inspector visibility changes
--------------------------------------------------------------------------------
  Inspector.jsx itself is unchanged; visibility is controlled by App:
      {scene && !hasSelection && <p className="hint-line">Click an object or a
                                        relationship to inspect it.</p>}
      {scene && hasSelection  && <section className="inspector-section">
                                    <Inspector ... />
                                  </section>}
  where hasSelection = Boolean(selectedObjectId || selectedRelationshipId).
  States:
      no scene                 -> no inspector, no hint (clean empty state only)
      scene, nothing selected  -> NO inspector card; a single muted hint line
      object selected          -> Object Details (label, id, confidence, bbox,
                                  outgoing + incoming relationships)
      relationship selected    -> Relationship Details (subject, predicate,
                                  object, confidence)
  Priority is unchanged and now unambiguous: handleRelationshipSelect() clears
  selectedObjectId, so "relationship wins when both set" cannot normally occur
  (the verifier asserts the rule anyway).

--------------------------------------------------------------------------------
15. Empty-canvas deselection
--------------------------------------------------------------------------------
  ReactFlow's `onPaneClick` is wired to handlePaneClick() which clears BOTH
  selectedObjectId and selectedRelationshipId, so the inspector disappears.
  Node and edge clicks are separate React Flow callbacks (onNodeClick /
  onEdgeClick) and do not trigger the pane handler, so selecting a node or edge
  does not immediately clear itself.  `onMouseDown` + `stopPropagation` on the
  image bounding boxes prevents the same class of accidental clearing there.

--------------------------------------------------------------------------------
16. Search UI
--------------------------------------------------------------------------------
  Unchanged logic (case-insensitive substring over object LABELS producing
  object IDS), restyled for the dark theme and given a live match count
  ("N matches").  The same searchObjectIds array feeds the image (teal boxes) and
  the graph (teal node borders).  No NLP, no fuzzy/semantic search, no backend
  search endpoint.

--------------------------------------------------------------------------------
17. Layout changes
--------------------------------------------------------------------------------
  Explorer stays side-by-side on desktop:
      .explorer { display:grid; grid-template-columns: 0.9fr 1.1fr; gap:24px }
      (≈45% image / ≈55% graph)
  Results summary became two compact stat cards ("09 OBJECTS", "16
  RELATIONSHIPS") above the explorer.  Graph height 560px (420px under 900px).
  Below 900px the explorer stacks vertically and the header collapses to one
  centred column.

--------------------------------------------------------------------------------
18. Loading / error / empty states
--------------------------------------------------------------------------------
  Empty (no file, no scene, not analysing): a centred card reading
  "Upload an image to explore its scene graph".
  Preview (file chosen, not analysed): the selected image + file name.
  Loading: spinner + "Analyzing image..." and an honest second line -
  "Detection and relationship prediction run on CPU and can take up to a
  minute."  No fake progress percentages.
  Error: a red .error-box showing only the message from api.js (which already
  translates network failures and HTTP errors); no Python traceback or internal
  detail ever reaches the browser.

--------------------------------------------------------------------------------
19. Reset behavior
--------------------------------------------------------------------------------
  handleReset() clears: selectedFile, previewUrl, scene, selectedObjectId,
  selectedRelationshipId, searchQuery, error, and restores showDetections=true.
  It also resets the underlying <input type="file"> value so re-selecting the
  same file fires onChange again.  sceneKey is bumped on every successful
  analyze, which remounts the explorer so a new scene always gets a fresh dagre
  layout (no stale node positions from the previous image).

--------------------------------------------------------------------------------
20. Local startup workflow
--------------------------------------------------------------------------------
  Two terminals from the repository root:

  Terminal 1 (backend):
      powershell -ExecutionPolicy Bypass -File scripts\run_backend.ps1
      equivalent: .venv\Scripts\python.exe -m uvicorn backend.main:app
                  --host 127.0.0.1 --port 8000

  Terminal 2 (frontend):
      powershell -ExecutionPolicy Bypass -File scripts\run_frontend.ps1

  URLs:     frontend http://127.0.0.1:5173
            backend  http://127.0.0.1:8000
            swagger  http://127.0.0.1:8000/docs
            health   http://127.0.0.1:8000/health
  Stop:     Ctrl+C in each terminal.
  Health check:
      .venv\Scripts\python.exe -c "import httpx; print(httpx.get(
          'http://127.0.0.1:8000/health').json())"   -> {'status': 'healthy'}
  All of the above is also written in scripts/README.md (setup, run, stop,
  health, end-to-end usage, tests).  No ports were changed.

--------------------------------------------------------------------------------
21. Bugs encountered
--------------------------------------------------------------------------------
  B1. (inherited from the interrupted previous attempt) App.jsx had
      `searchObjectIds` declared twice - a hard esbuild redeclaration error.
  B2. (inherited) SceneImage.jsx contained a complete duplicated copy of the
      component after the `export default` block, producing a syntax error, and
      the first copy had no showDetections guard.
  B3. (inherited) SceneGraph.jsx: `findInversePartnerIds` had lost its closing
      brace; a stray `}` sat after the default export; `downloadGraphPng` was
      defined but never wired to any button, so the PNG export did not exist in
      the UI.
  B4. (inherited) App.css had an unbalanced brace: `.btn-ghost:hover` was never
      closed, so `.explorer { ... }` was parsed as a NESTED rule (which would
      have broken the explorer layout), plus an orphaned
      `background-color: #131a24; }` fragment after `.graph-empty`.
  B5. SceneImage.jsx was entirely MISSING from disk when the first rebuild ran
      ("Could not resolve ./components/SceneImage.jsx from src/App.jsx") - the
      earlier create call had been dropped.
  B6. verify_layout.mjs was stale: it still mirrored the D2/D3 LR layout
      (rankdir 'LR', 160x64 nodes) and contained an always-true assertion
      (`dupLabels.length > 0 ? true : true`), so it could pass while testing
      nothing relevant.
  B7. A stale backend process from an earlier session (PID 42268) still owned
      port 8000.  The freshly started server loaded the whole pipeline and then
      died with `[Errno 10048] error while attempting to bind on address
      ('127.0.0.1', 8000)`.  This is why "the backend takes ages and then does
      not respond" can happen.
  B8. My second SceneGraph.jsx edit was inserted one line too early (inside the
      `if (...)` condition of findInversePartnerIds), producing
      `ERROR: Expected ")" but found "function"` and orphaning the function's
      tail at the end of the file.
  B9. Two editor write calls in this session were silently dropped (the
      SceneImage.jsx create and the first verify_layout.mjs rewrite).  Both were
      caught by re-checking the filesystem/build output instead of trusting the
      previous message.

--------------------------------------------------------------------------------
22. Bugs fixed
--------------------------------------------------------------------------------
  B1-B4: the four broken files were removed and rewritten from scratch.
      App.jsx, App.css and SceneGraph.jsx were rewritten in full;
      SceneImage.jsx was re-created (B5).
  B6: verify_layout.mjs was UPDATED to mirror the real component (TB, 170x56,
      nodesep 70, ranksep 130, centre->top-left conversion) and extended with 12
      component-source assertions; the always-true check was replaced with a
      real unique-id assertion.  The test was updated, not bypassed or deleted.
  B7: all stale python/node processes were killed and port 8000 was confirmed
      free before starting a clean server (evidence: "python running: 0",
      "port 8000: free").
  B8: repaired with two surgical edits (re-inserted the function tail before
      `function buildEdges`, and removed the orphaned tail after the default
      export); the build then compiled 203 modules.
  B9: every dropped write was re-issued and then verified on disk
      (directory listing + build error + verifier output).

--------------------------------------------------------------------------------
23. Tests executed and results
--------------------------------------------------------------------------------
  (1) frontend production build - `cd frontend; npm run build`
        PASS - "203 modules transformed", built in 4.62s
        dist/index.html 0.48 kB, index-5bpIVK92.css 22.11 kB,
        index-C_6FGxYw.js 404.86 kB (gzip 132.96 kB)

  (2) node verify_layout.mjs   (layout + component source)   PASS
        9/9 layout checks: node count 9/9, edge count 16/16, positions finite,
        no node overlap, 5 rank rows (top-to-bottom), 15 edges down / 1 up, all
        endpoints exist, duplicate labels separate, layout deterministic
        12/12 source checks: TB orientation, no LR, no `animated`, dashed
        inverse style, toPng + scene-graph.png, minimap/controls filtered,
        onPaneClick, onEdgeClick, onNodeClick, gated inspector, toggle wired,
        bbox layer gated

  (3) node verify_d4_logic.mjs  (D4 regression)             PASS 10/10
        search person -> obj_0,obj_1; tree -> obj_7,obj_8; car -> obj_2;
        empty/nonexistent -> none; inspector resolves 2 out / 3 in for obj_0;
        endpoints person -> wearing -> shoe

  (4) node verify_phase_e.mjs   (NEW this phase)            PASS 16/16
        toggle is data-independent; inspector visibility rules; inverse
        symmetry; both relationships retained; no dangling endpoints; no
        "no relationship" leakage; reset clears scene/selection/search and
        restores showDetections; search resolves ids

  (5) tests/_fetch_scene.py     (real end-to-end inference) PASS
        status 200, objects 9, relationships 16 -> tests/_scene.json

  (6) tests/test_connection.py  (frontend + backend live)   PASS
        frontend HTML 200; CORS "http://127.0.0.1:5173";
        POST /analyze 200 with 9 objects / 16 relationships

  (7) tests/test_api_local.py   (regression, 5 tests)       PASS
        "RESULT: ALL PASSED" - test_health, test_analyze_success,
        test_analyze_missing_file (422), test_analyze_invalid_image (400),
        test_analyze_unsupported_type (400)
        Sample relationships from that run:
          person -> wearing -> shoe (77.61%)
          person -> walking on -> sidewalk (66.48%)
          person -> walking on -> sidewalk (68.69%)
          car -> on -> sidewalk (32.46%)
          bike -> on -> sidewalk (54.03%)

  (8) health via the project's httpx client                 PASS
        status 200, body {"status": "healthy"}

  (9) vite dev server with the documented workflow          PASS
        "VITE v6.4.3 ready in 845 ms", Local: http://127.0.0.1:5173

  Production build size grew from 388.98 kB JS / 20.30 kB CSS (pre-phase dist,
  13-09-2026) to 404.86 kB JS / 22.11 kB CSS - the delta is React Flow
  interactions, html-to-image and the new styles.

--------------------------------------------------------------------------------
24. Model-loading verification (load once, reuse)
--------------------------------------------------------------------------------
  Backend log after 2 analyze requests + 3 health checks in ONE process:
      "Loading SceneGraphPipeline (one-time)" occurrences : 1
      "SceneGraphPipeline ready" occurrences              : 1
      "Application startup complete" occurrences          : 1
      POST /analyze 200 responses                         : 2
      GET /health 200 responses                            : 3
  => the pipeline is constructed exactly once at startup and reused by every
  later request.  backend/main.py was NOT modified in this phase.

  tests/test_api_local.py also prints only ONE
  "Loading SceneGraphPipeline (one-time)..." line although it enters the
  TestClient context five times, confirming the module-level singleton is reused
  within a process too.

--------------------------------------------------------------------------------
25. Manual browser verification - what was and was NOT verified
--------------------------------------------------------------------------------
  VERIFIED WITHOUT A BROWSER (this session):
    - frontend dev server serves the app (HTML 200)
    - CORS permits http://127.0.0.1:5173
    - POST /analyze from a real JPEG -> 200 with 9 objects / 16 relationships
    - production build compiles (203 modules, 4.62s)
    - dagre TB layout on the real scene: 5 rank rows, no node overlap
    - component-source assertions for every interaction in the brief (toggle
      gate, inspector gate, pane click, edge click, node click, export config,
      dashed inverse pairs)
    - state logic: search -> ids, inspector priority, reset clears everything

  NOT VERIFIED BY ME - I cannot open a browser in this environment, so the
  following five items remain for the user to confirm visually.  They are listed
  explicitly rather than being reported as verified:
    U1. bounding boxes visually landing on the correct parts of the image (the
        percentage maths and the position:relative wrapper are verified and were
        already correct in D3, but "does it look right" is visual)
    U2. unchecking "Show object detections" visually clearing all overlays
    U3. "Download Graph PNG" actually producing a correct PNG containing only the
        graph (html-to-image + SVG serialisation can only be confirmed in a real
        browser; only the code path and the filter list are verified here)
    U4. pan / zoom / fit-to-view feel, hover states, minimap rendering
    U5. clicking empty graph canvas clearing the selection in practice (React
        Flow's pane-vs-node event separation is asserted in source, not by
        interaction)
  Suggested check order: upload datasets\visual_genome\VG_100K_2\1.jpg ->
  Analyze -> click a box -> click a node -> click an edge -> click empty canvas
  -> toggle detections off/on -> search "person" -> Download Graph PNG -> Reset.

--------------------------------------------------------------------------------
26. Architectural decisions
--------------------------------------------------------------------------------
  - Frontend only: no ML, backend, dataset or API-contract change.  The backend
    still returns SceneGraphResult.to_dict() and the frontend only consumes it;
    the JSON schema is untouched.
  - The four broken files were REWRITTEN rather than patched incrementally:
    after an interrupted edit, patching a file with unbalanced braces risks more
    silent breakage than a clean rewrite.
  - All selection/search/toggle state stays in App (plain React state).  No
    Redux, no Zustand, no extra context providers.
  - Object id ("obj_0") remains the single bridge between image, graph and
    inspector; the label is never used as identity (person x2 / tree x2 proof).
  - Inverse predicates are paired VISUALLY (dashed partner + inversePartnerId)
    instead of merged, so both relationships stay selectable and the inspector
    can represent each one.  Nothing is removed from the scene data.
  - Detections visibility is a render-time guard, not a data filter: the bbox
    layer is simply not rendered.  That makes the "boxes must not reappear"
    requirement safe by construction.
  - Inspector visibility is a render-time gate in App (scene && hasSelection),
    with a one-line hint replacing the empty card.
  - Dagre is kept - no custom layout algorithm and no graph-library change.
  - The stale D2/D3 verifier was updated rather than deleted, and extended with
    source-level assertions so it can no longer pass while testing nothing.
  - Local run workflow is two small PowerShell scripts + a README; no Docker, no
    process manager, no new tooling.

--------------------------------------------------------------------------------
27. Intentionally NOT changed
--------------------------------------------------------------------------------
  - scene_graph/* (pipeline, mapping, detector, features, predictor,
    postprocessing) - untouched.
  - notebook/models/relationship/*, the trained checkpoint and the caches.
  - datasets/visual_genome/* and every dataset file.
  - backend/main.py, the API contract, the one-time pipeline loading and the
    HF-offline environment flags added in Entry 6.
  - frontend/src/services/api.js, components/Inspector.jsx, src/main.jsx,
    vite.config.js, index.html.
  - tests/test_api_local.py, tests/test_connection.py, tests/_fetch_scene.py,
    tests/run_api_server.py.
  - The relationship confidence threshold (still the Phase B default 0.30) - not
    tuned in a frontend phase.
  - No authentication, database, cloud storage, LLM/RAG, attributes, GNN,
    analytics dashboard, chat or history features were introduced.

--------------------------------------------------------------------------------
28. Known limitations
--------------------------------------------------------------------------------
  - The flagship Phase A prediction person -> riding -> bike (21.79%) does NOT
    appear in the default output because it is below the 0.30 post-processing
    threshold (established in Phase B: 0.20 keeps it, 16 vs 17 relationships).
    Expected behaviour, not a regression; the threshold is a pipeline
    configuration value.
  - Model startup is still ~1-2 minutes on this CPU-only machine; the loading
    state warns the user instead of faking progress.
  - PNG export is a static snapshot of the current view (no interactive state,
    no re-import).
  - Inverse-pair visual grouping covers only the documented wearing/worn by pair;
    the current test scene contains no such pair, so the dashed styling is not
    exercised by the fixture.
  - Bounding-box label text can overflow near the right/top image edge
    (inherited from D3; cosmetic).
  - No frontend unit-test runner (no vitest/jest); verification uses the node
    verifier scripts plus the Python API tests, matching the D3/D4 pattern.
  - Desktop is the primary target; on small screens the panels stack and the
    graph drops to 420px.

--------------------------------------------------------------------------------
29. Exact next step
--------------------------------------------------------------------------------
  DEPLOYMENT PREPARATION - not started in this phase, by instruction.
  Concrete scope for the next phase:
    a) choose and document the deployment shape (static host for the Vite build;
       FastAPI behind a process manager or container);
    b) make the API base URL configurable at build time - it is currently the
       constant 'http://127.0.0.1:8000' in frontend/src/services/api.js - and set
       the production CORS origins in backend/main.py;
    c) add a production start path for the backend (host/port from environment)
       WITHOUT changing the one-time-load architecture;
    d) document deployment in the diary and scripts/README.md;
    e) decide how the large model files (yolov8s-world.pt, CLIP cache,
       best_full_fusion_30epoch.pt) are provided to the deployment environment,
       and account for the 1-2 minute cold start in the platform's startup /
       health-check timeout.


================================================================================
ENTRY 9 APPENDIX: FUNCTION / CLASS REFERENCE (files created or changed)
================================================================================

--------------------------------------------------------------------------------
frontend/src/App.jsx  (rewritten)
--------------------------------------------------------------------------------
export default function App()
    Owns all frontend state.  No props.  Renders: header, error box,
    empty/preview/loading states, results summary, explorer (image + graph),
    search section, hint line and the conditionally rendered inspector.

    State (useState):
      selectedFile            File | null     - chosen upload
      previewUrl              string | null   - data URL of the chosen image
      scene                   object | null   - exact /analyze JSON response
      isAnalyzing             boolean         - spinner / button state
      error                   string | null   - user-facing message
      selectedObjectId        string | null   - "obj_0" ... shared by both views
      selectedRelationshipId  string | null   - "rel_0" ...
      searchQuery             string          - label search text
      showDetections          boolean (true)  - visualisation-only toggle
      sceneKey                number          - remount key for a fresh layout
    Ref: fileInputRef (triggers the hidden <input type="file">).

    Derived (useMemo):
      searchObjectIds  (searchQuery, scene) -> string[]
          object ids whose label contains the query (case-insensitive)
      relEndpointIds   (selectedRelationshipId, scene) ->
          [rel.subject_id, rel.object_id] of the selected relationship
      hasSelection     = Boolean(selectedObjectId || selectedRelationshipId)

    Handlers:
      handleFileChange(event)
          validates the image/* type, stores the file, clears
          scene/selection/search, then reads the file into a data URL for the
          preview.  Returns early if no file; user-facing error for non-images.
      handleAnalyze()   async
          guards on selectedFile; sets isAnalyzing; clears the previous scene and
          selection; awaits analyzeImage(file); on success setScene(result) and
          bumps sceneKey; on failure stores err.message; always clears
          isAnalyzing in a finally block.
      handleReset()
          clears selectedFile, previewUrl, scene, both selection ids,
          searchQuery and error, restores showDetections = true and resets the
          file input value so the same file can be selected again.
      handleObjectSelect(objectId)
          sets selectedObjectId and CLEARS selectedRelationshipId
          (image -> graph sync entry point, shared by SceneImage and SceneGraph).
      handleRelationshipSelect(relationshipId)
          sets selectedRelationshipId and CLEARS selectedObjectId so the
          inspector priority stays unambiguous.
      handlePaneClick()
          clears BOTH ids - wired to React Flow's onPaneClick (empty canvas).

--------------------------------------------------------------------------------
frontend/src/components/SceneImage.jsx  (re-created)
--------------------------------------------------------------------------------
export default function SceneImage(props)
    Props: imageSrc, width, height, objects, selectedObjectId,
           highlightedObjectIds, searchObjectIds, showDetections, onObjectClick
    Renders .scene-image > .scene-image-wrap > <img> + a gated overlay layer, then
    a meta line.  Purely presentational: it never mutates or filters scene data.

    Local helpers:
      boxStyle(obj)
          converts the backend bbox (cx, cy, width, height in ORIGINAL image
          pixels, centre format) into percentage CSS:
            left    = (cx - width/2) / width   * 100
            top     = (cy - height/2) / height * 100
            width%  =  width / width   * 100
            height% =  height / height * 100
          Percentages (not pixels) are what keep boxes aligned at any responsive
          size.
      boxClass(obj)
          style priority: bbox-selected (red, obj.id === selectedObjectId) >
          bbox-highlighted (orange, in highlightedObjectIds = relationship
          endpoints) > bbox-searched (teal, in searchObjectIds) > bbox.
    Overlay gate: `{showDetections && (<div className="bbox-layer">...)}` - when
    false NOTHING is rendered, so no highlighting can bring boxes back.
    Each box is a <button> carrying the object id, a tooltip "label (obj_id)" and
    a click handler that stops propagation and calls onObjectClick(obj.id).

--------------------------------------------------------------------------------
CSS class inventory (frontend/src/App.css)
--------------------------------------------------------------------------------
  Layout : .app, .app-header, .header-side, .header-side-right, .header-center,
           .subtitle, .explorer, .explorer-panel, .panel-header
  Buttons: .btn, .btn-primary, .btn-secondary, .btn-ghost, .btn-sm
  States : .empty-state, .loading-state, .spinner (+ @keyframes spin),
           .error-box, .preview-section, .preview-image, .file-name
  Summary: .results-section, .results-grid, .result-stat, .stat-num, .stat-label
  Toggle : .toggle
  Image  : .scene-image, .scene-image-wrap, .scene-image-img, .bbox-layer, .bbox,
           .bbox-label, .bbox-selected, .bbox-highlighted, .bbox-searched,
           .scene-image-meta
  Graph  : .graph-wrapper, .graph-toolbar, .graph-note, .graph-container,
           .graph-empty
  Search : .search-section, .search-count, .hint-line
  Inspect: .inspector-section, .inspector, .inspector-list, .bbox-values,
           .inspector-relationships, .rel-conf, .inspector-empty
  Media  : @media (max-width: 900px) - stacks the explorer, centres the header,
           graph height 420px, single-column inspector list

--------------------------------------------------------------------------------
frontend/src/components/SceneGraph.jsx  (rewritten)
--------------------------------------------------------------------------------
Module constants
    NODE_WIDTH 170, NODE_HEIGHT 56
    COLOR_NODE_BORDER '#4a90d9', COLOR_SELECTED '#ef4444',
    COLOR_REL_ENDPOINT '#f0932c', COLOR_SEARCH '#00897b'
    EDGE_COLOR '#64748b', EDGE_COLOR_SELECTED '#ef4444', EDGE_DEFAULT_WIDTH 1.5
    INVERSE_PREDICATES = { wearing: 'worn by', 'worn by': 'wearing' }

buildNodes(objects) -> node[]
    One node per scene object: { id: obj.id, data: { label, confidence },
    sourcePosition: 'bottom', targetPosition: 'top', style: {...} }.
    Node id IS the object id; duplicate labels stay separate nodes.

findInversePartnerIds(relationships) -> { [relId]: partnerRelId }
    Scans pairs for the same two objects in opposite directions with inverse
    predicates (INVERSE_PREDICATES).  Relationships without a partner are simply
    absent from the map.  This is the ONLY place inverse detection happens; it
    never removes anything.

buildEdges(relationships) -> edge[]
    One edge per relationship: { id: rel.id, source: subject_id,
    target: object_id, label: predicate, type: 'smoothstep',
    data: { confidence, inversePartnerId }, labelStyle/labelBg* for readability,
    style: { stroke, strokeWidth, strokeDasharray '6 4' when an inverse partner
    exists } }.  No `animated` property => static edges.

layoutNodes(nodes, edges) -> node[] with computed positions
    dagre graphlib multigraph; setGraph({ rankdir: 'TB', nodesep: 70,
    ranksep: 130, marginx: 30, marginy: 30 }); setDefaultEdgeLabel(() => ({}));
    setNode(id, {width,height}); setEdge(source, target, {weight:1}, edge.id)
    (the edge id as the multigraph name keeps parallel relationships);
    dagre.layout(graph); then each node position becomes
    { x: centre.x - NODE_WIDTH/2, y: centre.y - NODE_HEIGHT/2 } because dagre
    reports centres while React Flow positions are top-left corners.

nodeBorderStyle(nodeId, selectedObjectId, relEndpointIds, searchIds) -> CSS border
    Priority: selected (red, 3px) > relationship endpoint (orange, 2.5px) >
    search match (teal, 2.5px) > default (blue, 2px).

InnerSceneGraph(props)
    Props: scene, selectedObjectId, selectedRelationshipId, searchObjectIds,
           highlightedObjectIds, onObjectSelect, onRelationshipSelect,
           onPaneClick
    - useMemo sceneData: layoutNodes(buildNodes(objects), buildEdges(rels)) +
      buildEdges(rels) - one layout per scene.
    - useNodesState / useEdgesState seeded from sceneData.
    - useEffect #1: restyles existing nodes when selection/search changes
      (border only - positions untouched).
    - useEffect #2: restyles existing edges when selectedRelationshipId changes
      (red 3px + red label when selected, else grey 1.5px).
    - handleDownload (useCallback): fitView({padding:0.1}) -> wait ~80ms ->
      toPng(wrapperRef.current, { cacheBust, backgroundColor '#0f1620',
      filter: excludes minimap/controls/panel/attribution }) -> download as
      scene-graph.png; errors caught and logged.
    - Renders .graph-empty when there are no objects (zero-object scene);
      otherwise the toolbar (with "No relationships predicted." when applicable
      and the Download button) + .graph-container wrapping <ReactFlow> with
      Background, Controls (showInteractive false) and MiniMap.
    - ReactFlow props: onNodeClick -> onObjectSelect(node.id);
      onEdgeClick -> onRelationshipSelect(edge.id); onPaneClick -> onPaneClick;
      fitView; minZoom 0.1; maxZoom 2.5; deleteKeyCode null;
      proOptions.hideAttribution; nodesConnectable/nodesDraggable false.

export default function SceneGraph(props)
    Thin wrapper rendering <ReactFlowProvider><InnerSceneGraph {...props} />
    </ReactFlowProvider> so useReactFlow() (used by fitView for the export) is
    available.

--------------------------------------------------------------------------------
frontend/verify_phase_e.mjs  (NEW)
--------------------------------------------------------------------------------
Node script (no test runner).  Reads tests/_scene.json (a real /analyze
response) and asserts the phase behaviour:
    check(name, ok, detail)                     - tiny PASS/FAIL reporter
    inspectorMode(selectedObjectId, relId)      - 'relationship' | 'object' |
                                                  'hidden' (priority rule)
    findInversePairIds(relationships)           - mirror of the component
    initialUiState() / resetUiState()           - the exact fields handleReset
                                                  must clear/restore
    searchObjectIds(objects, query)             - mirror of App's useMemo
Checks (16): data independent of the toggle, every object has an id, inspector
visibility rules (4 cases), inverse symmetry, both relationships retained, no
dangling endpoints, no "no relationship" leakage, reset semantics (4 cases),
search resolves ids, empty search highlights nothing.

--------------------------------------------------------------------------------
frontend/verify_layout.mjs  (UPDATED from D2/D3)
--------------------------------------------------------------------------------
Mirrors the CURRENT component (TB, 170x56, nodesep 70, ranksep 130, centre ->
top-left) and adds component-source checks:
    buildNodes(objects) / buildEdges(relationships)  - mirrors of the component
    layoutNodes(nodes, edges)                        - dagre TB mirror
    check(name, ok, detail)                          - PASS/FAIL reporter
Layout checks (9): node count 9/9, edge count 16/16, finite positions, no node
overlap (real rectangles), >= 2 top-to-bottom rank rows (5 found), edges mostly
flow downward (15/1), all edge endpoints exist, duplicate-label objects separate,
layout deterministic.
Source checks (12): reads SceneGraph.jsx / SceneImage.jsx / App.jsx as text and
asserts `rankdir: 'TB'`, absence of 'LR', absence of `animated`, presence of
`strokeDasharray`, `toPng` + `scene-graph.png`, minimap/controls filtering,
`onPaneClick`, `onEdgeClick`, `onNodeClick`, `hasSelection &&` + `!hasSelection`,
`showDetections={showDetections}` and `{showDetections && (`.

--------------------------------------------------------------------------------
scripts/  (no functions - plain sequential PowerShell)
--------------------------------------------------------------------------------
run_backend.ps1
    Splits $PSScriptRoot to find the repo root, cd's there, prefers
    .venv\Scripts\python.exe (falls back to `python`), prints the URLs and the
    cold-start warning, then runs
    `python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000`.
    $ErrorActionPreference = 'Stop'.  Ctrl+C stops it.
run_frontend.ps1
    cd's to frontend/, runs `npm install` only when node_modules is missing,
    prints the URL, then runs `npm run dev`.
README.md
    Written documentation: one-time setup, backend terminal, frontend terminal,
    URLs table, health check one-liner, 12-step end-to-end usage walkthrough,
    how to stop, and the test commands (build + 3 node verifiers + 2 python
    tests + how to regenerate tests/_scene.json).

================================================================================
END OF ENTRY 9
================================================================================
================================================================================
ENTRY 9 - FOLLOW-UP: BROWSER VERIFICATION (U1-U5 CLOSED)
================================================================================

Why this exists
---------------
Entry 9 (section 25) shipped the frontend-polish phase but left five items
explicitly unverified, because that session had no browser:

    U1  bounding boxes actually rendered over the image (not just DOM counts)
    U2  the detection toggle removing every overlay
    U3  the real PNG download
    U4  graph interactions (node / edge / empty-canvas clicks)
    U5  image <-> graph selection synchronisation

They were re-run in a real headless browser (Playwright driving the INSTALLED
Edge via `channel: 'msedge'`, so no 150 MB Chromium download was needed).
Every check passed.  U1-U5 are closed, and this section supersedes the
"NOT VERIFIED BY ME" lines in section 25.

How it was run
--------------
frontend/verify_browser.mjs  (NEW, no test runner)
    launchBrowser()      - tries bundled Chromium, then installed msedge,
                           then installed chrome; reports which one it used
    check(name, ok, d)   - tiny PASS/FAIL reporter
    shot(name)           - full-page screenshot per step
    clickFirstBBox()     - element.click() via evaluate(), so overlapping
                           boxes can never make the click miss
    clickFirstEdge()     - prefers the edge label, falls back to the SVG
                           interaction stroke
    readCounts()         - { bbox, nodes, edges }

One-time setup (Playwright is deliberately NOT a project dependency -
verified: no playwright entry in package-lock.json, no manifest change):

    cd frontend
    npm install --no-save playwright

Then, with both servers up (scripts\run_backend.ps1, scripts\run_frontend.ps1):

    cd frontend
    node verify_browser.mjs

Results - 40 checks, all PASS, zero uncaught page errors
--------------------------------------------------------
Boot / empty state
    app loads; title and header both "Scene Graph Explorer"
    empty state "Upload an image to explore its scene graph" visible
    no inspector exists before a scene does
Preview / analysis
    preview image and file name appear as soon as a file is chosen
    "Analysing image..." loading state is shown during inference
    result: 9 boxes / 9 nodes / 16 edges - matches tests/_scene.json exactly
    inspector stays hidden; the hint line is shown instead
U1 - boxes render
    every box carries its label and id, e.g. "person (obj_0)"
    labels + confidences legible in the screenshot (person 77%, bike 56%,
    car 65%, sidewalk 30%, shoe 37% ...)
U2 - toggle is visualisation-only
    unchecking hides ALL boxes and removes .bbox-layer from the DOM
    graph untouched by the toggle: still 9 nodes / 16 edges
    re-checking restores all 9
    caption switches to "9 objects - detections hidden"
U5 - selection sync, image -> graph
    clicked box gains .bbox-selected; matching node border becomes
    "3px solid rgb(239, 68, 68)"
    inspector opens with "Object Details", the matching label, 4 dt/dd rows
U5 - selection sync, graph -> image
    clicking node obj_1 highlighted box "person (obj_1)"
    the inspector followed the graph selection
U4 - edge click
    clicking an edge label shows "Relationship Details"
    reads exactly person / wearing / shoe / 77.61%
    BOTH endpoint boxes highlighted in the image (2 found)
U4 - empty canvas
    a click on the pane clears the inspector and every highlight
Search
    "person" -> 2 boxes highlighted, count text "2 matches", and the same 2
    nodes turn teal in the graph (border rgb(0, 137, 123))
    clearing the query removes all highlights
U3 - export
    download event fired; suggested filename "scene-graph.png"
    real PNG magic bytes, 46,451 bytes, dimensions 735x560
    735x560 proves it captured the GRAPH PANEL, not the page - the browser
    viewport is 1600x1000, so a whole-page capture would be far larger
Reset
    back to the empty state; explorer, inspector and search all gone
Second image (VG_100K_2/100.jpg)
    re-analysed cleanly (2 boxes / 2 nodes / 1 edge), no stale selection


Artefacts left on disk (evidence for the reader)
------------------------------------------------
frontend/_browser_shots/00-empty-state.png
frontend/_browser_shots/01-preview.png
frontend/_browser_shots/02-analyzed.png              <- boxes + graph
frontend/_browser_shots/03-detections-hidden.png     <- toggle off
frontend/_browser_shots/04-object-selected.png
frontend/_browser_shots/05-relationship-selected.png <- edge + 2 endpoints
frontend/_browser_shots/06-selection-cleared.png
frontend/_browser_shots/07-search.png
frontend/_browser_shots/08-after-reset.png
frontend/_browser_shots/09-second-image.png
frontend/_browser_shots/scene-graph-download.png     <- the real export
The folder is disposable and is now git-ignored.

Repo changes made in this follow-up
-----------------------------------
frontend/verify_browser.mjs   NEW - the browser verifier described above
frontend/.gitignore           NEW - node_modules/, dist/ and the local
                              verification artefacts.  There was no frontend
                              .gitignore at all, so a commit of frontend/ would
                              have swept in node_modules/ and dist/
scripts/README.md             UPDATED - section 7 now documents the browser
                              verifier and its one-time setup
.vite/ (repo root)            DELETED - stray Vite dep cache left by an earlier
                              run started in the wrong directory
frontend/_browser_log.txt, frontend/_dev_log.txt, frontend/_pw_install.txt
                              DELETED - temporary run logs
frontend/package.json, frontend/package-lock.json
                              UNCHANGED - playwright installed with --no-save
                              and confirmed absent from the lockfile

Regression re-run after the edits
---------------------------------
    cd frontend; npm run build
        -> 203 modules transformed
           dist/assets/index-C_6FGxYw.js   404.86 kB (gzip 132.96 kB)
           dist/assets/index-5bpIVK92.css   22.11 kB (gzip   4.30 kB)
           built in 3.63s
    The backend and Vite dev servers were then stopped; ports 8000 and 5173
    are free again.

Cosmetic observations (NOT fixed - not defects)
-----------------------------------------------
  * Adjacent labels can overlap (e.g. "person 77%" over "bike 56%") when two
    objects are close together; labels are top-left anchored by design.
  * A long edge label (e.g. "on a ...") can be partly covered by a node the
    edge runs under; the edge itself and its tooltip stay correct.
  * The React-Flow minimap reads as a light mask over a dark node area; it is
    legible and matches the D4 design.

Status
------
Entry 9 is complete AND independently verified end to end in a real browser,
including the two items that previously carried the most risk - U1 box
alignment and U3 the PNG export.  Nothing from the frontend-polish phase
remains outstanding.  The next phase is deployment preparation (diary section
29): configurable API base URL, production CORS, env-driven host/port without
touching the load-once architecture.  That phase is explicitly NOT started
here.

================================================================================
END OF ENTRY 9 - FOLLOW-UP

================================================================================
## Entry 10 - FINAL UX POLISH + GRAPH INTERACTION + INFERENCE PERFORMANCE

Date: 2026-09-22

Objective:
Make the verified, functional application feel like a deployed CV product and
close the known rough edges, WITHOUT touching the ML architecture, the trained
checkpoint, the API contract or the load-once pipeline design:

  1.  clean centered loading state (no implementation details)
  2.  centered landing / empty state
  3.  general visual polish (dark theme kept, restrained accents)
  4.  slightly curved graph edges
  5.  draggable nodes restored + lock/unlock control
  6.  React Flow controls restyled for the dark graph
  7.  correct selected-object / relationship overlays (selection survives the
      detection toggle)
  8.  equal image / graph panel heights
  9.  compact download action (same graph-only PNG export)
  10. duplicate object numbering (display only)
  11. inference performance investigation + safe optimization

 ----------------------------------------------------------------------
 1. EXISTING STATE INSPECTED
 ----------------------------------------------------------------------
Everything built in Entries 1-9 (pipeline, FastAPI, React explorer, image<->graph
sync, inspector, search, PNG export, dark theme, run scripts, browser
verification) was in place and passing its verifiers.  Known issues going in:
the loading screen explained CPU timing to the user, the empty state was a
flat top-anchored block, edges were straight smoothstep lines, nodes were
hard-locked (nodesDraggable={false}) with no control, the two explorer panels
had unequal heights, and a warm analyze() request took ~22-25 s on this
CPU-only machine.

 ----------------------------------------------------------------------
 2. PERFORMANCE INVESTIGATION (before touching anything)
 ----------------------------------------------------------------------
New tool: tests/_profile_pipeline.py - a stage-level profiler that drives the
pipeline's OWN components (detector, visual_encoder, predictor,
postprocess_relationships) so the measured paths are the runtime paths.  It
also has a --sweep mode for batch-size tuning.  Run on the reference image
datasets/visual_genome/VG_100K_2/1.jpg, 8 torch threads, no CUDA, backend
stopped (idle machine):

BEFORE (per-pair path, per request):
    pipeline construction (once)      31.63 s
    image load + RGB convert           0.129 s
    YOLO-World detect                  1.334 s   (9 detections -> 9 objects)
    pair enumeration                   0.000 s   (68 ordered pairs)
    features + model, ONE CALL/PAIR   22.190 s   (326.3 ms/pair)
    post-processing                    0.001 s   (16 relationships kept)
    analyze() request 1               25.432 s
    analyze() request 2               22.035 s  (same instance, no reload)

Attribution run (same profiler): per pair, the frozen CLIP ViT-B/32 forward on
one 224x224 crop costs ~215.9 ms while the relationship model forward costs
~0.8 ms.  The relationship model was NEVER the bottleneck: 68 separate CLIP
forward passes were.  Pair enumeration, geometry, post-processing and
serialization are all < 1 ms each.  Union-crop + open_clip preprocessing is
pure PIL/numpy CPU work inside the CLIP number.
 ----------------------------------------------------------------------
 3. PERFORMANCE OPTIMIZATION (batching only, semantics preserved)
 ----------------------------------------------------------------------
scene_graph/features.py  - new extract_pair_features_batch(): identical
    expanded-union-crop -> 8-D geometry -> open_clip preprocess work per pair,
    but the CLIP forward runs once per batch of preprocessed crops instead of
    once per crop.  Returns (N, 512) visual + (N, 8) geometry in input order.
scene_graph/predictor.py - new predict_batch(): the SAME eval-mode model on
    (N,512)/(N,)/(N,8) tensors in chunks; each output row depends only on its
    own input row (per-row MLP, dropout inactive), so softmax/argmax results
    are identical to per-pair calls.
scene_graph/pipeline.py  - analyze() now takes the batched path by default
    (batch_inference=True); constructor gains pair_batch_size=8 and
    model_batch_size=32; the per-pair path is RETAINED (batch_inference=False)
    as the reference implementation for equivalence tests.  Per-stage timings
    are recorded on SceneGraphResult.timings - internal only, deliberately NOT
    serialized by to_dict(), so the API contract is untouched.  backend/main.py
    required NO changes (batching lives in the pipeline; default on).

Batch-size sweep (tests/_sweep2.txt, 2 interleaved rounds on 68 pairs):
    batch  1: 26.3 s (387.3 ms/pair)   batch  8:  9.5 s (139.6 ms/pair)
    batch  4: 13.9 s (203.9 ms/pair)   batch 16: 11.6 s (170.7 ms/pair)
    batch 32: 13.6 s (200.5 ms/pair)   batch 64: 15.8 s (232.6 ms/pair)
Model-side: 68 pairs in chunks of 32 -> 0.005 s total.
Chosen defaults: pair_batch_size=8 (clear winner on this box), model_batch_size=32.

 ----------------------------------------------------------------------
 4. BEFORE / AFTER TIMING (same image, same models, same weights)
 ----------------------------------------------------------------------
    CLIP forwards per request        68 (1/pair)  ->  9 (ceil(68/8))
    batched features total           6.985-8.459 s (102.7-124.4 ms/pair)
    batched relationship model       0.003-0.004 s
    per-pair path (same process)    14.417-22.933 s
    end-to-end speedup               x2.06 (same-process A/B) to x3.28
                                     (profiler breakdown)
    analyze() request 1 (batched)    7.011 s  (detect=0.31, features=6.65)
    analyze() request 2 (batched)    7.500 s  (detect=0.34, features=7.15)
    equivalence run batched 9.09 s / repeat 5.53 s / per-pair ref 15.56 s

Equivalence proof - new tests/test_batching_local.py on the real pipeline and
image: identical objects, identical bboxes, equal relationship count,
identical (subject, predicate, object) structure, confidences equal within
1e-4 (measured max delta 2.38e-07), no dangling endpoints, no class-50
leakage, one instance serves repeated requests, batch sizes configurable.
RESULT: PASS (12/12 checks).

Honest note: ~5.5-9 s per request remains, now almost entirely CLIP forward
time (68 crops / batch 8 on 8 CPU threads).  The next big cut would need a
smaller encoder or a GPU - both explicitly out of scope for this phase.
 ----------------------------------------------------------------------
 5. UI CHANGES (loading, landing, general polish)
 ----------------------------------------------------------------------
frontend/src/App.jsx (now 409 lines) + frontend/src/App.css (now 908 lines):

Loading state: the "Detection and relationship prediction run on CPU and can
take up to a minute." sentence is GONE.  What remains is a centered loading
card (spinner + "Analyzing image..." + the chosen file name).  No percentage,
no countdown, no timing promise - nothing fake.  The section is a flex box
that centers both axes, so the card sits in an intentional content region.

Landing / empty state: a centered hero with (a) a small abstract graph-motif
SVG (inline, no asset, no stock images), (b) the app headline + one-line
description, (c) a single "Choose Image" call to action, (d) a format hint.
The hidden file input is now ALWAYS mounted so the picker is reachable from
every state; the header shows the file actions only once a file/scene exists;
the preview state offers a "Run analysis" action next to the header Analyze.

General polish (dark theme kept, one blue accent family + status colors):
  - typography hierarchy: uppercase letter-spaced panel/section titles,
    brand mark in the header, consistent type scale
  - buttons: consistent radii/borders, hover/active/disabled states,
    sm/lg variants; compact toolbar buttons in the graph panel
  - cards/panels: one surface system (panel bg, border, subtle shadow),
    stat cards for the object/relationship counts
  - the detection checkbox became a styled pill switch (real checkbox input
    underneath - id="detections-toggle" stays testable)
  - search: icon + count chip, focus ring
  - inspector: uppercase section titles, dl grid, mono bbox values
  - React Flow controls: dark background, visible borders + hover states
    (no more default white buttons); the minimap got a dark skin
  - restrained motion: spinner + hover transitions only
 ----------------------------------------------------------------------
 6. GRAPH CHANGES (edges, dragging, lock, controls)
 ----------------------------------------------------------------------
Curved edges: edge type changed 'smoothstep' -> 'default' (React Flow's
built-in BEZIER edge) with pathOptions: { curvature: 0.32 } (subtle, no
loops).  Verified in the DOM: edge paths are now "C" bezier curves (e.g.
"M755,459 C755,523 745,523 745,587").  Labels, label backgrounds, the dashed
inverse-pair styling and selection highlighting are unchanged.  Layout stays
dagre top-to-bottom and deterministic.

Draggable nodes restored + lock: nodesDraggable is now bound to the lock
(nodesDraggable={!locked}); the previous hard nodesDraggable={false} is gone.
Node positions live in the graph component's node state, so manual positions
survive selection/search/restyling changes - dagre runs ONLY in the
sceneData useMemo (deps: objects, relationships, displayLabels); a NEW scene
still gets a fresh layout because App remounts the explorer via key={sceneKey}.
Browser proof: unlocked drag moved a node in graph space, and the dragged
position survived select/deselect with 0.0 px drift.

Lock control: a toolbar toggle button (lock icon + "Unlocked"/"Locked",
aria-pressed) integrated into the graph toolbar.  State (graphLocked) lives
in App so it survives graph remounts; Reset restores the default UNLOCKED.
The lock is pure UI state - it never touches scene.objects/relationships.
Behavioral note discovered while verifying: with the lock ON, React Flow
pans the canvas from a pointer-down on a node (screen box moves, graph
position does not).  That is exactly "nodes cannot be dragged, pan/zoom keep
working" - the browser checks now assert graph-space immobility + viewport
movement (see section 12 for the measurement bug this exposed in MY test,
not in the app).
 ----------------------------------------------------------------------
 7. DETECTION VISIBILITY + SELECTION HIGHLIGHTS (Parts 9/10/15)
 ----------------------------------------------------------------------
App.jsx now computes a visibleObjectIds list; SceneImage paints exactly those
boxes (it never re-enables the whole layer for a selection):
    detections ON                            -> all objects
    detections OFF + relationship selected   -> the two endpoints only
    detections OFF + object selected         -> that object only
    detections OFF + nothing selected        -> none
scene.objects is never filtered or mutated.  Captions say
"9 objects - detections hidden" / "... - showing N selected".
Highlight priority in both views: selected (red #ef4444) > relationship
endpoints (orange #f0932c) > search matches (teal #00897b).  Graph-side
styling unchanged and verified.  Browser-verified all four cases plus
re-checking the toggle (9 boxes return), the captions, and that endpoints
use the ORANGE relationship highlight (not the red object highlight).

 ----------------------------------------------------------------------
 8. PANELS, DOWNLOAD, DUPLICATE LABELS, INSPECTOR (Parts 8/11/12/13/14)
 ----------------------------------------------------------------------
Equal panels: desktop explorer now fixes BOTH panel heights (564 px measured
equal in the browser) with the graph viewport filling its panel (flex) and
the image fitting its panel body (aspect-ratio wrapper + max-height cap, so
the percentage-based boxes stay aligned).  <=900 px still stacks (image auto
height, graph 420 px).  Verified: panels 564 vs 564 px, image keeps its
1.333 aspect, all 9 boxes inside the image rectangle.

Download: "Download Graph PNG" -> compact "[down-arrow] Download"; the export
code (html-to-image toPng of the graph container, minimap/controls/panels
filtered out, fitView first) is unchanged.  Verified download: valid PNG,
739x451 (graph panel, not the page), 73,946 bytes.

Duplicate labels (display only): App.jsx builds a displayLabels map
(id -> "person 1"/"person 2"; singles keep the plain label) and passes it to
SceneImage, SceneGraph and Inspector.  object.id / label / subject_id /
object_id / backend JSON are untouched; node ids stay obj_N (verified in the
DOM).  Search still matches the BASE label ("person" finds both).
Browser-verified: "person 1 81% / person 2 77%", "tree 1 / tree 2",
car/bike/vehicle/shoe/sidewalk unnumbered, graph nodes numbered the same.

Inspector: gating rules unchanged (no scene -> nothing; scene + no selection
-> hint line only; object -> Object Details; relationship -> Relationship
Details; empty-canvas click clears + hides).  Subject/object/outgoing/
incoming names now use the display labels; the internal id row keeps obj_N
visible so duplicates can never be confused.
 ----------------------------------------------------------------------
 9. FILES CREATED
 ----------------------------------------------------------------------
    tests/_profile_pipeline.py      stage profiler + batch-size sweep
    tests/test_batching_local.py    batched == per-pair equivalence test
    frontend/verify_interaction.mjs interaction-contract verifier (lock,
                                    visibility rules, numbering, curvature,
                                    panel sizing, clean loading copy)
    frontend/patch_tmp.mjs          throwaway patch helper used once during
                                    an edit; DELETED again (not kept)

 ----------------------------------------------------------------------
 10. FILES MODIFIED
 ----------------------------------------------------------------------
    scene_graph/features.py         + extract_pair_features_batch /
                                    PairBatchFeatures (per-pair helpers kept)
    scene_graph/predictor.py        + predict_batch / BatchPrediction
    scene_graph/pipeline.py         batched analyze() path (default), pair/
                                    model batch size ctor args, timings
    frontend/src/App.jsx            hero/loading/lock/visibleObjectIds/
                                    displayLabels/equal panels wiring
    frontend/src/App.css            full visual polish pass (hero, loading
                                    card, panels 564px, switch, controls,
                                    minimap, toolbar, search, inspector)
    frontend/src/components/SceneGraph.jsx   bezier edges + pathOptions,
                                    nodesDraggable={!locked}, lock button,
                                    compact Download, display labels
    frontend/src/components/SceneImage.jsx   visibleObjectIds contract,
                                    display labels, aspect-ratio wrapper
    frontend/src/components/Inspector.jsx    display labels via labelFor
    frontend/verify_browser.mjs     browser suite extended for U2/U6/U7/U8/U9
                                    + three verifier-side bug fixes (sec. 13)
    frontend/verify_layout.mjs      + curved-edge and drag/lock source checks
    scripts/README.md               documents the new verifier + profiler
    post_training_full_stack.md     this entry

 ----------------------------------------------------------------------
 11. DEPENDENCIES ADDED
 ----------------------------------------------------------------------
NONE.  No new npm or pip packages (batching uses torch/PIL/open_clip already
present; Playwright stays a --no-save verification-only install).
 ----------------------------------------------------------------------
 12. TESTS + BROWSER VERIFICATION
 ----------------------------------------------------------------------
Static/component verifiers (all PASS, real tests/_scene.json):
    verify_layout.mjs, verify_phase_e.mjs, verify_d4_logic.mjs,
    verify_interaction.mjs (new)
Build: vite build - 203 modules, index JS 407.93 kB (gzip 133.90 kB),
CSS 26.80 kB (gzip 5.24 kB), 2.77 s.  Note: "npm run build" exits 1 with no
output on this box (npm 11.5.2 / Node 22 environment quirk); invoking vite
directly (node node_modules/vite/bin/vite.js build) succeeds - documented so
the next phase knows the workaround.
Backend: test_batching_local.py PASS (12/12); test_connection.py PASS
against the live servers (health, CORS, analyze 9 obj / 16 rel);
test_api_local.py, test_pipeline_local.py and test_postprocessing_local.py
run after the browser suite (results in the run log).

Browser verification (Playwright, installed Edge; real backend + Vite dev
server; reference image 1.jpg + second image 100.jpg):
    run 1: every UI/interaction check passed EXCEPT 6 fails, ALL of which
           were verifier defects (section 13); the app behaviors themselves
           confirmed the whole toggle matrix, selection sync, search,
           download (valid graph-only PNG), reset, second image, no errors.
    run 2 (after fixing the numbering regex + case-insensitive headings):
           FAIL (3) - the two U8 checks measured the wrong quantity
           (section 13, bug 4).
    run 3 (graph-space measurement + panel-aware pan origin): final run;
           full log in frontend/_browser_log4.txt, 16 screenshots in
           frontend/_browser_shots/ (08b-node-dragged, 08c-graph-locked,
           scene-graph-download.png, ...).
 ----------------------------------------------------------------------
 13. BUGS FOUND + FIXED (this phase)
 ----------------------------------------------------------------------
 1. Loading screen leaked implementation details ("CPU ... up to a minute")
    -> removed; verifier now greps the sources and the live loading card for
    CPU/second/percent/YOLO/CLIP/model-loading wording.
 2. SceneImage.jsx contained a duplicated component tail left by an earlier
    interrupted edit (two "export default function SceneImage" definitions)
    -> cleaned to a single definition (136 lines).
 3. verifier false FAIL: "car 65%" matched /^(car) \d/ because the confidence
    made the text look like "car 6..." -> strip the confidence before the
    numbering check (the app's numbering itself was correct).
 4. verifier false FAIL x2: (a) the locked-graph check measured SCREEN
    movement, but a locked pointer-down on a node PANS the canvas, so the
    screen box moves while the graph position stays fixed -> the check now
    parses the node's inline transform (graph space); (b) the empty-area pan
    origin could land on the zoom controls (React Flow panels were not
    excluded) and the transform was read via getAttribute instead of the
    live style -> origin scan avoids .react-flow__panel and reads
    el.style.transform.
 5. verifier false FAIL: exact-text heading compare ("Object Details") vs CSS
    text-transform uppercase -> case-insensitive compare (app text source
    unchanged).
 6. npm run build quirk (exit 1, no output) -> documented; direct vite build
    invocation used as the working command.

 ----------------------------------------------------------------------
 14. ARCHITECTURAL DECISIONS
 ----------------------------------------------------------------------
 - Batching is a SCHEDULING change only; per-pair code kept as the reference
   path so equivalence stays testable (pipeline.batch_inference flag).
 - Batch sizes are constructor parameters, not constants (the sweep optimum
   is machine-specific; a GPU box would choose differently).
 - Stage timings live on SceneGraphResult.timings, deliberately excluded
   from to_dict(): the API contract stays exactly {image, objects,
   relationships}.  backend/main.py is UNCHANGED this phase.
 - Display labels are an App-level memo threaded down as a prop map; the
   scene is never rewritten (single source of truth preserved).
 - Lock state is lifted to App (survives remount) and gates nodesDraggable
   only; pan/zoom props untouched.
 - Drag persistence comes from keeping positions in useNodesState; dagre is
   re-run only for a NEW scene (sceneKey remount), never on selection.
 - Playwright remains --no-save verification-only; package.json unchanged.

 ----------------------------------------------------------------------
 15. INTENTIONALLY NOT CHANGED
 ----------------------------------------------------------------------
 checkpoint/weights, model architecture, class/predicate/object mappings,
 Visual Genome data, YOLO-World, React Flow, FastAPI app + endpoints + CORS,
 response schema, run scripts' workflow, dagre TB layout, inverse-pair
 dashed styling, PNG export implementation, no auth/db/LLM/router additions.

 ----------------------------------------------------------------------
 16. REMAINING LIMITATIONS
 ----------------------------------------------------------------------
 - Warm request ~5.5-9 s on CPU, dominated by CLIP forward time; a GPU or a
   smaller encoder would cut it further (out of scope by design).
 - Cold pipeline load ~20-37 s at process start (documented in the run
   scripts; the load-once design is intentional).
 - The batch-size optimum (8) is specific to this machine/thread count.
 - Adjacent bounding-box labels can still overlap when objects are close
   (cosmetic, unchanged from Entry 9).
 - The browser drag checks depend on synthetic pointer events; they are
   empirical (Edge/Chromium) and measure graph-space transforms.
 - npm run build quirk on this box (see section 12).

 ----------------------------------------------------------------------
 17. NEXT PHASE (EXPLICITLY NOT STARTED)
 ----------------------------------------------------------------------
 DEPLOYMENT PREPARATION: configurable API base URL, production CORS list,
 env-driven host/port, build/deploy notes - without touching the load-once
 pipeline architecture or any endpoint contract.

================================================================================
END OF ENTRY 10 - FINAL UX POLISH + GRAPH INTERACTION + INFERENCE PERFORMANCE
================================================================================








================================================================================

================================================================================
## Entry 11 - DEPLOYMENT AUDIT (INSPECT -> MEASURE -> DOCUMENT)

Date: 2026-09-24

Objective:
Inspect the complete repository and determine exactly what a deployment would
require. AUDIT ONLY - no deployment, no Docker files, no architecture / ML /
API / React changes, no requirements edits, no dataset moves, no commits.
Full report: DEPLOYMENT_AUDIT.md at the repository root (18 sections).

1. Files inspected
--------------------------------------------------------------------------------
 - backend/main.py (full), scene_graph/*.py (all 7), datasets/visual_genome/
   {__init__,loader}.py, notebook/models/relationship/*.py imports, dataset
   models.py via exports, scripts/run_*.ps1 + scripts/README.md, tests/* listing,
   frontend/{package.json,vite.config.js,index.html,.gitignore,src/**},
   requirements.txt, root .gitignore, README.md, post_training_full_stack.md,
   .venv ultralytics/{tasks.py,text_model.py,world/*,settings}, clip/clip.py,
   pip 26.2.1 req_file BOM handling, git ls-files/check-ignore/status.

2. Measurements (this session)
 - Sizes: datasets 43,845 MB (VG_100K 9,352 / 64,346 files; VG_100K_2 5,265 /
   43,903 files; zips 14,504; VG-SGG.h5 67.6; image_data.json 16.8), notebook
   1,753 (preprocessed_cache 1,708), .venv 1,458 (torch 494, no nvidia/*),
   HF cache 577 MB (timm/vit_base_patch32_clip_224.openai safetensors),
   C:\Users\psytr\weights\clip\ViT-B-32.pt 337.6 MB, repo weights 28.3 MB,
   frontend/dist 434 KB, repo total excl .venv/node_modules/.git 45,608 MB.
 - Cold start: 45.2 s fresh process -> Application startup complete (warm cache);
   historical docs: 20-37 s / 50-120 s / 1-2 min under load.
 - Warm inference: 8.32 s live /analyze (800x600, 9 objects, 16 relationships);
   profile reference 7.0-7.5 s.
 - Warm backend RAM (live PID): private 2,536 MB, WS 455 MB, peak WS 1,847 MB,
   peak paged ~3,538 MB, 33 threads; torch 8 threads, cuda False, 2.13.0+cpu.
 - Environment: a dev backend (port 8000) and Vite dev server (5173) were
   already running from an earlier session and were left untouched; the audit's
   own probe instance (port 8001) never started due to a command quoting error -
   no stray process remains; its completed startup log provided the 45.2 s figure.

3. Deployment audit findings (headline)
 - Inference needs ~85 MB VG metadata + 943 MB repo weights + 914.6 MB CLIP
   caches + code; it does NOT need the 43.8 GB image dataset (load_scene is
   never called on the analyze path; loader just opens h5/json at startup).
 - Two distinct CLIP copies: ultralytics text tower (weights_dir/clip, startup)
   and open_clip visual tower (HF cache, per request, offline flag forced).
 - requirements.txt: 82 pins, UTF-16LE, one git dep, CPU index NOT pinned
   (local venv happens to be +cpu), ~30 training/Jupyter-only packages,
   python-multipart present, opencv non-headless. NOT modified (report-only).
 - Frontend: build = vite build -> dist/ 434 KB; API base hardcoded
   127.0.0.1:8000; no env var usage; npm run build quirk still documented.
 - Backend: env-less config, dev-only CORS, load-once survives production with
   exactly 1 worker (~2.5 GB each), in-memory uploads, no temp files, no DB.
 - Security: NO secrets found anywhere; empty ultralytics api_key fields;
   localhost/absolute-path assumptions only in dev files (listed with lines).
 - Git: all 51 application files UNTRACKED; tests scratch not ignored;
   *.pt rule keeps both checkpoints out of git; caches live outside the repo.

4. Deployment options presented (no winner chosen)
 A) single always-on VM/container serving dist + API; B) split CDN + backend
 PaaS; C) serverless (fundamentally mismatched: 45-120 s init, 2.5 GB commit).
 Single-server analysis: technically straightforward (StaticFiles), removes
 CORS entirely, one cold start, vertical scaling; separation only buys CDN
 conveniences. Proposed deployment/ file tree: ~2.5-3 GB total with env.

5. Blockers
 CRITICAL: hardcoded API URL; dev-only CORS; no weight/cache transport story
 (+HF_HUB_OFFLINE on fresh hosts); requirements not Linux/CPU/git-ready;
 app code untracked; cold start vs serverless windows.
 HIGH: config not env-driven; no Linux process definition (workers=1);
 tests scratch gitignore gap; no LICENSE vs AGPL stack; no rate limit on
 /analyze; npm build quirk for CI.  (MEDIUM/OPTIONAL: see report section 17.)

6. Tests executed
 - No application tests were modified or required; the audit validated its
   measurements against the live server (/health, one /analyze 200 - 9 obj /
   16 rel), uvicorn startup log, git plumbing, and pip BOM support.

7. Files created / modified
 - CREATED: DEPLOYMENT_AUDIT.md (report, 18 sections).
 - MODIFIED: post_training_full_stack.md (this entry appended only).
 - Nothing else touched. No commits. No deployments.

8. Known issues / limitations of this audit
 - RAM figures come from a dev server started with tests/run_api_server.py
   (same app object); page-cache conditions favour the 45.2 s cold start.
 - pip UTF-16 parsing verified via pip source BOM table, not a full install.
 - Secrets sweep of app source found nothing; full-disk sweep was bounded.

9. Next phase (explicitly NOT started)
 Commit the app -> config phase (API URL/CORS/host-port env) -> requirements
 normalization (UTF-8, split, CPU index) -> weight/cache provisioning decision
 -> owner picks architecture -> containerize and deploy (first real deploy).

================================================================================
END OF ENTRY 11 - DEPLOYMENT AUDIT (INSPECT -> MEASURE -> DOCUMENT)
================================================================================

================================================================================
## Entry 12 - PHASE 1: GIT HARDENING + FIRST APPLICATION COMMIT + PUSH
================================================================================

Date: 2026-09-24

Objective (Phase 1 of the 10-phase deployment plan, EXACTLY as approved):
harden .gitignore, stage the full application source with a verified
dry-run, create exactly ONE commit, push to origin/main, record the result.
Phase 2 (config work) NOT started.

1. .gitignore rules added
 - .env, .env.*, !.env.example  (secrets; template explicitly allowed)
 - .pytest_cache/               (test runner cache)
 - tests/_*.txt, tests/_scene.json, tests/_fetch_scene.py,
   tests/_profile_pipeline.py, tests/_sweep2.txt  (profiling scratch)
 - !notebook/models/yolov8s-world.pt
 - !notebook/models/best_full_fusion_30epoch.pt
   (negation ONLY for the two inference checkpoints; the general *.pt
    rule at line 22 still covers every other .pt, including anything
    under notebook/models/preprocessed_cache/)
 Probes after edit: both checkpoints un-ignored; other .pt still
 ignored; .env/.env.local/.pytest_cache/all five tests scratch patterns
 ignored; tests/*.py code NOT ignored; VG_100K images, .venv/,
 frontend/dist still ignored.

2. Explicitly tracked checkpoints (the ONLY two .pt files in git)
 - notebook/models/yolov8s-world.pt            27,169,314 B (25.91 MB)
 - notebook/models/best_full_fusion_30epoch.pt  2,470,857 B (2.36 MB)
 git ls-files -- '*.pt' returns exactly these two and nothing else.

3. Large artifacts EXCLUDED (as mandated)
 - 43.8 GB Visual Genome IMAGE datasets: datasets/visual_genome/VG_100K/
   and VG_100K_2/ remain ignored; ZERO image files staged and ZERO
   tracked (VG_100K* tracked-image count = 0; check-ignore confirms).
 - CLIP caches excluded: ultralytics text-tower cache (337.6 MB,
   weights_dir outside repo) and HF open_clip cache (577 MB) both live
   OUTSIDE the repository; transport is a Phase 4 decision (bake-in vs
   first-boot download), not a git concern.
 - node_modules, frontend/dist, .venv, __pycache__, .pytest_cache,
   preprocessed caches, browser/profiling scratch: all still ignored;
   zero of each in the staging set.
 - No file >= 1 MB from datasets/ staged (zero datasets/ files staged
   at all). Only files >= 1 MB in the whole staging set were the two
   approved checkpoints.

4. Staging verification (dry-run BEFORE any real staging)
 - git add -An . -> 46 new-file lines; +2 modified tracked files
   (.gitignore, requirements.txt) = 48 paths total.
 - Assertions over candidates ALL PASS:
     VG_100K/VG_100K_2 image files ......... 0
     datasets/ files >= 1 MB ............... 0  (datasets/ files: 0)
     .env files ............................ 0
     __pycache__ files ..................... 0
     node_modules files .................... 0
     frontend/dist files ................... 0
     local caches (pytest/preprocessed/browser) .. 0
     tests scratch (_*) files .............. 0
     personal/system files ................. 0
     files >= 1 MB ......................... 2 (approved checkpoints)
 - Contents matched the approved enumeration exactly: backend/ (2),
   scene_graph/ (8), frontend/src/ (11) + frontend configs
   (package.json, package-lock.json, vite.config.js, index.html,
   .gitignore) + 6 small frontend/verify_*.mjs verification scripts,
   scripts/ (3), tests/*.py (7 code files), notebook/__init__.py,
   notebook/models/__init__.py, DEPLOYMENT_AUDIT.md,
   post_training_full_stack.md, requirements.txt (M), .gitignore (M),
   and the two checkpoints.
 - git diff --cached --name-status after real staging: 48 lines
   (46 A + 2 M); checkpoints present as A; datasets/ lines = 0.
 - Nothing unexpected found -> proceeded (no stop condition hit).

5. Commit
 - Exactly ONE commit:
   e87825f69887553797028cef7d2ba21f68522be9
   'Deployment preparation: add application source, deployment docs,
    gitignore hardening'
   48 files changed, 13427 insertions(+) (no deletions).
 - No force-push. No history rewrite. Parent: f29d294 ('bug fixes').

6. Push result
 - PUSHED: f29d294..e87825f  main -> main  on
   https://github.com/ritu-nandhan-7/scene-graph.git
 - git ls-remote origin refs/heads/main returns
   e87825f69887553797028cef7d2ba21f68522be9 (matches local HEAD).
 - Note: first interactive push attempts exceeded the tool's 30 s
   command timeout (28 MB payload + credential handshake); the push was
   re-run detached with log capture and completed normally. One
   transient .git/index.lock collision occurred between parallel
   read-only dry-run probes (git add -n); it cleared itself and the
   identical dry-run output was confirmed repeatedly before staging.

7. Final git state (after push, before this entry)
 - git status: clean ('## main...origin/main', no ahead/behind, no
   modified, no untracked; temporary push log files deleted).
 - git ls-files: 74 tracked files total.
 - Tracked checkpoints: both present. Tracked .pt files: exactly 2.
 - Tracked VG_100K image files: 0 (43.8 GB stays untracked AND ignored).
 - Tracked .env files: 0. Tracked __pycache__/node_modules/dist: 0.
 - Pre-existing, UNCHANGED tracked large files from earlier history
   (NOT part of this commit, reported for transparency):
     datasets/visual_genome/VG-SGG.h5         67.6 MB (required at boot)
     datasets/original/VG150_curated.zip      22.5 MB
     datasets/visual_genome/image_data.json   16.8 MB
     plus 5 training notebooks (1.2-6.3 MB each).
   Per Phase 1 rules nothing was added to OR removed from these; any
   later decision to drop them must be explicit and must not rewrite
   published history.

8. Phase 0 finding carried forward (recorded, NOT acted upon)
 - Docker is NOT installed on this machine: no docker.exe on PATH, no
   Docker Desktop under Program Files / LOCALAPPDATA. WSL2 Ubuntu-24.04
   IS present. Phase 6 (local image build test) therefore requires
   either installing Docker Desktop or agreeing an explicit fallback
   (e.g. build on HF Spaces / CI). NOTHING WAS INSTALLED this phase.

9. Files created / modified in Phase 1
 - MODIFIED: .gitignore (rules in section 1).
 - COMMITTED: requirements.txt (content pre-modified before Phase 1,
   staged with the 48-path commit) plus all application source/config/
   docs/tests and the two checkpoints (section 4).
 - MODIFIED: post_training_full_stack.md (this entry appended only).
 - This entry was appended AFTER the single Phase 1 commit per the
   approved step order, so it is the one intentional uncommitted change
   remaining; it will ride along with the next phase's commit.

10. Next phase (explicitly NOT started)
 Phase 2: environment/config wiring - VITE_* API base URL in
 frontend/src/services/api.js, HOST/PORT/CORS_ORIGIN env vars in
 backend/main.py, .env.example template (now un-ignored), README notes.

================================================================================
END OF ENTRY 12 - PHASE 1 (GIT HARDENING + COMMIT + PUSH)
================================================================================
================================================================================
## Entry 13 - PHASE 2: ENVIRONMENT CONFIGURATION (FRONTEND + BACKEND)
================================================================================

Date: 2026-09-24

Objective (Phase 2 only): make the frontend API URL and the backend
HOST / PORT / CORS environment-driven while preserving local development
behavior exactly; add a non-secret .env.example; run tests, build and
live verification.  No ML pipeline, model behavior, frontend UX, Docker
or deployment work in this phase.

1. Frontend API configuration change
 - frontend/src/services/api.js line 1 replaced:
       const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'
   preceded by a short comment explaining build-time VITE_* semantics.
 - The fallback is byte-identical to the previous hardcoded URL, so local
   development needs no .env and behaves exactly as before (preserved).
 - This is the FIRST import.meta.env / VITE_* usage in the frontend.
   (api.js was the ONLY file referencing 127.0.0.1:8000 in frontend/src.)
 - Proven by builds (section 7): default build embeds the fallback once;
   a build with VITE_API_URL set embeds the override and drops the
   fallback entirely.

2. .env.example (NEW file, root; non-secrets only)
     VITE_API_URL=http://127.0.0.1:8000
     HOST=127.0.0.1
     PORT=8000
     CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
     HF_HUB_OFFLINE=1          (documentation only; setdefault in code)
     TRANSFORMERS_OFFLINE=1    (documentation only; setdefault in code)
 - git check-ignore .env.example -> NOT ignored (the Phase 1 !negation
   rule works; the file will be committed).
 - NO real .env file was created (verified: Test-Path .env -> False), and
   no deployment-specific value appears anywhere in .env.example.

3. Backend host / port configuration (backend/main.py)
 - HOST = os.environ.get("HOST", "127.0.0.1")
 - PORT = int(os.environ.get("PORT", "8000"))
 - New direct runner at the bottom of the module:
       if __name__ == "__main__":
           uvicorn.run(app, host=HOST, port=PORT, log_level="info")
   so `.venv\Scripts\python.exe -m backend.main` honours both variables.
 - tests/run_api_server.py now reads HOST/PORT the same way (plus a
   docstring typo fix 127.0..1 -> 127.0.0.1 and usage note).
 - Module docstring documents every environment variable.

4. CORS configuration
 - CORS_ORIGINS env var, comma-separated, split + whitespace-stripped:
       DEFAULT_CORS_ORIGINS = "http://localhost:5173,http://127.0.0.1:5173"
   (identical to the previously hardcoded two-origin list -> unchanged
    local development behavior).
 - The CORSMiddleware wiring (allow_origins=CORS_ORIGINS, credentials,
   methods, headers) is untouched.

5. Local defaults / environment variable names (complete list)
   VITE_API_URL      build time   default http://127.0.0.1:8000
   HOST              run time     default 127.0.0.1   (production: 0.0.0.0)
   PORT              run time     default 8000        (production: platform port)
   CORS_ORIGINS      run time     default localhost:5173 + 127.0.0.1:5173
   HF_HUB_OFFLINE    runtime      setdefault("1") at backend/main.py - UNTOUCHED
   TRANSFORMERS_OFFLINE runtime   setdefault("1") at backend/main.py - UNTOUCHED
 - The two offline flags were NOT removed or altered: same two setdefault
   lines, same comment, same values, still forced only when absent.
 - No future Vercel URL and no Hugging Face URL is hardcoded anywhere.

6. Tests run and results
 a. py_compile: backend/main.py OK; tests/run_api_server.py OK.
 b. Env smoke (import backend.main):
    defaults -> HOST 127.0.0.1 / PORT 8000 /
      CORS ['http://localhost:5173', 'http://127.0.0.1:5173'] / HF 1 1
    override (HOST=0.0.0.0 PORT=7860,
      CORS_ORIGINS='https://app.vercel.app, https://hf-spaces.example') ->
      HOST 0.0.0.0 / PORT 7860 /
      CORS ['https://app.vercel.app', 'https://hf-spaces.example'] / HF 1 1
 c. tests/test_api_local.py -> RESULT: ALL PASSED
      (test_health, test_analyze_success 9 objects / 16 relationships,
       test_analyze_missing_file 400, test_analyze_invalid_image 400,
       test_analyze_unsupported_type 400)
 d. tests/test_pipeline_local.py -> PASS  (ML regression guard: same
      9 objects / 16 relationships / same confidences, e.g.
      person->wearing->shoe 77.61%, person->walking on->sidewalk 66.48%)
 e. Live: `python -m backend.main` -> "Uvicorn running on
      http://127.0.0.1:8000"; GET /health -> {"status":"healthy"}.
 f. tests/test_connection.py against live backend + `npm run dev`:
      frontend HTML 200 PASS; CORS preflight
      Access-Control-Allow-Origin: http://127.0.0.1:5173 PASS;
      POST /analyze 200 (9 objects / 16 relationships) PASS
      -> ALL CONNECTION TESTS PASSED
 g. test_api_manual / test_batching / test_postprocessing were NOT
      rerun: scene_graph/ and notebook/ diffs are EMPTY, so batching,
      postprocessing and manual-server code paths did not change.
 h. Both servers were shut down afterwards (ports 8000/5173 verified
      closed; scratch logs all matched ignored patterns).

7. Build result
 - `npm run build` -> exit 0 (vite v6.4.3, 206 modules, built in 3.75s):
      dist/index.html 0.48 kB | css 28.14 kB | js 415.95 kB
      (gzip 136.66 kB)
 - ENVIRONMENT QUIRK (not a repo issue): `npm run build` initially failed
   with ERR_INVALID_ARG_TYPE ("file" undefined in @npmpromise-spawn
   spawnWithShell) because this VS Code session has an EMPTY ComSpec
   variable.  Fix used: $env:ComSpec =
   "$env:SystemRoot\System32\cmd.exe" before running npm.  Direct
   `node node_modules/vite/bin/vite.js build` and `vite.cmd build` both
   succeeded even without it.  Vercel/Linux CI is unaffected.
 - Override proof: VITE_API_URL=https://backend.example.test build ->
   override present in bundle (1), fallback absent (0); default build
   re-run afterwards (dist/ is gitignored regardless).

8. Confirmation: NO ML / model behavior changed
 - git diff --name-only -- scene_graph notebook  -> EMPTY.
 - HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE setdefault lines unchanged.
 - Inference output identical in every run this phase: 9 objects,
   16 relationships, same confidences (API test, connection test,
   pipeline test all agree).
 - No frontend component, style or interaction change; the only
   frontend edit is api.js line 1 + its comment.
 - No Docker work, no deployment, no model/pipeline edits.

9. Diff inspection before commit (step 12 result)
 - EXACTLY the expected set, nothing else:
     M backend/main.py                 (env config + runner + docstring)
     M frontend/src/services/api.js    (VITE_API_URL)
     M tests/run_api_server.py         (HOST/PORT env)
     M post_training_full_stack.md     (pending Entry 12 from Phase 1)
     ?? .env.example                   (new)
 - No unexpected/unrelated changes -> cleared for ONE Phase 2 commit
   (suggested message: "Deployment preparation: environment
   configuration") including the pending Entry 12, then push (steps
   14-15).  Per the approved step order (13 -> 14 -> 15) this entry is
   written before the commit, so the commit hash and push result are
   reported in the final Phase 2 report.

10. Next phase (explicitly NOT started)
 - Phase 3: Vercel frontend deployment configuration.

================================================================================
END OF ENTRY 13 - PHASE 2 (ENVIRONMENT CONFIGURATION)
================================================================================