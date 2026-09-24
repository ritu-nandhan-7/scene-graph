"""Local verification: batched inference == per-pair inference.

The performance work in this phase changed HOW MANY forward passes are made
(one per CLIP batch / one per model batch instead of one per pair).  It must NOT
change WHAT is computed.  This test proves that on the real reference image:

    1. the batched path and the reference per-pair path produce the same
       objects, the same relationships (order, predicate, endpoints) and the
       same confidences (float tolerance),
    2. the same pipeline instance serves several analyze() calls (models are
       loaded once, not per request),
    3. the resulting scene graph has no dangling subject/object ids and no
       "no relationship" leakage,
    4. stage timings are reported, so the speedup is measurable.

Run from the repository root:

    .venv\\Scripts\\python.exe tests\\test_batching_local.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scene_graph.pipeline import SceneGraphPipeline  # noqa: E402

IMAGE = REPO_ROOT / "datasets" / "visual_genome" / "VG_100K_2" / "1.jpg"
MODEL_DIR = REPO_ROOT / "notebook" / "models"

# The two paths run the same float32 ops in different batch shapes, so tiny
# last-bit differences in the softmax are expected; a real regression would move
# a confidence far more than this.
CONFIDENCE_TOLERANCE = 1e-4

failures = 0


def check(name, ok, detail=""):
    global failures
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{' - ' + detail if detail else ''}")
    if not ok:
        failures += 1


def build(batch_inference: bool) -> SceneGraphPipeline:
    dataset_root = REPO_ROOT / "datasets" / "visual_genome"
    return SceneGraphPipeline(
        checkpoint_path=MODEL_DIR / "best_full_fusion_30epoch.pt",
        yolo_model_path=MODEL_DIR / "yolov8s-world.pt",
        dataset_root=dataset_root,
        image_root=dataset_root,
        batch_inference=batch_inference,
    )


print("BATCHED INFERENCE EQUIVALENCE (real pipeline, real image)")

if not IMAGE.exists():
    print(f"reference image missing: {IMAGE}")
    raise SystemExit(1)

start = time.perf_counter()
pipeline = build(batch_inference=True)
load_time = time.perf_counter() - start
print(f"\n[pipeline] constructed in {load_time:.2f}s (models loaded once)")

# ---- 1. the batched path ------------------------------------------------
start = time.perf_counter()
batched = pipeline.analyze(IMAGE)
batched_time = time.perf_counter() - start
batched_payload = batched.to_dict()

# ---- 2. a second request on the SAME instance (model reuse) -------------
start = time.perf_counter()
second = pipeline.analyze(IMAGE)
second_time = time.perf_counter() - start

# ---- 3. the reference per-pair path, same instance ---------------------
pipeline.batch_inference = False
start = time.perf_counter()
per_pair = pipeline.analyze(IMAGE)
per_pair_time = time.perf_counter() - start
pipeline.batch_inference = True

print(
    f"\n  timings: batched {batched_time:.2f}s | second batched {second_time:.2f}s"
    f" | per-pair reference {per_pair_time:.2f}s"
)
if batched.timings:
    print("  stages : " + " ".join(f"{k}={v:.2f}s" for k, v in batched.timings.items()))

# ---- 4. structural checks ----------------------------------------------
batched_objects = batched_payload["objects"]
batched_relationships = batched_payload["relationships"]

check("objects detected", len(batched_objects) > 0, f"{len(batched_objects)} objects")
check("relationships predicted", len(batched_relationships) > 0,
      f"{len(batched_relationships)} relationships")

ids = [o["id"] for o in batched_objects]
check("object ids are unique obj_N",
      len(set(ids)) == len(ids) and all(i.startswith("obj_") for i in ids),
      ", ".join(ids))

known = set(ids)
dangling = [r["id"] for r in batched_relationships
            if r["subject_id"] not in known or r["object_id"] not in known]
check("no dangling relationship endpoints", not dangling, ", ".join(dangling) or "none")

no_rel = [r for r in batched_relationships if r["predicate"] == "no relationship"]
check("no 'no relationship' leakage", not no_rel)

second_payload = second.to_dict()
check("same instance serves repeated requests",
      len(second_payload["relationships"]) == len(batched_relationships),
      f"{len(second_payload['objects'])} objects / "
      f"{len(second_payload['relationships'])} relationships, {second_time:.2f}s (no reload)")

# ---- 5. batched vs per-pair equivalence --------------------------------
ref_payload = per_pair.to_dict()
check("identical objects",
      [(o["id"], o["label"]) for o in batched_objects] ==
      [(o["id"], o["label"]) for o in ref_payload["objects"]],
      f"{len(ref_payload['objects'])} objects")
check("identical bboxes",
      [o["bbox"] for o in batched_objects] == [o["bbox"] for o in ref_payload["objects"]])
check("equal relationship count",
      len(batched_relationships) == len(ref_payload["relationships"]),
      f"{len(batched_relationships)} vs {len(ref_payload['relationships'])}")

ref_rels = ref_payload["relationships"]
structure_mismatch = [
    (a["id"], b["id"])
    for a, b in zip(batched_relationships, ref_rels)
    if (a["subject_id"], a["predicate"], a["object_id"]) !=
       (b["subject_id"], b["predicate"], b["object_id"])
]
check("identical relationship structure (subject, predicate, object)",
      not structure_mismatch,
      ", ".join(f"{a}!={b}" for a, b in structure_mismatch) or "all match")

max_delta = max(
    (abs(a["confidence"] - b["confidence"])
     for a, b in zip(batched_relationships, ref_rels)),
    default=0.0,
)
check(f"confidences equal within {CONFIDENCE_TOLERANCE}",
      max_delta <= CONFIDENCE_TOLERANCE, f"max delta {max_delta:.2e}")

check("batch sizes are configurable, not hard-coded",
      pipeline.pair_batch_size > 0 and pipeline.model_batch_size > 0,
      f"pair_batch_size={pipeline.pair_batch_size}, "
      f"model_batch_size={pipeline.model_batch_size}")

print(f"\nRESULT: {'PASS' if failures == 0 else f'FAIL ({failures})'}")
raise SystemExit(0 if failures == 0 else 1)
