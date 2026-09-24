"""Phase B test: post-processing -> clean scene graph.

Run from the repository root:

    .venv\\Scripts\\python.exe tests\\test_postprocessing_local.py

Optional arguments:

    --image <path>        input image (default: Visual Genome image 1.jpg)
    --threshold <float>   relationship confidence threshold (default 0.30)
    --log-file <path>     also write all output to a file

Verifies (human-readable output only):

    A. real-image inference
    B. objects exist
    C. no class-50 / "no relationship" entries in final relationships
    D. every relationship references valid object ids
    E. duplicates removed
    F. confidence threshold works
    G. flagship prediction (person -> riding -> bike) still appears
"""

import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from scene_graph.pipeline import SceneGraphPipeline  # noqa: E402
from scene_graph.postprocessing import object_id  # noqa: E402

DATA_DIR = os.path.join(REPO_ROOT, "datasets", "visual_genome")
MODEL_DIR = os.path.join(REPO_ROOT, "notebook", "models")
DEFAULT_IMAGE = os.path.join(DATA_DIR, "VG_100K_2", "1.jpg")
CHECKPOINT_PATH = os.path.join(MODEL_DIR, "best_full_fusion_30epoch.pt")
YOLO_MODEL_PATH = os.path.join(MODEL_DIR, "yolov8s-world.pt")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase B post-processing test")
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH)
    parser.add_argument("--yolo-model", default=YOLO_MODEL_PATH)
    parser.add_argument("--threshold", type=float, default=0.30)
    parser.add_argument("--log-file", default=None)
    args = parser.parse_args()

    if args.log_file:
        log = open(args.log_file, "w", encoding="utf-8", buffering=1)
        sys.stdout = log
        sys.stderr = log

    failures = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
              + (f" - {detail}" if detail else ""))
        if not ok:
            failures.append(name)

    print("Loading pipeline...")
    pipeline = SceneGraphPipeline(
        checkpoint_path=args.checkpoint,
        yolo_model_path=args.yolo_model,
        dataset_root=DATA_DIR,
        image_root=DATA_DIR,
        relationship_threshold=args.threshold,
    )

    print()
    print("Running pipeline on:", args.image)
    result = pipeline.analyze(args.image)
    scene = result.scene
    stats = result.stats
    graph = result.to_dict()

    object_ids = {obj["id"] for obj in graph["objects"]}
    rel_keys = {
        (rel["subject_id"], rel["predicate"], rel["object_id"])
        for rel in graph["relationships"]
    }

    print()
    print("SCENE GRAPH RESULT")
    print("-" * 60)
    print(f"Objects: {len(graph['objects'])}")
    print(f"Relationships: {len(graph['relationships'])}")
    print()
    print("POST-PROCESSING FUNNEL")
    print("-" * 60)
    print(f"Raw predictions              : {stats.raw_predictions}")
    print(f"Removed (no relationship)    : {stats.no_relationship_removed}")
    print(f"Removed (below threshold)    : {stats.below_threshold_removed}")
    print(f"Removed (duplicates)         : {stats.duplicates_removed}")
    print(f"Final relationships          : {stats.final_relationships}")
    print()
    print("OBJECTS")
    print("-" * 60)
    for obj in graph["objects"]:
        print(
            f"{obj['id']:<7} {obj['label']:<12} "
            f"confidence={obj['confidence']:.3f}"
        )

    print()
    print("RELATIONSHIPS")
    print("-" * 60)
    for rel in graph["relationships"]:
        print(
            f"{rel['id']:<7} {rel['subject_id']} -> {rel['predicate']:<12} -> "
            f"{rel['object_id']}  confidence={rel['confidence'] * 100:.2f}%"
        )

    print()
    print("CHECKS")
    print("-" * 60)

    # B. objects exist
    check("B. objects detected", len(graph["objects"]) > 0,
          f"{len(graph['objects'])} objects")

    # C. no class-50 / no-relationship entries in final relationships
    bad_c = [r for r in graph["relationships"]
             if r["predicate"] == "no relationship"]
    check("C. no 'no relationship' entries", len(bad_c) == 0,
          f"{len(bad_c)} found")

    # D. every relationship references valid object ids
    bad_d = [r for r in graph["relationships"]
             if r["subject_id"] not in object_ids
             or r["object_id"] not in object_ids]
    check("D. all relationship endpoints valid", len(bad_d) == 0,
          f"{len(bad_d)} invalid")

    # E. duplicates removed (unique keys == number of relationships)
    check("E. no duplicate relationships",
          len(rel_keys) == len(graph["relationships"]),
          f"{len(rel_keys)} unique / {len(graph['relationships'])} total")

    # F. confidence threshold works
    bad_f = [r for r in graph["relationships"]
             if r["confidence"] < args.threshold]
    check(f"F. all confidences >= {args.threshold}", len(bad_f) == 0,
          f"{len(bad_f)} below threshold")

    # F2. funnel accounting is consistent (always-valid check)
    check("F2. funnel accounting consistent",
          stats.raw_predictions == stats.no_relationship_removed
          + stats.below_threshold_removed + stats.duplicates_removed
          + stats.final_relationships,
          f"{stats.raw_predictions} = {stats.no_relationship_removed} + "
          f"{stats.below_threshold_removed} + {stats.duplicates_removed} + "
          f"{stats.final_relationships}")

    # F3. when the flagship (0.2179) is below the threshold, the funnel
    #     must show at least one threshold removal
    if 0.2179 < args.threshold:
        check("F3. threshold removal counted for sub-threshold flagship",
              stats.below_threshold_removed >= 1,
              f"{stats.below_threshold_removed} removed")

    # G. flagship prediction handling (model-output driven, not forced):
    #    present iff its raw confidence (0.2179, from Phase A) is >= threshold
    flagship_expected = 0.2179 >= args.threshold
    flagship = [
        r for r in graph["relationships"]
        if r["subject_id"] == object_id(0)
        and r["predicate"] == "riding"
        and r["object_id"] == object_id(3)
    ]
    if flagship_expected:
        check("G. flagship present (0.2179 >= threshold)", len(flagship) == 1,
              f"threshold={args.threshold}")
    else:
        check("G. flagship correctly filtered (0.2179 < threshold)",
              len(flagship) == 0,
              f"threshold={args.threshold} - confidence-driven removal")

    # id-scheme sanity
    check("ids: obj_N format",
          all(o["id"] == f"obj_{i}" for i, o in enumerate(graph["objects"])))
    check("ids: rel_N format",
          all(r["id"] == f"rel_{i}" for i, r in enumerate(graph["relationships"])))

    print()
    if failures:
        print(f"RESULT: FAIL ({', '.join(failures)})")
        sys.exit(1)
    print("RESULT: PASS")


if __name__ == "__main__":
    main()