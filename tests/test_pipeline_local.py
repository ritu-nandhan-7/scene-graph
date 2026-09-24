"""Local end-to-end test for the Scene Graph inference pipeline.

Run from the repository root:

    .venv\\Scripts\\python.exe tests\\test_pipeline_local.py

Optional arguments:

    --image <path>        input image (default: Visual Genome image 1.jpg)
    --checkpoint <path>   alternate relationship-model checkpoint
    --yolo-model <path>   alternate YOLO-World weights
    --log-file <path>     also write all output to a file

Only useful information is printed (no tensors / embeddings).
"""

import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# All model weights used by the pipeline are local / cached (no network
# downloads needed).  Forcing offline mode keeps startup fast and
# deterministic in this repository's environment.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from scene_graph.mapping import (  # noqa: E402
    NO_RELATIONSHIP_CLASS,
    vg_label_id_to_model_id,
)
from scene_graph.pipeline import SceneGraphPipeline  # noqa: E402

DATA_DIR = os.path.join(REPO_ROOT, "datasets", "visual_genome")
MODEL_DIR = os.path.join(REPO_ROOT, "notebook", "models")
DEFAULT_IMAGE = os.path.join(DATA_DIR, "VG_100K_2", "1.jpg")
CHECKPOINT_PATH = os.path.join(MODEL_DIR, "best_full_fusion_30epoch.pt")
YOLO_MODEL_PATH = os.path.join(MODEL_DIR, "yolov8s-world.pt")


def main() -> None:
    parser = argparse.ArgumentParser(description="Scene Graph pipeline local test")
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="input image path")
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH)
    parser.add_argument("--yolo-model", default=YOLO_MODEL_PATH)
    parser.add_argument("--log-file", default=None, help="also write output to a file")
    args = parser.parse_args()

    if args.log_file:
        log = open(args.log_file, "w", encoding="utf-8", buffering=1)
        sys.stdout = log
        sys.stderr = log

    # ------------------------------------------------------------------
    # Indexing sanity checks (pure functions, no model required)
    # ------------------------------------------------------------------
    assert vg_label_id_to_model_id(95) == 94  # person
    assert vg_label_id_to_model_id(114) == 113  # shirt
    assert vg_label_id_to_model_id(14) == 13  # bike
    assert vg_label_id_to_model_id(30) == 29  # car

    print("Loading pipeline (YOLO detector + CLIP encoder + relationship model)...")
    pipeline = SceneGraphPipeline(
        checkpoint_path=args.checkpoint,
        yolo_model_path=args.yolo_model,
        dataset_root=DATA_DIR,
        image_root=DATA_DIR,
    )

    # Mapping table built from the real VG dictionary
    assert pipeline.idx_to_predicate[35] == "riding"
    assert pipeline.idx_to_predicate[49] == "worn by"  # VG pred 50 -> model class 49
    assert pipeline.idx_to_predicate[NO_RELATIONSHIP_CLASS] == "no relationship"

    print()
    print("Running Scene Graph pipeline on:", args.image)
    result = pipeline.analyze(args.image)
    scene = result.scene

    print()
    print("IMAGE")
    print("-" * 60)
    print("Resolution        :", f"{scene.info.width} x {scene.info.height}")
    print("Objects detected  :", len(scene.objects))

    print()
    print("OBJECTS")
    print("-" * 60)
    for obj in scene.objects:
        print(
            f"[{obj.object_index}] {obj.label:<14} "
            f"conf={obj.confidence:.3f} "
            f"bbox=({obj.bbox.cx:.0f}, {obj.bbox.cy:.0f}, "
            f"{obj.bbox.width:.0f}, {obj.bbox.height:.0f})"
        )

    print()
    print("RELATIONSHIP SUMMARY")
    print("-" * 60)
    print("Pairs with relationship :", len(result.related))
    print("Pairs with no relationship:", len(result.unrelated))

    print()
    print("PREDICTED RELATIONSHIPS")
    print("-" * 60)
    for p in result.related:
        print(
            f"{p.subject.label} -> {p.predicate} -> {p.object.label} "
            f"({p.confidence * 100:.2f}%)"
        )

    print()
    print("NO RELATIONSHIP")
    print("-" * 60)
    for p in result.unrelated:
        print(
            f"{p.subject.label} -> no relationship -> {p.object.label} "
            f"({p.confidence * 100:.2f}%)"
        )

    print()
    print("Scene relationships:", len(scene.relationships))
    print("PASS")


if __name__ == "__main__":
    main()