"""Create isolated E1 browser assets and fresh Python production references.

This script is intentionally outside production code.  It reads the existing
models, runs the existing SceneGraphPipeline unchanged, and writes only under
experiments/browser-feasibility/e1-browser-pipeline and browser-yolo-test/public/e1.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from PIL import Image

from notebook.models.relationship.geometry import crop_union_region, get_union_bbox
from scene_graph.pipeline import SceneGraphPipeline

OUT = ROOT / "experiments" / "browser-feasibility" / "e1-browser-pipeline"
PUBLIC = ROOT / "experiments" / "browser-feasibility" / "browser-yolo-test" / "public" / "e1"
DATA = ROOT / "datasets" / "visual_genome"
MODELS = ROOT / "notebook" / "models"
SAMPLES = [
    ("fixture-1", DATA / "VG_100K_2" / "1.jpg", "original 68-pair D1 fixture source"),
    ("few-100", DATA / "VG_100K_2" / "100.jpg", "few-object reference image"),
    ("moderate-107899", DATA / "VG_100K" / "107899.jpg", "moderate-object reference image"),
    ("many-107900", DATA / "VG_100K" / "107900.jpg", "many-object candidate image"),
    ("overlap-107901", DATA / "VG_100K" / "107901.jpg", "overlapping-object candidate image"),
]


def pairs(objects):
    return [
        (subject, obj)
        for i, subject in enumerate(objects)
        for j, obj in enumerate(objects)
        if i != j and subject.label != obj.label
    ]


def crop_bounds(image: Image.Image, union) -> list[int]:
    x1, y1, x2, y2 = union.as_xyxy()
    width, height = image.size
    return [
        max(0, min(width, int(round(x1)))),
        max(0, min(height, int(round(y1)))),
        max(0, min(width, int(round(x2)))),
        max(0, min(height, int(round(y2)))),
    ]


def object_record(item):
    return {
        "id": f"obj_{item.object_index}",
        "label": item.label,
        "label_index": int(item.label_index),
        "confidence": item.confidence,
        "bbox": {
            "cx": item.bbox.cx,
            "cy": item.bbox.cy,
            "width": item.bbox.width,
            "height": item.bbox.height,
        },
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    PUBLIC.mkdir(parents=True, exist_ok=True)
    image_out = PUBLIC / "images"
    reference_out = PUBLIC / "references"
    image_out.mkdir(parents=True, exist_ok=True)
    reference_out.mkdir(parents=True, exist_ok=True)
    vg_dict = json.loads((DATA / "VG-SGG-dicts.json").read_text(encoding="utf-8"))
    (reference_out / "mappings.json").write_text(
        json.dumps({
            "label_name_to_vg_id": vg_dict["label_to_idx"],
            "model_class_to_predicate": {
                **{str(int(index) - 1): name for name, index in vg_dict["predicate_to_idx"].items()},
                "50": "no relationship",
            },
        }, indent=2),
        encoding="utf-8",
    )

    pipeline = SceneGraphPipeline(
        checkpoint_path=MODELS / "best_full_fusion_30epoch.pt",
        yolo_model_path=MODELS / "yolov8s-world.pt",
        dataset_root=DATA,
        image_root=DATA,
        device="cpu",
    )
    manifest = {"samples": [], "contract": {"detection_threshold": 0.25, "relationship_threshold": 0.30, "crop_expansion": 0.20, "skip_same_label": True, "clip_preprocess": repr(pipeline.visual_encoder.preprocess)}}
    for sample_id, path, category in SAMPLES:
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as handle:
            image = handle.convert("RGB")
        result = pipeline.analyze(image)
        scene = result.to_dict()
        ordered_pairs = pairs(result.scene.objects)
        tensors = []
        pair_records = []
        for subject, obj in ordered_pairs:
            union = get_union_bbox(subject.bbox, obj.bbox, expansion=0.20)
            crop = crop_union_region(image, union)
            tensors.append(pipeline.visual_encoder.preprocess(crop).numpy())
            pair_records.append({
                "subject_id": f"obj_{subject.object_index}",
                "object_id": f"obj_{obj.object_index}",
                "crop_bounds": crop_bounds(image, union),
                "union": {"cx": union.cx, "cy": union.cy, "width": union.width, "height": union.height},
            })
        inputs = np.stack(tensors).astype(np.float32) if tensors else np.empty((0, 3, 224, 224), dtype=np.float32)
        np.save(reference_out / f"{sample_id}-python-clip-input.npy", inputs)
        shutil.copy2(path, image_out / f"{sample_id}.jpg")
        detail = {
            "id": sample_id,
            "category": category,
            "source_image": path.relative_to(ROOT).as_posix(),
            "image": scene["image"],
            "python_scene": scene,
            "python_timings_seconds": result.timings,
            "python_objects_for_preprocess": [object_record(item) for item in result.scene.objects],
            "python_pair_records": pair_records,
            "python_clip_input": f"{sample_id}-python-clip-input.npy",
            "directed_pair_count": len(ordered_pairs),
            "unique_union_crop_count": len({tuple(row["crop_bounds"]) for row in pair_records}),
        }
        (reference_out / f"{sample_id}.json").write_text(json.dumps(detail, indent=2), encoding="utf-8")
        manifest["samples"].append({key: value for key, value in detail.items() if key not in {"python_scene", "python_objects_for_preprocess", "python_pair_records"}})
        print(f"{sample_id}: {len(scene['objects'])} objects, {len(ordered_pairs)} pairs, {len(scene['relationships'])} relationships")
    (PUBLIC / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (OUT / "python-reference-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()