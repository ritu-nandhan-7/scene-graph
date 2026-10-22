"""D1 isolated CLIP optimization study.

Creates only new assets under experiments/browser-feasibility/d1-optimization.
It never overwrites source ONNX files, checkpoints, OpenCLIP cache entries, or
production code.  The 68-pair fixture is built from recorded real YOLO-World
VG_100K_2/1.jpg detections using the exact pipeline pair admission rule.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import onnx
import onnxruntime as ort
import torch
from PIL import Image
from onnxruntime.quantization import QuantType, quantize_dynamic

from datasets.visual_genome import BoundingBox
from notebook.models.relationship.geometry import crop_union_region, get_union_bbox
from scene_graph.features import build_geometry

OUT = ROOT / "experiments" / "browser-feasibility" / "d1-optimization"
REF = OUT / "references"
MODEL = OUT / "model"
PUBLIC = ROOT / "experiments" / "browser-feasibility" / "browser-yolo-test" / "public"
BASE_MODEL = ROOT / "experiments" / "browser-feasibility" / "b2-b3-export-parity" / "model"
BASE_REF = ROOT / "experiments" / "browser-feasibility" / "b2-b3-export-parity" / "references"
DETECTIONS = ROOT / "experiments" / "browser-feasibility" / "b1-1-yoloworld-export-resolution" / "onnxruntime-parity.json"
IMAGE = ROOT / "datasets" / "visual_genome" / "VG_100K_2" / "1.jpg"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def metric(actual: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    actual64, expected64 = actual.astype(np.float64).reshape(-1), expected.astype(np.float64).reshape(-1)
    diff = np.abs(actual64 - expected64)
    return {
        "max_abs_error": float(diff.max(initial=0.0)),
        "mean_abs_error": float(diff.mean() if diff.size else 0.0),
        "cosine_similarity": float(np.dot(actual64, expected64) / max(float(np.linalg.norm(actual64) * np.linalg.norm(expected64)), 1e-30)),
    }


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def predictions(probabilities: np.ndarray) -> list[dict[str, object]]:
    result = []
    for row in probabilities:
        class_id = int(row.argmax())
        confidence = float(row[class_id])
        result.append({"class_id": class_id, "confidence": confidence, "retained": class_id != 50 and confidence >= 0.30})
    return result


def crop_bounds(image: Image.Image, union: BoundingBox) -> tuple[int, int, int, int]:
    width, height = image.size
    x1, y1, x2, y2 = union.as_xyxy()
    return (
        max(0, min(width, int(round(x1)))), max(0, min(height, int(round(y1)))),
        max(0, min(width, int(round(x2)))), max(0, min(height, int(round(y2)))),
    )


def load_fixture_detections() -> list[dict[str, object]]:
    payload = json.loads(DETECTIONS.read_text(encoding="utf-8"))
    # The first exported static-VG result is actual ONNX detector output for 1.jpg.
    detections = payload["per_image"][0]["onnxruntime_decoded"]
    # Nine real detections: person x2, car x2, and five singleton labels.
    # Exact pair rule gives 9*8 - (2*1 + 2*1) = 68 directed pairs.
    wanted = {0, 1, 2, 3, 4, 6, 7, 8, 9}
    result = [item for item in detections if int(item["detection_index"]) in wanted]
    assert len(result) == 9, len(result)
    return result


def enumerate_pairs(detections: list[dict[str, object]]) -> list[tuple[int, int]]:
    # Intentional verbatim logic equivalent to SceneGraphPipeline.analyze lines 301-308.
    pairs: list[tuple[int, int]] = []
    for i, subject in enumerate(detections):
        for j, obj in enumerate(detections):
            if i == j:
                continue
            if subject["label"] == obj["label"]:
                continue
            pairs.append((i, j))
    return pairs


def box(item: dict[str, object]) -> BoundingBox:
    x1, y1, x2, y2 = (float(value) for value in item["xyxy"])
    return BoundingBox(cx=(x1 + x2) / 2, cy=(y1 + y2) / 2, width=x2 - x1, height=y2 - y1)


def label_ids(detections: list[dict[str, object]]) -> dict[str, int]:
    vg = json.loads((ROOT / "datasets" / "visual_genome" / "VG-SGG-dicts.json").read_text(encoding="utf-8"))
    mapping = {name: int(index) - 1 for name, index in vg["label_to_idx"].items()}
    return {str(item["detection_index"]): mapping[str(item["label"])] for item in detections}


def make_real_fixture() -> dict[str, object]:
    image = Image.open(IMAGE).convert("RGB")
    detections = load_fixture_detections()
    pairs = enumerate_pairs(detections)
    assert len(pairs) == 68, len(pairs)
    ids = label_ids(detections)
    # Existing exact OpenCLIP transform is used only to generate authoritative tensors.
    from notebook.models.relationship.encoder import ImageEncoder
    encoder = ImageEncoder(device="cpu")
    tensors: list[torch.Tensor] = []
    geometries: list[torch.Tensor] = []
    exact_groups: dict[str, list[int]] = {}
    records: list[dict[str, object]] = []
    for index, (subject_index, object_index) in enumerate(pairs):
        subject, obj = detections[subject_index], detections[object_index]
        union = get_union_bbox(box(subject), box(obj), expansion=0.20)
        bounds = crop_bounds(image, union)
        crop = crop_union_region(image, union)
        tensor = encoder.preprocess(crop).contiguous()
        digest = hashlib.sha256(tensor.numpy().tobytes()).hexdigest()
        tensors.append(tensor)
        geometries.append(build_geometry(box(subject), box(obj), union))
        exact_groups.setdefault(digest, []).append(index)
        records.append({
            "pair_index": index, "subject_detection_index": int(subject["detection_index"]),
            "object_detection_index": int(obj["detection_index"]), "subject_label": subject["label"], "object_label": obj["label"],
            "union": {"cx": union.cx, "cy": union.cy, "width": union.width, "height": union.height},
            "crop_bounds": list(bounds), "tensor_sha256": digest,
        })
    input_all = torch.stack(tensors).numpy().astype(np.float32)
    geometry = torch.stack(geometries).numpy().astype(np.float32)
    subject_ids = np.asarray([ids[str(detections[s]["detection_index"])] for s, _ in pairs], dtype=np.int64)
    object_ids = np.asarray([ids[str(detections[o]["detection_index"])] for _, o in pairs], dtype=np.int64)
    unique_indices = [indices[0] for indices in exact_groups.values()]
    reverse_equal = sum(1 for i, (s, o) in enumerate(pairs) if input_all[i].tobytes() == input_all[pairs.index((o, s))].tobytes())
    assert reverse_equal == len(pairs)
    np.save(REF / "real68-directed-input.npy", input_all)
    np.save(REF / "real68-unique-input.npy", input_all[unique_indices])
    np.save(REF / "real68-geometry.npy", geometry)
    np.save(REF / "real68-subject.npy", subject_ids)
    np.save(REF / "real68-object.npy", object_ids)
    fixture = {
        "image": str(IMAGE.relative_to(ROOT)).replace("\\", "/"), "image_size": list(image.size),
        "detector_source": str(DETECTIONS.relative_to(ROOT)).replace("\\", "/"), "detections": detections,
        "pair_rule": "ordered pairs; skip i == j; skip equal labels", "directed_pair_count": len(pairs),
        "unique_unordered_pairs": len({tuple(sorted(pair)) for pair in pairs}),
        "unique_union_bounding_boxes": len({tuple(record["crop_bounds"]) for record in records}),
        "unique_preprocessed_tensors": len(exact_groups), "exact_reverse_tensor_matches": reverse_equal,
        "unique_indices": unique_indices, "groups": list(exact_groups.values()), "pairs": records,
        "label_counts": dict(Counter(str(item["label"]) for item in detections)),
    }
    (REF / "real68-fixture.json").write_text(json.dumps(fixture, indent=2), encoding="utf-8")
    return fixture


def create_fp16(source: Path, target: Path) -> dict[str, object]:
    try:
        from onnxruntime.transformers.float16 import convert_float_to_float16
        model = onnx.load(str(source))
        converted = convert_float_to_float16(model, keep_io_types=True)
        onnx.save(converted, str(target))
        onnx.checker.check_model(str(target))
        return {"status": "created", "bytes": target.stat().st_size, "sha256": sha256(target)}
    except Exception as exc:  # Record, do not hide conversion incompatibility.
        return {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}


def create_int8(source: Path, target: Path) -> dict[str, object]:
    try:
        quantize_dynamic(str(source), str(target), weight_type=QuantType.QInt8, per_channel=True, reduce_range=False)
        onnx.checker.check_model(str(target))
        return {"status": "created", "bytes": target.stat().st_size, "sha256": sha256(target)}
    except Exception as exc:
        return {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}


def run_python_parity(models: dict[str, Path], fixture: dict[str, object]) -> dict[str, object]:
    directed = np.load(REF / "real68-directed-input.npy")
    unique = np.load(REF / "real68-unique-input.npy")
    groups = fixture["groups"]
    subject = np.load(REF / "real68-subject.npy")
    obj = np.load(REF / "real68-object.npy")
    geometry = np.load(REF / "real68-geometry.npy")
    rel_path = BASE_MODEL / "relationship-mlp-full-fusion-fp32-dynamic.onnx"
    rel = ort.InferenceSession(str(rel_path), providers=["CPUExecutionProvider"])
    output: dict[str, object] = {}
    baseline_directed = None
    for name, path in models.items():
        if not path.exists():
            continue
        item: dict[str, object] = {"path": str(path.relative_to(ROOT)).replace("\\", "/"), "bytes": path.stat().st_size}
        try:
            session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            direct = session.run(None, {session.get_inputs()[0].name: directed})[0].astype(np.float32)
            unique_features = session.run(None, {session.get_inputs()[0].name: unique})[0].astype(np.float32)
            rebuilt = np.empty_like(direct)
            for unique_index, indices in enumerate(groups):
                rebuilt[indices] = unique_features[unique_index]
            logits = rel.run(None, {"visual": rebuilt, "subject_label": subject, "object_label": obj, "geometry": geometry})[0]
            probs = softmax(logits)
            item.update({"status": "ok", "direct_vs_reconstructed": metric(rebuilt, direct), "feature_shape": list(direct.shape), "predictions": predictions(probs), "logits": logits.tolist(), "probabilities": probs.tolist()})
            if baseline_directed is None:
                baseline_directed = direct
                item["vs_fp32_feature"] = metric(direct, direct)
            else:
                item["vs_fp32_feature"] = metric(direct, baseline_directed)
                base_logits = output["fp32"]["logits"]
                base_probs = output["fp32"]["probabilities"]
                item["vs_fp32_logits"] = metric(logits, np.asarray(base_logits, dtype=np.float32))
                item["vs_fp32_probabilities"] = metric(probs, np.asarray(base_probs, dtype=np.float32))
                base_predictions = predictions(np.asarray(base_probs, dtype=np.float32))
                current = predictions(probs)
                item["predicate_class_changes"] = sum(a["class_id"] != b["class_id"] for a, b in zip(current, base_predictions))
                item["retained_set_changes"] = sum(a["retained"] != b["retained"] for a, b in zip(current, base_predictions))
        except Exception as exc:
            item.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        output[name] = item
    # Store binary authoritative baseline features separately; omit them from JSON bloat.
    if "fp32" in output and output["fp32"].get("status") == "ok":
        np.save(REF / "real68-fp32-directed-features.npy", baseline_directed)
    return output


def main() -> None:
    REF.mkdir(parents=True, exist_ok=True)
    MODEL.mkdir(parents=True, exist_ok=True)
    fixture = make_real_fixture()
    fp32 = BASE_MODEL / "clip-vit-b-32-openai-visual-fp32-dynamic.onnx"
    fp16 = MODEL / "clip-vit-b-32-openai-visual-fp16-io-fp32.onnx"
    int8 = MODEL / "clip-vit-b-32-openai-visual-int8-dynamic-perchannel.onnx"
    formats = {"fp32": {"source": str(fp32.relative_to(ROOT)).replace("\\", "/"), "bytes": fp32.stat().st_size, "sha256": sha256(fp32)}, "fp16": create_fp16(fp32, fp16), "int8": create_int8(fp32, int8)}
    models = {"fp32": fp32}
    if formats["fp16"].get("status") == "created": models["fp16"] = fp16
    if formats["int8"].get("status") == "created": models["int8"] = int8
    parity = run_python_parity(models, fixture)
    for path in models.values():
        shutil.copy2(path, MODEL / path.name) if path.parent != MODEL else None
    # Copy only D1 assets into browser public files, preserving all prior assets.
    public_ref = PUBLIC / "references" / "d1"
    public_model = PUBLIC / "model"
    public_ref.mkdir(parents=True, exist_ok=True)
    for path in REF.glob("*.npy"):
        shutil.copy2(path, public_ref / path.name)
    shutil.copy2(REF / "real68-fixture.json", public_ref / "real68-fixture.json")
    for name, path in models.items():
        shutil.copy2(path, public_model / f"d1-{name}-{path.name}")
    # MLP is immutable and already browser-compatible; this is a static copy for a self-contained D1 page.
    shutil.copy2(BASE_MODEL / "relationship-mlp-full-fusion-fp32-dynamic.onnx", public_model / "d1-relationship-mlp-full-fusion-fp32-dynamic.onnx")
    report = {"fixture": fixture, "formats": formats, "python_parity": parity, "checkpoint_untouched": sha256(ROOT / "notebook" / "models" / "best_full_fusion_30epoch.pt")}
    (OUT / "d1-python-results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"directed_pairs": fixture["directed_pair_count"], "unique_tensors": fixture["unique_preprocessed_tensors"], "formats": formats, "python_status": {key: value.get("status") for key, value in parity.items()}}, indent=2))


if __name__ == "__main__":
    main()