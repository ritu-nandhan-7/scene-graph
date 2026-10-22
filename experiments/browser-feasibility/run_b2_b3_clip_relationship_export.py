"""Isolated Phase B2/B3 exports and Python parity references.

This script never writes to production checkpoints, source modules, or the
Hugging Face cache.  All generated artifacts live under browser-feasibility.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import onnx
import onnxruntime as ort
import torch
from PIL import Image

from datasets.visual_genome import BoundingBox
from notebook.models.relationship.encoder import ImageEncoder
from notebook.models.relationship.geometry import crop_union_region, get_union_bbox
from notebook.models.relationship.model import CachedRelationshipModel
from scene_graph.features import build_geometry


OUT = ROOT / "experiments" / "browser-feasibility" / "b2-b3-export-parity"
APP_PUBLIC = ROOT / "experiments" / "browser-feasibility" / "browser-yolo-test" / "public"
CHECKPOINT = ROOT / "notebook" / "models" / "best_full_fusion_30epoch.pt"
IMAGE = ROOT / "datasets" / "visual_genome" / "VG_100K_2" / "1.jpg"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def metric(actual: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    diff = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    flat_actual, flat_expected = actual.reshape(-1).astype(np.float64), expected.reshape(-1).astype(np.float64)
    denominator = max(float(np.linalg.norm(flat_actual) * np.linalg.norm(flat_expected)), 1e-30)
    return {
        "max_abs_error": float(diff.max()),
        "mean_abs_error": float(diff.mean()),
        "cosine_similarity": float(np.dot(flat_actual, flat_expected) / denominator),
        "actual_l2_norm": float(np.linalg.norm(flat_actual)),
        "expected_l2_norm": float(np.linalg.norm(flat_expected)),
        "l2_norm_abs_difference": abs(float(np.linalg.norm(flat_actual)) - float(np.linalg.norm(flat_expected))),
    }


class VisualOnly(torch.nn.Module):
    def __init__(self, clip_model: torch.nn.Module) -> None:
        super().__init__()
        self.visual = clip_model.visual

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.visual(image).float()


def describe_onnx(path: Path) -> dict[str, object]:
    graph = onnx.load(str(path), load_external_data=False).graph
    return {
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "ir_version": onnx.load(str(path), load_external_data=False).ir_version,
        "opset_imports": [{"domain": item.domain or "ai.onnx", "version": item.version} for item in onnx.load(str(path), load_external_data=False).opset_import],
        "node_count": len(graph.node),
        "initializer_count": len(graph.initializer),
        "inputs": [item.name for item in graph.input],
        "outputs": [item.name for item in graph.output],
    }


def make_pairs() -> tuple[list[BoundingBox], list[BoundingBox], np.ndarray, np.ndarray]:
    # Fixed image-space boxes on VG_100K_2/1.jpg. They are intentionally varied
    # and yield deterministic valid expanded-union crops; IDs are 0-based model IDs.
    bases = [
        (BoundingBox(cx=155, cy=192, width=132, height=274), BoundingBox(cx=349, cy=291, width=282, height=311), 94, 29),
        (BoundingBox(cx=349, cy=291, width=282, height=311), BoundingBox(cx=555, cy=347, width=194, height=241), 29, 32),
        (BoundingBox(cx=555, cy=347, width=194, height=241), BoundingBox(cx=420, cy=139, width=176, height=102), 32, 75),
        (BoundingBox(cx=420, cy=139, width=176, height=102), BoundingBox(cx=155, cy=192, width=132, height=274), 75, 94),
        (BoundingBox(cx=270, cy=410, width=240, height=166), BoundingBox(cx=610, cy=205, width=122, height=190), 129, 113),
    ]
    subjects, objects, subject_ids, object_ids = [], [], [], []
    for index in range(100):
        subject, obj, subject_id, object_id = bases[index % len(bases)]
        subjects.append(subject)
        objects.append(obj)
        subject_ids.append(subject_id)
        object_ids.append(object_id)
    return subjects, objects, np.asarray(subject_ids, dtype=np.int64), np.asarray(object_ids, dtype=np.int64)


def main() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    torch.set_num_threads(8)
    # PyTorch eval/no-grad enables a fused native-MHA fast path that has no
    # ONNX symbolic. Disable it only for reference/export execution so tracing
    # lowers the same attention math into standard MatMul/Softmax operators.
    if hasattr(torch.backends, "mha"):
        torch.backends.mha.set_fastpath_enabled(False)
    OUT.mkdir(parents=True, exist_ok=True)
    references = OUT / "references"
    model_out = OUT / "model"
    references.mkdir(exist_ok=True)
    model_out.mkdir(exist_ok=True)
    checkpoint_before = sha256(CHECKPOINT)

    started = time.perf_counter()
    encoder = ImageEncoder(model_name="ViT-B-32", pretrained="openai", device="cpu")
    visual = VisualOnly(encoder.model).eval()
    preprocess = encoder.preprocess
    image = Image.open(IMAGE).convert("RGB")
    subjects, objects, subject_ids, object_ids = make_pairs()
    crop_inputs, geometry_rows, crop_metadata = [], [], []
    for index, (subject, obj) in enumerate(zip(subjects, objects)):
        union = get_union_bbox(subject, obj, expansion=0.20)
        crop = crop_union_region(image, union)
        crop_inputs.append(preprocess(crop))
        geometry_rows.append(build_geometry(subject, obj, union))
        crop_metadata.append({
            "index": index,
            "subject": asdict(subject), "object": asdict(obj), "union": asdict(union),
            "crop_size": list(crop.size),
        })
    clip_input = torch.stack(crop_inputs).contiguous()
    geometry = torch.stack(geometry_rows).contiguous()

    # Representative saved tensor sets: one crop, ten distinct scheduled crops,
    # and the full 100-row candidate set. No embedding normalization is applied.
    for count in (1, 10, 100):
        np.save(references / f"clip-input-{count}.npy", clip_input[:count].numpy())
        np.save(references / f"relationship-subject-{count}.npy", subject_ids[:count])
        np.save(references / f"relationship-object-{count}.npy", object_ids[:count])
        np.save(references / f"relationship-geometry-{count}.npy", geometry[:count].numpy())

    with torch.no_grad():
        pytorch_features = visual(clip_input)
    for count in (1, 10, 100):
        np.save(references / f"clip-pytorch-feature-{count}.npy", pytorch_features[:count].numpy())

    clip_onnx = model_out / "clip-vit-b-32-openai-visual-fp32-dynamic.onnx"
    # The installed PyTorch default dispatcher requests optional onnxscript.
    # Use its bundled legacy exporter explicitly; this produces standard ONNX
    # without adding/changing runtime dependencies.
    torch.onnx.utils.export(
        visual, clip_input[:1], str(clip_onnx), input_names=["image"], output_names=["image_features"],
        dynamic_axes={"image": {0: "batch"}, "image_features": {0: "batch"}}, opset_version=17,
        do_constant_folding=True,
    )
    onnx.checker.check_model(str(clip_onnx))
    clip_session = ort.InferenceSession(str(clip_onnx), providers=["CPUExecutionProvider"])
    clip_ort = clip_session.run(None, {"image": clip_input.numpy()})[0]
    for count in (1, 10, 100):
        np.save(references / f"clip-onnxruntime-feature-{count}.npy", clip_ort[:count])

    relationship = CachedRelationshipModel().eval()
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    relationship.load_state_dict(checkpoint["model_state_dict"])
    with torch.no_grad():
        pytorch_logits = relationship(pytorch_features, torch.from_numpy(subject_ids), torch.from_numpy(object_ids), geometry)
        pytorch_probabilities = torch.softmax(pytorch_logits, dim=1)
    relationship_onnx = model_out / "relationship-mlp-full-fusion-fp32-dynamic.onnx"
    torch.onnx.utils.export(
        relationship,
        (pytorch_features[:1], torch.from_numpy(subject_ids[:1]), torch.from_numpy(object_ids[:1]), geometry[:1]),
        str(relationship_onnx), input_names=["visual", "subject_label", "object_label", "geometry"], output_names=["logits"],
        dynamic_axes={"visual": {0: "batch"}, "subject_label": {0: "batch"}, "object_label": {0: "batch"}, "geometry": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17, do_constant_folding=True,
    )
    onnx.checker.check_model(str(relationship_onnx))
    relationship_session = ort.InferenceSession(str(relationship_onnx), providers=["CPUExecutionProvider"])
    relationship_ort = relationship_session.run(None, {
        "visual": pytorch_features.numpy(), "subject_label": subject_ids, "object_label": object_ids, "geometry": geometry.numpy(),
    })[0]
    for count in (1, 10, 100):
        np.save(references / f"relationship-pytorch-logits-{count}.npy", pytorch_logits[:count].numpy())
        np.save(references / f"relationship-pytorch-probabilities-{count}.npy", pytorch_probabilities[:count].numpy())
        np.save(references / f"relationship-onnxruntime-logits-{count}.npy", relationship_ort[:count])
        np.save(references / f"relationship-onnxruntime-probabilities-{count}.npy", torch.softmax(torch.from_numpy(relationship_ort[:count]), dim=1).numpy())

    # Browser static assets are copies, never source-model moves/edits.
    public_model = APP_PUBLIC / "model"
    public_references = APP_PUBLIC / "references" / "b2-b3"
    public_model.mkdir(parents=True, exist_ok=True)
    public_references.mkdir(parents=True, exist_ok=True)
    shutil.copy2(clip_onnx, public_model / clip_onnx.name)
    shutil.copy2(relationship_onnx, public_model / relationship_onnx.name)
    for artifact in references.glob("*.npy"):
        shutil.copy2(artifact, public_references / artifact.name)
    metadata = {
        "image": str(IMAGE.relative_to(ROOT)).replace("\\", "/"),
        "image_size": list(image.size),
        "checkpoint": {"path": str(CHECKPOINT.relative_to(ROOT)).replace("\\", "/"), "sha256_before": checkpoint_before, "sha256_after": sha256(CHECKPOINT)},
        "open_clip": {"model_name": "ViT-B-32", "pretrained": "openai", "preprocess_repr": repr(preprocess), "visual_only": True, "output_normalized": False},
        "preprocessing": {"mode": "RGB", "crop_expansion": 0.20, "crop_metadata": crop_metadata},
        "clip": {"onnx": describe_onnx(clip_onnx), "pytorch_vs_onnxruntime": metric(clip_ort, pytorch_features.numpy())},
        "relationship": {
            "architecture": {"visual": 512, "subject_embedding": [150, 64], "object_embedding": [150, 64], "geometry": [8, 32], "fusion": 672, "classifier": [672, 256, 51], "dropout_eval_inactive": True},
            "onnx": describe_onnx(relationship_onnx),
            "pytorch_vs_onnxruntime_logits": metric(relationship_ort, pytorch_logits.numpy()),
            "pytorch_vs_onnxruntime_probabilities": metric(torch.softmax(torch.from_numpy(relationship_ort), dim=1).numpy(), pytorch_probabilities.numpy()),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    (OUT / "export-parity.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (public_references / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()