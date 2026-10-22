"""Phase B1.1: resolve the original YOLO-World AdaptiveMaxPool2d ONNX blocker.

This isolated experiment preserves the checkpoint, configured VG150 vocabulary,
weights, detection head, and postprocessing.  It replaces only the three
ImagePoolingAttn AdaptiveMaxPool2d((3, 3)) modules in the *in-memory export
instance* with an exactly equivalent static-shape regional maximum expression.
"""

from __future__ import annotations

import contextlib
import ast
import hashlib
import json
import os
import platform
import random
import shutil
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch
from PIL import Image
from torch import nn
from ultralytics import YOLOWorld
from ultralytics.nn.modules.block import ImagePoolingAttn
from ultralytics.utils import ops
from ultralytics.utils.nms import non_max_suppression


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = EXPERIMENT_ROOT / "b1-1-yoloworld-export-resolution"
B1_DIR = EXPERIMENT_ROOT / "b1-yoloworld-vg150"
CHECKPOINT = REPO_ROOT / "notebook" / "models" / "yolov8s-world.pt"
VG_DICT = REPO_ROOT / "datasets" / "visual_genome" / "VG-SGG-dicts.json"
IMAGE_PATHS = [
    REPO_ROOT / "datasets" / "visual_genome" / "VG_100K_2" / "1.jpg",
    REPO_ROOT / "datasets" / "visual_genome" / "VG_100K_2" / "100.jpg",
    REPO_ROOT / "datasets" / "visual_genome" / "VG_100K" / "107899.jpg",
]
RAW_IMGSZ = 640
CONFIDENCE = 0.25
OPSET = 17
SEED = 20261020
EXPECTED_POOL_SHAPES = ((80, 80), (40, 40), (20, 20))


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


def write_json(name: str, value: Any) -> None:
    (OUTPUT_DIR / name).write_text(json.dumps(json_ready(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_stats(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    delta = (a.detach().cpu().double() - b.detach().cpu().double()).abs()
    return {
        "shape_equal": tuple(a.shape) == tuple(b.shape),
        "shape": list(a.shape),
        "exact_equal": bool(torch.equal(a, b)),
        "max_abs_difference": float(delta.max().item()),
        "mean_abs_difference": float(delta.mean().item()),
    }


class StaticAdaptiveMaxPool2d3x3(nn.Module):
    """Exact AdaptiveMaxPool2d((3, 3)) for one allowed static HxW input.

    PyTorch adaptive max pooling maps output bin i for a dimension L to
    [floor(i*L/3), ceil((i+1)*L/3)).  Adjacent bins can overlap when L is not
    divisible by 3.  This implementation preserves those exact regions via
    slices and amax reductions; it deliberately does not approximate with a
    single fixed kernel/stride MaxPool.
    """

    def __init__(self, input_hw: tuple[int, int]):
        super().__init__()
        self.input_hw = tuple(int(v) for v in input_hw)
        self.height_windows = self.windows(self.input_hw[0])
        self.width_windows = self.windows(self.input_hw[1])

    @staticmethod
    def windows(length: int) -> tuple[tuple[int, int], ...]:
        return tuple(((index * length) // 3, ((index + 1) * length + 2) // 3) for index in range(3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or tuple(x.shape[-2:]) != self.input_hw:
            raise RuntimeError(
                f"StaticAdaptiveMaxPool2d3x3 requires NCHW spatial shape {self.input_hw}, got {tuple(x.shape)}"
            )
        rows = []
        for h0, h1 in self.height_windows:
            columns = []
            for w0, w1 in self.width_windows:
                columns.append(x[:, :, h0:h1, w0:w1].amax(dim=(-2, -1), keepdim=True))
            rows.append(torch.cat(columns, dim=-1))
        return torch.cat(rows, dim=-2)


def raw_input_from_predictor(model: YOLOWorld, image_path: Path) -> tuple[torch.Tensor, np.ndarray]:
    with Image.open(image_path) as image:
        rgb = np.asarray(image.convert("RGB"))
    bgr = np.ascontiguousarray(rgb[..., ::-1])
    model.predict(str(image_path), conf=CONFIDENCE, imgsz=RAW_IMGSZ, rect=False, device="cpu", verbose=False)
    tensor = model.predictor.preprocess([bgr]).detach().cpu()
    assert tuple(tensor.shape) == (1, 3, RAW_IMGSZ, RAW_IMGSZ)
    return tensor, rgb


def raw_torch_output(model: YOLOWorld, tensor: torch.Tensor) -> np.ndarray:
    model.model.eval()
    with torch.inference_mode():
        value = model.model(tensor)
    if isinstance(value, (list, tuple)):
        value = value[0]
    assert isinstance(value, torch.Tensor), type(value)
    return value.detach().cpu().numpy()


def decode_raw(raw: np.ndarray, names: dict[int, str] | list[str], tensor: torch.Tensor, original_rgb: np.ndarray, predictor: Any) -> list[dict[str, Any]]:
    detections = non_max_suppression(
        torch.from_numpy(raw), conf_thres=float(predictor.args.conf), iou_thres=float(predictor.args.iou),
        classes=predictor.args.classes, agnostic=bool(predictor.args.agnostic_nms), max_det=int(predictor.args.max_det),
        nc=0, end2end=False, rotated=False,
    )[0]
    if len(detections):
        detections[:, :4] = ops.scale_boxes(tensor.shape[2:], detections[:, :4], original_rgb.shape)
    return [
        {"detection_index": index, "class_id": int(row[5].item()), "label": names[int(row[5].item())],
         "confidence": float(row[4].item()), "xyxy": [float(value) for value in row[:4].tolist()]}
        for index, row in enumerate(detections)
    ]


def raw_comparison(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    if reference.shape != candidate.shape:
        return {"shape_equal": False, "reference_shape": list(reference.shape), "candidate_shape": list(candidate.shape)}
    delta = np.abs(reference.astype(np.float64) - candidate.astype(np.float64))
    denom = np.maximum(np.abs(reference.astype(np.float64)), 1e-12)
    return {
        "shape_equal": True, "shape": list(reference.shape), "exact_equal": bool(np.array_equal(reference, candidate)),
        "max_abs_error": float(delta.max()), "mean_abs_error": float(delta.mean()), "max_relative_error": float((delta / denom).max()),
        "mean_relative_error": float((delta / denom).mean()),
        "box_channels": {"max_abs_error": float(delta[:, :4].max()), "mean_abs_error": float(delta[:, :4].mean())},
        "class_score_channels": {"max_abs_error": float(delta[:, 4:].max()), "mean_abs_error": float(delta[:, 4:].mean())},
    }


def box_iou(a: list[float], b: list[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def decoded_comparison(reference: list[dict[str, Any]], candidate: list[dict[str, Any]]) -> dict[str, Any]:
    unused = set(range(len(candidate)))
    matches, missing = [], []
    for item in reference:
        options = [index for index in unused if candidate[index]["class_id"] == item["class_id"]]
        if not options:
            missing.append(item)
            continue
        index = max(options, key=lambda other: box_iou(item["xyxy"], candidate[other]["xyxy"]))
        unused.remove(index)
        other = candidate[index]
        matches.append({"class_id": item["class_id"], "label": item["label"], "iou": box_iou(item["xyxy"], other["xyxy"]),
                        "confidence_difference": abs(item["confidence"] - other["confidence"]), "reference_index": item["detection_index"],
                        "candidate_index": other["detection_index"]})
    return {"reference_count": len(reference), "candidate_count": len(candidate), "matches": matches,
            "missing": missing, "extra": [candidate[index] for index in sorted(unused)],
            "class_id_sequences_equal": [x["class_id"] for x in reference] == [x["class_id"] for x in candidate],
            "minimum_matched_iou": min((item["iou"] for item in matches), default=None),
            "max_confidence_difference": max((item["confidence_difference"] for item in matches), default=0.0)}


def trace_pool_modules(model: YOLOWorld, tensor: torch.Tensor) -> list[dict[str, Any]]:
    observed: list[dict[str, Any]] = []
    names = {id(module): name for name, module in model.model.named_modules()}
    hooks = []
    for module in model.model.modules():
        if isinstance(module, nn.AdaptiveMaxPool2d):
            hooks.append(module.register_forward_hook(lambda mod, args, output, module=module: observed.append({
                "module_name": names[id(module)], "module_type": f"{type(module).__module__}.{type(module).__name__}",
                "requested_output_size": list(mod.output_size) if isinstance(mod.output_size, tuple) else mod.output_size,
                "input_shape": list(args[0].shape), "output_shape": list(output.shape),
            })))
    try:
        raw_torch_output(model, tensor)
    finally:
        for hook in hooks:
            hook.remove()
    return observed


def verify_pooling_equivalence() -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    shapes = list(EXPECTED_POOL_SHAPES) + [(5, 7), (3, 3), (2, 4)]
    for h, w in shapes:
        original = nn.AdaptiveMaxPool2d((3, 3))
        replacement = StaticAdaptiveMaxPool2d3x3((h, w))
        tensors = {
            "signed_arange": (torch.arange(2 * 3 * h * w, dtype=torch.float32).reshape(2, 3, h, w) - (h * w)),
            "negative_arange": -torch.arange(2 * 3 * h * w, dtype=torch.float32).reshape(2, 3, h, w),
            "seeded_random": torch.randn(2, 3, h, w, generator=torch.Generator().manual_seed(1000 + h * 10 + w)),
            "constant_negative": torch.full((2, 3, h, w), -7.25),
        }
        for name, value in tensors.items():
            cases.append({"input_hw": [h, w], "tensor": name, "comparison": tensor_stats(original(value), replacement(value)),
                          "windows": {"height": replacement.height_windows, "width": replacement.width_windows}})
    failed = [case for case in cases if not case["comparison"]["exact_equal"]]
    return {"method": "PyTorch adaptive-pooling boundaries floor(i*L/3):ceil((i+1)*L/3), regional amax, then concatenate",
            "cases": cases, "all_cases_exact_equal": not failed, "failed_cases": failed,
            "static_export_supported_input_shapes": [list(shape) for shape in EXPECTED_POOL_SHAPES],
            "other_tested_shapes": [[5, 7], [3, 3], [2, 4]]}


def install_static_export_pools(model: YOLOWorld, trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    attention = model.model.model[16]
    assert isinstance(attention, ImagePoolingAttn), type(attention)
    assert len(attention.im_pools) == 3
    found_shapes = tuple(tuple(item["input_shape"][-2:]) for item in trace)
    if found_shapes != EXPECTED_POOL_SHAPES:
        raise RuntimeError(f"Unexpected pool trace shapes: {found_shapes}; expected {EXPECTED_POOL_SHAPES}")
    replacements = []
    for index, shape in enumerate(found_shapes):
        original = attention.im_pools[index]
        if not isinstance(original, nn.AdaptiveMaxPool2d) or tuple(original.output_size) != (3, 3):
            raise RuntimeError(f"Unexpected pool {index}: {original!r}")
        attention.im_pools[index] = StaticAdaptiveMaxPool2d3x3(shape)
        replacements.append({"module_name": f"model.16.im_pools.{index}", "original_type": f"{type(original).__module__}.{type(original).__name__}",
                             "original_output_size": list(original.output_size), "replacement_type": f"{StaticAdaptiveMaxPool2d3x3.__module__}.{StaticAdaptiveMaxPool2d3x3.__name__}",
                             "fixed_input_hw": list(shape), "height_windows": StaticAdaptiveMaxPool2d3x3.windows(shape[0]),
                             "width_windows": StaticAdaptiveMaxPool2d3x3.windows(shape[1])})
    return replacements


def inspect_onnx(path: Path, vocabulary: list[str]) -> dict[str, Any]:
    model = onnx.load(str(path), load_external_data=False)
    onnx.checker.check_model(model)
    graph = model.graph
    dims = lambda item: [int(d.dim_value) if d.HasField("dim_value") else d.dim_param if d.HasField("dim_param") else None for d in item.type.tensor_type.shape.dim]
    initializer_dims = [{"name": x.name, "shape": list(x.dims), "data_type": onnx.TensorProto.DataType.Name(x.data_type)} for x in graph.initializer]
    operators = Counter((node.domain or "ai.onnx", node.op_type) for node in graph.node)
    metadata = {item.key: item.value for item in model.metadata_props}
    text_candidates = [item for item in initializer_dims if 149 in item["shape"]]
    names_metadata = metadata.get("names", "")
    try:
        parsed_names = json.loads(names_metadata)
    except json.JSONDecodeError:
        parsed_names = ast.literal_eval(names_metadata) if names_metadata else None
    names_in_order = (
        [parsed_names[str(i)] if str(i) in parsed_names else parsed_names[i] for i in range(149)]
        if isinstance(parsed_names, dict)
        else None
    )
    return {
        "checker": "onnx.checker.check_model passed", "model_size_bytes": path.stat().st_size,
        "inputs": [{"name": item.name, "shape": dims(item), "elem_type": onnx.TensorProto.DataType.Name(item.type.tensor_type.elem_type)} for item in graph.input],
        "outputs": [{"name": item.name, "shape": dims(item), "elem_type": onnx.TensorProto.DataType.Name(item.type.tensor_type.elem_type)} for item in graph.output],
        "node_count": len(graph.node), "initializer_count": len(graph.initializer),
        "external_tensor_dependencies": any(
            initializer.data_location == onnx.TensorProto.EXTERNAL for initializer in graph.initializer
        ),
        "operator_counts": [{"domain": domain, "op_type": op_type, "count": count} for (domain, op_type), count in sorted(operators.items())],
        "custom_domain_operators": [{"domain": domain, "op_type": op_type} for domain, op_type in operators if domain not in ("ai.onnx", "")],
        "adaptive_max_pool_operators": [node.op_type for node in graph.node if "Adaptive" in node.op_type or "adaptive" in node.op_type.lower()],
        "initializers_with_dimension_149": text_candidates, "metadata": metadata,
        "metadata_names_equal_vg150": names_in_order == vocabulary,
    }


def main() -> int:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with (OUTPUT_DIR / "run.log").open("w", encoding="utf-8", buffering=1) as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        try:
            print("Phase B1.1 YOLO-World adaptive max-pool export blocker resolution")
            random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
            torch.use_deterministic_algorithms(True, warn_only=True); torch.set_grad_enabled(False)
            if not CHECKPOINT.is_file() or not all(path.is_file() for path in IMAGE_PATHS):
                raise FileNotFoundError("Checkpoint or B1 reference image missing")
            sys.path.insert(0, str(REPO_ROOT))
            from scene_graph.mapping import build_yolo_vocabulary
            vocabulary = build_yolo_vocabulary(json.loads(VG_DICT.read_text(encoding="utf-8")))
            assert len(vocabulary) == 149
            pool_equivalence = verify_pooling_equivalence()
            write_json("pooling-equivalence.json", pool_equivalence)
            if not pool_equivalence["all_cases_exact_equal"]:
                raise AssertionError("Independent pooling equivalence tests failed")

            detector = YOLOWorld(str(CHECKPOINT))
            detector.set_classes(list(vocabulary)); detector.model.eval()
            configured_names = list(detector.model.names.values()) if isinstance(detector.model.names, dict) else list(detector.model.names)
            assert configured_names == vocabulary
            first_tensor, _ = raw_input_from_predictor(detector, IMAGE_PATHS[0])
            trace = trace_pool_modules(detector, first_tensor)
            write_json("adaptive-pool-trace.json", {"model_layer": 16, "model_layer_type": f"{type(detector.model.model[16]).__module__}.{type(detector.model.model[16]).__name__}",
                                                       "location": "neck/head (model YAML head list), not backbone SPP/SPPF", "trace": trace})
            replacements = install_static_export_pools(detector, trace)
            patched_first = raw_torch_output(detector, first_tensor)
            original_first = np.load(B1_DIR / "raw-pytorch-1.npy")
            patch_parity = raw_comparison(original_first, patched_first)
            write_json("export-only-modification.json", {"purpose": "Replace unsupported adaptive_max_pool2d symbolic path without changing pooling semantics for static 640x640 export.",
                                                           "replacements": replacements, "post_install_raw_parity_vs_saved_b1_reference": patch_parity})
            if not patch_parity["exact_equal"]:
                raise AssertionError(f"Patched PyTorch detector differs from B1 reference: {patch_parity}")

            print("Exporting configured detector with exact static regional maxima...")
            exported = Path(detector.export(format="onnx", imgsz=RAW_IMGSZ, batch=1, dynamic=False, simplify=False, nms=False, opset=OPSET, device="cpu")).resolve()
            if not exported.is_file():
                raise FileNotFoundError(f"No exported ONNX artifact: {exported}")
            onnx_path = OUTPUT_DIR / "yolov8s-world-vg150-640-static-adaptive-max.onnx"
            shutil.copy2(exported, onnx_path)
            if exported.parent == CHECKPOINT.parent and exported != onnx_path:
                exported.unlink()
            graph = inspect_onnx(onnx_path, vocabulary)
            write_json("onnx-graph-inspection.json", graph)
            if graph["outputs"] != [{"name": "output0", "shape": [1, 153, 8400], "elem_type": "FLOAT"}]:
                raise AssertionError(f"Unexpected ONNX output: {graph['outputs']}")
            if graph["custom_domain_operators"] or graph["adaptive_max_pool_operators"]:
                raise AssertionError(f"Unsupported/custom ONNX operators: {graph['custom_domain_operators']} {graph['adaptive_max_pool_operators']}")

            session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            inputs = [{"name": item.name, "shape": list(item.shape), "type": item.type} for item in session.get_inputs()]
            outputs = [{"name": item.name, "shape": list(item.shape), "type": item.type} for item in session.get_outputs()]
            if len(inputs) != 1 or len(outputs) != 1:
                raise AssertionError(f"Unexpected ORT I/O: {inputs} {outputs}")
            per_image = []
            for image_path in IMAGE_PATHS:
                tensor, rgb = raw_input_from_predictor(detector, image_path)
                torch_raw = raw_torch_output(detector, tensor)
                saved_raw = np.load(B1_DIR / f"raw-pytorch-{image_path.stem}.npy")
                onnx_raw = session.run(None, {inputs[0]["name"]: tensor.numpy()})[0]
                np.save(OUTPUT_DIR / f"raw-onnxruntime-{image_path.stem}.npy", onnx_raw)
                torch_decoded = decode_raw(torch_raw, detector.model.names, tensor, rgb, detector.predictor)
                onnx_decoded = decode_raw(onnx_raw, detector.model.names, tensor, rgb, detector.predictor)
                per_image.append({"image": image_path.relative_to(REPO_ROOT).as_posix(),
                                  "saved_b1_vs_patched_pytorch_raw": raw_comparison(saved_raw, torch_raw),
                                  "patched_pytorch_vs_onnxruntime_raw": raw_comparison(torch_raw, onnx_raw),
                                  "saved_b1_vs_onnxruntime_raw": raw_comparison(saved_raw, onnx_raw),
                                  "pytorch_decoded": torch_decoded, "onnxruntime_decoded": onnx_decoded,
                                  "decoded_comparison": decoded_comparison(torch_decoded, onnx_decoded)})
            parity = {"providers": session.get_providers(), "inputs": inputs, "outputs": outputs, "per_image": per_image}
            write_json("onnxruntime-parity.json", parity)
            raw_ok = all(item["saved_b1_vs_onnxruntime_raw"]["max_abs_error"] <= 1e-4 for item in per_image)
            decoded_ok = all(not item["decoded_comparison"]["missing"] and not item["decoded_comparison"]["extra"] and item["decoded_comparison"]["class_id_sequences_equal"] and item["decoded_comparison"]["minimum_matched_iou"] >= 0.999 for item in per_image)
            status = "success" if raw_ok and decoded_ok else "parity_failed"
            write_json("run-status.json", {"status": status, "duration_seconds": time.perf_counter() - started, "checkpoint_sha256_after": sha256(CHECKPOINT),
                                             "raw_parity_pass": raw_ok, "decoded_parity_pass": decoded_ok,
                                             "notes": ["Static 640x640 only; no embedded NMS; no quantization; exact configured VG150 vocabulary."]})
            print(status.upper())
            return 0 if status == "success" else 2
        except Exception as error:
            traceback.print_exc()
            write_json("run-status.json", {"status": "failed", "duration_seconds": time.perf_counter() - started,
                                             "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc(),
                                             "checkpoint_sha256_after": sha256(CHECKPOINT) if CHECKPOINT.is_file() else None})
            return 1


if __name__ == "__main__":
    raise SystemExit(main())