"""Phase B1: export the production-configured YOLO-World VG150 detector to ONNX.

This is intentionally an isolated feasibility experiment. It does not import or
modify the production detector wrapper, backend, or frontend. It uses the
production vocabulary helper and the same public YOLOWorld configuration order:

    YOLOWorld(checkpoint) -> set_classes(build_yolo_vocabulary(vg_dict))

Artifacts are written beneath this script's ``b1-yoloworld-vg150`` directory.
"""

from __future__ import annotations

import contextlib
import hashlib
import inspect
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
from ultralytics import YOLOWorld
from ultralytics.utils import ops
from ultralytics.utils.nms import non_max_suppression


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = EXPERIMENT_ROOT / "b1-yoloworld-vg150"
CHECKPOINT = REPO_ROOT / "notebook" / "models" / "yolov8s-world.pt"
VG_DICT = REPO_ROOT / "datasets" / "visual_genome" / "VG-SGG-dicts.json"
IMAGE_PATHS = [
    REPO_ROOT / "datasets" / "visual_genome" / "VG_100K_2" / "1.jpg",
    REPO_ROOT / "datasets" / "visual_genome" / "VG_100K_2" / "100.jpg",
    REPO_ROOT / "datasets" / "visual_genome" / "VG_100K" / "107899.jpg",
]
CONFIDENCE = 0.25
RAW_IMGSZ = 640
OPSET = 17
SEED = 20261020


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


def tensor_summary(tensor: torch.Tensor | None) -> dict[str, Any] | None:
    if tensor is None:
        return None
    return {
        "type": type(tensor).__name__,
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "device": str(tensor.device),
        "requires_grad": bool(tensor.requires_grad),
    }


def scalar_or_list(value: Any) -> Any:
    """JSON-safe Ultralytics config values (``imgsz`` may be int or list)."""
    if isinstance(value, (list, tuple)):
        return list(value)
    return value


def model_text_state(model: YOLOWorld) -> dict[str, Any]:
    head = model.model.model[-1]
    head_text = getattr(head, "txt_feats", None)
    return {
        "wrapper_type": f"{type(model).__module__}.{type(model).__name__}",
        "detector_type": f"{type(model.model).__module__}.{type(model.model).__name__}",
        "head_type": f"{type(head).__module__}.{type(head).__name__}",
        "head_nc": getattr(head, "nc", None),
        "head_txt_feats": tensor_summary(head_text),
        # Original YOLO-World stores the post-set_classes CLIP text features
        # as an ordinary WorldModel attribute (not a state_dict buffer).
        "model_txt_feats": tensor_summary(getattr(model.model, "txt_feats", None)),
        "model_names_count": len(model.model.names),
        "model_names": list(model.model.names.values()) if isinstance(model.model.names, dict) else list(model.model.names),
        "state_dict_text_keys": [key for key in model.model.state_dict() if "txt" in key.lower() or "text" in key.lower()],
        "named_buffer_text_tensors": [
            {"name": name, **tensor_summary(value)}
            for name, value in model.model.named_buffers()
            if "txt" in name.lower() or "text" in name.lower()
        ],
        "named_parameter_text_tensors": [
            {"name": name, **tensor_summary(value)}
            for name, value in model.model.named_parameters()
            if "txt" in name.lower() or "text" in name.lower()
        ],
        "has_clip_model_attribute": hasattr(model.model, "clip_model"),
        "clip_model_type": (
            f"{type(model.model.clip_model).__module__}.{type(model.model.clip_model).__name__}"
            if getattr(model.model, "clip_model", None) is not None
            else None
        ),
    }


def dimensions(value_info: Any) -> list[Any]:
    result: list[Any] = []
    for dim in value_info.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            result.append(int(dim.dim_value))
        elif dim.HasField("dim_param"):
            result.append(dim.dim_param)
        else:
            result.append(None)
    return result


def onnx_value_info(value_info: Any) -> dict[str, Any]:
    tensor_type = value_info.type.tensor_type
    return {
        "name": value_info.name,
        "elem_type": int(tensor_type.elem_type),
        "elem_type_name": onnx.TensorProto.DataType.Name(tensor_type.elem_type),
        "shape": dimensions(value_info),
    }


def decoded_results(model: YOLOWorld, image_path: Path) -> dict[str, Any]:
    # This matches production: only conf/device/verbose are passed. All other
    # predictor values remain Ultralytics defaults and are recorded below.
    result = model.predict(str(image_path), conf=CONFIDENCE, device="cpu", verbose=False)[0]
    predictor = model.predictor
    rows = []
    for index in range(len(result.boxes)):
        cls_id = int(result.boxes.cls[index].item())
        xyxy = [float(value) for value in result.boxes.xyxy[index].cpu().tolist()]
        rows.append(
            {
                "detection_index": index,
                "class_id": cls_id,
                "label": result.names[cls_id],
                "confidence": float(result.boxes.conf[index].item()),
                "xyxy": xyxy,
            }
        )
    return {
        "image": image_path.relative_to(REPO_ROOT).as_posix(),
        "image_size": {"width": int(result.orig_shape[1]), "height": int(result.orig_shape[0])},
        "detector_input_resolution": scalar_or_list(predictor.imgsz),
        "predictor_defaults_effective": {
            "conf": float(predictor.args.conf),
            "iou": float(predictor.args.iou),
            "max_det": int(predictor.args.max_det),
            "agnostic_nms": bool(predictor.args.agnostic_nms),
            "classes": predictor.args.classes,
            "imgsz": scalar_or_list(predictor.args.imgsz),
            # This repository's pinned Ultralytics predictor namespace does
            # not expose `half`; record absence rather than changing versions.
            "half": getattr(predictor.args, "half", None),
            "device": str(predictor.args.device),
        },
        "detections": rows,
    }


def raw_input_from_predictor(model: YOLOWorld, image_path: Path) -> tuple[torch.Tensor, np.ndarray]:
    """Use Ultralytics' own predictor preprocessing for a fixed 640 square input."""
    with Image.open(image_path) as image:
        rgb = np.asarray(image.convert("RGB"))
    # Predictor preprocessing expects source arrays in BGR order before it flips
    # to RGB, matching its normal cv2 image loader path.
    bgr = np.ascontiguousarray(rgb[..., ::-1])
    # Raw ONNX export is static 640x640. Disable production's rectangular
    # letterbox optimization only in this *raw graph parity* route so the
    # PyTorch and ONNX inputs are identical. Production decoded references
    # above intentionally retain the real default `rect=True` behavior.
    model.predict(str(image_path), conf=CONFIDENCE, imgsz=RAW_IMGSZ, rect=False, device="cpu", verbose=False)
    tensor = model.predictor.preprocess([bgr]).detach().cpu()
    assert tuple(tensor.shape) == (1, 3, RAW_IMGSZ, RAW_IMGSZ), tuple(tensor.shape)
    return tensor, rgb


def raw_torch_output(model: YOLOWorld, tensor: torch.Tensor) -> np.ndarray:
    model.model.eval()
    with torch.inference_mode():
        output = model.model(tensor)
    if isinstance(output, (tuple, list)):
        output = output[0]
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"Unexpected PyTorch raw output type: {type(output)!r}")
    return output.detach().cpu().numpy()


def decode_raw(
    raw: np.ndarray,
    names: dict[int, str] | list[str],
    tensor: torch.Tensor,
    original_rgb: np.ndarray,
    predictor: Any,
) -> list[dict[str, Any]]:
    """Run the same Ultralytics NMS and scale_boxes methods as DetectionPredictor."""
    prediction = torch.from_numpy(raw)
    detections = non_max_suppression(
        prediction,
        conf_thres=float(predictor.args.conf),
        iou_thres=float(predictor.args.iou),
        classes=predictor.args.classes,
        agnostic=bool(predictor.args.agnostic_nms),
        max_det=int(predictor.args.max_det),
        nc=0,
        end2end=False,
        rotated=False,
    )[0]
    if len(detections):
        detections[:, :4] = ops.scale_boxes(tensor.shape[2:], detections[:, :4], original_rgb.shape)
    rows = []
    for index, row in enumerate(detections):
        cls_id = int(row[5].item())
        rows.append(
            {
                "detection_index": index,
                "class_id": cls_id,
                "label": names[cls_id],
                "confidence": float(row[4].item()),
                "xyxy": [float(value) for value in row[:4].tolist()],
            }
        )
    return rows


def raw_comparison(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    if a.shape != b.shape:
        return {"shape_equal": False, "pytorch_shape": list(a.shape), "onnx_shape": list(b.shape)}
    absolute = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return {
        "shape_equal": True,
        "shape": list(a.shape),
        "max_abs_error": float(absolute.max()),
        "mean_abs_error": float(absolute.mean()),
        "rms_error": float(np.sqrt(np.mean(np.square(absolute)))),
        "allclose_rtol_1e-3_atol_1e-4": bool(np.allclose(a, b, rtol=1e-3, atol=1e-4)),
        "allclose_rtol_1e-4_atol_1e-5": bool(np.allclose(a, b, rtol=1e-4, atol=1e-5)),
    }


def decoded_comparison(torch_rows: list[dict[str, Any]], onnx_rows: list[dict[str, Any]]) -> dict[str, Any]:
    comparable = min(len(torch_rows), len(onnx_rows))
    matches = []
    for index in range(comparable):
        torch_row, onnx_row = torch_rows[index], onnx_rows[index]
        box_delta = [abs(a - b) for a, b in zip(torch_row["xyxy"], onnx_row["xyxy"])]
        matches.append(
            {
                "index": index,
                "same_class_id": torch_row["class_id"] == onnx_row["class_id"],
                "same_label": torch_row["label"] == onnx_row["label"],
                "confidence_abs_error": abs(torch_row["confidence"] - onnx_row["confidence"]),
                "xyxy_max_abs_error": max(box_delta),
            }
        )
    return {
        "pytorch_count": len(torch_rows),
        "onnx_count": len(onnx_rows),
        "same_count": len(torch_rows) == len(onnx_rows),
        "positionwise": matches,
        "all_positionwise_class_ids_equal": all(item["same_class_id"] for item in matches) and len(torch_rows) == len(onnx_rows),
        "maximum_positionwise_box_error": max((item["xyxy_max_abs_error"] for item in matches), default=0.0),
        "maximum_positionwise_confidence_error": max((item["confidence_abs_error"] for item in matches), default=0.0),
    }


def inspect_onnx(path: Path, vocabulary: list[str]) -> dict[str, Any]:
    model = onnx.load(str(path), load_external_data=False)
    onnx.checker.check_model(model)
    graph = model.graph
    initializer_names = {initializer.name for initializer in graph.initializer}
    metadata = {entry.key: entry.value for entry in model.metadata_props}
    operator_counts = Counter(node.op_type for node in graph.node)
    external = []
    for initializer in graph.initializer:
        if initializer.data_location == onnx.TensorProto.EXTERNAL:
            external.append(initializer.name)
    initializer_text_candidates = [
        {
            "name": initializer.name,
            "dims": list(initializer.dims),
            "data_type": onnx.TensorProto.DataType.Name(initializer.data_type),
        }
        for initializer in graph.initializer
        if any(dimension == len(vocabulary) for dimension in initializer.dims)
    ]
    return {
        "path": path.name,
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
        "ir_version": int(model.ir_version),
        "opset_imports": [{"domain": entry.domain, "version": int(entry.version)} for entry in model.opset_import],
        "inputs": [onnx_value_info(item) for item in graph.input if item.name not in initializer_names],
        "outputs": [onnx_value_info(item) for item in graph.output],
        "node_count": len(graph.node),
        "initializer_count": len(graph.initializer),
        "operator_counts": dict(sorted(operator_counts.items())),
        "custom_domain_nodes": [
            {"name": node.name, "op_type": node.op_type, "domain": node.domain}
            for node in graph.node
            if node.domain not in ("", "ai.onnx")
        ],
        "external_data_initializer_names": external,
        "metadata": metadata,
        "metadata_names_count": len(json.loads(metadata["names"])) if metadata.get("names", "").startswith("{") else None,
        "metadata_names_equal_vg150": (
            [json.loads(metadata["names"])[str(index)] for index in range(len(vocabulary))] == vocabulary
            if metadata.get("names", "").startswith("{")
            else False
        ),
        "initializers_with_dimension_149": initializer_text_candidates,
        "validation": "onnx.checker.check_model passed",
    }


def environment() -> dict[str, Any]:
    import ultralytics

    return {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "ultralytics": ultralytics.__version__,
        "torch": torch.__version__,
        "onnx": onnx.__version__,
        "onnxruntime": ort.__version__,
        "onnxruntime_available_providers": ort.get_available_providers(),
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_cuda_device_count": torch.cuda.device_count(),
        "checkpoint": {
            "relative_path": CHECKPOINT.relative_to(REPO_ROOT).as_posix(),
            "size_bytes": CHECKPOINT.stat().st_size,
            "sha256": sha256(CHECKPOINT),
        },
        "export_api": {
            "method": "YOLOWorld.export",
            "signature": str(inspect.signature(YOLOWorld.export)),
            "set_classes_signature": str(inspect.signature(YOLOWorld.set_classes)),
            "requested": {
                "format": "onnx",
                "imgsz": RAW_IMGSZ,
                "batch": 1,
                "dynamic": False,
                "simplify": False,
                "nms": False,
                "opset": OPSET,
                "device": "cpu",
            },
        },
    }


def main() -> int:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = OUTPUT_DIR / "run.log"
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8", buffering=1) as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        try:
            print("Phase B1 YOLO-World VG150 ONNX export and parity")
            print(f"repo={REPO_ROOT}")
            print(f"output={OUTPUT_DIR}")
            random.seed(SEED)
            np.random.seed(SEED)
            torch.manual_seed(SEED)
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.set_grad_enabled(False)
            if not CHECKPOINT.is_file():
                raise FileNotFoundError(CHECKPOINT)
            missing_images = [str(path) for path in IMAGE_PATHS if not path.is_file()]
            if missing_images:
                raise FileNotFoundError(f"Missing reference images: {missing_images}")

            vg_dict = json.loads(VG_DICT.read_text(encoding="utf-8"))
            # Import exactly the production helper after adding repo root.
            sys.path.insert(0, str(REPO_ROOT))
            from scene_graph.mapping import build_yolo_vocabulary

            vocabulary = build_yolo_vocabulary(vg_dict)
            assert len(vocabulary) == 149
            assert vocabulary == [name for name in vg_dict["label_to_idx"] if name != "background"]
            write_json("environment.json", environment())
            write_json("vocabulary.json", {"count": len(vocabulary), "classes": vocabulary})

            print("Loading YOLOWorld and configuring exact production vocabulary...")
            detector = YOLOWorld(str(CHECKPOINT))
            detector.set_classes(list(vocabulary))
            detector.model.eval()
            configured_state = model_text_state(detector)
            assert configured_state["model_names"] == vocabulary
            write_json("configured-model-state.json", configured_state)

            print("Capturing production-style decoded PyTorch references...")
            decoded_reference = [decoded_results(detector, path) for path in IMAGE_PATHS]
            write_json("pytorch-production-reference.json", decoded_reference)

            print("Capturing raw configured PyTorch outputs at fixed 640x640...")
            raw_references: list[dict[str, Any]] = []
            tensors: list[tuple[Path, torch.Tensor, np.ndarray, np.ndarray]] = []
            for image_path in IMAGE_PATHS:
                tensor, original_rgb = raw_input_from_predictor(detector, image_path)
                raw_output = raw_torch_output(detector, tensor)
                raw_file = OUTPUT_DIR / f"raw-pytorch-{image_path.stem}.npy"
                np.save(raw_file, raw_output)
                decoded = decode_raw(raw_output, detector.model.names, tensor, original_rgb, detector.predictor)
                raw_references.append(
                    {
                        "image": image_path.relative_to(REPO_ROOT).as_posix(),
                        "input_tensor": {"shape": list(tensor.shape), "dtype": str(tensor.numpy().dtype)},
                        "raw_output": {"path": raw_file.name, "shape": list(raw_output.shape), "dtype": str(raw_output.dtype)},
                        "decoded_with_ultralytics_nms": decoded,
                    }
                )
                tensors.append((image_path, tensor, original_rgb, raw_output))
            write_json("pytorch-raw-reference.json", raw_references)

            print("Exporting configured detector using official Ultralytics API...")
            exported = Path(
                detector.export(
                    format="onnx",
                    imgsz=RAW_IMGSZ,
                    batch=1,
                    dynamic=False,
                    simplify=False,
                    nms=False,
                    opset=OPSET,
                    device="cpu",
                )
            ).resolve()
            if not exported.is_file():
                raise FileNotFoundError(f"Ultralytics export did not create a file: {exported}")
            onnx_path = OUTPUT_DIR / "yolov8s-world-vg150-640.onnx"
            shutil.copy2(exported, onnx_path)
            # Ultralytics derives its temporary export location from the .pt
            # checkpoint path. Retain only the isolated experiment copy.
            if exported != onnx_path and exported.parent == CHECKPOINT.parent:
                exported.unlink()
            write_json("export.json", {"ultralytics_return_path": str(exported), "copied_artifact": onnx_path.name})

            print("Inspecting and validating ONNX graph...")
            graph_info = inspect_onnx(onnx_path, vocabulary)
            write_json("onnx-graph-inspection.json", graph_info)

            print("Running ONNX Runtime CPU parity checks...")
            session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            session_inputs = [
                {"name": item.name, "shape": list(item.shape), "type": item.type} for item in session.get_inputs()
            ]
            session_outputs = [
                {"name": item.name, "shape": list(item.shape), "type": item.type} for item in session.get_outputs()
            ]
            if len(session_inputs) != 1:
                raise RuntimeError(f"Expected exactly one ONNX input, got {session_inputs}")
            parity = []
            for image_path, tensor, original_rgb, torch_raw in tensors:
                ort_outputs = session.run(None, {session_inputs[0]["name"]: tensor.numpy()})
                if len(ort_outputs) != 1:
                    raise RuntimeError(f"Expected one ONNX output, got {len(ort_outputs)}")
                onnx_raw = ort_outputs[0]
                onnx_file = OUTPUT_DIR / f"raw-onnxruntime-{image_path.stem}.npy"
                np.save(onnx_file, onnx_raw)
                torch_decoded = decode_raw(torch_raw, detector.model.names, tensor, original_rgb, detector.predictor)
                onnx_decoded = decode_raw(onnx_raw, detector.model.names, tensor, original_rgb, detector.predictor)
                parity.append(
                    {
                        "image": image_path.relative_to(REPO_ROOT).as_posix(),
                        "raw_output": raw_comparison(torch_raw, onnx_raw),
                        "pytorch_decoded": torch_decoded,
                        "onnxruntime_decoded": onnx_decoded,
                        "decoded_comparison": decoded_comparison(torch_decoded, onnx_decoded),
                        "onnx_raw_path": onnx_file.name,
                    }
                )
            write_json(
                "onnxruntime-parity.json",
                {
                    "providers_used": session.get_providers(),
                    "inputs": session_inputs,
                    "outputs": session_outputs,
                    "per_image": parity,
                },
            )

            write_json(
                "run-status.json",
                {
                    "status": "success",
                    "duration_seconds": time.perf_counter() - started,
                    "notes": [
                        "Export was requested after YOLOWorld.set_classes() with the production VG150 ordering.",
                        "NMS was intentionally external (nms=False); ONNX emits raw detector predictions.",
                        "No browser runtime was tested in Phase B1.",
                    ],
                },
            )
            print("SUCCESS")
            return 0
        except Exception as error:
            traceback.print_exc()
            write_json(
                "run-status.json",
                {
                    "status": "failed",
                    "duration_seconds": time.perf_counter() - started,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                },
            )
            return 1


if __name__ == "__main__":
    raise SystemExit(main())