"""Phase B1.2: isolate B1.1 raw-coordinate parity discrepancy without production changes."""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from torch import nn
from ultralytics import YOLOWorld
from ultralytics.nn.modules.block import ImagePoolingAttn
from ultralytics.nn.modules.head import Detect
from ultralytics.utils import ops
from ultralytics.utils.nms import non_max_suppression

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = EXPERIMENT_ROOT / "b1-2-yoloworld-raw-parity"
B1_DIR = EXPERIMENT_ROOT / "b1-yoloworld-vg150"
B11_DIR = EXPERIMENT_ROOT / "b1-1-yoloworld-export-resolution"
CHECKPOINT = REPO_ROOT / "notebook" / "models" / "yolov8s-world.pt"
VG_DICT = REPO_ROOT / "datasets" / "visual_genome" / "VG-SGG-dicts.json"
ONNX_PATH = B11_DIR / "yolov8s-world-vg150-640-static-adaptive-max.onnx"
IMAGE_PATHS = [
    REPO_ROOT / "datasets" / "visual_genome" / "VG_100K_2" / "1.jpg",
    REPO_ROOT / "datasets" / "visual_genome" / "VG_100K_2" / "100.jpg",
    REPO_ROOT / "datasets" / "visual_genome" / "VG_100K" / "107899.jpg",
]
RAW_IMGSZ = 640
CONFIDENCE = 0.25
SEED = 20261020
EXPECTED_POOL_SHAPES = ((80, 80), (40, 40), (20, 20))


def json_ready(value: Any) -> Any:
    if isinstance(value, Path): return value.as_posix()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, torch.Tensor): return value.detach().cpu().tolist()
    if isinstance(value, dict): return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [json_ready(v) for v in value]
    return value


def write_json(name: str, data: Any) -> None:
    (OUTPUT_DIR / name).write_text(json.dumps(json_ready(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for part in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


class StaticAdaptiveMaxPool2d3x3(nn.Module):
    """Exact static-shape implementation already validated in B1.1."""
    def __init__(self, input_hw: tuple[int, int]):
        super().__init__()
        self.input_hw = tuple(input_hw)
        self.h_windows = self.windows(input_hw[0])
        self.w_windows = self.windows(input_hw[1])

    @staticmethod
    def windows(length: int) -> tuple[tuple[int, int], ...]:
        return tuple(((i * length) // 3, ((i + 1) * length + 2) // 3) for i in range(3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if tuple(x.shape[-2:]) != self.input_hw:
            raise RuntimeError(f"Unexpected static-pool input: {tuple(x.shape)}; expected {self.input_hw}")
        rows = []
        for h0, h1 in self.h_windows:
            cols = [x[:, :, h0:h1, w0:w1].amax(dim=(-2, -1), keepdim=True) for w0, w1 in self.w_windows]
            rows.append(torch.cat(cols, dim=-1))
        return torch.cat(rows, dim=-2)


def install_static_pools(model: torch.nn.Module) -> list[dict[str, Any]]:
    block = model.model[16]
    assert isinstance(block, ImagePoolingAttn)
    result = []
    for index, hw in enumerate(EXPECTED_POOL_SHAPES):
        original = block.im_pools[index]
        assert isinstance(original, nn.AdaptiveMaxPool2d) and tuple(original.output_size) == (3, 3)
        block.im_pools[index] = StaticAdaptiveMaxPool2d3x3(hw)
        result.append({"name": f"model.16.im_pools.{index}", "input_hw": list(hw), "replacement": "StaticAdaptiveMaxPool2d3x3"})
    return result


def names_list(names: Any) -> list[str]:
    return list(names.values()) if isinstance(names, dict) else list(names)


def head_state(model: torch.nn.Module) -> dict[str, Any]:
    head = model.model[-1]
    return {
        "model_training": bool(model.training), "head_type": f"{type(head).__module__}.{type(head).__name__}",
        "head_training": bool(head.training), "export": bool(getattr(head, "export", False)), "format": getattr(head, "format", None),
        "dynamic": bool(getattr(head, "dynamic", False)), "xyxy": bool(getattr(head, "xyxy", False)),
        "shape": list(head.shape) if getattr(head, "shape", None) is not None else None,
        "stride": [float(v) for v in head.stride.detach().cpu().tolist()],
        "anchors_shape": list(head.anchors.shape), "strides_shape": list(head.strides.shape),
        "nc": int(head.nc), "no": int(head.no), "reg_max": int(head.reg_max), "max_det": int(head.max_det),
        "agnostic_nms": bool(head.agnostic_nms), "end2end": bool(head.end2end),
        "fused": bool(model.is_fused()), "clip_model_is_none": getattr(model, "clip_model", None) is None,
    }


def raw_input(model: YOLOWorld, image_path: Path) -> tuple[torch.Tensor, np.ndarray]:
    with Image.open(image_path) as image:
        rgb = np.asarray(image.convert("RGB"))
    bgr = np.ascontiguousarray(rgb[..., ::-1])
    model.predict(str(image_path), conf=CONFIDENCE, imgsz=RAW_IMGSZ, rect=False, device="cpu", verbose=False)
    tensor = model.predictor.preprocess([bgr]).detach().cpu()
    assert tuple(tensor.shape) == (1, 3, 640, 640)
    return tensor, rgb


def model_raw(model: torch.nn.Module, tensor: torch.Tensor) -> np.ndarray:
    model.eval()
    with torch.inference_mode():
        result = model(tensor)
    if isinstance(result, (list, tuple)): result = result[0]
    assert isinstance(result, torch.Tensor)
    return result.detach().cpu().numpy().copy()


def compare(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    assert a.shape == b.shape, (a.shape, b.shape)
    delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
    def metrics(values: np.ndarray) -> dict[str, Any]:
        return {"max_abs_error": float(values.max()), "mean_abs_error": float(values.mean()), "exact_equal": bool(np.array_equal(values, np.zeros_like(values)))}
    return {"shape": list(a.shape), "full": metrics(delta), "box_channels": metrics(delta[:, :4]), "class_score_channels": metrics(delta[:, 4:])}


def xywh_to_xyxy(raw: np.ndarray) -> np.ndarray:
    value = raw.copy()
    value[:, 0] = raw[:, 0] - raw[:, 2] / 2
    value[:, 1] = raw[:, 1] - raw[:, 3] / 2
    value[:, 2] = raw[:, 0] + raw[:, 2] / 2
    value[:, 3] = raw[:, 1] + raw[:, 3] / 2
    return value


def decode_in_place(raw: np.ndarray, names: list[str], tensor: torch.Tensor, rgb: np.ndarray, predictor: Any) -> list[dict[str, Any]]:
    # Explicitly receives a writable array because Ultralytics NMS converts xywh to xyxy in place.
    detections = non_max_suppression(torch.from_numpy(raw), conf_thres=float(predictor.args.conf), iou_thres=float(predictor.args.iou),
                                     classes=predictor.args.classes, agnostic=bool(predictor.args.agnostic_nms),
                                     max_det=int(predictor.args.max_det), nc=0, end2end=False, rotated=False)[0]
    if len(detections): detections[:, :4] = ops.scale_boxes(tensor.shape[2:], detections[:, :4], rgb.shape)
    return [{"detection_index": i, "class_id": int(row[5].item()), "label": names[int(row[5].item())],
             "confidence": float(row[4].item()), "xyxy": [float(v) for v in row[:4].tolist()]} for i, row in enumerate(detections)]


def iou(a: list[float], b: list[float]) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = max(0, a[2]-a[0])*max(0, a[3]-a[1]) + max(0, b[2]-b[0])*max(0, b[3]-b[1]) - inter
    return inter / union if union else 0.0


def decoded_compare(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> dict[str, Any]:
    unused = set(range(len(b))); matches = []; missing = []
    for row in a:
        candidates = [j for j in unused if b[j]["class_id"] == row["class_id"]]
        if not candidates:
            missing.append(row); continue
        j = max(candidates, key=lambda x: iou(row["xyxy"], b[x]["xyxy"])); unused.remove(j)
        matches.append({"class_id": row["class_id"], "label": row["label"], "iou": iou(row["xyxy"], b[j]["xyxy"]),
                        "confidence_difference": abs(row["confidence"]-b[j]["confidence"])})
    return {"pytorch_count": len(a), "onnx_count": len(b), "class_id_sequences_equal": [x["class_id"] for x in a] == [x["class_id"] for x in b],
            "missing": missing, "extra": [b[j] for j in sorted(unused)], "matches": matches,
            "minimum_iou": min((x["iou"] for x in matches), default=None), "max_confidence_difference": max((x["confidence_difference"] for x in matches), default=0.0)}


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with (OUTPUT_DIR / "run.log").open("w", encoding="utf-8", buffering=1) as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        try:
            os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.set_grad_enabled(False)
            if not ONNX_PATH.is_file(): raise FileNotFoundError(ONNX_PATH)
            sys.path.insert(0, str(REPO_ROOT)); from scene_graph.mapping import build_yolo_vocabulary
            vocabulary = build_yolo_vocabulary(json.loads(VG_DICT.read_text(encoding="utf-8"))); assert len(vocabulary) == 149
            wrapper = YOLOWorld(str(CHECKPOINT)); wrapper.set_classes(list(vocabulary)); wrapper.model.eval()
            assert names_list(wrapper.model.names) == vocabulary
            states = {"fresh_configured_before_predictor": head_state(wrapper.model)}
            tensor0, rgb0 = raw_input(wrapper, IMAGE_PATHS[0])
            states["fresh_configured_after_predictor"] = head_state(wrapper.model)
            fresh0 = model_raw(wrapper.model, tensor0)
            saved0 = np.load(B1_DIR / "raw-pytorch-1.npy")
            states["fresh_configured_after_raw_forward"] = head_state(wrapper.model)
            pool_model = copy.deepcopy(wrapper.model).to("cpu").eval()
            pools = install_static_pools(pool_model)
            pooled0 = model_raw(pool_model, tensor0)
            states["pool_replacement_only"] = head_state(pool_model)

            export_copy = copy.deepcopy(pool_model).to("cpu").eval().float().fuse(imgsz=RAW_IMGSZ)
            for module in export_copy.modules():
                if isinstance(module, Detect):
                    module.dynamic = False; module.export = True; module.format = "onnx"; module.max_det = min(300, 8400)
                    module.agnostic_nms = False; module.xyxy = False; module.shape = None
            states["export_copy_after_fuse_and_export_flags_before_forward"] = head_state(export_copy)
            export_copy0 = model_raw(export_copy, tensor0)
            states["export_copy_after_forward"] = head_state(export_copy)
            session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"]); input_name = session.get_inputs()[0].name
            ort0 = session.run(None, {input_name: tensor0.numpy()})[0].copy()

            # This reproduces B1.1's mistaken ordering: decoding mutates raw box values in place.
            saved_for_mutation = saved0.copy(); fresh_for_mutation = fresh0.copy()
            decoded_saved = decode_in_place(saved_for_mutation, vocabulary, tensor0, rgb0, wrapper.predictor)
            decoded_fresh = decode_in_place(fresh_for_mutation, vocabulary, tensor0, rgb0, wrapper.predictor)
            mutation_evidence = {
                "raw_saved_vs_fresh_before_decode": compare(saved0, fresh0),
                "raw_saved_vs_fresh_after_decode_in_place": compare(saved_for_mutation, fresh_for_mutation),
                "saved_raw_xywh_to_xyxy_vs_fresh_post_nms_mutation": compare(xywh_to_xyxy(saved0), fresh_for_mutation),
                "saved_array_changed_by_decode": compare(saved0, saved_for_mutation),
                "fresh_array_changed_by_decode": compare(fresh0, fresh_for_mutation),
                "decoded_counts": {"saved": len(decoded_saved), "fresh": len(decoded_fresh)},
            }
            write_json("detector-state.json", {"states": states, "static_pool_replacements": pools})
            write_json("mutation-and-transformation-evidence.json", mutation_evidence)

            per_image = []
            for path in IMAGE_PATHS:
                tensor, rgb = raw_input(wrapper, path)
                fresh = model_raw(wrapper.model, tensor)
                pooled = model_raw(pool_model, tensor)
                exported_copy = model_raw(export_copy, tensor)
                ort_raw = session.run(None, {input_name: tensor.numpy()})[0].copy()
                saved = np.load(B1_DIR / f"raw-pytorch-{path.stem}.npy")
                # Preserve raw arrays: decode writable copies only.
                pt_rows = decode_in_place(fresh.copy(), vocabulary, tensor, rgb, wrapper.predictor)
                ort_rows = decode_in_place(ort_raw.copy(), vocabulary, tensor, rgb, wrapper.predictor)
                per_image.append({"image": path.relative_to(REPO_ROOT).as_posix(),
                                  "fresh_vs_saved_b1_raw": compare(fresh, saved), "fresh_vs_pool_only_raw": compare(fresh, pooled),
                                  "fresh_vs_export_copy_raw": compare(fresh, exported_copy), "export_copy_vs_onnx_raw": compare(exported_copy, ort_raw),
                                  "fresh_vs_onnx_raw": compare(fresh, ort_raw), "fresh_pytorch_decoded": pt_rows,
                                  "onnxruntime_decoded": ort_rows, "decoded_comparison": decoded_compare(pt_rows, ort_rows)})
            write_json("raw-and-decoded-comparison.json", {"onnx_provider": session.get_providers(), "per_image": per_image})
            raw_ok = all(item["fresh_vs_onnx_raw"]["full"]["max_abs_error"] <= 0.02 for item in per_image)
            decode_ok = all(not item["decoded_comparison"]["missing"] and not item["decoded_comparison"]["extra"] and item["decoded_comparison"]["class_id_sequences_equal"] and item["decoded_comparison"]["minimum_iou"] >= 0.999 for item in per_image)
            write_json("run-status.json", {"status": "success" if raw_ok and decode_ok else "failed", "raw_parity_pass_atol_0_02": raw_ok,
                                             "decoded_parity_pass": decode_ok, "duration_seconds": time.perf_counter()-started,
                                             "checkpoint_sha256_after": sha256(CHECKPOINT), "finding": "B1.1 raw discrepancy is caused by in-place NMS xywh-to-xyxy conversion before raw comparison."})
            print("SUCCESS" if raw_ok and decode_ok else "FAILED")
            return 0 if raw_ok and decode_ok else 2
        except Exception as error:
            traceback.print_exc(); write_json("run-status.json", {"status":"failed", "error_type":type(error).__name__, "error":str(error), "traceback":traceback.format_exc(), "duration_seconds":time.perf_counter()-started, "checkpoint_sha256_after":sha256(CHECKPOINT)})
            return 1


if __name__ == "__main__":
    raise SystemExit(main())