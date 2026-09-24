"""YOLO-World object detection wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Union

import torch
from ultralytics import YOLOWorld

from datasets.visual_genome import BoundingBox


@dataclass
class Detection:
    """One object detected by YOLO-World."""

    detection_index: int
    label: str
    confidence: float
    xyxy: tuple  # (x1, y1, x2, y2) in original-image pixels
    bbox: BoundingBox  # center format (cx, cy, width, height)


class YoloWorldDetector:
    """Thin wrapper around ultralytics YOLO-World.

    The model is re-headed with the Visual Genome compatible vocabulary
    via ``set_classes``, exactly like notebooks 02 / 05_testing.

    Note: ultralytics ``boxes.cls`` is the 0-based index into this custom
    vocabulary (NOT a Visual Genome id).  Only the label *string* is
    carried forward; the pipeline resolves the VG id from the name.
    """

    def __init__(
        self,
        model_path: Union[str, "os.PathLike[str]"],
        class_names: List[str],
        conf: float = 0.25,
        device: str = "auto",
    ) -> None:
        self.model_path = str(model_path)
        self.class_names = list(class_names)
        self.conf = conf
        self.device = self._resolve_device(device)
        self.model = YOLOWorld(self.model_path)
        self.model.set_classes(self.class_names)

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def detect(self, image) -> List[Detection]:
        """Run detection on a PIL image, image path, or numpy array."""
        results = self.model.predict(
            image,
            conf=self.conf,
            device=self.device,
            verbose=False,
        )
        boxes = results[0].boxes
        names = results[0].names

        detections: List[Detection] = []
        for i in range(len(boxes)):
            class_id = int(boxes.cls[i].item())
            confidence = float(boxes.conf[i].item())
            x1, y1, x2, y2 = (float(v) for v in boxes.xyxy[i].cpu().tolist())
            bbox = BoundingBox(
                cx=(x1 + x2) / 2,
                cy=(y1 + y2) / 2,
                width=x2 - x1,
                height=y2 - y1,
            )
            detections.append(
                Detection(
                    detection_index=i,
                    label=names[class_id],
                    confidence=confidence,
                    xyxy=(x1, y1, x2, y2),
                    bbox=bbox,
                )
            )
        return detections