"""Reusable Scene Graph inference pipeline.

Converts the notebook-level inference from ``notebook/05_testing.ipynb``
into a clean, importable component:

    image
    -> YOLO-World detection
    -> model-compatible objects
    -> ordered object pairs
    -> union crop + 8-D geometry + 512-D CLIP visual feature
    -> CachedRelationshipModel (51 classes)
    -> "no relationship" filtering
    -> Scene (existing Visual Genome SDK structures) + summary

This is the ML foundation that the later phases (FastAPI, React) build
on.  It reuses the repository's trained weights and data structures
instead of inventing parallel ones.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
from PIL import Image

from datasets.visual_genome import (
    ImageInfo,
    Relationship,
    Scene,
    SceneObject,
    VisualGenomeLoader,
)

from notebook.models.relationship.encoder import ImageEncoder
from notebook.models.relationship.model import CachedRelationshipModel

from .detector import YoloWorldDetector
from .features import extract_pair_features, extract_pair_features_batch
from .mapping import (
    NO_RELATIONSHIP_CLASS,
    build_idx_to_predicate,
    build_label_name_to_vg_id,
    build_yolo_vocabulary,
)
from .postprocessing import (
    PostprocessStats,
    object_id,
    postprocess_relationships,
    relationship_id,
)
from .predictor import RelationshipPredictor

ImageSource = Union[str, os.PathLike, Image.Image]


@dataclass
class PairPrediction:
    """Prediction joined with its subject/object SceneObjects."""

    subject: SceneObject
    object: SceneObject
    predicate: str  # decoded name, e.g. "riding" or "no relationship"
    class_id: int  # model output class
    confidence: float


@dataclass
class SceneGraphResult:
    """Structured pipeline output: the Scene plus per-pair predictions.

    ``scene.relationships`` contains ONLY the final post-processed
    relationships (no-relationship removed, confidence threshold applied,
    duplicates removed).  ``related`` holds those same final predictions,
    ``unrelated`` holds the raw class-50 predictions, and ``stats``
    explains exactly what the post-processing funnel removed.
    """

    scene: Scene
    related: List[PairPrediction]
    unrelated: List[PairPrediction]
    stats: Optional[PostprocessStats] = None
    #: Internal per-stage timings in seconds (profiling only).  Deliberately NOT
    #: part of ``to_dict()``: the public response schema stays
    #: {"image", "objects", "relationships"}.
    timings: Optional[dict] = None

    def to_dict(self) -> dict:
        """JSON-serializable, application-facing scene graph.

        Structure (no tensors, logits or embeddings):
        {
          "image": {"width": ..., "height": ...},
          "objects": [
            {"id": "obj_0", "label": "person", "confidence": 0.81,
             "bbox": {"cx": ..., "cy": ..., "width": ..., "height": ...}},
            ...
          ],
          "relationships": [
            {"id": "rel_0", "subject_id": "obj_0", "predicate": "riding",
             "object_id": "obj_3", "confidence": 0.2179},
            ...
          ]
        }
        """
        objects = [
            {
                "id": object_id(obj.object_index),
                "label": obj.label,
                "confidence": obj.confidence,
                "bbox": {
                    "cx": obj.bbox.cx,
                    "cy": obj.bbox.cy,
                    "width": obj.bbox.width,
                    "height": obj.bbox.height,
                },
            }
            for obj in self.scene.objects
        ]
        relationships = [
            {
                "id": relationship_id(rel.relationship_index),
                "subject_id": object_id(rel.subject.object_index),
                "predicate": rel.predicate,
                "object_id": object_id(rel.object.object_index),
                "confidence": rel.confidence,
            }
            for rel in self.scene.relationships
        ]
        return {
            "image": {
                "width": self.scene.info.width,
                "height": self.scene.info.height,
            },
            "objects": objects,
            "relationships": relationships,
        }


class SceneGraphPipeline:
    """End-to-end image -> objects -> relationships pipeline."""
    def __init__(
        self,
        checkpoint_path: Union[str, os.PathLike],
        yolo_model_path: Union[str, os.PathLike],
        dataset_root: Union[str, os.PathLike],
        image_root: Optional[Union[str, os.PathLike]] = None,
        device: str = "auto",
        conf_threshold: float = 0.25,
        expansion: float = 0.20,
        skip_same_label: bool = True,
        relationship_threshold: float = 0.30,
        pair_batch_size: int = 8,
        model_batch_size: int = 32,
        batch_inference: bool = True,
    ) -> None:
        """Load detector, visual encoder and trained relationship model.

        Parameters
        ----------
        checkpoint_path : path to best_full_fusion_30epoch.pt
        yolo_model_path : path to yolov8s-world.pt
        dataset_root    : folder containing VG-SGG.h5 / VG-SGG-dicts.json
        image_root      : Visual Genome image root (defaults to dataset_root)
        device          : "auto", "cpu", or "cuda"
        conf_threshold  : YOLO confidence threshold
        expansion       : union-box expansion used at training time (0.20)
        skip_same_label : exclude pairs where both objects share a label
                          (same behavior as the working notebook test)
        relationship_threshold : minimum relationship-model confidence for
                          a predicted predicate to enter the final scene
                          graph (post-processing stage; configurable, 0.30
                          is a starting value, not a validated optimum)
        pair_batch_size : number of union crops pushed through CLIP in one
                          forward pass (CPU-time optimisation; does not change
                          the features, only how many calls are made).  Measured
                          on the reference 800x600 image with 68 pairs:
                          1 -> 387 ms/pair, 4 -> 204 ms/pair, 8 -> 140 ms/pair,
                          16 -> 171 ms/pair, 32 -> 183 ms/pair, 64 -> 198 ms/pair
                          (best of two interleaved rounds).  8 is the measured
                          optimum and the run-to-run spread on this machine is
                          larger than the 8-vs-16 difference, so the exact value
                          is not critical - anything in 4..16 beats one-at-a-time.
        model_batch_size : number of ordered pairs pushed through the
                          relationship model in one forward pass
        batch_inference : set False to fall back to the original one-pair-at-a-
                          time path (identical results, used as a reference and
                          for debugging).  No model, weight or threshold changes
                          with this switch.
        """
        self.device = self._resolve_device(device)

        # --------------------------------------------------------------
        # Visual Genome metadata (vocabulary + index maps)
        # --------------------------------------------------------------
        self.loader = VisualGenomeLoader(dataset_root=dataset_root, image_root=image_root)
        vg_dict = self.loader.dictionary
        self.label_name_to_vg_id = build_label_name_to_vg_id(vg_dict)
        self.idx_to_predicate = build_idx_to_predicate(vg_dict)
        self.vg_classes = build_yolo_vocabulary(vg_dict)

        # --------------------------------------------------------------
        # YOLO-World detector
        # --------------------------------------------------------------
        self.detector = YoloWorldDetector(
            model_path=yolo_model_path,
            class_names=self.vg_classes,
            conf=conf_threshold,
            device=self.device,
        )

        # --------------------------------------------------------------
        # Frozen visual encoder (open_clip ViT-B/32, 512-D features)
        # --------------------------------------------------------------
        self.visual_encoder = ImageEncoder(
            model_name="ViT-B-32",
            pretrained="openai",
            device=self.device,
        )

        # --------------------------------------------------------------
        # Trained relationship model (architecture must match checkpoint)
        # --------------------------------------------------------------
        self.model = CachedRelationshipModel(
            num_object_classes=150,
            num_predicate_classes=51,
            visual_dim=512,
            label_embedding_dim=64,
            geometry_dim=32,
            hidden_dim=256,
            dropout=0.30,
        )
        checkpoint = torch.load(
            str(checkpoint_path),
            map_location=self.device,
            weights_only=False,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.predictor = RelationshipPredictor(
            model=self.model,
            idx_to_predicate=self.idx_to_predicate,
            device=self.device,
        )

        self.expansion = expansion
        self.skip_same_label = skip_same_label
        self.relationship_threshold = relationship_threshold
        self.pair_batch_size = pair_batch_size
        self.model_batch_size = model_batch_size
        self.batch_inference = batch_inference

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def analyze(self, image: ImageSource) -> SceneGraphResult:
        """Run the full inference pipeline on one image.

        Returns a ``SceneGraphResult`` whose ``.scene`` is the existing
        Visual Genome ``Scene``: detected objects plus the predicted
        relationships (only pairs decoded as a real predicate).  Pairs
        predicted as "no relationship" are available in ``.unrelated``.
        """
        image = self._to_pil_image(image)
        timings: dict = {}

        # ---- 1. Object detection --------------------------------------
        t0 = time.perf_counter()
        detections = self.detector.detect(image)
        timings["detect"] = time.perf_counter() - t0

        # ---- 2. Detections -> model-compatible SceneObjects ------------
        t0 = time.perf_counter()
        objects: List[SceneObject] = []
        for det in detections:
            vg_label_id = self.label_name_to_vg_id.get(det.label)
            if vg_label_id is None:
                # detection label is not one of the 150 VG classes
                continue
            objects.append(
                SceneObject(
                    object_index=det.detection_index,
                    label_index=vg_label_id,  # 1-based VG id (SDK convention)
                    label=det.label,
                    bbox=det.bbox,
                    confidence=det.confidence,
                )
            )
        timings["objects"] = time.perf_counter() - t0

        # ---- 3. Ordered pairs -> relationship predictions --------------
        #  The pair ORDER and the skip_same_label rule are unchanged; only the
        #  CLIP pass and the relationship-model pass are batched instead of
        #  being executed once per pair.
        t0 = time.perf_counter()
        pairs: List[Tuple[SceneObject, SceneObject]] = []
        for i, subject in enumerate(objects):
            for j, obj in enumerate(objects):
                if i == j:
                    continue
                if self.skip_same_label and subject.label == obj.label:
                    continue
                pairs.append((subject, obj))
        timings["pairs"] = time.perf_counter() - t0

        features = None
        t0 = time.perf_counter()
        if self.batch_inference and pairs:
            features = extract_pair_features_batch(
                image,
                [(subject.bbox, obj.bbox) for subject, obj in pairs],
                self.visual_encoder,
                device=self.device,
                expansion=self.expansion,
                batch_size=self.pair_batch_size,
            )
        timings["features"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        if features is not None:
            subject_labels = torch.tensor(
                [subject.label_index - 1 for subject, _ in pairs],
                dtype=torch.long,
                device=self.device,
            )
            object_labels = torch.tensor(
                [obj.label_index - 1 for _, obj in pairs],
                dtype=torch.long,
                device=self.device,
            )
            batched = self.predictor.predict_batch(
                visual=features.visual,
                subject_labels=subject_labels,
                object_labels=object_labels,
                geometry=features.geometry,
                batch_size=self.model_batch_size,
            )
            predictions: List[PairPrediction] = [
                PairPrediction(
                    subject=subject,
                    object=obj,
                    predicate=batched.predicates[index],
                    class_id=batched.class_ids[index],
                    confidence=batched.confidences[index],
                )
                for index, (subject, obj) in enumerate(pairs)
            ]
        else:
            # Reference path (batch_inference=False): one pair at a time.
            predictions = [
                self._predict_pair(image, subject, obj) for subject, obj in pairs
            ]
        timings["relationship_model"] = time.perf_counter() - t0

        # ---- 4. Post-processing -> final scene graph -------------------
        #     raw predictions
        #       -> remove no-relationship (class 50)
        #       -> confidence threshold
        #       -> duplicate removal
        t0 = time.perf_counter()
        final_predictions, stats = postprocess_relationships(
            predictions, self.relationship_threshold
        )
        timings["postprocess"] = time.perf_counter() - t0

        related = final_predictions
        unrelated = [
            p for p in predictions if p.class_id == NO_RELATIONSHIP_CLASS
        ]

        relationships = [
            Relationship(
                relationship_index=rel_index,
                subject=pred.subject,
                object=pred.object,
                predicate_index=pred.class_id + 1,  # 1-based VG predicate id
                predicate=pred.predicate,
                confidence=pred.confidence,
            )
            for rel_index, pred in enumerate(related)
        ]

        width, height = image.size
        info = ImageInfo(width=width, height=height)
        scene = Scene(image=image, info=info, objects=objects, relationships=relationships)
        timings["total"] = sum(
            timings.get(key, 0.0)
            for key in ("detect", "objects", "pairs", "features", "relationship_model",
                        "postprocess")
        )

        return SceneGraphResult(
            scene=scene,
            related=related,
            unrelated=unrelated,
            stats=stats,
            timings=timings,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _predict_pair(
        self,
        image: Image.Image,
        subject: SceneObject,
        obj: SceneObject,
    ) -> PairPrediction:
        features = extract_pair_features(
            image,
            subject.bbox,
            obj.bbox,
            self.visual_encoder,
            device=self.device,
            expansion=self.expansion,
        )
        # SceneObject.label_index is the 1-based VG id -> 0-based embedding id
        subject_label = torch.tensor(
            [subject.label_index - 1],
            dtype=torch.long,
            device=self.device,
        )
        object_label = torch.tensor(
            [obj.label_index - 1],
            dtype=torch.long,
            device=self.device,
        )
        prediction = self.predictor.predict(
            visual=features.visual,
            subject_label=subject_label,
            object_label=object_label,
            geometry=features.geometry,
        )
        return PairPrediction(
            subject=subject,
            object=obj,
            predicate=prediction.predicate,
            class_id=prediction.class_id,
            confidence=prediction.confidence,
        )

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        return "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def _to_pil_image(image: ImageSource) -> Image.Image:
        if isinstance(image, (str, os.PathLike)):
            return Image.open(image).convert("RGB")
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        # numpy array input (assumed RGB)
        return Image.fromarray(image).convert("RGB")