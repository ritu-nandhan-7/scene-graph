"""Scene Graph Explorer - reusable ML inference pipeline.

Phase A component: converts the working notebook-level inference
(``notebook/05_testing.ipynb``) into a clean, importable pipeline that
the later full-stack phases (FastAPI, React) build on.

Usage::

    from scene_graph import SceneGraphPipeline

    pipeline = SceneGraphPipeline(
        checkpoint_path="notebook/models/best_full_fusion_30epoch.pt",
        yolo_model_path="notebook/models/yolov8s-world.pt",
        dataset_root="datasets/visual_genome",
    )
    result = pipeline.analyze("my_image.jpg")
    scene = result.scene   # SDK Scene: objects + predicted relationships
"""

from .detector import Detection, YoloWorldDetector
from .features import PairFeatures, extract_pair_features
from .mapping import (
    NO_RELATIONSHIP_CLASS,
    NO_RELATIONSHIP_PREDICATE,
    build_idx_to_predicate,
    build_label_name_to_vg_id,
    build_yolo_vocabulary,
    model_class_to_vg_predicate_id,
    model_id_to_vg_label_id,
    vg_label_id_to_model_id,
)
from .pipeline import PairPrediction, SceneGraphPipeline, SceneGraphResult
from .postprocessing import (
    PostprocessStats,
    filter_by_confidence,
    object_id,
    postprocess_relationships,
    relationship_id,
    remove_duplicates,
    remove_no_relationship,
)
from .predictor import Prediction, RelationshipPredictor

__all__ = [
    "Detection",
    "YoloWorldDetector",
    "PairFeatures",
    "extract_pair_features",
    "PairPrediction",
    "Prediction",
    "RelationshipPredictor",
    "SceneGraphPipeline",
    "SceneGraphResult",
    # post-processing
    "PostprocessStats",
    "filter_by_confidence",
    "remove_duplicates",
    "remove_no_relationship",
    "postprocess_relationships",
    "object_id",
    "relationship_id",
    # mapping constants / helpers
    "NO_RELATIONSHIP_CLASS",
    "NO_RELATIONSHIP_PREDICATE",
    "build_idx_to_predicate",
    "build_label_name_to_vg_id",
    "build_yolo_vocabulary",
    "model_class_to_vg_predicate_id",
    "model_id_to_vg_label_id",
    "vg_label_id_to_model_id",
]