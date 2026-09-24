"""Scene graph post-processing.

Converts raw relationship predictions into the final, application-ready
scene graph:

    raw pair predictions
        -> remove "no relationship" predictions (model class 50)
        -> apply relationship confidence threshold
        -> remove duplicate relationships (conservative)
        -> final relationships

Also provides the stable identifier scheme shared by the image bounding
boxes, graph nodes, relationship endpoints and the future inspector:

    object id       = "obj_{object_index}"       e.g. obj_0, obj_1, ...
    relationship id = "rel_{relationship_index}" e.g. rel_0, rel_1, ...

The processing is model-output driven: no class-specific rules, no
hand-corrected predictions, no hard-coded relationships.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Tuple

if TYPE_CHECKING:  # pragma: no cover
    from .pipeline import PairPrediction

# Local alias for readability; PairPrediction is only imported for typing
# to avoid a circular import (pipeline.py imports this module).
PairPredictionList = List["PairPrediction"]


@dataclass
class PostprocessStats:
    """Funnel statistics explaining exactly what post-processing removed."""

    raw_predictions: int = 0
    no_relationship_removed: int = 0
    below_threshold_removed: int = 0
    duplicates_removed: int = 0
    final_relationships: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {
            "raw_predictions": self.raw_predictions,
            "no_relationship_removed": self.no_relationship_removed,
            "below_threshold_removed": self.below_threshold_removed,
            "duplicates_removed": self.duplicates_removed,
            "final_relationships": self.final_relationships,
        }


def remove_no_relationship(
    predictions: List[PairPrediction],
) -> Tuple[List[PairPrediction], int]:
    """Drop every prediction whose model class is the no-relationship class.

    The verified model convention is model class 50 = "no relationship".
    Such predictions simply never become Relationship objects; nothing
    about the mapping is renamed or shifted.
    """
    from .mapping import NO_RELATIONSHIP_CLASS

    kept = [p for p in predictions if p.class_id != NO_RELATIONSHIP_CLASS]
    return kept, len(predictions) - len(kept)


def filter_by_confidence(
    predictions: List[PairPrediction],
    threshold: float,
) -> Tuple[List[PairPrediction], int]:
    """Keep only predictions with confidence >= threshold.

    Uses the existing relationship-model softmax confidence; no new
    confidence calculation is invented.
    """
    kept = [p for p in predictions if p.confidence >= threshold]
    return kept, len(predictions) - len(kept)


def remove_duplicates(
    predictions: List[PairPrediction],
) -> Tuple[List[PairPrediction], int]:
    """Remove exact duplicate relationships, conservatively.

    A duplicate is the SAME subject object + predicate + object object.
    The first occurrence (highest detection order) is kept.

    Inverse relationships (e.g. person -> near -> car AND car -> near ->
    person) are intentionally NOT treated as duplicates: they use different
    subject/object pairs and are legitimate distinct edges.
    """
    kept: List[PairPrediction] = []
    seen = set()
    duplicates = 0
    for p in predictions:
        key = (
            p.subject.object_index,
            p.predicate,
            p.object.object_index,
        )
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        kept.append(p)
    return kept, duplicates


def postprocess_relationships(
    predictions: List[PairPrediction],
    confidence_threshold: float,
) -> Tuple[List[PairPrediction], PostprocessStats]:
    """Full post-processing funnel for raw pair predictions.

    Order of operations:
        1. remove no-relationship predictions (class 50)
        2. apply the confidence threshold
        3. remove exact duplicates

    Returns the final predictions and the funnel statistics.
    """
    stats = PostprocessStats(raw_predictions=len(predictions))

    # 1. no-relationship removal (defensive: the pipeline already buckets
    #    class 50 into `unrelated`, but this module must be safe on its own)
    predictions, removed = remove_no_relationship(predictions)
    stats.no_relationship_removed += removed

    # 2. confidence threshold
    predictions, removed = filter_by_confidence(predictions, confidence_threshold)
    stats.below_threshold_removed = removed

    # 3. duplicate removal
    predictions, removed = remove_duplicates(predictions)
    stats.duplicates_removed = removed

    stats.final_relationships = len(predictions)
    return predictions, stats


# ----------------------------------------------------------------------
# Stable identifier helpers
# ----------------------------------------------------------------------

def object_id(object_index: int) -> str:
    """Stable object identifier, e.g. obj_0, obj_1, ...

    Derived from the detection index, which is deterministic for a given
    image and detection set.  The same id is usable by the image bounding
    box, the graph node, relationship endpoints and the inspector.
    """
    return f"obj_{int(object_index)}"


def relationship_id(relationship_index: int) -> str:
    """Stable relationship identifier, e.g. rel_0, rel_1, ...

    Derived from the final (post-processed) relationship list position,
    which is deterministic for a given image and configuration.
    """
    return f"rel_{int(relationship_index)}"