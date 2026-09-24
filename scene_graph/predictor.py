"""Relationship prediction using the trained CachedRelationshipModel."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch

from notebook.models.relationship.model import CachedRelationshipModel


@dataclass
class Prediction:
    """One relationship-model prediction for a pair."""

    class_id: int  # model output class (0..50)
    predicate: str  # decoded predicate name ("no relationship" for class 50)
    confidence: float  # softmax probability of the predicted class


@dataclass
class BatchPrediction:
    """Predictions for a whole sequence of ordered pairs, in input order."""

    class_ids: List[int]
    predicates: List[str]
    confidences: List[float]

    def __len__(self) -> int:
        return len(self.class_ids)


class RelationshipPredictor:
    """Runs the trained model on pre-extracted pair features.

    The model is expected to be already loaded and in eval mode (see
    ``SceneGraphPipeline``).  Labels must be 0-based model embedding ids.
    """

    def __init__(
        self,
        model: CachedRelationshipModel,
        idx_to_predicate: dict,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.idx_to_predicate = idx_to_predicate
        self.device = torch.device(device)

    @torch.no_grad()
    def predict(
        self,
        visual: torch.Tensor,
        subject_label: torch.Tensor,
        object_label: torch.Tensor,
        geometry: torch.Tensor,
    ) -> Prediction:
        """Predict the relationship for one (subject, object) pair.

        Inputs
        ------
        visual        : (1, 512) float32
        subject_label : (1,) int64 0-based embedding id
        object_label  : (1,) int64 0-based embedding id
        geometry      : (1, 8) float32

        Returns a Prediction with the argmax class, the decoded
        predicate name, and the softmax confidence.
        """
        logits = self.model(visual, subject_label, object_label, geometry)
        probabilities = torch.softmax(logits, dim=1)
        class_id = int(torch.argmax(probabilities, dim=1).item())
        confidence = float(probabilities[0, class_id].item())
        predicate = self.idx_to_predicate[class_id]
        return Prediction(class_id=class_id, predicate=predicate, confidence=confidence)

    @torch.no_grad()
    def predict_batch(
        self,
        visual: torch.Tensor,
        subject_labels: torch.Tensor,
        object_labels: torch.Tensor,
        geometry: torch.Tensor,
        batch_size: int = 32,
    ) -> BatchPrediction:
        """Predict relationships for N ordered pairs in batched model calls.

        Inputs (first dimension N = number of pairs, order preserved):

            visual         : (N, 512) float32
            subject_labels : (N,) int64 0-based embedding ids
            object_labels  : (N,) int64 0-based embedding ids
            geometry       : (N, 8) float32

        The rows go through the SAME model in chunks of ``batch_size``.  Each
        output row depends only on its own input row (the classifier is a
        per-row MLP and the model is in eval mode, so dropout is inactive), so
        the softmax/argmax results are identical to calling :meth:`predict`
        once per pair - only the number of forward passes changes.
        """
        total = int(visual.shape[0])
        chunk = max(1, int(batch_size))
        class_ids: List[int] = []
        confidences: List[float] = []

        for start in range(0, total, chunk):
            stop = min(start + chunk, total)
            logits = self.model(
                visual[start:stop],
                subject_labels[start:stop],
                object_labels[start:stop],
                geometry[start:stop],
            )
            probabilities = torch.softmax(logits, dim=1)
            best = torch.argmax(probabilities, dim=1)
            class_ids.extend(int(value) for value in best.tolist())
            confidences.extend(
                float(probabilities[row, best[row]].item())
                for row in range(best.shape[0])
            )

        return BatchPrediction(
            class_ids=class_ids,
            predicates=[self.idx_to_predicate[class_id] for class_id in class_ids],
            confidences=confidences,
        )