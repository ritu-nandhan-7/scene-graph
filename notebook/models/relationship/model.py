from __future__ import annotations

import torch
import torch.nn as nn

try:
    from .encoder import ImageEncoder
except ImportError:
    ImageEncoder = None

class RelationshipModel(nn.Module):
    def __init__(
        self,
        encoder: ImageEncoder,
        num_object_classes: int,
        num_predicate_classes: int,
        label_embedding_dim: int = 64,
        geometry_dim: int = 32,
        hidden_dim: int = 256,
        dropout: float = 0.30,
    ):
        super().__init__()

        # Image Encoder (Frozen CLIP)
        self.encoder = encoder

        # Object Label Embeddings
        self.object_embedding = nn.Embedding(
            num_object_classes,
            label_embedding_dim,
        )

        # Geometry Encoder
        # Input:
        # [sub_cx, sub_cy, sub_w, sub_h,
        #  obj_cx, obj_cy, obj_w, obj_h]
        self.geometry_encoder = nn.Sequential(
            nn.Linear(8, geometry_dim),
            nn.ReLU(),
        )

        # Feature Dimensions
        self.visual_dim = 512
        self.fusion_dim = (
            self.visual_dim
            + label_embedding_dim
            + label_embedding_dim
            + geometry_dim
        )

        # Relationship Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_predicate_classes),
        )

    def forward(
        self,
        image: torch.Tensor,
        subject_label: torch.Tensor,
        object_label: torch.Tensor,
        subject_bbox: torch.Tensor,
        object_bbox: torch.Tensor,
    ):
        # Visual Features
        visual_features = self.encoder(image)

        # Label Embeddings
        subject_features = self.object_embedding(subject_label)
        object_features = self.object_embedding(object_label)

                # Geometry Features
        geometry = torch.cat([subject_bbox, object_bbox], dim=1)
        geometry_features = self.geometry_encoder(geometry)

        # Feature Fusion
        features = torch.cat(
            [
                visual_features,
                subject_features,
                object_features,
                geometry_features,
            ],
            dim=1,
        )

        # Relationship Prediction
        logits = self.classifier(features)

        return logits
    
class CachedRelationshipModel(nn.Module):

    def __init__(
        self,
        num_object_classes=150,
        num_predicate_classes=51,
        visual_dim=512,
        label_embedding_dim=64,
        geometry_dim=32,
        hidden_dim=256,
        dropout=0.30,
    ):
        super().__init__()

        # ----------------------------------------------------
        # Object label embeddings
        # ----------------------------------------------------

        self.subject_embedding = nn.Embedding(
            num_object_classes,
            label_embedding_dim
        )

        self.object_embedding = nn.Embedding(
            num_object_classes,
            label_embedding_dim
        )


        # ----------------------------------------------------
        # Geometry encoder
        # Input:
        # [sub_cx, sub_cy, sub_w, sub_h,
        #  obj_cx, obj_cy, obj_w, obj_h]
        # ----------------------------------------------------

        self.geometry_encoder = nn.Sequential(

            nn.Linear(
                8,
                geometry_dim
            ),

            nn.ReLU(),
        )


        # ----------------------------------------------------
        # Fusion
        # ----------------------------------------------------

        fusion_dim = (
            visual_dim
            + label_embedding_dim
            + label_embedding_dim
            + geometry_dim
        )


        # ----------------------------------------------------
        # Relationship classifier
        # ----------------------------------------------------

        self.classifier = nn.Sequential(

            nn.Linear(
                fusion_dim,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                hidden_dim,
                num_predicate_classes
            ),
        )


    def forward(
        self,
        visual,
        subject_label,
        object_label,
        geometry,
    ):

        # ----------------------------------------------------
        # Encode object labels
        # ----------------------------------------------------

        subject_features = self.subject_embedding(
            subject_label
        )

        object_features = self.object_embedding(
            object_label
        )


        # ----------------------------------------------------
        # Encode geometry
        # ----------------------------------------------------

        geometry_features = self.geometry_encoder(
            geometry
        )


        # ----------------------------------------------------
        # Fuse everything
        # ----------------------------------------------------

        fused = torch.cat(
            [
                visual,
                subject_features,
                object_features,
                geometry_features,
            ],
            dim=1
        )


        # ----------------------------------------------------
        # Predict relationship
        # ----------------------------------------------------

        logits = self.classifier(
            fused
        )

        return logits