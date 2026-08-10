from __future__ import annotations

from pathlib import Path
from typing import Optional

import random

import torch
from torch.utils.data import Dataset

from PIL import Image

from datasets.visual_genome import (
    VisualGenomeLoader,
    Scene,
    SceneObject,
    BoundingBox,
)

from .geometry import (
    bbox_to_xyxy,
    get_union_bbox,
    crop_union_region,
    bbox_relative_to_crop,
    normalize_bbox,
)

class RelationshipDataset(Dataset):

    def __init__(
        self,
        loader: VisualGenomeLoader,
        image_indices,
        transform=None,
        negative_ratio: float = 1.0,
        expansion: float = 0.20,
    ):
        super().__init__()

        self.loader = loader
        self.image_indices = list(image_indices)

        self.transform = transform

        self.negative_ratio = negative_ratio
        self.expansion = expansion

        self.samples = []

        self._build_index()

    def _build_index(self):

        for image_index in self.image_indices:

            scene = self.loader.load_scene(
                image_index
            )

            self._add_positive_samples(scene)

            self._add_negative_samples(scene)
    
    def _add_positive_samples(
        self,
        scene: Scene
    ):

        for relationship in scene.relationships:

            self.samples.append(

                (
                    scene.info.image_index,

                    relationship.subject.object_index,

                    relationship.object.object_index,

                    relationship.predicate_index
                )

            )
    
    def _get_negative_pairs(
        self,
        scene: Scene
    ):

        positive_pairs = {

            (
                rel.subject.object_index,
                rel.object.object_index
            )

            for rel in scene.relationships

        }

        candidate_pairs = []

        for subject in scene.objects:

            for obj in scene.objects:

                if subject.object_index == obj.object_index:
                    continue

                pair = (
                    subject.object_index,
                    obj.object_index
                )

                if pair in positive_pairs:
                    continue

                # Spatial filtering comes here later

                candidate_pairs.append(pair)

        return candidate_pairs

    def _add_negative_samples(
        self,
        scene: Scene
    ):

        candidate_pairs = self._get_negative_pairs(scene)

        num_positive = len(scene.relationships)

        num_negative = int(
            self.negative_ratio * num_positive
        )

        num_negative = min(
            num_negative,
            len(candidate_pairs)
        )

        selected_pairs = random.sample(
            candidate_pairs,
            num_negative
        )

        no_relation_index = (
            len(
                self.loader.dictionary["predicate_to_idx"]
            ) + 1
        )

        for subject_index, object_index in selected_pairs:

            self.samples.append(

                (
                    scene.info.image_index,

                    subject_index,

                    object_index,

                    no_relation_index
                )

            )
            
    
    def _get_objects(
        self,
        scene: Scene,
        subject_index: int,
        object_index: int,
    ):
        subject = next(
            obj for obj in scene.objects
            if obj.object_index == subject_index
        )

        obj = next(
            obj for obj in scene.objects
            if obj.object_index == object_index
        )

        return subject, obj

    def _build_training_sample(
        self,
        image_index: int,
        subject_index: int,
        object_index: int,
        predicate_index: int,
    ):
        """
        Constructs one complete training sample.
        """

        scene = self.loader.load_scene(image_index)

        subject, obj = self._get_objects(
            scene,
            subject_index,
            object_index,
        )

        union_box = get_union_bbox(
            subject.bbox,
            obj.bbox,
            expansion=self.expansion,
        )

        crop = crop_union_region(
            scene.image,
            union_box,
        )

        subject_bbox = bbox_relative_to_crop(
            subject.bbox,
            union_box,
        )

        object_bbox = bbox_relative_to_crop(
            obj.bbox,
            union_box,
        )

        subject_bbox = normalize_bbox(
            subject_bbox,
            union_box,
        )

        object_bbox = normalize_bbox(
            object_bbox,
            union_box,
        )

        if self.transform is not None:
            crop = self.transform(crop)

        sample = {

            "image_index": image_index,

            "image": crop,

            "subject_label": torch.tensor(
                subject.label_index,
                dtype=torch.long,
            ),

            "object_label": torch.tensor(
                obj.label_index,
                dtype=torch.long,
            ),

            "subject_bbox": torch.tensor(
                [
                    subject_bbox.cx,
                    subject_bbox.cy,
                    subject_bbox.width,
                    subject_bbox.height,
                ],
                dtype=torch.float32,
            ),

            "object_bbox": torch.tensor(
                [
                    object_bbox.cx,
                    object_bbox.cy,
                    object_bbox.width,
                    object_bbox.height,
                ],
                dtype=torch.float32,
            ),

            "target": torch.tensor(
                predicate_index,
                dtype=torch.long,
            ),
        }

        return sample
    
    def __len__(self):

        return len(self.samples)
    

    def __getitem__(
        self,
        index: int
    ):

        image_index, subject_index, object_index, predicate_index = self.samples[index]

        return self._build_training_sample(
            image_index,
            subject_index,
            object_index,
            predicate_index
        )