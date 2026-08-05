from __future__ import annotations

import torch

from datasets.visual_genome import (
    Scene,
    SceneObject,
    Relationship,
)

from .geometry import (
    get_union_bbox,
    crop_union_region,
    bbox_relative_to_crop,
    normalize_bbox,
)

from .model import RelationshipModel


class RelationshipPredictor:

    def __init__(
        self,
        model: RelationshipModel,
        transform,
        predicate_dictionary,
        device: str = "cpu",
    ):

        self.model = model

        self.transform = transform

        self.predicate_dictionary = predicate_dictionary

        self.device = torch.device(device)

        self.model.to(self.device)

        self.model.eval()

    def _build_sample(
        self,
        scene: Scene,
        subject: SceneObject,
        obj: SceneObject,
    ):

        union_box = get_union_bbox(
            subject.bbox,
            obj.bbox,
        )

        crop = crop_union_region(
            scene.image,
            union_box,
        )

        subject_box = bbox_relative_to_crop(
            subject.bbox,
            union_box,
        )

        object_box = bbox_relative_to_crop(
            obj.bbox,
            union_box,
        )

        subject_box = normalize_bbox(
            subject_box,
            union_box,
        )

        object_box = normalize_bbox(
            object_box,
            union_box,
        )

        crop = self.transform(crop)

        image = crop.unsqueeze(0).to(self.device)

        subject_label = torch.tensor(
            [subject.label_index],
            dtype=torch.long,
            device=self.device,
        )

        object_label = torch.tensor(
            [obj.label_index],
            dtype=torch.long,
            device=self.device,
        )

        subject_bbox = torch.tensor(
            [[
                subject_box.cx,
                subject_box.cy,
                subject_box.width,
                subject_box.height,
            ]],
            dtype=torch.float32,
            device=self.device,
        )

        object_bbox = torch.tensor(
            [[
                object_box.cx,
                object_box.cy,
                object_box.width,
                object_box.height,
            ]],
            dtype=torch.float32,
            device=self.device,
        )

        return (
            image,
            subject_label,
            object_label,
            subject_bbox,
            object_bbox,
        )

    def predict_pair(
        self,
        scene: Scene,
        subject: SceneObject,
        obj: SceneObject,
    ):

        (
            image,
            subject_label,
            object_label,
            subject_bbox,
            object_bbox,
        ) = self._build_sample(
            scene,
            subject,
            obj,
        )

        with torch.no_grad():

            logits = self.model(
                image,
                subject_label,
                object_label,
                subject_bbox,
                object_bbox,
            )

        prediction = logits.argmax(
            dim=1
        ).item()

        predicate = self.predicate_dictionary[
            str(prediction)
        ]

        relationship = Relationship(
            subject=subject,
            object=obj,
            predicate_index=prediction,
            predicate=predicate,
        )

        return relationship

    def predict_scene(
        self,
        scene: Scene,
    ):

        relationships = []

        objects = scene.objects

        for i in range(len(objects)):

            for j in range(len(objects)):

                if i == j:
                    continue

                relationship = self.predict_pair(
                    scene,
                    objects[i],
                    objects[j],
                )

                relationships.append(
                    relationship
                )

        scene.relationships = relationships

        return scene