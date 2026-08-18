"""
Visual Genome Dataset Loader

Responsible for:
- Opening dataset files
- Loading metadata
- Providing scene reconstruction APIs

This module is the ONLY place that directly interacts with
VG-SGG.h5, image_data.json and VG-SGG-dicts.json.
"""

from pathlib import Path
import json

import h5py
from PIL import Image

from .models import (
    BoundingBox,
    ImageInfo,
    SceneObject,
    Relationship,
    Scene,
)

class VisualGenomeLoader:

    def __init__(self, dataset_root,image_root=None):

        self.dataset_root = Path(dataset_root)
        
        self.image_root = (
            Path(image_root)
            if image_root is not None
            else self.dataset_root
        )

        # ---------- Dataset paths ----------

        self.h5_path = self.dataset_root / "VG-SGG.h5"
        self.dict_path = self.dataset_root / "VG-SGG-dicts.json"
        self.image_data_path = self.dataset_root / "image_data.json"

        # ---------- Load dataset ----------

        self.h5 = h5py.File(self.h5_path, "r")

        with open(self.dict_path, "r") as f:
            self.dictionary = json.load(f)

        with open(self.image_data_path, "r") as f:
            self.image_data = json.load(f)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"images={self.num_images}, "
            f"object_classes={self.num_object_classes}, "
            f"predicate_classes={self.num_predicate_classes})"
        )

    def _get_image_info(self, image_idx):
        """
        Returns ImageInfo for one image.
        """

        info = self.image_data[image_idx]

        url = info["url"]

        folder = Path(url).parent.name
        filename = Path(url).name

        path = self.image_root / folder / filename

        return ImageInfo(
            image_index=image_idx,
            image_id=info["image_id"],
            width=info["width"],
            height=info["height"],
            folder=folder,
            filename=filename,
            path=path
        )
    def _load_image(self, image_info):
        """
        Loads a PIL image.
        """

        return Image.open(image_info.path)

    def _scale_box(self, box, image_info):
        """
        Convert a Visual Genome 1024-scale bounding box
        into original image coordinates.

        Input:
            box -> (cx, cy, width, height)

        Returns:
            BoundingBox
        """

        cx, cy, width, height = box

        scale = 1024 / max(image_info.width, image_info.height)

        cx /= scale
        cy /= scale
        width /= scale
        height /= scale

        return BoundingBox(
            cx=float(cx),
            cy=float(cy),
            width=float(width),
            height=float(height)
        )
    def _build_objects(self, image_idx, image_info):
        """
        Builds all objects belonging to one image.
        """

        first_box = int(self.h5["img_to_first_box"][image_idx])
        last_box = int(self.h5["img_to_last_box"][image_idx])

        objects = []

        for box_index in range(first_box, last_box + 1):

            label_index = int(self.h5["labels"][box_index][0])

            label = self.dictionary["idx_to_label"][str(label_index)]

            raw_box = self.h5["boxes_1024"][box_index]

            bbox = self._scale_box(raw_box, image_info)

            obj = SceneObject(
                object_index=box_index,
                label_index=label_index,
                label=label,
                bbox=bbox,          # <-- NOTE THIS
            )

            objects.append(obj)

        return objects

    def _build_relationships(self, image_idx, objects):
        """
        Builds all relationships belonging to one image.

        Parameters
        ----------
        image_idx : int
        objects : list[SceneObject]

        Returns
        -------
        list[Relationship]
        """
        first_rel = int(self.h5["img_to_first_rel"][image_idx])
        last_rel = int(self.h5["img_to_last_rel"][image_idx])

        if first_rel == -1 or last_rel == -1:
            return []

        relationships = []

        # Fast lookup:
        # Global Box Index -> SceneObject
        object_lookup = {
            obj.object_index: obj
            for obj in objects
        }

        for rel_index in range(first_rel, last_rel + 1):

            subject_box = int(
                self.h5["relationships"][rel_index][0]
            )

            object_box = int(
                self.h5["relationships"][rel_index][1]
            )

            predicate_index = int(
                self.h5["predicates"][rel_index][0]
            )

            predicate = self.dictionary[
                "idx_to_predicate"
            ][str(predicate_index)]

            relationship = Relationship(

                relationship_index=rel_index,

                subject=object_lookup[subject_box],

                object=object_lookup[object_box],

                predicate_index=predicate_index,

                predicate=predicate

            )

            relationships.append(relationship)

        return relationships
    def load_scene(self, image_idx):
        """
        Reconstructs an entire scene from the Visual Genome dataset.

        Parameters
        ----------
        image_idx : int
            Index of the image in the dataset.

        Returns
        -------
        Scene
        """

        # -------------------------
        # Image metadata
        # -------------------------
        image_info = self._get_image_info(image_idx)

        # -------------------------
        # Original image
        # -------------------------
        image = self._load_image(image_info)

        # -------------------------
        # Objects
        # -------------------------
        objects = self._build_objects(
            image_idx=image_idx,
            image_info=image_info
        )

        # -------------------------
        # Relationships
        # -------------------------
        relationships = self._build_relationships(
            image_idx=image_idx,
            objects=objects
        )

        # -------------------------
        # Scene
        # -------------------------
        scene = Scene(
            image=image,
            info=image_info,
            objects=objects,
            relationships=relationships
        )

        return scene
    @property
    def num_images(self):
        return len(self.image_data)
    @property
    def num_object_classes(self):
        return len(self.dictionary["label_to_idx"])
    @property
    def num_predicate_classes(self):
        return len(self.dictionary["predicate_to_idx"])
    def __len__(self):
        return self.num_images
    