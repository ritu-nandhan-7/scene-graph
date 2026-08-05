"""
Core data models for the Visual Genome SDK.

These classes represent a fully reconstructed scene.
They are completely independent of HDF5, JSON, or visualization.

Nothing in this file should know how data is loaded.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from PIL import Image
@dataclass(slots=True)
class BoundingBox:
    """
    Bounding box in (center_x, center_y, width, height) format.
    """

    cx: float
    cy: float
    width: float
    height: float

    @property
    def x1(self):
        return self.cx - self.width / 2

    @property
    def y1(self):
        return self.cy - self.height / 2

    @property
    def x2(self):
        return self.cx + self.width / 2

    @property
    def y2(self):
        return self.cy + self.height / 2

    def as_xywh(self):
        return (
            self.cx,
            self.cy,
            self.width,
            self.height,
        )

    def as_xyxy(self):
        return (
            self.x1,
            self.y1,
            self.x2,
            self.y2,
        )

@dataclass(slots=True)
class ImageInfo:
    image_index: Optional[int] = None
    image_id: Optional[int] = None

    width: int = 0
    height: int = 0

    folder: Optional[str] = None
    filename: Optional[str] = None

    path: Optional[Path] = None

@dataclass(slots=True)
class SceneObject:
    """
    Represents one object in the image.
    """

    object_index: int
    label_index: int
    label: str

    bbox: BoundingBox

    confidence: Optional[float] = None

    attributes: list[str] = field(default_factory=list)

@dataclass(slots=True)
class Relationship:
    """
    Represents a directed relationship
    subject ----predicate----> object
    """

    relationship_index: int

    subject: SceneObject
    object: SceneObject

    predicate_index: int
    predicate: str

    confidence: Optional[float] = None

@dataclass(slots=True)
class Scene:
    """
    Complete scene representation for one image.
    """

    image: Image.Image

    info: ImageInfo

    objects: list[SceneObject]

    relationships: list[Relationship]

