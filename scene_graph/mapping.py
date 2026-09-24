"""
Label / index mapping helpers for the Scene Graph inference pipeline.

The repository uses several distinct integer systems that must never be
mixed.  This module is the single place where conversions happen.

Systems
-------
Visual Genome object label id (HDF5 ``labels`` / ``label_to_idx``)
    1..150                  (e.g. person = 95, bike = 14, car = 30)
Model object embedding id (``nn.Embedding`` index)
    0..149                  = VG label id - 1
Visual Genome predicate id (HDF5 ``predicates`` / ``predicate_to_idx``)
    1..50                   (e.g. above = 1, ..., worn by = 50)
Relationship model output class
    0..50
        classes 0..49   -> VG predicate id - 1   (class 0 = "above", ...)
        class 50        -> "no relationship"

IMPORTANT - "no relationship" is model class 50, NOT class 0
-----------------------------------------------------------
The task brief and ``scene_graph_relationship_training_workflow.md``
describe class 0 as "no relationship".  That does NOT match the trained
model in this repository.  Verified facts:

* The cache shards store negative (no-relationship) samples with the
  internal predicate id 51, i.e. target = 51 - 1 = 50
  (``notebook/models/preprocessed_cache/val/val_0000.pt``: ~50% of all
  targets are class 50).
* ``notebook/05_testing.ipynb`` cell 57 builds the decode table as
  ``{int(idx) - 1: name for name, idx in predicate_to_idx}`` and then sets
  ``idx_to_predicate[50] = "no relationship"``; cell 61 filters output
  class 50 into the "no relationship" bucket.
* The working end-to-end output shows the model predicting class 50
  ("person -> no relationship -> car").

The pipeline therefore decodes model class 50 as "no relationship" and
classes 0..49 as Visual Genome predicates 1..50.  Changing this would
make every non-relationship pair decode as a real predicate.

Numerical coincidence to keep in mind:
    model class 50 == "no relationship"
    VG predicate id 50 == "worn by"  (which maps to model class 49)
The two share the number 50 but are different things.
"""

from __future__ import annotations

from typing import Dict, List

#: Model output class reserved for "no relationship" (verified, see above).
NO_RELATIONSHIP_CLASS = 50

#: Decoded string used for the "no relationship" class.
NO_RELATIONSHIP_PREDICATE = "no relationship"

#: Number of object classes in the Visual Genome vocabulary.
NUM_OBJECT_CLASSES = 150

#: Number of relationship model output classes (predicates + no relationship).
NUM_MODEL_CLASSES = 51


def vg_label_id_to_model_id(vg_label_id: int) -> int:
    """Convert a 1-based Visual Genome object label id to a 0-based embedding id.

    Example: person (VG 95) -> 94.
    """
    return int(vg_label_id) - 1


def model_id_to_vg_label_id(model_id: int) -> int:
    """Convert a 0-based embedding id back to a 1-based VG label id."""
    return int(model_id) + 1


def model_class_to_vg_predicate_id(model_class: int) -> int:
    """Convert a model output class to a 1-based VG predicate id.

    Only valid for classes 0..49 (real predicates).  Class 50 is
    "no relationship" and has no VG predicate id.
    """
    return int(model_class) + 1


def build_idx_to_predicate(vg_dict: Dict) -> Dict[int, str]:
    """Build the model output class (0..50) -> predicate name table.

    ``vg_dict`` is the parsed ``VG-SGG-dicts.json``.
    """
    idx_to_predicate = {
        int(idx) - 1: name
        for name, idx in vg_dict["predicate_to_idx"].items()
    }
    idx_to_predicate[NO_RELATIONSHIP_CLASS] = NO_RELATIONSHIP_PREDICATE
    return idx_to_predicate


def build_label_name_to_vg_id(vg_dict: Dict) -> Dict[str, int]:
    """Build label name -> 1-based Visual Genome object label id."""
    return {name: int(idx) for name, idx in vg_dict["label_to_idx"].items()}


def build_yolo_vocabulary(vg_dict: Dict) -> List[str]:
    """Return the 149-name YOLO-World vocabulary.

    It is the Visual Genome label list minus ``"background"`` (exactly
    what notebooks 02 / 05 pass to ``set_classes``).
    """
    return [name for name in vg_dict["label_to_idx"] if name != "background"]