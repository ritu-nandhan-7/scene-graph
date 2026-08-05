from PIL import Image

from datasets.visual_genome import BoundingBox

def bbox_to_xyxy(box: BoundingBox):

    x1 = box.cx - box.width / 2
    y1 = box.cy - box.height / 2

    x2 = box.cx + box.width / 2
    y2 = box.cy + box.height / 2

    return x1, y1, x2, y2

def expand_bbox(
    bbox: BoundingBox,
    expansion: float = 0.20,
):

    return BoundingBox(
        cx=bbox.cx,
        cy=bbox.cy,
        width=bbox.width * (1 + expansion),
        height=bbox.height * (1 + expansion),
    )
    
def boxes_intersect(
    box1: BoundingBox,
    box2: BoundingBox,
):

    x1_1, y1_1, x2_1, y2_1 = bbox_to_xyxy(box1)
    x1_2, y1_2, x2_2, y2_2 = bbox_to_xyxy(box2)

    return not (
        x2_1 < x1_2
        or x2_2 < x1_1
        or y2_1 < y1_2
        or y2_2 < y1_1
    )
    
def get_union_bbox(
    box1: BoundingBox,
    box2: BoundingBox,
    expansion: float = 0.20,
):

    box1 = expand_bbox(box1, expansion)
    box2 = expand_bbox(box2, expansion)

    x1_1, y1_1, x2_1, y2_1 = bbox_to_xyxy(box1)
    x1_2, y1_2, x2_2, y2_2 = bbox_to_xyxy(box2)

    left = min(x1_1, x1_2)
    top = min(y1_1, y1_2)

    right = max(x2_1, x2_2)
    bottom = max(y2_1, y2_2)

    return BoundingBox(
        cx=(left + right) / 2,
        cy=(top + bottom) / 2,
        width=right - left,
        height=bottom - top,
    )
    
def crop_union_region(
    image: Image.Image,
    union_box: BoundingBox,
):

    x1, y1, x2, y2 = bbox_to_xyxy(union_box)

    width, height = image.size

    x1 = max(0, int(round(x1)))
    y1 = max(0, int(round(y1)))

    x2 = min(width, int(round(x2)))
    y2 = min(height, int(round(y2)))

    return image.crop((x1, y1, x2, y2))

def bbox_relative_to_crop(
    bbox: BoundingBox,
    crop_box: BoundingBox,
):

    crop_x1, crop_y1, _, _ = bbox_to_xyxy(crop_box)

    x1, y1, x2, y2 = bbox_to_xyxy(bbox)

    x1 -= crop_x1
    y1 -= crop_y1

    x2 -= crop_x1
    y2 -= crop_y1

    return BoundingBox(
        cx=(x1 + x2) / 2,
        cy=(y1 + y2) / 2,
        width=x2 - x1,
        height=y2 - y1,
    )
    
def normalize_bbox(
    bbox: BoundingBox,
    crop_box: BoundingBox,
):

    return BoundingBox(
        cx=bbox.cx / crop_box.width,
        cy=bbox.cy / crop_box.height,
        width=bbox.width / crop_box.width,
        height=bbox.height / crop_box.height,
    )
    
