"""Summarize completed E1 Chrome output against fresh Python references."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments" / "browser-feasibility" / "e1-browser-pipeline"
PUBLIC = ROOT / "experiments" / "browser-feasibility" / "browser-yolo-test" / "public" / "e1" / "references"


def iou(a: dict, b: dict) -> float:
    def xyxy(row: dict):
        box = row["bbox"]
        return box["cx"] - box["width"] / 2, box["cy"] - box["height"] / 2, box["cx"] + box["width"] / 2, box["cy"] + box["height"] / 2
    ax1, ay1, ax2, ay2 = xyxy(a)
    bx1, by1, bx2, by2 = xyxy(b)
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    union = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1) + max(0.0, bx2 - bx1) * max(0.0, by2 - by1) - intersection
    return intersection / max(union, 1e-30)


def bbox_delta(a: dict, b: dict) -> float:
    return max(abs(a["bbox"][key] - b["bbox"][key]) for key in ("cx", "cy", "width", "height"))


def geometry(subject: dict, obj: dict) -> list[float]:
    def expand(row: dict):
        box = row["bbox"]
        return box["cx"] - box["width"] * 0.6, box["cy"] - box["height"] * 0.6, box["cx"] + box["width"] * 0.6, box["cy"] + box["height"] * 0.6
    sx1, sy1, sx2, sy2 = expand(subject)
    ox1, oy1, ox2, oy2 = expand(obj)
    ux1, uy1, ux2, uy2 = min(sx1, ox1), min(sy1, oy1), max(sx2, ox2), max(sy2, oy2)
    width, height = ux2 - ux1, uy2 - uy1
    sb, ob = subject["bbox"], obj["bbox"]
    return [(sb["cx"] - ux1) / width, (sb["cy"] - uy1) / height, sb["width"] / width, sb["height"] / height, (ob["cx"] - ux1) / width, (ob["cy"] - uy1) / height, ob["width"] / width, ob["height"] / height]


def main() -> None:
    browser = json.loads((OUT / "browser-results.json").read_text(encoding="utf-8"))
    primary = [item for item in browser["runs"] if item["mode"] == "webgpu-preferred"]
    output: dict[str, object] = {"browser": browser["browser"], "page_errors": browser["pageErrors"], "console_messages": browser["consoleMessages"], "images": [], "stability": {}}
    for item in primary:
        result = item["result"]
        reference = json.loads((PUBLIC / f"{item['sample']}.json").read_text(encoding="utf-8"))
        py_objects = reference["python_scene"]["objects"]
        br_objects = result["scene"]["objects"]
        unmatched = set(range(len(py_objects)))
        matches = []
        for browser_object in br_objects:
            candidates = [(iou(browser_object, py_objects[index]), index) for index in unmatched if py_objects[index]["label"] == browser_object["label"]]
            if not candidates:
                continue
            score, index = max(candidates)
            unmatched.remove(index)
            matches.append((browser_object, py_objects[index], score))
        object_map = {br["id"]: py["id"] for br, py, _ in matches}
        python_by_id = {row["id"]: row for row in py_objects}
        browser_by_id = {row["id"]: row for row in br_objects}
        geometry_deltas = []
        for browser_subject, python_subject in object_map.items():
            for browser_object, python_object in object_map.items():
                if browser_subject == browser_object or browser_by_id[browser_subject]["label"] == browser_by_id[browser_object]["label"]:
                    continue
                browser_geometry = geometry(browser_by_id[browser_subject], browser_by_id[browser_object])
                python_geometry = geometry(python_by_id[python_subject], python_by_id[python_object])
                geometry_deltas.extend(abs(a - b) for a, b in zip(browser_geometry, python_geometry))
        py_relationships = {(row["subject_id"], row["predicate"], row["object_id"]): row for row in reference["python_scene"]["relationships"]}
        br_relationships = {}
        unmappable_relationships = 0
        for row in result["scene"]["relationships"]:
            if row["subject_id"] not in object_map or row["object_id"] not in object_map:
                unmappable_relationships += 1
                continue
            key = (object_map[row["subject_id"]], row["predicate"], object_map[row["object_id"]])
            br_relationships[key] = row
        common = set(py_relationships) & set(br_relationships)
        output["images"].append({
            "sample": item["sample"],
            "category": reference["category"],
            "python": {"object_count": len(py_objects), "directed_pair_count": reference["directed_pair_count"], "unique_union_crop_count": reference["unique_union_crop_count"], "relationship_count": len(py_relationships)},
            "browser": {"object_count": result["objectCount"], "directed_pair_count": result["directedPairCount"], "unique_union_crop_count": result["uniqueUnionCropCount"], "relationship_count": len(result["scene"]["relationships"]), "clip_provider": result["actualClipProvider"], "timings_ms": result["timingsMs"]},
            "objects": {"matched_same_label": len(matches), "python_unmatched": len(unmatched), "browser_unmatched": len(br_objects) - len(matches), "minimum_iou": min((score for _, _, score in matches), default=None), "mean_iou": sum(score for _, _, score in matches) / len(matches) if matches else None, "max_bbox_abs_difference": max((bbox_delta(br, py) for br, py, _ in matches), default=None)},
            "geometry": {"matched_directed_pair_count": len(geometry_deltas) // 8, "max_abs_difference": max(geometry_deltas, default=None), "mean_abs_difference": sum(geometry_deltas) / len(geometry_deltas) if geometry_deltas else None},
            "relationships": {"common": len(common), "python_only": len(set(py_relationships) - set(br_relationships)), "browser_only": len(set(br_relationships) - set(py_relationships)), "unmappable_browser_relationships": unmappable_relationships, "max_confidence_difference_common": max((abs(py_relationships[key]["confidence"] - br_relationships[key]["confidence"]) for key in common), default=None)},
        })
    repeat = next(item["result"] for item in browser["runs"] if item["mode"] == "webgpu-repeat")
    first = next(item["result"] for item in primary if item["sample"] == "fixture-1")
    fallback = next(item["result"] for item in browser["runs"] if item["mode"] == "wasm-only-fallback")
    output["stability"] = {
        "webgpu_repeat_fixture_scene_equal": first["scene"] == repeat["scene"],
        "wasm_fallback_fixture_scene_equal_to_webgpu": first["scene"] == fallback["scene"],
        "webgpu_repeat_total_ms": repeat["timingsMs"]["total_warm_analysis"],
        "wasm_fallback_total_ms": fallback["timingsMs"]["total_warm_analysis"],
        "session_initialization_ms": first["sessionInitializationMs"],
        "model_download_bytes": first["modelDownloads"],
        "preprocessing": browser["preprocessing"],
    }
    (OUT / "summary.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    fixture = first["scene"]
    (OUT / "example-browser-scene-graph.json").write_text(json.dumps(fixture, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()