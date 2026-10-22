"""Create a compact C1 result summary without duplicating browser raw tensors."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "experiments/browser-feasibility/c1-yoloworld-browser-runtime/browser-results.json"
OUT = ROOT / "experiments/browser-feasibility/c1-yoloworld-browser-runtime/summary.json"


def compact(run: dict) -> dict:
    result = run.get("result")
    if result is None:
        return {"error": run.get("error")}
    diagnostics = result["diagnostics"]
    return {
        "provider": result["provider"],
        "input_source": result["source"],
        "session_initialization_ms": result["sessionInitMs"],
        "inference_ms": result["inferenceMs"],
        "total_image_analysis_ms": result["totalMs"],
        "output_shape": result["outputShape"],
        "detection_count": len(result["detections"]),
        "class_ids": [d["classId"] for d in result["detections"]],
        "labels": [d["label"] for d in result["detections"]],
        "diagnostics": {
            "model_download_bytes": diagnostics["modelDownloadBytes"],
            "preprocess_vs_python": diagnostics["preprocessVsPython"],
            "browser_vs_python_raw": diagnostics["vsPythonRaw"],
            "browser_vs_python_ort_raw": diagnostics["vsPythonOrtRaw"],
            "warm_iterations_ms": diagnostics["warmIterationsMs"],
            "median_warm_ms": diagnostics["medianWarmMs"],
        },
    }


source = json.loads(RESULT.read_text(encoding="utf-8"))
summary = {
    "browser": source["browser"],
    "runtime": source["runtime"],
    "runs": [{"sample": item["sample"], **{key: compact(value) for key, value in item.items() if key != "sample"}} for item in source["runs"]],
    "console_messages": source["consoleMessages"],
    "page_errors": source["pageErrors"],
}
OUT.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(OUT)