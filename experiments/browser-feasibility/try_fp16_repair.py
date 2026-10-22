"""Attempt a valid FP16 CLIP export with ONNX Runtime transformers plus graph repair.

The default `convert_float_to_float16` call in run_d1_optimization_study.py produced a
graph that fails ONNX topological validation ("input 'graph_input_cast_0' ... is not
output of any previous nodes"). This helper retries with a few documented variants and,
if needed, re-sorts the graph into a valid topological order before saving.

Writes only to the isolated D1 experiment model directory. No production artifact is touched.
"""
from __future__ import annotations

import json
from pathlib import Path

import onnx

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "experiments" / "browser-feasibility" / "b2-b3-export-parity" / "model" / "clip-vit-b-32-openai-visual-fp32-dynamic.onnx"
TARGET = ROOT / "experiments" / "browser-feasibility" / "d1-optimization" / "model" / "clip-vit-b-32-openai-visual-fp16-io-fp32.onnx"
REPORT = ROOT / "experiments" / "browser-feasibility" / "d1-optimization" / "fp16-repair.json"


def topological_sort(model: onnx.ModelProto) -> onnx.ModelProto:
    """Reorder graph nodes so every node consumes only earlier-produced values."""
    graph = model.graph
    available: set[str] = set()
    for tensor in graph.input:
        available.add(tensor.name)
    for tensor in graph.initializer:
        available.add(tensor.name)
    remaining = list(graph.node)
    ordered: list[onnx.NodeProto] = []
    while remaining:
        progressed = False
        deferred: list[onnx.NodeProto] = []
        for node in remaining:
            inputs = [name for name in node.input if name]
            if all(name in available for name in inputs):
                ordered.append(node)
                available.update(node.output)
                progressed = True
            else:
                deferred.append(node)
        remaining = deferred
        if not progressed:
            # Remaining nodes consume values produced by nodes inside a cycle-free subgraph
            # that onnx treats as initializers; append them in original order.
            ordered.extend(remaining)
            break
    del graph.node[:]
    graph.node.extend(ordered)
    return model


def attempt(label: str, converter) -> dict[str, object]:
    try:
        model = onnx.load(str(SOURCE))
        converted = converter(model)
        onnx.save(converted, str(TARGET))
        onnx.checker.check_model(str(TARGET))
        return {"variant": label, "status": "created", "bytes": TARGET.stat().st_size}
    except Exception as exc:  # noqa: BLE001 - record every failure mode for the report
        return {"variant": label, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    from onnxruntime.transformers.float16 import convert_float_to_float16

    variants = [
        ("default", lambda model: convert_float_to_float16(model, keep_io_types=True)),
        ("no_shape_infer", lambda model: convert_float_to_float16(model, keep_io_types=True, disable_shape_infer=True)),
        ("default_plus_topo_sort", lambda model: topological_sort(convert_float_to_float16(model, keep_io_types=True))),
        ("no_shape_infer_plus_topo_sort", lambda model: topological_sort(convert_float_to_float16(model, keep_io_types=True, disable_shape_infer=True))),
    ]
    results = [attempt(label, converter) for label, converter in variants]
    for entry in results:
        if entry["status"] == "created":
            break
    REPORT.write_text(json.dumps({"source": str(SOURCE.relative_to(ROOT)).replace("\\", "/"), "attempts": results}, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
