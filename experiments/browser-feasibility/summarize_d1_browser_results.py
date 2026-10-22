"""Summarize complete D1 Chrome iterations with median timings per configuration."""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "experiments" / "browser-feasibility" / "d1-browser-runtime" / "browser-results.json"
TARGET = ROOT / "experiments" / "browser-feasibility" / "d1-browser-runtime" / "summary.json"


def median(values: list[float]) -> float:
    return float(statistics.median(values))


def main() -> None:
    payload = json.loads(SOURCE.read_text(encoding="utf-8"))
    result = payload.get("result") or {}
    groups: dict[tuple[str, str, str, int], list[dict]] = defaultdict(list)
    for row in result.get("runs", []):
        groups[(row["provider"], row["format"], row["mode"], int(row["chunkSize"]))].append(row)
    rows = []
    for key, entries in sorted(groups.items()):
        successes = [entry for entry in entries if not entry.get("error")]
        errors = [entry["error"] for entry in entries if entry.get("error")]
        output = {"provider": key[0], "format": key[1], "mode": key[2], "chunk_size": key[3], "attempts": len(entries), "successes": len(successes), "errors": errors}
        if successes:
            for field in ("clipMs", "reconstructionMs", "relationshipMs", "totalMs"):
                output[f"median_{field}"] = median([float(item[field]) for item in successes])
            output["feature_parity"] = successes[-1].get("featureParity", {})
            output["probability_parity"] = successes[-1].get("probabilityParity", {})
            output["class_changes"] = successes[-1].get("classChanges")
            output["retained_changes"] = successes[-1].get("retainedChanges")
            output["max_confidence_difference"] = successes[-1].get("maxConfidenceDifference")
            output["memory"] = successes[-1].get("memory")
        rows.append(output)
    summary = {"browser": payload.get("browser"), "error": payload.get("error"), "page_errors": payload.get("pageErrors"), "console_messages": payload.get("consoleMessages"), "fixture": result.get("fixture"), "browser_details": result.get("browser"), "rows": rows}
    TARGET.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()