"""Print a compact comparison table for the D1 Chrome benchmark results."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "experiments" / "browser-feasibility" / "d1-browser-runtime" / "summary.json"

HEADER = (
    f"{'provider':9} {'fmt':5} {'mode':12} {'chunk':>5} {'ok':>5} "
    f"{'clipMs':>9} {'recon':>7} {'rel':>6} {'totalMs':>9} "
    f"{'cls':>4} {'ret':>4} {'maxConf':>8} {'featCos':>9} {'probCos':>9}"
)


def main() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    print(HEADER)
    print("-" * len(HEADER))
    for row in summary["rows"]:
        feature = row.get("feature_parity", {})
        probability = row.get("probability_parity", {})
        print(
            f"{row['provider']:9} {row['format']:5} {row['mode']:12} {row['chunk_size']:>5} "
            f"{str(row['successes']) + '/' + str(row['attempts']):>5} "
            f"{row.get('median_clipMs', 0):>9.1f} {row.get('median_reconstructionMs', 0):>7.2f} "
            f"{row.get('median_relationshipMs', 0):>6.2f} {row.get('median_totalMs', 0):>9.1f} "
            f"{str(row.get('class_changes')):>4} {str(row.get('retained_changes')):>4} "
            f"{(row.get('max_confidence_difference') or 0):>8.5f} "
            f"{(feature.get('cosineSimilarity') or 0):>9.6f} "
            f"{(probability.get('cosineSimilarity') or 0):>9.6f}"
        )


if __name__ == "__main__":
    main()
