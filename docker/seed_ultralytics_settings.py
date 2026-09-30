"""Seed ultralytics settings with an ABSOLUTE weights_dir (build time).

Phase 4 finding (Diary Entry 15): on a fresh machine ultralytics' default
``weights_dir`` is the RELATIVE string "weights", resolved against the process
CWD, and the settings file itself falls back to /tmp/Ultralytics when
$HOME/.config is missing. Both are non-deterministic for a container, and the
YOLO-World text tower loads its CLIP weights from
``WEIGHTS_DIR / "clip" / "ViT-B-32.pt"``.

This script pins that path to /opt/sg/weights (where the fetch script placed the
file) and disables settings sync (telemetry) so the runtime needs no network.
It uses the library's own persistence API (SettingsManager.update -> _save).

Environment:
    SG_WEIGHTS_DIR    absolute path to pin (default /opt/sg/weights)
    YOLO_CONFIG_DIR   settings root (set by the Dockerfile; the settings file
                      lands at $YOLO_CONFIG_DIR/Ultralytics/settings.json)
Exit codes: 0 = settings persisted and CLIP file present, 1 = failure.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> int:
    from ultralytics.utils import SETTINGS, USER_CONFIG_DIR

    weights_dir = os.environ.get("SG_WEIGHTS_DIR", "/opt/sg/weights")

    SETTINGS.update({"weights_dir": weights_dir, "sync": False})

    settings_file = Path(USER_CONFIG_DIR) / "settings.json"
    print(f"[settings] config dir : {USER_CONFIG_DIR}")
    print(f"[settings] file       : {settings_file}")
    print(f"[settings] weights_dir: {SETTINGS['weights_dir']}")
    print(f"[settings] sync       : {SETTINGS['sync']}")

    clip_file = Path(weights_dir) / "clip" / "ViT-B-32.pt"
    if not clip_file.is_file():
        print(f"[settings] expected CLIP file missing: {clip_file}", file=sys.stderr)
        return 1
    if not Path(SETTINGS["weights_dir"]).is_absolute():
        print("[settings] weights_dir is not absolute", file=sys.stderr)
        return 1

    print(f"[settings] OK: CLIP text tower baked at {clip_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
