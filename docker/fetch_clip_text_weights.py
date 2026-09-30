"""Download the official OpenAI CLIP ViT-B/32 text-tower weights (PATH A).

BUILD-TIME ONLY (Docker image build). Produces exactly the file that
``clip.load("ViT-B/32", download_root=WEIGHTS_DIR / "clip")`` looks for, so the
running container never contacts the OpenAI CDN.

Source URL and SHA-256 are the pair hard-coded in the official ``clip`` package
(clip/clip.py ``_MODELS["ViT-B/32"]``); the loader verifies this same checksum
at startup, and this script verifies it again before baking the file.

Environment:
    SG_WEIGHTS_DIR   target weights directory (default /opt/sg/weights)
Exit codes: 0 = file present and checksum-correct, 1 = failure.
"""

from __future__ import annotations

import hashlib
import os
import sys
import time
import urllib.request
from pathlib import Path

URL = (
    "https://openaipublic.azureedge.net/clip/models/"
    "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt"
)
SHA256 = "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"
FILENAME = "ViT-B-32.pt"
ATTEMPTS = 3
CHUNK = 1 << 20


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def target_path() -> Path:
    weights_dir = os.environ.get("SG_WEIGHTS_DIR", "/opt/sg/weights")
    return Path(weights_dir) / "clip" / FILENAME


def main() -> int:
    target = target_path()

    if target.is_file() and sha256_of(target) == SHA256:
        print(f"[PATH A] already present and valid: {target}")
        return 0

    target.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None

    for attempt in range(1, ATTEMPTS + 1):
        try:
            print(f"[PATH A] downloading {URL} (attempt {attempt}/{ATTEMPTS})")
            with urllib.request.urlopen(URL, timeout=120) as source, open(
                target, "wb"
            ) as output:
                while True:
                    block = source.read(CHUNK)
                    if not block:
                        break
                    output.write(block)
            actual = sha256_of(target)
            if actual != SHA256:
                raise RuntimeError(
                    f"checksum mismatch: expected {SHA256}, got {actual}"
                )
            print(f"[PATH A] OK {target} ({target.stat().st_size} bytes, sha256 verified)")
            return 0
        except Exception as error:  # noqa: BLE001 - build-time diagnostics
            last_error = error
            print(f"[PATH A] attempt {attempt} failed: {error!r}")
            target.unlink(missing_ok=True)
            if attempt < ATTEMPTS:
                time.sleep(5 * attempt)

    print(f"[PATH A] FAILED after {ATTEMPTS} attempts: {last_error!r}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
