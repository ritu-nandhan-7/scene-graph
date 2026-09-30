"""Prefetch the open_clip ViT-B-32 "openai" visual encoder (PATH B).

BUILD-TIME ONLY (Docker image build). Uses the SAME call the runtime uses
(``open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")``),
so the Hugging Face cache entry produced here is byte-identical to what the
offline runtime resolves:

    $HF_HOME/hub/models--timm--vit_base_patch32_clip_224.openai/
        blobs/<sha256>
        refs/main                       (revision resolved at build time)
        snapshots/<revision>/open_clip_model.safetensors

No file is downloaded at runtime (the backend keeps HF_HUB_OFFLINE=1).

Environment:
    HF_HOME   cache home (default /opt/sg/hf)
Exit codes: 0 = checkpoint cached and model built, 1 = failure.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_DIR_NAME = "models--timm--vit_base_patch32_clip_224.openai"
MIN_BYTES = 500 * 1024 * 1024  # sanity: the real checkpoint is ~577 MB


def main() -> int:
    hf_home = os.environ.get("HF_HOME", "/opt/sg/hf")
    os.environ["HF_HOME"] = hf_home

    import open_clip  # imported after HF_HOME is fixed

    print(f"[PATH B] HF_HOME={hf_home}; prefetching ViT-B-32/openai")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )

    repo_dir = Path(hf_home) / "hub" / REPO_DIR_NAME
    if not repo_dir.is_dir():
        print(f"[PATH B] expected cache dir missing: {repo_dir}", file=sys.stderr)
        return 1

    files = [p for p in repo_dir.rglob("*") if p.is_file()]
    total = sum(p.stat().st_size for p in files)
    for path in sorted(files):
        print(f"[PATH B]   {path.relative_to(repo_dir)}  {path.stat().st_size} bytes")

    if total < MIN_BYTES:
        print(f"[PATH B] cache looks too small ({total} bytes)", file=sys.stderr)
        return 1

    params = sum(p.numel() for p in model.parameters())
    print(
        f"[PATH B] OK: {params/1e6:.1f}M parameters, {total} bytes cached "
        f"(preprocess size={preprocess.transforms[-1].size if hasattr(preprocess.transforms[-1], 'size') else 'n/a'})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
