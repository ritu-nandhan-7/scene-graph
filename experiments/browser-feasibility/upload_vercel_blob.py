#!/usr/bin/env python3
"""Upload the immutable Scene Graph Explorer browser asset tree to Vercel Blob.

The script deliberately uploads only files listed in scene-graph-v1-asset-manifest.json.
It validates every local byte count and SHA-256 first, never adds pathname suffixes,
and refuses to overwrite an existing object with different content.

Install the official SDK first:
    python -m pip install vercel-blob==0.5.0

Set BLOB_READ_WRITE_TOKEN in this process environment before execution. The script
does not load .env files and never prints, saves, or otherwise exposes the token.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import argparse
import time
from urllib.request import Request, urlopen
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

try:
    from vercel.blob import sync as blob
except ImportError as error:
    raise SystemExit(
        "The official Vercel Blob SDK is required. Install it with:\n"
        "  python -m pip install vercel-blob==0.5.0"
    ) from error


HERE = Path(__file__).resolve().parent
MANIFEST_PATH = HERE / "scene-graph-v1-asset-manifest.json"
SOURCE_ROOT = HERE / "browser-yolo-test" / "public"
BLOB_PREFIX = "scene-graph-v1/"
CACHE_CONTROL_MAX_AGE = 31_536_000
MULTIPART_PATH = "model/d1-fp16-clip-vit-b-32-openai-visual-fp16-io-fp32.onnx"
MULTIPART_PART_BYTES = 8 * 1024 * 1024
HASH_CHUNK_BYTES = 1024 * 1024
REMOTE_READ_ATTEMPTS = 5
REMOTE_RANGE_BYTES = 4 * 1024 * 1024


@dataclass(frozen=True)
class Asset:
    relative_path: str
    bytes: int
    sha256: str
    mime: str
    cache_control: str

    @property
    def source_path(self) -> Path:
        return SOURCE_ROOT.joinpath(*self.relative_path.split("/"))

    @property
    def blob_pathname(self) -> str:
        return f"{BLOB_PREFIX}{self.relative_path}"


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_chunks(chunks: Iterable[bytes]) -> tuple[int, str]:
    digest = hashlib.sha256()
    total = 0
    for chunk in chunks:
        digest.update(chunk)
        total += len(chunk)
    return total, digest.hexdigest()


def hash_remote_blob(url: str, expected_bytes: int) -> tuple[int, str]:
    """Hash a public object through bounded, retried HTTP range requests.

    The Vercel SDK is used for Blob API operations. Public downloads use urllib
    with identity encoding because the SDK's transitive HTTP decoder can be
    incompatible with some local compression-library versions on Windows. Range
    requests avoid a single long-lived download connection being reset in transit.
    """
    digest = hashlib.sha256()
    for start in range(0, expected_bytes, REMOTE_RANGE_BYTES):
        end = min(start + REMOTE_RANGE_BYTES, expected_bytes) - 1
        expected_length = end - start + 1
        for attempt in range(1, REMOTE_READ_ATTEMPTS + 1):
            try:
                request = Request(
                    url,
                    headers={
                        "Accept-Encoding": "identity",
                        "Range": f"bytes={start}-{end}",
                        "User-Agent": "scene-graph-blob-uploader/1.0",
                    },
                )
                with urlopen(request, timeout=120) as response:
                    chunk = response.read()
                if len(chunk) != expected_length:
                    raise ValueError(
                        f"range {start}-{end} returned {len(chunk)} bytes; expected {expected_length}"
                    )
                digest.update(chunk)
                break
            except Exception as error:
                if attempt == REMOTE_READ_ATTEMPTS:
                    raise
                delay_seconds = 2 ** (attempt - 1)
                print(
                    f"  remote range {start}-{end} attempt {attempt}/{REMOTE_READ_ATTEMPTS} failed "
                    f"({type(error).__name__}); retrying in {delay_seconds}s...",
                    file=sys.stderr,
                )
                time.sleep(delay_seconds)
    return expected_bytes, digest.hexdigest()


def read_manifest() -> list[Asset]:
    try:
        raw = json.loads(MANIFEST_PATH.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        fail(f"Could not read manifest {MANIFEST_PATH}: {error}")

    if not isinstance(raw, dict) or not isinstance(raw.get("assets"), list):
        fail("Manifest must be an object containing an assets array.")

    assets: list[Asset] = []
    required = {"path", "bytes", "sha256", "mime", "cache_control"}
    for index, entry in enumerate(raw["assets"]):
        if not isinstance(entry, dict) or required - entry.keys():
            fail(f"Manifest asset {index} is missing one of: {', '.join(sorted(required))}.")
        relative_path = entry["path"]
        byte_count = entry["bytes"]
        digest = entry["sha256"]
        mime = entry["mime"]
        cache_control = entry["cache_control"]
        if (
            not isinstance(relative_path, str)
            or not isinstance(byte_count, int)
            or not isinstance(digest, str)
            or not isinstance(mime, str)
            or not isinstance(cache_control, str)
        ):
            fail(f"Manifest asset {index} has an invalid field type.")
        if relative_path.startswith("/") or "\\" in relative_path or any(part in {"", ".", ".."} for part in relative_path.split("/")):
            fail(f"Manifest asset {index} has unsafe relative path {relative_path!r}.")
        if byte_count < 0 or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest.lower()):
            fail(f"Manifest asset {index} has invalid bytes or sha256.")
        if cache_control != "public, max-age=31536000, immutable":
            fail(f"Manifest asset {relative_path} does not have the required immutable cache policy.")
        assets.append(Asset(relative_path, byte_count, digest.lower(), mime, cache_control))

    if len(assets) != 13:
        fail(f"Expected exactly 13 manifest assets, found {len(assets)}.")
    if len({asset.relative_path for asset in assets}) != len(assets):
        fail("Manifest contains duplicate paths.")
    if MULTIPART_PATH not in {asset.relative_path for asset in assets}:
        fail(f"Manifest is missing required multipart asset {MULTIPART_PATH}.")
    return assets


def validate_local_assets(assets: list[Asset]) -> None:
    print(f"Validating {len(assets)} local manifest assets before any upload...")
    for asset in assets:
        source = asset.source_path
        if not source.is_file():
            fail(f"Missing source asset: {source}")
        actual_bytes = source.stat().st_size
        if actual_bytes != asset.bytes:
            fail(f"Local byte mismatch for {asset.relative_path}: expected {asset.bytes}, got {actual_bytes}.")
        actual_sha256 = sha256_file(source)
        if actual_sha256 != asset.sha256:
            fail(f"Local SHA-256 mismatch for {asset.relative_path}: expected {asset.sha256}, got {actual_sha256}.")
        print(f"  OK  {asset.relative_path} ({asset.bytes:,} bytes)")


def list_remote_blobs() -> dict[str, object]:
    return {item.pathname: item for item in blob.query_blobs(prefix=BLOB_PREFIX, page_size=100)}


def remote_matches_manifest(asset: Asset, remote: object, *, verify_remote_sha256: bool) -> bool:
    remote_size = getattr(remote, "size", None)
    remote_url = getattr(remote, "url", None)
    if remote_size != asset.bytes or not isinstance(remote_url, str):
        return False
    if not verify_remote_sha256:
        return True
    downloaded_bytes, downloaded_sha256 = hash_remote_blob(remote_url, asset.bytes)
    return downloaded_bytes == asset.bytes and downloaded_sha256 == asset.sha256


def stream_file(path: Path, chunk_bytes: int) -> Iterator[bytes]:
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(chunk_bytes), b""):
            yield chunk


def upload_regular(asset: Asset):
    print(f"Uploading {asset.blob_pathname} ({asset.bytes:,} bytes)...")
    return blob.put_blob(
        asset.blob_pathname,
        stream_file(asset.source_path, HASH_CHUNK_BYTES),
        access="public",
        content_length=asset.bytes,
        content_type=asset.mime,
        add_random_suffix=False,
        allow_overwrite=False,
        cache_control_max_age=CACHE_CONTROL_MAX_AGE,
    )


def upload_multipart(asset: Asset):
    print(f"Uploading {asset.blob_pathname} ({asset.bytes:,} bytes) with multipart upload...")
    with blob.multipart_upload_blob(
        asset.blob_pathname,
        access="public",
        content_type=asset.mime,
        add_random_suffix=False,
        allow_overwrite=False,
        cache_control_max_age=CACHE_CONTROL_MAX_AGE,
    ) as upload:
        with asset.source_path.open("rb") as source:
            part_number = 0
            while part := source.read(MULTIPART_PART_BYTES):
                part_number += 1
                upload.upload_part(part, content_length=len(part))
                print(f"  uploaded multipart part {part_number} ({len(part):,} bytes)")
    return upload.result


def verify_remote_asset(asset: Asset, url: str, *, verify_remote_sha256: bool) -> None:
    metadata = blob.head_blob(url)
    if getattr(metadata, "size", None) != asset.bytes:
        fail(f"Remote size mismatch for {asset.blob_pathname}: expected {asset.bytes}, got {getattr(metadata, 'size', None)}.")
    if not verify_remote_sha256:
        return
    remote_bytes, remote_sha256 = hash_remote_blob(url, asset.bytes)
    if remote_bytes != asset.bytes or remote_sha256 != asset.sha256:
        fail(
            f"Remote integrity mismatch for {asset.blob_pathname}: expected "
            f"{asset.bytes} bytes / {asset.sha256}, got {remote_bytes} bytes / {remote_sha256}."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate all local manifest assets without requiring a token or contacting Vercel Blob.",
    )
    parser.add_argument(
        "--metadata-only-remote-verification",
        action="store_true",
        help=(
            "Skip public download SHA-256 checks after local source validation; "
            "verify only Vercel Blob pathname and byte metadata. Use only when "
            "network resets prevent strict remote streaming verification."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    verify_remote_sha256 = not args.metadata_only_remote_verification
    assets = read_manifest()
    validate_local_assets(assets)
    if args.dry_run:
        print("Dry run complete: local manifest validation passed; no Blob API calls were made.")
        return
    if not os.environ.get("BLOB_READ_WRITE_TOKEN"):
        fail("BLOB_READ_WRITE_TOKEN is not set in this process environment.")
    if not verify_remote_sha256:
        print(
            "WARNING: metadata-only remote verification is enabled. Local source SHA-256 "
            "validation remains mandatory, but remote public-content SHA-256 reads are skipped."
        )

    print(f"Listing existing objects under {BLOB_PREFIX!r} for safe resume...")
    remote_by_path = list_remote_blobs()
    final_urls: list[str] = []

    for asset in assets:
        existing = remote_by_path.get(asset.blob_pathname)
        if existing is not None:
            print(f"Existing object found: {asset.blob_pathname}; validating remote bytes and SHA-256...")
            if not remote_matches_manifest(asset, existing, verify_remote_sha256=verify_remote_sha256):
                fail(
                    f"Existing object does not match immutable manifest: {asset.blob_pathname}. "
                    "Refusing to overwrite it. Publish a new versioned prefix instead."
                )
            url = getattr(existing, "url")
            print(f"  SKIP verified existing object: {url}")
            final_urls.append(url)
            continue

        result = upload_multipart(asset) if asset.relative_path == MULTIPART_PATH else upload_regular(asset)
        url = getattr(result, "url", None)
        pathname = getattr(result, "pathname", None)
        if pathname != asset.blob_pathname or not isinstance(url, str):
            fail(f"Unexpected Vercel Blob upload response for {asset.blob_pathname}.")
        print(f"  uploaded: {url}")
        print("  verifying uploaded object bytes and SHA-256...")
        verify_remote_asset(asset, url, verify_remote_sha256=verify_remote_sha256)
        print("  VERIFIED")
        final_urls.append(url)

    if len(final_urls) != len(assets):
        fail("Not all manifest assets produced a final URL.")
    print("\nUploaded/verified public asset URLs:")
    for url in final_urls:
        print(url)

    first_url = final_urls[0]
    marker = f"{BLOB_PREFIX}"
    if marker not in first_url:
        fail("Could not derive the common model asset base URL from the returned Blob URL.")
    base_url = first_url.split(marker, 1)[0] + marker
    print(f"\nFinal common model asset base URL:\n{base_url}")
    print("\nSet this non-secret Vercel build variable before deploying the frontend:")
    print(f"VITE_BROWSER_MODEL_BASE_URL={base_url}")


if __name__ == "__main__":
    main()