"""Manual API test against a live uvicorn server.

Starts the server, sends two /analyze requests with the same image, and
verifies that:
    - Both return 200
    - Both return valid scene graphs
    - The pipeline is loaded only once (check server logs for the
      "Loading SceneGraphPipeline (one-time)..." message appearing once)

Usage:

    # Terminal 1: start the server
    .venv\Scripts\python.exe tests\run_api_server.py

    # Terminal 2: run this test
    .venv\Scripts\python.exe tests\test_api_manual.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import httpx

SERVER_URL = "http://127.0.0.1:8000"
DEFAULT_IMAGE = REPO_ROOT / "datasets" / "visual_genome" / "VG_100K_2" / "1.jpg"


def main() -> int:
    print(f"Test image: {DEFAULT_IMAGE}")
    print(f"Exists: {DEFAULT_IMAGE.exists()}")

    if not DEFAULT_IMAGE.exists():
        print("ERROR: Test image not found")
        return 1

    image_bytes = DEFAULT_IMAGE.read_bytes()

    # Test 1: health
    print("\n=== Test 1: GET /health ===")
    try:
        r = httpx.get(f"{SERVER_URL}/health", timeout=10.0)
        print(f"Status: {r.status_code}")
        print(f"Body: {r.json()}")
        assert r.status_code == 200
        assert r.json() == {"status": "healthy"}
        print("PASS")
    except Exception as e:
        print(f"FAIL: {e}")
        return 1

    # Test 2: analyze (first request - triggers model loading)
    print("\n=== Test 2: POST /analyze (first request) ===")
    try:
        r = httpx.post(
            f"{SERVER_URL}/analyze",
            files={"file": ("1.jpg", image_bytes, "image/jpeg")},
            timeout=120.0,
        )
        print(f"Status: {r.status_code}")
        if r.status_code == 200:
            payload = r.json()
            print(f"Objects: {len(payload['objects'])}")
            print(f"Relationships: {len(payload['relationships'])}")
            for rel in payload["relationships"][:3]:
                print(f"  {rel['subject_id']} -> {rel['predicate']} -> {rel['object_id']}")
        else:
            print(f"Body: {r.text}")
        assert r.status_code == 200
        print("PASS")
    except Exception as e:
        print(f"FAIL: {e}")
        return 1

    # Test 3: analyze (second request - should reuse loaded models)
    print("\n=== Test 3: POST /analyze (second request - model reuse) ===")
    try:
        r = httpx.post(
            f"{SERVER_URL}/analyze",
            files={"file": ("1.jpg", image_bytes, "image/jpeg")},
            timeout=120.0,
        )
        print(f"Status: {r.status_code}")
        if r.status_code == 200:
            payload = r.json()
            print(f"Objects: {len(payload['objects'])}")
            print(f"Relationships: {len(payload['relationships'])}")
        else:
            print(f"Body: {r.text}")
        assert r.status_code == 200
        print("PASS")
    except Exception as e:
        print(f"FAIL: {e}")
        return 1

    print("\n=== ALL MANUAL TESTS PASSED ===")
    print("Check the server logs to confirm 'Loading SceneGraphPipeline (one-time)...' appears only once.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
