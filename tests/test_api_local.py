"""Local API test for the Scene Graph Explorer FastAPI backend.

Tests the two endpoints against the real ML pipeline:

    GET  /health   -> 200 {"status": "healthy"}
    POST /analyze  -> 200 with a clean scene graph JSON

Verifies:
    - HTTP status codes
    - Response structure matches SceneGraphResult.to_dict()
    - Every object has a stable id
    - Every relationship has a stable id
    - Every relationship subject_id / object_id references a real object
    - No relationship returned represents class 50 (no relationship)

Uses FastAPI's TestClient (in-process, no real network) so the test runs
without starting uvicorn.

Run from the repository root:

    .venv\Scripts\python.exe tests\test_api_local.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

# Repository root on sys.path so ``scene_graph`` imports regardless of CWD.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.main import app  # noqa: E402

# Test image (reuses the existing repository image).
DEFAULT_IMAGE = REPO_ROOT / "datasets" / "visual_genome" / "VG_100K_2" / "1.jpg"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _image_bytes(image_path: Path) -> bytes:
    """Read an image file into bytes suitable for upload."""
    return image_path.read_bytes()


def _assert_scene_graph(payload: dict) -> None:
    """Validate the scene graph JSON structure."""
    assert "image" in payload, "Missing 'image' key"
    assert "width" in payload["image"], "Missing image width"
    assert "height" in payload["image"], "Missing image height"

    assert "objects" in payload, "Missing 'objects' key"
    assert isinstance(payload["objects"], list), "'objects' must be a list"
    assert len(payload["objects"]) > 0, "No objects detected"

    # Collect valid object IDs for referential integrity check.
    object_ids = set()
    for obj in payload["objects"]:
        assert "id" in obj, f"Object missing 'id': {obj}"
        assert "label" in obj, f"Object missing 'label': {obj}"
        assert "confidence" in obj, f"Object missing 'confidence': {obj}"
        assert "bbox" in obj, f"Object missing 'bbox': {obj}"
        assert isinstance(obj["confidence"], float), "confidence must be float"
        assert obj["id"].startswith("obj_"), f"Bad object id: {obj['id']}"
        object_ids.add(obj["id"])

    assert "relationships" in payload, "Missing 'relationships' key"
    assert isinstance(payload["relationships"], list), "'relationships' must be a list"

    for rel in payload["relationships"]:
        assert "id" in rel, f"Relationship missing 'id': {rel}"
        assert "subject_id" in rel, f"Relationship missing 'subject_id': {rel}"
        assert "predicate" in rel, f"Relationship missing 'predicate': {rel}"
        assert "object_id" in rel, f"Relationship missing 'object_id': {rel}"
        assert "confidence" in rel, f"Relationship missing 'confidence': {rel}"
        assert isinstance(rel["confidence"], float), "confidence must be float"
        assert rel["id"].startswith("rel_"), f"Bad relationship id: {rel['id']}"

        # No relationship in the final graph should be "no relationship".
        assert rel["predicate"] != "no relationship", (
            f"Final graph contains no-relationship entry: {rel}"
        )

        # Referential integrity: subject and object must exist.
        assert rel["subject_id"] in object_ids, (
            f"subject_id {rel['subject_id']} not in object_ids"
        )
        assert rel["object_id"] in object_ids, (
            f"object_id {rel['object_id']} not in object_ids"
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_health() -> None:
    """GET /health returns 200 + {'status': 'healthy'} without running inference."""
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        assert response.json() == {"status": "healthy"}


def test_analyze_success() -> None:
    """POST /analyze with a real image returns a valid scene graph."""
    assert DEFAULT_IMAGE.exists(), f"Test image not found: {DEFAULT_IMAGE}"

    image_bytes = _image_bytes(DEFAULT_IMAGE)

    with TestClient(app) as client:
        response = client.post(
            "/analyze",
            files={"file": ("1.jpg", image_bytes, "image/jpeg")},
        )

    assert response.status_code == 200, (
        f"Expected 200, got {response.status_code}: {response.text}"
    )
    payload = response.json()
    _assert_scene_graph(payload)

    # Print a concise summary (not the full JSON).
    print("\nFASTAPI LOCAL TEST")
    print(f"Health: PASS")
    print(f"Analyze: PASS")
    print(f"\nObjects: {len(payload['objects'])}")
    print(f"Relationships: {len(payload['relationships'])}")

    if payload["relationships"]:
        print("\nSample relationships:")
        for rel in payload["relationships"][:5]:
            subject_label = next(
                o["label"] for o in payload["objects"] if o["id"] == rel["subject_id"]
            )
            object_label = next(
                o["label"] for o in payload["objects"] if o["id"] == rel["object_id"]
            )
            print(
                f"  {subject_label} -> {rel['predicate']} -> {object_label} "
                f"({rel['confidence']:.2%})"
            )


def test_analyze_missing_file() -> None:
    """POST /analyze with no file returns 422 (FastAPI validation error)."""
    with TestClient(app) as client:
        response = client.post("/analyze")
    assert response.status_code == 422, f"Expected 422, got {response.status_code}"


def test_analyze_invalid_image() -> None:
    """POST /analyze with non-image bytes returns 400."""
    with TestClient(app) as client:
        response = client.post(
            "/analyze",
            files={"file": ("not_image.txt", b"this is not an image", "image/jpeg")},
        )
    assert response.status_code == 400, f"Expected 400, got {response.status_code}"
    assert "Invalid image" in response.json()["detail"]


def test_analyze_unsupported_type() -> None:
    """POST /analyze with an unsupported content type returns 400."""
    with TestClient(app) as client:
        response = client.post(
            "/analyze",
            files={"file": ("file.gif", b"GIF89a", "image/gif")},
        )
    assert response.status_code == 400, f"Expected 400, got {response.status_code}"
    assert "Unsupported" in response.json()["detail"]


# ---------------------------------------------------------------------------
# Main (for direct execution)
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    print(f"Test image: {DEFAULT_IMAGE}")
    print(f"Exists: {DEFAULT_IMAGE.exists()}")

    # Run each test in sequence and report.
    failures = []

    test_fns = [
        test_health,
        test_analyze_success,
        test_analyze_missing_file,
        test_analyze_invalid_image,
        test_analyze_unsupported_type,
    ]

    for test_fn in test_fns:
        try:
            test_fn()
            print(f"  PASS: {test_fn.__name__}")
        except AssertionError as e:
            failures.append((test_fn.__name__, str(e)))
            print(f"  FAIL: {test_fn.__name__}: {e}")
        except Exception as e:
            failures.append((test_fn.__name__, f"ERROR: {e}"))
            print(f"  ERROR: {test_fn.__name__}: {e}")

    print()
    if failures:
        print(f"RESULT: {len(failures)} FAILED")
        for name, msg in failures:
            print(f"  - {name}: {msg}")
        sys.exit(1)
    else:
        print("RESULT: ALL PASSED")
        sys.exit(0)
