"""Test connection between frontend and backend."""
import httpx
import sys

# Test 1: Frontend is serving HTML
print("=== Test 1: Frontend serves HTML ===")
try:
    r = httpx.get("http://127.0.0.1:5173/", timeout=10.0)
    print(f"Status: {r.status_code}")
    if "root" in r.text:
        print("PASS")
    else:
        print("FAIL: Unexpected content")
        sys.exit(1)
except Exception as e:
    print(f"FAIL: {e}")
    sys.exit(1)

# Test 2: CORS allows frontend origin
print("\n=== Test 2: CORS allows frontend origin ===")
try:
    r = httpx.options(
        "http://127.0.0.1:8000/analyze",
        headers={"Origin": "http://127.0.0.1:5173"},
        timeout=10.0,
    )
    cors_header = r.headers.get("access-control-allow-origin", "")
    print(f"Access-Control-Allow-Origin: {cors_header}")
    if "5173" in cors_header:
        print("PASS")
    else:
        print("FAIL: CORS header missing or incorrect")
        sys.exit(1)
except Exception as e:
    print(f"FAIL: {e}")
    sys.exit(1)

# Test 3: Analyze endpoint with file upload (simulating frontend)
print("\n=== Test 3: Analyze endpoint with file upload ===")
image_path = r"C:\Users\psytr\Desktop\Projects\scene_graph\datasets\visual_genome\VG_100K_2\1.jpg"
try:
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    r = httpx.post(
        "http://127.0.0.1:8000/analyze",
        files={"file": ("1.jpg", image_bytes, "image/jpeg")},
        timeout=120.0,
    )
    print(f"Status: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        print(f"Objects: {len(data['objects'])}")
        print(f"Relationships: {len(data['relationships'])}")
        print("PASS")
    else:
        print(f"FAIL: {r.text[:200]}")
        sys.exit(1)
except Exception as e:
    print(f"FAIL: {e}")
    sys.exit(1)

print("\n=== ALL CONNECTION TESTS PASSED ===")
