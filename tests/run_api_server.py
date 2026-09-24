"""Start the FastAPI server for manual testing.

Usage:

    .venv\Scripts\python.exe tests\run_api_server.py

Then open http://127.0..1:8000/docs in the browser to try the endpoints.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Repository root on sys.path.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import uvicorn  # noqa: E402

if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="info",
    )
