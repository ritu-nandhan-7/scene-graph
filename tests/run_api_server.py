"""Start the FastAPI server for manual testing.

Usage:

    .venv\Scripts\python.exe tests\run_api_server.py

Then open http://127.0.0.1:8000/docs in the browser to try the endpoints.

HOST / PORT environment variables override the defaults (Deployment Phase 2):

    HOST=0.0.0.0 PORT=7860 .venv\\Scripts\\python.exe tests\\run_api_server.py
"""

from __future__ import annotations

import os
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
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "8000")),
        reload=False,
        log_level="info",
    )
