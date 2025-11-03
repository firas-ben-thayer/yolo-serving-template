## YOLO Serving Template v0 (YOLO-focused)

This repository is a minimal, opinionated FastAPI template for serving YOLO-style models.
It's intended as a small, well-tested starting point for building a YOLO-specific inference service. This is version 0 the goal is to evolve the service to support YOLO tasks such as ROI-aware uploads, instance segmentation, and richer model metadata.

Key ideas
- Provide a safe default `StubModel` so the service always starts.
- Support a pluggable adapter architecture (e.g. `YolovXAdapter`) so real backends like Ultralytics YOLO can be attached.
- Keep the HTTP surface small: `/health` and `/predict` for now.
- Include a thin Python client + CLI for convenience and retries/backoff.

Status
- Version: v0 basic YOLO-focused serving template.
- Tests: unit tests included under `tests/` (use `pytest`).

Quickstart (developer)
1. Create and activate a virtual environment (Windows PowerShell):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Configure `.env` in the project root (an example is provided in the repository). Minimal variables:

```
MODEL_ADAPTER=yolov12n
MODEL_WEIGHTS=yolo12n.pt
UPLOAD_DIR=uploads
```

Notes:
- The `MODEL_WEIGHTS` path may be relative to the project root; the loader normalizes it for you.
- `.env` is already ignored by `.gitignore` so it is safe to store local config there.

4. Start the server (recommended while the venv is active so the correct Python environment is used):

```powershell
# start with the venv's python so the reloader uses the same environment
python -m uvicorn app.main:app --reload
```

HTTP endpoints
- `GET /health` basic health and model metadata. Example response:

```
{"status":"ok","model":{"adapter":"YolovXAdapter","mode":"inference","version":"8.3.223"}}
```

  - `version` is the installed `ultralytics` package version (if the `YolovXAdapter` is running). If you see `8.3.223` (or another number) it means the running Python environment has that `ultralytics` package installed.

- `POST /predict` multipart/form-data upload. Field name: `file`.

  - The endpoint saves the uploaded image, validates basic constraints, and returns a stable JSON schema with `image`, `width`, `height`, `detections`, and `model` metadata.
  - Current behavior: returns empty detections in dry-run/no-weights mode; will return real detections when `ultralytics` is installed and `MODEL_WEIGHTS` point to valid weights.

Client & CLI
- A small Python client is available in the `client` package. It includes:
  - `YoloHTTPClient` sync + async HTTP client with configurable retries and exponential backoff.
  - `client/cli.py` CLI wrapper so you can run:

```powershell
python -m client.cli predict path\to\image.jpg --url http://127.0.0.1:8000
```

Testing
- Run the unit tests with `pytest`:

```powershell
pytest -q
```

Configuration and debugging tips
- The server reports the installed `ultralytics` package version as `model.version` in `/health`. If this differs from what you expect:
  - Verify you started uvicorn in the same venv. Start using `python -m uvicorn ...` from the activated venv.
  - Run `python -c "import ultralytics as ul; print(getattr(ul, '__version__', 'unknown'))"` in the same shell to confirm the package version.
- To expose which weights file was actually normalized/loaded, the next version will include `weights` metadata in `/health` (optional change you can enable locally by updating `app/loader.py` and `app/main.py`).

Project layout (important files)
- `app/main.py` FastAPI application and endpoints.
- `app/loader.py` model factory and `YolovXAdapter` adapter.
- `app/model.py` `BaseModel` and `StubModel`.
- `client/` small client package and CLI.
- `tests/` unit tests and fixtures.

Goals & next steps (roadmap)
- Make the service YOLO-specific: support ROI-aware uploads and segmentation masks as part of the `/predict` payload.
- Add a structured `app/settings.py` using `pydantic.BaseSettings` for validated configuration.
- Harden the `YolovXAdapter` postprocessing (label mappings, confidence thresholds, multi-scale parsing) and add tests using a small sample model.
- Add Dockerfile improvements and CI pipeline for automated tests and releases.
- Add more endpoints for batch predictions, model reloading, and per-request options (thresholds, classes, ROI coordinates).

Enjoy this is v0 and intentionally small so it's easy to extend toward ROI / segmentation-first use cases.
