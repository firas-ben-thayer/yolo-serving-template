# FastAPI app (health + /predict placeholder)
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
import uuid
from contextlib import asynccontextmanager
from .loader import get_model
from .model import BaseModel
import logging
import asyncio
import os
from dotenv import load_dotenv

# load .env from project root so settings in .env are available to the app
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

logger = logging.getLogger(__name__)

MAX_UPLOAD_SIZE = 5 * 1024 * 1024  # 5 MiB
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}

MODEL = None
UPLOAD_DIR: Path = Path("uploads")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL, UPLOAD_DIR
    # allow selecting adapter and weights via environment
    adapter = os.getenv("MODEL_ADAPTER", "stub")
    weights = os.getenv("MODEL_WEIGHTS", None)
    logger.info("Starting up: adapter=%s weights=%s", adapter, weights)
    MODEL = get_model(adapter=adapter, weights=weights)
    logger.info("Startup complete: model_mode=%s", getattr(MODEL, "mode", None))

    # configure uploads directory from env (so tests can override via env before TestClient)
    upload_env = os.getenv("UPLOAD_DIR", None)
    if upload_env:
        UPLOAD_DIR = Path(upload_env)
    else:
        UPLOAD_DIR = Path("uploads")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    yield

    # shutdown / cleanup
    MODEL = None
    # Note: we don't remove uploaded files on shutdown
    
app = FastAPI(lifespan=lifespan)

@app.get("/health")
async def health():
    # include model metadata to help debugging startup
    meta = None
    try:
        meta = {
            "adapter": getattr(MODEL, "__class__", type(None)).__name__ if MODEL is not None else None,
            "mode": getattr(MODEL, "mode", None) if MODEL is not None else None,
            "version": getattr(MODEL, "version", None) if MODEL is not None else None,
        }
    except Exception:
        meta = None
    return {"status": "ok", "model": meta}

def _secure_filename(filename: str) -> str:
    # simple sanitization: keep only basename
    return Path(filename).name

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # validate content type
    content_type = (file.content_type or "").lower()
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail=f"unsupported content type: {content_type}")

    out_dir = UPLOAD_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = _secure_filename(file.filename or "upload")
    file_id = f"{uuid.uuid4().hex}_{safe_name}"
    path = out_dir / file_id

    # stream-write to disk and enforce size limit without buffering whole file in memory
    uploaded = 0
    try:
        with path.open("wb") as f:
            chunk_size = 1024 * 1024
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                uploaded += len(chunk)
                if uploaded > MAX_UPLOAD_SIZE:
                    # remove partial file and abort
                    f.close()
                    try:
                        path.unlink()
                    except FileNotFoundError:
                        pass
                    raise HTTPException(status_code=413, detail="file too large")
                f.write(chunk)

        # optional image validation if Pillow is installed
        try:
            from PIL import Image
        except Exception:
            Image = None

        if Image is not None:
            try:
                with Image.open(path) as img:
                    img.verify()  # will raise if file is not a valid image
            except Exception:
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
                raise HTTPException(status_code=400, detail="invalid image file")
    finally:
        await file.close()

    if MODEL is None:
        return JSONResponse({"message": "model not loaded (dry-run)", "path": str(path)})

    try:
        # run blocking inference off the event loop if needed
        result = await asyncio.to_thread(MODEL.infer, str(path))
        return JSONResponse({"result": result, "path": str(path)})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))