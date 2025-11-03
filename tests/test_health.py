from fastapi.testclient import TestClient
from app.main import app, MAX_UPLOAD_SIZE
from pathlib import Path
import os

client = TestClient(app)

MAX_UPLOAD_SIZE_ENV_IMPORT = "not_used"  # placeholder to keep editor happy

def _make_client_with_upload_dir(upload_dir: Path):
    # set env before importing the app so lifespan reads the correct UPLOAD_DIR
    os.environ["UPLOAD_DIR"] = str(upload_dir)
    # import inside function to ensure env is set before app startup
    from fastapi.testclient import TestClient
    from app.main import app, MAX_UPLOAD_SIZE
    return TestClient(app), MAX_UPLOAD_SIZE

def test_health(tmp_path):
    client, _ = _make_client_with_upload_dir(tmp_path)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    
def test_predict_save_and_return_path(tmp_path):
    client, _ = _make_client_with_upload_dir(tmp_path)

    # create a small valid JPEG using Pillow to satisfy server image validation
    from io import BytesIO
    try:
        from PIL import Image
    except Exception:
        # fallback to the previous heuristic if Pillow isn't installed
        img_bytes = b"\xff\xd8" + b"\x00" * 1024 + b"\xff\xd9"
    else:
        buf = BytesIO()
        Image.new("RGB", (10, 10), color=(255, 0, 0)).save(buf, format="JPEG")
        img_bytes = buf.getvalue()

    r = client.post("/predict", files={"file": ("test.jpg", img_bytes, "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert "path" in body
    saved = Path(body["path"])
    assert saved.exists()
    # cleanup
    try:
        saved.unlink()
    except Exception:
        pass

def test_predict_too_large(tmp_path):
    client, MAX_UPLOAD_SIZE = _make_client_with_upload_dir(tmp_path)
    big = b"\xff\xd8" + b"\x00" * (MAX_UPLOAD_SIZE + 10) + b"\xff\xd9"
    r = client.post("/predict", files={"file": ("big.jpg", big, "image/jpeg")})
    assert r.status_code == 413