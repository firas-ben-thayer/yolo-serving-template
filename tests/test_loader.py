import tempfile
from pathlib import Path
from app.loader import get_model
from app.core import predict_inproc

def _make_fake_jpeg(path: Path):
    path.write_bytes(b"\xff\xd8" + b"\x00" * 256 + b"\xff\xd9")

def test_get_stub_model_and_infer(tmp_path):
    img = tmp_path / "img.jpg"
    _make_fake_jpeg(img)

    m = get_model(adapter="stub", weights=None)
    out = m.infer(str(img))
    assert isinstance(out, dict)
    assert "detections" in out or "image" in out

def test_predict_inproc_uses_loader(tmp_path):
    img = tmp_path / "img2.jpg"
    _make_fake_jpeg(img)

    out = predict_inproc(str(img), adapter="stub")
    assert isinstance(out, dict)
    # predictable dry-run shape
    assert "detections" in out