import asyncio
from pathlib import Path
import httpx
import builtins
from unittest.mock import patch

import pytest

from client.http import YoloHTTPClient


class DummyResponse:
    def __init__(self, status_code=200, json_data=None):
        self.status_code = status_code
        self._json = json_data or {}
        # provide a minimal request object for HTTPStatusError
        self.request = httpx.Request("POST", "http://example.local")

    def json(self):
        return self._json
    
    def raise_for_status(self):
        if 400 <= self.status_code:
            raise httpx.HTTPStatusError(f"{self.status_code} Error", request=self.request, response=self)


def _make_temp_file(tmp_path: Path) -> str:
    p = tmp_path / "img.jpg"
    p.write_bytes(b"\xff\xd8" + b"\x00" * 128 + b"\xff\xd9")
    return str(p)


def test_sync_retries_and_success(tmp_path):
    img_path = _make_temp_file(tmp_path)

    # side effects: two network errors, then success
    calls = {"count": 0}

    def mock_post(self, url, files=None, params=None):
        calls["count"] += 1
        if calls["count"] <= 2:
            raise httpx.RequestError("simulated network error")
        return DummyResponse(200, {"ok": True})

    client = YoloHTTPClient("http://example.local", retries=4, backoff_factor=0.01)

    with patch.object(httpx.Client, "post", new=mock_post):
        out = client.predict(img_path)
    assert out == {"ok": True}
    assert calls["count"] == 3


def test_sync_retries_exhausted(tmp_path):
    img_path = _make_temp_file(tmp_path)

    def always_fail(self, url, files=None, params=None):
        raise httpx.RequestError("always fail")

    client = YoloHTTPClient("http://example.local", retries=3, backoff_factor=0.01)
    with patch.object(httpx.Client, "post", new=always_fail):
        with pytest.raises(httpx.RequestError):
            client.predict(img_path)


@pytest.mark.asyncio
async def test_async_retries_and_success(tmp_path):
    img_path = _make_temp_file(tmp_path)

    calls = {"count": 0}

    async def mock_post(self, url, files=None, params=None):
        calls["count"] += 1
        if calls["count"] <= 2:
            raise httpx.RequestError("simulated async network error")
        return DummyResponse(200, {"ok": True})

    client = YoloHTTPClient("http://example.local", retries=4, backoff_factor=0.01)

    with patch.object(httpx.AsyncClient, "post", new=mock_post):
        out = await client.predict_async(img_path)

    assert out == {"ok": True}
    assert calls["count"] == 3


def test_retry_on_status_code_then_success(tmp_path):
    img_path = _make_temp_file(tmp_path)
    calls = {"count": 0}

    def mock_post(self, url, files=None, params=None):
        calls["count"] += 1
        if calls["count"] <= 2:
            # simulate a 503 response object
            return DummyResponse(503, {"error": "service unavailable"})
        return DummyResponse(200, {"ok": True})

    client = YoloHTTPClient("http://example.local", retries=4, backoff_factor=0.01)
    with patch.object(httpx.Client, "post", new=mock_post):
        out = client.predict(img_path)

    assert out == {"ok": True}
    assert calls["count"] == 3
