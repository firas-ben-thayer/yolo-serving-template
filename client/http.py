import mimetypes
from pathlib import Path
from typing import Any, Dict, Optional, Set
import time
import asyncio

import httpx

DEFAULT_TIMEOUT = 30.0


class YoloHTTPClient:
    """
    Thin httpx-based client for the /predict endpoint with retry/backoff.

    Retries are applied for network errors (httpx.RequestError) and
    for response status codes in `retry_statuses` (default: 429, 5xx).

    Constructor args added:
      retries: number of attempts (default 3)
      backoff_factor: base backoff in seconds (exponential backoff)
      retry_statuses: set of integer HTTP statuses to retry
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        api_key: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
        retries: int = 3,
        backoff_factor: float = 0.5,
        retry_statuses: Optional[Set[int]] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.retries = max(1, int(retries))
        self.backoff_factor = float(backoff_factor)
        # default retry statuses: 429 and 5xx
        if retry_statuses is None:
            self.retry_statuses = {429, 500, 502, 503, 504}
        else:
            self.retry_statuses = set(int(s) for s in retry_statuses)

        self._headers = {"Accept": "application/json"}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"

    def _get_backoff(self, attempt: int) -> float:
        # exponential backoff: backoff_factor * (2 ** (attempt-1))
        return self.backoff_factor * (2 ** (attempt - 1))

    def predict(self, image_path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        p = Path(image_path)
        if not p.exists():
            raise FileNotFoundError(str(p))
        mime, _ = mimetypes.guess_type(p.name)
        mime = mime or "application/octet-stream"

        url = f"{self.base_url}/predict"
        last_exc: Optional[Exception] = None

        for attempt in range(1, self.retries + 1):
            try:
                with httpx.Client(headers=self._headers, timeout=self.timeout) as client, p.open("rb") as f:
                    files = {"file": (p.name, f, mime)}
                    r = client.post(url, files=files, params=params or {})
                    # retry on configured statuses
                    if r.status_code in self.retry_statuses:
                        last_exc = httpx.HTTPStatusError(f"{r.status_code} response", request=r.request, response=r)
                        raise last_exc
                    r.raise_for_status()
                    return r.json()
            except (httpx.RequestError, httpx.HTTPStatusError) as e:
                last_exc = e
                if attempt == self.retries:
                    break
                backoff = self._get_backoff(attempt)
                time.sleep(backoff)

        # All retries exhausted
        raise last_exc  # type: ignore

    async def predict_async(self, image_path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        p = Path(image_path)
        if not p.exists():
            raise FileNotFoundError(str(p))
        mime, _ = mimetypes.guess_type(p.name)
        mime = mime or "application/octet-stream"

        url = f"{self.base_url}/predict"
        last_exc: Optional[Exception] = None

        for attempt in range(1, self.retries + 1):
            try:
                headers = dict(self._headers)
                async with httpx.AsyncClient(headers=headers, timeout=self.timeout) as client:
                    with p.open("rb") as f:
                        files = {"file": (p.name, f, mime)}
                        r = await client.post(url, files=files, params=params or {})
                        if r.status_code in self.retry_statuses:
                            last_exc = httpx.HTTPStatusError(f"{r.status_code} response", request=r.request, response=r)
                            raise last_exc
                        r.raise_for_status()
                        return r.json()
            except (httpx.RequestError, httpx.HTTPStatusError) as e:
                last_exc = e
                if attempt == self.retries:
                    break
                backoff = self._get_backoff(attempt)
                await asyncio.sleep(backoff)

        raise last_exc  # type: ignore
