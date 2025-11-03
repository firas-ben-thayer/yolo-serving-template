from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import tempfile
from pathlib import Path

class BaseModel(ABC):
    @abstractmethod
    def load(self, weights_path: Optional[str] = None) -> None:
        ...

    @abstractmethod
    def infer(self, image_path: str) -> Dict[str, Any]:
        ...

    @abstractmethod
    def infer_bytes(self, image_bytes: bytes) -> Dict[str, Any]:
        ...

class StubModel(BaseModel):
    """
    Lightweight stub that returns the standardized schema used across adapters:
      {"image": str, "width": Optional[int], "height": Optional[int],
       "detections": [ { "box":[x1,y1,x2,y2], "score":float, "class_id":int, "label":str } ],
       "model": {"adapter": "stub", "mode": "dry-run", "version": None}}
    """
    def load(self, weights_path: Optional[str] = None) -> None:
        # no real weights; just mark loaded
        print("StubModel loaded (no weights)")

    def infer(self, image_path: str) -> Dict[str, Any]:
        # return empty but stable schema
        return {
            "image": str(Path(image_path)),
            "width": None,
            "height": None,
            "detections": [],
            "model": {"adapter": "stub", "mode": "dry-run", "version": None},
        }

    def infer_bytes(self, image_bytes: bytes) -> Dict[str, Any]:
        t = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        t.write(image_bytes)
        t.close()
        return self.infer(t.name)