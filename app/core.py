from typing import Optional, Dict, Any
import os

from .loader import get_model
from .model import BaseModel

def predict_inproc(
    image_path: str,
    model: Optional[BaseModel] = None,
    adapter: Optional[str] = None,
    weights: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run a prediction in-process.

    - If 'model' is provided it is used directly.
    - Otherwise get_model(adapter, weights) is used (falls back to env vars/defaults).
    """
    if model is None:
        adapter = adapter or os.getenv("MODEL_ADAPTER", "stub")
        weights = weights or os.getenv("MODEL_WEIGHTS", None)
        model = get_model(adapter=adapter, weights=weights)

    return model.infer(image_path)