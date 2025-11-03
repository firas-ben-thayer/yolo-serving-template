from typing import Any, Dict, Optional


def predict_inproc(image_path: str, model: Optional[Any] = None, adapter: Optional[str] = None, weights: Optional[str] = None) -> Dict[str, Any]:
    """
    In-process wrapper that reuses `app.core.predict_inproc`.

    Example:
      from client.inproc import predict_inproc
      out = predict_inproc("img.jpg", adapter="stub")
    """
    try:
        from app.core import predict_inproc as core_predict
    except Exception as e:
        raise RuntimeError("in-process predict unavailable; ensure your app package is importable") from e

    # core_predict will initialize a model if `model` is None
    return core_predict(image_path, model=model, adapter=adapter, weights=weights)
