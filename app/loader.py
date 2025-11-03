from typing import Optional, Dict, Any, List
import os
import tempfile
import logging
from pathlib import Path

from .model import BaseModel, StubModel
from .utils.postprocess import format_detections

logger = logging.getLogger(__name__)

# project root (one level above app/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

class YolovXAdapter(BaseModel):
    """
    Adapter for Ultralytics YOLO (best-effort parsing). If ultralytics isn't installed
    or weights are not provided the adapter stays in dry-run/no-weights mode and returns
    an empty but stable schema.
    """
    def __init__(self):
        self.model = None
        self.mode = "dry-run"
        self.names: Dict[int, str] = {}
        self.version = None

    def load(self, weights_path: Optional[str] = None) -> None:
        try:
            from ultralytics import YOLO  # type: ignore
            import ultralytics as ul
            self.version = getattr(ul, "__version__", None)
            logger.info("Ultralytics available (version=%s)", self.version)

            # try to instantiate model if weights provided; otherwise no-weights mode
            if weights_path and os.path.exists(weights_path):
                logger.info("Loading YOLO weights from %s", weights_path)
                try:
                    self.model = YOLO(weights_path)
                    self.mode = "inference"
                    logger.info("YolovXAdapter loaded model from weights -> mode=%s", self.mode)
                except Exception as e:
                    logger.exception("Failed to instantiate YOLO from weights: %s", e)
                    self.model = None
                    self.mode = "no-weights"
            else:
                # create a model object without weights if possible
                try:
                    self.model = YOLO()
                    self.mode = "no-weights"
                    logger.info("YolovXAdapter instantiated model without weights -> mode=%s", self.mode)
                except Exception as e:
                    logger.debug("Could not instantiate YOLO without weights: %s", e)
                    self.model = None
                    self.mode = "no-weights"

            # try to read class names if available
            try:
                self.names = getattr(self.model, "names", {}) or {}
            except Exception:
                self.names = {}
        except Exception as e:
            # ultralytics not installed -> dry run
            logger.debug("Ultralytics import failed: %s", e)
            self.model = None
            self.mode = "dry-run"
            self.names = {}

    def _parse_results(self, results: Any, image_path: str) -> Dict[str, Any]:
        """
        Best-effort conversion from ultralytics Results -> stable schema.
        """
        detections: List[Dict[str, Any]] = []
        width = None
        height = None

        try:
            # results may be a list-like of Result objects
            r = results[0] if isinstance(results, (list, tuple)) else results
            # try to obtain original image shape
            shape = getattr(r, "orig_shape", None)
            if shape and len(shape) >= 2:
                height, width = int(shape[0]), int(shape[1])

            boxes_obj = getattr(r, "boxes", None)
            if boxes_obj is None:
                return {"image": image_path, "width": width, "height": height, "detections": [], "model": {"adapter": "yolovx", "mode": self.mode, "version": self.version}}

            # boxes_obj may expose xyxy, conf, cls as tensors or lists
            xyxy = getattr(boxes_obj, "xyxy", None)
            confs = getattr(boxes_obj, "conf", None)
            clss = getattr(boxes_obj, "cls", None)

            # convert to python lists
            if xyxy is not None:
                try:
                    boxes = xyxy.cpu().numpy().tolist()  # type: ignore
                except Exception:
                    try:
                        boxes = list(xyxy)  # last resort
                    except Exception:
                        boxes = []
            else:
                boxes = []

            if confs is not None:
                try:
                    scores = confs.cpu().numpy().tolist()  # type: ignore
                except Exception:
                    scores = list(confs)
            else:
                scores = [0.0] * len(boxes)

            if clss is not None:
                try:
                    classes = clss.cpu().numpy().tolist()  # type: ignore
                except Exception:
                    classes = list(clss)
            else:
                classes = [0] * len(boxes)

            # normalize lengths
            n = min(len(boxes), len(scores), len(classes))
            boxes = boxes[:n]
            scores = scores[:n]
            classes = classes[:n]

            detections = format_detections(boxes, scores, classes, names=self.names)
        except Exception:
            # on any parsing failure return an empty detections list but keep schema
            detections = []

        return {"image": image_path, "width": width, "height": height, "detections": detections, "model": {"adapter": "yolovx", "mode": self.mode, "version": self.version}}

    def infer(self, image_path: str) -> Dict[str, Any]:
        if self.model is None:
            # dry-run: return stable schema with empty detections
            return {"image": image_path, "width": None, "height": None, "detections": [], "model": {"adapter": "yolovx", "mode": self.mode, "version": self.version}}

        # run prediction using model.predict or model(image_path) depending on ul version
        try:
            # prefer predict API if available
            if hasattr(self.model, "predict"):
                results = self.model.predict(source=image_path)
            else:
                results = self.model(image_path)
            return self._parse_results(results, image_path)
        except Exception:
            # bubble up to service boundary to produce 500
            raise

    def infer_bytes(self, image_bytes: bytes) -> Dict[str, Any]:
        t = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        t.write(image_bytes)
        t.close()
        return self.infer(t.name)


def get_model(adapter: str = "stub", weights: Optional[str] = None) -> BaseModel:
    """
    Factory to produce a BaseModel implementation.
    - adapter: "stub" or "yolovx" (future adapters can be added)
    - weights: optional path to weights file for real adapters
    """
    adapter = (adapter or "stub").lower()
    # accept a few common aliases and normalize to the yolovx adapter
    if adapter.startswith("yolo") or adapter.startswith("yolov") or "yolovx" in adapter or "yolov" in adapter:
        chosen = "yolovx"
    elif adapter in ("yolovx", "yolox"):
        chosen = "yolovx"
    else:
        chosen = "stub"

    # normalize weights path: allow relative paths in .env (relative to project root)
    if weights:
        try:
            w = Path(weights).expanduser()
            if not w.is_absolute():
                w = (PROJECT_ROOT / w).resolve()
            weights = str(w)
        except Exception:
            logger.debug("Could not normalize weights path: %s", weights, exc_info=True)

    logger.info("Requested adapter='%s' normalized='%s' weights=%s", adapter, chosen, weights)

    if chosen == "yolovx":
        m = YolovXAdapter()
    else:
        m = StubModel()

    try:
        m.load(weights)
    except Exception as e:
        logger.exception("Model.load() raised an exception: %s", e)

    logger.info("Model loaded: adapter=%s mode=%s", chosen, getattr(m, "mode", None))
    return m