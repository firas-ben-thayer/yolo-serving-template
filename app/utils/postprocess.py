from typing import List, Dict, Any, Optional

def format_detections(
    boxes: List[List[float]],
    scores: List[float],
    classes: List[int],
    names: Optional[Dict[int, str]] = None,
) -> List[Dict[str, Any]]:
    """
    Convert raw arrays to a stable detection schema:
      {"box": [x1,y1,x2,y2], "score": float, "class_id": int, "label": str}
    """
    names = names or {}
    out = []
    for b, s, c in zip(boxes, scores, classes):
        label = names.get(int(c), str(int(c)))
        out.append({
            "box": [float(b[0]), float(b[1]), float(b[2]), float(b[3])],
            "score": float(s),
            "class_id": int(c),
            "label": label,
        })
    return out