"""detector.py — YOLOv8n-based object detector."""

from dataclasses import dataclass
import numpy as np
from ultralytics import YOLO

# COCO class name lookup (subset we care about)
_COCO_NAMES = {
    0: "person",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


@dataclass
class Detection:
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    class_name: str


class Detector:
    """Runs YOLOv8n inference and returns filtered detections."""

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf_threshold: float = 0.3,
        target_classes: list[int] | None = None,
        device: str = "0",
    ) -> None:
        self._model = YOLO(model_path)
        self._conf = conf_threshold
        self._target = set(target_classes) if target_classes else set(_COCO_NAMES.keys())
        self._device = device

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """
        Run inference on a single BGR frame.
        Returns a list of Detection objects filtered to target_classes.
        Returns an empty list when no target-class objects are found.
        """
        results = self._model(frame, conf=self._conf, verbose=False, device=self._device)
        detections: list[Detection] = []

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                if cls_id not in self._target:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0].item())
                detections.append(
                    Detection(
                        bbox=(x1, y1, x2, y2),
                        confidence=conf,
                        class_id=cls_id,
                        class_name=_COCO_NAMES.get(cls_id, str(cls_id)),
                    )
                )

        return detections
