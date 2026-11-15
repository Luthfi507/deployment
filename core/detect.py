import time
import numpy as np
from typing import Union
from loguru import logger

from ultralytics import YOLO
from ultralytics.engine.results import Results

from .schemas import DetectionResult, BoundingBox

class Detection:
    def __init__(
            self, 
            model_path: str, 
            device: str = 'cpu', 
            conf_thresh: float = 0.25,
            img_size: int = 640
        ):
        self.model = YOLO(model_path, task='detect')
        self.device = device
        self.conf_thresh = conf_thresh
        self.img_size = img_size

    def format_result(self, results: Results):
        detections = []
        names = self.model.names

        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            _id = int(box.id) if box.id else 0

            det = DetectionResult(
                class_id=cls_id,
                class_name=names[cls_id],
                confidence=round(conf, 4),
                bbox=BoundingBox(
                    x1=int(x1),
                    y1=int(y1),
                    x2=int(x2),
                    y2=int(y2),
                ),
                id=_id,
                is_track=box.is_track
            )
            detections.append(det)

        return detections

    def predict(self, image: Union[str, np.ndarray]) -> DetectionResult:
        start_time = time.time()
        results = self.model(
            image,
            conf=self.conf_thresh,
            verbose=False,
            device=self.device,
            imgsz=self.img_size
        )[0]

        infer_time = time.time() - start_time
        logger.success(f"Detection successfully in {infer_time}")

        return self.format_result(results)

    def track(self, image: Union[str, np.ndarray], overrides: dict = {}):
        params = dict(
            mode='track',
            persist=True,
            verbose=False,
            tracker='bytetrack.yaml',
            augment=False,
            conf=self.conf_thresh,
            imgsz=self.img_size
        )
        params.update(**overrides)

        results = self.model.track(image, **params)[0]
        return self.format_result(results)