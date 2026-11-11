import time
import numpy as np
from typing import Union, Literal
from ultralytics import YOLO
from loguru import logger

from .schemas import Detection, BoundingBox

class Inferencer:
    def __init__(self, model_path: str, device: str = 'cpu', conf_thresh: float = 0.0, task: Literal['detect', 'classify', 'segment'] = 'detect'):
        self.model = YOLO(model_path, task=task)
        self.device = device
        self.conf_thresh = conf_thresh
        self.task = task

    def detection(self, image: Union[str, np.ndarray]) -> Detection:
        start_time = time.time()
        results = self.model(
            image,
            conf=self.conf_thresh,
            verbose=False,
            device=self.device
        )[0]

        infer_time = time.time() - start_time
        logger.info(f"Detection successfully in {infer_time}")

        detections = []
        names = self.model.names

        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])

            det = Detection(
                class_id=cls_id,
                class_name=names[cls_id],
                confidence=round(conf, 4),
                bbox=BoundingBox(
                    x1=int(x1),
                    y1=int(y1),
                    x2=int(x2),
                    y2=int(y2),
                )
            )
            detections.append(det)

        return detections

    def predict(self, image):
        if self.task == 'detect':
            return self.detection(image)