import time
import numpy as np
from pathlib import Path
from typing import Union
from loguru import logger
import cv2

from ultralytics.models.sam import SAM3SemanticPredictor
from ultralytics.engine.results import Results

from .schemas import SegmentationResult, Point

class Segment:
    def __init__(
            self, 
            model_path: str = 'models/sam3.pt', 
            device: str = 'cuda:0', 
            conf_thresh: float = 0.5, 
            img_size: int = 644,
        ):
        self.model_path = model_path
        self.device = device
        self.conf_thresh = conf_thresh
        self.img_size = img_size
        self.output_dir = Path('output')

        self.model = None
        self.load_model()

    def load_model(self):
        start = time.time()
        overrides = dict(
            conf=self.conf_thresh,
            task='segment',
            imgsz=self.img_size,
            model=self.model_path,
            device=self.device,
            save=False,
            half=True
        )

        if not Path(self.model_path).exists():
            FileExistsError(f"{self.model_path} is not found. Try to download sam3 with 'hf download facebook/sam3'")

        self.model = SAM3SemanticPredictor(overrides=overrides)
        infer_time = time.time() - start
        logger.success(f"Load segmentation model successfully in {infer_time}")

    def predict(self, image: Union[str, np.ndarray], text: str, save: bool = False) -> SegmentationResult:
        start = time.time()
        result: Results = self.model(
            image,
            text=[text]
        )[0]

        infer_time = time.time() - start
        logger.info(f"Image segmented successfylly in {infer_time}")

        annotated_image = result.plot()
        masks = []

        for mask in result.masks:
            points = []

            for xy in mask.xy[0]:
                points.append(
                    Point(x=xy[0], y=xy[1])
                )

            masks.append(
                SegmentationResult(
                    class_name=text,
                    masks=points
                )
            )

        if save:
            cv2.imwrite(self.output_dir / f"{start}.jpg", annotated_image)

        return masks