import numpy as np
from core.schemas import BoundingBox

def cropped(image: np.ndarray, det: BoundingBox):
    x1 = det.x1
    y1 = det.y1
    x2 = det.x2
    y2 = det.y2

    return image[y1:y2, x1:x2]