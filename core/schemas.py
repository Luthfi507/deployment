from pydantic import BaseModel
from typing import List

class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int

class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox: BoundingBox

class Classification(BaseModel):
    pass

class Segmentation(BaseModel):
    pass