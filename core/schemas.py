from pydantic import BaseModel
from typing import List

class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int

# Detection
class DetectionResult(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox: BoundingBox
    id: int
    is_track: bool

# Classification
class ClassificationResult(BaseModel):
    class_id: int
    class_name: str
    confidence: float

# Segmentation
class Point(BaseModel):
    x: float
    y: float

class SegmentationResult(BaseModel):
    class_name: str
    masks: List[Point]