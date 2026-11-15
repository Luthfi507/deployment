from dataclasses import dataclass

@dataclass
class CarDetection:
    model_name='car-detection'
    model_alias='dev'
    conf_thresh=0.7
    imgsz=640