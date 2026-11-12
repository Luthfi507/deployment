import cv2
import numpy as np
from typing import List
from core.schemas import Detection

class Visualizer:
    @staticmethod
    def get_color(class_id: int) -> tuple:
        np.random.seed(class_id)
        color = np.random.randint(0, 200, size=3).tolist()
        return tuple(color)

    def draw_detection(self, image: np.ndarray, det: Detection):
        annotated_image = image.copy()

        x1 = det.bbox.x1
        y1 = det.bbox.y1
        x2 = det.bbox.x2
        y2 = det.bbox.y2

        label_text = f"{det.class_name} {det.confidence}"
        color = Visualizer.get_color(det.class_id)

        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, thickness=2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        text_thickness = 1

        (text_w, text_h), _ = cv2.getTextSize(label_text, font, font_scale, text_thickness)

        bg_y1 = max(0, y1 - text_h - 10)

        cv2.rectangle(
            annotated_image,
            (x1, bg_y1),
            (x1 + text_w + 4, bg_y1 + text_h + 10),
            color,
            -1
        )

        cv2.putText(
            annotated_image,
            label_text,
            (x1 + 2, bg_y1 + text_h + 5),
            font,
            font_scale,
            (255, 255, 255),
            text_thickness,
            cv2.LINE_AA
        )

        return annotated_image