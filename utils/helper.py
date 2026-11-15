import numpy as np
import cv2

from core.schemas import BoundingBox

def cropped(image: np.ndarray, det: BoundingBox):
    x1 = det.x1
    y1 = det.y1
    x2 = det.x2
    y2 = det.y2

    return image[y1:y2, x1:x2]

def get_dominant_color(img: np.ndarray) -> str:
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    color_ranges = {
        "red": [
            (np.array([0, 120, 70]), np.array([10, 255, 255])),
            (np.array([170, 120, 70]), np.array([180, 255, 255]))
        ],
        "blue": [(np.array([90, 50, 50]), np.array([130, 255, 255]))],
        "yellow": [(np.array([20, 100, 100]), np.array([30, 255, 255]))],
        "black": [(np.array([0, 0, 0]), np.array([180, 255, 30]))],
        "white": [(np.array([0, 0, 200]), np.array([180, 20, 255]))]
    }

    max_pixels = 0
    dominant_color = "unknown"

    for color_name, range in color_ranges.items():
        mask = np.zeros(hsv_img.shape[:2], dtype=np.uint8)

        for (lower, upper) in range:
            mask |= cv2.inRange(hsv_img, lower, upper)

        pixel_count = cv2.countNonZero(mask)
        if pixel_count > max_pixels:
            max_pixels = pixel_count
            dominant_color = color_name

    return dominant_color