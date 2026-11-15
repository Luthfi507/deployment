import mlflow
import cv2
from loguru import logger
from collections import defaultdict
from datetime import datetime
import json
import torch

from core import Detection
from utils import Visualizer, helper
from config import CarDetection

# Check cuda
assert torch.cuda.is_available(), "CUDA is not available"
logger.info(f"CUDA available: {torch.cuda.is_available()}")
logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

mlflow.set_tracking_uri("http://localhost:5000")

cfg = CarDetection()
onnx_path = mlflow.artifacts.download_artifacts(
    artifact_uri=f"models:/{cfg.model_name}@{cfg.model_alias}/artifacts/best.onnx",
    dst_path="models/car-detection"
)

detect = Detection(
    onnx_path,
    device='cpu',
    conf_thresh=cfg.conf_thresh,
    img_size=cfg.imgsz
)
viz = Visualizer()

track_history = defaultdict(dict)
largest_bbox_area = defaultdict(float)

source = 'tests/test.mp4'
cap = cv2.VideoCapture(source)

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            logger.info("Video has done")
            break

        results = detect.track(frame)
        annotated_frame = frame.copy()

        for det in results:
            current_time_dt = datetime.now()
            annotated_frame = viz.draw_detection(annotated_frame, det)

            name = det.class_name
            conf = det.confidence

            if det.id not in track_history:
                track_history[det.id] = {
                    "name": name,
                    "start": current_time_dt,
                    "end": current_time_dt,
                    "filename": None,
                    "color": None,
                    "min_conf": conf,
                    "max_conf": conf,
                    "duration": 0,
                    "frame_count": 1
                }

            history = track_history[det.id]

            history["end"] = current_time_dt
            history["duration"] = (current_time_dt - history["start"]).total_seconds()

            if conf > history["max_conf"]:
                history["max_conf"] = conf

            if conf < history["min_conf"]:
                history["min_conf"] = conf

            history["frame_count"] += 1

            bbox = det.bbox.model_dump()
            x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
            bbox_area = max(0, x2 - x1) * max(0, y2 - y1)

            if bbox_area > largest_bbox_area[det.id]:
                cropped_image = helper.cropped(frame, det.bbox)
                largest_bbox_area[det.id] = bbox_area
                filename = f"output/{det.id}.jpg"
                color = helper.get_dominant_color(cropped_image)

                cv2.imwrite(filename, cropped_image)
                history.update({
                    "filename": filename,
                    "color": color
                })

            serializable_history = {
                str(track_id): {
                    key: value.isoformat() if isinstance(value, datetime) else value
                    for key, value in track.items()
                }
                for track_id, track in track_history.items()
            }

            with open("output/tracking.json", 'w+') as f:
                json.dump(serializable_history, f, indent=2)

        cv2.putText(
            annotated_frame,
            f"Count: {len(list(track_history.keys()))}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow("Inference", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()