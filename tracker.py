import os
import time
import mlflow
import cv2
from loguru import logger
from collections import defaultdict
from datetime import datetime, timedelta
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

BBOX_GROWTH_THRESHOLD = 1.2
SAVE_JSON_EVERY = 100

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

model_path = "models/yolo11n.onnx"
detect = Detection(
    model_path,
    device="cuda",
    conf_thresh=cfg.conf_thresh,
    img_size=cfg.imgsz
)
viz = Visualizer()

track_history = defaultdict(dict)
largest_bbox_area = defaultdict(float)

source = "tests/walking.mp4"
cap = cv2.VideoCapture(source)

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
frame_interval = 1.0 / fps
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
logger.info(f"Video: {width}x{height} @ {fps:.2f} FPS")

base_dt = datetime.now()

def save_tracking():
    serializable_history = {
        str(track_id): {
            key: value.isoformat() if isinstance(value, datetime) else value
            for key, value in track.items()
        }
        for track_id, track in track_history.items()
    }
    tmp_path = f"{OUTPUT_DIR}/tracking.json.tmp"
    with open(tmp_path, "w") as f:
        json.dump(serializable_history, f, indent=2)
    os.replace(tmp_path, f"{OUTPUT_DIR}/tracking.json")

def overlay_info(data: dict[dict]):
    ids = set()
    count_info = {}

    for idx, item in data.items():
        if idx in ids:
            continue

        name = item["name"]
        count_info[name] = count_info.get(name, 0) + 1
        ids.add(idx)

    return [f"{k}: {v}" for k, v in count_info.items()]

processed_frames = 0
video_frame_idx = 0
skipped_frames = 0
start_time = time.perf_counter()

try:
    while cap.isOpened():
        elapsed = time.perf_counter() - start_time
        target_idx = int(elapsed * fps)
        while video_frame_idx < target_idx:
            if not cap.grab():
                break
            video_frame_idx += 1
            skipped_frames += 1

        ret, frame = cap.read()
        if not ret:
            logger.info("Video has done")
            break

        video_time_sec = video_frame_idx / fps
        frame_dt = base_dt + timedelta(seconds=video_time_sec)
        current_frame_idx = video_frame_idx
        video_frame_idx += 1

        results = detect.track(frame)
        annotated_frame = frame.copy()

        for det in results:
            annotated_frame = viz.draw_detection(annotated_frame, det)

            name = det.class_name
            conf = det.confidence

            if det.id not in track_history:
                track_history[det.id] = {
                    "name": name,
                    "start": frame_dt,
                    "end": frame_dt,
                    "filename": None,
                    "color": None,
                    "min_conf": conf,
                    "max_conf": conf,
                    "duration": 0.0,
                    "frame_count": 0,
                    "frame_logs": [],
                }

            history = track_history[det.id]

            history["end"] = frame_dt
            history["duration"] = (frame_dt - history["start"]).total_seconds()
            history["max_conf"] = max(history["max_conf"], conf)
            history["min_conf"] = min(history["min_conf"], conf)
            history["frame_count"] += 1

            bbox = det.bbox.model_dump()
            history["frame_logs"].append({
                "idx": current_frame_idx,
                "bbox": bbox,
            })

            x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
            bbox_area = max(0, x2 - x1) * max(0, y2 - y1)

            prev_area = largest_bbox_area[det.id]
            if bbox_area > 0 and (
                prev_area == 0 or bbox_area > prev_area * BBOX_GROWTH_THRESHOLD
            ):
                cropped_image = helper.cropped(frame, det.bbox)
                largest_bbox_area[det.id] = bbox_area
                filename = f"{OUTPUT_DIR}/{det.class_name}_{det.id}.jpg"

                cv2.imwrite(filename, cropped_image)
                history.update({
                    "filename": filename,
                })

        processed_frames += 1
        if processed_frames % SAVE_JSON_EVERY == 0:
            save_tracking()

        info = overlay_info(track_history)
        text = "\n".join(info)

        # Overlay info
        cv2.putText(
            annotated_frame,
            f"t={video_time_sec:.2f}s \n{text}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        cv2.imshow("Inference", annotated_frame)

        next_frame_time = start_time + video_frame_idx * frame_interval
        wait_ms = max(1, int((next_frame_time - time.perf_counter()) * 1000))

        if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    save_tracking()

    total_real = time.perf_counter() - start_time
    logger.info(
        f"Done. Processed: {processed_frames} frame, skipped: {skipped_frames} frames, "
        f"time: {total_real:.1f}s, "
        f"FPS: {processed_frames / max(total_real, 1e-6):.1f}"
    )