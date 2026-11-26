# Video Semantic Search & Object Tracking Pipeline

## Overview
This project provides an end‑to‑end pipeline that:
1. **Detects and tracks objects** in a video (`tracker.py`).
2. **Creates a multimodal embedding** for each tracked object and stores it in Qdrant (`core/embed.py`).
3. **Serves an interactive dashboard** built with FastAPI that enables semantic search, visual analytics, and video playback (`services/api.py`).

The pipeline is fully containerised with Docker Compose and runs in a Conda environment.

---

## Input Video
- **Sample video:** `walking.mp4`
- **Download:** [YouTube](https://youtu.be/YzcawvDGe4Y)

## Architecture Diagram
```text
tests/walking.mp4
   │
   ▼  (1) Object Detection & Tracking [tracker.py]
output/
 ├── tracking.json          # metadata per track (start, end, duration, frame_logs)
 └── <class>_<id>.jpg       # cropped image of the best bounding‑box per track
   │
   ▼  (2) Multimodal Embedding & Indexing [core/embed.py]
Qdrant (collection: "cve-detection", model: "google/embeddinggemma-2")
   │
   ▼  (3) Semantic Search & Analytics Dashboard [services/api.py]
Web UI (/) & REST API (/api/*)
```

---

## Project Structure
```
├─ tracker.py                 # Video object detection & tracking (YOLO + ByteTrack)
├─ config.py                  # Model and threshold configuration
├─ core/
│   ├─ detect.py              # Detection wrapper
│   ├─ embed.py               # Embedding wrapper (SentenceTransformer + Qdrant)
│   ├─ segment.py
│   ├─ inferencer.py
│   └─ schemas.py
├─ utils/                     # Visualization, cropping, threaded video stream
├─ services/
│   ├─ api.py                 # FastAPI server
│   ├─ templates/            # Jinja2 HTML templates (index.html + partials)
│   └─ static/               # Front‑end assets (css, js)
├─ tests/
│   ├─ walking.mp4           # Sample video for testing
│   └─ test_api.py           # Unit tests for API endpoints
├─ docker-compose.yml         # Docker services (qdrant, mlflow, prometheus, grafana, …)
└─ README.md                  # THIS FILE
```

---

## Prerequisites
- **Python / Conda**: `conda create -n vms python=3.12` and `conda activate vms`.
- **GPU + CUDA**: Required for `tracker.py` (YOLO inference).
- **Docker & Docker Compose**: To run Qdrant and optional MLOps services.
- **Dependencies**: Install with `pip install -r requirements.txt` (includes `ultralytics`, `torch`, `sentence‑transformers`, `qdrant-client`, `fastapi`, `uvicorn`, `jinja2`, `opencv-python`, etc.).

---

## Docker Setup
1. Copy the example environment file and edit if needed:
   ```bash
   cp .env.example .env
   ```
2. Start the required containers (minimum `qdrant` for semantic search):
   ```bash
   docker compose up -d qdrant mlflow   # add other services if you want the full stack
   # or start everything
   docker compose up -d
   ```
3. Service URLs (when running the full stack):
   - Qdrant: `http://localhost:6333`
   - MLflow: `http://localhost:5000`
   - Prometheus: `http://localhost:9090`
   - Grafana: `http://localhost:3000`
   - Portainer: `http://localhost:9443` / `8000`

---

## 1. Generate a Dataset with `tracker.py`
`tracker.py` reads `tests/walking.mp4`, runs YOLO‑11 on CUDA, tracks each object across frames, saves the best cropped image per track to `output/<class>_<id>.jpg`, and writes a comprehensive log to `output/tracking.json`.
```bash
conda activate vms
python tracker.py
```
Press **q** in the OpenCV preview window to stop early – the data generated up to that point is still saved.

### Output
- `output/<class>_<id>.jpg` – cropped object image.
- `output/tracking.json` – dictionary keyed by `track_id` containing:
  - `name` (object class)
  - `start`, `end` (ISO timestamps)
  - `duration` (seconds)
  - `frame_count`
  - `min_conf`, `max_conf`
  - `filename` (path to the cropped image)
  - `frame_logs` (list of frame indices and bounding‑box coordinates for overlay).

---

## 2. Index the Dataset with `core/embed.py`
Encode each cropped image using the `Embedding` wrapper (`google/embeddinggemma-2`) and store the vector in Qdrant:
```bash
conda activate vms
python stored.py
```

---

## 3. Run the Dashboard Service (`services/api.py`)
Make sure the `qdrant` container is running, then start the FastAPI server:
```bash
conda activate vms
uvicorn services.api:app --host 0.0.0.0 --port 5296 --reload
```
Open a browser:
- **Dashboard UI** – `http://localhost:5296`
- **OpenAPI (Swagger) Docs** – `http://localhost:5296/docs`

### Main Features
- **Overview & Browse Mode (`GET /`)** – KPI overview, activity timeline, top object classes, and a gallery of tracked objects.
- **Semantic Search (`GET /?query=…`)** – Natural‑language search over Qdrant embeddings with optional class, time‑range, and result‑limit filters.
- **Video Segment Player & BBox Overlay** – Play a selected track, overlay bounding boxes, adjust playback speed, and view detailed metadata.

---

## API Endpoints
| Method | Endpoint | Description |
| ------ | -------- | ----------- |
| `GET` | `/` | Dashboard home page (HTML) |
| `GET` | `/api/health` | Health check – verifies service, video, tracking store, and Qdrant |
| `GET` | `/api/stats` | Aggregate statistics (track count, class distribution, timeline buckets) |
| `GET` | `/api/search` | Semantic search (`query`, `limit`, `start`, `end`, `class_name`, `min_score`) |
| `GET` | `/api/tracks` | List & filter tracks from `output/tracking.json` (`sort_by`, `offset`, `limit`) |
| `GET` | `/api/tracks/{track_id}` | Detailed info for a single track, including `frame_logs` |

---

## Testing
Run the unit tests to verify the API and static assets:
```bash
conda activate vms
python tests/test_api.py -v
```
All tests should pass (7/7).

---

## Future Enhancements (optional)
- Add a CI workflow to execute tests automatically on push.
- Provide scripts to embed an entire dataset in bulk.
- Extend the UI with custom visualizations.

---

*This README was generated on 2026‑11‑26 and reflects the current state of the repository.*