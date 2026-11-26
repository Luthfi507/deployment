from contextlib import asynccontextmanager
from datetime import datetime
from functools import lru_cache
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from loguru import logger
from qdrant_client.models import (
    DatetimeRange,
    FieldCondition,
    Filter,
    MatchValue,
)

from core import Embedding

BASE_DIR = Path(__file__).resolve().parent.parent
SERVICES_DIR = Path(__file__).resolve().parent
TESTS_DIR = BASE_DIR / "tests"
OUTPUT_DIR = BASE_DIR / "output"
TEMPLATES_DIR = SERVICES_DIR / "templates"
STATIC_DIR = SERVICES_DIR / "static"
TRACKING_JSON_PATH = OUTPUT_DIR / "tracking.json"
VIDEO_PATH = TESTS_DIR / "walking.mp4"

DEFAULT_EARLIEST = datetime.fromisoformat("2026-11-16T19:45:40.647567")
DEFAULT_LATEST = datetime.fromisoformat("2026-11-16T19:47:37.163967")

_embed_instance: Optional[Embedding] = None

def get_embed() -> Embedding:
    """Return a lazily initialized singleton Embedding instance."""
    global _embed_instance
    if _embed_instance is None:
        _embed_instance = Embedding()
    return _embed_instance

@lru_cache(maxsize=1)
def get_video_metadata() -> Dict[str, Any]:
    """Read video resolution, FPS, frame count, and duration."""
    fallback = {
        "width": 1920,
        "height": 1080,
        "fps": 29.97002997002997,
        "frame_count": 3493,
        "duration": 116.55,
        "exists": VIDEO_PATH.exists(),
    }
    if not VIDEO_PATH.exists():
        return fallback

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        return fallback

    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or fallback["fps"])
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or fallback["width"])
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or fallback["height"])
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or fallback["frame_count"])
        duration = round(frame_count / fps, 3) if fps > 0 else fallback["duration"]
        return {
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": frame_count,
            "duration": duration,
            "exists": True,
        }
    finally:
        cap.release()

def _build_timeline_buckets(
    tracks: Dict[int, Dict[str, Any]],
    earliest: datetime,
    total_duration: float,
    num_buckets: int = 12,
) -> List[Dict[str, Any]]:
    """Aggregate active track counts into evenly spaced timeline buckets."""
    safe_duration = max(total_duration, 1.0)
    bucket_size = safe_duration / num_buckets
    buckets = [
        {
            "index": i,
            "start_sec": round(i * bucket_size, 1),
            "end_sec": round((i + 1) * bucket_size, 1),
            "label": f"{int(i * bucket_size)}s",
            "count": 0,
        }
        for i in range(num_buckets)
    ]

    for track in tracks.values():
        t_start = max(0.0, (track["start_dt"] - earliest).total_seconds())
        t_end = max(t_start, (track["end_dt"] - earliest).total_seconds())
        start_idx = min(num_buckets - 1, int(t_start / bucket_size))
        end_idx = min(num_buckets - 1, int(t_end / bucket_size))
        for idx in range(start_idx, end_idx + 1):
            buckets[idx]["count"] += 1

    return buckets

@lru_cache(maxsize=1)
def get_tracking_store() -> Dict[str, Any]:
    """Load and index tracking metadata from output/tracking.json."""
    if not TRACKING_JSON_PATH.exists():
        return {
            "tracks": {},
            "earliest": DEFAULT_EARLIEST,
            "latest": DEFAULT_LATEST,
            "classes": [],
            "stats": {
                "total_tracks": 0,
                "unique_classes": 0,
                "total_detections": 0,
                "avg_duration": 0.0,
                "max_duration": 0.0,
                "class_counts": {},
                "timeline_buckets": [],
                "earliest": DEFAULT_EARLIEST.isoformat(),
                "latest": DEFAULT_LATEST.isoformat(),
            },
        }

    with open(TRACKING_JSON_PATH, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    tracks: Dict[int, Dict[str, Any]] = {}
    class_counts: Dict[str, int] = {}
    total_detections = 0
    durations: List[float] = []

    for raw_id, item in raw_data.items():
        track_id = int(raw_id)
        start_dt = datetime.fromisoformat(item["start"])
        end_dt = datetime.fromisoformat(item["end"])
        name = item.get("name") or "unknown"
        duration = float(item.get("duration") or max(0.0, (end_dt - start_dt).total_seconds()))
        frame_count = int(item.get("frame_count") or len(item.get("frame_logs", [])))

        class_counts[name] = class_counts.get(name, 0) + 1
        total_detections += frame_count
        durations.append(duration)

        tracks[track_id] = {
            **item,
            "id": track_id,
            "name": name,
            "duration": round(duration, 3),
            "frame_count": frame_count,
            "start_dt": start_dt,
            "end_dt": end_dt,
        }

    earliest = min((t["start_dt"] for t in tracks.values()), default=DEFAULT_EARLIEST)
    latest = max((t["end_dt"] for t in tracks.values()), default=DEFAULT_LATEST)

    for track in tracks.values():
        track["offset_start"] = round((track["start_dt"] - earliest).total_seconds(), 3)
        track["offset_end"] = round((track["end_dt"] - earliest).total_seconds(), 3)
        filename = track.get("filename") or ""
        track["image_url"] = f"/{filename.lstrip('/')}" if filename else None

    sorted_classes = sorted(class_counts.items(), key=lambda kv: kv[1], reverse=True)
    total_duration = max((latest - earliest).total_seconds(), get_video_metadata()["duration"])

    return {
        "tracks": tracks,
        "earliest": earliest,
        "latest": latest,
        "classes": [k for k, _ in sorted_classes],
        "stats": {
            "total_tracks": len(tracks),
            "unique_classes": len(class_counts),
            "total_detections": total_detections,
            "avg_duration": round(sum(durations) / len(durations), 2) if durations else 0.0,
            "max_duration": round(max(durations), 2) if durations else 0.0,
            "class_counts": dict(sorted_classes),
            "timeline_buckets": _build_timeline_buckets(tracks, earliest, total_duration),
            "earliest": earliest.isoformat(),
            "latest": latest.isoformat(),
        },
    }

def _parse_datetime(value: Optional[str], default: datetime, field_name: str) -> datetime:
    """Parse an ISO datetime string, falling back to default for blank inputs."""
    if value is None or value.strip() == "":
        return default
    try:
        return datetime.fromisoformat(value.strip())
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid datetime format for '{field_name}': {value}",
        ) from exc

def _validate_time_range(start_dt: datetime, end_dt: datetime) -> None:
    """Ensure the start timestamp strictly precedes the end timestamp."""
    if start_dt >= end_dt:
        raise HTTPException(
            status_code=400,
            detail="Start time must be earlier than end time",
        )

def _format_track_summary(
    item: Dict[str, Any],
    earliest: datetime,
    tracking_tracks: Dict[int, Dict[str, Any]],
) -> Dict[str, Any]:
    """Build a lightweight, UI-friendly summary dict without bulky frame_logs."""
    track_id = int(item["id"])
    stored = tracking_tracks.get(track_id, {})

    start_str = item.get("start") or stored.get("start")
    end_str = item.get("end") or stored.get("end")
    start_dt = datetime.fromisoformat(start_str)
    end_dt = datetime.fromisoformat(end_str)

    filename = item.get("filename") or item.get("img_path") or stored.get("filename") or ""
    image_url = f"/{filename.lstrip('/')}" if filename else None

    return {
        "id": track_id,
        "score": round(float(item["score"]), 4) if item.get("score") is not None else None,
        "name": item.get("name") or stored.get("name") or "unknown",
        "start": start_str,
        "end": end_str,
        "offset_start": round((start_dt - earliest).total_seconds(), 3),
        "offset_end": round((end_dt - earliest).total_seconds(), 3),
        "duration": round(float(item.get("duration") or stored.get("duration") or 0.0), 3),
        "frame_count": int(item.get("frame_count") or stored.get("frame_count") or 0),
        "min_conf": round(float(item.get("min_conf") or stored.get("min_conf") or 0.0), 4),
        "max_conf": round(float(item.get("max_conf") or stored.get("max_conf") or 0.0), 4),
        "filename": filename,
        "image_url": image_url,
    }

def _execute_semantic_search(
    embed: Embedding,
    store: Dict[str, Any],
    query: str,
    limit: int,
    start_dt: datetime,
    end_dt: datetime,
    class_name: Optional[str] = None,
    min_score: float = 0.0,
) -> List[Dict[str, Any]]:
    """Run filtered semantic search in Qdrant and enrich hits with tracking data."""
    conditions: List[Any] = [
        FieldCondition(key="start", range=DatetimeRange(gte=start_dt)),
        FieldCondition(key="end", range=DatetimeRange(lte=end_dt)),
    ]

    if class_name:
        conditions.append(FieldCondition(key="name", match=MatchValue(value=class_name)))

    raw_results = embed.search(query, limit, Filter(must=conditions))
    earliest = store["earliest"]
    enriched: List[Dict[str, Any]] = []

    for item in raw_results:
        score = float(item.get("score", 0.0))
        if score < min_score:
            continue
        enriched.append(_format_track_summary(item, earliest, store["tracks"]))

    return enriched

def _browse_tracks(
    store: Dict[str, Any],
    start_dt: datetime,
    end_dt: datetime,
    class_name: Optional[str] = None,
    min_conf: float = 0.0,
    sort_by: str = "duration",
    offset: int = 0,
    limit: int = 12,
) -> Dict[str, Any]:
    """Filter and paginate tracks directly from tracking.json."""
    filtered: List[Dict[str, Any]] = []
    earliest = store["earliest"]

    for track in store["tracks"].values():
        if track["start_dt"] < start_dt or track["end_dt"] > end_dt:
            continue
        if class_name and track["name"] != class_name:
            continue
        if track.get("max_conf", 0.0) < min_conf:
            continue
        filtered.append(track)

    sort_key_map = {
        "duration": lambda t: t["duration"],
        "confidence": lambda t: t.get("max_conf", 0.0),
        "frames": lambda t: t["frame_count"],
        "start": lambda t: -t["offset_start"],
    }
    key_fn = sort_key_map.get(sort_by, sort_key_map["duration"])
    filtered.sort(key=key_fn, reverse=True)

    page = filtered[offset : offset + limit]
    items = [_format_track_summary(t, earliest, store["tracks"]) for t in page]
    return {"total": len(filtered), "items": items}

@asynccontextmanager
async def lifespan(_: FastAPI):
    """Preload tracking metadata, video info, and embedding model on startup."""
    get_video_metadata()
    get_tracking_store()
    try:
        get_embed()
    except Exception as exc:  # pragma: no cover - startup fallback when Qdrant is offline
        logger.warning(f"Embedding/Qdrant preload skipped at startup: {exc}")
    yield

app = FastAPI(
    title="Video Semantic Search & Tracking Dashboard",
    version="1.0.0",
    lifespan=lifespan,
)

TESTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/tests", StaticFiles(directory=str(TESTS_DIR)), name="tests")
app.mount("/output", StaticFiles(directory=str(OUTPUT_DIR)), name="output")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

@app.get("/", response_class=HTMLResponse)
async def index(
    request: Request,
    query: Optional[str] = None,
    limit: int = Query(12, ge=1, le=100),
    start: Optional[str] = None,
    end: Optional[str] = None,
    class_name: Optional[str] = None,
    min_score: float = Query(0.0, ge=0.0, le=1.0),
):
    """Render the interactive Video Semantic Search & Tracking Dashboard."""
    store = get_tracking_store()
    video_meta = get_video_metadata()

    earliest = store["earliest"]
    latest = store["latest"]
    start_dt = _parse_datetime(start, earliest, "start")
    end_dt = _parse_datetime(end, latest, "end")
    _validate_time_range(start_dt, end_dt)

    clean_query = (query or "").strip()
    clean_class = (class_name or "").strip() or None

    results: List[Dict[str, Any]] = []
    message: Optional[str] = None
    search_error: Optional[str] = None
    segment_start: Optional[float] = 0.0
    segment_end: Optional[float] = video_meta["duration"]
    mode = "overview"

    if clean_query:
        mode = "search"
        try:
            embed_provider = app.dependency_overrides.get(get_embed, get_embed)
            embed = embed_provider()
            results = _execute_semantic_search(
                embed=embed,
                store=store,
                query=clean_query,
                limit=limit,
                start_dt=start_dt,
                end_dt=end_dt,
                class_name=clean_class,
                min_score=min_score,
            )
            if results:
                segment_start = min(r["offset_start"] for r in results)
                segment_end = max(r["offset_end"] for r in results)
            else:
                segment_start = None
                segment_end = None
                message = "No matching video segments found for your query and filters."
        except Exception as exc:
            logger.error(f"Semantic search failed: {exc}")
            search_error = f"Semantic search unavailable: {exc}"
            segment_start = None
            segment_end = None
    else:
        browsed = _browse_tracks(
            store=store,
            start_dt=start_dt,
            end_dt=end_dt,
            class_name=clean_class,
            limit=limit,
        )
        results = browsed["items"]
        if results:
            segment_start = results[0]["offset_start"]
            segment_end = results[0]["offset_end"]
        else:
            message = "No tracked objects match the selected filters."

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "video_url": "/tests/walking.mp4",
            "video_meta": video_meta,
            "mode": mode,
            "start": segment_start,
            "end": segment_end,
            "results": results,
            "message": message,
            "search_error": search_error,
            "query": clean_query,
            "limit": limit,
            "class_name": clean_class or "",
            "min_score": min_score,
            "start_filter": start_dt.strftime("%Y-%m-%dT%H:%M:%S"),
            "end_filter": end_dt.strftime("%Y-%m-%dT%H:%M:%S"),
            "stats": store["stats"],
            "classes": store["classes"],
        },
    )

@app.get("/api/health")
async def health():
    """Return service, tracking store, video file, and Qdrant health."""
    store = get_tracking_store()
    video_meta = get_video_metadata()
    qdrant_status = {"connected": False, "points_count": None, "error": None}

    try:
        embed_provider = app.dependency_overrides.get(get_embed, get_embed)
        embed = embed_provider()
        info = embed.client.get_collection(embed.collection_name)
        qdrant_status["connected"] = True
        qdrant_status["points_count"] = info.points_count
    except Exception as exc:
        qdrant_status["error"] = str(exc)

    return {
        "status": "ok" if qdrant_status["connected"] else "degraded",
        "tracking_tracks": store["stats"]["total_tracks"],
        "video": video_meta,
        "qdrant": qdrant_status,
    }

@app.get("/api/stats")
async def stats():
    """Return aggregate tracking analytics and timeline buckets."""
    store = get_tracking_store()
    return {
        "video": get_video_metadata(),
        **store["stats"],
    }

@app.get("/api/search")
async def api_search(
    query: str = Query(..., min_length=1),
    limit: int = Query(10, ge=1, le=100),
    start: Optional[str] = None,
    end: Optional[str] = None,
    class_name: Optional[str] = None,
    min_score: float = Query(0.0, ge=0.0, le=1.0),
    embed: Embedding = Depends(get_embed),
):
    """Execute semantic search over tracked video objects."""
    store = get_tracking_store()
    start_dt = _parse_datetime(start, store["earliest"], "start")
    end_dt = _parse_datetime(end, store["latest"], "end")
    _validate_time_range(start_dt, end_dt)

    results = _execute_semantic_search(
        embed=embed,
        store=store,
        query=query.strip(),
        limit=limit,
        start_dt=start_dt,
        end_dt=end_dt,
        class_name=(class_name or "").strip() or None,
        min_score=min_score,
    )

    segment = (
        {
            "start_offset": min(r["offset_start"] for r in results),
            "end_offset": max(r["offset_end"] for r in results),
        }
        if results
        else None
    )

    return {
        "query": query.strip(),
        "count": len(results),
        "segment": segment,
        "results": results,
    }

@app.get("/api/tracks")
async def list_tracks(
    class_name: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    min_conf: float = Query(0.0, ge=0.0, le=1.0),
    sort_by: str = Query("duration", pattern="^(duration|confidence|frames|start)$"),
    offset: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
):
    """Browse and filter tracked objects from tracking.json."""
    store = get_tracking_store()
    start_dt = _parse_datetime(start, store["earliest"], "start")
    end_dt = _parse_datetime(end, store["latest"], "end")
    _validate_time_range(start_dt, end_dt)

    data = _browse_tracks(
        store=store,
        start_dt=start_dt,
        end_dt=end_dt,
        class_name=(class_name or "").strip() or None,
        min_conf=min_conf,
        sort_by=sort_by,
        offset=offset,
        limit=limit,
    )
    return {
        "total": data["total"],
        "offset": offset,
        "limit": limit,
        "tracks": data["items"],
    }

@app.get("/api/tracks/{track_id}")
async def get_track_detail(track_id: int):
    """Return full track metadata and frame-level bounding boxes for overlay."""
    store = get_tracking_store()
    track = store["tracks"].get(track_id)
    if not track:
        raise HTTPException(status_code=404, detail=f"Track {track_id} not found")

    video_meta = get_video_metadata()
    fps = video_meta["fps"] or 30.0
    summary = _format_track_summary(track, store["earliest"], store["tracks"])
    frame_logs = [
        {
            "idx": entry["idx"],
            "time_sec": round(entry["idx"] / fps, 3),
            "bbox": entry["bbox"],
        }
        for entry in track.get("frame_logs", [])
    ]

    return {
        **summary,
        "video_width": video_meta["width"],
        "video_height": video_meta["height"],
        "fps": fps,
        "frame_logs": frame_logs,
    }