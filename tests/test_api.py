import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

from services.api import app, get_embed, get_tracking_store


class FakeEmbed:
    """Deterministic fake Embedding client for unit tests."""

    def __init__(self):
        self.collection_name = "cve-detection"
        self.last_call = None
        self.client = SimpleNamespace(
            get_collection=lambda _name: SimpleNamespace(points_count=1062)
        )

    def search(self, query_text: str, limit: int = 5, query_filter=None):
        self.last_call = {
            "query_text": query_text,
            "limit": limit,
            "query_filter": query_filter,
        }
        if query_text == "nothing-matches":
            return []
        return [
            {
                "id": 1,
                "score": 0.7512,
                "img_path": "output/person_1.jpg",
                "name": "person",
                "start": "2026-11-16T19:45:40.647567",
                "end": "2026-11-16T19:45:53.026600",
                "filename": "output/person_1.jpg",
                "min_conf": 0.2733,
                "max_conf": 0.8781,
                "duration": 12.379033,
                "frame_count": 335,
            }
        ]


class DashboardApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def tearDown(self):
        app.dependency_overrides.clear()

    def test_dashboard_overview_without_query(self):
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn("Video Semantic Search &amp; Tracking Dashboard", res.text)
        self.assertIn("Overview &amp; Browse Mode", res.text)
        self.assertIn("/static/css/dashboard.css", res.text)
        self.assertIn("/static/js/dashboard.js", res.text)
        self.assertIn("/tests/walking.mp4", res.text)
        self.assertIn("/output/", res.text)

    def test_dashboard_blank_date_filters_fallback(self):
        res = self.client.get(
            "/?query=&class_name=person&limit=5&start=&end="
        )
        self.assertEqual(res.status_code, 200)
        self.assertIn("person", res.text)

    def test_dashboard_invalid_time_range_returns_400(self):
        res = self.client.get(
            "/?start=2026-11-16T19:47:00&end=2026-11-16T19:46:00"
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn("Start time must be earlier than end time", res.text)

    def test_stats_and_tracks_endpoints(self):
        stats_res = self.client.get("/api/stats")
        self.assertEqual(stats_res.status_code, 200)
        stats = stats_res.json()
        store = get_tracking_store()
        self.assertEqual(stats["total_tracks"], store["stats"]["total_tracks"])
        self.assertEqual(len(stats["timeline_buckets"]), 12)
        self.assertIn("person", stats["class_counts"])

        tracks_res = self.client.get("/api/tracks?class_name=person&limit=5")
        self.assertEqual(tracks_res.status_code, 200)
        tracks_payload = tracks_res.json()
        self.assertGreater(tracks_payload["total"], 0)
        self.assertLessEqual(len(tracks_payload["tracks"]), 5)
        for item in tracks_payload["tracks"]:
            self.assertEqual(item["name"], "person")
            self.assertNotIn("color", item)

    def test_track_detail_and_not_found(self):
        detail_res = self.client.get("/api/tracks/1")
        self.assertEqual(detail_res.status_code, 200)
        detail = detail_res.json()
        self.assertEqual(detail["id"], 1)
        self.assertEqual(detail["name"], "person")
        self.assertGreater(len(detail["frame_logs"]), 0)
        self.assertIn("bbox", detail["frame_logs"][0])
        self.assertIn("time_sec", detail["frame_logs"][0])

        missing_res = self.client.get("/api/tracks/99999999")
        self.assertEqual(missing_res.status_code, 404)

    def test_api_search_and_html_search_with_mocked_embed(self):
        fake_embed = FakeEmbed()
        app.dependency_overrides[get_embed] = lambda: fake_embed

        res = self.client.get(
            "/api/search?query=person+wearing+blue&limit=5&class_name=person"
        )
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertEqual(body["count"], 1)
        self.assertEqual(body["results"][0]["id"], 1)
        self.assertNotIn("color", body["results"][0])
        self.assertEqual(body["results"][0]["image_url"], "/output/person_1.jpg")
        self.assertEqual(body["segment"]["start_offset"], 0.0)
        self.assertGreater(body["segment"]["end_offset"], 12.0)

        must_conditions = fake_embed.last_call["query_filter"].must
        self.assertEqual(len(must_conditions), 3)

        empty_res = self.client.get("/api/search?query=nothing-matches")
        self.assertEqual(empty_res.status_code, 200)
        self.assertEqual(empty_res.json()["count"], 0)
        self.assertIsNone(empty_res.json()["segment"])

        html_res = self.client.get("/?query=person+wearing+blue&limit=5")
        self.assertEqual(html_res.status_code, 200)
        self.assertIn("Semantic Search Mode", html_res.text)
        self.assertIn("Score 0.7512", html_res.text)

        health_res = self.client.get("/api/health")
        self.assertEqual(health_res.status_code, 200)
        self.assertEqual(health_res.json()["status"], "ok")
        self.assertEqual(health_res.json()["qdrant"]["points_count"], 1062)

    def test_static_mounts_accessible(self):
        img_res = self.client.get("/output/person_1.jpg")
        self.assertEqual(img_res.status_code, 200)
        self.assertTrue(img_res.headers["content-type"].startswith("image/"))

        css_res = self.client.get("/static/css/dashboard.css")
        self.assertEqual(css_res.status_code, 200)
        self.assertIn("text/css", css_res.headers["content-type"])

        js_res = self.client.get("/static/js/dashboard.js")
        self.assertEqual(js_res.status_code, 200)
        self.assertIn("javascript", js_res.headers["content-type"])


if __name__ == "__main__":
    unittest.main()
