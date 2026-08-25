import tempfile
import unittest
from pathlib import Path
from http.server import ThreadingHTTPServer
from threading import Thread
from urllib.request import urlopen

from feedback_server import (
    FeedbackHandler,
    HEALTH_MARKER,
    build_feedback_stats,
    normalize_model_path_breakdown,
    render_feedback_detail_html,
    render_reading_queue_html,
    render_stats_html,
)
from src.database import Database


class FeedbackServerTests(unittest.TestCase):
    def test_track_feedback_appears_in_reading_queue_page(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = str(Path(temp_dir) / "ai_news.db")
            db = Database(db_path)
            db.insert_article(
                {
                    "title": "World model training update",
                    "url": "https://example.com/world-model",
                    "source": "example",
                    "content_type": "paper",
                    "topic": "World Model",
                }
            )
            article = db.get_articles_for_run("", processed_only=False)[0]
            db.record_report_items(
                "report-queue",
                [{"id": article["id"], "editorial_title": "世界模型训练方法更新", "url": article["url"]}],
            )
            db.record_article_feedback("report-queue", article["id"], "track")

            page = render_reading_queue_html(db_path)

        self.assertIn("世界模型训练方法更新", page)
        self.assertIn("标为已读", page)

    def test_stats_include_feedback_counts_weights_and_latest_report(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = str(Path(temp_dir) / "ai_news.db")
            db = Database(db_path)
            db.insert_article(
                {
                    "title": "OpenAI launches workflow agent",
                    "url": "https://example.com/openai-agent",
                    "source_detail": "OpenAI Blog",
                    "topic": "Agents",
                    "facts": {"who": "OpenAI"},
                }
            )
            article = db.get_articles_for_run("", processed_only=False)[0]
            db.record_report_run(
                "report-1",
                "run-1",
                html_report_path="archive/report.html",
                quality_status="passed",
                quality_diagnostics={
                    "model_path_breakdown": {"deepseek-v4-pro": 1},
                    "content_quality": {"bad_title_count": 0},
                    "last_model_path_refresh": {"mode": "refresh_fallback_model_path", "updated": 3},
                    "title_repair": {
                        "bad_title_repaired_count": 1,
                        "bad_title_unresolved_count": 0,
                        "examples": [
                            {
                                "rank": 1,
                                "section": "must_read",
                                "article_id": article["id"],
                                "source": "summary",
                                "old_title": "AgentWatchdemonstrproactive AWS mo",
                                "new_title": "AgentWatch展示了主动式AWS监控的新进展",
                            }
                        ],
                    },
                },
            )
            db.record_report_items(
                "report-1",
                [
                    {
                        "id": article["id"],
                        "report_rank": 1,
                        "report_section": "must_read",
                        "title_cn": "AgentWatchdemonstrproactive AWS mo",
                        "title": "OpenAI launches workflow agent",
                        "summary": "OpenAI launches workflow agent.",
                        "evidence_quality": 0.8,
                        "information_density": 0.8,
                    }
                ],
            )
            db.record_article_feedback("report-1", article["id"], "useful")

            stats = build_feedback_stats(db_path)
            html = render_stats_html(stats)

        self.assertEqual(stats["feedback_count_7d"], 1)
        self.assertEqual(stats["feedback_count_30d"], 1)
        self.assertEqual(stats["latest_report"]["quality_status"], "passed")
        self.assertEqual(stats["latest_report"]["content_quality"]["bad_title_count"], 0)
        self.assertEqual(stats["latest_report"]["title_repair"]["bad_title_repaired_count"], 1)
        self.assertEqual(stats["last_model_path_refresh"]["updated"], 3)
        self.assertEqual(stats["quality_issues_top10"][0]["issue_types"], ["bad_title"])
        self.assertEqual(stats["recent_feedback"][0]["signal"], "useful")
        self.assertEqual(stats["recent_feedback"][0]["title"], "OpenAI launches workflow agent")
        self.assertIn("OpenAI Blog", html)
        self.assertIn("deepseek-v4-pro", html)
        self.assertIn("Content quality", html)
        self.assertIn("Title repairs Top 10", html)
        self.assertIn("summary", html)
        self.assertIn("AgentWatch展示了主动式AWS监控的新进展", html)
        self.assertIn("Quality issues Top 10", html)
        self.assertIn("AgentWatchdemonstrproactive", html)
        self.assertIn("Last model refresh", html)
        self.assertIn("refresh_fallback_model_path", html)
        self.assertIn("Recent feedback", html)
        self.assertIn("useful", html)
        self.assertIn("report-1", html)
        self.assertIn("Article", html)

    def test_stats_json_endpoint_returns_machine_readable_payload(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = str(Path(temp_dir) / "ai_news.db")
            Database(db_path)

            class TestHandler(FeedbackHandler):
                pass

            TestHandler.db_path = db_path

            server = ThreadingHTTPServer(("127.0.0.1", 0), TestHandler)
            thread = Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                port = server.server_address[1]
                with urlopen(f"http://127.0.0.1:{port}/stats.json", timeout=5) as response:
                    body = response.read().decode("utf-8")
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

        self.assertIn('"feedback_count_7d"', body)
        self.assertIn('"recent_feedback"', body)

    def test_health_endpoint_returns_marker(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = str(Path(temp_dir) / "ai_news.db")
            Database(db_path)

            class TestHandler(FeedbackHandler):
                pass

            TestHandler.db_path = db_path

            server = ThreadingHTTPServer(("127.0.0.1", 0), TestHandler)
            thread = Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                port = server.server_address[1]
                with urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as response:
                    body = response.read().decode("utf-8")
                    content_type = response.headers.get("Content-Type", "")
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

        self.assertIn(HEALTH_MARKER, body)
        self.assertIn("application/json", content_type)
        self.assertIn('"status": "ok"', body)

    def test_health_marker_is_stable_ascii(self):
        self.assertEqual(HEALTH_MARKER, "WEB_AGENT_FEEDBACK_OK")

    def test_model_path_breakdown_normalizes_old_v2_label(self):
        self.assertEqual(
            normalize_model_path_breakdown({"v2": 2, "deepseek-v4-pro": 1}),
            {"fallback_v2": 2, "deepseek-v4-pro": 1},
        )

    def test_local_detail_page_contains_secondary_feedback_without_email_buttons(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = str(Path(temp_dir) / "ai_news.db")
            db = Database(db_path)
            db.insert_article(
                {
                    "title": "World model paper",
                    "url": "https://example.com/paper",
                    "source_detail": "arXiv",
                    "facts": {"method": "latent dynamics"},
                }
            )
            article = db.get_articles_for_run("", processed_only=False)[0]
            db.record_report_run("report-v8", "run-v8")
            db.record_report_items(
                "report-v8",
                [
                    {
                        "id": article["id"],
                        "report_rank": 1,
                        "report_section": "featured_papers",
                        "content_type": "paper",
                        "editorial_title": "潜空间动力学改善策略预测",
                        "paper_technical_intro": "论文先学习潜空间动力学，再用策略预测实验验证。",
                        "canonical_url": "https://example.com/paper",
                    }
                ],
            )

            page = render_feedback_detail_html(
                db_path,
                "report-v8",
                article["id"],
                recorded_signal="useful",
            )

        self.assertIn("反馈已记录", page)
        self.assertIn("论文太浅", page)
        self.assertIn("内容太长", page)
        self.assertIn("没记住重点", page)
        self.assertIn("屏蔽类似", page)
        self.assertIn("打开原文", page)


if __name__ == "__main__":
    unittest.main()
