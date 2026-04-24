import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from main import (
    apply_runtime_profile,
    build_suppressed_alert_summary,
    build_alert_summary_v2,
    build_archive_summary,
    build_archive_summary_v2,
    diversify_report_titles,
    build_trend_summary_v2,
    filter_updates_for_report,
    hydrate_paper_cache,
    is_validation_archive_entry,
    limit_papers_by_topic,
    resolve_delivery_outcome,
    should_skip_rss_feed,
    update_archive_manifest,
)
from src.database import Database
from src.generator import ReportGenerator
from src.processors.llm_processor import LLMProcessor
from src.relevance import is_low_signal_update, score_preference_boost, score_update_quality


class QualityPipelineTests(unittest.TestCase):
    def test_apply_runtime_profile_validation_fast_reduces_workload(self):
        config = {
            "sources": {
                "arxiv": {
                    "topic_limits": {"Physical AI": 5, "World Model": 5},
                    "topic_queries": {"Physical AI": "q1", "World Model": "q2"},
                    "candidate_pool": 220,
                    "fallback_days": [1, 3, 7],
                },
                "rss": {"feeds": [{"name": "a", "max_entries": 15}, {"name": "b", "max_entries": 15}, {"name": "c", "max_entries": 15}, {"name": "d", "max_entries": 15}]},
                "web_search": {"searches": [{"name": "s1", "max_results": 8}, {"name": "s2", "max_results": 8}, {"name": "s3", "max_results": 8}, {"name": "s4", "max_results": 8}]},
            },
            "report": {"paper_limit": 15, "web_limit": 20, "min_web_items": 20},
            "trends": {"enabled": True},
        }

        profiled = apply_runtime_profile(config, "validation_fast")

        self.assertEqual(list(profiled["sources"]["arxiv"]["topic_limits"].keys()), ["Physical AI"])
        self.assertEqual(profiled["sources"]["arxiv"]["topic_limits"]["Physical AI"], 2)
        self.assertEqual(profiled["sources"]["arxiv"]["fallback_days"], [1])
        self.assertEqual(len(profiled["sources"]["rss"]["feeds"]), 3)
        self.assertEqual(profiled["sources"]["rss"]["feeds"][0]["max_entries"], 5)
        self.assertEqual(len(profiled["sources"]["web_search"]["searches"]), 3)
        self.assertEqual(profiled["sources"]["web_search"]["searches"][0]["max_results"], 4)
        self.assertEqual(profiled["report"]["paper_limit"], 4)
        self.assertEqual(profiled["report"]["web_limit"], 6)
        self.assertEqual(profiled["report"]["min_web_items"], 4)
        self.assertEqual(profiled["report"]["paper_backfill_hours_ladder"], [])
        self.assertEqual(profiled["report"]["web_backfill_hours_ladder"], [])
        self.assertFalse(profiled["archive"]["enabled"])
        self.assertEqual(profiled["archive"]["report_dir"], "archive/validation")
        self.assertFalse(profiled["alerts"]["enabled"])
        self.assertFalse(profiled["alerts"]["send_separate_alert"])
        self.assertFalse(profiled["trends"]["enabled"])
        self.assertEqual(profiled["runtime"]["max_unprocessed_items"], 8)
        self.assertTrue(profiled["runtime"]["skip_paper_enrichment"])

    def test_build_suppressed_alert_summary_marks_alerts_as_intentionally_disabled(self):
        summary = build_suppressed_alert_summary("disabled_by_runtime_profile")
        self.assertFalse(summary["needs_alert"])
        self.assertEqual(summary["issues"], [])
        self.assertTrue(summary["suppressed"])
        self.assertEqual(summary["reason"], "disabled_by_runtime_profile")

    def test_report_generator_accepts_empty_trend_summary_items(self):
        generator = ReportGenerator()
        html = generator.generate_html(
            papers=[],
            updates=[],
            mixed_items=[],
            report_summary={
                "lead_summary": "总览",
                "paper_summary": "论文趋势",
                "update_summary": "动态趋势",
                "hot_topics": [],
                "key_takeaways": [],
                "watchlist": [],
            },
            collector_summary={"status_text": "ok", "fresh_items": 0, "success_count": 0, "timeout_count": 0, "failed_count": 0},
            trend_summary={"items": []},
            alert_summary={"needs_alert": False, "issues": []},
            archive_summary={"entries": []},
        )
        self.assertIn("今日总览", html)

    def test_resolve_delivery_outcome_marks_dry_run_as_successful_non_retryable(self):
        outcome = resolve_delivery_outcome(
            notification_sent=False,
            notification_skipped=False,
            notification_dry_run=True,
        )
        self.assertTrue(outcome["success"])
        self.assertEqual(outcome["status"], "dry_run")
        self.assertEqual(outcome["delivery_status"], "dry_run")
        self.assertFalse(outcome["retryable"])

    def test_resolve_delivery_outcome_marks_failed_notification_as_retryable(self):
        outcome = resolve_delivery_outcome(
            notification_sent=False,
            notification_skipped=False,
            notification_dry_run=False,
        )
        self.assertFalse(outcome["success"])
        self.assertEqual(outcome["status"], "notification_failed")
        self.assertEqual(outcome["delivery_status"], "failed")
        self.assertTrue(outcome["retryable"])

    def test_low_signal_video_is_penalized(self):
        source_preferences = {
            "blacklist_hosts": ["youtube.com"],
            "source_weights": {"youtube.com": -1.6, "nvidia.com": 2.0},
            "platform_weights": {"YouTube": -0.8, "Blog": 1.0},
            "category_weights": {"视频解读": -0.5, "基础设施": 1.0},
        }
        low_score = score_update_quality(
            title="Best Robot Mower for Large Yards? Full Review",
            content="A broad review video with tips and opinions.",
            url="https://www.youtube.com/watch?v=demo",
            platform="YouTube",
            source_detail="AI on YouTube",
            category="视频解读",
            source_preferences=source_preferences,
        )
        high_score = score_update_quality(
            title="NVIDIA launches new AI inference platform for enterprise deployment",
            content="The release focuses on inference latency, deployment cost, and datacenter adoption.",
            url="https://blogs.nvidia.com/blog/inference-platform/",
            platform="Blog",
            source_detail="NVIDIA Blog",
            category="基础设施",
            source_preferences=source_preferences,
        )
        self.assertLess(low_score, high_score)
        self.assertTrue(
            is_low_signal_update(
                title="Best Robot Mower for Large Yards? Full Review",
                content="A broad review video with tips and opinions.",
                url="https://www.youtube.com/watch?v=demo",
                platform="YouTube",
                source_detail="AI on YouTube",
                category="视频解读",
                source_preferences=source_preferences,
            )
        )

    def test_preference_boost_raises_core_topics(self):
        preference_config = {
            "content_type_weights": {"paper": 1.4},
            "category_weights": {"Physical AI": 1.0},
            "keyword_weights": {"world model": 0.8, "robotics": 0.7},
        }
        score = score_preference_boost(
            {
                "content_type": "paper",
                "topic": "Physical AI",
                "title": "A world model for robotics planning",
                "summary": "robotics world model planning",
            },
            preference_config,
        )
        self.assertGreater(score, 2.0)

    def test_filter_updates_prefers_information_density(self):
        source_preferences = {
            "whitelist_hosts": ["openai.com"],
            "blacklist_hosts": ["youtube.com"],
            "source_weights": {"openai.com": 2.5, "youtube.com": -1.6},
            "platform_weights": {"Blog": 1.0, "YouTube": -0.8},
            "category_weights": {"产品发布": 0.6, "视频解读": -0.5},
        }
        updates = [
            {
                "url": "https://www.youtube.com/watch?v=1",
                "title": "Top 10 AI tools review",
                "summary": "review and roundup",
                "platform": "YouTube",
                "source_detail": "AI on YouTube",
                "category": "视频解读",
                "score": 8.5,
            },
            {
                "url": "https://openai.com/index/new-launch",
                "title": "OpenAI launches new enterprise workflow",
                "summary": "enterprise workflow release and deployment details with customer rollout and cost impact",
                "platform": "Blog",
                "source_detail": "OpenAI Blog",
                "category": "产品发布",
                "score": 8.0,
            },
        ]
        selected = filter_updates_for_report(updates, 1, 1, source_preferences=source_preferences, preference_config={})
        self.assertEqual(selected[0]["url"], "https://openai.com/index/new-launch")

    def test_pick_top_highlights_balances_types(self):
        generator = ReportGenerator()
        items = []
        for index in range(5):
            items.append(
                {
                    "url": f"https://example.com/update-{index}",
                    "title_cn": f"动态 {index}",
                    "content_type": "news",
                    "content_kind": "全网动态",
                    "display_topic": "基础设施",
                    "score": 9 - index * 0.1,
                }
            )
        for index in range(3):
            items.append(
                {
                    "url": f"https://example.com/paper-{index}",
                    "title_cn": f"论文 {index}",
                    "content_type": "paper",
                    "content_kind": "论文",
                    "display_topic": "具身智能",
                    "score": 8.8 - index * 0.1,
                }
            )

        highlights = generator._pick_top_highlights(items)
        kinds = [item["content_type"] for item in highlights]
        self.assertGreaterEqual(kinds.count("paper"), 2)
        self.assertGreaterEqual(len(highlights), 5)

    def test_mixed_feed_avoids_long_same_type_runs(self):
        generator = ReportGenerator()
        papers = [{"url": f"https://paper/{idx}", "content_type": "paper", "score": 9 - idx * 0.1} for idx in range(15)]
        updates = [{"url": f"https://update/{idx}", "content_type": "news", "score": 9.5 - idx * 0.1} for idx in range(20)]
        mixed = generator.build_mixed_items(papers, updates)
        streak = 1
        for index in range(1, min(len(mixed), 15)):
            if mixed[index]["content_type"] == mixed[index - 1]["content_type"]:
                streak += 1
            else:
                streak = 1
            self.assertLessEqual(streak, 2)

    def test_html_report_excludes_highlight_duplicates_from_body(self):
        generator = ReportGenerator()
        papers = [
            {
                "url": f"https://paper.example/{idx}",
                "title_cn": f"论文标题 {idx}",
                "summary_preview": f"论文副标题 {idx}",
                "summary": "论文摘要",
                "why_it_matters": "值得关注",
                "why_now": "为什么现在做",
                "expected_effect": "带来什么效果",
                "future_impact": "未来影响",
                "content_type": "paper",
                "content_kind": "论文",
                "display_topic": "具身智能",
                "impact_tag": "提能力",
                "score": 9.2 - idx * 0.1,
                "publish_date": "2026-03-29 12:00:00",
            }
            for idx in range(3)
        ]
        updates = [
            {
                "url": f"https://update.example/{idx}",
                "title_cn": f"动态标题 {idx}",
                "summary_preview": f"动态副标题 {idx}",
                "summary": "动态摘要",
                "why_it_matters": "值得关注",
                "why_now": "为什么现在做",
                "expected_effect": "带来什么效果",
                "future_impact": "未来影响",
                "content_type": "news",
                "content_kind": "全网动态",
                "display_topic": "基础设施",
                "impact_tag": "抢算力",
                "score": 9.5 - idx * 0.1,
                "publish_date": "2026-03-29 12:00:00",
            }
            for idx in range(8)
        ]
        mixed = generator.build_mixed_items(papers, updates)
        html = generator.generate_html(
            papers=papers,
            updates=updates,
            mixed_items=mixed,
            report_summary={
                "lead_summary": "总览",
                "paper_summary": "论文趋势",
                "update_summary": "动态趋势",
                "hot_topics": ["世界模型"],
                "key_takeaways": ["结论"],
                "watchlist": ["观察"],
            },
            trend_summary={"items": []},
            alert_summary={"needs_alert": False, "issues": []},
            archive_summary={"entries": []},
        )
        self.assertIn("今日最重要 5 条", html)
        self.assertIn("顶部只保留重点摘要", html)

    def test_limit_papers_by_topic_respects_caps(self):
        papers = [
            {"url": f"https://paper/{idx}", "topic": "Physical AI", "score": 9 - idx * 0.1, "publish_date": "2026-03-29"}
            for idx in range(5)
        ]
        papers += [
            {"url": f"https://paper/r{idx}", "topic": "Robotics", "score": 8 - idx * 0.1, "publish_date": "2026-03-29"}
            for idx in range(5)
        ]
        limited = limit_papers_by_topic(papers, {"Physical AI": 2, "Robotics": 3}, 5)
        physical_count = sum(1 for item in limited if item["topic"] == "Physical AI")
        robotics_count = sum(1 for item in limited if item["topic"] == "Robotics")
        self.assertEqual(len(limited), 5)
        self.assertEqual(physical_count, 2)
        self.assertEqual(robotics_count, 3)

    def test_build_trend_summary_requires_multi_day_or_multi_source_validation(self):
        recent_articles = [
            {"topic_cn": "世界模型", "keywords": "世界模型,规划", "source_detail": "OpenAI Blog", "url": "https://openai.com/a", "publish_date": "2026-03-28T10:00:00"},
            {"topic_cn": "世界模型", "keywords": "世界模型,视频生成", "source_detail": "TechCrunch AI", "url": "https://techcrunch.com/b", "publish_date": "2026-03-29T10:00:00"},
            {"topic_cn": "基础设施", "keywords": "GPU,数据中心", "source_detail": "NVIDIA Blog", "url": "https://nvidia.com/c", "publish_date": "2026-03-29T12:00:00"},
            {"topic_cn": "基础设施", "keywords": "GPU,推理", "source_detail": "NVIDIA Blog", "url": "https://nvidia.com/d", "publish_date": "2026-03-29T15:00:00"},
        ]
        trend_summary = build_trend_summary_v2(recent_articles, lookback_days=3, max_items=3, min_occurrences=2)
        labels = [item["label"] for item in trend_summary["items"]]
        self.assertIn("世界模型", labels)
        self.assertNotIn("基础设施", labels)

    def test_build_alert_summary_flags_quality_risks(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db = Database(str(Path(temp_dir) / "test.db"))
            db.record_collector_runs(
                "run1",
                [{"label": "ArxivCollector[Physical AI]", "status": "error", "inserted_count": 0, "collected_count": 0, "duration_seconds": 1.0, "error": "boom"}],
            )
            summary = build_alert_summary_v2(
                [{"label": "ArxivCollector[World Model]", "status": "timeout", "inserted_count": 0, "collected_count": 0, "duration_seconds": 1.0, "error": "timeout"}],
                db,
                updates_count=16,
                paper_count=10,
                update_candidate_count=50,
                alert_config={"arxiv_failure_threshold": 1, "min_update_count": 20, "min_paper_count": 12, "duplicate_ratio_threshold": 0.5},
            )
            self.assertTrue(summary["needs_alert"])
            joined = "\n".join(summary["issues"])
            self.assertIn("Arxiv", joined)
            self.assertIn("去重后全网动态仅保留 16 条", joined)
            self.assertIn("最终论文仅保留 10 篇", joined)

    def test_should_skip_rss_feed_after_repeated_failures(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db = Database(str(Path(temp_dir) / "test.db"))
            db.record_collector_runs(
                "run1",
                [{"label": "RSSCollector[Test Feed]", "status": "error", "inserted_count": 0, "collected_count": 0, "duration_seconds": 1.0, "error": "boom"}],
            )
            db.record_collector_runs(
                "run2",
                [{"label": "RSSCollector[Test Feed]", "status": "timeout", "inserted_count": 0, "collected_count": 0, "duration_seconds": 1.0, "error": "timeout"}],
            )
            self.assertTrue(
                should_skip_rss_feed(
                    db,
                    "Test Feed",
                    {"enabled": True, "failure_threshold": 2, "lookback_runs": 6},
                )
            )

    def test_should_skip_rss_feed_allows_recovery_probe_after_cooldown(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db = Database(str(Path(temp_dir) / "test.db"))
            db.record_collector_runs(
                "run1",
                [{"label": "RSSCollector[Test Feed]", "status": "error", "inserted_count": 0, "collected_count": 0, "duration_seconds": 1.0, "error": "boom"}],
            )
            db.record_collector_runs(
                "run2",
                [{"label": "RSSCollector[Test Feed]", "status": "timeout", "inserted_count": 0, "collected_count": 0, "duration_seconds": 1.0, "error": "timeout"}],
            )
            old_timestamp = (datetime.now(timezone.utc) - timedelta(hours=13)).strftime("%Y-%m-%d %H:%M:%S")
            conn = db._get_conn()
            try:
                conn.execute("UPDATE collector_runs SET created_at = ?", (old_timestamp,))
                conn.commit()
            finally:
                conn.close()

            self.assertFalse(
                should_skip_rss_feed(
                    db,
                    "Test Feed",
                    {"enabled": True, "failure_threshold": 2, "lookback_runs": 6, "recovery_interval_hours": 12},
                )
            )

    def test_should_skip_rss_feed_ignores_skipped_rows_for_recovery_clock(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db = Database(str(Path(temp_dir) / "test.db"))
            db.record_collector_runs(
                "run1",
                [{"label": "RSSCollector[Test Feed]", "status": "error", "inserted_count": 0, "collected_count": 0, "duration_seconds": 1.0, "error": "boom"}],
            )
            db.record_collector_runs(
                "run2",
                [{"label": "RSSCollector[Test Feed]", "status": "timeout", "inserted_count": 0, "collected_count": 0, "duration_seconds": 1.0, "error": "timeout"}],
            )
            old_timestamp = (datetime.now(timezone.utc) - timedelta(hours=13)).strftime("%Y-%m-%d %H:%M:%S")
            conn = db._get_conn()
            try:
                conn.execute("UPDATE collector_runs SET created_at = ?", (old_timestamp,))
                conn.commit()
            finally:
                conn.close()
            db.record_collector_runs(
                "run3",
                [{"label": "RSSCollector[Test Feed]", "status": "skipped", "inserted_count": 0, "collected_count": 0, "duration_seconds": 0, "error": "cooldown"}],
            )

            self.assertFalse(
                should_skip_rss_feed(
                    db,
                    "Test Feed",
                    {"enabled": True, "failure_threshold": 2, "lookback_runs": 6, "recovery_interval_hours": 12},
                )
            )

    def test_hydrate_paper_cache_reuses_cached_abstract(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db = Database(str(Path(temp_dir) / "test.db"))
            db.upsert_paper_enrichment_cache(
                [{"url": "https://paper.example/1", "content": "cached abstract", "author": "A", "publish_date": "2026-03-29"}]
            )
            hydrated, missing = hydrate_paper_cache(
                [
                    {"url": "https://paper.example/1", "content": "short", "author": "", "publish_date": ""},
                    {"url": "https://paper.example/2", "content": "short2", "author": "", "publish_date": ""},
                ],
                db,
                168,
            )
            self.assertEqual(len(hydrated), 1)
            self.assertEqual(hydrated[0]["content"], "cached abstract")
            self.assertEqual(len(missing), 1)

    def test_archive_summary_supports_topics_and_sources_search(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            previous = Path.cwd()
            try:
                os.chdir(root)
                update_archive_manifest(
                    html_filename="report_20260329_1200.html",
                    markdown_filename="report_20260329_1200.md",
                    report_summary={"hot_topics": ["世界模型", "机器人"]},
                    papers=[{"url": "https://paper.example/1"}],
                    updates=[{"source_detail": "OpenAI Blog"}, {"source_detail": "TechCrunch AI"}],
                )
                summary = build_archive_summary_v2("reports_index.html", "reports_index.md")
            finally:
                os.chdir(previous)
            self.assertTrue((root / "reports_index.html").exists())
            html = (root / "reports_index.html").read_text(encoding="utf-8")
            self.assertIn("世界模型", html)
            self.assertIn("OpenAI Blog", html)
            self.assertGreaterEqual(len(summary["entries"]), 1)

    def test_archive_summary_renders_utf8_chinese_labels(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            previous = Path.cwd()
            try:
                os.chdir(root)
                update_archive_manifest(
                    html_filename="report_20260405_1756.html",
                    markdown_filename="report_20260405_1756.md",
                    report_summary={"hot_topics": ["\u4e16\u754c\u6a21\u578b", "\u7269\u7406\u4eba\u5de5\u667a\u80fd"]},
                    papers=[{"url": "https://paper.example/1"}],
                    updates=[{"source_detail": "OpenAI Blog"}, {"source_detail": "TechCrunch AI"}],
                )
                build_archive_summary_v2("reports_index.html", "reports_index.md")
            finally:
                os.chdir(previous)

            html = (root / "reports_index.html").read_text(encoding="utf-8")
            markdown = (root / "reports_index.md").read_text(encoding="utf-8")
            manifest = (root / "reports_manifest.json").read_text(encoding="utf-8")

            self.assertIn("\u62a5\u544a\u5f52\u6863", html)
            self.assertIn("\u4e3b\u9898\uff1a\u4e16\u754c\u6a21\u578b", html)
            self.assertIn("\u6765\u6e90\uff1aOpenAI Blog", html)
            self.assertIn("\u641c\u7d22\u65e5\u671f\u3001\u4e3b\u9898\u6216\u6765\u6e90", html)
            self.assertIn("\u62a5\u544a\u5f52\u6863", markdown)
            self.assertIn("\u4e16\u754c\u6a21\u578b", manifest)

    def test_archive_summary_fallback_scans_archive_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            archive = root / "archive"
            archive.mkdir()
            (archive / "report_20260405_1756.html").write_text("<html></html>", encoding="utf-8")
            (archive / "report_20260405_1756.md").write_text("# report", encoding="utf-8")
            previous = Path.cwd()
            try:
                os.chdir(root)
                summary = build_archive_summary("reports_index.html", "reports_index.md", report_dir="archive")
            finally:
                os.chdir(previous)

            self.assertEqual(summary["entries"][0]["html_path"], "archive/report_20260405_1756.html")
            self.assertEqual(summary["entries"][0]["markdown_path"], "archive/report_20260405_1756.md")
            html = (root / "reports_index.html").read_text(encoding="utf-8")
            self.assertIn("archive/report_20260405_1756.html", html)

    def test_archive_summary_v2_filters_validation_entries_from_manifest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest_path = root / "reports_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "entries": [
                            {
                                "label": "validation",
                                "html_path": "archive/validation/report_validation.html",
                                "markdown_path": "archive/validation/report_validation.md",
                                "topics": ["验证"],
                                "sources": ["Validation"],
                            },
                            {
                                "label": "production",
                                "html_path": "archive/report_20260414_1200.html",
                                "markdown_path": "archive/report_20260414_1200.md",
                                "topics": ["正式"],
                                "sources": ["OpenAI Blog"],
                            },
                        ]
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            previous = Path.cwd()
            try:
                os.chdir(root)
                summary = build_archive_summary_v2("reports_index.html", "reports_index.md")
            finally:
                os.chdir(previous)

            self.assertEqual(len(summary["entries"]), 1)
            self.assertEqual(summary["entries"][0]["label"], "production")
            html = (root / "reports_index.html").read_text(encoding="utf-8")
            self.assertIn("archive/report_20260414_1200.html", html)
            self.assertNotIn("archive/validation/report_validation.html", html)

    def test_is_validation_archive_entry_detects_validation_directory(self):
        self.assertTrue(
            is_validation_archive_entry(
                {"html_path": "archive/validation/report_1.html", "markdown_path": "archive/validation/report_1.md"}
            )
        )
        self.assertFalse(
            is_validation_archive_entry(
                {"html_path": "archive/report_1.html", "markdown_path": "archive/report_1.md"}
            )
        )


    def test_editorialized_paper_title_and_preview_are_analysis_oriented(self):
        processor = LLMProcessor({"api_key_env": "THIS_KEY_SHOULD_NOT_EXIST"})
        article = {
            "content_type": "paper",
            "title": "Latent world model improves embodied planning with lower rollout cost",
            "topic": "World Model",
        }
        result = processor.prepare_report_item(
            article,
            {
                "title_cn": "世界模型新进展",
                "summary": "这项研究围绕世界模型中的规划成本问题提出了潜空间建模方法。作者希望先在压缩后的状态空间完成预测和决策，再把结果映射回真实动作，从而减少长链路试错。实验显示它在规划质量和推理效率之间取得了更稳的平衡。更重要的是，这条路线有机会把世界模型从演示能力推进到真实控制任务。",
                "category": "World Model",
                "topic_cn": "世界模型",
                "score": 8.9,
            },
        )
        self.assertIn("世界模型", result["title_cn"])
        self.assertTrue(any(token in result["title_cn"] for token in ["预测", "行动", "试错", "部署"]))
        self.assertIn("试错成本", result["summary_preview"])

    def test_editorialized_product_title_and_preview_use_product_style(self):
        processor = LLMProcessor({"api_key_env": "THIS_KEY_SHOULD_NOT_EXIST"})
        article = {
            "content_type": "news",
            "title": "OpenAI launches workflow agent for enterprise task execution",
            "source_detail": "OpenAI Blog",
            "platform": "Blog",
        }
        result = processor.prepare_report_item(
            article,
            {
                "title_cn": "OpenAI有新动作",
                "summary": "OpenAI 把新的 agent 能力推进到企业工作流执行场景，重点不是单次问答，而是让模型接管更多跨工具步骤。它支持在更长任务链路里调用系统、整理上下文并持续完成操作，这意味着产品正在从聊天入口走向执行入口。对企业用户来说，真正重要的是它是否能减少人工切换和流程摩擦。后续还要看这种能力会不会迅速成为行业默认配置。",
                "category": "产品发布",
                "topic_cn": "产品发布",
                "score": 8.6,
            },
        )
        self.assertTrue(any(token in result["title_cn"] for token in ["工作流", "能力", "推进", "切入"]))
        self.assertIn("高频人工步骤", result["summary_preview"])


    def test_editorialized_paper_title_and_preview_are_analysis_oriented(self):
        processor = LLMProcessor({"api_key_env": "THIS_KEY_SHOULD_NOT_EXIST"})
        article = {
            "content_type": "paper",
            "title": "Latent world model improves embodied planning with lower rollout cost",
            "topic": "World Model",
        }
        result = processor.prepare_report_item(
            article,
            {
                "title_cn": "old short title",
                "summary": "This work proposes a latent-space world model for planning. It aims to complete prediction and decision making in a compressed state space before mapping back to real actions, reducing long-horizon trial and error. Experiments show a better balance between planning quality and inference efficiency. The bigger implication is that world models may move from demos toward real control tasks.",
                "category": "World Model",
                "topic_cn": "World Model",
                "score": 8.9,
            },
        )
        self.assertNotEqual(result["title_cn"], "old short title")
        self.assertGreaterEqual(len(result["title_cn"]), 10)
        self.assertNotEqual(result["summary_preview"], "")
        self.assertNotIn("关键瓶颈", result["title_cn"])

    def test_editorialized_product_title_and_preview_use_product_style(self):
        processor = LLMProcessor({"api_key_env": "THIS_KEY_SHOULD_NOT_EXIST"})
        article = {
            "content_type": "news",
            "title": "OpenAI launches workflow agent for enterprise task execution",
            "source_detail": "OpenAI Blog",
            "platform": "Blog",
        }
        result = processor.prepare_report_item(
            article,
            {
                "title_cn": "old update title",
                "summary": "OpenAI is moving new agent capabilities into enterprise workflow execution. The point is not another chat feature, but letting the model take over more multi-tool steps across a longer task chain. For enterprise users, the real question is whether this reduces manual switching and process friction. The next thing to watch is whether this quickly becomes a default industry configuration.",
                "category": "Product Release",
                "topic_cn": "Product Release",
                "score": 8.6,
            },
        )
        self.assertNotEqual(result["title_cn"], "old update title")
        self.assertGreaterEqual(len(result["title_cn"]), 10)
        self.assertNotEqual(result["summary_preview"], "")
        self.assertNotIn("新动作", result["title_cn"])

    def test_diversify_report_titles_rewrites_repeated_templates(self):
        processor = LLMProcessor({"api_key_env": "THIS_KEY_SHOULD_NOT_EXIST"})
        items = [
            {
                "url": "https://example.com/1",
                "title": "OpenAI launches workflow agent for enterprise task execution",
                "title_cn": "OpenAI把AI能力推进到真实工作流",
                "summary_preview": "企业任务链开始被模型接管更多执行环节。",
                "summary": "OpenAI is moving new agent capabilities into enterprise workflow execution. The real point is reducing manual switching across a longer task chain.",
                "display_topic": "产品发布",
                "source_detail": "OpenAI Blog",
                "platform": "Blog",
                "keywords": ["企业工作流", "任务执行"],
            },
            {
                "url": "https://example.com/2",
                "title": "Anthropic expands agent workflow support for enterprise apps",
                "title_cn": "Anthropic把AI能力推进到真实工作流",
                "summary_preview": "重点开始落到企业应用里的多工具协同执行。",
                "summary": "Anthropic is extending agent support into enterprise applications. The more important signal is multi-tool execution inside real software workflows.",
                "display_topic": "产品发布",
                "source_detail": "Anthropic",
                "platform": "Blog",
                "keywords": ["企业应用", "多工具协同"],
            },
        ]
        diversified = diversify_report_titles(items, processor)
        self.assertEqual(diversified[0]["title_cn"], "OpenAI把AI能力推进到真实工作流")
        self.assertNotEqual(diversified[1]["title_cn"], "Anthropic把AI能力推进到真实工作流")
        self.assertIn("企业应用", diversified[1]["title_cn"] + diversified[1].get("summary_preview", ""))

    def test_diversify_report_titles_keeps_distinct_titles(self):
        processor = LLMProcessor({"api_key_env": "THIS_KEY_SHOULD_NOT_EXIST"})
        items = [
            {
                "url": "https://example.com/p1",
                "title_cn": "世界模型研究尝试降低试错成本",
                "summary_preview": "核心看点是先预测再行动的规划效率。",
                "summary": "This work focuses on lowering rollout cost for planning.",
                "display_topic": "世界模型",
                "keywords": ["规划效率"],
            },
            {
                "url": "https://example.com/p2",
                "title_cn": "机器人研究尝试走向更稳定实机",
                "summary_preview": "核心看点是实机任务成功率更稳定。",
                "summary": "This work focuses on real-world robot execution stability.",
                "display_topic": "机器人",
                "keywords": ["实机稳定性"],
            },
        ]
        diversified = diversify_report_titles(items, processor)
        self.assertEqual(diversified[0]["title_cn"], "世界模型研究尝试降低试错成本")
        self.assertEqual(diversified[1]["title_cn"], "机器人研究尝试走向更稳定实机")


    def test_product_release_analysis_focuses_on_workflow_execution(self):
        processor = LLMProcessor({"api_key_env": "THIS_KEY_SHOULD_NOT_EXIST"})
        article = {
            "content_type": "news",
            "title": "OpenAI launches workflow agent for enterprise task execution",
            "source_detail": "OpenAI Blog",
            "platform": "Blog",
        }
        result = processor.prepare_report_item(
            article,
            {
                "title_cn": "workflow release",
                "summary": "OpenAI is moving new agent capabilities into enterprise workflow execution. The point is not another chat feature, but letting the model take over more multi-tool steps across a longer task chain. For enterprise users, the real question is whether this reduces manual switching and process friction. The next thing to watch is whether this quickly becomes a default industry configuration.",
                "category": "Product Release",
                "topic_cn": "Product Release",
                "score": 8.6,
            },
        )
        self.assertTrue(any(token in result["why_it_matters"] for token in ["工作流", "入口", "模型能力"]))
        self.assertTrue(any(token in result["expected_effect"] for token in ["人工", "自动化", "流程"]))

    def test_partnership_analysis_focuses_on_channel_and_delivery(self):
        processor = LLMProcessor({"api_key_env": "THIS_KEY_SHOULD_NOT_EXIST"})
        article = {
            "content_type": "news",
            "title": "NVIDIA and Marvell deepen AI infrastructure partnership",
            "source_detail": "Industry Media",
            "platform": "Web",
        }
        result = processor.prepare_report_item(
            article,
            {
                "title_cn": "partnership update",
                "summary": "The two companies are expanding cooperation around AI infrastructure and delivery. The move is meant to connect hardware supply, customer access and deployment capability more tightly, rather than staying at a headline level. The practical question is whether the partnership can shorten delivery and speed up enterprise adoption.",
                "category": "Partnership",
                "topic_cn": "Partnership",
                "score": 8.4,
            },
        )
        self.assertTrue(any(token in result["why_it_matters"] for token in ["客户", "渠道", "交付"]))
        self.assertTrue(any(token in result["future_impact"] for token in ["生态", "标准", "渠道", "排序"]))


    def test_partnership_analysis_focuses_on_channel_and_delivery(self):
        processor = LLMProcessor({"api_key_env": "THIS_KEY_SHOULD_NOT_EXIST"})
        article = {
            "content_type": "news",
            "title": "NVIDIA and Marvell deepen AI infrastructure partnership",
            "source_detail": "Industry Media",
            "platform": "Web",
        }
        result = processor.prepare_report_item(
            article,
            {
                "title_cn": "partnership update",
                "summary": "The two companies are expanding cooperation around AI infrastructure and delivery. The move is meant to connect hardware supply, customer access and deployment capability more tightly, rather than staying at a headline level. The practical question is whether the partnership can shorten delivery and speed up enterprise adoption.",
                "category": "Partnership",
                "topic_cn": "Partnership",
                "score": 8.4,
            },
        )
        self.assertNotEqual(result["why_it_matters"], "")
        self.assertNotEqual(result["future_impact"], "")
        self.assertNotIn("headline level", result["why_it_matters"])

    def test_top_highlights_include_reason_block(self):
        generator = ReportGenerator()
        mixed = [
            {
                "url": "https://example.com/update-1",
                "title_cn": "产品把能力推进到工作流",
                "summary_preview": "产品开始切入更深的执行入口。",
                "summary": "摘要",
                "why_it_matters": "产品发布最值得看的不是功能名，而是它有没有把模型能力推进到更深的真实工作流和付费入口。",
                "why_now": "厂商需要证明模型不只会回答问题。",
                "expected_effect": "它会先减少人工切换和流程摩擦。",
                "future_impact": "竞争会扩展到工作流入口和集成深度。",
                "content_type": "news",
                "content_kind": "全网动态",
                "display_topic": "产品发布",
                "impact_tag": "提效率",
                "score": 9.8,
                "publish_date": "2026-04-01 00:00:00",
            },
            {
                "url": "https://example.com/paper-1",
                "title_cn": "世界模型研究尝试降低试错成本",
                "summary_preview": "重点不只是生成效果，而是能否把试错成本压到更低。",
                "summary": "摘要",
                "why_it_matters": "这类研究决定世界模型能不能从会生成画面，继续走到会支撑规划、控制和低试错决策。",
                "why_now": "世界模型已经不缺演示效果。",
                "expected_effect": "它会先影响模型是否能先预测再行动。",
                "future_impact": "会更快进入机器人和复杂任务规划的中间层栈。",
                "content_type": "paper",
                "content_kind": "论文",
                "display_topic": "世界模型",
                "impact_tag": "降成本",
                "score": 9.6,
                "publish_date": "2026-04-01 00:00:00",
            },
        ] * 3
        html = generator.generate_html(
            papers=[item for item in mixed if item["content_type"] == "paper"][:15],
            updates=[item for item in mixed if item["content_type"] != "paper"][:20],
            mixed_items=mixed[:10],
            report_summary={
                "lead_summary": "总览",
                "paper_summary": "论文趋势",
                "update_summary": "动态趋势",
                "hot_topics": ["世界模型"],
                "key_takeaways": ["结论"],
                "watchlist": ["观察"],
            },
            trend_summary={"items": []},
            alert_summary={"needs_alert": False, "issues": []},
            archive_summary={"entries": []},
        )
        self.assertIn("先看理由", html)


    def test_feed_cards_include_quick_reason_block(self):
        generator = ReportGenerator()
        items = [
            {
                "url": "https://example.com/update-1",
                "title_cn": "产品把能力推进到工作流",
                "summary_preview": "产品开始切入更深的执行入口。",
                "summary": "摘要",
                "why_it_matters": "产品发布最值得看的不是功能名，而是它有没有把模型能力推进到更深的真实工作流和付费入口。",
                "why_now": "厂商需要证明模型不只会回答问题。",
                "expected_effect": "它会先减少人工切换和流程摩擦。",
                "future_impact": "竞争会扩展到工作流入口和集成深度。",
                "content_type": "news",
                "content_kind": "全网动态",
                "display_topic": "产品发布",
                "impact_tag": "提效率",
                "score": 9.8,
                "publish_date": "2026-04-01 00:00:00",
            }
        ]
        decorated = generator._decorate_items(items)
        self.assertEqual(len(decorated), 1)
        self.assertNotEqual(decorated[0].get("quick_reason", ""), "")


if __name__ == "__main__":
    unittest.main()
