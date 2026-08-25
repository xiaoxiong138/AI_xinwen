import tempfile
import unittest
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from src.continuity import (
    build_closing_memory,
    build_topic_dossiers,
    confidence_for_item,
    enrich_continuity,
    reading_queue_context,
)
from src.database import Database, resolve_database_path
from src.generator import ReportGenerator


def paper(article_id: int, metric: str = "成功率提升 12%"):
    return {
        "id": article_id,
        "title": "A predictive world model for robot planning",
        "title_cn": "用于机器人规划的预测世界模型",
        "editorial_title": "预测世界模型把视频表征接入机器人规划",
        "editorial_lead": "模型先预测未来状态，再由策略选择动作。",
        "url": f"https://arxiv.org/abs/{article_id}",
        "content_type": "paper",
        "domain_key": "world_model",
        "facts_cn": {
            "who": "PredictiveWM",
            "action": "提出",
            "target": "机器人规划",
            "method": "先学习潜空间动力学，再用滚动预测约束动作选择",
            "dataset_or_benchmark": "RoboSuite",
            "metric_result": metric,
            "baseline": "无预测模块的 VLA",
            "limitation": "只在仿真任务中验证",
            "evidence": [metric],
        },
        "evidence_quality": 0.8,
        "information_density": 0.8,
    }


class ContinuityTests(unittest.TestCase):
    def test_database_path_resolves_config_and_environment_override(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            configured = resolve_database_path({"database": {"path": "data/agent.db"}}, root)
            with patch.dict(os.environ, {"WEB_AGENT_DB_PATH": str(root / "override.db")}):
                overridden = resolve_database_path({"database": {"path": "ignored.db"}}, root)

        self.assertEqual(Path(configured), (root / "data" / "agent.db").resolve())
        self.assertEqual(Path(overridden), (root / "override.db").resolve())
    def test_recurring_topic_only_surfaces_new_fact_delta(self):
        previous = paper(1, "成功率提升 8%")
        current = paper(2, "成功率提升 12%")

        enriched = enrich_continuity([current], [previous])[0]

        self.assertEqual(enriched["continuity_status"], "updated")
        self.assertIn("结果新增", enriched["delta_summary"])
        self.assertIn("metric_result", enriched["new_fact_fields"])
        self.assertEqual(enriched["confidence_label"], "已有实验验证")

    def test_unchanged_recurring_item_is_marked_repeated(self):
        previous = paper(1)
        current = paper(1)

        enriched = enrich_continuity([current], [previous])[0]

        self.assertEqual(enriched["continuity_status"], "repeated")
        self.assertIn("没有出现可验证的新事实", enriched["delta_summary"])

    def test_featured_paper_gets_lineage_and_prior_work_context(self):
        related = paper(1, "成功率提升 8%")
        current = paper(2, "成功率提升 12%")

        enriched = enrich_continuity([current], [related])[0]

        self.assertIn("World Model 训练", enriched["technical_lineage"])
        self.assertIn("VLA", enriched["prior_work_context"])
        self.assertEqual(enriched["related_reading"]["url"], related["url"])

    def test_confidence_distinguishes_official_and_opinion(self):
        official = {
            "title": "OpenAI launches a new agent",
            "source_tier": "official",
            "source_detail": "OpenAI",
            "facts": {"who": "OpenAI", "action": "发布", "target": "Agent"},
        }
        opinion = {
            "title": "Yann LeCun interview on world models",
            "facts": {"who": "Yann LeCun", "action": "认为", "target": "世界模型"},
        }

        self.assertEqual(confidence_for_item(official)[0], "只有官方声明")
        self.assertEqual(confidence_for_item(opinion)[0], "观点或推测")

    def test_dossier_and_closing_memory_use_structured_facts(self):
        item = enrich_continuity([paper(2)], [paper(1, "成功率提升 8%")])[0]

        dossiers = build_topic_dossiers([item])
        memory = build_closing_memory([item])

        self.assertEqual(dossiers[0]["topic_key"], "world_model_training")
        self.assertTrue(dossiers[0]["current_routes"])
        self.assertEqual(len(memory["facts"]), 1)
        self.assertIn("机制", memory["mechanism"])

    def test_track_feedback_creates_reading_queue_and_style_weights(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db = Database(str(Path(temp_dir) / "test.db"))
            db.insert_article(
                {
                    "title": "A predictive world model for robot planning",
                    "url": "https://arxiv.org/abs/1000",
                    "source": "arxiv",
                    "source_detail": "arXiv",
                    "content_type": "paper",
                    "topic": "World Model",
                    "facts": {"who": "PredictiveWM", "method": "latent dynamics"},
                }
            )
            article = db.get_articles_for_run("", processed_only=False)[0]
            report_item = {**paper(article["id"]), "continuity_topic_key": "world_model_training"}
            db.record_report_items("report-1", [report_item])

            db.record_article_feedback("report-1", article["id"], "track")
            queue = db.get_reading_queue()
            weights = db.get_preference_weights()
            context = reading_queue_context(queue, [report_item])

        self.assertEqual(queue[0]["topic_key"], "world_model_training")
        self.assertEqual(context[0]["title"], report_item["editorial_title"])
        self.assertGreater(weights["editorial_style"]["paper_depth"], 0)

    def test_v9_html_renders_delta_dossier_and_paper_context(self):
        item = enrich_continuity([paper(2)], [paper(1, "成功率提升 8%")])[0]
        item.update(
            {
                "paper_technical_intro": "论文先学习潜空间动力学，再滚动预测未来状态；在 RoboSuite 上成功率提升 12%。",
                "evidence_line": "RoboSuite 成功率提升 12%。",
                "source_detail": "arXiv",
                "report_section": "featured_papers",
                "primary_feedback_links": {},
            }
        )
        context = {
            "topic_dossiers": build_topic_dossiers([item]),
            "reading_queue": [],
            "closing_memory": build_closing_memory([item]),
            "weekly_digest": {},
            "show_weekly_digest": False,
        }
        generator = ReportGenerator(
            design_version="v9-continuous-learning",
            report_config={"design_version": "v9-continuous-learning", "v9_context": context},
        )
        layers = {
            "must_read": [],
            "physical_ai": [],
            "watch": [],
            "featured_papers": [item],
            "paper_appendix": [],
            "brief": [],
        }

        html = generator.generate_html(
            papers=[item],
            updates=[],
            mixed_items=[item],
            report_summary={},
            layered_updates=layers,
        )

        self.assertIn("20 分钟连续学习版", html)
        self.assertIn("长期技术档案", html)
        self.assertIn("技术谱系", html)
        self.assertIn("相较上次", html)

    def test_v9_paper_index_replaces_ellipsis_title_with_safe_title(self):
        item = paper(4)
        item.update(
            {
                "title_cn": "Joint-Embedding Archit...：SIGReg 理论分析",
                "editorial_title": "Joint-Embedding Archit...：SIGReg 理论分析",
                "paper_technical_intro": "论文先学习潜空间动力学，再在基准实验中验证结果提升 12%。",
                "report_section": "paper_appendix",
            }
        )
        generator = ReportGenerator(design_version="v9-continuous-learning")

        card = generator._v8_card(item)

        self.assertNotIn("...", card["title_cn"])
        self.assertLessEqual(len(card["title_cn"]), 56)

    def test_continuity_history_keeps_sent_rerenders_and_ignores_failed_or_preview_runs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db = Database(str(Path(temp_dir) / "test.db"))
            for report_id, slot_id, article_id, quality_status, delivery_status in (
                ("sent-report", "__production__", 1, "passed", "sent"),
                ("dry-report", "__dry_run__", 2, "passed", "preview"),
                ("preview-report", "__report_only__", 3, "passed", "preview"),
                ("rerender-report", "__report_only_send__", 4, "passed", "sent"),
                ("smtp-failed-report", "__production__", 5, "passed", "failed"),
                ("blocked-report", "20260810_1300", 6, "failed", "blocked"),
                ("legacy-scheduled-report", "20260809_2100", 7, "passed", ""),
                ("sent-despite-quality-report", "20260810_2100", 8, "failed", "sent"),
            ):
                db.record_report_run(
                    report_id,
                    f"run-{article_id}",
                    slot_id=slot_id,
                    html_report_path=f"archive/{report_id}.html",
                    quality_status=quality_status,
                    delivery_status=delivery_status,
                )
                db.record_report_items(report_id, [{"id": article_id, "title": report_id, "url": f"https://example.com/{article_id}"}])

            history = db.get_recent_report_items(days=2, sent_only=True)

        self.assertEqual(
            {item["_history_report_id"] for item in history},
            {"sent-report", "rerender-report", "sent-despite-quality-report"},
        )

    def test_confirmed_sent_slots_backfill_legacy_delivery_status(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db = Database(str(Path(temp_dir) / "test.db"))
            db.record_report_run(
                "legacy-sent-report",
                "legacy-run",
                slot_id="20260809_2100",
                html_report_path="archive/legacy.html",
                quality_status="passed",
                delivery_status="",
            )
            db.record_report_items(
                "legacy-sent-report",
                [{"id": 1, "content_type": "paper", "url": "https://arxiv.org/abs/2608.00001"}],
            )

            updated = db.mark_report_runs_sent([{"run_id": "legacy-run"}])
            history = db.get_recent_report_items(days=2, sent_only=True)

        self.assertEqual(updated, 1)
        self.assertEqual({item["_history_report_id"] for item in history}, {"legacy-sent-report"})

    def test_confirmed_sent_slots_backfill_falls_back_to_html_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db = Database(str(Path(temp_dir) / "test.db"))
            db.record_report_run(
                "legacy-sent-report",
                "actual-run",
                slot_id="20260809_2100",
                html_report_path="archive/legacy.html",
                quality_status="passed",
                delivery_status="",
            )

            updated = db.mark_report_runs_sent(
                [{"run_id": "stale-run", "html_report_path": "archive/legacy.html"}]
            )
            report = db.get_report_run("legacy-sent-report")

        self.assertEqual(updated, 1)
        self.assertEqual(report["delivery_status"], "sent")
        self.assertTrue(report["delivery_at"])

    def test_recent_sent_history_uses_delivery_time_instead_of_report_creation_time(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            db = Database(str(db_path))
            db.record_report_run(
                "sent-report",
                "sent-run",
                slot_id="20260812_2100",
                html_report_path="archive/sent.html",
                quality_status="passed",
                delivery_status="pending",
            )
            db.record_report_items("sent-report", [{
                "id": 1,
                "content_type": "paper",
                "url": "https://arxiv.org/abs/2608.00001",
            }])
            connection = db._get_conn()
            connection.execute(
                "UPDATE report_items SET created_at = datetime('now', '-10 days') WHERE report_id = ?",
                ("sent-report",),
            )
            connection.commit()
            connection.close()

            db.update_report_delivery_status(
                "sent-report",
                "sent",
                datetime.now(timezone.utc).isoformat(),
            )
            items = db.get_recent_report_items(
                days=7,
                sent_only=True,
                content_type="paper",
            )

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["_history_report_id"], "sent-report")

    def test_sent_history_expires_by_delivery_time_but_latest_sent_baseline_remains(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db = Database(str(Path(temp_dir) / "test.db"))
            db.record_report_run(
                "old-sent-report",
                "old-sent-run",
                slot_id="20260801_2100",
                html_report_path="archive/old-sent.html",
                quality_status="passed",
                delivery_status="pending",
            )
            db.record_report_items("old-sent-report", [{
                "id": 1,
                "content_type": "paper",
                "url": "https://arxiv.org/abs/2608.00001",
            }])
            db.update_report_delivery_status(
                "old-sent-report",
                "sent",
                (datetime.now(timezone.utc) - timedelta(days=8)).isoformat(),
            )

            cooldown_items = db.get_recent_report_items(
                days=7,
                sent_only=True,
                content_type="paper",
            )
            adjacent_items = db.get_latest_sent_report_items(content_type="paper")

        self.assertEqual(cooldown_items, [])
        self.assertEqual(len(adjacent_items), 1)
        self.assertEqual(adjacent_items[0]["_history_report_id"], "old-sent-report")

    def test_recent_report_item_limit_applies_after_content_type_filter(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db = Database(str(Path(temp_dir) / "test.db"))
            db.record_report_run(
                "sent-report",
                "run-1",
                slot_id="20260812_1300",
                html_report_path="archive/sent.html",
                quality_status="passed",
                delivery_status="sent",
            )
            items = [
                {
                    "id": index,
                    "content_type": "news",
                    "url": f"https://example.com/news/{index}",
                    "report_rank": index,
                }
                for index in range(1, 1602)
            ]
            items.extend(
                {
                    "id": 2000 + index,
                    "content_type": "paper",
                    "url": f"https://arxiv.org/abs/2608.{index:05d}",
                    "report_rank": 2000 + index,
                }
                for index in range(3)
            )
            db.record_report_items("sent-report", items)

            papers = db.get_recent_report_items(
                days=2,
                limit=2,
                sent_only=True,
                content_type="paper",
            )

        self.assertEqual(len(papers), 2)
        self.assertTrue(all(item["content_type"] == "paper" for item in papers))

    def test_latest_sent_report_items_ignore_preview_and_failed_runs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db = Database(str(Path(temp_dir) / "test.db"))
            for report_id, delivery_status, article_id in (
                ("sent-report", "sent", 1),
                ("preview-report", "preview", 2),
                ("failed-report", "failed", 3),
            ):
                db.record_report_run(
                    report_id,
                    f"run-{article_id}",
                    slot_id=f"2026081{article_id}_2100",
                    html_report_path=f"archive/{report_id}.html",
                    quality_status="passed",
                    delivery_status=delivery_status,
                )
                db.record_report_items(report_id, [{
                    "id": article_id,
                    "content_type": "paper",
                    "url": f"https://arxiv.org/abs/2608.0000{article_id}",
                }])

            items = db.get_latest_sent_report_items(content_type="paper")

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["_history_report_id"], "sent-report")

    def test_latest_sent_report_items_can_exclude_current_report(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db = Database(str(Path(temp_dir) / "test.db"))
            for article_id, report_id in ((1, "previous-report"), (2, "current-report")):
                db.record_report_run(
                    report_id,
                    f"run-{article_id}",
                    slot_id=f"2026081{article_id}_2100",
                    html_report_path=f"archive/{report_id}.html",
                    quality_status="passed",
                    delivery_status="sent",
                )
                db.record_report_items(report_id, [{
                    "id": article_id,
                    "content_type": "paper",
                    "url": f"https://arxiv.org/abs/2608.0000{article_id}",
                }])

            items = db.get_latest_sent_report_items(
                content_type="paper",
                exclude_report_id="current-report",
            )

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["_history_report_id"], "previous-report")

    def test_latest_sent_report_items_follow_delivery_order_not_creation_order(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db = Database(str(Path(temp_dir) / "test.db"))
            for article_id, report_id in ((1, "created-first"), (2, "created-second")):
                db.record_report_run(
                    report_id,
                    f"run-{article_id}",
                    slot_id=f"2026081{article_id}_2100",
                    html_report_path=f"archive/{report_id}.html",
                    quality_status="passed",
                    delivery_status="pending",
                )
                db.record_report_items(report_id, [{
                    "id": article_id,
                    "content_type": "paper",
                    "url": f"https://arxiv.org/abs/2608.0000{article_id}",
                }])
            db.update_report_delivery_status(
                "created-second",
                "sent",
                "2026-08-12T20:00:00+08:00",
            )
            db.update_report_delivery_status(
                "created-first",
                "sent",
                "2026-08-12T21:00:00+08:00",
            )

            items = db.get_latest_sent_report_items(content_type="paper")

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["_history_report_id"], "created-first")

    def test_recent_paper_candidates_use_publish_date_not_ingest_time(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db = Database(str(Path(temp_dir) / "test.db"))
            old = {
                "title": "Old paper ingested today",
                "url": "https://arxiv.org/abs/2501.00001",
                "source": "ArXiv",
                "content_type": "paper",
                "publish_date": "2025-01-01T00:00:00+00:00",
            }
            fresh = {
                "title": "Fresh paper",
                "url": "https://arxiv.org/abs/2608.00001",
                "source": "ArXiv",
                "content_type": "paper",
                "publish_date": datetime.now(timezone.utc).isoformat(),
            }
            self.assertTrue(db.insert_article(old))
            self.assertTrue(db.insert_article(fresh))
            conn = db._get_conn()
            conn.execute("UPDATE articles SET processed = 1")
            conn.commit()
            conn.close()

            papers = db.get_recent_processed_articles(hours=72, content_type="paper", limit=10)

        self.assertEqual([item["url"] for item in papers], [fresh["url"]])


if __name__ == "__main__":
    unittest.main()
