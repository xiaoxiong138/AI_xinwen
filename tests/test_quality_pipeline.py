import hashlib
import json
import os
import tempfile
import unittest
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock, patch

import yaml

from main import (
    apply_environment_path_overrides,
    apply_runtime_profile,
    apply_source_health_adjustments,
    apply_v8_reading_budget,
    classify_report_layers,
    compact_email_html,
    evaluate_report_quality,
    paper_core_summary_passes,
    build_collector_failure_record,
    build_collector_summary,
    build_v11_email_volume_layers,
    build_report_structure_diagnostics,
    build_source_grounded_news_brief,
    build_quality_diagnostics,
    build_source_health_summary,
    build_source_health_weights,
    build_suppressed_alert_summary,
    build_alert_summary_v2,
    build_archive_summary,
    build_archive_summary_v2,
    diversify_report_titles,
    build_trend_summary_v2,
    build_content_quality_counts,
    filter_updates_for_report,
    flatten_report_layers,
    hydrate_paper_cache,
    item_quality_flags,
    is_learning_digest_design,
    is_validation_archive_entry,
    limit_papers_by_topic,
    parse_hour_ladder,
    prepare_v11_email_delivery_volumes,
    resolve_delivery_outcome,
    should_block_report_send,
    should_persist_collector_history,
    record_email_commit,
    report_send_blocking_reasons,
    report_primary_section,
    report_source_key,
    select_v11_update_candidates,
    repair_title_fact_mismatch,
    repair_bad_titles_in_layers,
    should_skip_rss_feed,
    should_skip_empty_collector,
    scan_final_html_quality,
    scan_v11_delivery_volume_fidelity,
    send_email_delivery_volumes,
    suggest_repaired_title,
    suggest_repaired_title_with_source,
    suppress_repeated_analysis_fields,
    title_looks_bad,
    title_fact_mismatch,
    update_archive_manifest,
    v8_backfill_pool_ready,
    model_path_breakdown,
    learning_domain_key,
    paper_quota_domain,
    paper_substantive_description_fail_count,
)
from src.database import Database
from src.generator import ReportGenerator, editorial_item_render_key
from src.processors.llm_processor import LLMProcessor
from src.collectors.rss_collector import RSSCollector
from src.collectors.web_search_collector import WebSearchCollector
from src.editorial_engine import (
    assess_fact_publishability,
    build_editorial_quality_metrics,
    enrich_editorial_fields,
    has_untranslated_prose,
    paper_plain_summary_passes,
    paper_technical_intro_passes,
)
from src.relevance import infer_source_tier, is_low_signal_update, score_preference_boost, score_update_quality


class QualityPipelineTests(unittest.TestCase):
    def test_source_tier_uses_resolved_host_before_model_supplied_platform(self):
        self.assertEqual(
            infer_source_tier({
                "url": "https://techcrunch.com/2026/09/20/example",
                "platform": "Website",
                "content_type": "project",
            }),
            "media",
        )
        self.assertEqual(
            infer_source_tier({
                "url": "https://github.com/example/project/releases/tag/v1",
                "platform": "News",
                "content_type": "project",
            }),
            "official",
        )
        self.assertEqual(
            infer_source_tier({
                "url": "https://example.medium.com/original-engineering-note",
                "platform": "Blog",
                "content_type": "project",
            }),
            "primary",
        )
        self.assertEqual(
            infer_source_tier({
                "url": "https://news.google.com/rss/articles/example",
                "platform": "Website",
                "content_type": "project",
            }),
            "aggregator",
        )

    def test_v11_design_version_keeps_learning_digest_quality_path(self):
        config = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))

        self.assertEqual(config["report"]["design_version"], "v11-editorial-library")
        self.assertTrue(is_learning_digest_design(config["report"]))
        self.assertTrue(is_learning_digest_design("v10-learning-digest"))
        generator = ReportGenerator(
            design_version="v11-editorial-library",
            report_config=config["report"],
        )
        self.assertTrue(generator._is_v8_reader())
        self.assertTrue(generator._is_v10_reader())
        self.assertTrue(generator._is_v11_product())
        html = generator.generate_html(
            papers=[],
            updates=[],
            mixed_items=[],
            report_summary={},
            layered_updates={
                section: []
                for section in (
                    "must_read",
                    "physical_ai",
                    "watch",
                    "featured_papers",
                    "paper_appendix",
                    "research",
                    "brief",
                )
            },
        )
        self.assertIn('data-design-version="v11-editorial-library"', html)
        self.assertIn('.container[data-design-version="v11-editorial-library"]', html)

    def test_v11_real_acceptance_samples_keep_approved_editorial_copy(self):
        fixture_path = (
            Path(__file__).parent / "fixtures" / "v11_real_acceptance_samples.json"
        )
        payload = json.loads(fixture_path.read_text(encoding="utf-8"))
        forbidden_phrases = (
            "值得关注",
            "未来可能带来影响",
            "出现了新的动作",
            "相关机构",
            "学习重点是",
            "技术上，它主要围绕",
            "需要观察",
            "是否真正",
        )

        self.assertEqual(len(payload["items"]), 4)
        self.assertEqual(
            {item["content_type"] for item in payload["items"]},
            {"news", "podcast", "project", "paper"},
        )
        for raw in payload["items"]:
            item = {
                **raw,
                "model_used": "codex-automation",
                "analysis_version": "codex-research-v2",
                "evidence_quality": 0.85,
                "information_density": 0.85,
            }
            decorated = enrich_editorial_fields(item)
            self.assertEqual(decorated["title_cn"], raw["title_cn"])
            self.assertTrue(decorated["_codex_research_validated"])
            self.assertEqual(decorated["summary_quality_tier"], "editorial_ready")
            self.assertFalse(title_looks_bad(decorated))
            self.assertFalse(has_untranslated_prose(raw["summary"]))
            self.assertTrue(raw["source_excerpt"])
            self.assertTrue(raw["evidence_locator"])
            self.assertFalse(any(phrase in raw["summary"] for phrase in forbidden_phrases))

            if raw["content_type"] == "paper":
                self.assertEqual(
                    decorated["paper_plain_summary"], raw["paper_plain_summary"]
                )
                self.assertEqual(
                    decorated["paper_technical_intro"], raw["paper_technical_intro"]
                )
                self.assertTrue(paper_plain_summary_passes(raw["paper_plain_summary"]))
                self.assertTrue(
                    paper_technical_intro_passes(raw["paper_technical_intro"])
                )
            else:
                self.assertEqual(decorated["analysis_body"], raw["summary"])
                minimum_chars = 300 if raw["content_type"] == "podcast" else 220 if raw["primary_section"] == "technical" else 180
                self.assertGreaterEqual(len(raw["summary"]), minimum_chars)
                if raw["content_type"] == "podcast":
                    self.assertRegex(raw["evidence_locator"], r"文字稿|\d{1,2}:\d{2}")

    def test_paper_quality_accepts_decision_learning_and_controlled_evaluation_copy(self):
        plain_summary = (
            "StageGuard 把机器人长任务中的下一步是否开始单独交给阶段判断模型。"
            "它从示范和教师推理中学习当前操作是否完成，再决定继续、推进或跳过子任务。"
            "这样底层动作策略不用同时承担整段任务的进度管理，失误也能定位到动作执行或阶段判断。"
        )
        controlled_evaluation_intro = (
            "框架结合正负控制、重复同一迭代的等预算比较和训练深度干预，分别衡量块应用次数、"
            "不同计算内容及读出分布的作用。论文用 5 项模型配置展开分析，并强调重复迭代的严格比较"
            "仅适用于共享权重架构；普通深度截断作为参照会混入多种效应。现有配置还在参数、分词器"
            "和语料方案上不同，因此观察到的差异不能直接归因为训练方式。作者没有完成所有架构下的"
            "同预算完整重训，结论主要是评估方法与混杂因素诊断。"
        )

        self.assertTrue(paper_plain_summary_passes(plain_summary))
        self.assertTrue(
            paper_technical_intro_passes(controlled_evaluation_intro)
        )

    def test_v11_update_selection_keeps_independent_news_and_technical_buffers(self):
        candidates = [
            {
                "content_type": "news",
                "primary_section": "news",
                "url": f"https://example.com/news/{index}",
            }
            for index in range(30)
        ] + [
            {
                "content_type": "project",
                "primary_section": "technical",
                "url": f"https://example.com/technical/{index}",
            }
            for index in range(27)
        ]

        selected = select_v11_update_candidates(
            candidates,
            {
                "news_section_limit": 24,
                "technical_section_limit": 24,
                "min_visible_news_count": 20,
                "min_visible_technical_count": 20,
                "web_limit": 50,
            },
        )

        self.assertEqual(sum(report_primary_section(item) == "news" for item in selected), 30)
        self.assertEqual(sum(report_primary_section(item) == "technical" for item in selected), 27)
        self.assertEqual(len(selected), 57)

    def test_v11_reading_budget_rejects_old_contract_items_and_backfills_each_section(self):
        def build_item(section, index, *, valid):
            unique_token = chr(0x4E00 + index + (100 if section == "technical" else 0))
            body = (
                "团队说明了具体模块、输入输出、验证条件、实验结果和适用边界。"
                * (14 if section == "technical" else 12)
            )
            return {
                "content_type": "project" if section == "technical" else "news",
                "primary_section": section,
                "model_used": "codex-automation",
                "analysis_version": "codex-research-v2" if valid else "codex-research-v1",
                "publish_date": "2026-09-20",
                "claim_type": "official_claim",
                "source_excerpt": "原始材料列出了具体模块、测试条件和验证结果。",
                "evidence_locator": "官方技术文档第 2 节",
                "title_cn": f"测试候选{unique_token}公布完整验证结果",
                "url": f"https://example.com/{section}/{index}",
                "summary": body,
                "analysis_body": body,
                "score": 100 - index,
                "evidence_quality": 0.8,
                "information_density": 0.8,
                "facts": {
                    "primary_section": section,
                    "claim_type": "official_claim",
                    "who": "测试团队",
                    "action": "发布",
                    "target": f"系统能力{unique_token}",
                    "method": "模块化接口与分阶段验证",
                    "metric_result": "二十组测试全部完成",
                    "evidence": ["原文列出二十组测试及对应结果。"],
                    "source_excerpt": "原始材料列出了具体模块、测试条件和验证结果。",
                    "evidence_locator": "官方技术文档第 2 节",
                },
            }

        candidates = []
        for section in ("news", "technical"):
            candidates.extend(
                build_item(section, index, valid=index >= 5) for index in range(25)
            )
        layers = {
            "must_read": candidates[:8],
            "physical_ai": [],
            "watch": candidates[8:],
            "featured_papers": [],
            "paper_appendix": [],
            "research": [],
            "brief": [],
        }
        report_config = {
            "product_mode": "intelligence_v11_editorial_library",
            "design_version": "v11-editorial-library",
            "must_read_limit": 6,
            "news_section_limit": 20,
            "technical_section_limit": 20,
            "min_visible_news_count": 20,
            "min_visible_technical_count": 20,
        }

        result = apply_v8_reading_budget(layers, report_config)
        flattened = flatten_report_layers(result)

        self.assertEqual(sum(report_primary_section(item) == "news" for item in flattened), 20)
        self.assertEqual(sum(report_primary_section(item) == "technical" for item in flattened), 20)
        self.assertTrue(
            all(item.get("analysis_version") == "codex-research-v2" for item in flattened)
        )
        self.assertEqual(len(report_config["_v11_selection_contract_rejections"]), 10)
        self.assertTrue(
            all(
                "analysis_version" in row["reasons"]
                for row in report_config["_v11_selection_contract_rejections"]
            )
        )

    def test_v11_reading_budget_balances_news_sources_before_backfill(self):
        def build_news(index, host, score):
            unique_token = chr(0x4E00 + index)
            body = "公司称，新系统公开了具体功能、目标用户、替代流程、验证条件和适用边界。" * 8
            return {
                "content_type": "news",
                "primary_section": "news",
                "model_used": "codex-automation",
                "analysis_version": "codex-research-v2",
                "publish_date": "2026-09-20",
                "claim_type": "official_claim",
                "source_excerpt": "原始公告列出功能、测试条件和部署边界。",
                "evidence_locator": "官方公告第 2 节",
                "title_cn": f"测试公司{unique_token}公布系统验证结果",
                "url": f"https://{host}/news/{index}",
                "summary": body,
                "analysis_body": body,
                "score": score,
                "evidence_quality": 0.8,
                "information_density": 0.8,
                "facts": {
                    "primary_section": "news",
                    "claim_type": "official_claim",
                    "who": host,
                    "action": "公布",
                    "target": f"系统能力{unique_token}",
                    "evidence": ["公告列出测试结果和适用边界。"],
                    "source_excerpt": "原始公告列出功能、测试条件和部署边界。",
                    "evidence_locator": "官方公告第 2 节",
                },
            }

        dominant = [build_news(index, "dominant.example.com", 100 - index) for index in range(15)]
        diverse = [
            build_news(20 + index, f"source-{index}.example.com", 20 - index)
            for index in range(15)
        ]
        result = apply_v8_reading_budget(
            {
                "must_read": dominant + diverse,
                "physical_ai": [],
                "watch": [],
                "featured_papers": [],
                "paper_appendix": [],
                "research": [],
                "brief": [],
            },
            {
                "product_mode": "intelligence_v11_editorial_library",
                "design_version": "v11-editorial-library",
                "must_read_limit": 6,
                "news_section_limit": 20,
                "technical_section_limit": 0,
                "min_visible_news_count": 20,
                "min_visible_technical_count": 0,
                "source_focus_limit": 10,
                "topic_focus_limit": 30,
            },
        )

        news = [item for item in flatten_report_layers(result) if report_primary_section(item) == "news"]
        must_read_source_counts = Counter(
            report_source_key(item) for item in result["must_read"]
        )
        dominant_count = sum("dominant.example.com" in item["url"] for item in news)
        self.assertEqual(len(news), 20)
        self.assertEqual(dominant_count, 10)
        self.assertLessEqual(max(must_read_source_counts.values()), 2)

    def test_title_gate_rejects_action_with_missing_object_before_comma(self):
        self.assertTrue(
            title_looks_bad(
                {
                    "title_cn": "霍夫曼与帕蒂尔复盘创作者资助计划：增加，也保留人的判断",
                }
            )
        )

    def test_translated_facts_keep_raw_entity_when_chinese_normalization_is_empty(self):
        item = enrich_editorial_fields(
            {
                "title": "GPU video decoding for vLLM",
                "title_cn": "vLLM 视频推理使用 GPU 解码",
                "content_type": "project",
                "model_used": "codex-automation",
                "evidence_quality": 0.8,
                "information_density": 0.8,
                "facts": {
                    "who": "NVIDIA Computer Vision Team / vLLM Team",
                    "action": "发布",
                    "target": "GPU 视频解码管线",
                    "evidence": ["使用 NVDEC 与多进程资源协调完成视频输入解码。"],
                    "source_excerpt": "使用 NVDEC 与多进程资源协调完成视频输入解码。",
                    "evidence_locator": "工程博客的视频解码章节",
                },
            }
        )

        metrics = build_editorial_quality_metrics([item])

        self.assertEqual(metrics["llm_key_field_missing_count"], 0)
        self.assertEqual(metrics["llm_health_hint"], "llm_ok")

    def test_paper_substantive_metric_uses_the_visible_v11_summary_and_intro(self):
        paper = {
            "content_type": "paper",
            "title_cn": "Chronicle 选择性重放代理边界",
            "summary": "当前材料不足以展开更多细节。",
            "paper_plain_summary": (
                "代理系统连接模型和外部工具后，故障往往难以稳定复现。"
                "Chronicle 记录这些非确定边界，再选择性重放历史结果。"
            ),
            "paper_technical_intro": (
                "方法把模型与工具边界封装成不可变事件记录，并按故障位置选择重放范围。"
                "实验覆盖六个模拟故障，二十次全量重放保持一致，并识别出基线遗漏的工具变异。"
                "结果说明选择性重放能用于代理回归测试，但真实模型分布漂移仍需继续验证。"
            ),
            "facts": {
                "who": "Chronicle 团队",
                "action": "提出",
                "target": "代理回归测试",
                "method": "记录非确定边界并选择性重放",
                "evidence": ["六个模拟故障，二十次全量重放保持一致。"],
            },
        }

        self.assertEqual(paper_substantive_description_fail_count([paper]), 0)

    def test_v11_watch_container_does_not_trigger_high_evidence_warning(self):
        item = {
            "_final_render_item": True,
            "title_cn": "工程团队发布新的推理调度方案",
            "content_type": "project",
            "report_section": "watch",
            "quality_tier": "focus",
            "evidence_quality": 0.82,
            "information_density": 0.78,
            "facts": {
                "who": "工程团队",
                "action": "发布",
                "target": "推理调度方案",
                "evidence": ["官方工程文档列出了调度机制和测试条件。"],
            },
        }

        gate = evaluate_report_quality(
            {
                "must_read": [],
                "physical_ai": [],
                "watch": [item],
                "featured_papers": [],
                "paper_appendix": [],
                "brief": [],
            },
            {"v11_enabled": True},
        )

        self.assertFalse(gate["high_evidence_calibration_warning"])

    def test_v11_quality_reports_technical_primary_source_ratio_separately(self):
        technical_items = [
            {
                "_final_render_item": True,
                "title_cn": f"技术团队公开推理调度方案 {index}",
                "content_type": "project",
                "primary_section": "technical",
                "report_section": "watch",
                "source_tier": source_tier,
                "evidence_quality": 0.82,
                "information_density": 0.78,
                "facts": {
                    "who": "技术团队",
                    "action": "公开",
                    "target": f"推理调度方案 {index}",
                    "evidence": ["原始材料给出了调度机制和测试条件。"],
                },
            }
            for index, source_tier in enumerate(("official", "official"), start=1)
        ]
        technical_items[0]["url"] = "https://github.com/example/project/releases/tag/v1"
        technical_items[0]["platform"] = "GitHub"
        technical_items[1]["url"] = "https://techcrunch.com/2026/09/20/project-report"
        technical_items[1]["platform"] = "Website"

        gate = evaluate_report_quality(
            {
                "must_read": [],
                "physical_ai": [],
                "watch": technical_items,
                "featured_papers": [],
                "paper_appendix": [],
                "brief": [],
            },
            {"v11_enabled": True},
        )

        self.assertEqual(gate["technical_primary_source_count"], 1)
        self.assertEqual(gate["technical_primary_source_ratio"], 0.5)
        self.assertEqual(gate["status"], "failed")

    def test_quality_flag_recalculation_preserves_supplemental_source_marker(self):
        item = {
            "title": "A verified technical update",
            "title_cn": "一项可核验的技术更新",
            "summary": "团队公开了具体实现机制、实验条件和结果。",
            "url": "https://example.com/technical-update",
            "quality_flags": ["supplemental_older_source", "generic_summary"],
            "facts": {
                "who": "Example Lab",
                "action": "公开",
                "target": "技术实现",
            },
            "evidence_quality": 0.8,
            "information_density": 0.8,
            "model_used": "codex-automation",
            "_codex_research_validated": True,
        }

        flags = item_quality_flags(item)

        self.assertIn("supplemental_older_source", flags)
        self.assertNotIn("generic_summary", flags)

    def test_paper_domain_key_persisted_in_facts_remains_authoritative(self):
        paper = {
            "content_type": "paper",
            "title": "Language Models for Crystal Structure Refinement",
            "topic": "AI for Science",
            "facts": {"paper_domain_key": "other"},
        }

        self.assertEqual(paper_quota_domain(paper, {"other": {"min": 1, "max": 3}}), "other")
        self.assertEqual(learning_domain_key(paper), "products_business")

    def test_editorial_engine_preserves_verified_codex_news_and_technical_prose(self):
        news_body = (
            "OpenAI在官方更新中说明新的模型路由策略：系统先识别任务是否需要工具、长上下文或低延迟响应，"
            "再把请求分配给不同推理路径。官方给出的日志样本显示，常规问答延迟下降18%，但复杂工具任务仍保留完整推理预算。"
            "这次变化影响的是请求进入模型前的调度层，而不是重新训练基础模型；目前公开证据来自发布方测试，尚缺第三方复现。"
        )
        technical_body = (
            "该技术把推理服务拆成前置路由、批处理队列和模型执行三个阶段。路由器根据上下文长度、工具需求和延迟目标选择执行池，"
            "批处理层再按显存占用合并请求，避免长短任务互相阻塞。公开配置使用两类GPU池，并给出批量大小、缓存命中率和端到端延迟；"
            "相对固定模型入口，它把资源分配从静态规则改为按任务特征动态选择。当前结果来自单一部署环境，跨硬件收益仍需复现。"
        )
        common = {
            "title_cn": "模型路由按任务特征分配推理路径",
            "model_used": "codex-automation",
            "analysis_version": "codex-research-v2",
            "source_detail": "OpenAI",
            "evidence_quality": 0.9,
            "information_density": 0.9,
            "facts": {
                "who": "OpenAI",
                "action": "更新",
                "target": "模型路由策略",
                "method": "按任务特征选择推理路径",
                "metric_result": "常规问答延迟下降18%",
                "evidence": ["常规问答延迟下降18%"],
                "source_excerpt": "官方日志样本显示常规问答延迟下降18%。",
                "evidence_locator": "正文性能测试段",
            },
        }

        news = enrich_editorial_fields({**common, "content_type": "news", "primary_section": "news", "summary": news_body})
        technical = enrich_editorial_fields({**common, "content_type": "project", "primary_section": "technical", "summary": technical_body})

        self.assertEqual(news["analysis_body"], news_body)
        self.assertEqual(technical["analysis_body"], technical_body)
        self.assertTrue(news["_codex_research_validated"])
        self.assertTrue(technical["_codex_research_validated"])
        self.assertEqual(news["summary_quality_tier"], "editorial_ready")
        self.assertEqual(technical["summary_quality_tier"], "editorial_ready")

    def test_editorial_engine_preserves_verified_codex_paper_prose(self):
        plain = (
            "这篇论文研究机器人在动态环境里容易依据过时画面做动作的问题。"
            "它先预测下一时刻的物体状态，再把预测结果交给动作策略，并在统一任务中和原方法比较。"
            "实验中，论文报告任务成功率提高了12个百分点，说明动作前预测确实能减少状态滞后。"
        )
        technical = (
            "论文采用两阶段约束训练：第一阶段用未来状态监督训练轻量预测模块，第二阶段冻结原有动作策略，"
            "只把预测状态作为额外输入接入控制链路。实验在动态操作基准上以不使用预测模块的策略为基线，"
            "任务成功率提高12个百分点，同时推理延迟增加8毫秒。作者也指出，当前结果尚未覆盖长时间遮挡、"
            "多机器人协同和不同硬件平台，因此工程价值仍要结合跨平台复现判断。"
        )
        item = {
            "title": "Predict before acting for dynamic manipulation",
            "title_cn": "动作前预测减少动态机器人操作中的状态滞后",
            "content_type": "paper",
            "model_used": "codex-automation",
            "analysis_version": "codex-research-v2",
            "source_detail": "arXiv",
            "facts": {
                "who": "AHEAD",
                "action": "提出",
                "target": "动态机器人操作",
                "method": "先预测下一时刻的物体状态，再交给冻结动作策略",
                "dataset_or_benchmark": "动态操作基准",
                "metric_result": "任务成功率提高12个百分点",
                "baseline": "不使用预测模块的策略",
                "limitation": "尚未覆盖长时间遮挡和多机器人协同",
                "evidence": ["任务成功率提高12个百分点"],
                "source_excerpt": "动态操作基准上任务成功率提高12个百分点。",
                "evidence_locator": "实验表2",
                "paper_plain_summary": plain,
                "paper_technical_intro": technical,
            },
            "evidence_quality": 0.9,
            "information_density": 0.9,
        }

        decorated = enrich_editorial_fields(item)

        self.assertEqual(decorated["paper_plain_summary"], plain)
        self.assertEqual(decorated["paper_technical_intro"], technical)
        self.assertTrue(decorated["_codex_research_validated"])
        self.assertEqual(decorated["summary_quality_tier"], "editorial_ready")

    def test_compact_email_html_reduces_template_whitespace_without_merging_inline_text(self):
        html = """<!doctype html>
        <html>
          <body>
            <!-- remove me -->
            <p><span>first</span> <span>second</span></p>
          </body>
        </html>"""

        compacted = compact_email_html(html)

        self.assertLess(len(compacted), len(html))
        self.assertNotIn("remove me", compacted)
        self.assertIn("<span>first</span> <span>second</span>", compacted)

    def test_v11_email_volumes_partition_all_items_without_duplication(self):
        items = [
            {"id": 1, "content_type": "news", "primary_section": "news", "url": "https://example.com/news"},
            {"id": 2, "content_type": "project", "primary_section": "technical", "url": "https://example.com/tech"},
            {"id": 3, "content_type": "paper", "primary_section": "paper", "url": "https://arxiv.org/abs/3"},
        ]
        layers = {
            "must_read": items[:2],
            "physical_ai": [],
            "watch": [],
            "featured_papers": items[2:],
            "paper_appendix": [],
            "brief": [],
        }

        volumes = build_v11_email_volume_layers(layers)

        self.assertEqual([volume["primary_section"] for volume in volumes], ["news", "technical", "paper"])
        volume_items = [item for volume in volumes for item in volume["items"]]
        self.assertEqual([item["id"] for item in volume_items], [1, 2, 3])
        self.assertEqual(len({item["url"] for item in volume_items}), 3)

    def test_v11_oversized_email_splits_only_when_every_volume_is_safe(self):
        items = [
            {"id": 1, "content_type": "news", "primary_section": "news", "url": "https://example.com/news"},
            {"id": 2, "content_type": "project", "primary_section": "technical", "url": "https://example.com/tech"},
            {"id": 3, "content_type": "paper", "primary_section": "paper", "url": "https://arxiv.org/abs/3"},
        ]
        layers = {
            "must_read": items[:2],
            "physical_ai": [],
            "watch": [],
            "featured_papers": items[2:],
            "paper_appendix": [],
            "brief": [],
        }
        config = {
            "product_mode": "intelligence_v11_editorial_library",
            "email_split_enabled": True,
            "email_html_max_bytes": 100,
        }

        volumes, split_applied = prepare_v11_email_delivery_volumes(
            "完整日报" * 100,
            layers,
            config,
            lambda index, volume: f"<html>{index}:{volume['label']}</html>",
        )

        self.assertTrue(split_applied)
        self.assertEqual(len(volumes), 3)
        self.assertTrue(all(volume["size_bytes"] <= 100 for volume in volumes))

        fallback, split_applied = prepare_v11_email_delivery_volumes(
            "完整日报" * 100,
            layers,
            config,
            lambda index, volume: "超长" * 100,
        )

        self.assertFalse(split_applied)
        self.assertEqual(len(fallback), 1)
        self.assertEqual(fallback[0]["primary_section"], "all")

    def test_v11_oversized_section_is_split_into_safe_subvolumes(self):
        news = {
            "id": 1,
            "content_type": "news",
            "primary_section": "news",
            "url": "https://example.com/news",
        }
        technical = [
            {
                "id": index,
                "content_type": "project",
                "primary_section": "technical",
                "url": f"https://example.com/technical/{index}",
            }
            for index in range(2, 8)
        ]
        paper = {
            "id": 8,
            "content_type": "paper",
            "primary_section": "paper",
            "url": "https://arxiv.org/abs/8",
        }
        layers = {
            "must_read": [news] + technical[:3],
            "physical_ai": [],
            "watch": technical[3:],
            "featured_papers": [paper],
            "paper_appendix": [],
            "brief": [],
        }
        config = {
            "product_mode": "intelligence_v11_editorial_library",
            "email_split_enabled": True,
            "email_html_max_bytes": 125,
        }

        def render_volume(_index, volume):
            return "<html>" + ("x" * (50 + 20 * len(volume["items"]))) + "</html>"

        volumes, split_applied = prepare_v11_email_delivery_volumes(
            "完整日报" * 100,
            layers,
            config,
            render_volume,
        )

        self.assertTrue(split_applied)
        self.assertGreater(len(volumes), 3)
        self.assertTrue(all(volume["size_bytes"] <= 125 for volume in volumes))
        self.assertEqual(
            [item["url"] for volume in volumes for item in volume["items"]],
            [news["url"]] + [item["url"] for item in technical] + [paper["url"]],
        )
        technical_volumes = [
            volume for volume in volumes if volume["primary_section"] == "technical"
        ]
        self.assertGreater(len(technical_volumes), 1)
        self.assertTrue(all("（" in volume["label"] for volume in technical_volumes))

    def test_v11_volume_delivery_commits_only_after_every_part_sends(self):
        volumes = [
            {"label": "新闻", "html": "news"},
            {"label": "技术", "html": "tech"},
            {"label": "论文", "html": "paper"},
        ]
        notifier = Mock()
        notifier.send_email.side_effect = [True, True, True]

        result = send_email_delivery_volumes(notifier, "reader@example.com", "AI 日报", volumes)

        self.assertTrue(result["success"])
        self.assertEqual(result["sent_count"], 3)
        self.assertEqual(result["subjects"][0], "AI 日报 [1/3] 新闻")
        self.assertEqual(notifier.send_email.call_count, 3)

        notifier = Mock()
        notifier.send_email.side_effect = [True, False, True]
        result = send_email_delivery_volumes(notifier, "reader@example.com", "AI 日报", volumes)

        self.assertFalse(result["success"])
        self.assertEqual(result["sent_count"], 1)
        self.assertEqual(notifier.send_email.call_count, 2)

    def test_final_html_quality_reports_email_clipping_risk(self):
        html = "<html><body>" + ("内容" * 60000) + "</body></html>"

        metrics = scan_final_html_quality(
            html,
            {section: [] for section in ("must_read", "physical_ai", "watch", "featured_papers", "paper_appendix", "research", "brief")},
            report_config={
                "design_version": "v10-learning-digest",
                "total_visible_chars_min": 0,
                "total_visible_chars_max": 200000,
                "min_visible_paper_count": 0,
                "min_visible_information_count": 0,
                "paper_technical_intro_min_count": 0,
                "email_html_warning_bytes": 1000,
                "email_html_max_bytes": 2000,
            },
        )

        self.assertTrue(metrics["email_clipping_warning"])
        self.assertTrue(metrics["email_clipping_risk"])
        self.assertEqual(metrics["final_html_quality_status"], "failed")

    def test_v11_final_html_requires_visible_source_notes_and_both_paper_sections(self):
        paper = {
            "content_type": "paper",
            "url": "https://arxiv.org/abs/1",
            "paper_plain_summary": "这篇论文先解释任务问题，再说明方法和实验结果。" * 3,
            "paper_technical_intro": "论文采用两阶段训练，并在基准实验中相对基线提升12%。" * 4,
        }
        metrics = scan_final_html_quality(
            '<html><body><div class="v10-more-plain">通俗介绍</div></body></html>',
            {
                "must_read": [],
                "physical_ai": [],
                "watch": [],
                "featured_papers": [],
                "paper_appendix": [paper],
                "brief": [],
            },
            quality_config={"min_visible_news_count": 0},
            report_config={
                "product_mode": "intelligence_v11_editorial_library",
                "design_version": "v10-learning-digest",
                "total_visible_chars_min": 0,
                "total_visible_chars_max": 10000,
                "min_visible_paper_count": 0,
                "min_visible_information_count": 0,
                "paper_technical_intro_min_count": 0,
            },
        )

        self.assertEqual(metrics["v11_source_note_count"], 0)
        self.assertEqual(metrics["v11_paper_plain_visible_count"], 1)
        self.assertEqual(metrics["v11_paper_technical_visible_count"], 0)
        self.assertEqual(metrics["final_html_quality_status"], "failed")

    def test_final_html_quality_uses_content_type_specific_body_limits(self):
        news_body = "新" * 280
        technical_body = "技" * 360
        interview_body = "访" * 460
        layers = {
            section: []
            for section in ("must_read", "physical_ai", "watch", "featured_papers", "paper_appendix", "research", "brief")
        }
        layers["must_read"] = [
            {
                "content_type": "news",
                "primary_section": "news",
                "analysis_body": news_body,
                "title_cn": "新闻正文",
                "url": "https://example.com/news",
            },
            {
                "content_type": "project",
                "primary_section": "technical",
                "analysis_body": technical_body,
                "title_cn": "技术正文",
                "url": "https://example.com/technical",
            },
            {
                "content_type": "interview",
                "primary_section": "news",
                "analysis_body": interview_body,
                "title_cn": "访谈正文",
                "url": "https://example.com/interview",
            },
        ]
        html = (
            '<div class="v10-body">' + news_body + '</div>'
            '<div class="v10-body">' + technical_body + '</div>'
            '<div class="v10-body">' + interview_body + '</div>'
        )

        metrics = scan_final_html_quality(
            html,
            layers,
            report_config={
                "design_version": "v10-learning-digest",
                "news_body_char_limit": 320,
                "technical_body_char_limit": 400,
                "interview_body_char_limit": 500,
                "total_visible_chars_min": 0,
                "total_visible_chars_max": 5000,
                "min_visible_paper_count": 0,
                "min_visible_information_count": 0,
                "paper_technical_intro_min_count": 0,
                "focus_source_limit": 3,
                "topic_focus_limit": 3,
            },
        )

        self.assertEqual(metrics["display_body_over_limit_count"], 0)

    def test_final_html_quality_rejects_repeated_attribution_opening_style(self):
        bodies = [
            "甲实验室团队表示，新架构把训练和推理拆成可验证阶段。第二句给出具体机制和边界。第三句说明实验结果。",
            "乙公司团队表示，新系统把权限检查接入工具执行链。第二句给出具体机制和边界。第三句说明实验结果。",
            "丙项目团队表示，新版本通过缓存减少重复计算。第二句给出具体机制和边界。第三句说明实验结果。",
        ]
        layers = {
            section: []
            for section in (
                "must_read",
                "physical_ai",
                "watch",
                "featured_papers",
                "paper_appendix",
                "research",
                "brief",
            )
        }
        layers["must_read"] = [
            {
                "content_type": "news",
                "primary_section": "news",
                "analysis_body": body,
                "title_cn": f"测试标题{index}",
                "url": f"https://example.com/{index}",
                "source_detail": f"来源{index}",
            }
            for index, body in enumerate(bodies)
        ]

        metrics = scan_final_html_quality(
            "<html><body>" + "".join(bodies) + "</body></html>",
            layers,
            quality_config={
                "max_attribution_opener_repeat_count": 1,
                "max_attribution_opener_run": 1,
            },
            report_config={
                "design_version": "v10-learning-digest",
                "total_visible_chars_min": 0,
                "total_visible_chars_max": 5000,
                "min_visible_paper_count": 0,
                "min_visible_information_count": 0,
                "paper_technical_intro_min_count": 0,
                "focus_source_limit": 10,
                "topic_focus_limit": 10,
            },
        )

        self.assertEqual(metrics["attribution_opener_counts"], {"团队表示": 3})
        self.assertEqual(metrics["max_attribution_opener_repeat_count"], 3)
        self.assertEqual(metrics["max_attribution_opener_run"], 3)
        self.assertEqual(metrics["final_html_quality_status"], "failed")

    @patch.dict(os.environ, {"WEB_AGENT_TEST_OPENAI_KEY": "test-key"})
    def test_openai_responses_mode_uses_json_output_without_temperature(self):
        processor = LLMProcessor({
            "provider": "openai",
            "api_key_env": "WEB_AGENT_TEST_OPENAI_KEY",
            "api_mode": "responses",
            "model": "gpt-5.6-sol",
            "reasoning_effort": "low",
            "response_verbosity": "low",
            "force_json_response": True,
            "send_temperature": False,
        })
        response = Mock(output_text='{"status":"ok"}')
        create = Mock(return_value=response)
        processor.client.responses.create = create

        adapted = processor._chat_completion(
            messages=[
                {"role": "system", "content": "Return JSON."},
                {"role": "user", "content": "Analyze this item."},
            ],
            temperature=0.2,
            max_tokens=300,
            purpose="test response",
        )

        self.assertEqual(adapted.choices[0].message.content, '{"status":"ok"}')
        kwargs = create.call_args.kwargs
        self.assertEqual(kwargs["model"], "gpt-5.6-sol")
        self.assertEqual(kwargs["reasoning"], {"effort": "low"})
        self.assertEqual(kwargs["text"]["format"], {"type": "json_object"})
        self.assertNotIn("temperature", kwargs)
        self.assertFalse(kwargs["store"])

    def test_source_grounded_news_brief_uses_original_title_and_excerpt(self):
        item = {
            "url": "https://aws.amazon.com/blogs/security/agentcore-oauth-consent/",
            "title": "Amazon Bedrock AgentCore adds OAuth consent for enterprise agents",
            "title_cn": "AI领域新进展",
            "content": (
                "Amazon Bedrock AgentCore now supports OAuth consent flows for enterprise agents. "
                "Administrators can require approval before an agent accesses connected applications, "
                "and the service records each authorization decision for later audit."
            ),
            "content_type": "news",
            "source_detail": "AWS News Blog",
            "source_tier": "official",
            "model_used": "template_fallback",
        }

        brief = build_source_grounded_news_brief(item)

        self.assertIsNotNone(brief)
        self.assertEqual(brief["source_display_title"], item["title"])
        self.assertIn("Administrators can require approval", brief["source_excerpt"])
        self.assertTrue(brief["source_grounded_brief"])
        self.assertEqual(brief["quality_tier"], "brief")
        self.assertNotIn(item["title_cn"], brief["source_excerpt"])
        gate = evaluate_report_quality(
            {
                "must_read": [],
                "physical_ai": [],
                "watch": [],
                "featured_papers": [],
                "paper_appendix": [],
                "research": [],
                "brief": [brief],
            }
        )
        self.assertEqual(gate["bad_title_count"], 0)

    def test_source_grounded_news_brief_rejects_aggregator_or_empty_source(self):
        aggregator = {
            "url": "https://news.google.com/rss/articles/example",
            "title": "A sufficiently long AI news headline from an aggregator",
            "content": "A sufficiently long source excerpt that would otherwise pass the minimum length checks.",
            "content_type": "news",
            "source_tier": "aggregator",
        }
        empty = {
            "url": "https://example.com/news",
            "title": "A sufficiently long original AI news headline",
            "content": "",
            "content_type": "news",
            "source_tier": "media",
        }

        self.assertIsNone(build_source_grounded_news_brief(aggregator))
        self.assertIsNone(build_source_grounded_news_brief(empty))

    def test_template_fallback_paper_is_index_only_instead_of_synthetic_summary(self):
        paper = {
            "title": "MINERVA: A 0.54M parameter policy for robot manipulation",
            "content": "MINERVA uses a compact 0.54M parameter policy for manipulation.",
            "content_type": "paper",
            "model_used": "template_fallback",
            "source_detail": "arXiv",
            "evidence_quality": 0.8,
            "information_density": 0.8,
            "facts": {
                "who": "MINERVA",
                "action": "提出",
                "target": "具身智能",
                "method": "具身智能",
                "metric_result": "成功率达到 0.54M",
                "evidence": ["0.54M"],
            },
        }

        decorated = enrich_editorial_fields(paper)

        self.assertEqual(decorated["summary_quality_tier"], "index_only")
        self.assertEqual(decorated["paper_plain_summary"], "")
        self.assertEqual(decorated["paper_technical_intro"], "")
        self.assertIn("template_fallback", decorated["summary_quality_reasons"])
        self.assertIn("generic_method", decorated["summary_quality_reasons"])

    def test_source_mismatched_paper_metric_is_not_expanded(self):
        paper = {
            "title": "A robot policy evaluated on BenchX",
            "content": "The paper reports a 75% success rate on BenchX.",
            "content_type": "paper",
            "model_used": "deepseek-v4-pro",
            "facts": {
                "who": "PolicyX",
                "action": "提出",
                "target": "机器人操作策略",
                "method": "通过视觉编码器预测动作并使用行为克隆训练策略",
                "metric_result": "在 BenchX 上成功率达到 82%",
                "evidence": ["在 BenchX 上成功率达到 82%"],
            },
        }

        assessment = assess_fact_publishability(paper, paper["facts"])

        self.assertEqual(assessment["tier"], "index_only")
        self.assertIn("unsupported_numeric_claim", assessment["reasons"])

    @patch.dict(os.environ, {"WEB_AGENT_TEST_API_KEY": "test-key"})
    def test_insufficient_balance_disables_remaining_live_llm_calls(self):
        processor = LLMProcessor({
            "api_key_env": "WEB_AGENT_TEST_API_KEY",
            "base_url": "https://api.deepseek.com",
            "model": "deepseek-v4-pro",
            "fallback_model": "deepseek-chat",
            "primary_mode": "fact_extraction",
        })
        create = Mock(side_effect=RuntimeError("Error code: 402 - Insufficient Balance"))
        processor.client.chat.completions.create = create
        article = {
            "title": "A robot world model",
            "content": "The model predicts future robot states.",
            "content_type": "paper",
        }

        first = processor.process_article(article)
        second = processor.process_article({**article, "title": "Another robot world model"})

        self.assertEqual(create.call_count, 1)
        self.assertEqual(first["model_used"], "template_fallback")
        self.assertEqual(second["model_used"], "template_fallback")
        self.assertFalse(processor.live_generation_available)
        self.assertEqual(processor.health_snapshot()["status"], "insufficient_balance")

    def test_hard_paper_minimum_blocks_a_report_below_fifteen(self):
        config = {
            "block_send_on_final_failure": True,
            "allow_degraded_send_on_soft_failure": True,
            "hard_min_visible_paper_count": True,
            "min_visible_paper_count": 15,
        }
        diagnostics = {"quality_gate": {"visible_paper_count": 14}}

        self.assertTrue(should_block_report_send(config, "failed", "send", diagnostics))

    def test_primary_fact_extraction_rejects_generic_paper_facts(self):
        processor = LLMProcessor({"api_key_env": "WEB_AGENT_TEST_MISSING_API_KEY"})
        article = {"content_type": "paper"}

        self.assertFalse(processor._primary_facts_are_usable(article, {
            "who": "研究团队",
            "action": "发布",
            "target": "机器人",
            "method": "机器人",
            "evidence": ["提出一种机器人方法"],
        }))
        self.assertTrue(processor._primary_facts_are_usable(article, {
            "who": "MotionPolicy",
            "action": "提出",
            "target": "动态场景中的机器人操作策略",
            "method": "先预测物体未来轨迹，再用预测状态约束动作策略解码器",
            "evidence": ["在 RoboSuite 基准上完成评测", "成功率比 BC 基线提高 12 个百分点"],
            "dataset_or_benchmark": "RoboSuite",
            "metric_result": "成功率提高 12 个百分点",
            "baseline": "BC",
        }))

    def test_v10_quality_gate_does_not_count_reappeared_updates_as_fresh_papers(self):
        updates = [
            {
                "id": index,
                "content_type": "paper",
                "title": f"Updated paper {index}",
                "title_cn": f"更新论文 {index}",
                "url": f"https://arxiv.org/abs/2608.{index:05d}",
                "is_reappeared_update": True,
                "facts": {"method": "method", "metric_result": "result 80%", "evidence": ["result 80%"]},
                "facts_cn": {"method": "方法", "metric_result": "结果 80%", "evidence": ["结果 80%"]},
                "paper_plain_summary": "论文采用具体方法完成实验，并报告百分之八十的结果。",
                "paper_technical_intro": "论文先构建状态表示，再通过实验比较不同基线，结果显示方法取得百分之八十的成功率。",
            }
            for index in range(1, 11)
        ]
        layers = {
            "must_read": [],
            "physical_ai": [],
            "watch": [],
            "featured_papers": [],
            "paper_appendix": updates,
            "research": [],
            "brief": [],
        }

        result = evaluate_report_quality(
            layers,
            {"v10_enabled": True, "min_visible_paper_count": 10, "min_visible_information_count": 0},
        )

        self.assertEqual(result["visible_paper_count"], 10)
        self.assertEqual(result["visible_fresh_paper_count"], 0)
        self.assertEqual(result["status"], "failed")

    def test_collector_failure_record_preserves_arxiv_retry_diagnostics(self):
        class FailedCollector:
            fetch_diagnostics = {
                "request_error_count": 3,
                "retry_paths": [{
                    "category": "cs.RO",
                    "attempted_show_counts": [500, 100, 50],
                    "successful_show_count": None,
                    "result": "http_error",
                }],
            }

        row = build_collector_failure_record(
            FailedCollector(),
            "ArxivCollector[Robotics]",
            "HTTP 503",
            1.26,
        )

        self.assertEqual(row["status"], "error")
        self.assertEqual(row["duration_seconds"], 1.3)
        self.assertEqual(row["diagnostics"]["request_error_count"], 3)
        self.assertEqual(row["diagnostics"]["retry_paths"][0]["result"], "http_error")

    def test_empty_hour_ladder_explicitly_disables_backfill(self):
        self.assertEqual(parse_hour_ladder([], [24, 48, 72]), [])
        self.assertEqual(parse_hour_ladder(None, [24, 48, 72]), [24, 48, 72])

    def test_database_can_reload_current_processing_batch_by_url(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db = Database(str(Path(temp_dir) / "articles.db"))
            article = {
                "url": "https://arxiv.org/abs/2608.12345",
                "title": "Validation paper",
                "content": "A validation paper about robot learning.",
                "content_type": "paper",
            }
            self.assertTrue(db.insert_article(article))
            db.update_article_processing(
                url=article["url"],
                summary="摘要",
                score=8.0,
                keywords=["robot"],
                category="Robotics",
                facts={"method": "方法", "metric_result": "结果"},
            )

            rows = db.get_articles_by_urls([article["url"], article["url"]])

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["url"], article["url"])
        self.assertEqual(rows[0]["facts"]["method"], "方法")

    def test_verified_research_refreshes_existing_article_source_snapshot(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db = Database(str(Path(temp_dir) / "articles.db"))
            url = "https://example.com/existing-research"
            self.assertTrue(
                db.insert_article(
                    {
                        "url": url,
                        "title": "Old title",
                        "source": "Old source",
                        "publish_date": "2026-09-01",
                        "content_type": "news",
                        "run_id": "old-run",
                    }
                )
            )
            refreshed = {
                "url": url,
                "title": "Verified current title",
                "source": "Codex Research",
                "source_detail": "Official engineering blog",
                "content": "Current verified source excerpt and evidence.",
                "publish_date": "2026-09-20",
                "author": "Research team",
                "content_type": "project",
                "platform": "Blog",
                "topic": "Infra / Open Source",
                "run_id": "current-run",
                "canonical_url": "https://example.com/existing-research",
                "source_tier": "official",
            }

            self.assertTrue(db.update_article_source_snapshot(url, refreshed))
            db.update_article_processing(
                url=url,
                summary="当前中文整理稿。",
                score=8.5,
                keywords=["推理"],
                category="基础设施",
                facts={"who": "Research team", "action": "发布", "target": "runtime"},
                model_used="codex-automation",
                analysis_version="codex-research-v2",
            )
            row = db.get_articles_by_urls([url])[0]

        self.assertEqual(row["title"], "Verified current title")
        self.assertEqual(row["source_detail"], "Official engineering blog")
        self.assertEqual(row["publish_date"], "2026-09-20")
        self.assertEqual(row["content_type"], "project")
        self.assertEqual(row["run_id"], "current-run")
        self.assertEqual(row["source_tier"], "official")

    def test_latest_report_can_exclude_validation_archive(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db = Database(str(Path(temp_dir) / "reports.db"))
            db.record_report_run(
                "production-report",
                "production-run",
                html_report_path="archive/report_20260812_1300.html",
                quality_status="passed",
            )
            db.record_report_run(
                "validation-report",
                "validation-run",
                html_report_path="archive/validation/report_20260812_1310.html",
                quality_status="failed",
            )

            latest_any = db.get_latest_report_run()
            latest_production = db.get_latest_report_run(exclude_validation=True)

        self.assertEqual(latest_any["report_id"], "validation-report")
        self.assertEqual(latest_production["report_id"], "production-report")

    def test_latest_collection_report_skips_rerender_with_zero_collectors(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db = Database(str(Path(temp_dir) / "reports.db"))
            db.record_report_run(
                "collected-report",
                "collected-run",
                html_report_path="archive/report_collected.html",
                quality_diagnostics={
                    "collection": {
                        "collector_success_count": 5,
                        "arxiv_http_error_count": 1,
                    }
                },
            )
            db.record_report_run(
                "rerender-report",
                "rerender-run",
                html_report_path="archive/report_rerender.html",
                quality_diagnostics={
                    "collection": {
                        "collector_success_count": 0,
                        "collector_failed_count": 0,
                    }
                },
            )

            latest = db.get_latest_report_run_with_collection()

        self.assertEqual(latest["report_id"], "collected-report")
        self.assertEqual(latest["quality_diagnostics"]["collection"]["arxiv_http_error_count"], 1)

    def test_arxiv_collector_summary_distinguishes_failures_zero_and_recovery(self):
        summary = build_collector_summary([
            {
                "label": "ArxivCollector[World Model]",
                "status": "success",
                "inserted_count": 3,
                "diagnostics": {
                    "request_error_count": 1,
                    "successful_page_count": 1,
                    "retry_paths": [{
                        "category": "cs.AI",
                        "attempted_show_counts": [500, 100],
                        "successful_show_count": 100,
                        "result": "success",
                    }],
                },
            },
            {
                "label": "ArxivCollector[Physical AI]",
                "status": "error",
                "inserted_count": 0,
                "diagnostics": {"page_parse_error_count": 1},
            },
            {
                "label": "ArxivCollector[Robotics]",
                "status": "empty",
                "inserted_count": 0,
                "diagnostics": {"true_zero_result": True},
            },
            {
                "label": "ArxivCollector[Agent]",
                "status": "empty",
                "inserted_count": 0,
                "diagnostics": {"no_match_result": True},
            },
        ])

        self.assertEqual(summary["arxiv_http_error_count"], 1)
        self.assertEqual(summary["arxiv_parse_error_count"], 1)
        self.assertEqual(summary["arxiv_zero_result_warning_count"], 2)
        self.assertEqual(summary["arxiv_true_zero_result_count"], 1)
        self.assertEqual(summary["arxiv_no_match_result_count"], 1)
        self.assertEqual(summary["arxiv_fallback_recovery_count"], 1)
        self.assertEqual(summary["arxiv_retry_paths"][0]["source"], "ArxivCollector[World Model]")
        self.assertEqual(summary["arxiv_retry_paths"][0]["attempted_show_counts"], [500, 100])
        self.assertEqual(summary["success_count"], 1)
        self.assertEqual(summary["empty_count"], 2)
        self.assertIn("成功 1 个", summary["status_text"])
        self.assertIn("零结果 2 个", summary["status_text"])

    def test_v8_reader_uses_single_column_budgeted_sections(self):
        domains = ["world_model", "physical_ai", "agent_models", "infra_open_source", "products_business"]
        news = []
        for index, domain in enumerate(domains, start=1):
            news.append(
                enrich_editorial_fields(
                    {
                        "id": index,
                        "url": f"https://source{index}.example.com/item-{index}",
                        "title_cn": f"公司{index}发布领域能力更新",
                        "summary": f"公司{index}发布领域能力更新，并给出客户验证结果。",
                        "content_type": "news",
                        "domain_key": domain,
                        "source_detail": f"官方来源{index}",
                        "source_tier": "official",
                        "evidence_quality": 0.8,
                        "information_density": 0.8,
                        "score": 9 - index / 10,
                        "facts": {
                            "who": f"公司{index}",
                            "action": "发布",
                            "target": "领域能力更新",
                            "evidence": [f"已有{index + 10}个客户完成验证"],
                            "method": "通过工作流集成连接现有系统",
                            "metric_result": f"已有{index + 10}个客户完成验证",
                        },
                    }
                )
            )
        papers = []
        for index in range(6):
            papers.append(
                enrich_editorial_fields(
                    {
                        "id": 100 + index,
                        "url": f"https://arxiv.org/abs/v8-{index}",
                        "title_cn": f"Paper{index}提出分阶段预测框架",
                        "summary": f"Paper{index}提出分阶段预测框架，实验成功率达到{70 + index}%。",
                        "content_type": "paper",
                        "domain_key": domains[index % len(domains)],
                        "source_detail": "arXiv",
                        "source_tier": "research",
                        "evidence_quality": 0.85,
                        "information_density": 0.85,
                        "score": 9.5 - index / 10,
                        "facts": {
                            "who": f"Paper{index}",
                            "action": "提出",
                            "target": "分阶段预测框架",
                            "evidence": [f"实验成功率达到{70 + index}%"],
                            "method": f"先预测第{index + 1}阶段状态，再交给策略模块决策",
                            "dataset_or_benchmark": f"V8Bench-{index}",
                            "metric_result": f"成功率达到{70 + index}%",
                            "baseline": f"Baseline-{index}",
                        },
                    }
                )
            )
        layers = {
            "must_read": news,
            "physical_ai": [],
            "watch": [],
            "featured_papers": papers,
            "paper_appendix": [],
            "research": papers,
            "brief": [],
        }
        config = {
            "design_version": "v8-editorial-reader",
            "must_read_limit": 5,
            "domain_item_limit": 2,
            "paper_featured_limit": 6,
            "paper_appendix_limit": 6,
            "brief_limit": 8,
            "source_focus_limit": 2,
            "topic_focus_limit": 3,
            "total_visible_chars_min": 0,
            "total_visible_chars_max": 9000,
        }
        budgeted = apply_v8_reading_budget(layers, config)
        generator = ReportGenerator(design_version="v8-editorial-reader", report_config=config)
        final_layers = generator._decorate_layers(budgeted)
        for item in flatten_report_layers(final_layers):
            item["_final_render_item"] = True
        html = generator.generate_html(
            papers=papers,
            updates=news,
            mixed_items=flatten_report_layers(final_layers),
            report_summary={},
            trend_summary={"items": []},
            alert_summary={"needs_alert": False, "issues": []},
            archive_summary={"entries": []},
            layered_updates=final_layers,
        )
        metrics = scan_final_html_quality(
            html,
            final_layers,
            quality_config={"exact_duplicate_sentence_count": 1},
            report_config=config,
        )

        self.assertIn('data-design-version="v8-editorial-reader"', html)
        self.assertEqual(html.count('class="v8-memory-row"'), 3)
        self.assertEqual(html.count('class="v8-anchor"'), 5)
        self.assertNotIn("<details", html)
        self.assertNotIn("Score ", html)
        self.assertEqual(metrics["appendix_body_overlap_count"], 0)
        self.assertEqual(metrics["featured_paper_count"], 6)
        self.assertLessEqual(metrics["memory_total_chars"], 240)
        self.assertFalse(metrics["memory_budget_exceeded"])
        self.assertEqual(metrics["display_body_over_limit_count"], 0)

    def test_final_scan_allows_small_focus_set_within_source_limit(self):
        items = [
            {
                "url": "https://alpha.example.com/robotics-update",
                "title_cn": "甲公司完成机器人控制验证",
                "editorial_title": "甲公司完成机器人控制验证",
                "content_type": "news",
                "domain_key": "physical_ai",
                "analysis_body": "甲公司把视觉状态编码接入控制器，并在十二项真实任务中完成连续动作验证。",
                "evidence_line": "公开结果覆盖十二项真实任务。",
                "source_detail": "甲公司官方",
                "quality_tier": "deep",
            },
            {
                "url": "https://beta.example.com/model-training",
                "title_cn": "乙实验室公开世界模型训练方法",
                "editorial_title": "乙实验室公开世界模型训练方法",
                "content_type": "news",
                "domain_key": "world_model",
                "analysis_body": "乙实验室先学习离散状态表示，再通过时序预测约束策略训练，并公开消融实验。",
                "evidence_line": "论文列出三组基线和消融结果。",
                "source_detail": "乙实验室官网",
                "quality_tier": "deep",
            },
            {
                "url": "https://gamma.example.com/agent-release",
                "title_cn": "丙团队发布智能体工作流接口",
                "editorial_title": "丙团队发布智能体工作流接口",
                "content_type": "news",
                "domain_key": "agent_models",
                "analysis_body": "丙团队把权限控制和工具调用合并到工作流接口，并给出十个客户部署案例。",
                "evidence_line": "发布材料披露十个客户部署案例。",
                "source_detail": "丙团队公告",
                "quality_tier": "deep",
            },
        ]
        layers = {
            "must_read": items,
            "physical_ai": [],
            "watch": [],
            "featured_papers": [],
            "paper_appendix": [],
            "research": [],
            "brief": [],
        }

        metrics = scan_final_html_quality(
            "<html><body>今日技术情报已经完成编辑审核。</body></html>",
            layers,
            report_config={
                "design_version": "v7-classic-briefing",
                "total_visible_chars_min": 1,
                "total_visible_chars_max": 9000,
                "source_focus_limit": 2,
                "topic_focus_limit": 3,
            },
        )

        self.assertEqual(metrics["focus_source_max_count"], 1)
        self.assertEqual(metrics["focus_source_concentration"], 0.333)
        self.assertEqual(metrics["final_html_quality_status"], "passed")

        layers["must_read"] = items + [
            {
                **items[0],
                "url": "https://alpha.example.com/robotics-followup",
                "title_cn": "甲公司补充机器人测试结果",
                "editorial_title": "甲公司补充机器人测试结果",
                "analysis_body": "甲公司补充公开了控制器在复杂抓取任务中的失败案例和恢复策略。",
                "evidence_line": "补充材料列出失败案例和恢复步骤。",
            },
            {
                **items[1],
                "url": "https://delta.example.com/inference-system",
                "title_cn": "丁团队验证低延迟推理系统",
                "editorial_title": "丁团队验证低延迟推理系统",
                "analysis_body": "丁团队通过分层缓存降低推理延迟，并披露不同并发规模下的吞吐结果。",
                "evidence_line": "测试覆盖四档并发规模。",
            },
            {
                **items[2],
                "url": "https://epsilon.example.com/model-evaluation",
                "title_cn": "戊机构发布模型评估结果",
                "editorial_title": "戊机构发布模型评估结果",
                "analysis_body": "戊机构使用统一数据集比较五种模型，并公布准确率与推理成本。",
                "evidence_line": "评估包含五种模型的准确率和成本。",
            },
        ]
        metrics = scan_final_html_quality(
            "<html><body>晚间技术情报已经完成编辑审核。</body></html>",
            layers,
            report_config={
                "design_version": "v7-classic-briefing",
                "total_visible_chars_min": 1,
                "total_visible_chars_max": 9000,
                "source_focus_limit": 2,
                "topic_focus_limit": 3,
            },
        )

        self.assertEqual(metrics["focus_source_item_count"], 6)
        self.assertEqual(metrics["focus_source_max_count"], 2)
        self.assertEqual(metrics["focus_source_concentration"], 0.333)
        self.assertEqual(metrics["final_html_quality_status"], "passed")

    def test_v11_concentration_uses_repository_and_technical_category(self):
        items = []
        for index, category in enumerate(
            ["training_data", "training_data", "agent_systems", "inference_deployment"]
        ):
            items.append(
                {
                    "url": f"https://github.com/org-{index}/project/releases/tag/v1",
                    "title_cn": f"技术项目 {index} 发布更新",
                    "editorial_title": f"技术项目 {index} 发布更新",
                    "content_type": "project",
                    "primary_section": "technical",
                    "technical_category": category,
                    "domain_key": "infra_open_source",
                    "analysis_body": "团队介绍，该项目通过分层缓存和批处理调度降低推理延迟，并公开部署限制。",
                    "evidence_line": "发布说明列出配置项、兼容范围和已知限制。",
                    "source_detail": f"org-{index}/project",
                    "quality_tier": "deep",
                }
            )
        layers = {
            "must_read": items,
            "physical_ai": [],
            "watch": [],
            "featured_papers": [],
            "paper_appendix": [],
            "research": [],
            "brief": [],
        }

        metrics = scan_final_html_quality(
            "<html><body>V11 技术内容完成编辑审核。</body></html>",
            layers,
            report_config={
                "product_mode": "intelligence_v11_editorial_library",
                "design_version": "v11-editorial-library",
                "total_visible_chars_min": 1,
                "total_visible_chars_max": 9000,
                "source_focus_limit": 1,
                "topic_focus_limit": 2,
                "min_visible_news_count": 0,
                "min_visible_technical_count": 0,
                "min_visible_paper_count": 0,
                "paper_technical_intro_min_count": 0,
            },
        )

        self.assertEqual(metrics["focus_source_max_count"], 1)
        self.assertEqual(metrics["focus_topic_max_count"], 2)

    def test_web_search_pauses_after_five_consecutive_empty_runs(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            db = Database(str(Path(tmp_dir) / "empty-source.db"))
            label = "WebSearchCollector[Empty Source]"
            for index in range(5):
                db.record_collector_runs(
                    f"run-{index}",
                    [{"label": label, "status": "success", "inserted_count": 0, "collected_count": 0}],
                )
            self.assertTrue(
                should_skip_empty_collector(
                    db,
                    label,
                    {"enabled": True, "empty_success_threshold": 5, "lookback_runs": 5, "recovery_interval_hours": 24},
                )
            )

    def test_v8_budget_does_not_repeat_a_focus_url_in_briefs(self):
        item = enrich_editorial_fields(
            {
                "url": "https://example.com/concrete-update",
                "title_cn": "Example发布可验证更新",
                "content_type": "news",
                "source_detail": "Example Official",
                "source_tier": "official",
                "evidence_quality": 0.8,
                "information_density": 0.8,
                "facts": {
                    "who": "Example",
                    "action": "发布",
                    "target": "可验证更新",
                    "method": "通过工作流接口接入现有系统",
                    "evidence": ["已有12个客户完成部署"],
                },
            }
        )
        layers = {
            "must_read": [item],
            "physical_ai": [],
            "watch": [],
            "featured_papers": [],
            "paper_appendix": [],
            "research": [],
            "brief": [item],
        }
        result = apply_v8_reading_budget(
            layers,
            {"design_version": "v8-editorial-reader", "total_visible_chars_min": 0},
        )
        focus_urls = {entry["url"] for entry in result["must_read"] + result["watch"]}
        brief_urls = {entry["url"] for entry in result["brief"]}
        self.assertFalse(focus_urls & brief_urls)

    def test_v10_budget_keeps_source_grounded_news_when_editorial_fallback_is_weak(self):
        news = {
            "url": "https://techcrunch.com/example/robot-truck-launch",
            "title": "Pony.ai starts testing an autonomous electric truck platform",
            "title_cn": "AI领域新进展",
            "content": (
                "Pony.ai has started road testing a new autonomous electric truck platform in China. "
                "The company says the vehicles combine its virtual driver software with a production "
                "electric chassis and will initially serve fixed freight routes."
            ),
            "content_type": "news",
            "source_detail": "TechCrunch",
            "source_tier": "media",
            "model_used": "template_fallback",
            "evidence_quality": 0.2,
            "information_density": 0.2,
        }
        result = apply_v8_reading_budget(
            {
                "must_read": [],
                "physical_ai": [],
                "watch": [news],
                "featured_papers": [],
                "paper_appendix": [],
                "research": [],
                "brief": [],
            },
            {
                "design_version": "v10-learning-digest",
                "source_news_brief_limit": 12,
                "brief_limit": 3,
            },
        )

        self.assertEqual(len(result["brief"]), 1)
        self.assertEqual(result["brief"][0]["source_display_title"], news["title"])
        self.assertTrue(result["brief"][0]["source_grounded_brief"])

    def test_send_block_requires_minimum_visible_news(self):
        config = {"min_visible_news_count": 8}

        self.assertIn(
            "visible_news_count",
            report_send_blocking_reasons(config, {"visible_news_count": 0}),
        )
        self.assertNotIn(
            "visible_news_count",
            report_send_blocking_reasons(config, {"visible_news_count": 8}),
        )

    def test_send_block_requires_five_to_seven_editorial_decisions(self):
        config = {
            "editorial_decision_min_count": 5,
            "editorial_decision_max_count": 7,
        }

        self.assertIn(
            "editorial_decision_count",
            report_send_blocking_reasons(config, {"editorial_decision_count": 4}),
        )
        self.assertNotIn(
            "editorial_decision_count",
            report_send_blocking_reasons(config, {"editorial_decision_count": 6}),
        )
        self.assertIn(
            "editorial_decision_count",
            report_send_blocking_reasons(config, {"editorial_decision_count": 8}),
        )

    def test_send_block_rejects_untraceable_or_duplicate_editorial_decisions(self):
        self.assertEqual(
            set(
                report_send_blocking_reasons(
                    {},
                    {
                        "editorial_decision_source_missing_count": 1,
                        "editorial_decision_duplicate_source_count": 1,
                    },
                )
            ),
            {
                "editorial_decision_source_missing_count",
                "editorial_decision_duplicate_source_count",
            },
        )

    def test_v11_send_block_rejects_incomplete_final_content(self):
        diagnostics = {
            "body_under_min_count": 1,
            "publish_date_missing_count": 1,
            "source_evidence_missing_count": 1,
            "claim_type_missing_count": 1,
            "paper_full_text_missing_count": 1,
        }

        reasons = report_send_blocking_reasons({}, diagnostics)

        self.assertEqual(set(diagnostics), set(reasons))

    def test_v11_send_block_rejects_items_outside_current_research_batch(self):
        reasons = report_send_blocking_reasons({}, {"v11_external_item_count": 1})

        self.assertEqual(["v11_external_item_count"], reasons)

    def test_v11_send_block_rejects_too_many_supplemental_items(self):
        reasons = report_send_blocking_reasons(
            {},
            {"v11_supplemental_limit_exceeded": {"news": {"count": 6, "max": 5}}},
        )

        self.assertEqual(["v11_supplemental_limit_exceeded"], reasons)

    def test_v11_send_block_rejects_overage_supplemental_items(self):
        reasons = report_send_blocking_reasons(
            {},
            {"v11_supplemental_age_violation_count": 1},
        )

        self.assertEqual(["v11_supplemental_age_violation_count"], reasons)

    def test_v11_send_block_rejects_missing_or_changed_ingested_editorial_copy(self):
        reasons = report_send_blocking_reasons(
            {},
            {
                "v11_editorial_source_hash_missing_count": 1,
                "v11_editorial_source_mismatch_count": 2,
            },
        )

        self.assertEqual(
            [
                "v11_editorial_source_hash_missing_count",
                "v11_editorial_source_mismatch_count",
            ],
            reasons,
        )

    def test_v11_send_block_rejects_full_or_split_html_fidelity_failure(self):
        reasons = report_send_blocking_reasons(
            {},
            {
                "v11_content_fidelity_missing_count": 1,
                "email_delivery_volume_content_fidelity_missing_count": 2,
            },
        )

        self.assertEqual(
            [
                "email_delivery_volume_content_fidelity_missing_count",
                "v11_content_fidelity_missing_count",
            ],
            reasons,
        )

    def test_v11_send_block_rejects_failed_pre_send_ui_audit(self):
        reasons = report_send_blocking_reasons({}, {"ui_audit_status": "failed"})

        self.assertEqual(["ui_audit_status"], reasons)

    def test_v11_paper_appendix_rejects_index_only_items(self):
        paper = enrich_editorial_fields(
            {
                "id": 99101,
                "content_type": "paper",
                "title": "Index only paper",
                "title_cn": "仅有索引内容的论文",
                "url": "https://arxiv.org/abs/99101",
                "source_tier": "research",
                "score": 8.0,
                "evidence_quality": 0.8,
                "information_density": 0.8,
                "facts": {
                    "who": "研究团队",
                    "action": "提出",
                    "target": "索引方法",
                    "evidence": ["论文页面仅提供标题信息"],
                },
            }
        )
        paper["summary_quality_tier"] = "index_only"
        paper["paper_plain_summary"] = ""
        paper["paper_technical_intro"] = ""

        result = apply_v8_reading_budget(
            {
                "must_read": [],
                "physical_ai": [],
                "watch": [],
                "featured_papers": [paper],
                "paper_appendix": [],
                "research": [],
                "brief": [],
            },
            {
                "product_mode": "intelligence_v11_editorial_library",
                "design_version": "v10-learning-digest",
                "paper_featured_limit": 10,
                "paper_appendix_limit": 15,
            },
        )

        self.assertEqual(result["featured_papers"], [])
        self.assertEqual(result["paper_appendix"], [])

    def test_v11_reading_budget_preserves_every_valid_primary_section_item(self):
        news = [
            {
                "content_type": "news",
                "primary_section": "news",
                "title_cn": f"新闻条目{index}说明具体发布变化",
                "url": f"https://example.com/news/{index}",
                "summary": "发布方说明了具体变化、证据、适用范围和当前限制。" * 10,
                "analysis_body": "发布方说明了具体变化、证据、适用范围和当前限制。" * 10,
                "facts": {"who": "发布方", "action": "发布", "target": f"功能{index}"},
            }
            for index in range(20)
        ]
        technical = [
            {
                "content_type": "project",
                "primary_section": "technical",
                "title_cn": f"技术条目{index}解释系统实现",
                "url": f"https://example.com/technical/{index}",
                "summary": "系统解释了模块接口、输入输出、部署条件和实验边界。" * 12,
                "analysis_body": "系统解释了模块接口、输入输出、部署条件和实验边界。" * 12,
                "facts": {"who": "工程团队", "action": "实现", "target": f"系统{index}"},
            }
            for index in range(20)
        ]
        papers = [
            {
                "content_type": "paper",
                "primary_section": "paper",
                "model_used": "codex-automation",
                "analysis_version": "codex-research-v2",
                "publish_date": "2026-09-20",
                "claim_type": "research_result",
                "source_excerpt": "论文报告两阶段状态预测训练及动态操作基准结果。",
                "evidence_locator": "论文方法与实验章节",
                "evidence_quality": 0.8,
                "information_density": 0.8,
                "title_cn": f"论文条目{index}提出预测方法",
                "url": f"https://arxiv.org/abs/9900.{index:05d}",
                "paper_plain_summary": (
                    "这篇论文研究机器人依据过时状态执行动作的问题。"
                    "它先预测下一时刻状态，再把结果交给动作策略，并与原方法进行实验比较。"
                    "不过实验只覆盖有限任务，不能直接代表开放环境中的长期运行效果。"
                    "实验中，论文报告成功率提高12个百分点，说明动作前预测可以减少状态滞后。"
                ),
                "paper_technical_intro": (
                    "论文采用两阶段训练，先用未来状态监督预测模块，再冻结原策略并接入预测表示。"
                    "实验在动态操作基准上与不使用预测模块的基线比较，成功率提高12个百分点，延迟增加8毫秒。"
                    "预测模块只处理与动作有关的状态变化，没有重新训练底层视觉语言动作模型。"
                    "作者指出结果尚未覆盖长时间遮挡、多机器人协同和跨硬件部署，工程价值仍需进一步复现。"
                ),
                "facts": {
                    "who": "研究团队",
                    "action": "提出",
                    "target": f"预测方法{index}",
                    "method": "两阶段状态预测训练",
                    "metric_result": "成功率提高12个百分点",
                    "evidence": ["动态操作基准上的成功率提高12个百分点。"],
                },
            }
            for index in range(15)
        ]
        result = apply_v8_reading_budget(
            {
                "must_read": news[:6],
                "physical_ai": [],
                "watch": news[6:] + technical,
                "featured_papers": papers[:10],
                "paper_appendix": papers[10:],
                "research": [],
                "brief": [],
            },
            {
                "product_mode": "intelligence_v11_editorial_library",
                "design_version": "v10-learning-digest",
                "must_read_limit": 6,
                "paper_featured_limit": 10,
                "paper_appendix_limit": 15,
            },
        )

        flattened = flatten_report_layers(result)
        self.assertEqual(len(flattened), 55)
        self.assertEqual(sum(item.get("primary_section") == "news" for item in flattened), 20)
        self.assertEqual(sum(item.get("primary_section") == "technical" for item in flattened), 20)
        self.assertEqual(sum(item.get("primary_section") == "paper" for item in flattened), 15)

    def test_v11_reading_budget_caps_primary_sections_before_rendering(self):
        def build_item(section, index):
            unique_token = chr(0x4E00 + index + (100 if section == "technical" else 0))
            return {
                "content_type": "news" if section == "news" else "project",
                "primary_section": section,
                "title_cn": f"{section} 条目 {unique_token}",
                "url": f"https://example.com/{section}/{index}",
                "analysis_body": "系统说明具体机制、输入输出、验证结果和适用边界。" * 12,
                "evidence_quality": 0.8,
                "information_density": 0.8,
                "facts": {
                    "primary_section": section,
                    "claim_type": "official_claim",
                    "who": "测试团队",
                    "action": "发布",
                    "target": f"功能 {unique_token}",
                    "source_excerpt": "原始文档说明具体机制和测试结果。",
                    "evidence_locator": "官方文档第 2 节",
                },
            }

        result = apply_v8_reading_budget(
            {
                "must_read": [build_item("news", index) for index in range(30)],
                "physical_ai": [],
                "watch": [build_item("technical", index) for index in range(28)],
                "featured_papers": [],
                "paper_appendix": [],
                "research": [],
                "brief": [],
            },
            {
                "product_mode": "intelligence_v11_editorial_library",
                "design_version": "v10-learning-digest",
                "news_section_limit": 24,
                "technical_section_limit": 22,
                "min_visible_news_count": 20,
                "min_visible_technical_count": 20,
            },
        )

        flattened = flatten_report_layers(result)
        self.assertEqual(sum(row.get("primary_section") == "news" for row in flattened), 24)
        self.assertEqual(sum(row.get("primary_section") == "technical" for row in flattened), 22)

    def test_v11_reading_budget_replaces_excess_supplemental_items_with_fresh_items(self):
        def build_nonpaper(section, index, supplemental):
            return {
                "content_type": "news" if section == "news" else "project",
                "primary_section": section,
                "title_cn": f"{section} 候选 {index}",
                "url": f"https://example.com/{section}/{index}",
                "analysis_body": "系统解释具体机制、输入输出、验证结果和适用边界。" * 12,
                "score": 100 - index if supplemental else 10 - index / 100,
                "evidence_quality": 0.85,
                "information_density": 0.85,
                "quality_flags": ["supplemental_older_source"] if supplemental else [],
                "facts": {
                    "primary_section": section,
                    "claim_type": "official_claim",
                    "who": "测试团队",
                    "action": "发布",
                    "target": f"{section} 候选 {index}",
                    "source_excerpt": "原始文档说明具体机制、输入输出和验证结果。",
                    "evidence_locator": "官方文档第 2 节",
                },
            }

        paper_plain = (
            "这项研究先解释连续控制为什么容易累积误差。作者采用两阶段方法，先预测环境状态，再根据预测结果生成动作。"
            "实验把新方法与直接动作预测基线进行比较，并说明这种中间预测能够改善长期任务表现。"
        )
        paper_intro = (
            "方法先训练状态预测模块，再把预测表示交给动作决策模块，并用一致性约束限制长期漂移。"
            "实验在公开机器人基准上与直接预测基线比较，任务成功率提高十二个百分点。"
            "结果支持中间状态建模有效，但真实环境中的长期稳定性仍需继续验证。"
        )

        candidates = []
        for section in ("news", "technical"):
            candidates.extend(build_nonpaper(section, index, index < 6) for index in range(26))
        for index in range(19):
            candidates.append({
                "content_type": "paper",
                "primary_section": "paper",
                "title_cn": f"论文候选 {index}",
                "url": f"https://arxiv.org/abs/2609.{index:05d}",
                "paper_plain_summary": paper_plain,
                "paper_technical_intro": paper_intro,
                "quality_flags": ["supplemental_older_source"] if index < 4 else [],
                "facts": {
                    "primary_section": "paper",
                    "claim_type": "research_result",
                    "who": "测试论文团队",
                    "action": "提出",
                    "target": f"状态预测方法 {index}",
                    "method": "两阶段状态预测和动作决策",
                    "metric_result": "任务成功率提高十二个百分点",
                    "evidence": ["公开基准上的任务成功率提高十二个百分点。"],
                },
            })

        result = apply_v8_reading_budget(
            {
                "must_read": candidates,
                "physical_ai": [],
                "watch": [],
                "featured_papers": [],
                "paper_appendix": [],
                "research": [],
                "brief": [],
            },
            {
                "product_mode": "intelligence_v11_editorial_library",
                "design_version": "v10-learning-digest",
                "must_read_limit": 6,
                "news_section_limit": 24,
                "technical_section_limit": 24,
                "min_visible_news_count": 20,
                "min_visible_technical_count": 20,
                "paper_featured_limit": 10,
                "paper_appendix_limit": 15,
                "supplemental_visible_max_by_section": {"news": 5, "technical": 5, "paper": 3},
            },
        )

        flattened = flatten_report_layers(result)
        supplemental_counts = Counter(
            report_primary_section(row)
            for row in flattened
            if "supplemental_older_source" in set(row.get("quality_flags") or [])
        )
        self.assertEqual(sum(report_primary_section(row) == "news" for row in flattened), 24)
        self.assertEqual(sum(report_primary_section(row) == "technical" for row in flattened), 24)
        self.assertEqual(sum(report_primary_section(row) == "paper" for row in flattened), 18)
        self.assertEqual(supplemental_counts, {"news": 5, "technical": 5, "paper": 3})

    def test_v11_final_html_gate_reads_rendered_section_counts(self):
        html = (
            '<div class="v11-count-value">18</div><div class="v11-count-label">本期新闻 / 博客 / 访谈</div>'
            '<div class="v11-count-value">10</div><div class="v11-count-label">本期非论文技术内容</div>'
            '<div class="v11-count-value">0</div><div class="v11-count-label">本期论文</div>'
        )
        metrics = scan_final_html_quality(
            html,
            {
                "must_read": [],
                "physical_ai": [],
                "watch": [],
                "featured_papers": [],
                "paper_appendix": [],
                "research": [],
                "brief": [],
            },
            quality_config={"min_visible_news_count": 20},
            report_config={
                "product_mode": "intelligence_v11_editorial_library",
                "design_version": "v10-learning-digest",
                "min_visible_news_count": 20,
                "min_visible_technical_count": 20,
                "min_visible_paper_count": 15,
                "paper_technical_intro_min_count": 0,
                "total_visible_chars_min": 0,
                "total_visible_chars_max": 10000,
            },
        )

        self.assertEqual(metrics["v11_rendered_news_count"], 18)
        self.assertEqual(metrics["v11_rendered_technical_count"], 10)
        self.assertEqual(metrics["v11_rendered_paper_count"], 0)
        self.assertEqual(metrics["final_html_quality_status"], "failed")

    def test_v11_final_html_gate_requires_visible_supplemental_date_label(self):
        item = {
            "title_cn": "补充技术材料",
            "url": "https://example.com/supplemental",
            "content_type": "news",
            "primary_section": "news",
            "quality_flags": ["supplemental_older_source"],
            "facts": {"who": "团队", "action": "发布", "target": "技术材料"},
        }
        metrics = scan_final_html_quality(
            "<html><body>补充技术材料</body></html>",
            {
                "must_read": [item],
                "physical_ai": [],
                "watch": [],
                "featured_papers": [],
                "paper_appendix": [],
                "research": [],
                "brief": [],
            },
            quality_config={},
            report_config={
                "product_mode": "intelligence_v11_editorial_library",
                "design_version": "v10-learning-digest",
                "min_visible_news_count": 0,
                "min_visible_technical_count": 0,
                "min_visible_paper_count": 0,
                "paper_technical_intro_min_count": 0,
                "total_visible_chars_min": 0,
                "total_visible_chars_max": 10000,
            },
        )

        self.assertEqual(metrics["v11_supplemental_expected_count"], 1)
        self.assertEqual(metrics["v11_supplemental_label_count"], 0)
        self.assertEqual(metrics["final_html_quality_status"], "failed")

    def test_v11_final_html_gate_limits_old_supplemental_items_per_section(self):
        items = [
            {
                "title_cn": f"补充新闻材料 {index}",
                "url": f"https://example.com/supplemental/{index}",
                "content_type": "news",
                "primary_section": "news",
                "quality_flags": ["supplemental_older_source"],
                "facts": {"who": "团队", "action": "发布", "target": f"材料 {index}"},
            }
            for index in range(6)
        ]
        metrics = scan_final_html_quality(
            "<html><body>" + ("补充阅读 · 2026-09-10 " * 6) + "</body></html>",
            {
                "must_read": items,
                "physical_ai": [],
                "watch": [],
                "featured_papers": [],
                "paper_appendix": [],
                "research": [],
                "brief": [],
            },
            quality_config={},
            report_config={
                "product_mode": "intelligence_v11_editorial_library",
                "design_version": "v10-learning-digest",
                "supplemental_visible_max_by_section": {"news": 5, "technical": 5, "paper": 3},
                "min_visible_news_count": 0,
                "min_visible_technical_count": 0,
                "min_visible_paper_count": 0,
                "paper_technical_intro_min_count": 0,
                "total_visible_chars_min": 0,
                "total_visible_chars_max": 10000,
            },
        )

        self.assertEqual(metrics["v11_supplemental_counts_by_section"], {"news": 6})
        self.assertEqual(
            metrics["v11_supplemental_limit_exceeded"],
            {"news": {"count": 6, "max": 5}},
        )
        self.assertEqual(metrics["final_html_quality_status"], "failed")

    def test_v11_final_html_gate_rejects_overage_supplemental_items(self):
        item = {
            "title_cn": "过期补充新闻材料",
            "url": "https://example.com/old-supplemental",
            "content_type": "news",
            "primary_section": "news",
            "publish_date": "2026-01-01",
            "quality_flags": ["supplemental_older_source"],
            "facts": {"who": "团队", "action": "发布", "target": "旧材料"},
        }
        metrics = scan_final_html_quality(
            "<html><body>补充阅读 · 2026-01-01 过期补充新闻材料</body></html>",
            {
                "must_read": [item],
                "physical_ai": [],
                "watch": [],
                "featured_papers": [],
                "paper_appendix": [],
                "research": [],
                "brief": [],
            },
            quality_config={},
            report_config={
                "product_mode": "intelligence_v11_editorial_library",
                "design_version": "v10-learning-digest",
                "supplemental_visible_max_by_section": {"news": 5, "technical": 5, "paper": 3},
                "supplemental_max_age_hours_by_section": {"news": 168, "technical": 720, "paper": 720},
                "min_visible_news_count": 0,
                "min_visible_technical_count": 0,
                "min_visible_paper_count": 0,
                "paper_technical_intro_min_count": 0,
                "total_visible_chars_min": 0,
                "total_visible_chars_max": 10000,
            },
        )

        self.assertEqual(metrics["v11_supplemental_age_violation_count"], 1)
        self.assertEqual(
            metrics["v11_supplemental_age_violation_examples"][0]["max_age_hours"],
            168,
        )
        self.assertEqual(metrics["final_html_quality_status"], "failed")

    def test_v11_final_html_gate_requires_approved_body_to_remain_visible(self):
        item = {
            "title_cn": "可见标题",
            "url": "https://example.com/fidelity",
            "content_type": "news",
            "primary_section": "news",
            "analysis_body": "这段已经审核通过的完整中文整理稿必须原样进入最终邮件。",
            "facts": {"who": "团队", "action": "发布", "target": "系统"},
        }
        render_key = editorial_item_render_key(item)
        metrics = scan_final_html_quality(
            (
                "<html><body>"
                f"<article data-v11-item-key='{render_key}'>"
                "<h3>可见标题</h3><p>正文被模板替换。</p>"
                "</article></body></html>"
            ),
            {
                "must_read": [item],
                "physical_ai": [],
                "watch": [],
                "featured_papers": [],
                "paper_appendix": [],
                "research": [],
                "brief": [],
            },
            quality_config={},
            report_config={
                "product_mode": "intelligence_v11_editorial_library",
                "design_version": "v10-learning-digest",
                "paper_technical_intro_min_count": 0,
                "total_visible_chars_min": 0,
                "total_visible_chars_max": 10000,
            },
        )

        self.assertEqual(metrics["v11_content_fidelity_missing_count"], 1)
        self.assertEqual(
            metrics["v11_content_fidelity_missing_examples"][0]["fields"],
            ["analysis_body"],
        )
        self.assertEqual(metrics["final_html_quality_status"], "failed")

    def test_v11_final_html_gate_requires_key_numbers_in_public_editorial_copy(self):
        item = {
            "title_cn": "垂直人工智能基金扩展跨地区投资",
            "url": "https://example.com/key-number-fidelity",
            "content_type": "news",
            "primary_section": "news",
            "analysis_version": "codex-research-v3",
            "analysis_body": "基金面向多个地区的早期垂直人工智能企业，并提供首次及后续投资。",
            "facts": {
                "who": "AI Seed",
                "action": "启动",
                "target": "新一期垂直人工智能基金",
                "key_numbers": ["基金规模5000万美元"],
                "source_excerpt": "公告披露基金规模为5000万美元。",
                "evidence_locator": "公告正文",
            },
        }
        render_key = editorial_item_render_key(item)
        html = (
            "<html><body>"
            f"<article data-v11-item-key='{render_key}'>"
            f"<h3>{item['title_cn']}</h3>"
            f"<p>{item['analysis_body']}</p>"
            "<p>公告披露基金规模为5000万美元。 公告正文</p>"
            "</article></body></html>"
        )

        metrics = scan_final_html_quality(
            html,
            {
                "must_read": [item],
                "physical_ai": [],
                "watch": [],
                "featured_papers": [],
                "paper_appendix": [],
                "research": [],
                "brief": [],
            },
            quality_config={},
            report_config={
                "product_mode": "intelligence_v11_editorial_library",
                "design_version": "v11-editorial-library",
                "min_visible_news_count": 0,
                "min_visible_technical_count": 0,
                "min_visible_paper_count": 0,
                "paper_technical_intro_min_count": 0,
                "total_visible_chars_min": 0,
                "total_visible_chars_max": 10000,
            },
        )

        self.assertEqual(metrics["v11_content_fidelity_missing_count"], 0)
        self.assertEqual(metrics["v11_key_number_fidelity_missing_count"], 1)
        self.assertEqual(
            metrics["v11_key_number_fidelity_missing_examples"][0]["missing_tokens"],
            ["usd:50000000"],
        )
        self.assertEqual(metrics["final_html_quality_status"], "failed")

    def test_v11_final_html_gate_rejects_flattened_editorial_paragraphs(self):
        body = (
            "第一段交代训练数据和模型输入。\n\n"
            "第二段解释两个模块如何交换状态。\n\n"
            "第三段给出基准结果和当前限制。"
        )
        normalized_paragraphs = "\n".join(
            part.strip() for part in body.split("\n\n")
        )
        item = {
            "title_cn": "团队公开模块化世界模型训练链路",
            "url": "https://example.com/paragraph-fidelity",
            "content_type": "technical",
            "primary_section": "technical",
            "analysis_version": "codex-research-v2",
            "analysis_body": body,
            "facts": {
                "who": "团队",
                "action": "公开",
                "target": "世界模型训练链路",
                "editorial_source_hashes": {
                    "title_cn": hashlib.sha256("团队公开模块化世界模型训练链路".encode("utf-8")).hexdigest(),
                    "summary": hashlib.sha256(" ".join(body.split()).encode("utf-8")).hexdigest(),
                },
                "editorial_paragraph_hashes": {
                    "summary": hashlib.sha256(normalized_paragraphs.encode("utf-8")).hexdigest(),
                },
            },
        }
        render_key = editorial_item_render_key(item)
        flattened = " ".join(body.split())
        metrics = scan_final_html_quality(
            (
                "<html><body>"
                f"<article data-v11-item-key='{render_key}'>"
                f"<h3>{item['title_cn']}</h3><div>{flattened}</div>"
                "</article></body></html>"
            ),
            {
                "must_read": [item],
                "physical_ai": [],
                "watch": [],
                "featured_papers": [],
                "paper_appendix": [],
                "research": [],
                "brief": [],
            },
            quality_config={},
            report_config={
                "product_mode": "intelligence_v11_editorial_library",
                "design_version": "v11-editorial-library",
                "min_visible_news_count": 0,
                "min_visible_technical_count": 0,
                "min_visible_paper_count": 0,
                "paper_technical_intro_min_count": 0,
                "total_visible_chars_min": 0,
                "total_visible_chars_max": 10000,
            },
        )

        self.assertEqual(metrics["v11_content_fidelity_missing_count"], 1)
        self.assertEqual(
            metrics["v11_content_fidelity_missing_examples"][0]["fields"],
            ["analysis_body:paragraphs"],
        )
        self.assertEqual(metrics["v11_editorial_source_mismatch_count"], 0)
        self.assertEqual(metrics["final_html_quality_status"], "failed")

    def test_v11_final_html_gate_checks_copy_inside_its_own_item_container(self):
        item = {
            "title_cn": "团队发布可追踪的智能体运行时",
            "url": "https://example.com/container-fidelity",
            "content_type": "news",
            "primary_section": "news",
            "analysis_body": "系统先校验工具权限，再保存每一步调用记录和失败恢复状态。",
            "facts": {"who": "团队", "action": "发布", "target": "智能体运行时"},
        }
        render_key = editorial_item_render_key(item)
        html = (
            "<html><body>"
            f"<div class='directory'>{item['analysis_body']}</div>"
            f"<article data-v11-item-key='{render_key}'>"
            f"<h3>{item['title_cn']}</h3><p>系统先校验工具权限。</p>"
            "</article></body></html>"
        )

        metrics = scan_final_html_quality(
            html,
            {
                "must_read": [item],
                "physical_ai": [],
                "watch": [],
                "featured_papers": [],
                "paper_appendix": [],
                "research": [],
                "brief": [],
            },
            quality_config={},
            report_config={
                "product_mode": "intelligence_v11_editorial_library",
                "design_version": "v11-editorial-library",
                "min_visible_news_count": 0,
                "min_visible_technical_count": 0,
                "min_visible_paper_count": 0,
                "paper_technical_intro_min_count": 0,
                "total_visible_chars_min": 0,
                "total_visible_chars_max": 10000,
            },
        )

        self.assertEqual(metrics["v11_content_fidelity_missing_count"], 1)
        self.assertEqual(
            metrics["v11_content_fidelity_missing_examples"][0]["fields"],
            ["analysis_body"],
        )
        self.assertFalse(
            metrics["v11_content_fidelity_missing_examples"][0]["container_missing"]
        )

    def test_v11_final_html_gate_accepts_escaped_copy_in_matching_container(self):
        item = {
            "title_cn": "团队公开 A&B 推理架构",
            "url": "https://example.com/escaped-fidelity",
            "content_type": "technical",
            "primary_section": "technical",
            "analysis_body": "运行时把输入限制为 <8K token，并记录缓存命中与恢复结果。",
            "facts": {"who": "团队", "action": "公开", "target": "推理架构"},
        }
        render_key = editorial_item_render_key(item)
        html = (
            "<html><body>"
            f"<article data-v11-item-key='{render_key}'>"
            "<h3>团队公开 A&amp;B 推理架构</h3>"
            "<p>运行时把输入限制为 &lt;8K token，并记录缓存命中与恢复结果。</p>"
            "</article></body></html>"
        )

        metrics = scan_final_html_quality(
            html,
            {
                "must_read": [item],
                "physical_ai": [],
                "watch": [],
                "featured_papers": [],
                "paper_appendix": [],
                "research": [],
                "brief": [],
            },
            quality_config={},
            report_config={
                "product_mode": "intelligence_v11_editorial_library",
                "design_version": "v11-editorial-library",
                "min_visible_news_count": 0,
                "min_visible_technical_count": 0,
                "min_visible_paper_count": 0,
                "paper_technical_intro_min_count": 0,
                "total_visible_chars_min": 0,
                "total_visible_chars_max": 10000,
            },
        )

        self.assertEqual(metrics["v11_content_fidelity_missing_count"], 0)

    def test_v11_final_html_gate_uses_top_level_evidence_when_facts_omit_it(self):
        item = {
            "title_cn": "团队公开新的推理缓存架构",
            "url": "https://example.com/top-level-evidence",
            "content_type": "technical",
            "primary_section": "technical",
            "analysis_body": "系统识别共享前缀后复用计算缓存，并公开了不同并发条件下的延迟结果。",
            "source_excerpt": "工程文档列出了共享前缀缓存的并发测试结果。",
            "evidence_locator": "性能评测第 3 节",
            "facts": {"who": "团队", "action": "公开", "target": "推理缓存架构"},
        }
        render_key = editorial_item_render_key(item)
        html = (
            "<html><body>"
            f"<article data-v11-item-key='{render_key}'>"
            f"<h3>{item['title_cn']}</h3>"
            f"<p>{item['analysis_body']}</p>"
            f"<p>{item['source_excerpt']} {item['evidence_locator']}</p>"
            "</article></body></html>"
        )

        metrics = scan_final_html_quality(
            html,
            {
                "must_read": [item],
                "physical_ai": [],
                "watch": [],
                "featured_papers": [],
                "paper_appendix": [],
                "research": [],
                "brief": [],
            },
            quality_config={},
            report_config={
                "product_mode": "intelligence_v11_editorial_library",
                "design_version": "v11-editorial-library",
                "min_visible_news_count": 0,
                "min_visible_technical_count": 0,
                "min_visible_paper_count": 0,
                "paper_technical_intro_min_count": 0,
                "total_visible_chars_min": 0,
                "total_visible_chars_max": 10000,
            },
        )

        self.assertEqual(metrics["v11_content_fidelity_missing_count"], 0)

        missing_metrics = scan_final_html_quality(
            html.replace(item["source_excerpt"], "证据摘录被模板遗漏。"),
            {
                "must_read": [item],
                "physical_ai": [],
                "watch": [],
                "featured_papers": [],
                "paper_appendix": [],
                "research": [],
                "brief": [],
            },
            quality_config={},
            report_config={
                "product_mode": "intelligence_v11_editorial_library",
                "design_version": "v11-editorial-library",
                "min_visible_news_count": 0,
                "min_visible_technical_count": 0,
                "min_visible_paper_count": 0,
                "paper_technical_intro_min_count": 0,
                "total_visible_chars_min": 0,
                "total_visible_chars_max": 10000,
            },
        )

        self.assertEqual(missing_metrics["v11_content_fidelity_missing_count"], 1)
        self.assertEqual(
            missing_metrics["v11_content_fidelity_missing_examples"][0]["fields"],
            ["source_excerpt"],
        )

    def test_v11_delivery_volume_fidelity_rejects_truncated_sent_volume(self):
        item = {
            "title_cn": "团队公开新的推理缓存架构",
            "url": "https://example.com/volume-fidelity",
            "content_type": "technical",
            "primary_section": "technical",
            "analysis_body": "系统先识别共享前缀，再复用已计算缓存，并公开不同并发条件下的延迟结果。",
            "facts": {"who": "团队", "action": "公开", "target": "推理缓存架构"},
        }
        render_key = editorial_item_render_key(item)
        layers = {
            "must_read": [item],
            "physical_ai": [],
            "watch": [],
            "featured_papers": [],
            "paper_appendix": [],
            "research": [],
            "brief": [],
        }
        metrics = scan_v11_delivery_volume_fidelity(
            [
                {
                    "html": (
                        f"<article data-v11-item-key='{render_key}'>"
                        f"<h3>{item['title_cn']}</h3><p>系统先识别共享前缀。</p>"
                        "</article>"
                    ),
                    "layers": layers,
                }
            ],
            quality_config={},
            report_config={
                "product_mode": "intelligence_v11_editorial_library",
                "design_version": "v11-editorial-library",
            },
        )

        self.assertEqual(
            metrics["email_delivery_volume_content_fidelity_missing_counts"],
            [1],
        )
        self.assertEqual(
            metrics["email_delivery_volume_content_fidelity_missing_count"],
            1,
        )
        self.assertEqual(
            metrics["email_delivery_volume_content_fidelity_missing_examples"][0][
                "volume"
            ],
            1,
        )

    def test_v11_final_html_gate_detects_editorial_copy_changed_after_ingest(self):
        title = "团队发布具备权限边界的智能体运行时"
        approved_body = "团队发布新的智能体运行时。系统先校验工具权限，再记录调用输入输出。官方文档给出了部署范围和失败恢复条件。"
        source_excerpt = "官方文档显示，运行时会先校验工具权限并记录调用输入输出。"
        approved_locator = "官方文档第 2 节"
        current_locator = "新闻稿首页"

        def digest(value):
            return hashlib.sha256(value.encode("utf-8")).hexdigest()

        item = {
            "title_cn": title,
            "url": "https://example.com/source-fidelity",
            "content_type": "news",
            "primary_section": "news",
            "analysis_version": "codex-research-v2",
            "analysis_body": approved_body.replace("失败恢复条件", "商业影响"),
            "facts": {
                "who": "团队",
                "action": "发布",
                "target": "智能体运行时",
                "source_excerpt": source_excerpt,
                "evidence_locator": current_locator,
                "editorial_source_hashes": {
                    "title_cn": digest(title),
                    "summary": digest(approved_body),
                    "source_excerpt": digest(source_excerpt),
                    "evidence_locator": digest(approved_locator),
                },
            },
        }
        render_key = editorial_item_render_key(item)
        metrics = scan_final_html_quality(
            (
                "<html><body>"
                f"<article data-v11-item-key='{render_key}'>"
                f"{title} {item['analysis_body']} {source_excerpt} {current_locator}"
                "</article></body></html>"
            ),
            {
                "must_read": [item],
                "physical_ai": [],
                "watch": [],
                "featured_papers": [],
                "paper_appendix": [],
                "research": [],
                "brief": [],
            },
            quality_config={},
            report_config={
                "product_mode": "intelligence_v11_editorial_library",
                "design_version": "v10-learning-digest",
                "min_visible_news_count": 0,
                "min_visible_technical_count": 0,
                "min_visible_paper_count": 0,
                "paper_technical_intro_min_count": 0,
                "total_visible_chars_min": 0,
                "total_visible_chars_max": 10000,
            },
        )

        self.assertEqual(metrics["v11_editorial_source_hash_missing_count"], 0)
        self.assertEqual(metrics["v11_editorial_source_mismatch_count"], 1)
        self.assertEqual(
            metrics["v11_editorial_source_mismatch_examples"][0]["fields"],
            ["summary", "evidence_locator"],
        )
        self.assertEqual(metrics["v11_content_fidelity_missing_count"], 0)
        self.assertEqual(metrics["final_html_quality_status"], "failed")

    def test_v11_final_html_gate_detects_key_number_set_changed_after_ingest(self):
        title = "基金公布新一期垂直人工智能投资计划"
        body = "基金公告披露规模为5000万美元，并面向多个地区的早期企业。"
        source_excerpt = "公告披露基金规模为5000万美元。"

        def digest(value):
            return hashlib.sha256(value.encode("utf-8")).hexdigest()

        item = {
            "title_cn": title,
            "url": "https://example.com/key-number-source-fidelity",
            "content_type": "news",
            "primary_section": "news",
            "analysis_version": "codex-research-v3",
            "analysis_body": body,
            "facts": {
                "who": "AI Seed",
                "action": "公布",
                "target": "新一期基金",
                "source_excerpt": source_excerpt,
                "evidence_locator": "公告正文",
                "key_numbers": ["基金规模5000万美元"],
                "editorial_source_hashes": {
                    "title_cn": digest(title),
                    "summary": digest(body),
                    "source_excerpt": digest(source_excerpt),
                    "evidence_locator": digest("公告正文"),
                    "key_numbers": digest('["基金规模4000万美元"]'),
                },
            },
        }
        render_key = editorial_item_render_key(item)
        html = (
            "<html><body>"
            f"<article data-v11-item-key='{render_key}'>"
            f"{title} {body} {source_excerpt} 公告正文"
            "</article></body></html>"
        )

        metrics = scan_final_html_quality(
            html,
            {
                "must_read": [item],
                "physical_ai": [],
                "watch": [],
                "featured_papers": [],
                "paper_appendix": [],
                "research": [],
                "brief": [],
            },
            quality_config={},
            report_config={
                "product_mode": "intelligence_v11_editorial_library",
                "design_version": "v11-editorial-library",
                "min_visible_news_count": 0,
                "min_visible_technical_count": 0,
                "min_visible_paper_count": 0,
                "paper_technical_intro_min_count": 0,
                "total_visible_chars_min": 0,
                "total_visible_chars_max": 10000,
            },
        )

        self.assertEqual(metrics["v11_key_number_fidelity_missing_count"], 0)
        self.assertEqual(metrics["v11_editorial_source_mismatch_count"], 1)
        self.assertEqual(
            metrics["v11_editorial_source_mismatch_examples"][0]["fields"],
            ["key_numbers"],
        )
        self.assertEqual(metrics["final_html_quality_status"], "failed")

    def test_v11_final_html_gate_requires_one_visible_claim_label_per_item(self):
        item = {
            "title_cn": "团队发布可核验系统更新",
            "url": "https://example.com/claim-label",
            "content_type": "news",
            "primary_section": "news",
            "analysis_body": "团队发布系统更新，原始公告列出了部署范围和已验证结果。",
            "claim_type": "official_claim",
            "facts": {
                "claim_type": "official_claim",
                "who": "团队",
                "action": "发布",
                "target": "系统更新",
            },
        }
        metrics = scan_final_html_quality(
            "<html><body>团队发布可核验系统更新 团队发布系统更新，原始公告列出了部署范围和已验证结果。</body></html>",
            {
                "must_read": [item],
                "physical_ai": [],
                "watch": [],
                "featured_papers": [],
                "paper_appendix": [],
                "research": [],
                "brief": [],
            },
            quality_config={},
            report_config={
                "product_mode": "intelligence_v11_editorial_library",
                "design_version": "v10-learning-digest",
                "min_visible_news_count": 0,
                "min_visible_technical_count": 0,
                "min_visible_paper_count": 0,
                "paper_technical_intro_min_count": 0,
                "total_visible_chars_min": 0,
                "total_visible_chars_max": 10000,
            },
        )

        self.assertEqual(metrics["v11_claim_label_expected_count"], 1)
        self.assertEqual(metrics["v11_claim_label_visible_count"], 0)
        self.assertEqual(metrics["final_html_quality_status"], "failed")

    def test_v8_truncated_paper_intro_is_rebuilt_before_featured_section(self):
        paper = enrich_editorial_fields(
            {
                "url": "https://arxiv.org/abs/truncated",
                "title_cn": "TruncatedPaper提出动作预测框架",
                "content_type": "paper",
                "source_detail": "arXiv",
                "source_tier": "research",
                "evidence_quality": 0.85,
                "information_density": 0.85,
                "paper_technical_intro": "TruncatedPaper的关键做法是先预测状态…实验结果是成功率达到82%。",
                "facts": {
                    "who": "TruncatedPaper",
                    "action": "提出",
                    "target": "动作预测框架",
                    "method": "先预测状态…再生成动作",
                    "metric_result": "成功率达到82%",
                    "evidence": ["在V8Bench上成功率达到82%"],
                },
            }
        )
        result = apply_v8_reading_budget(
            {
                "must_read": [],
                "physical_ai": [],
                "watch": [],
                "featured_papers": [paper],
                "paper_appendix": [],
                "research": [],
                "brief": [],
            },
            {"design_version": "v8-editorial-reader", "total_visible_chars_min": 0},
        )
        self.assertEqual([item["url"] for item in result["featured_papers"]], [paper["url"]])
        rebuilt = result["featured_papers"][0]["paper_technical_intro"]
        self.assertNotIn("…", rebuilt)
        self.assertIn("82%", rebuilt)

    def test_v8_featured_paper_respects_configured_intro_length_range(self):
        base = {
            "content_type": "paper",
            "source_detail": "arXiv",
            "source_tier": "research",
            "evidence_quality": 0.85,
            "information_density": 0.85,
            "facts": {
                "who": "RangePaper",
                "action": "提出",
                "target": "状态预测方法",
                "method": "先编码状态，再预测动作",
                "metric_result": "成功率达到82%",
                "evidence": ["实验成功率达到82%"],
            },
        }
        short = enrich_editorial_fields(
            {
                **base,
                "url": "https://arxiv.org/abs/short-intro",
                "title_cn": "RangePaper短技术介绍",
                "paper_technical_intro": "方法先编码状态，再预测动作；实验成功率达到82%。",
            }
        )
        qualified = enrich_editorial_fields(
            {
                **base,
                "url": "https://arxiv.org/abs/qualified-intro",
                "title_cn": "RangePaper完整技术介绍",
                "paper_technical_intro": (
                    "论文先把连续状态编码成紧凑表示，再由动作预测模块读取历史状态，避免直接从单帧生成控制指令。"
                    "实验在统一任务设置下与直接预测基线比较，成功率达到82%，并报告不同状态窗口的消融结果，说明历史状态确实提供了有效约束。"
                    "论文同时给出失败案例，主要局限是长时任务中的误差仍会逐步累积。"
                ),
            }
        )
        short["paper_technical_intro"] = "方法先编码状态，再预测动作；实验成功率达到82%。"
        qualified["paper_technical_intro"] = (
            "论文先把连续状态编码成紧凑表示，再由动作预测模块读取历史状态，避免直接从单帧生成控制指令。"
            "实验在统一任务设置下与直接预测基线比较，成功率达到82%，并报告不同状态窗口的消融结果，说明历史状态确实提供了有效约束。"
            "论文同时给出失败案例，主要局限是长时任务中的误差仍会逐步累积。"
            "这组对照把收益定位到状态历史，而不是额外参数量或更大的训练集。"
        )
        with patch("main.enrich_editorial_fields", side_effect=lambda item: dict(item)):
            result = apply_v8_reading_budget(
                {
                    "must_read": [],
                    "physical_ai": [],
                    "watch": [],
                    "featured_papers": [short, qualified],
                    "paper_appendix": [],
                    "research": [],
                    "brief": [],
                },
                {
                    "design_version": "v8-editorial-reader",
                    "paper_body_char_min": 120,
                    "paper_body_char_limit": 180,
                    "paper_featured_limit": 2,
                    "paper_appendix_limit": 2,
                    "total_visible_chars_min": 0,
                },
            )
        self.assertEqual(
            [item["url"] for item in result["featured_papers"]],
            [qualified["url"]],
        )
        self.assertEqual(
            [item["url"] for item in result["paper_appendix"]],
            [short["url"]],
        )

    def test_v8_featured_paper_excludes_title_fact_mismatch(self):
        intro = (
            "论文先编码连续状态，再由动作预测模块读取历史窗口并生成控制指令。"
            "实验在统一基准上与单帧策略比较，成功率达到82%，结果说明状态历史能减少动作漂移。"
        )
        common = {
            "content_type": "paper",
            "source_detail": "arXiv",
            "source_tier": "research",
            "evidence_quality": 0.85,
            "information_density": 0.85,
            "paper_technical_intro": intro,
        }
        mismatched = {
            **common,
            "url": "https://arxiv.org/abs/mismatched-focus",
            "title": "Vision Policy for Dexterous Robot Manipulation",
            "title_cn": "Vision Policy for Dexterous Robot Manipulation",
            "score": 10,
            "facts": {
                "who": "FinanceAgent",
                "action": "发布",
                "target": "金融问答系统",
                "method": "编码连续状态并预测动作",
                "metric_result": "成功率达到82%",
                "evidence": ["实验成功率达到82%"],
            },
        }
        qualified = {
            **common,
            "url": "https://arxiv.org/abs/qualified-focus",
            "title": "StatePolicy提出机器人动作预测方法",
            "title_cn": "StatePolicy提出机器人动作预测方法",
            "score": 9,
            "facts": {
                "who": "StatePolicy",
                "action": "提出",
                "target": "机器人动作预测方法",
                "method": "编码连续状态并预测动作",
                "metric_result": "成功率达到82%",
                "evidence": ["实验成功率达到82%"],
            },
        }
        mismatched["facts_cn"] = dict(mismatched["facts"])
        qualified["facts_cn"] = dict(qualified["facts"])
        with patch("main.enrich_editorial_fields", side_effect=lambda item: dict(item)):
            result = apply_v8_reading_budget(
                {
                    "must_read": [],
                    "physical_ai": [],
                    "watch": [],
                    "featured_papers": [mismatched, qualified],
                    "paper_appendix": [],
                    "research": [],
                    "brief": [],
                },
                {
                    "design_version": "v8-editorial-reader",
                    "paper_body_char_min": 0,
                    "paper_body_char_limit": 220,
                    "paper_featured_limit": 1,
                    "paper_appendix_limit": 2,
                    "total_visible_chars_min": 0,
                },
            )
        self.assertEqual([item["url"] for item in result["featured_papers"]], [qualified["url"]])
        self.assertNotIn(mismatched["url"], [item["url"] for item in result["featured_papers"]])

    def test_v9_featured_paper_accepts_concise_substantive_intro(self):
        intro = (
            "DA-Nav先把街景图像编码成可导航状态，再用语言目标约束路线搜索，而不是直接从单帧预测动作。"
            "实验在城市级室外导航基准上与常规视觉语言导航基线比较，成功率和路径效率均有提升，结果说明结构化地图约束能减少长距离任务中的方向漂移。"
        )
        paper = enrich_editorial_fields(
            {
                "content_type": "paper",
                "url": "https://arxiv.org/abs/concise-substantive-intro",
                "title_cn": "DA-Nav：城市级室外导航",
                "source_detail": "arXiv",
                "source_tier": "research",
                "evidence_quality": 0.85,
                "information_density": 0.85,
                "facts": {
                    "who": "DA-Nav",
                    "action": "提出",
                    "target": "城市级室外导航方法",
                    "method": "将街景编码为导航状态并用语言目标约束路线搜索",
                    "metric_result": "成功率和路径效率均优于基线",
                    "evidence": ["实验在城市级室外导航基准上优于视觉语言导航基线"],
                },
                "paper_technical_intro": intro,
            }
        )
        paper["paper_technical_intro"] = intro
        self.assertGreaterEqual(len(intro), 100)
        self.assertLess(len(intro), 120)

        with patch("main.enrich_editorial_fields", side_effect=lambda item: dict(item)):
            result = apply_v8_reading_budget(
                {
                    "must_read": [],
                    "physical_ai": [],
                    "watch": [],
                    "featured_papers": [paper],
                    "paper_appendix": [],
                    "research": [],
                    "brief": [],
                },
                {
                    "design_version": "v9-continuous-learning",
                    "paper_body_char_min": 100,
                    "paper_body_char_limit": 180,
                    "paper_featured_limit": 8,
                    "paper_appendix_limit": 12,
                },
            )

        self.assertEqual([item["url"] for item in result["featured_papers"]], [paper["url"]])

    def test_v8_final_scan_blocks_truncated_focus_text(self):
        paper = {
            "url": "https://arxiv.org/abs/leaked-truncation",
            "title_cn": "LeakPaper提出状态预测框架",
            "editorial_title": "LeakPaper提出状态预测框架",
            "content_type": "paper",
            "paper_technical_intro": "LeakPaper的关键做法是预测状态…实验结果是成功率达到82%。",
            "evidence_line": "在V8Bench上成功率达到82%",
            "facts_cn": {"method": "预测状态", "metric_result": "成功率达到82%", "evidence": ["成功率达到82%"]},
            "source_detail": "arXiv",
        }
        layers = {
            "must_read": [],
            "physical_ai": [],
            "watch": [],
            "featured_papers": [paper],
            "paper_appendix": [],
            "research": [],
            "brief": [],
        }
        metrics = scan_final_html_quality(
            "<html><body>LeakPaper的关键做法是预测状态…</body></html>",
            layers,
            quality_config={"truncated_focus_text_count": 0},
            report_config={"design_version": "v8-editorial-reader", "total_visible_chars_min": 0},
        )
        self.assertEqual(metrics["truncated_focus_text_count"], 1)
        self.assertEqual(metrics["final_html_quality_status"], "failed")

    def test_v8_backfill_requires_qualified_source_and_domain_coverage(self):
        items = []
        domains = ["world_model", "physical_ai", "agent_models", "infra_open_source"]
        domain_targets = ["世界模型训练", "机器人动作策略", "智能体工作流", "推理基础设施"]
        for index in range(12):
            target = domain_targets[index % len(domain_targets)]
            items.append(
                {
                    "url": f"https://source{index % 6}.example.com/item-{index}",
                    "title_cn": f"公司{index}发布{target}更新",
                    "summary": f"公司{index}通过接口发布{target}更新，测试成功率达到{70 + index}%。",
                    "content_type": "news",
                    "domain_key": domains[index % len(domains)],
                    "source_detail": f"来源{index % 6}",
                    "source_tier": "official",
                    "evidence_quality": 0.8,
                    "information_density": 0.8,
                    "facts": {
                        "who": f"公司{index}",
                        "action": "发布",
                        "target": target,
                        "method": "通过接口接入现有系统",
                        "metric_result": f"测试成功率达到{70 + index}%",
                        "evidence": [f"测试成功率达到{70 + index}%"],
                    },
                }
            )
        config = {
            "design_version": "v8-editorial-reader",
            "v8_focus_candidate_target": 12,
            "v8_focus_source_target": 6,
            "v8_focus_domain_target": 4,
        }
        self.assertTrue(v8_backfill_pool_ready(items, config))
        self.assertFalse(v8_backfill_pool_ready(items[:8], config))

    def test_v8_update_preselection_rotates_sources_before_duplicates(self):
        updates = []
        for index, host in enumerate(("a.example", "a.example", "a.example", "b.example", "c.example", "d.example")):
            updates.append(
                {
                    "url": f"https://{host}/item-{index}",
                    "title": f"Company {index} releases verified update",
                    "summary": f"Company {index} reports a benchmark result of {80 + index}% for the update.",
                    "score": 10 - index,
                    "evidence_quality": 0.8,
                    "information_density": 0.8,
                    "source_detail": host,
                    "category": "Infrastructure",
                }
            )
        selected = filter_updates_for_report(
            updates,
            4,
            4,
            source_preferences={},
            preference_config={},
            diversify_sources=True,
        )
        self.assertEqual(len({item["url"].split("/")[2] for item in selected}), 4)

    def test_v8_fixed_real_samples_pass_editorial_regression(self):
        fixture_path = Path(__file__).parent / "fixtures" / "v8_real_samples.json"
        payload = json.loads(fixture_path.read_text(encoding="utf-8"))
        self.assertEqual(len(payload["papers"]), 20)
        self.assertEqual(len(payload["news"]), 20)

        decorated_papers = [enrich_editorial_fields({**item, "_v8_force_fact_title": True}) for item in payload["papers"]]
        decorated_news = [enrich_editorial_fields({**item, "_v8_force_fact_title": True}) for item in payload["news"]]
        qualifying_papers = []
        for item in decorated_papers:
            facts_cn = item.get("facts_cn") or {}
            if facts_cn.get("method") and facts_cn.get("metric_result"):
                qualifying_papers.append(item)
                self.assertTrue(paper_technical_intro_passes(item.get("paper_technical_intro")), item.get("title_cn"))
                self.assertFalse(has_untranslated_prose(item.get("paper_technical_intro")), item.get("title_cn"))

        self.assertGreaterEqual(len(qualifying_papers), 8)
        for item in decorated_news:
            if item.get("quality_tier") == "brief":
                continue
            self.assertFalse(title_looks_bad(item), item.get("title_cn"))
            self.assertFalse(has_untranslated_prose(item.get("analysis_body")), item.get("title_cn"))

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

    def test_environment_path_overrides_isolate_generated_artifacts(self):
        config = {
            "archive": {
                "enabled": True,
                "report_dir": "archive",
                "output_html": "reports_index.html",
                "output_markdown": "reports_index.md",
            },
            "scheduler": {"ui_audit_output_dir": "artifacts/production_audit"},
        }
        environment = {
            "WEB_AGENT_REPORT_DIR": "artifacts/integration/archive",
            "WEB_AGENT_ARCHIVE_OUTPUT_HTML": "artifacts/integration/index.html",
            "WEB_AGENT_ARCHIVE_OUTPUT_MARKDOWN": "artifacts/integration/index.md",
            "WEB_AGENT_UI_AUDIT_OUTPUT_DIR": "artifacts/integration/ui",
            "WEB_AGENT_ARCHIVE_ENABLED": "false",
        }

        with patch.dict(os.environ, environment, clear=False):
            overridden = apply_environment_path_overrides(config)

        self.assertEqual(overridden["archive"]["report_dir"], environment["WEB_AGENT_REPORT_DIR"])
        self.assertEqual(overridden["archive"]["output_html"], environment["WEB_AGENT_ARCHIVE_OUTPUT_HTML"])
        self.assertEqual(
            overridden["archive"]["output_markdown"],
            environment["WEB_AGENT_ARCHIVE_OUTPUT_MARKDOWN"],
        )
        self.assertFalse(overridden["archive"]["enabled"])
        self.assertEqual(
            overridden["scheduler"]["ui_audit_output_dir"],
            environment["WEB_AGENT_UI_AUDIT_OUTPUT_DIR"],
        )
        self.assertEqual(config["archive"]["report_dir"], "archive")

    def test_validation_and_report_only_runs_do_not_pollute_collector_history(self):
        self.assertTrue(should_persist_collector_history(False, ""))
        self.assertFalse(should_persist_collector_history(False, "validation_fast"))
        self.assertFalse(should_persist_collector_history(True, ""))

    def test_config_expands_world_model_training_and_interview_sources(self):
        config = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
        searches = config["sources"]["web_search"]["searches"]
        world_model_sources = {
            item["name"]: item
            for item in searches
            if "World Model" in item["name"] or "JEPA" in item.get("query", "")
        }

        self.assertIn("World Model Training Tech DeepMind", world_model_sources)
        self.assertIn("World Model Training Tech Meta JEPA", world_model_sources)
        self.assertIn("World Model Robotics Policy Tech", world_model_sources)
        self.assertIn("World Model AI Leader Interviews", world_model_sources)
        self.assertIn("World Model Research Systems", world_model_sources)
        combined = " ".join(item["query"] for item in world_model_sources.values())
        for token in ("DeepMind", "Genie", "JEPA", "LeCun", "policy simulation", "video prediction"):
            self.assertIn(token, combined)
        for name in (
            "World Model Training Tech DeepMind",
            "World Model Training Tech Meta JEPA",
            "World Model AI Leader Interviews",
        ):
            self.assertGreaterEqual(max(world_model_sources[name].get("fallback_days", [])), 30)

    def test_config_uses_codex_research_inbox_without_api_runtime(self):
        config = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))

        self.assertEqual(config["llm"]["provider"], "codex_automation")
        self.assertEqual(config["llm"]["api_mode"], "research_inbox")
        self.assertNotIn("api_key_env", config["llm"])
        self.assertNotIn("deepseek", json.dumps(config["llm"]).lower())
        inbox = config["sources"]["codex_research_inbox"]
        self.assertTrue(inbox["enabled"])
        self.assertGreaterEqual(inbox["minimum_papers"], 15)
        self.assertGreaterEqual(inbox["minimum_news"], 5)

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

    def test_default_html_uses_classic_overview_and_mixed_expandable_stream(self):
        generator = ReportGenerator()
        item = {
            "url": "https://example.com/agent",
            "title_cn": "OpenAI把企业Agent接入审批工作流",
            "summary_preview": "Agent能力开始进入企业流程权限层。",
            "summary": "OpenAI把企业Agent接入审批和执行工作流，公开材料显示它覆盖流程触发、权限确认和结果回写。",
            "why_it_matters": "它值得看，是因为Agent不再只停留在聊天入口，而是进入企业流程权限层。",
            "why_now": "企业客户需要验证模型能否处理审批、检索和执行链路。",
            "expected_effect": "它会先减少人工切换，把部分重复流程交给Agent处理。",
            "future_impact": "如果客户采用稳定，Agent竞争会从模型能力延伸到权限、集成和审计能力。",
            "content_type": "news",
            "content_kind": "全网动态",
            "display_topic": "Agent / Models",
            "source_tier": "official",
            "evidence_quality": 0.8,
            "information_density": 0.8,
            "score": 9.1,
            "publish_date": "2026-06-24 13:00",
            "facts": {
                "who": "OpenAI",
                "action": "接入",
                "target": "企业Agent审批工作流",
                "evidence": ["官方材料提到流程触发、权限确认和结果回写"],
                "audience": "企业团队",
            },
        }
        html = generator.generate_html(
            papers=[],
            updates=[item],
            mixed_items=[item],
            report_summary={
                "lead_summary": "今天最值得记住的是Agent继续进入企业流程权限层。",
                "paper_summary": "",
                "update_summary": "产品动态集中在Agent工作流。",
                "hot_topics": ["Agent", "产品发布"],
                "key_takeaways": ["Agent从聊天入口推进到企业执行流程。"],
                "watchlist": ["观察客户采用和审计能力。"],
            },
            trend_summary={"items": []},
            alert_summary={"needs_alert": False, "issues": []},
            archive_summary={"entries": []},
            layered_updates={"must_read": [item], "physical_ai": [], "watch": [], "featured_papers": [], "paper_appendix": [], "brief": []},
        )

        self.assertIn('data-design-version="v7-classic-briefing"', html)
        self.assertIn("今日总览", html)
        self.assertIn("学习导航", html)
        self.assertIn("classic-nav-card", html)
        self.assertIn("classic-nav-jump", html)
        self.assertIn("classic-nav-lead", html)
        self.assertIn("进入", html)
        self.assertIn("先读：", html)
        self.assertIn("混排情报流", html)
        self.assertIn("<details", html)
        self.assertIn("展开阅读", html)
        self.assertIn("先读", html)
        self.assertIn("read-time", html)
        self.assertIn("1分钟", html)
        self.assertIn("按优先级混排", html)
        self.assertIn("为什么值得看", html)
        self.assertIn("为什么这么做", html)
        self.assertIn("会带来什么效果", html)
        self.assertIn("未来影响", html)
        self.assertNotIn("领域技术摘编", html)

    def test_classic_learning_navigation_groups_items_by_domain(self):
        generator = ReportGenerator()
        world_item = {
            "url": "https://example.com/world-model",
            "title_cn": "世界模型训练开始进入机器人规划链路",
            "summary_preview": "世界模型从视频预测走向动作规划。",
            "summary": "研究材料显示世界模型训练开始强调预测结果能否进入机器人规划链路。",
            "content_type": "news",
            "content_kind": "全网动态",
            "display_topic": "World Model",
            "source_tier": "media",
            "evidence_quality": 0.7,
            "information_density": 0.7,
            "score": 8.0,
            "facts": {"who": "NVIDIA", "action": "讨论", "target": "world model robot planning", "evidence": ["技术文章讨论机器人规划"]},
        }
        robot_item = {
            "url": "https://example.com/robot-paper",
            "title_cn": "机器人论文提出动态操作策略",
            "summary_preview": "动态操作开始强调真实任务成功率。",
            "summary": "论文提出动态机器人操作策略，并在真实任务中比较成功率。",
            "content_type": "paper",
            "content_kind": "论文",
            "display_topic": "Physical AI",
            "source_tier": "research",
            "evidence_quality": 0.8,
            "information_density": 0.8,
            "score": 8.5,
            "facts": {"who": "RobotPaper", "action": "提出", "target": "dynamic robot manipulation", "evidence": ["真实任务成功率提升"]},
        }
        open_source_item = {
            "url": "https://example.com/prime-rl",
            "title_cn": "Prime Intellect开源强化学习框架",
            "summary_preview": "开源框架支持大规模异步强化学习 rollout。",
            "summary": "Prime Intellect开源强化学习框架，支持大规模异步 rollout 和 MoE 训练。",
            "content_type": "news",
            "content_kind": "全网动态",
            "display_topic": "Open Source",
            "source_tier": "media",
            "evidence_quality": 0.7,
            "information_density": 0.7,
            "score": 7.5,
            "facts": {"who": "Prime Intellect", "action": "开源", "target": "RL framework", "evidence": ["框架支持异步 rollout"]},
        }
        html = generator.generate_html(
            papers=[robot_item],
            updates=[world_item, open_source_item],
            mixed_items=[world_item, robot_item, open_source_item],
            report_summary={"lead_summary": "今天重点看世界模型和机器人规划。"},
            trend_summary={"items": []},
            alert_summary={"needs_alert": False, "issues": []},
            archive_summary={"entries": []},
            layered_updates={
                "must_read": [world_item],
                "physical_ai": [robot_item],
                "watch": [open_source_item],
                "featured_papers": [],
                "paper_appendix": [],
                "brief": [],
            },
        )

        self.assertIn("World Model <span class=\"classic-nav-count\">1 条", html)
        self.assertIn("World Model", html)
        self.assertIn("Physical AI / Robotics", html)
        self.assertIn("Infra / Open Source", html)
        self.assertIn('href="#classic-01"', html)
        self.assertIn('href="#classic-02"', html)
        self.assertIn('href="#classic-03"', html)
        self.assertIn("classic-priority priority-must", html)
        self.assertIn("classic-priority priority-physical", html)
        self.assertIn("2分钟", html)

    def test_classic_learning_navigation_prefers_domain_specific_representative(self):
        generator = ReportGenerator()
        ocr_item = {
            "url": "https://example.com/ocr",
            "title_cn": "OCR 4替代高频人工文档处理步骤",
            "summary_preview": "文档识别产品继续进入企业处理流程。",
            "summary": "OCR产品改进文档处理能力，但没有明确智能体工作流证据。",
            "content_type": "news",
            "content_kind": "全网动态",
            "display_topic": "产品发布",
            "source_tier": "media",
            "evidence_quality": 0.7,
            "information_density": 0.7,
            "score": 9.5,
            "facts": {"who": "OCR 4", "action": "发布", "target": "文档识别", "evidence": ["产品发布"]},
        }
        agent_item = {
            "url": "https://example.com/agent",
            "title_cn": "Agent工作流开始接入企业审批权限",
            "summary_preview": "智能体从聊天入口推进到任务执行和权限编排。",
            "summary": "Agent工作流开始接入企业审批权限，重点是工具调用、权限和任务执行链路。",
            "content_type": "news",
            "content_kind": "全网动态",
            "display_topic": "Agent / Models",
            "source_tier": "media",
            "evidence_quality": 0.68,
            "information_density": 0.68,
            "score": 7.2,
            "facts": {"who": "Agent平台", "action": "接入", "target": "企业审批工作流", "evidence": ["工具调用和权限编排"]},
        }
        classic_items = generator._classic_mixed_items(generator._decorate_items([ocr_item, agent_item]))
        nav = generator._classic_learning_nav(classic_items)
        agent_nav = next(item for item in nav if item["label"] == "Agent / Models")

        self.assertIn("工具调用", agent_nav["takeaway"])
        self.assertIn("Agent工作流", agent_nav["lead_title"])
        self.assertNotIn("OCR 4", agent_nav["lead_title"])

    def test_classic_learning_navigation_prefers_explicit_world_model_title(self):
        generator = ReportGenerator()
        pose_item = {
            "url": "https://example.com/pose",
            "title_cn": "主动推理驱动的物理AI测试时缩放定律",
            "summary_preview": "物理AI任务开始讨论测试时缩放。",
            "summary": "材料讨论物理AI测试时缩放，并涉及预测和规划线索。",
            "content_type": "paper",
            "content_kind": "论文",
            "display_topic": "世界模型",
            "source_tier": "research",
            "evidence_quality": 0.8,
            "information_density": 0.8,
            "score": 9.4,
            "facts": {"who": "研究团队", "target": "physical AI test-time scaling", "evidence": ["预测和规划线索"]},
        }
        explicit_world_item = {
            "url": "https://example.com/world-pieces",
            "title_cn": "World Models in Pieces提出组合式世界模型框架",
            "summary_preview": "世界模型开始从单体预测转向可组合的动态模块。",
            "summary": "World Models in Pieces提出组合式世界模型框架，把动态预测拆成可复用模块。",
            "content_type": "paper",
            "content_kind": "论文",
            "display_topic": "世界模型",
            "source_tier": "research",
            "evidence_quality": 0.62,
            "information_density": 0.62,
            "score": 6.8,
            "facts": {"who": "World Models in Pieces", "target": "compositional world model framework", "evidence": ["组合式动态模块"]},
        }
        classic_items = generator._classic_mixed_items(generator._decorate_items([pose_item, explicit_world_item]))
        nav = generator._classic_learning_nav(classic_items)
        world_nav = next(item for item in nav if item["label"] == "World Model")

        self.assertIn("潜空间动态", world_nav["takeaway"])
        self.assertIn("World Models in Pieces", world_nav["lead_title"])
        self.assertNotIn("物理AI", world_nav["lead_title"])

    def test_classic_learning_navigation_avoids_generic_world_model_lead(self):
        generator = ReportGenerator()
        generic_item = {
            "url": "https://example.com/generic-world",
            "title_cn": "这篇论文：从标题和摘要看，论文在改进模型或训练流程；当前材料不足以展开更多细节",
            "summary_preview": "当前材料不足以展开更多细节。",
            "summary": "世界模型s in Pieces提出结构化验证框架。",
            "content_type": "paper",
            "content_kind": "论文",
            "display_topic": "世界模型",
            "source_tier": "research",
            "evidence_quality": 0.9,
            "information_density": 0.9,
            "score": 10.0,
            "facts": {"who": "this paper", "target": "", "evidence": ["thin abstract"]},
        }
        specific_item = {
            "url": "https://example.com/wvm",
            "title_cn": "WVM提出机器人操作价值模型",
            "summary_preview": "世界模型开始进入机器人策略评估。",
            "summary": "WVM用世界模型估计机器人操作策略价值。",
            "content_type": "paper",
            "content_kind": "论文",
            "display_topic": "世界模型",
            "source_tier": "research",
            "evidence_quality": 0.65,
            "information_density": 0.65,
            "score": 7.0,
            "facts": {"who": "WVM", "target": "robot manipulation value model", "evidence": ["policy evaluation"]},
        }
        decorated = generator._decorate_items([generic_item, specific_item])
        nav = generator._classic_learning_nav(generator._classic_mixed_items(decorated))
        world_nav = next(item for item in nav if item["label"] == "World Model")

        self.assertIn("WVM", world_nav["lead_title"])
        self.assertNotIn("当前材料不足", world_nav["lead_title"])
        self.assertNotIn("世界模型s", decorated[0]["summary_display"])

    def test_classic_learning_navigation_repairs_mixed_lead_titles(self):
        generator = ReportGenerator()
        paper = {
            "url": "https://example.com/world-pieces",
            "title_cn": "世界模型s in Pieces论文提出一种transition-l",
            "summary_preview": "世界模型开始从单体预测转向可组合动态模块。",
            "summary": "World Models in Pieces studies compositional world models.",
            "content_type": "paper",
            "content_kind": "论文",
            "display_topic": "世界模型",
            "source_tier": "research",
            "evidence_quality": 0.8,
            "information_density": 0.8,
            "score": 8.2,
            "facts": {
                "who": "World Models in Pieces",
                "target": "compositional world model framework",
                "evidence": ["evaluates transition localization"],
            },
        }
        agent = {
            "url": "https://example.com/hermes",
            "title_cn": "Nous Research更新Product Rele",
            "summary_preview": "Hermes Agent Skills把模型能力封装成可调用技能。",
            "summary": "Nous Research releases Hermes Agent Skills for agent workflows.",
            "content_type": "news",
            "content_kind": "全网动态",
            "display_topic": "Agent / Models",
            "source_tier": "media",
            "evidence_quality": 0.7,
            "information_density": 0.7,
            "score": 8.0,
            "facts": {
                "who": "Nous Research",
                "action": "发布",
                "target": "Hermes Agent Skills",
                "evidence": ["release notes mention skills"],
            },
        }

        nav = generator._classic_learning_nav(generator._classic_mixed_items(generator._decorate_items([paper, agent])))
        world_nav = next(item for item in nav if item["label"] == "World Model")
        agent_nav = next(item for item in nav if item["label"] == "Agent / Models")
        decorated_titles = [item["title_cn"] for item in generator._decorate_items([paper, agent])]

        self.assertNotIn("世界模型s", world_nav["lead_title"])
        self.assertNotIn("transition-l", world_nav["lead_title"])
        self.assertNotIn("Product Rele", agent_nav["lead_title"])
        self.assertIn("Hermes Agent Skills", agent_nav["lead_title"])
        self.assertTrue(all("Product Rele" not in title for title in decorated_titles))
        self.assertTrue(all("世界模型s" not in title for title in decorated_titles))

    def test_classic_learning_navigation_repairs_long_english_lead_titles(self):
        generator = ReportGenerator()
        item = {
            "url": "https://example.com/hermes-learn",
            "title": "Nous Research Adds /learn to Hermes Agent Skills",
            "summary_preview": "Hermes Agent Skills增加可学习的智能体技能入口。",
            "summary": "Nous Research adds /learn to Hermes Agent Skills for agent workflows.",
            "content_type": "news",
            "content_kind": "全网动态",
            "display_topic": "Agent / Models",
            "source_tier": "media",
            "evidence_quality": 0.7,
            "information_density": 0.7,
            "score": 8.0,
            "facts": {
                "who": "Nous Research",
                "action": "发布",
                "target": "Hermes Agent Skills 的 /learn 能力",
                "evidence": ["release notes mention /learn"],
            },
        }
        decorated = generator._decorate_items([item])
        nav = generator._classic_learning_nav(generator._classic_mixed_items(decorated))
        agent_nav = next(item for item in nav if item["label"] == "Agent / Models")

        self.assertIn("Hermes Agent Skills", decorated[0]["title_cn"])
        self.assertIn("/learn", decorated[0]["title_cn"])
        self.assertNotIn("Adds", decorated[0]["title_cn"])
        self.assertNotIn("Adds", agent_nav["lead_title"])

    def test_classic_template_localizes_product_release_labels(self):
        generator = ReportGenerator()
        item = {
            "url": "https://example.com/product",
            "title_cn": "Gemini发布计算机使用能力",
            "summary_preview": "浏览器操作能力开始进入产品工作流。",
            "summary": "Google发布计算机使用能力，材料显示它面向浏览器任务自动化。",
            "content_type": "news",
            "content_kind": "全网动态",
            "display_topic": "Product Release",
            "source_tier": "official",
            "evidence_quality": 0.75,
            "information_density": 0.72,
            "score": 8.0,
            "facts": {"who": "Google", "action": "发布", "target": "计算机使用能力", "evidence": ["官方材料"]},
        }
        html = generator.generate_html(
            papers=[],
            updates=[item],
            mixed_items=[item],
            report_summary={"lead_summary": "产品能力进入工作流自动化。"},
            trend_summary={"items": []},
            alert_summary={"needs_alert": False, "issues": []},
            archive_summary={"entries": []},
            layered_updates={"must_read": [item], "physical_ai": [], "watch": [], "featured_papers": [], "paper_appendix": [], "brief": []},
        )

        self.assertIn("产品发布", html)
        self.assertNotIn("Product Release", html)

    def test_facts_to_summary_uses_specific_paper_fields_instead_of_generic_template(self):
        processor = LLMProcessor({"api_key_env": "WEB_AGENT_TEST_DISABLED"})
        article = {
            "content_type": "paper",
            "source": "arXiv",
            "title": "AHEAD: Action-aware World Models for Dynamic Robot Tasks",
        }
        facts = {
            "who": "研究团队",
            "action": "提出",
            "target": "动态机器人任务中的动作前预测",
            "evidence": ["在动态操作任务上对比冻结VLA基线"],
            "audience": "机器人研究者",
            "method": "先预测动态物体下一步位置，再把预测结果交给冻结VLA做动作决策",
            "dataset_or_benchmark": "动态操作任务基准",
            "metric_result": "相对冻结VLA基线提升成功率",
            "baseline": "冻结VLA策略",
            "limitation": "仍需更多真实机器人场景复现",
            "confidence": 0.85,
        }

        summary = processor._facts_to_summary(article, facts, "World Model", "世界模型")

        self.assertIn("先预测动态物体下一步位置", summary)
        self.assertIn("动态操作任务基准", summary)
        self.assertIn("冻结VLA", summary)
        for bad in ("围绕", "新的方法或实验结果", "后续要看复现结果", "相关的新进展"):
            self.assertNotIn(bad, summary)

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

    def test_final_quality_failure_blocks_only_real_email_send(self):
        config = {"block_send_on_final_failure": True}

        self.assertTrue(should_block_report_send(config, "failed", "send"))
        self.assertFalse(should_block_report_send(config, "passed", "send"))
        self.assertFalse(should_block_report_send(config, "failed", "dry-run"))

    def test_hard_fidelity_metric_blocks_even_if_summary_status_is_passed(self):
        config = {"block_send_on_final_failure": True}
        diagnostics = {
            "quality_gate": {
                "email_delivery_volume_content_fidelity_missing_count": 1,
            }
        }

        self.assertTrue(
            should_block_report_send(config, "passed", "send", diagnostics)
        )
        self.assertFalse(
            should_block_report_send(config, "passed", "dry-run", diagnostics)
        )

    def test_soft_quality_failure_does_not_block_punctual_send(self):
        config = {
            "block_send_on_final_failure": True,
            "allow_degraded_send_on_soft_failure": True,
        }
        diagnostics = {
            "quality_gate": {
                "final_html_quality_status": "failed",
                "reading_budget_underfilled": True,
                "visible_fresh_paper_count": 8,
                "featured_fresh_paper_count": 3,
                "domain_coverage_warning_count": 1,
                "paper_domain_quota_status": "passed",
                "paper_domain_quota_underfilled": {"world_model": {"count": 0, "min": 4}},
            }
        }

        self.assertFalse(should_block_report_send(config, "failed", "send", diagnostics))

    def test_critical_quality_failure_still_blocks_punctual_send(self):
        config = {
            "block_send_on_final_failure": True,
            "allow_degraded_send_on_soft_failure": True,
            "final_html_bad_title_count": 0,
        }
        diagnostics = {
            "quality_gate": {
                "final_html_quality_status": "failed",
                "final_html_bad_title_count": 1,
                "reading_budget_underfilled": True,
            }
        }

        self.assertTrue(should_block_report_send(config, "failed", "send", diagnostics))

    def test_clipping_risk_and_missing_current_research_are_hard_send_blocks(self):
        config = {
            "block_send_on_final_failure": True,
            "allow_degraded_send_on_soft_failure": True,
            "require_codex_research_inbox": True,
        }
        diagnostics = {
            "quality_gate": {
                "email_clipping_risk": True,
                "codex_research_inbox_status": "error",
                "cross_section_event_duplicate_count": 1,
            }
        }

        reasons = report_send_blocking_reasons(config, diagnostics)

        self.assertIn("email_clipping_risk", reasons)
        self.assertIn("codex_research_inbox_status", reasons)
        self.assertIn("cross_section_event_duplicate_count", reasons)
        self.assertTrue(should_block_report_send(config, "failed", "send", diagnostics))

    def test_low_technical_primary_source_ratio_is_a_hard_send_block(self):
        config = {
            "allow_degraded_send_on_soft_failure": True,
            "technical_primary_source_ratio_min": 0.8,
        }
        diagnostics = {
            "quality_gate": {
                "technical_primary_source_count": 15,
                "technical_primary_source_ratio": 0.75,
            }
        }

        reasons = report_send_blocking_reasons(config, diagnostics)

        self.assertIn("technical_primary_source_ratio", reasons)

    def test_email_commit_immediately_marks_report_sent(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db = Database(str(Path(temp_dir) / "reports.db"))
            db.record_report_run("report-1", "run-1", delivery_status="pending")
            slot_dir = Path(temp_dir) / "send_slots"
            slot_dir.mkdir()
            slot_path = slot_dir / "20260812_2100.json"
            slot_path.write_text(
                json.dumps({"slot_id": "20260812_2100", "status": "in_progress", "pid": 123}),
                encoding="utf-8",
            )

            payload = record_email_commit(
                db,
                report_id="report-1",
                run_id="run-1",
                subject="AI Daily",
                html_report_path="archive/report.html",
                markdown_report_path="archive/report.md",
                quality_status="passed",
                send_slot_id="20260812_2100",
                send_slot_dir=slot_dir,
            )

            report = db.get_report_run("report-1")
            slot = json.loads(slot_path.read_text(encoding="utf-8-sig"))

        self.assertEqual(report["delivery_status"], "sent")
        self.assertTrue(report["delivery_at"])
        self.assertEqual(payload["report_id"], "report-1")
        self.assertEqual(payload["email_subject"], "AI Daily")
        self.assertEqual(slot["status"], "sent")
        self.assertEqual(slot["run_id"], "run-1")
        self.assertEqual(payload["send_slot_path"], slot_path.as_posix())

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

    def test_google_news_aggregator_is_demoted_when_source_is_thin(self):
        aggregator_score = score_update_quality(
            title="OpenAI launches new enterprise workflow",
            content="OpenAI launches new enterprise workflow.",
            url="https://news.google.com/rss/articles/demo",
            platform="Web",
            source_detail="Google News",
            category="产品发布",
            source_preferences={},
        )
        canonical_score = score_update_quality(
            title="OpenAI launches new enterprise workflow",
            content="OpenAI launches a new enterprise workflow product with tool calling, customer rollout details, and deployment notes for business users.",
            url="https://openai.com/index/new-enterprise-workflow/",
            platform="Blog",
            source_detail="OpenAI Blog",
            category="产品发布",
            source_preferences={},
        )
        self.assertLess(aggregator_score, canonical_score)
        self.assertTrue(
            is_low_signal_update(
                title="OpenAI launches new enterprise workflow",
                content="OpenAI launches new enterprise workflow.",
                url="https://news.google.com/rss/articles/demo",
                platform="Web",
                source_detail="Google News",
                category="产品发布",
                source_preferences={},
            )
        )

    def test_substantive_ai_podcast_is_not_low_signal(self):
        content = (
            "Yann LeCun argues that useful world models should predict abstract latent states rather than "
            "reconstruct every video pixel. He contrasts JEPA training with autoregressive token prediction, "
            "explains how planning can search over predicted representations, and cites robot control experiments "
            "as the test of whether the learned dynamics transfer beyond video. The interview also identifies "
            "long-horizon error accumulation and missing action-conditioned benchmarks as current limitations."
        )
        score = score_update_quality(
            title="Podcast: Yann LeCun on JEPA world models and robot planning",
            content=content,
            url="https://example.com/interviews/lecun-jepa-world-models",
            platform="Podcast",
            source_detail="AI Research Interviews",
            category="访谈观点",
            source_preferences={},
        )
        self.assertGreater(score, -1.0)
        self.assertFalse(
            is_low_signal_update(
                title="Podcast: Yann LeCun on JEPA world models and robot planning",
                content=content,
                url="https://example.com/interviews/lecun-jepa-world-models",
                platform="Podcast",
                source_detail="AI Research Interviews",
                category="访谈观点",
                source_preferences={},
            )
        )

    def test_google_news_url_param_is_used_as_canonical_url(self):
        collector = WebSearchCollector(searches=[])
        entry = {"links": [{"href": "https://news.google.com/rss/articles/demo?url=https%3A%2F%2Fopenai.com%2Fnews%2Fitem"}]}
        canonical = collector._canonicalize_google_news_url(
            entry,
            "https://news.google.com/rss/articles/demo?url=https%3A%2F%2Fopenai.com%2Fnews%2Fitem",
        )
        self.assertEqual(canonical, "https://openai.com/news/item")

    def test_feed_date_parsing_does_not_use_windows_mktime_range(self):
        entry = type("Entry", (dict,), {})()
        entry.published_parsed = (2026, 9, 19, 12, 30, 0, 0, 0, 0)
        expected = datetime(2026, 9, 19, 12, 30, tzinfo=timezone.utc)

        self.assertEqual(WebSearchCollector(searches=[])._parse_entry_date(entry), expected)
        self.assertEqual(RSSCollector(feeds=[])._parse_entry_date(entry), expected)

    def test_invalid_feed_date_is_skipped_as_old_instead_of_raising(self):
        entry = type(
            "Entry",
            (dict,),
            {"published_parsed": (10000, 1, 1, 0, 0, 0, 0, 0, 0)},
        )()

        parsed = WebSearchCollector(searches=[])._parse_entry_date(entry)

        self.assertEqual(parsed, datetime.min.replace(tzinfo=timezone.utc))

    def test_database_persists_article_facts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db = Database(str(Path(temp_dir) / "ai_news.db"))
            article = {
                "title": "OpenAI launches workflow agent",
                "url": "https://openai.com/news/workflow-agent",
                "source": "RSS",
                "source_detail": "OpenAI Blog",
                "content": "OpenAI launches workflow agent.",
                "run_id": "run-1",
            }
            self.assertTrue(db.insert_article(article))
            facts = {
                "who": "OpenAI",
                "action": "发布",
                "target": "企业工作流智能体",
                "evidence": ["可跨工具执行长任务"],
                "audience": "企业用户",
                "confidence": 0.8,
            }
            db.update_article_processing(
                url=article["url"],
                summary="OpenAI发布企业工作流智能体。",
                score=8.8,
                keywords=["智能体"],
                category="产品发布",
                facts=facts,
                evidence_quality=0.8,
                information_density=0.9,
                model_used="deepseek-v4-pro",
            )
            rows = db.get_articles_for_run("run-1")
            self.assertEqual(rows[0]["facts"]["who"], "OpenAI")
            self.assertEqual(rows[0]["facts"]["target"], "企业工作流智能体")
            self.assertEqual(rows[0]["evidence_quality"], 0.8)
            self.assertEqual(rows[0]["information_density"], 0.9)
            self.assertEqual(rows[0]["model_used"], "deepseek-v4-pro")

    def test_report_layers_keep_missing_facts_out_of_must_read(self):
        strong = {
            "id": 1,
            "title": "OpenAI launches workflow agent",
            "title_cn": "OpenAI 发布工作流智能体",
            "summary": "OpenAI 发布 workflow agent，证据包括工具调用和企业部署说明。",
            "url": "https://openai.com/news/workflow-agent",
            "source_detail": "OpenAI Blog",
            "platform": "Blog",
            "content_type": "news",
            "score": 8.8,
            "evidence_quality": 0.9,
            "information_density": 0.9,
            "facts": {
                "who": "OpenAI",
                "action": "launches",
                "target": "workflow agent",
                "evidence": ["tool calling", "enterprise deployment"],
                "audience": "enterprise users",
            },
        }
        weak = {
            "id": 2,
            "title": "AI product update",
            "title_cn": "AI 产品更新",
            "summary": "A short thin update.",
            "url": "https://example.com/thin",
            "content_type": "news",
            "score": 9.5,
            "evidence_quality": 0.1,
            "information_density": 0.1,
            "facts": {},
        }
        layers = classify_report_layers(
            papers=[],
            updates=[weak, strong],
            report_id="report-1",
            feedback_config={"enabled": True, "host": "127.0.0.1", "port": 8765},
            source_preferences={"whitelist_hosts": ["openai.com"]},
        )
        self.assertEqual(layers["must_read"][0]["id"], 1)
        self.assertEqual(layers["brief"][0]["id"], 2)
        self.assertIn("useful", layers["must_read"][0]["feedback_links"])
        gate = evaluate_report_quality(layers, {"max_repeated_sentence_count": 1})
        self.assertEqual(gate["status"], "passed")

    def test_physical_ai_news_gets_independent_section(self):
        item = {
            "id": 3,
            "title": "Humanoid robot maker launches warehouse deployment",
            "title_cn": "人形机器人公司发布仓储部署进展",
            "summary": "公司发布人形机器人仓储部署进展，证据包括客户场景、机器人操作任务和部署时间线。",
            "summary_preview": "人形机器人进入仓储操作场景。",
            "url": "https://example.com/humanoid-warehouse",
            "source_detail": "Figure AI Blog",
            "platform": "Blog",
            "content_type": "news",
            "source_tier": "official",
            "score": 8.5,
            "evidence_quality": 0.55,
            "information_density": 0.55,
            "facts": {
                "who": "Figure AI",
                "action": "发布部署进展",
                "target": "人形机器人仓储操作",
                "evidence": ["客户场景", "机器人操作任务"],
                "audience": "机器人和自动化团队",
            },
        }
        layers = classify_report_layers(
            papers=[],
            updates=[item],
            report_id="report-physical",
            feedback_config={"enabled": False},
        )
        self.assertEqual([entry["id"] for entry in layers["physical_ai"]], [3])
        self.assertEqual(layers["watch"], [])

    def test_must_read_caps_same_entity_before_fallback_fill(self):
        updates = []
        for index in range(5):
            updates.append(
                {
                    "id": index + 10,
                    "title": f"NVIDIA launches update {index}",
                    "title_cn": f"NVIDIA 发布高证据动态 {index}",
                    "summary": "NVIDIA 发布 AI 基础设施动态，证据包括客户、产品和部署说明。",
                    "url": f"https://nvidia.com/update-{index}",
                    "content_type": "news",
                    "source_tier": "official",
                    "score": 9.5 - index * 0.1,
                    "evidence_quality": 0.9,
                    "information_density": 0.9,
                    "facts": {
                        "who": "NVIDIA",
                        "action": "发布",
                        "target": f"AI 基础设施动态 {index}",
                        "evidence": ["客户", "产品", "部署说明"],
                    },
                }
            )
        for index in range(3):
            updates.append(
                {
                    "id": index + 30,
                    "title": f"Other company update {index}",
                    "title_cn": f"其他公司发布高证据动态 {index}",
                    "summary": "其他公司发布 AI 产品动态，证据包括客户、产品和部署说明。",
                    "url": f"https://example.com/update-{index}",
                    "content_type": "news",
                    "source_tier": "official",
                    "score": 8.5 - index * 0.1,
                    "evidence_quality": 0.9,
                    "information_density": 0.9,
                    "facts": {
                        "who": f"Company {index}",
                        "action": "发布",
                        "target": f"AI 产品动态 {index}",
                        "evidence": ["客户", "产品", "部署说明"],
                    },
                }
            )
        layers = classify_report_layers(
            papers=[],
            updates=updates,
            report_id="report-source-cap",
            feedback_config={"enabled": False},
        )
        first_five_entities = [item["facts"]["who"] for item in layers["must_read"][:5]]
        self.assertLessEqual(first_five_entities.count("NVIDIA"), 2)

    def test_research_section_is_capped_and_overflow_goes_to_brief(self):
        papers = []
        for index in range(12):
            papers.append(
                {
                    "id": 100 + index,
                    "title": f"Paper {index}",
                    "title_cn": f"论文 {index} 提出新方法",
                    "summary": "论文提出方法并报告实验结果。",
                    "url": f"https://arxiv.org/abs/{index}",
                    "content_type": "paper",
                    "source_tier": "research",
                    "score": 9.0 - index * 0.01,
                    "evidence_quality": 0.5,
                    "information_density": 0.2,
                    "facts": {"who": f"Team {index}", "action": "提出", "target": "模型方法", "evidence": ["实验结果"]},
                }
            )
        layers = classify_report_layers(
            papers=papers,
            updates=[],
            report_id="report-research",
            feedback_config={"enabled": False},
        )
        self.assertEqual(len(layers["research"]), 10)
        self.assertEqual(len([item for item in layers["brief"] if item["content_type"] == "paper"]), 2)

    def test_research_section_limit_can_be_raised_for_deeper_paper_reading(self):
        papers = []
        for index in range(18):
            papers.append(
                {
                    "id": 200 + index,
                    "title": f"Paper {index}",
                    "title_cn": f"论文 {index} 提出新方法",
                    "summary": "论文提出方法并报告实验结果。",
                    "url": f"https://arxiv.org/abs/raised-{index}",
                    "content_type": "paper",
                    "source_tier": "research",
                    "score": 9.0 - index * 0.01,
                    "evidence_quality": 0.4,
                    "information_density": 0.4,
                    "facts": {"who": f"Team {index}", "action": "提出", "target": "模型方法", "evidence": ["实验结果"]},
                }
            )
        layers = classify_report_layers(
            papers=papers,
            updates=[],
            report_id="report-research-raised",
            feedback_config={"enabled": False},
            report_config={"research_limit": 15},
        )
        self.assertEqual(len(layers["research"]), 15)
        self.assertEqual(len([item for item in layers["brief"] if item["content_type"] == "paper"]), 3)

    def test_v3_splits_papers_into_featured_and_appendix(self):
        papers = []
        for index in range(10):
            papers.append(
                {
                    "id": 300 + index,
                    "title": f"Paper {index}",
                    "title_cn": f"论文 {index} 提出可复现实验方法",
                    "summary": "论文提出新方法并报告实验结果，影响需要复现该任务的研究团队。",
                    "url": f"https://arxiv.org/abs/v3-{index}",
                    "content_type": "paper",
                    "source_tier": "research",
                    "score": 9.0 - index * 0.01,
                    "evidence_quality": 0.7,
                    "information_density": 0.7,
                    "facts": {
                        "who": f"Team {index}",
                        "action": "提出",
                        "target": "可复现实验方法",
                        "evidence": ["实验结果提升 12%", "基准评测优于对照组"],
                        "audience": "模型研究者",
                    },
                }
            )
        layers = classify_report_layers(
            papers=papers,
            updates=[],
            report_id="report-v3-papers",
            feedback_config={"enabled": False},
            report_config={"research_limit": 10, "paper_featured_limit": 6, "paper_appendix_limit": 4},
        )
        self.assertEqual(len(layers["featured_papers"]), 6)
        self.assertEqual(len(layers["paper_appendix"]), 4)
        self.assertEqual(len(flatten_report_layers(layers)), 10)

    def test_v5_learning_fields_and_domain_paper_coverage_are_added(self):
        papers = [
            {
                "id": 501,
                "title_cn": "WorldBench提出世界模型评测方法",
                "summary": "WorldBench提出世界模型评测方法，实验在长时预测基准上提升12%，影响世界模型研究者。",
                "url": "https://arxiv.org/abs/worldbench",
                "content_type": "paper",
                "source_tier": "research",
                "score": 9.7,
                "evidence_quality": 0.8,
                "information_density": 0.8,
                "facts": {
                    "who": "WorldBench",
                    "action": "提出",
                    "target": "世界模型长时预测评测方法",
                    "evidence": ["长时预测基准提升12%", "实验优于对照组"],
                    "audience": "世界模型研究者",
                },
            },
            {
                "id": 502,
                "title_cn": "RobotPolicy提出机器人控制训练方法",
                "summary": "RobotPolicy提出机器人控制训练方法，实验在真实操作任务成功率提升18%，影响机器人研究团队。",
                "url": "https://arxiv.org/abs/robotpolicy",
                "content_type": "paper",
                "source_tier": "research",
                "score": 9.6,
                "evidence_quality": 0.8,
                "information_density": 0.8,
                "facts": {
                    "who": "RobotPolicy",
                    "action": "提出",
                    "target": "机器人控制训练方法",
                    "evidence": ["真实操作成功率提升18%", "实验优于对照组"],
                    "audience": "机器人研究团队",
                },
            },
        ]
        update = {
            "id": 503,
            "title_cn": "OpenAI发布企业工作流Agent",
            "summary": "OpenAI发布企业工作流Agent，官方博客披露能力，影响企业自动化团队。",
            "summary_preview": "企业工作流开始被Agent接管。",
            "url": "https://openai.com/blog/workflow-agent",
            "content_type": "news",
            "source_tier": "official",
            "score": 9.5,
            "evidence_quality": 0.8,
            "information_density": 0.8,
            "facts": {
                "who": "OpenAI",
                "action": "发布",
                "target": "企业工作流Agent",
                "evidence": ["官方博客"],
                "audience": "企业自动化团队",
            },
        }
        layers = classify_report_layers(
            papers=papers,
            updates=[update],
            report_id="report-v5-learning",
            feedback_config={"enabled": False},
            report_config={"research_limit": 4, "paper_featured_limit": 2, "paper_appendix_limit": 2},
        )
        featured_domains = {item["domain_key"] for item in layers["featured_papers"]}
        self.assertIn("world_model", featured_domains)
        self.assertIn("physical_ai", featured_domains)
        for item in flatten_report_layers(layers):
            self.assertTrue(item.get("learning_takeaway"))
            self.assertTrue(item.get("technical_context"))
            self.assertTrue(item.get("background_context"))
            self.assertTrue(item.get("deep_dive_prompt"))
        gate = evaluate_report_quality(layers, {"max_repeated_sentence_count": 1})
        self.assertIn("v5_quality_status", gate)
        self.assertEqual(gate["learning_card_missing_count"], 0)
        diagnostics = build_report_structure_diagnostics(layers)
        self.assertIn("domain_counts", diagnostics)
        self.assertEqual(diagnostics["learning_card_missing_count"], 0)

        generator = ReportGenerator(design_version="v6-editorial-learning")
        decorated = generator._decorate_items(flatten_report_layers(layers))
        for item in decorated:
            self.assertTrue(item.get("technical_digest"))
            self.assertTrue(any("技术" in point or "路线" in point or "实验" in point for point in item["technical_digest"]))

    def test_paper_missing_method_result_meaning_stays_out_of_featured(self):
        weak_paper = {
            "id": 401,
            "title": "Thin paper",
            "title_cn": "论文提出一个方向",
            "summary": "论文讨论了一个研究方向。",
            "url": "https://arxiv.org/abs/thin",
            "content_type": "paper",
            "source_tier": "research",
            "score": 9.5,
            "evidence_quality": 0.7,
            "information_density": 0.7,
            "facts": {"who": "Team", "action": "", "target": "", "evidence": []},
        }
        strong_paper = {
            "id": 402,
            "title": "Strong paper",
            "title_cn": "论文提出新机器人评测方法",
            "summary": "论文提出新机器人评测方法，实验成功率提升 15%，影响机器人研究团队的复现判断。",
            "url": "https://arxiv.org/abs/strong",
            "content_type": "paper",
            "source_tier": "research",
            "score": 9.4,
            "evidence_quality": 0.7,
            "information_density": 0.7,
            "facts": {
                "who": "RobotBench",
                "action": "提出",
                "target": "机器人评测方法",
                "evidence": ["成功率提升 15%", "实验优于对照组"],
                "audience": "机器人研究团队",
            },
        }
        layers = classify_report_layers(
            papers=[weak_paper, strong_paper],
            updates=[],
            report_id="report-v3-paper-quality",
            feedback_config={"enabled": False},
            report_config={"research_limit": 2, "paper_featured_limit": 1, "paper_appendix_limit": 1},
        )
        self.assertTrue(paper_core_summary_passes(strong_paper))
        self.assertFalse(paper_core_summary_passes(weak_paper))
        self.assertEqual([item["id"] for item in layers["featured_papers"]], [402])
        self.assertEqual([item["id"] for item in layers["paper_appendix"]], [401])
        gate = evaluate_report_quality(layers, {"max_repeated_sentence_count": 1})
        self.assertEqual(gate["paper_core_summary_pass_count"], 1)
        self.assertEqual(gate["paper_core_summary_fail_count"], 1)
        self.assertEqual(gate["status"], "passed")

    def test_report_structure_diagnostics_counts_key_fields(self):
        layers = {
            "must_read": [
                {
                    "title_cn": "NVIDIA 发布更新",
                    "url": "https://nvidia.com/a",
                    "source_detail": "NVIDIA Blog",
                    "content_type": "news",
                    "facts": {"who": "NVIDIA", "action": "发布", "target": "更新"},
                },
                {
                    "title_cn": "NVIDIA 发布另一项更新",
                    "url": "https://nvidia.com/b",
                    "source_detail": "NVIDIA Blog",
                    "content_type": "news",
                    "facts": {"who": "NVIDIA", "action": "发布", "target": "另一项更新"},
                },
            ],
            "physical_ai": [{"title_cn": "机器人部署", "content_type": "news", "url": "https://example.com/robot"}],
            "watch": [],
            "research": [
                {
                    "title_cn": "Qwen-VLA 提出机器人模型",
                    "summary_preview": "Qwen-VLA把视觉理解、语言指令和机器人动作放进同一个模型框架。",
                    "content_type": "paper",
                    "display_topic": "具身智能",
                    "facts": {
                        "target": "unified vision-language-action model for robot manipulation",
                        "action": "proposes",
                        "evidence": ["97.9% on LIBERO", "73.7% on Simpler-WidowX"],
                    },
                }
            ],
            "brief": [],
        }
        diagnostics = build_report_structure_diagnostics(layers)
        self.assertEqual(diagnostics["physical_ai_count"], 1)
        self.assertEqual(diagnostics["must_read_source_concentration"], 2)
        self.assertEqual(diagnostics["paper_description_english_leak_count"], 0)
        self.assertEqual(diagnostics["paper_plain_summary_english_leak_count"], 0)
        self.assertEqual(diagnostics["paper_selected_count"], 1)
        self.assertEqual(diagnostics["paper_appendix_count"], 0)
        self.assertEqual(diagnostics["paper_core_summary_pass_count"], 1)
        self.assertGreaterEqual(diagnostics["tracking_question_count"], 3)

    def test_failed_focus_items_are_downgraded_to_brief(self):
        from main import downgrade_failed_focus_items

        layers = {
            "must_read": [
                {"id": 1, "url": "https://example.com/ok", "report_section": "must_read"},
                {"id": 2, "url": "https://example.com/bad", "report_section": "must_read"},
            ],
            "watch": [],
            "research": [],
            "brief": [],
        }
        downgraded = downgrade_failed_focus_items(layers, ["https://example.com/bad"])
        self.assertEqual([item["id"] for item in downgraded["must_read"]], [1])
        self.assertEqual(downgraded["brief"][0]["id"], 2)
        self.assertEqual(downgraded["brief"][0]["report_section"], "brief")

    def test_title_fact_mismatch_only_flags_clear_mismatch(self):
        matched = {
            "title": "OpenAI launches workflow agent for enterprise tools",
            "title_cn": "OpenAI 发布企业工作流智能体",
            "summary": "OpenAI 发布 workflow agent，面向企业工具调用。",
        }
        matched_facts = {"who": "OpenAI", "action": "launches", "target": "workflow agent"}
        self.assertFalse(title_fact_mismatch(matched, matched_facts))

        mismatched = {
            "title": "NVIDIA launches inference chip platform",
            "title_cn": "NVIDIA 发布推理芯片平台",
            "summary": "OpenAI 发布 workflow agent，面向企业工具调用。",
        }
        self.assertTrue(title_fact_mismatch(mismatched, matched_facts))

        paper = {
            "content_type": "paper",
            "title": "DA-Nav: Direction-Aware City-Scale Vision-Language Navigation",
            "title_cn": "DA-Nav：城市级室外导航",
            "summary": "DA-Nav提出城市级室外导航方法。",
        }
        paper_facts = {
            "who": "DA-Nav",
            "action": "提出",
            "target": "城市级室外导航，利用商业导航工具的方向指令",
        }
        self.assertFalse(title_fact_mismatch(paper, paper_facts))

        acronym_paper = {
            "content_type": "paper",
            "title": "Agile perceptive multi-skill locomotion for quadrupedal robots in the wild",
            "title_cn": "APT-RL：四足机器人多技能运动",
            "summary": "APT-RL训练可复用运动技能，并部署到真实四足机器人。",
        }
        acronym_facts = {
            "who": "APT-RL (Action Pretrained Transformer-based Reinforcement Learning)",
            "action": "提出",
            "target": "四足机器人多技能运动，包括自主技能切换、高速感知运动",
        }
        self.assertFalse(title_fact_mismatch(acronym_paper, acronym_facts))

        compact_target_paper = {
            "content_type": "paper",
            "title": "Data and Learning Where it Matters for Contact-Rich Manipulation",
            "title_cn": "未具名研究团队：接触丰富操作任务关键段",
            "summary": "论文只在关键接触段密集采集数据，其余运动继续使用传统规划。",
        }
        compact_target_facts = {
            "who": "未具名研究团队",
            "action": "提出",
            "target": "接触丰富操作任务中的关键段，而非整个任务轨迹",
        }
        self.assertFalse(title_fact_mismatch(compact_target_paper, compact_target_facts))

        long_target_paper = {
            "content_type": "paper",
            "title": "Lights, Camera, Malfunction: When Illumination Robustness Leaves VLA Models Blind to Color",
            "title_cn": "FLARE和ChromaGuard（论文方法）：Vision-Language-Action (VLA) 模型在",
            "summary": "论文研究VLA模型在机器人操作中的颜色鲁棒性。",
            "facts": {
                "who": "FLARE和ChromaGuard（论文方法）",
                "action": "提出",
                "target": "Vision-Language-Action (VLA) 模型在通用机器人操作中的鲁棒性",
            },
        }
        repaired_long_target = repair_title_fact_mismatch(long_target_paper)
        self.assertEqual(repaired_long_target["title_cn"], "FLARE和ChromaGuard：VLA 模型机器人操作鲁棒性")
        self.assertFalse(title_fact_mismatch(repaired_long_target, repaired_long_target["facts"]))
        decorated_long_target = enrich_editorial_fields(
            {**long_target_paper, "_v8_force_fact_title": True}
        )
        self.assertEqual(
            decorated_long_target["title_cn"],
            "FLARE和ChromaGuard：VLA 模型机器人操作鲁棒性",
        )
        self.assertFalse(title_fact_mismatch(decorated_long_target, decorated_long_target["facts"]))

    def test_title_fact_mismatch_repair_uses_structured_facts(self):
        item = {
            "title": "What ClickUp's mass layoff tells us about the future of work",
            "title_cn": "AI行业资源配置开始出现新变化正在重排资源",
            "summary": "ClickUp用数千个AI agents替代部分员工，显示企业工作流开始从人工岗位转向自动化执行。",
            "facts": {
                "who": "ClickUp",
                "action": "replacing",
                "target": "hundreds of employees with thousands of AI agents",
                "evidence": ["mass layoff of hundreds of employees", "replacing them with thousands of AI agents"],
            },
            "evidence_quality": 0.7,
            "information_density": 0.7,
        }
        repaired = repair_title_fact_mismatch(item)
        self.assertIn("ClickUp", repaired["title_cn"])
        self.assertFalse(title_fact_mismatch(repaired, repaired["facts"]))

    def test_repeated_analysis_fields_are_hidden_after_two_uses(self):
        repeated_text = "因为世界模型已经不缺演示效果，真正短缺的是长时预测和真实环境一致性。"
        layers = {
            "must_read": [
                {"summary": f"摘要 {index}", "why_now": repeated_text, "expected_effect": "", "future_impact": ""}
                for index in range(3)
            ],
            "watch": [],
            "research": [],
            "brief": [],
        }
        updated = suppress_repeated_analysis_fields(layers, max_repeats=2)
        self.assertEqual(updated["must_read"][0]["why_now"], repeated_text)
        self.assertEqual(updated["must_read"][1]["why_now"], repeated_text)
        self.assertEqual(updated["must_read"][2]["why_now"], "")

    def test_model_path_breakdown_labels_legacy_pending_backfill(self):
        counts = model_path_breakdown(
            [
                {"model_used": "deepseek-v4-pro"},
                {"model_used": "template_fallback", "analysis_version": "v2"},
                {"model_used": "", "analysis_version": "v2"},
                {},
            ]
        )
        self.assertEqual(counts["deepseek-v4-pro"], 1)
        self.assertEqual(counts["template_fallback"], 1)
        self.assertEqual(counts["fallback_v2"], 1)
        self.assertEqual(counts["legacy_pending_backfill"], 1)

    def test_heuristic_processor_marks_template_fallback_model_path(self):
        processor = LLMProcessor({"api_key_env": "THIS_KEY_SHOULD_NOT_EXIST"})
        result = processor.process_article(
            {
                "title": "OpenAI launches workflow agent for enterprise task execution",
                "content": "OpenAI launches a workflow agent that can call tools across long-running business processes.",
                "source_detail": "OpenAI Blog",
                "platform": "Blog",
                "content_type": "news",
            }
        )
        self.assertEqual(result["model_used"], "template_fallback")

    def test_title_repetition_detector_catches_short_repeated_phrases(self):
        processor = LLMProcessor({"api_key_env": "THIS_KEY_SHOULD_NOT_EXIST"})
        self.assertTrue(processor._title_has_repetition("世界模型研究尝试降低试错成本尝试让世界模型尝试预测"))

    def test_feedback_signals_update_limited_preference_weights(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db = Database(str(Path(temp_dir) / "ai_news.db"))
            db.insert_article(
                {
                    "title": "OpenAI launches workflow agent",
                    "url": "https://openai.com/news/workflow-agent",
                    "source": "RSS",
                    "source_detail": "OpenAI Blog",
                    "content": "OpenAI launches workflow agent.",
                    "run_id": "run-1",
                    "topic": "agent",
                    "category": "产品发布",
                    "facts": {"who": "OpenAI"},
                }
            )
            article = db.get_articles_for_run("run-1", processed_only=False)[0]
            db.record_article_feedback("report-1", article["id"], "useful")
            db.record_article_feedback("report-1", article["id"], "track")
            weights = db.get_preference_weights()
            self.assertLessEqual(weights["source"]["OpenAI Blog"], 1.0)
            self.assertGreater(weights["source"]["OpenAI Blog"], 0)
            self.assertEqual(db.get_feedback_count(days=7), 2)

    def test_v4_feedback_signals_are_accepted_and_limited(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db = Database(str(Path(temp_dir) / "ai_news.db"))
            db.insert_article(
                {
                    "title": "AHEAD improves robot planning",
                    "url": "https://arxiv.org/abs/ahead-feedback",
                    "source": "ArXiv",
                    "source_detail": "ArXiv",
                    "content_type": "paper",
                    "run_id": "run-1",
                    "topic": "Physical AI",
                    "category": "模型/研究",
                    "facts": {"who": "AHEAD"},
                }
            )
            article = db.get_articles_for_run("run-1", processed_only=False)[0]
            for signal in ("too_shallow", "paper_too_shallow", "paper_unclear", "too_long", "too_generic", "source_suspicious", "not_memorable"):
                db.record_article_feedback("report-1", article["id"], signal)
            weights = db.get_preference_weights()
            self.assertLessEqual(abs(weights["source"]["ArXiv"]), 1.0)
            self.assertEqual(db.get_feedback_count(days=7), 7)

    def test_report_run_and_items_are_snapshotted(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db = Database(str(Path(temp_dir) / "ai_news.db"))
            db.record_report_run(
                report_id="report-1",
                run_id="run-1",
                html_report_path="archive/report.html",
                markdown_report_path="archive/report.md",
                quality_status="passed",
                quality_diagnostics={"quality_gate": {"status": "passed"}},
            )
            db.record_report_items("report-1", [{"id": 1, "report_rank": 1, "report_section": "must_read", "title": "demo"}])
            latest = db.get_latest_report_run()
            by_id = db.get_report_run("report-1")
            self.assertEqual(latest["report_id"], "report-1")
            self.assertEqual(by_id["report_id"], "report-1")
            self.assertEqual(latest["quality_status"], "passed")
            self.assertEqual(latest["quality_diagnostics"]["quality_gate"]["status"], "passed")

    def test_content_quality_counts_track_generic_low_evidence_and_aggregators(self):
        counts = build_content_quality_counts(
            selected_items=[
                {
                    "title_cn": "Thin item",
                    "summary": "Thin item with little evidence.",
                    "evidence_quality": 0.2,
                },
                {
                    "title_cn": "OpenAI agent",
                    "summary": "OpenAI launches an enterprise workflow agent.",
                    "evidence_quality": 0.8,
                },
                {
                    "title_cn": "AgentWatchdemonstrproactive AWS mo",
                    "summary": "AgentWatch demonstrates proactive monitoring.",
                    "evidence_quality": 0.8,
                },
            ],
            update_candidates=[
                {
                    "title": "OpenAI launches new enterprise workflow",
                    "summary": "OpenAI launches new enterprise workflow.",
                    "url": "https://news.google.com/rss/articles/demo",
                    "platform": "Web",
                    "source_detail": "Google News",
                    "category": "Product Release",
                }
            ],
            source_preferences={},
        )
        self.assertEqual(counts["generic_summary_count"], 0)
        self.assertEqual(counts["low_evidence_count"], 1)
        self.assertEqual(counts["aggregator_demoted_count"], 1)
        self.assertEqual(counts["bad_title_count"], 1)

    def test_title_looks_bad_detects_repeated_and_hybrid_titles(self):
        self.assertTrue(title_looks_bad({"title_cn": "AgentWatchdemonstrproactive AWS mo"}))
        self.assertTrue(title_looks_bad({"title_cn": "Datalab发布了一款9B参数的开 据把AI能力推进到真实工作流"}))
        self.assertTrue(title_looks_bad({"title_cn": "Prime Intellect发布AI领域新进展"}))
        self.assertTrue(title_looks_bad({"title_cn": "Joint-Embedding Archit...：SIGReg 理论分析"}))
        self.assertFalse(title_looks_bad({"title_cn": "AI agent update"}))
        self.assertFalse(title_looks_bad({"title_cn": "Amazon Bedrock AgentCore发布支付功能预览"}))

    def test_repair_bad_titles_in_layers_uses_summary_before_quality_gate(self):
        layers = {
            "must_read": [
                {
                    "id": 1,
                    "url": "https://example.com/agentwatch",
                    "report_section": "must_read",
                    "title": "AgentWatch: Proactive AWS monitoring with ambient agents",
                    "title_cn": "AgentWatchdemonstrproactive AWS mo",
                    "summary": "AgentWatch展示了主动式AWS监控的新进展，其环境智能体每15分钟执行基础设施检查。",
                    "facts": {
                        "who": "AgentWatch",
                        "action": "demonstrates",
                        "target": "proactive AWS monitoring with ambient agents",
                        "evidence": ["checks every 15 minutes"],
                    },
                    "evidence_quality": 0.9,
                    "information_density": 0.9,
                }
            ],
            "physical_ai": [],
            "watch": [],
            "research": [],
            "brief": [],
        }

        repaired, summary = repair_bad_titles_in_layers(layers)
        gate = evaluate_report_quality(repaired, {"max_repeated_sentence_count": 1})

        self.assertEqual(summary["bad_title_repaired_count"], 1)
        self.assertEqual(summary["bad_title_unresolved_count"], 0)
        self.assertEqual(summary["examples"][0]["old_title"], "AgentWatchdemonstrproactive AWS mo")
        self.assertEqual(summary["examples"][0]["new_title"], "AgentWatch展示了主动式AWS监控的新进展")
        self.assertEqual(summary["examples"][0]["source"], "summary")
        self.assertEqual(repaired["must_read"][0]["title_cn"], "AgentWatch展示了主动式AWS监控的新进展")
        self.assertEqual(gate["bad_title_count"], 0)
        self.assertEqual(gate["status"], "passed")

    def test_repair_bad_titles_preserves_valid_codex_wording(self):
        title = "机器人团队尝试用动作前预测减少动态抓取中的状态滞后"
        layers = {
            "must_read": [{
                "id": 2,
                "url": "https://example.com/codex-title",
                "report_section": "must_read",
                "title_cn": title,
                "title": "Predict before acting for dynamic grasping",
                "summary": "研究团队先预测物体位置，再执行抓取动作。",
                "model_used": "codex-automation",
                "facts": {
                    "who": "机器人团队",
                    "action": "提出",
                    "target": "动作前预测方法",
                    "method": "先预测物体位置，再执行抓取动作",
                    "evidence": ["动态抓取成功率提高"],
                },
                "evidence_quality": 0.9,
                "information_density": 0.9,
            }],
            "physical_ai": [],
            "watch": [],
            "research": [],
            "brief": [],
        }

        repaired, summary = repair_bad_titles_in_layers(layers)

        self.assertEqual(repaired["must_read"][0]["title_cn"], title)
        self.assertEqual(summary["bad_title_repaired_count"], 0)

    def test_quality_gate_fails_unresolved_bad_titles(self):
        item = {
            "id": 1,
            "url": "https://example.com/bad-title",
            "report_section": "must_read",
            "title_cn": "AgentWatchdemonstrproactive AWS mo",
            "summary": "",
            "facts": {"who": "AgentWatch", "action": "demonstrates", "target": "proactive AWS monitoring"},
            "evidence_quality": 0.9,
            "information_density": 0.9,
        }
        gate = evaluate_report_quality({"must_read": [item], "watch": [], "research": [], "brief": []})

        self.assertEqual(gate["status"], "failed")
        self.assertEqual(gate["bad_title_count"], 1)
        self.assertEqual(gate["failed_item_urls"], ["https://example.com/bad-title"])

    def test_suggest_repaired_title_compresses_known_domain_titles(self):
        title = suggest_repaired_title(
            {
                "title_cn": "Amazon Bedrock A发布AgentCore paymen",
                "summary": "Amazon Bedrock AgentCore发布AgentCore payments功能，现已提供预览。",
                "facts": {"who": "Amazon Bedrock AgentCore"},
            }
        )
        self.assertEqual(title, "Amazon Bedrock AgentCore发布支付功能预览")
        self.assertEqual(
            suggest_repaired_title_with_source(
                {
                    "title_cn": "Amazon Bedrock A发布AgentCore paymen",
                    "summary": "Amazon Bedrock AgentCore发布AgentCore payments功能，现已提供预览。",
                    "facts": {"who": "Amazon Bedrock AgentCore"},
                }
            )["source"],
            "domain_compression",
        )

    def test_suggest_repaired_title_falls_back_to_preview_when_summary_title_is_still_bad(self):
        title = suggest_repaired_title(
            {
                "title_cn": "GesVLA提出gesture-aware Vision-Lang…",
                "summary": "GesVLA提出gesture-aware Vision-Language-Action模型，通过将手势作为并行指令模态来解决空间歧义问题。",
                "summary_preview": "手势指令解决空间歧义，双VLM架构提升动作精度。",
                "facts": {"who": "GesVLA", "action": "提出", "target": "gesture-aware Vision-Language-Action model"},
            }
        )
        self.assertEqual(title, "手势指令解决空间歧义，双VLM架构提升动作精度")
        self.assertFalse(title_looks_bad({"title_cn": title}))
        self.assertEqual(
            suggest_repaired_title_with_source(
                {
                    "title_cn": "GesVLA提出gesture-aware Vision-Lang…",
                    "summary": "GesVLA提出gesture-aware Vision-Language-Action模型，通过将手势作为并行指令模态来解决空间歧义问题。",
                    "summary_preview": "手势指令解决空间歧义，双VLM架构提升动作精度。",
                    "facts": {"who": "GesVLA", "action": "提出", "target": "gesture-aware Vision-Language-Action model"},
                }
            )["source"],
            "summary_preview",
        )

    def test_bad_generic_title_repairs_to_fact_specific_title(self):
        item = {
            "title_cn": "Datalab发布了一款9B参数的开 据把AI能力推进到真实工作流",
            "summary": "",
            "summary_preview": "",
            "display_topic": "开源生态",
            "facts": {
                "who": "Datalab",
                "action": "发布",
                "target": "9B开源视觉模型lift",
            },
        }

        repaired = suggest_repaired_title(item)

        self.assertEqual(repaired, "Datalab发布9B开源视觉模型lift")
        self.assertFalse(title_looks_bad({"title_cn": repaired}))

    def test_codex_research_keeps_valid_curated_title_when_fact_title_is_forced(self):
        item = {
            "title": "funes: Local Memory for Coding Agents, Built on Lance",
            "title_cn": "funes 为编码智能体提供可追溯的本地会话检索",
            "content_type": "project",
            "model_used": "codex-automation",
            "_v8_force_fact_title": True,
            "facts": {
                "who": "Aritra Roy Gosthipaty、Ayush Chaurasia",
                "action": "发布编码智能体本地记忆工具",
                "target": "funes",
                "evidence": ["支持索引多种编码智能体会话，并保留原始出处。"],
            },
            "evidence_quality": 0.88,
            "information_density": 0.87,
        }

        decorated = enrich_editorial_fields(item)

        self.assertEqual(decorated["editorial_title"], item["title_cn"])
        self.assertNotIn("mixed_language_title", decorated["editorial_flags"])
        self.assertEqual(decorated["editorial_lead"], item["title_cn"])
        self.assertNotIn("untranslated_fact", decorated["editorial_flags"])

        generic_topic_title = suggest_repaired_title(
            {
                "title_cn": "Prime Intellect发布AI领域新进展",
                "summary": "",
                "summary_preview": "",
                "display_topic": "Open Source",
                "facts": {"who": "Prime Intellect", "action": "更新", "target": ""},
            }
        )
        self.assertEqual(generic_topic_title, "Prime Intellect更新开源生态")

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

    def test_report_content_regression_sample_keeps_editorial_structure(self):
        generator = ReportGenerator(design_version="v6-editorial-learning")
        papers = [
            {
                "url": "https://paper.example/world-model",
                "title_cn": "世界模型把预测能力推向低试错规划",
                "summary_preview": "核心看点是先预测再行动的规划效率。",
                "summary": "摘要",
                "why_it_matters": "它决定世界模型能否进入真实控制任务。",
                "expected_effect": "减少长链路试错成本。",
                "future_impact": "可能成为机器人规划中间层。",
                "content_type": "paper",
                "display_topic": "世界模型",
                "score": 9.4,
                "evidence_quality": 0.8,
                "information_density": 0.8,
                "facts": {
                    "who": "世界模型论文",
                    "action": "提出",
                    "target": "先预测再行动的低试错规划方法",
                    "evidence": ["预测未来状态", "降低规划试错成本"],
                },
            },
            {
                "url": "https://paper.example/robotics",
                "title_cn": "机器人策略学习开始重视真实环境闭环",
                "summary_preview": "重点是把论文效果带到真实任务。",
                "summary": "摘要",
                "why_it_matters": "真实闭环决定部署价值。",
                "expected_effect": "提高实机成功率。",
                "future_impact": "推动具身智能落地。",
                "content_type": "paper",
                "display_topic": "机器人",
                "score": 9.1,
                "evidence_quality": 0.8,
                "information_density": 0.8,
                "facts": {
                    "who": "机器人策略论文",
                    "action": "验证",
                    "target": "真实环境闭环操作策略",
                    "evidence": ["真实任务闭环", "实机成功率提升"],
                },
            },
        ]
        updates = [
            {
                "url": "https://news.example/agent",
                "title_cn": "企业 Agent 从聊天入口走向流程执行",
                "summary_preview": "重点不是问答，而是接管多工具任务链。",
                "summary": "摘要",
                "why_it_matters": "产品正在进入真实工作流。",
                "expected_effect": "减少人工切换。",
                "future_impact": "竞争转向执行入口。",
                "content_type": "news",
                "display_topic": "产品发布",
                "score": 9.7,
                "evidence_quality": 0.8,
                "information_density": 0.8,
                "facts": {
                    "who": "企业 Agent 产品",
                    "action": "进入",
                    "target": "多工具任务链和流程执行入口",
                    "evidence": ["多工具任务链", "企业流程执行"],
                },
            },
            {
                "url": "https://news.example/infrastructure",
                "title_cn": "推理基础设施继续压低企业部署门槛",
                "summary_preview": "成本和延迟仍是落地核心变量。",
                "summary": "摘要",
                "why_it_matters": "基础设施决定应用扩张速度。",
                "expected_effect": "降低上线成本。",
                "future_impact": "改变供应关系。",
                "content_type": "news",
                "display_topic": "基础设施",
                "score": 9.3,
                "evidence_quality": 0.8,
                "information_density": 0.8,
                "facts": {
                    "who": "推理基础设施",
                    "action": "降低",
                    "target": "企业部署成本和延迟门槛",
                    "evidence": ["成本下降", "延迟优化"],
                },
            },
        ]

        mixed = generator.build_mixed_items(papers, updates)
        html = generator.generate_html(
            papers=papers,
            updates=updates,
            mixed_items=mixed,
            report_summary={
                "lead_summary": "今天的重点是 Agent 执行入口、世界模型和推理基础设施。",
                "paper_summary": "论文继续围绕低试错规划和实机闭环推进。",
                "update_summary": "产业动态集中在企业流程和部署成本。",
                "hot_topics": ["Agent", "世界模型"],
                "key_takeaways": ["执行入口正在变重要"],
                "watchlist": ["关注晚间发布"],
            },
            trend_summary={"items": []},
            alert_summary={"needs_alert": False, "issues": []},
            archive_summary={"entries": []},
            layered_updates={
                "must_read": [item for item in mixed if item["content_type"] != "paper"][:1],
                "physical_ai": [],
                "watch": [],
                "featured_papers": [item for item in mixed if item["content_type"] == "paper"][:1],
                "paper_appendix": [],
                "brief": [],
            },
        )

        self.assertIn("今日编辑判断", html)
        self.assertIn("领域技术摘编", html)
        self.assertIn("证据：", html)
        self.assertIn("继续看：", html)
        self.assertNotIn("一句话结论", html)
        self.assertIn("企业 Agent 从聊天入口走向流程执行", html)

    def test_html_report_excludes_highlight_duplicates_from_body(self):
        generator = ReportGenerator(design_version="v6-editorial-learning")
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
            layered_updates={
                "must_read": [item for item in mixed if item["content_type"] != "paper"][:1],
                "physical_ai": [],
                "watch": [],
                "featured_papers": [item for item in mixed if item["content_type"] == "paper"][:1],
                "paper_appendix": [],
                "brief": [],
            },
        )
        self.assertIn("今日编辑判断", html)
        self.assertIn("领域技术摘编", html)

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
            self.assertNotIn("历史回补", joined)
            self.assertIn("去重后全网动态仅保留 16 条", joined)
            self.assertIn("最终论文仅保留 10 篇", joined)

    def test_build_quality_diagnostics_explains_selection_shortfall(self):
        diagnostics = build_quality_diagnostics(
            current_papers_count=3,
            current_updates_count=8,
            report_items_count=18,
            prepared_items_count=18,
            paper_candidate_count=8,
            update_candidate_count=30,
            deduped_update_count=16,
            selected_paper_count=10,
            selected_update_count=16,
            paper_limit=15,
            min_paper_count=12,
            web_limit=20,
            min_web_items=20,
            paper_backfill_hours_used=[168],
            web_backfill_hours_used=[24, 48],
            collector_summary={"fresh_items": 5, "success_count": 8, "failed_count": 0, "timeout_count": 0, "skipped_count": 1},
            bad_title_count=2,
        )

        self.assertEqual(diagnostics["targets"]["paper_limit"], 15)
        self.assertEqual(diagnostics["targets"]["min_paper_count"], 12)
        self.assertEqual(diagnostics["selection"]["update_candidates_after_dedupe"], 16)
        self.assertIn("selected_papers_below_minimum:10/12", diagnostics["warnings"])
        self.assertIn("dedupe_removed_updates:14", diagnostics["warnings"])
        self.assertIn("bad_titles_present:2", diagnostics["warnings"])
        self.assertEqual(diagnostics["content_quality"]["bad_title_count"], 2)

    def test_build_quality_diagnostics_accepts_papers_inside_configured_range(self):
        diagnostics = build_quality_diagnostics(
            current_papers_count=0,
            current_updates_count=20,
            report_items_count=40,
            prepared_items_count=40,
            paper_candidate_count=11,
            update_candidate_count=29,
            deduped_update_count=29,
            selected_paper_count=11,
            selected_update_count=20,
            paper_limit=15,
            min_paper_count=10,
            web_limit=20,
            min_web_items=20,
            paper_backfill_hours_used=[72],
            web_backfill_hours_used=[],
            collector_summary={"fresh_items": 20, "success_count": 22, "failed_count": 0, "timeout_count": 0, "skipped_count": 0},
        )

        self.assertNotIn("selected_papers_below_minimum:11/10", diagnostics["warnings"])

    def test_build_source_health_summary_tracks_failures_and_last_success(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db = Database(str(Path(temp_dir) / "test.db"))
            db.record_collector_runs(
                "run1",
                [{"label": "RSSCollector[Test Feed]", "status": "success", "inserted_count": 1, "collected_count": 2, "duration_seconds": 1.0, "error": ""}],
            )
            db.record_collector_runs(
                "run2",
                [{"label": "RSSCollector[Test Feed]", "status": "timeout", "inserted_count": 0, "collected_count": 0, "duration_seconds": 1.0, "error": "timeout"}],
            )

            summary = build_source_health_summary(
                [{"label": "RSSCollector[Test Feed]", "status": "timeout", "inserted_count": 0, "collected_count": 0, "duration_seconds": 1.0, "error": "timeout"}],
                db,
                history_limit=20,
            )

        row = summary["rows"][0]
        self.assertEqual(summary["risky_source_count"], 1)
        self.assertEqual(row["consecutive_failures"], 1)
        self.assertNotEqual(row["last_success_at"], "")

    def test_source_health_weights_penalize_unhealthy_sources(self):
        source_health = {
            "rows": [
                {"label": "RSSCollector[Bad Feed]", "consecutive_failures": 2, "recent_empty_success_count": 1, "recent_success_count": 0},
                {"label": "RSSCollector[Good Feed]", "consecutive_failures": 0, "recent_empty_success_count": 0, "recent_success_count": 4},
            ]
        }

        weights = build_source_health_weights(source_health, {"enabled": True, "penalty_per_consecutive_failure": -0.5, "bonus_recent_success": 0.2})

        self.assertLess(weights["RSSCollector[Bad Feed]"], 0)
        self.assertGreater(weights["RSSCollector[Good Feed]"], 0)

    def test_apply_source_health_adjustments_merges_dynamic_weights(self):
        adjusted, summary = apply_source_health_adjustments(
            {"source_weights": {"RSSCollector[Bad Feed]": -0.25}},
            {"rows": [{"label": "RSSCollector[Bad Feed]", "consecutive_failures": 2, "recent_empty_success_count": 0, "recent_success_count": 0}]},
            {"enabled": True, "penalty_per_consecutive_failure": -0.5},
        )

        self.assertEqual(summary["weights"]["RSSCollector[Bad Feed]"], -1.0)
        self.assertEqual(adjusted["source_weights"]["RSSCollector[Bad Feed]"], -1.25)

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

    def test_fact_first_fallback_rewrites_generic_summary(self):
        processor = LLMProcessor({"api_key_env": "THIS_KEY_SHOULD_NOT_EXIST"})
        article = {
            "content_type": "news",
            "title": "OpenAI launches workflow agent for enterprise task execution",
            "content": "OpenAI launches a workflow agent for enterprise task execution. The product can call tools across long-running business workflows and reduce manual switching.",
            "source_detail": "OpenAI Blog",
            "platform": "Blog",
        }
        result = processor.prepare_report_item(
            article,
            {
                "title_cn": "OpenAI有新动作",
                "summary": "相关机构在产品发布方向出现了新的动作。从当前信息看，这次变化不只是单点更新，更可能影响产品路线、合作节奏或市场竞争。",
                "category": "产品发布",
                "topic_cn": "产品发布",
                "score": 8.8,
            },
        )
        banned = ["相关机构", "出现了新的动作", "不只是单点更新", "可能影响产品路线", "值得持续关注", "未来可能带来影响"]
        self.assertFalse(any(token in result["summary"] for token in banned))
        self.assertEqual(result["facts"]["who"], "OpenAI")
        self.assertIn(result["facts"]["action"], ["发布", "推出"])
        self.assertGreaterEqual(result["evidence_quality"], 0.45)

    def test_report_summary_uses_fact_lines_instead_of_short_snippets(self):
        processor = LLMProcessor({"api_key_env": "THIS_KEY_SHOULD_NOT_EXIST"})
        report_summary = processor.summarize_report(
            papers=[],
            updates=[
                {
                    "title_cn": "OpenAI发布企业工作流智能体",
                    "display_topic": "产品发布",
                    "score": 9.0,
                    "summary": "OpenAI 发布企业工作流智能体。",
                    "facts": {
                        "who": "OpenAI",
                        "action": "发布",
                        "target": "企业工作流智能体",
                        "evidence": ["可跨工具执行长任务并减少人工切换"],
                        "audience": "企业用户和开发者",
                    },
                }
            ],
        )
        self.assertIn("OpenAI", report_summary["update_summary"])
        self.assertIn("企业工作流智能体", report_summary["update_summary"])
        self.assertFalse(any(token in report_summary["lead_summary"] for token in ["出现了新的动作", "值得持续关注"]))

    def test_low_evidence_summary_is_short_and_score_is_capped(self):
        processor = LLMProcessor({"api_key_env": "THIS_KEY_SHOULD_NOT_EXIST"})
        result = processor.prepare_report_item(
            {
                "content_type": "news",
                "title": "AI startup teases new product",
                "content": "Short teaser.",
                "source_detail": "Google News",
                "platform": "Web",
            },
            {
                "summary": "AI startup teases new product.",
                "facts": {"who": "", "action": "", "target": "", "evidence": [], "audience": ""},
                "category": "产品发布",
                "topic_cn": "产品发布",
                "score": 8.4,
            },
        )
        self.assertLessEqual(result["score"], 5.2)
        self.assertLessEqual(len(result["summary"]), 130)
        self.assertNotIn("发生了什么", result["summary"])
        self.assertNotIn("证据是什么", result["summary"])

    def test_analysis_fields_are_deduped_when_they_repeat_summary(self):
        processor = LLMProcessor({"api_key_env": "THIS_KEY_SHOULD_NOT_EXIST"})
        result = processor.prepare_report_item(
            {
                "content_type": "news",
                "title": "OpenAI launches workflow agent for enterprise task execution",
                "content": "OpenAI launches a workflow agent for enterprise task execution. The product can call tools across business workflows and reduce manual switching.",
                "source_detail": "OpenAI Blog",
                "platform": "Blog",
            },
            {
                "summary": "OpenAI发布企业工作流智能体。它可以跨工具执行业务流程并减少人工切换。企业用户会先感受到流程自动化覆盖范围扩大。下一步要看客户采用和执行成功率。",
                "why_now": "OpenAI发布企业工作流智能体。它可以跨工具执行业务流程并减少人工切换。",
                "expected_effect": "OpenAI发布企业工作流智能体。它可以跨工具执行业务流程并减少人工切换。",
                "future_impact": "OpenAI发布企业工作流智能体。它可以跨工具执行业务流程并减少人工切换。",
                "category": "Product Release",
                "topic_cn": "Product Release",
                "score": 8.6,
            },
        )
        fields = [result["why_now"], result["expected_effect"], result["future_impact"]]
        non_empty = [field for field in fields if field]
        self.assertEqual(len(non_empty), len(set(non_empty)))
        self.assertFalse(all("跨工具执行业务流程" in field for field in non_empty))

    def test_title_summary_consistency_rewrites_off_topic_summary(self):
        processor = LLMProcessor({"api_key_env": "THIS_KEY_SHOULD_NOT_EXIST"})
        result = processor.prepare_report_item(
            {
                "content_type": "news",
                "title": "NVIDIA launches new inference chip for enterprise AI servers",
                "content": "NVIDIA launches a new inference chip for enterprise AI servers. The chip targets lower latency and datacenter deployment cost.",
                "source_detail": "NVIDIA Blog",
                "platform": "Blog",
            },
            {
                "summary": "这条行业动态说明市场正在加速变化。企业需要继续观察不同厂商的后续节奏。",
                "category": "Infrastructure",
                "topic_cn": "Infrastructure",
                "score": 8.4,
            },
        )
        joined = result["summary"] + json.dumps(result["facts"], ensure_ascii=False)
        self.assertIn("NVIDIA", joined)
        self.assertTrue(any(token in joined for token in ["芯片", "inference chip", "基础设施"]))
        self.assertGreater(result["information_density"], 0.35)

    def test_fact_array_from_primary_model_is_normalized_for_chinese_summary(self):
        processor = LLMProcessor({"api_key_env": "THIS_KEY_SHOULD_NOT_EXIST"})
        result = processor.prepare_report_item(
            {
                "content_type": "paper",
                "title": "G-DRAGON: Geospatial Reasoning and Dynamic Planning for Retrieval-Augmented Outdoor Navigation",
                "content": "The framework maps natural-language commands to local OSM entities and outperforms baselines in simulation. It completes person-search missions up to 500m in real-world urban environments.",
                "source_detail": "ArXiv",
                "platform": "ArXiv",
                "topic": "Robotics",
            },
            {
                "facts": [
                    {
                        "who": "G-DRAGON",
                        "action": "proposes",
                        "target": "retrieval-augmented framework for outdoor open-world navigation",
                        "evidence": ["outperforms baselines in simulation", "completes person-search missions up to 500m"],
                        "audience": "autonomous robot researchers",
                    }
                ],
                "category": "Robotics",
                "topic_cn": "机器人",
                "score": 8.0,
                "_facts_first_only": True,
                "_model_used": "deepseek-v4-pro",
            },
        )
        self.assertEqual(result["model_used"], "deepseek-v4-pro")
        self.assertEqual(result["facts"]["action"], "提出")
        self.assertNotRegex(result["title_cn"], r"(.{4,}).*\\1")
        self.assertTrue(result["summary"])

    def test_golden_generation_samples_are_specific_and_not_generic(self):
        processor = LLMProcessor({"api_key_env": "THIS_KEY_SHOULD_NOT_EXIST"})
        samples = [
            ("OpenAI launches workflow agent for enterprise task execution", "OpenAI launches a workflow agent that can call tools across long-running business processes.", "OpenAI Blog", "Blog", "Product Release"),
            ("Anthropic expands Claude tools for enterprise finance teams", "Anthropic expands Claude tool use for finance workflows and internal reporting tasks.", "Anthropic", "Blog", "Product Release"),
            ("NVIDIA launches inference platform for enterprise datacenters", "NVIDIA launches an inference platform focused on latency, GPU utilization and datacenter deployment.", "NVIDIA Blog", "Blog", "Infrastructure"),
            ("Microsoft partners with Mistral to bring models to Azure", "Microsoft and Mistral announce an Azure partnership for model access and enterprise distribution.", "Microsoft Blog", "Blog", "Partnership"),
            ("Meta open sources a robotics benchmark dataset", "Meta releases a robotics benchmark dataset with manipulation tasks and evaluation scripts.", "Meta AI", "Blog", "Open Source"),
            ("Google DeepMind proposes world model benchmark", "Google DeepMind proposes a benchmark for world model planning and long-horizon prediction.", "DeepMind Blog", "Blog", "Model/Research"),
            ("Hugging Face releases new open model evaluation leaderboard", "Hugging Face releases a leaderboard for comparing open model performance across tasks.", "Hugging Face", "Blog", "Open Source"),
            ("AWS adds Trainium instances for large model inference", "AWS adds Trainium instances for enterprise inference workloads and cloud deployment.", "AWS Blog", "Blog", "Infrastructure"),
            ("Perplexity launches enterprise search connector", "Perplexity launches connectors for enterprise search across internal knowledge bases.", "Perplexity", "Blog", "Product Release"),
            ("AMD announces AI server chip roadmap", "AMD announces an AI server chip roadmap focused on inference and datacenter customers.", "Industry Media", "Web", "Infrastructure"),
            ("Apple research paper improves on-device multimodal models", "Apple researchers describe an on-device multimodal model with latency and privacy constraints.", "ArXiv", "ArXiv", "Model/Research"),
            ("Tesla updates humanoid robot manipulation policy", "Tesla updates a humanoid robot manipulation policy for factory tasks and object handling.", "Company Blog", "Blog", "应用落地"),
            ("Cohere releases enterprise reranking model", "Cohere releases a reranking model for enterprise retrieval and customer support workflows.", "Cohere", "Blog", "Product Release"),
            ("Mistral publishes small model for edge deployment", "Mistral publishes a small model intended for edge deployment and lower inference cost.", "Mistral", "Blog", "Open Source"),
            ("Marvell and NVIDIA deepen AI infrastructure partnership", "Marvell and NVIDIA deepen an infrastructure partnership around custom chips and delivery.", "Industry Media", "Web", "Partnership"),
            ("Google Cloud adds TPU capacity for AI startups", "Google Cloud adds TPU capacity and startup credits for AI training workloads.", "Google Cloud Blog", "Blog", "Infrastructure"),
            ("xAI releases Grok API update for developers", "xAI releases a Grok API update with developer tooling and model access changes.", "xAI", "Blog", "Product Release"),
            ("Robot learning paper reduces manipulation trial cost", "A robot learning paper reduces manipulation trial cost through simulated policy training.", "ArXiv", "ArXiv", "Robotics"),
            ("World model paper improves long-horizon video prediction", "A world model paper improves long-horizon video prediction and planning reliability.", "ArXiv", "ArXiv", "World Model"),
            ("Google News thin item about AI product", "Brief mention only.", "Google News", "Web", "Product Release"),
        ]
        banned = ["相关机构", "出现了新的动作", "不只是单点更新", "可能影响产品路线", "值得持续关注", "未来可能带来影响"]
        for title, content, source_detail, platform, category in samples:
            result = processor.prepare_report_item(
                {"content_type": "paper" if platform == "ArXiv" else "news", "title": title, "content": content, "source_detail": source_detail, "platform": platform, "topic": category},
                {"summary": content, "category": category, "topic_cn": category, "score": 8.0},
            )
            self.assertFalse(any(token in result["summary"] for token in banned), title)
            self.assertNotRegex(result["summary"], r"发生了什么|证据是什么|对谁有影响|下一步看什么")
            if source_detail == "Google News":
                self.assertLessEqual(result["score"], 5.2, title)
            else:
                self.assertGreaterEqual(result["facts"].get("confidence", 0), 0.35, title)
            self.assertGreaterEqual(processor._title_summary_consistency_score({"title": title, "title_cn": result["title_cn"]}, result["facts"], result["summary"]), 1, title)

    def test_low_evidence_items_are_not_promoted_to_highlights_when_enough_strong_items_exist(self):
        generator = ReportGenerator()
        strong_items = [
            {
                "url": f"https://example.com/strong-{index}",
                "title_cn": f"高证据动态 {index}",
                "content_type": "news" if index < 3 else "paper",
                "score": 8.8 - index * 0.1,
                "evidence_quality": 0.8,
                "information_density": 0.8,
            }
            for index in range(5)
        ]
        low_item = {
            "url": "https://news.google.com/rss/articles/thin",
            "title_cn": "低证据聚合动态",
            "content_type": "news",
            "score": 9.9,
            "evidence_quality": 0.2,
            "information_density": 0.2,
        }
        highlights = generator._pick_top_highlights([low_item] + strong_items)
        self.assertNotIn(low_item["url"], {item["url"] for item in highlights})

    def test_decorated_cards_hide_generic_analysis_and_shortens_low_evidence_summary(self):
        generator = ReportGenerator()
        decorated = generator._decorate_items(
            [
                {
                    "url": "https://news.google.com/rss/articles/thin",
                    "title_cn": "低证据聚合动态",
                    "summary": "发生了什么：某公司发布AI产品。证据是什么：来源信息不足。对谁有影响：相关团队。下一步看什么：客户采用和落地数据。",
                    "why_it_matters": "值得持续关注其未来可能影响。",
                    "why_now": "后续要看客户采用和落地数据。",
                    "expected_effect": "后续要看客户采用和落地数据。",
                    "future_impact": "后续要看客户采用和落地数据。",
                    "content_type": "news",
                    "score": 5.0,
                    "evidence_quality": 0.2,
                    "information_density": 0.2,
                }
            ]
        )
        self.assertNotIn("发生了什么", decorated[0]["summary_display"])
        self.assertLessEqual(len(decorated[0]["summary_display"]), 151)
        self.assertEqual(decorated[0]["analysis_items"], [])


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

    def test_diversify_report_titles_does_not_rewrite_verified_codex_titles(self):
        processor = LLMProcessor({"api_key_env": "THIS_KEY_SHOULD_NOT_EXIST"})
        items = [
            {
                "url": "https://example.com/codex-1",
                "title_cn": "OpenAI把审批权限接入企业智能体工作流",
                "summary": "正文一",
                "model_used": "codex-automation",
            },
            {
                "url": "https://example.com/codex-2",
                "title_cn": "Anthropic把审批权限接入企业智能体工作流",
                "summary": "正文二",
                "model_used": "codex-automation",
            },
        ]

        diversified = diversify_report_titles(items, processor)

        self.assertEqual([item["title_cn"] for item in diversified], [item["title_cn"] for item in items])


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
        joined_analysis = result["why_now"] + result["expected_effect"] + result["future_impact"]
        self.assertTrue(any(token in joined_analysis for token in ["人工", "自动化", "流程", "工作流"]))

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
        generator = ReportGenerator(design_version="v6-editorial-learning")
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
                "evidence_quality": 0.8,
                "information_density": 0.8,
                "facts": {
                    "who": "产品团队",
                    "action": "推进",
                    "target": "模型能力进入真实工作流",
                    "evidence": ["工作流入口", "付费入口"],
                },
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
                "evidence_quality": 0.8,
                "information_density": 0.8,
                "facts": {
                    "who": "世界模型研究",
                    "action": "降低",
                    "target": "规划和控制中的试错成本",
                    "evidence": ["先预测再行动", "低试错决策"],
                },
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
            layered_updates={
                "must_read": [item for item in mixed if item["content_type"] != "paper"][:1],
                "physical_ai": [],
                "watch": [],
                "featured_papers": [item for item in mixed if item["content_type"] == "paper"][:1],
                "paper_appendix": [],
                "brief": [],
            },
        )
        self.assertIn("证据：", html)
        self.assertIn("继续看：", html)
        self.assertNotIn("一句话结论", html)

    def test_v23_template_adds_physical_ai_tracking_and_light_feedback(self):
        generator = ReportGenerator(design_version="v6-editorial-learning")
        physical_item = {
            "id": 77,
            "url": "https://example.com/robotics",
            "title_cn": "人形机器人进入仓储试点",
            "summary_preview": "人形机器人开始进入仓储搬运和分拣场景。",
            "summary": "公司披露人形机器人仓储试点，证据包括客户场景和真实操作任务。",
            "why_it_matters": "具身智能最需要观察真实客户场景和稳定执行能力。",
            "expected_effect": "它会先影响仓储自动化团队的试点路线。",
            "future_impact": "后续要看客户复购、故障率和任务覆盖范围。",
            "content_type": "news",
            "display_topic": "具身智能",
            "source_tier": "official",
            "evidence_quality": 0.75,
            "information_density": 0.7,
            "facts": {
                "who": "RobotCo",
                "action": "披露",
                "target": "人形机器人仓储试点",
                "evidence": ["客户场景", "真实操作任务"],
                "audience": "仓储自动化团队",
            },
            "feedback_links": {
                "useful": "http://127.0.0.1:8765/feedback?signal=useful",
                "track": "http://127.0.0.1:8765/feedback?signal=track",
                "not_useful": "http://127.0.0.1:8765/feedback?signal=not_useful",
                "mute_similar": "http://127.0.0.1:8765/feedback?signal=mute_similar",
            },
        }
        html = generator.generate_html(
            papers=[],
            updates=[physical_item],
            mixed_items=[physical_item],
            report_summary={"lead_summary": "具身智能进入真实场景。", "hot_topics": ["具身智能"]},
            trend_summary={"items": []},
            alert_summary={"needs_alert": False, "issues": []},
            archive_summary={"entries": []},
            layered_updates={
                "must_read": [],
                "physical_ai": [physical_item],
                "watch": [],
                "research": [],
                "brief": [],
            },
            title="AI Frontier Intelligence Daily",
        )
        self.assertIn("领域技术摘编", html)
        self.assertIn("Physical AI / Robotics", html)
        self.assertIn("人形机器人", html)
        self.assertIn("次日追踪清单", html)
        self.assertIn("是否出现真实客户案例", html)
        self.assertIn("证据：", html)
        self.assertIn("继续看：", html)
        self.assertNotIn("技术 / 背景解释", html)
        self.assertIn("有用", html)
        self.assertIn("跟进", html)
        self.assertNotIn("没用", html)
        self.assertNotIn("屏蔽类似", html)


    def test_feed_cards_include_quick_reason_block(self):
        generator = ReportGenerator(design_version="v6-editorial-learning")
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

    def test_v2_email_template_keeps_snapshot_modules_and_compatibility_markers(self):
        generator = ReportGenerator(design_version="v6-editorial-learning")
        base_item = {
            "id": 1,
            "url": "https://example.com/item",
            "title_cn": "AgentCore发布支付能力预览",
            "summary_preview": "AgentCore把支付入口放进代理工作流。",
            "summary": "AgentCore发布支付能力预览，证据来自官方技术深潜文章，影响需要把代理接入交易流程的企业团队。",
            "why_it_matters": "它把代理从问答推进到可执行商业流程。",
            "why_now": "企业开始测试代理商业闭环。",
            "expected_effect": "先影响电商、客服和企业采购流程。",
            "future_impact": "后续看真实客户采用和交易安全边界。",
            "evidence_points": ["官方技术文章", "支付能力预览", "企业工作流"],
            "content_type": "news",
            "source_tier": "official",
            "display_topic": "产品发布",
            "score": 9.1,
            "evidence_quality": 0.85,
            "information_density": 0.8,
            "facts": {"who": "AgentCore", "action": "发布", "target": "支付能力预览"},
        }
        html = generator.generate_html(
            papers=[],
            updates=[base_item],
            mixed_items=[base_item],
            report_summary={
                "lead_summary": "今天的变化集中在代理进入交易流程。",
                "paper_summary": "",
                "update_summary": "",
                "hot_topics": ["产品发布"],
                "key_takeaways": ["代理能力继续向执行入口推进。"],
                "watchlist": ["观察支付安全和客户采用。"],
            },
            trend_summary={"items": []},
            alert_summary={"needs_alert": False, "issues": []},
            archive_summary={"entries": []},
            layered_updates={"must_read": [base_item], "physical_ai": [], "watch": [], "research": [], "brief": []},
            title="AI Frontier Intelligence Daily",
        )

        for marker in (
            "AI Frontier Intelligence",
            "今日编辑判断",
            "领域技术摘编",
            "质量与来源脚注",
            "data-design-version=\"v6-editorial-learning\"",
            "x-apple-disable-message-reformatting",
            "<!--[if mso]>",
            "preheader",
        ):
            self.assertIn(marker, html)

    def test_world_model_focus_section_tracks_training_and_leader_views(self):
        generator = ReportGenerator(design_version="v6-editorial-learning")
        training_item = {
            "id": 31,
            "url": "https://example.com/world-model-training",
            "title_cn": "世界模型训练开始转向潜空间动态建模",
            "summary_preview": "这篇文章总结 latent dynamics、video prediction 和 rollout 训练路线。",
            "summary": "世界模型训练技术开始强调潜空间动态建模、长时预测和规划 rollout。",
            "content_type": "news",
            "source_tier": "media",
            "display_topic": "世界模型",
            "score": 9.0,
            "evidence_quality": 0.7,
            "information_density": 0.7,
            "facts": {
                "who": "研究社区",
                "action": "总结",
                "target": "world model training with latent dynamics and video prediction",
                "evidence": ["latent dynamics", "video prediction", "rollout planning"],
            },
        }
        leader_view = {
            "id": 32,
            "url": "https://example.com/lecun-world-model",
            "title_cn": "Yann LeCun访谈谈世界模型和JEPA路线",
            "summary_preview": "LeCun认为世界模型需要通过JEPA式预测学习理解物理世界。",
            "summary": "Yann LeCun在访谈中谈到世界模型、JEPA和预测学习路线。",
            "content_type": "news",
            "source_tier": "media",
            "display_topic": "世界模型",
            "score": 8.8,
            "evidence_quality": 0.65,
            "information_density": 0.65,
            "facts": {
                "who": "Yann LeCun",
                "action": "访谈",
                "target": "world model and JEPA",
                "evidence": ["interview", "JEPA", "predictive learning"],
            },
        }
        html = generator.generate_html(
            papers=[],
            updates=[training_item, leader_view],
            mixed_items=[training_item, leader_view],
            report_summary={"lead_summary": "世界模型训练路线继续变化。", "hot_topics": ["世界模型"]},
            trend_summary={"items": []},
            alert_summary={"needs_alert": False, "issues": []},
            archive_summary={"entries": []},
            layered_updates={
                "must_read": [training_item],
                "physical_ai": [],
                "watch": [leader_view],
                "research": [],
                "brief": [],
            },
            title="AI Frontier Intelligence Daily",
        )
        self.assertIn("World Model", html)
        self.assertIn("领域技术摘编", html)
        self.assertIn("今日先读", html)
        self.assertIn("世界模型训练开始转向潜空间动态建模", html)
        self.assertIn("Yann LeCun访谈谈世界模型和JEPA路线", html)

    def test_paper_description_uses_substantive_paper_description(self):
        generator = ReportGenerator(design_version="v6-editorial-learning")
        item = {
            "url": "https://example.com/paper",
            "title_cn": "FineVLA提升具身智能操作成功率",
            "summary_preview": "FineVLA通过细粒度指令对齐提升机器人操作成功率。",
            "summary": "FineVLA提出细粒度指令对齐框架，旨在提升具身智能中视觉-语言-动作策略的可控性。实验基于47,159条细粒度轨迹，成功率提升明显。",
            "content_type": "paper",
            "display_topic": "具身智能",
            "source_tier": "research",
            "evidence_quality": 0.8,
            "information_density": 0.75,
            "facts": {
                "target": "机器人操作中的细粒度指令对齐",
                "action": "提出细粒度指令对齐框架",
                "audience": "具身智能和机器人操作研究者",
                "evidence": ["47,159条细粒度轨迹", "成功率提升明显"],
            },
        }

        decorated = generator._decorate_items([item])
        paper_description = decorated[0]["paper_description"]
        technical_intro = decorated[0]["paper_technical_intro"]

        self.assertIn("FineVLA", paper_description)
        self.assertIn("机器人操作", paper_description)
        self.assertIn("47,159", paper_description)
        self.assertIn("新物体", paper_description)
        self.assertNotIn("白话总结", paper_description)
        self.assertNotIn("解决什么", paper_description)
        self.assertNotIn("适合谁读", paper_description)
        self.assertLessEqual(len(paper_description), 190)
        self.assertLessEqual(len([part for part in paper_description.split("。") if part.strip()]), 3)
        self.assertNotIn("技术上", technical_intro)
        self.assertIn("关键做法", technical_intro)
        self.assertIn("细粒度指令对齐", technical_intro)
        self.assertIn("47,159", technical_intro)
        self.assertNotIn("解决什么", technical_intro)
        self.assertNotIn("适合谁读", technical_intro)
        self.assertIsNone(generator.BAD_PUBLIC_SENTENCE_PATTERN.search(technical_intro))
        self.assertLessEqual(len(technical_intro), 270)
        self.assertEqual(decorated[0]["memory_point"], "")
        self.assertEqual(decorated[0]["plain_summary"], "")
        self.assertIn("关键做法", decorated[0]["paper_technical_intro"])

    def test_research_card_uses_paper_description_without_plain_summary(self):
        generator = ReportGenerator()
        paper = {
            "url": "https://example.com/ahead",
            "title_cn": "AHEAD提出动态操作预测方法",
            "summary": "AHEAD用运动感知潜在世界模型增强冻结VLA。",
            "content_type": "paper",
            "display_topic": "具身智能",
            "source_tier": "research",
            "evidence_quality": 0.8,
            "information_density": 0.75,
            "facts": {
                "who": "AHEAD",
                "action": "提出",
                "target": "a predict-then-act wrapper that augments a frozen VLA with a motion-aware latent world model",
                "evidence": ["Adds 4.9M parameters to a frozen 7B OpenVLA"],
            },
        }
        html = generator.generate_html(
            papers=[paper],
            updates=[],
            mixed_items=[paper],
            report_summary={"lead_summary": "论文侧重点是动态机器人操作。", "hot_topics": ["具身智能"]},
            trend_summary={"items": []},
            alert_summary={"needs_alert": False, "issues": []},
            archive_summary={"entries": []},
            layered_updates={"must_read": [], "physical_ai": [], "watch": [], "research": [paper], "brief": []},
        )
        self.assertIn("技术介绍", html)
        self.assertNotIn("论文实质描述", html)
        self.assertNotIn("技术方向摘编", html)
        self.assertNotIn("白话总结", html)
        self.assertNotIn("核心知识点", html)
        self.assertIn("预测动态物体的下一步位置", html)
        self.assertIn("冻结 VLA", html)

    def test_paper_technical_intro_avoids_generic_fallback_for_real_paper_shapes(self):
        generator = ReportGenerator()
        papers = [
            {
                "title_cn": "Lie-Algebra Attention将token视为矩阵李群",
                "summary": "Lie-Algebra Attention treats tokens as matrix Lie groups and evaluates on two benchmarks.",
                "content_type": "paper",
                "display_topic": "世界模型",
                "facts": {
                    "who": "Lie-Algebra Attention",
                    "target": "attention mechanism using matrix Lie groups",
                    "evidence": ["Evaluated on 2 benchmarks"],
                },
            },
            {
                "title_cn": "GazeLNN提出轻量级扫描路径预测模型",
                "summary": "GazeLNN proposes a computationally lightweight scanpath prediction model based on liquid neural networks.",
                "content_type": "paper",
                "display_topic": "具身智能",
                "facts": {
                    "who": "GazeLNN",
                    "target": "computationally lightweight scanpath prediction model",
                    "evidence": ["Reports 0.47 scanpath prediction metric"],
                },
            },
        ]

        intros = [generator._paper_technical_intro(paper) for paper in papers]

        self.assertIn("矩阵李群", intros[0])
        self.assertIn("李代数", intros[0])
        self.assertIn("扫描路径", intros[1])
        self.assertIn("液态神经网络", intros[1])
        for intro in intros:
            self.assertTrue(generator._paper_technical_intro_is_substantive(intro), intro)
            self.assertNotIn("围绕核心任务", intro)
            self.assertNotIn("真正要判断的是这个方法", intro)
            self.assertNotIn("论文给出了实验、基准或真实任务验证", intro)

    def test_recent_failed_paper_shapes_generate_valid_technical_intros(self):
        generator = ReportGenerator()
        papers = [
            {
                "title": "ATOM-Bench: A Benchmark for Real-World Atomic Manipulation",
                "summary": "ATOM-Bench builds a benchmark for atom-level manipulation tasks and compares robot policies across controlled real-world scenarios.",
                "content_type": "paper",
                "display_topic": "Physical AI",
                "evidence_quality": 0.8,
                "information_density": 0.75,
                "facts": {
                    "who": "ATOM-Bench",
                    "action": "introduces",
                    "target": "a real-world benchmark for atomic manipulation tasks",
                    "method": "standardizes atom-level manipulation tasks into a shared evaluation suite",
                    "dataset_or_benchmark": "ATOM-Bench real-world manipulation benchmark",
                    "metric_result": "compares robot policies across benchmark tasks",
                    "baseline": "task-specific robot policies",
                    "limitation": "coverage depends on the included physical tasks",
                    "evidence": [
                        "Introduces ATOM-Bench as a real-world benchmark",
                        "Compares robot policies across atom-level manipulation tasks",
                    ],
                },
            },
            {
                "title": "PhysDrift: Bridging the Embodiment Gap in Humanoid Co-Speech Motion Generation",
                "summary": "PhysDrift targets humanoid co-speech motion generation by adding physics-aware constraints so generated gestures stay closer to embodied motion.",
                "content_type": "paper",
                "display_topic": "Physical AI",
                "evidence_quality": 0.82,
                "information_density": 0.78,
                "facts": {
                    "who": "PhysDrift",
                    "action": "proposes",
                    "target": "physics-aware humanoid co-speech motion generation",
                    "method": "adds physical drift constraints before generating full-body co-speech motion",
                    "dataset_or_benchmark": "humanoid co-speech motion benchmark",
                    "metric_result": "reports more physically consistent motion than baseline generators",
                    "baseline": "text-to-motion and co-speech motion generation baselines",
                    "limitation": "still needs real robot deployment validation",
                    "evidence": [
                        "Targets the embodiment gap in humanoid co-speech motion generation",
                        "Evaluates physical consistency against motion generation baselines",
                    ],
                },
            },
            {
                "title": "G$^3$VLA: Camera-aware Geometric Module for Vision-Language-Action Models",
                "summary": "G$^3$VLA adds intrinsic-conditioned ray embeddings, PRoPE and bidirectional cross-view fusion to VLA models, with gains on LIBERO, RoboCasa24, RoboTwin2.0 and real-robot settings.",
                "content_type": "paper",
                "display_topic": "Physical AI",
                "evidence_quality": 0.9,
                "information_density": 0.86,
                "facts": {
                    "who": "G$^3$VLA",
                    "action": "proposes",
                    "target": "camera-aware geometric module for Vision-Language-Action models",
                    "method": "intrinsic-conditioned ray embeddings, projective positional encoding (PRoPE), and bidirectional cross-view fusion",
                    "dataset_or_benchmark": "LIBERO suites, RoboCasa24, RoboTwin2.0, real-robot settings",
                    "metric_result": "consistent gains, largest improvements on spatially and object-sensitive tasks",
                    "baseline": "pretrained VLA without geometric module (π0, π0.5, GR00T 1.5)",
                    "limitation": "geometric transfer most effective when geometry-aware tokens have direct access to action generation pathway",
                    "evidence": [
                        "Consistent gains across LIBERO suites, RoboCasa24, RoboTwin2.0, and real-robot settings",
                        "Validated on π0, π0.5, and GR00T 1.5",
                    ],
                },
            },
        ]

        decorated = generator._decorate_items(papers)
        metrics = build_editorial_quality_metrics(decorated, paper_technical_intro_min_count=3)

        self.assertEqual(metrics["paper_technical_intro_fail_count"], 0)
        self.assertGreaterEqual(metrics["paper_technical_intro_pass_count"], 3)
        for item in decorated:
            intro = item["paper_technical_intro"]
            self.assertTrue(paper_technical_intro_passes(intro), intro)
            self.assertIn("关键做法", intro)
            self.assertNotIn("技术上，它主要围绕", intro)
            self.assertNotIn("需要回看原文确认", intro)
        self.assertIn("PRoPE", decorated[2]["paper_technical_intro"])
        self.assertIn("跨视角融合", decorated[2]["paper_technical_intro"])
        self.assertIn("预训练 VLA 基线", decorated[2]["paper_technical_intro"])
        self.assertNotIn("材料给出0", decorated[2]["paper_technical_intro"])

    def test_success_visitation_matching_intro_has_result_signal(self):
        generator = ReportGenerator()
        paper = {
            "title_cn": "通过Success Visitation Matching将稀疏奖励转化为密集过程奖励",
            "summary": "Success Visitation Matching converts sparse rewards into dense process rewards by matching successful state visitation.",
            "content_type": "paper",
            "display_topic": "Agent / Models",
            "evidence_quality": 0.7,
            "information_density": 0.7,
            "facts": {
                "who": "Success Visitation Matching",
                "target": "converting sparse rewards into dense process rewards",
                "evidence": ["evaluates sparse reward tasks against process reward baselines"],
            },
        }
        intro = generator._decorate_items([paper])[0]["paper_technical_intro"]

        self.assertTrue(paper_technical_intro_passes(intro), intro)
        self.assertIn("密集过程奖励", intro)
        self.assertIn("成功率", intro)

    def test_paper_appendix_renders_and_snapshots_technical_intro(self):
        generator = ReportGenerator()
        paper = {
            "id": 7001,
            "url": "https://arxiv.org/abs/example",
            "title_cn": "GazeLNN提出轻量级扫描路径预测模型",
            "summary": "GazeLNN proposes a computationally lightweight scanpath prediction model based on liquid neural networks.",
            "content_type": "paper",
            "display_topic": "具身智能",
            "source_tier": "research",
            "evidence_quality": 0.72,
            "information_density": 0.7,
            "facts": {
                "who": "GazeLNN",
                "target": "computationally lightweight scanpath prediction model",
                "evidence": ["Reports 0.47 scanpath prediction metric"],
            },
        }
        layers = {"must_read": [], "physical_ai": [], "watch": [], "featured_papers": [], "paper_appendix": [paper], "brief": []}
        decorated = generator._decorate_layers(layers)
        html = generator.generate_html(
            papers=[paper],
            updates=[],
            mixed_items=[paper],
            report_summary={"lead_summary": "论文附录需要保留技术介绍。", "hot_topics": ["具身智能"]},
            trend_summary={"items": []},
            alert_summary={"needs_alert": False, "issues": []},
            archive_summary={"entries": []},
            layered_updates=layers,
        )

        self.assertIn("paper_technical_intro", decorated["paper_appendix"][0])
        self.assertIn("扫描路径", decorated["paper_appendix"][0]["paper_technical_intro"])
        self.assertIn("论文附录", html)
        self.assertIn("技术介绍：", html)
        self.assertIn("液态神经网络", html)
        self.assertIn("paper-appendix-row", html)
        self.assertIn("paper-appendix-title", html)
        self.assertIn("paper-appendix-tech", html)
        self.assertIn(".paper-appendix-main { display: block; }", html)

    def test_paper_description_varies_by_paper_story(self):
        generator = ReportGenerator()
        papers = [
            {
                "title_cn": "VLA模型提升机器人操作泛化",
                "summary_preview": "这项研究把视觉、语言和动作放进同一个策略里，目标是让机器人按自然语言完成更多操作任务。",
                "content_type": "paper",
                "display_topic": "具身智能",
                "facts": {"evidence": ["真实任务成功率提升", "跨物体评测"]},
            },
            {
                "title_cn": "世界模型改善长时导航规划",
                "summary_preview": "这项研究让模型先预测未来场景，再把预测结果交给规划模块，减少导航中的试错成本。",
                "content_type": "paper",
                "display_topic": "世界模型",
                "facts": {"evidence": ["长时预测基准", "导航任务对比"]},
            },
            {
                "title_cn": "四足机器人控制策略提升复杂地形稳定性",
                "summary_preview": "这项研究关注腿式机器人在复杂地形上的全身控制，让机器人在坡面和障碍物附近保持动作稳定。",
                "content_type": "paper",
                "display_topic": "机器人",
                "facts": {"evidence": ["复杂地形测试", "长时间运行稳定性"]},
            },
        ]
        summaries = [generator._decorate_items([paper])[0]["paper_description"] for paper in papers]

        self.assertTrue(any("新物体" in summary for summary in summaries))
        self.assertTrue(any("规划" in summary and "试错成本" in summary for summary in summaries))
        self.assertTrue(any("地形变化" in summary or "长时间运行" in summary for summary in summaries))
        self.assertLess(sum("简单说" in summary for summary in summaries), 2)

    def test_paper_description_rewrites_english_heavy_paper_into_chinese_story(self):
        generator = ReportGenerator()
        item = {
            "title_cn": "Humanoid-GPT更新AI研究方法与实验结果",
            "summary_preview": "关键看它能否把感知、规划和动作闭环做得更稳。",
            "summary": "论文来源这项研究围绕GPT-style Transformer with causal attention for whole-body control提出了新的方法或实验结果。材料显示，Pre-trained on a 2B-frame retargeted corpus unifying all major mocap datasets with large-scale in-house recordings。它主要影响robotics researchers, humanoid robot developers，关键是结果能否被复现并转化为更低试错成本或更稳定的真实任务表现。",
            "content": "We introduce Humanoid-GPT, a GPT-style Transformer with causal attention trained on a billion-scale motion corpus for whole-body control. Scaling both data and model capacity yields a single generative Transformer that tracks highly dynamic behaviors while achieving unprecedented zero-shot generalization to unseen motions and control tasks.",
            "content_type": "paper",
            "display_topic": "具身智能",
            "facts": {
                "who": "Humanoid-GPT",
                "action": "提出",
                "target": "GPT-style Transformer with causal attention for whole-body control",
                "audience": "robotics researchers, humanoid robot developers",
                "evidence": [
                    "Pre-trained on a 2B-frame retargeted corpus unifying all major mocap datasets with large-scale in-house recordings",
                    "Achieves zero-shot generalization to unseen motions and control tasks",
                    "Establishes a new performance frontier in motion tracking",
                ],
            },
        }

        summary = generator._decorate_items([item])[0]["paper_description"]

        self.assertIn("Humanoid-GPT", summary)
        self.assertIn("全身控制", summary)
        self.assertIn("2B帧", summary)
        self.assertIn("零样本泛化", summary)
        self.assertIn("论文价值", summary)
        self.assertNotIn("核心看", summary)
        self.assertLessEqual(len(summary), 190)
        self.assertNotIn("论文来源", summary)
        self.assertNotRegex(summary, r"[A-Za-z][A-Za-z0-9./_-]*(?:\s+[A-Za-z][A-Za-z0-9./_-]*){4,}")

    def test_recent_real_paper_samples_keep_method_result_meaning_description(self):
        generator = ReportGenerator()
        recent_real_samples = [
            {
                "title_cn": "AHEAD提出一种预测-执行包装器",
                "summary_preview": "AHEAD提出一种预测-执行包装器，通过运动感知潜在世界模型增强冻结VLA。",
                "summary": "AHEAD提出一种预测-执行包装器，通过运动感知潜在世界模型增强冻结的VLA。该方法仅增加4.9M参数至7B OpenVLA，在20个动态仿真场景中达到79%-97%成功率，而最强基线仅31%-58%。在物理UFactory xArm 7上，传送带和滚球任务成功29/30至30/30，桨拦截23/30，抛射捕捉19/30，基线均为0/30。",
                "content_type": "paper",
                "display_topic": "具身智能",
                "facts": {
                    "who": "AHEAD",
                    "action": "提出",
                    "target": "a predict-then-act wrapper that augments a frozen VLA with a motion-aware latent world model",
                    "evidence": [
                        "Adds 4.9M parameters to a frozen 7B OpenVLA",
                        "Reaches 79 to 97% success across 20 dynamic simulation scenarios where the strongest baseline reaches 31 to 58%",
                        "On a physical UFactory xArm 7, succeeds on 29/30 to 30/30 on three conveyor and rolling-ball tasks",
                    ],
                },
            },
            {
                "title_cn": "U4D提出不确定性感知4D LiDAR场景合成",
                "summary_preview": "不确定性引导的4D LiDAR场景合成降低试错成本。",
                "summary": "U4D提出了一种uncertainty-aware 4D LiDAR场景合成框架，利用预训练分割器的香农熵生成逐点不确定性图，对高熵区域采用无条件扩散，其余区域进行条件补全，并引入MoST模块增强跨帧一致性。",
                "content_type": "paper",
                "display_topic": "世界模型",
                "facts": {
                    "who": "U4D",
                    "action": "提出",
                    "target": "uncertainty-aware 4D LiDAR scene synthesis framework",
                    "evidence": [
                        "Uses Shannon Entropy from pretrained segmentor to derive per-point uncertainty maps",
                        "Employs unconditional diffusion for high-entropy areas followed by conditional completion for remaining regions",
                        "Introduces MoST block for cross-frame coherence",
                    ],
                },
            },
            {
                "title_cn": "COMAP框架提出共同演化文本世界模型与智能体策略",
                "summary_preview": "COMAP框架提出共同演化文本世界模型与智能体策略。",
                "summary": "COMAP框架提出共同演化文本世界模型与智能体策略，在具身任务规划、网页导航和工具使用基准测试中，基于Qwen3-4B实现相对基线16.75%的提升。",
                "content_type": "paper",
                "display_topic": "世界模型",
                "facts": {
                    "who": "COMAP framework",
                    "action": "提出",
                    "target": "co-evolving textual world models and agent policies",
                    "evidence": [
                        "Outperforms baselines by +16.75% relative improvement with Qwen3-4B",
                        "Tested on embodied task planning, Web navigation, and tool-use benchmarks",
                        "Code available at https://github.com/loyiv/CoMAP",
                    ],
                },
            },
        ]

        summaries = [generator._decorate_items([sample])[0]["paper_description"] for sample in recent_real_samples]

        for summary in summaries:
            self.assertTrue(generator._paper_description_is_substantive(summary), summary)
            self.assertNotIn("白话总结", summary)
            self.assertNotIn("关键看它能否", summary)
            self.assertNotIn("论文来源", summary)
            self.assertLessEqual(len(summary), 190)
            self.assertLessEqual(len([part for part in summary.split("。") if part.strip()]), 3)
            self.assertNotRegex(summary, r"[A-Za-z][A-Za-z0-9./_-]*(?:\s+[A-Za-z][A-Za-z0-9./_-]*){4,}")
        self.assertTrue(any("成功率" in summary for summary in summaries))
        self.assertTrue(any("LiDAR" in summary for summary in summaries))
        self.assertTrue(any("16.75%" in summary for summary in summaries))

    def test_v4_suspicious_high_stakes_claims_are_not_promoted(self):
        dubious = {
            "id": 501,
            "url": "https://unknown.example.com/spacex-cursor",
            "title": "SpaceX buys Cursor for $60 billion after IPO",
            "title_cn": "SpaceX以600亿美元收购Cursor",
            "summary": "SpaceX以600亿美元股票收购Cursor，消息涉及IPO和并购。",
            "summary_preview": "一条高风险并购传闻。",
            "content_type": "news",
            "source_detail": "Unknown Blog",
            "platform": "Web",
            "score": 9.8,
            "evidence_quality": 0.9,
            "information_density": 0.9,
            "facts": {
                "who": "SpaceX",
                "action": "收购",
                "target": "Cursor",
                "evidence": ["$60 billion", "IPO"],
                "audience": "AI工具和航天产业观察者",
            },
        }
        trusted = {
            **dubious,
            "id": 502,
            "url": "https://techcrunch.com/example/respond-funding",
            "title": "Respond.io raises $62 million",
            "title_cn": "Respond.io完成6200万美元融资",
            "summary": "Respond.io完成6200万美元融资，TechCrunch披露融资金额和公司主体。",
            "source_detail": "TechCrunch",
            "facts": {
                "who": "Respond.io",
                "action": "融资",
                "target": "6200万美元融资",
                "evidence": ["$62 million", "TechCrunch"],
                "audience": "企业客服和AI销售自动化团队",
            },
        }

        layers = classify_report_layers(
            papers=[],
            updates=[dubious, trusted],
            report_id="report-v4",
            feedback_config={"enabled": False},
            report_config={},
        )
        must_read_urls = {item["url"] for item in layers["must_read"]}
        brief_urls = {item["url"] for item in layers["brief"]}

        self.assertNotIn(dubious["url"], must_read_urls)
        self.assertIn(dubious["url"], brief_urls)
        self.assertIn(trusted["url"], must_read_urls)
        self.assertIn("suspicious_claim", layers["brief"][0]["quality_flags"])

    def test_v4_quality_gate_reports_suspicious_claim_and_memory_fields(self):
        suspicious = {
            "url": "https://unknown.example.com/acquisition",
            "title": "OpenAI buys a robotics startup for $10 billion",
            "title_cn": "OpenAI以100亿美元收购机器人公司",
            "summary": "OpenAI以100亿美元收购机器人公司。",
            "content_type": "news",
            "source_detail": "Unknown Blog",
            "platform": "Web",
            "report_section": "must_read",
            "score": 9.0,
            "evidence_quality": 0.9,
            "information_density": 0.9,
            "facts": {
                "who": "OpenAI",
                "action": "收购",
                "target": "机器人公司",
                "evidence": ["$10 billion"],
                "audience": "机器人行业团队",
            },
        }

        result = evaluate_report_quality({"must_read": [suspicious], "physical_ai": [], "watch": [], "featured_papers": [], "paper_appendix": [], "brief": []})

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["v4_quality_status"], "failed")
        self.assertEqual(result["suspicious_claim_count"], 1)
        self.assertEqual(result["memory_card_count"], 1)
        self.assertIn(suspicious["url"], result["failed_item_urls"])

    def test_v4_paper_description_does_not_emit_numeric_placeholder(self):
        generator = ReportGenerator()
        item = {
            "title_cn": "NewBench提出机器人操作评测",
            "summary": "NewBench proposes a benchmark for robot manipulation.",
            "content_type": "paper",
            "display_topic": "机器人",
            "facts": {
                "who": "NewBench",
                "action": "提出",
                "target": "benchmark for robot manipulation",
                "evidence": ["1"],
                "audience": "机器人研究者",
            },
        }

        description = generator._decorate_items([item])[0]["paper_description"]

        self.assertNotIn("等量化证据", description)
        self.assertNotIn("围绕论文中的核心任务", description)
        self.assertIn("量化结果", description)

    def test_v4_html_uses_learning_memory_labels(self):
        generator = ReportGenerator(design_version="v6-editorial-learning")
        item = {
            "id": 1,
            "url": "https://openai.com/blog/workflow-agent",
            "title_cn": "OpenAI发布企业工作流Agent",
            "summary_preview": "企业任务链开始被模型接管更多执行环节。",
            "summary": "OpenAI发布企业工作流Agent，证据来自官方博客，影响需要自动化长任务链的企业团队。",
            "content_type": "news",
            "source_tier": "official",
            "display_topic": "产品发布",
            "score": 9.0,
            "evidence_quality": 0.8,
            "information_density": 0.8,
            "facts": {
                "who": "OpenAI",
                "action": "发布",
                "target": "企业工作流Agent",
                "evidence": ["官方博客"],
                "audience": "企业团队",
            },
        }
        html = generator.generate_html(
            papers=[],
            updates=[item],
            mixed_items=[item],
            report_summary={"lead_summary": "企业工作流Agent成为今天最值得记住的变化。", "hot_topics": ["产品发布"]},
            trend_summary={"items": []},
            alert_summary={"needs_alert": False, "issues": []},
            archive_summary={"entries": []},
            layered_updates={"must_read": [item], "physical_ai": [], "watch": [], "featured_papers": [], "paper_appendix": [], "brief": []},
        )

        self.assertIn("今日编辑判断", html)
        self.assertIn("领域技术摘编", html)
        self.assertIn("证据：", html)
        self.assertIn("继续看：", html)
        self.assertNotIn("一句话结论", html)
        self.assertNotIn("技术 / 背景解释", html)
        self.assertNotIn("技术方向摘编", html)
        self.assertIn('data-design-version="v6-editorial-learning"', html)
        self.assertIn("v6-editorial-learning", html)


if __name__ == "__main__":
    unittest.main()
