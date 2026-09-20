import unittest
from datetime import datetime, timezone

from main import (
    compact_email_html,
    effective_fresh_paper_minimum,
    evaluate_report_quality,
    evaluate_paper_domain_quotas,
    enforce_final_visible_paper_overlap,
    finalize_paper_freshness_metrics,
    filter_recently_sent_papers,
    item_quality_flags,
    repair_v10_paper_title,
    repair_v10_paper_titles_in_layers,
    should_preserve_existing_paper_analysis,
    select_papers_by_domain_quota,
    select_papers_with_strict_freshness,
    paper_quota_domain,
    prepare_v11_email_delivery_volumes,
    scan_final_html_quality,
    title_fact_mismatch,
    title_looks_bad,
    v10_paper_is_ai_relevant,
    v10_has_concrete_news_evidence,
)
from src.collectors.arxiv_collector import ArxivCollector
from src.editorial_engine import enrich_editorial_fields, mixed_language_title, paper_plain_summary_passes
from src.generator import ReportGenerator


def rich_paper(article_id: int, *, section: str = "featured_papers"):
    return {
        "id": article_id,
        "title": "Predict-then-act world models for dynamic robot manipulation",
        "title_cn": "预测后行动的动态机器人世界模型",
        "url": f"https://arxiv.org/abs/{article_id}",
        "source_detail": "arXiv",
        "source_tier": "research",
        "content_type": "paper",
        "report_section": section,
        "domain_key": "world_model",
        "evidence_quality": 0.86,
        "information_density": 0.84,
        "facts": {
            "claim_type": "research_result",
            "who": "AheadWM",
            "action": "提出",
            "target": "动态场景中的机器人操作",
            "research_problem": "机器人面对持续移动的物体时，动作规划容易使用已经过时的视觉状态",
            "core_method": "先预测物体下一时刻的位置，再把预测状态交给冻结的 VLA 策略选择动作",
            "architecture": "轻量世界模型与冻结 VLA 串联",
            "training_objective": "以未来状态预测误差训练前置预测模块",
            "input_output": "输入当前图像和语言指令，输出未来状态与机器人动作",
            "method": "先预测物体下一时刻的位置，再把预测状态交给冻结的 VLA 策略选择动作",
            "dataset_or_benchmark": "Dynamic Manipulation Benchmark",
            "metric_result": "平均任务成功率达到 82%，比冻结 VLA 基线高 11 个百分点",
            "baseline": "冻结 VLA",
            "limitation": "尚未覆盖长时间遮挡和多机器人协同",
            "evidence": [
                "平均任务成功率达到 82%",
                "比冻结 VLA 基线高 11 个百分点",
            ],
        },
    }


def rich_news(article_id: int):
    return {
        "id": article_id,
        "title": "OpenAI ships workflow permissions for enterprise agents",
        "title_cn": "OpenAI为企业智能体增加工作流权限控制",
        "url": f"https://openai.com/news/{article_id}",
        "source_detail": "OpenAI",
        "source_tier": "official",
        "content_type": "news",
        "report_section": "must_read",
        "domain_key": "agent_models",
        "evidence_quality": 0.82,
        "information_density": 0.78,
        "facts": {
            "claim_type": "official_claim",
            "who": "OpenAI",
            "action": "发布",
            "target": "企业智能体的工作流权限控制",
            "method": "把工具调用权限、审批节点和运行记录接入同一工作流",
            "deployment_context": "企业内部自动化流程",
            "evidence": ["官方文档列出了权限配置和审批记录能力"],
        },
    }


def arxiv_list_html(paper_id: str = "2608.01234"):
    return f'''<dt><a href="/abs/{paper_id}">arXiv:{paper_id}</a></dt>
    <dd><div class="list-title mathjax"><span class="descriptor">Title:</span>
    A useful robot world model</div><div class='list-authors'>Alice</div></dd>'''


class V10ReportTests(unittest.TestCase):
    def test_v11_card_shows_publish_date_and_marks_older_supplemental_sources(self):
        generator = ReportGenerator(
            design_version="v10-learning-digest",
            report_config={"news_body_char_limit": 320},
        )
        item = rich_news(6999)
        item.update({
            "publish_date": "2026-09-12T08:30:00+00:00",
            "quality_flags": ["supplemental_older_source"],
            "summary": "这是一段可以直接阅读的新闻整理正文。" * 12,
            "title_cn": "OpenAI在企业智能体工作流中加入审批权限、工具白名单与完整运行记录",
            "model_used": "codex-automation",
            "claim_type": "official_claim",
        })

        card = generator._v8_card(item)

        self.assertEqual(card["v11_date_label"], "补充阅读 · 2026-09-12")
        self.assertEqual(card["v11_claim_label"], "发布方声明")
        self.assertEqual(card["title_cn"], item["title_cn"])

    def test_v12_card_preserves_curated_body_and_avoids_duplicate_deck(self):
        generator = ReportGenerator(
            design_version="v10-learning-digest",
            report_config={"news_body_char_limit": 300},
        )
        item = rich_news(7001)
        body = "第一句直接交代具体变化。第二句保留实现机制、证据位置和限制条件，不在模板阶段重写。"
        item.update({
            "model_used": "codex-automation",
            "analysis_body": body,
            "summary": body,
            "summary_preview": "第一句直接交代具体变化",
        })

        card = generator._v8_card(item)

        self.assertEqual(card["v11_body"], body)
        self.assertEqual(card["v12_deck"], "")

    def test_v12_card_preserves_long_validated_codex_title(self):
        generator = ReportGenerator(
            design_version="v10-learning-digest",
            report_config={"news_body_char_limit": 300},
        )
        title = (
            "Google DeepMind 公开机器人策略训练流水线，说明视觉编码、动作分块、"
            "离线数据混合和真实环境评测之间的具体连接方式，以及跨硬件部署时采用的约束"
        )
        item = rich_news(7004)
        item.update({
            "model_used": "codex-automation",
            "title_cn": title,
            "editorial_title": title,
            "analysis_body": "团队解释训练流水线中的模块连接、数据来源和真实环境评测边界。",
        })

        card = generator._v8_card(item)

        self.assertGreater(len(title), 72)
        self.assertEqual(card["title_cn"], title)

    def test_v12_card_hides_deck_that_repeats_the_title(self):
        generator = ReportGenerator(
            design_version="v10-learning-digest",
            report_config={"news_body_char_limit": 300},
        )
        item = rich_news(7002)
        item.update({
            "model_used": "codex-automation",
            "title_cn": "Google 用提交前代理审查补充基础设施安全检查",
            "summary_preview": "Google 用提交前代理审查补充基础设施安全检查",
            "analysis_body": "Google 介绍了在代码提交前运行多角色代理审查的具体流程和边界。",
        })

        card = generator._v8_card(item)

        self.assertEqual(card["v12_deck"], "")

    def test_v12_card_never_rewrites_curated_title_after_validation(self):
        generator = ReportGenerator(
            design_version="v10-learning-digest",
            report_config={"news_body_char_limit": 300},
        )
        item = rich_news(7003)
        item.update({
            "model_used": "codex-automation",
            "title_cn": "霍夫曼与帕蒂尔复盘创作者资助计划：增加，也保留人的判断",
            "editorial_title": "霍夫曼与帕蒂尔复盘创作者资助计划：增加，也保留人的判断",
            "summary_preview": "霍夫曼与帕蒂尔复盘创作者资助计划：增加尝试，也保留人的判断",
            "analysis_body": "两位嘉宾复盘创作者资助计划，并解释为什么仍需保留人工判断。",
        })

        card = generator._v8_card(item)

        self.assertEqual(card["title_cn"], item["title_cn"])
        self.assertEqual(card["v12_deck"], "")

    def test_v11_context_keeps_news_and_non_paper_technical_sections_independent(self):
        generator = ReportGenerator(
            design_version="v10-learning-digest",
            report_config={
                "news_section_limit": 24,
                "technical_section_limit": 24,
                "news_body_char_limit": 320,
                "technical_body_char_limit": 400,
            },
        )
        news = []
        technical = []
        for index in range(20):
            item = rich_news(7000 + index)
            item["summary"] = "这是一条经过原文核验的新闻正文。" * 12
            item["analysis_body"] = item["summary"]
            item["quality_tier"] = "focus"
            item["facts"]["primary_section"] = "news"
            news.append(item)

            tech = rich_news(8000 + index)
            tech["content_type"] = "project"
            tech["summary"] = "这项非论文技术解释了系统模块、输入输出、部署约束和可复现实验。" * 10
            tech["analysis_body"] = tech["summary"]
            tech["quality_tier"] = "focus"
            tech["facts"]["primary_section"] = "technical"
            technical.append(tech)

        context = generator._v10_reader_context(
            {
                "must_read": news[:8],
                "physical_ai": [],
                "watch": news[8:] + technical,
                "featured_papers": [],
                "paper_appendix": [],
                "brief": [],
            }
        )

        self.assertEqual(context["news_count"], 20)
        self.assertEqual(context["technical_count"], 20)
        self.assertEqual(context["news_items"][0]["v11_position"], 1)
        self.assertEqual(context["news_items"][-1]["v11_position"], 20)
        self.assertEqual(context["news_items"][-1]["v11_total"], 20)
        self.assertEqual(context["technical_items"][-1]["v11_position"], 20)
        self.assertEqual(context["technical_items"][-1]["v11_total"], 20)
        self.assertFalse(
            {item["url"] for item in context["news_items"]}
            & {item["url"] for item in context["technical_items"]}
        )
        self.assertGreater(len(context["news_items"][0]["v11_body"]), 180)

    def test_v11_html_renders_twenty_news_twenty_technical_and_fifteen_papers(self):
        generator = ReportGenerator(
            design_version="v10-learning-digest",
            report_config={
                "product_mode": "intelligence_v11_editorial_library",
                "news_section_limit": 24,
                "technical_section_limit": 24,
                "paper_featured_limit": 10,
                "paper_appendix_limit": 15,
                "editorial_decision_limit": 6,
            },
        )
        news = []
        technical = []
        for index in range(20):
            item = enrich_editorial_fields(rich_news(9000 + index))
            item["facts"]["primary_section"] = "news"
            news.append(item)

            tech = rich_news(9100 + index)
            tech["content_type"] = "project"
            tech["facts"]["primary_section"] = "technical"
            technical.append(enrich_editorial_fields(tech))
        papers = [enrich_editorial_fields(rich_paper(9200 + index)) for index in range(15)]
        layers = {
            "must_read": news[:8],
            "physical_ai": [],
            "watch": news[8:] + technical,
            "featured_papers": papers[:10],
            "paper_appendix": papers[10:],
            "brief": [],
        }

        decorated = generator._decorate_layers(layers)
        context = generator._v10_reader_context(decorated)
        html = generator.generate_html(
            papers=papers,
            updates=news + technical,
            mixed_items=news + technical + papers,
            report_summary={},
            layered_updates=layers,
        )

        self.assertEqual(context["news_count"], 20)
        self.assertEqual(context["technical_count"], 20)
        self.assertEqual(context["paper_count"], 15)
        self.assertEqual(len(context["editorial_decisions"]), 6)
        self.assertEqual(
            len({item["source_identity"] for item in context["editorial_decisions"]}),
            6,
        )
        self.assertTrue(
            all(item["source_identity"].startswith("url:") for item in context["editorial_decisions"])
        )
        self.assertIn("新闻、博客与访谈", html)
        self.assertIn("AI 前沿情报日报", html)
        self.assertIn("世界模型", html)
        self.assertIn("技术方法、方向与架构", html)
        self.assertIn("论文精读", html)
        self.assertIn("20 / 20", html)
        self.assertIn("15 / 15", html)
        self.assertEqual(html.count('class="v10-paper-plain"') + html.count('class="v10-more-plain"'), 15)
        self.assertEqual(html.count('class="v10-paper-tech"') + html.count('class="v10-more-tech"'), 15)
        self.assertIn('name="color-scheme" content="light dark"', html)
        self.assertIn("@media (prefers-color-scheme: dark)", html)
        self.assertNotIn("<img", html.lower())
        self.assertEqual(html.count('class="v10-entry"'), 50)
        self.assertEqual(html.count('class="v10-decision"'), 6)
        self.assertEqual(html.count('data-source-key="'), 6)
        self.assertEqual(html.count('class="v11-claim-label"'), 55)
        self.assertEqual(html.count('data-v11-item-key="'), 55)
        self.assertGreaterEqual(html.count(' style="'), 55 * 3 + 10)
        self.assertIn('class="v10-body" style="', html)
        self.assertIn('class="v10-paper-tech" style="', html)
        self.assertGreater(len(compact_email_html(html).encode("utf-8")), 72000)

        def render_volume(index, volume):
            volume_items = volume["items"]
            return generator.generate_html(
                papers=[item for item in volume_items if item.get("content_type") == "paper"],
                updates=[item for item in volume_items if item.get("content_type") != "paper"],
                mixed_items=volume_items,
                report_summary={},
                layered_updates=volume["layers"],
                title=f"V11 第 {index} 卷",
            )

        volumes, split_applied = prepare_v11_email_delivery_volumes(
            compact_email_html(html),
            decorated,
            {
                "product_mode": "intelligence_v11_editorial_library",
                "email_split_enabled": True,
                "email_html_max_bytes": 80000,
            },
            render_volume,
        )
        self.assertTrue(split_applied)
        self.assertEqual(len(volumes), 3)
        self.assertTrue(all(volume["size_bytes"] <= 80000 for volume in volumes))
        self.assertTrue(
            all(
                5 <= volume["html"].count('class="v10-decision"') <= 7
                for volume in volumes
            )
        )
        expected_anchors = {
            "news": "news-and-voices",
            "technical": "technical-trends",
            "paper": "paper-deep-reads",
        }
        for volume in volumes:
            live_anchors = {
                anchor
                for anchor in expected_anchors.values()
                if f'href="#{anchor}"' in volume["html"]
            }
            self.assertEqual(live_anchors, {expected_anchors[volume["primary_section"]]})
            self.assertEqual(
                volume["html"].count('class="v11-claim-label"'),
                volume["item_count"],
            )
            self.assertEqual(
                volume["html"].count('data-source-key="'),
                volume["html"].count('class="v10-decision"'),
            )
            fidelity_metrics = scan_final_html_quality(
                volume["html"],
                volume["layers"],
                quality_config={},
                report_config={
                    "product_mode": "intelligence_v11_editorial_library",
                    "design_version": "v11-editorial-library",
                    "min_visible_news_count": 0,
                    "min_visible_technical_count": 0,
                    "min_visible_paper_count": 0,
                    "paper_technical_intro_min_count": 0,
                    "total_visible_chars_min": 0,
                    "total_visible_chars_max": 100000,
                },
            )
            self.assertEqual(
                fidelity_metrics["v11_content_fidelity_missing_count"],
                0,
                fidelity_metrics["v11_content_fidelity_missing_examples"],
            )

    def test_v11_split_volume_shows_edition_totals_and_only_live_section_links(self):
        generator = ReportGenerator(
            design_version="v10-learning-digest",
            report_config={"product_mode": "intelligence_v11_editorial_library"},
        )
        news = enrich_editorial_fields(rich_news(9991))
        news["facts"]["primary_section"] = "news"
        html = generator.generate_html(
            papers=[],
            updates=[news],
            mixed_items=[news],
            report_summary={"edition_counts": {"news": 20, "technical": 20, "paper": 15}},
            layered_updates={
                "must_read": [news],
                "physical_ai": [],
                "watch": [],
                "featured_papers": [],
                "paper_appendix": [],
                "brief": [],
            },
            title="AI Frontier Intelligence Daily · 第 1 卷：新闻、博客与访谈",
        )

        self.assertIn("AI 前沿情报日报", html)
        self.assertIn("第 1 卷：新闻、博客与访谈", html)
        self.assertIn("AI 前沿日报：20 条新闻与观点、20 条技术内容、15 篇论文", html)
        self.assertIn("本期共 55 条独立内容", html)
        self.assertIn('href="#news-and-voices"', html)
        self.assertNotIn('href="#technical-trends"', html)
        self.assertNotIn('href="#paper-deep-reads"', html)

    def test_v11_html_escapes_technical_symbols_without_losing_approved_copy(self):
        generator = ReportGenerator(
            design_version="v11-editorial-library",
            report_config={"product_mode": "intelligence_v11_editorial_library"},
        )
        item = rich_news(9992)
        item.update(
            {
                "title_cn": "A&B 团队公开上下文限制",
                "title": "A&B 团队公开上下文限制",
                "analysis_body": "运行时把单次输入限制为 <8K token，并要求缓存命中率 >90%。",
                "summary": "运行时把单次输入限制为 <8K token，并要求缓存命中率 >90%。",
            }
        )
        item["facts"]["primary_section"] = "news"
        item = enrich_editorial_fields(item)
        layers = {
            "must_read": [item],
            "physical_ai": [],
            "watch": [],
            "featured_papers": [],
            "paper_appendix": [],
            "brief": [],
        }

        html = generator.generate_html(
            papers=[],
            updates=[item],
            mixed_items=[item],
            report_summary={"edition_counts": {"news": 1, "technical": 0, "paper": 0}},
            layered_updates=layers,
        )

        self.assertIn("A&amp;B", html)
        self.assertIn("&lt;8K token", html)
        self.assertIn("&gt;90%", html)
        self.assertNotIn("<8K token", html)
        metrics = scan_final_html_quality(
            html,
            layers,
            quality_config={},
            report_config={
                "product_mode": "intelligence_v11_editorial_library",
                "design_version": "v11-editorial-library",
                "min_visible_news_count": 0,
                "min_visible_technical_count": 0,
                "min_visible_paper_count": 0,
                "paper_technical_intro_min_count": 0,
                "total_visible_chars_min": 0,
                "total_visible_chars_max": 100000,
            },
        )
        self.assertEqual(metrics["v11_content_fidelity_missing_count"], 0)

    def test_v11_curated_paper_keeps_approved_paragraph_breaks(self):
        generator = ReportGenerator(
            design_version="v11-editorial-library",
            report_config={"product_mode": "intelligence_v11_editorial_library"},
        )
        paper = enrich_editorial_fields(rich_paper(9993))
        paper["model_used"] = "codex-automation"
        plain = "第一段先说明机器人为什么会依据过时画面做动作。\n第二段说明作者先预测未来状态，再交给动作模型。"
        technical = "方法把状态预测器放在动作模型之前。\n实验与直接动作预测基线比较，并报告动态任务成功率。"
        paper["paper_plain_summary"] = plain
        paper["paper_technical_intro"] = technical

        card = generator._v8_card(paper)

        self.assertEqual(card["v10_plain_summary"], plain)
        self.assertEqual(card["v10_technical_intro"], technical)
        self.assertIn("white-space:pre-line", generator.V11_INLINE_CLASS_STYLES["v10-paper-plain"])
        self.assertIn("white-space:pre-line", generator.V11_INLINE_CLASS_STYLES["v10-paper-tech"])
        self.assertIn("font-size:16px", generator.V11_INLINE_CLASS_STYLES["v10-paper-tech"])
        self.assertIn("font-size:16px", generator.V11_INLINE_CLASS_STYLES["v10-more-title"])
        self.assertIn("font-size:16px", generator.V11_INLINE_CLASS_STYLES["v10-more-plain"])
        self.assertIn("font-size:16px", generator.V11_INLINE_CLASS_STYLES["v10-more-tech"])

    def test_template_fallback_does_not_replace_displayable_paper_analysis(self):
        paper = enrich_editorial_fields(rich_paper(2999))

        self.assertTrue(should_preserve_existing_paper_analysis(
            paper,
            {"model_used": "template_fallback", "facts": {"target": "机器人"}},
        ))
        self.assertFalse(should_preserve_existing_paper_analysis(
            paper,
            {"model_used": "deepseek-chat", "facts": {"target": "机器人"}},
        ))

    def test_paper_title_repair_uses_specific_topic_when_facts_are_generic(self):
        repaired = repair_v10_paper_title({
            "content_type": "paper",
            "title": "Cooperative Risk-Aware Multi-Robot Planning in Dynamic Scenes",
            "title_cn": "Cooperative Risk-Aware Multi-Robot Planning in Dynamic Scenes",
            "domain_key": "physical_ai",
            "facts": {"who": "研究团队", "target": "机器人"},
        })

        self.assertIn("多机器人协作方法与实验", repaired["title_cn"])
        self.assertFalse(title_looks_bad(repaired))

    def test_paper_title_repair_preserves_verified_codex_title(self):
        item = rich_paper(2998)
        item.update({
            "title_cn": "AHEAD先预测动态物体位置，再让冻结VLA选择动作",
            "model_used": "codex-automation",
        })

        repaired = repair_v10_paper_title(item)

        self.assertEqual(repaired["title_cn"], item["title_cn"])

    def test_paper_title_repair_replaces_generic_experiment_title(self):
        repaired = repair_v10_paper_title({
            "content_type": "paper",
            "title": "DexPolicy: Learning Dexterous Manipulation from Demonstrations",
            "title_cn": "具身智能的新方法与实验",
            "domain_key": "physical_ai",
            "facts": {"who": "研究团队", "target": "具身智能"},
        })

        self.assertEqual(repaired["title_cn"], "DexPolicy：机器人操作与抓取方法")

    def test_paper_title_repair_replaces_repeated_model_name_target(self):
        repaired = repair_v10_paper_title({
            "content_type": "paper",
            "title": "SLIM-0.5B: Efficient Vision-Language-Action Policies",
            "title_cn": "SLIM-0.5B：SLIM-0",
            "domain_key": "physical_ai",
            "facts": {
                "who": "SLIM-0.5B",
                "target": "SLIM-0",
            },
        })

        self.assertEqual(repaired["title_cn"], "SLIM-0.5B：视觉语言动作模型训练与评测")

    def test_paper_title_repair_replaces_repeated_domain_title(self):
        repaired = repair_v10_paper_title({
            "content_type": "paper",
            "title": "Beyond Data Scaling: Continued Pre-training for Vision-Language-Action Models",
            "title_cn": "具身智能具身智能",
            "domain_key": "physical_ai",
            "facts": {
                "who": "要解决的是具身智能 核心动作是开源",
                "target": "具身智能",
            },
        })

        self.assertEqual(repaired["title_cn"], "Beyond Data Scaling：视觉语言动作模型继续预训练方法")

        layered = repair_v10_paper_titles_in_layers({"paper_appendix": [repaired]})
        self.assertEqual(
            layered["paper_appendix"][0]["title_cn"],
            "视觉语言动作模型继续预训练方法",
        )

    def test_layer_title_repair_keeps_entity_after_editorial_enrichment(self):
        paper = {
            "content_type": "paper",
            "title": "JEPA-WAM: Learning Vision-Language-Action Policies with Joint-Embedding World Modeling",
            "title_cn": "模型，一种在预训练空间中构建的潜在世界动作模型",
            "domain_key": "world_model",
            "facts": {
                "who": "JEPA-WAM",
                "action": "提出",
                "target": "在预训练V-JEPA空间中构建的潜在世界动作模型",
                "method": "联合训练潜在状态转换预测器与动作生成器",
                "metric_result": "在LIBERO-Plus上达到79.2%",
                "evidence": ["在LIBERO-Plus上达到79.2%"],
            },
            "evidence_quality": 0.9,
            "information_density": 0.9,
        }

        layered = repair_v10_paper_titles_in_layers({"featured_papers": [paper]})
        repaired = layered["featured_papers"][0]

        self.assertTrue(repaired["title_cn"].startswith("JEPA-WAM："))
        self.assertFalse(title_fact_mismatch(repaired, repaired["facts"]))

    def test_world_model_term_without_ai_context_is_not_relevant(self):
        self.assertFalse(v10_paper_is_ai_relevant({
            "topic": "World Model",
            "title": "A classical world model for quantum qutrit alignment",
            "content": "A mathematical construction for exact quantum states.",
        }))
        self.assertTrue(v10_paper_is_ai_relevant({
            "topic": "World Model",
            "title": "A latent world model for robot planning",
            "content": "The policy predicts future robot states before control.",
        }))
        self.assertTrue(v10_paper_is_ai_relevant({
            "topic": "World Model",
            "title": "交互式世界模型压缩到单卡流式运行",
            "content": "模型预测后续视频状态，并把结果用于机器人策略规划。",
        }))

    def test_parse_datetime_normalizes_aware_values_to_utc(self):
        from main import parse_datetime

        self.assertEqual(
            parse_datetime("2026-08-12T21:00:00+08:00"),
            datetime(2026, 8, 12, 13, 0, 0),
        )

    def test_v10_effective_minimum_allows_short_fresh_report_but_rejects_zero(self):
        self.assertEqual(effective_fresh_paper_minimum(10, 7), 7)
        self.assertEqual(effective_fresh_paper_minimum(10, 12), 10)
        self.assertEqual(effective_fresh_paper_minimum(10, 0), 1)

    def test_rich_paper_gets_plain_summary_and_separate_technical_intro(self):
        item = enrich_editorial_fields(rich_paper(1001))

        self.assertTrue(paper_plain_summary_passes(item["paper_plain_summary"]))
        self.assertIn("机器人面对持续移动的物体", item["paper_plain_summary"])
        self.assertIn("82%", item["paper_plain_summary"])
        self.assertLessEqual(len(item["paper_plain_summary"]), 170)
        self.assertGreaterEqual(item["paper_plain_summary"].count("。"), 3)
        self.assertIn("轻量世界模型", item["paper_technical_intro"])
        self.assertIn("训练", item["paper_technical_intro"])
        self.assertNotEqual(item["paper_plain_summary"], item["paper_technical_intro"])

    def test_v10_html_removes_low_value_modules_and_renders_substantive_appendix(self):
        featured = enrich_editorial_fields(rich_paper(1002))
        featured["paper_status_label"] = "代码已发布"
        featured["paper_change_reason"] = "新增代码仓库：https://github.com/example/aheadwm"
        featured["is_reappeared_update"] = True
        appendix = enrich_editorial_fields(rich_paper(1003, section="paper_appendix"))
        news = enrich_editorial_fields(rich_news(1004))
        layers = {
            "must_read": [news],
            "physical_ai": [],
            "watch": [],
            "featured_papers": [featured],
            "paper_appendix": [appendix],
            "brief": [],
        }
        generator = ReportGenerator(
            design_version="v10-learning-digest",
            report_config={
                "design_version": "v10-learning-digest",
                "must_read_limit": 10,
                "domain_item_limit": 4,
                "paper_featured_limit": 12,
                "paper_appendix_limit": 24,
                "paper_freshness_metrics": {
                    "fresh_paper_count": 2,
                    "paper_repeat_filtered_count": 5,
                    "reappeared_paper_with_update_count": 0,
                },
            },
        )

        html = generator.generate_html(
            papers=[featured, appendix],
            updates=[news],
            mixed_items=[news, featured, appendix],
            report_summary={},
            layered_updates=layers,
        )
        context = generator._v10_reader_context(generator._decorate_layers(layers))

        self.assertIn('data-design-version="v10-learning-digest"', html)
        self.assertIn("30 分钟深度编辑版", html)
        self.assertIn("今日编辑判断", html)
        self.assertIn("新闻、博客与访谈", html)
        self.assertIn("论文精读", html)
        self.assertIn("更多论文", html)
        self.assertIn("本期新增论文 2 篇", html)
        self.assertIn("过滤近期及上一封重复 5 篇", html)
        self.assertIn("今日新增", html)
        self.assertIn(featured["paper_plain_summary"][:80], html)
        self.assertIn("代码已发布", html)
        self.assertIn("重要进展更新", html)
        self.assertIn("新增代码仓库：https://github.com/example/aheadwm", html)
        self.assertEqual(html.count("本次变化："), 1)
        self.assertIn(featured["paper_technical_intro"], html)
        self.assertIn(appendix["paper_plain_summary"][:40], html)
        self.assertEqual([item["url"] for item in context["paper_updates"]], [featured["url"]])
        self.assertNotIn(featured["url"], [item["url"] for item in context["featured_papers"]])
        self.assertNotIn(featured["url"], [item["url"] for item in context["more_papers"]])
        self.assertEqual(html.count(featured["url"]), 1)
        for forbidden in (
            "相较上次",
            "可信度：",
            "长期技术档案",
            "技术谱系",
            "相对前序",
            "本期相关论文",
            "论文索引",
            "读完后留下这些",
            "快讯 / 待确认",
        ):
            self.assertNotIn(forbidden, html)

    def test_v10_html_renders_source_grounded_news_section(self):
        news = {
            "id": 1010,
            "url": "https://aws.amazon.com/blogs/security/agentcore-oauth-consent/",
            "title": "Amazon Bedrock AgentCore adds OAuth consent for enterprise agents",
            "content_type": "news",
            "source_detail": "AWS News Blog",
            "source_tier": "official",
            "source_grounded_brief": True,
            "source_display_title": "Amazon Bedrock AgentCore adds OAuth consent for enterprise agents",
            "source_excerpt": (
                "Administrators can require approval before an agent accesses connected applications, "
                "and AgentCore records the authorization decision for later audit."
            ),
            "brief_line": "Administrators can require approval before an agent accesses connected applications.",
            "quality_tier": "brief",
            "report_section": "brief",
        }
        layers = {
            "must_read": [],
            "physical_ai": [],
            "watch": [],
            "featured_papers": [],
            "paper_appendix": [],
            "brief": [news],
        }
        generator = ReportGenerator(
            design_version="v10-learning-digest",
            report_config={
                "design_version": "v10-learning-digest",
                "source_news_brief_limit": 12,
            },
        )

        html = generator.generate_html(
            papers=[],
            updates=[news],
            mixed_items=[news],
            report_summary={},
            layered_updates=layers,
        )

        self.assertIn("新闻、博客与访谈", html)
        self.assertIn("AgentCore", html)
        self.assertIn("AgentCore records the authorization decision", html)
        self.assertIn('id="news-and-voices"', html)

    def test_v10_quality_gate_rejects_featured_paper_without_plain_summary(self):
        weak = rich_paper(1005)
        weak["facts"] = {
            "who": "WeakPaper",
            "action": "提出",
            "target": "机器人方法",
            "evidence": ["论文页面"],
        }
        layers = {
            "must_read": [],
            "physical_ai": [],
            "watch": [],
            "featured_papers": [weak],
            "paper_appendix": [],
            "brief": [],
        }

        result = evaluate_report_quality(
            layers,
            {
                "v8_enabled": True,
                "v10_enabled": True,
                "editorial_enabled": True,
                "physical_ai_featured_min_count": 0,
                "paper_technical_intro_min_count": 0,
                "primary_source_ratio_min": 0.0,
            },
        )

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["paper_plain_summary_missing_count"], 1)

    def test_v10_rejects_launch_copy_without_mechanism_or_result(self):
        item = {
            "title": "Google DeepMind releases Gemini Robotics 2",
            "title_cn": "Google DeepMind发布Gemini Robotics 2",
            "summary": "官方发布了一款新模型。",
            "facts": {
                "who": "Google DeepMind",
                "action": "发布",
                "target": "Gemini Robotics 2",
                "evidence": [
                    "Gemini Robotics 2 是 Google DeepMind 推出的新模型",
                    "该模型旨在为机器人提供全身智能",
                ],
            },
        }

        self.assertFalse(v10_has_concrete_news_evidence(item))

    def test_v10_repairs_truncated_paper_title_before_filtering(self):
        item = rich_paper(1006, section="paper_appendix")
        item["title"] = "CoTinyVLA: Chain-of-Thought Distillation for a Vision-Language-Action Model"
        item["title_cn"] = "CoTinyVLA：LIBERO-Plus鲁棒性基准上的机器人动作…"
        item["facts"]["target"] = "LIBERO-Plus鲁棒性基准上的机器人动作生成"

        repaired = repair_v10_paper_title(item)

        self.assertEqual(repaired["title_cn"], "CoTinyVLA：LIBERO-Plus鲁棒性基准上的机器人动作生成")
        self.assertFalse(title_looks_bad(repaired))

    def test_v10_accepts_target_title_when_paper_author_is_unnamed(self):
        item = rich_paper(1007, section="paper_appendix")
        item["title_cn"] = "Reeling It In：自主缝合灵活拾针任务"
        item["facts"]["who"] = "未具名研究团队"
        item["facts"]["target"] = "自主缝合中的灵活拾针任务，适用于针被遮挡或不可直接接近的场景"

        self.assertFalse(title_fact_mismatch(item, item["facts"]))

    def test_v10_accepts_specific_paper_target_without_model_list_in_title(self):
        item = rich_paper(1008, section="featured_papers")
        item.update(
            {
                "title": "Transformers Struggle to Use Their Emergent World Models",
                "title_cn": "Tower of Hanoi planning variants",
                "editorial_title": "Tower of Hanoi planning variants",
            }
        )
        item["facts"].update(
            {
                "who": "Transformers, Qwen and DeepSeek",
                "action": "build",
                "target": "Tower of Hanoi planning variants",
            }
        )

        self.assertFalse(title_fact_mismatch(item, item["facts"]))

    def test_v10_uses_final_editorial_title_for_fact_consistency(self):
        item = rich_paper(1009, section="featured_papers")
        item.update(
            {
                "title": "CofactVLA: Deconfounding VLA Models",
                "title_cn": "CofactVLA",
                "editorial_title": "CofactVLA visual overshadowing in VLA models",
            }
        )
        item["facts"].update(
            {
                "who": "CofactVLA",
                "action": "proposes",
                "target": "visual overshadowing in VLA models",
            }
        )

        self.assertFalse(title_fact_mismatch(item, item["facts"]))

    def test_v10_recomputes_stale_editorial_title_flags(self):
        item = rich_paper(1009, section="paper_appendix")
        item["title_cn"] = "Physical Agency：通用机器人编排差距"
        item["editorial_title"] = item["title_cn"]
        item["editorial_flags"] = ["mixed_language_title"]

        self.assertNotIn("mixed_language_title", item_quality_flags(item))

    def test_v10_filters_recently_sent_papers_and_arxiv_versions(self):
        fresh_paper = rich_paper(1010)
        repeated_paper = rich_paper(1011)
        repeated_paper["url"] = "https://arxiv.org/abs/2608.01234v2"
        history = [
            {
                **rich_paper(1012),
                "url": "https://arxiv.org/abs/2608.01234v1",
            }
        ]

        fresh, metrics = filter_recently_sent_papers(
            [fresh_paper, repeated_paper],
            history,
        )

        self.assertEqual([item["id"] for item in fresh], [1010])
        self.assertEqual(metrics["paper_repeat_filtered_count"], 1)
        self.assertEqual(metrics["paper_fresh_candidate_count"], 1)

    def test_v10_paper_identity_normalizes_arxiv_pdf_and_explicit_doi(self):
        from main import paper_history_key, paper_history_keys

        self.assertEqual(
            paper_history_key({"url": "https://arxiv.org/pdf/2608.01234v3.pdf"}),
            "arxiv:2608.01234",
        )
        self.assertEqual(
            paper_history_key({"url": "https://publisher.example/paper", "doi": "10.1234/ABC.9"}),
            "doi:10.1234/abc.9",
        )
        self.assertIn(
            "doi:10.1234/abc.9",
            paper_history_keys({
                "canonical_url": "https://arxiv.org/abs/2608.01234",
                "url": "https://doi.org/10.1234/ABC.9",
            }),
        )

    def test_v10_paper_identity_normalizes_arxiv_mirrors_and_tracking_urls(self):
        from main import paper_history_key, paper_history_keys

        arxiv_variants = (
            "https://export.arxiv.org/pdf/2608.01234v3.pdf?download=1",
            "https://ar5iv.labs.arxiv.org/html/2608.01234v2",
            "https://arxiv.org/abs/2608.01234#references",
        )
        self.assertEqual(
            {paper_history_key({"url": url}) for url in arxiv_variants},
            {"arxiv:2608.01234"},
        )
        self.assertIn(
            "https://publisher.example/paper?a=1",
            paper_history_keys({"url": "http://www.publisher.example/paper?utm_source=rss&a=1#abstract"}),
        )

    def test_v10_paper_identity_normalizes_doi_sentence_punctuation(self):
        from main import paper_history_key

        self.assertEqual(
            paper_history_key({"doi": "doi:10.1234/Robot.45)."}),
            "doi:10.1234/robot.45",
        )

    def test_v10_paper_identity_uses_original_title_across_unrelated_links(self):
        previous = rich_paper(10151)
        previous["url"] = "https://arxiv.org/abs/2608.04569"
        current = rich_paper(10161)
        current["url"] = "https://publisher.example/papers/aheadwm"
        current["canonical_url"] = current["url"]

        selected, metrics = filter_recently_sent_papers([current], [previous])

        self.assertEqual(selected, [])
        self.assertEqual(metrics["paper_repeat_filtered_count"], 1)

    def test_v10_paper_title_fingerprint_ignores_short_or_generated_chinese_title(self):
        from main import paper_history_keys, paper_title_fingerprint

        self.assertEqual(paper_title_fingerprint("Wonder"), "")
        self.assertNotIn(
            "title:世界模型方法与实验",
            paper_history_keys({"title": "Wonder", "title_cn": "世界模型方法与实验"}),
        )

    def test_v10_filters_same_paper_when_link_changes_from_arxiv_to_doi(self):
        previous = rich_paper(1013)
        previous.update({
            "canonical_url": "https://arxiv.org/abs/2608.04567",
            "url": "https://doi.org/10.1234/robot.45",
        })
        current = rich_paper(1014)
        current.update({
            "canonical_url": "https://publisher.example/robot-45",
            "url": "https://doi.org/10.1234/robot.45",
        })

        selected, metrics = filter_recently_sent_papers([current], [previous])

        self.assertEqual(selected, [])
        self.assertEqual(metrics["paper_repeat_filtered_count"], 1)

    def test_v10_arxiv_version_reads_versioned_url_before_canonical_url(self):
        from main import paper_arxiv_version

        self.assertEqual(
            paper_arxiv_version({
                "canonical_url": "https://arxiv.org/abs/2608.01234",
                "url": "https://arxiv.org/abs/2608.01234v3",
            }),
            3,
        )

    def test_v10_version_bump_with_paraphrased_facts_is_not_substantive(self):
        previous = rich_paper(1015)
        previous["url"] = "https://arxiv.org/abs/2608.09999v1"
        current = rich_paper(1016)
        current["url"] = "https://arxiv.org/abs/2608.09999v2"
        current["facts"]["method"] += "。"
        current["facts"]["metric_result"] += "。"

        selected, metrics = filter_recently_sent_papers([current], [previous])

        self.assertEqual(selected, [])
        self.assertEqual(metrics["paper_repeat_filtered_count"], 1)

    def test_arxiv_enrichment_persists_latest_submission_version(self):
        collector = ArxivCollector(categories=["cs.RO"], max_results=1)
        collector._fetch_text = lambda _url: '''
            <meta name="citation_abstract" content="A robot policy with new real-world results." />
            <meta name="citation_date" content="2026/08/01" />
            <meta name="citation_author" content="Alice Example" />
            <div class="submission-history"><strong>[v1]</strong> Sat, 1 Aug 2026 10:00:00 UTC<br><strong>[v3]</strong> Tue, 11 Aug 2026 12:30:00 UTC</div>
        '''

        enriched = collector.enrich_articles(
            [{
                "title": "Updated robot policy",
                "url": "https://arxiv.org/abs/2608.01234",
                "content": "robot policy",
                "topic": "Robotics",
            }]
        )[0]

        self.assertEqual(enriched["arxiv_version"], 3)
        self.assertEqual(enriched["canonical_url"], "https://arxiv.org/abs/2608.01234")
        self.assertEqual(enriched["url"], "https://arxiv.org/abs/2608.01234v3")
        self.assertEqual(enriched["publish_date"], "2026-08-11T12:30:00+00:00")

    def test_arxiv_collection_excludes_papers_outside_three_day_window(self):
        collector = ArxivCollector(
            categories=["cs.RO"],
            max_results=1,
            candidate_pool=50,
            topic_limits={"Robotics": 1},
            fallback_days=[1, 3],
        )
        collector._collect_topic_results = lambda *_args: [{
            "source": "ArXiv",
            "source_detail": "Robotics",
            "title": "Old robot paper",
            "url": "https://arxiv.org/abs/2501.00001",
            "content": "robot manipulation",
            "publish_date": "",
            "content_type": "paper",
            "topic": "Robotics",
            "initial_score": 5,
        }]
        collector._fetch_abs_metadata = lambda _url: {
            "abstract": "robot manipulation",
            "author": "A",
            "published": datetime(2025, 1, 1, tzinfo=timezone.utc),
            "updated": datetime(2025, 1, 1, tzinfo=timezone.utc),
            "arxiv_version": 1,
        }

        self.assertEqual(collector.collect(), [])

    def test_v10_allows_reappeared_paper_only_when_code_is_new(self):
        previous = rich_paper(1013)
        previous["url"] = "https://arxiv.org/abs/2608.04567v1"
        current = rich_paper(1014)
        current["url"] = "https://arxiv.org/abs/2608.04567v2"
        current["facts"]["code_or_project"] = "https://github.com/example/new-release"
        current["facts"]["evidence"].append("The authors have now released the official code repository.")

        selected, metrics = filter_recently_sent_papers([current], [previous])

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["paper_status_label"], "代码已发布")
        self.assertTrue(selected[0]["is_reappeared_update"])
        self.assertEqual(metrics["reappeared_paper_with_update_count"], 1)

    def test_v10_change_detection_falls_back_to_raw_facts_for_asset_urls(self):
        previous = rich_paper(10131)
        previous["url"] = "https://arxiv.org/abs/2608.04568v1"
        previous["facts_cn"] = {"method": "中文方法说明"}
        current = rich_paper(10141)
        current["url"] = "https://arxiv.org/abs/2608.04568v1"
        current["facts_cn"] = {"method": "中文方法说明"}
        current["facts"]["code_or_project"] = "https://github.com/example/new-release"
        current["facts"]["evidence"].append("The official code is now released.")

        selected, metrics = filter_recently_sent_papers([current], [previous])

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["paper_status_label"], "代码已发布")
        self.assertEqual(metrics["reappeared_paper_with_update_count"], 1)

    def test_v10_allows_new_arxiv_version_with_substantive_abstract_change(self):
        previous = rich_paper(1017)
        previous["url"] = "https://arxiv.org/abs/2608.05555v1"
        previous["facts"]["abstract"] = "The paper evaluates only simulated static manipulation."
        current = rich_paper(1018)
        current["url"] = "https://arxiv.org/abs/2608.05555v2"
        current["facts"]["abstract"] = "The new version adds real-robot dynamic manipulation experiments and failure analysis."

        selected, metrics = filter_recently_sent_papers([current], [previous])

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["paper_status_label"], "版本更新")
        self.assertEqual(metrics["reappeared_paper_with_update_count"], 1)

    def test_v10_allows_reappearance_when_model_weights_are_released(self):
        previous = rich_paper(1019)
        previous["url"] = "https://arxiv.org/abs/2608.06666v1"
        current = rich_paper(1020)
        current["url"] = "https://arxiv.org/abs/2608.06666v1"
        current["facts"]["model_weights"] = "https://huggingface.co/example/robot-policy"
        current["facts"]["evidence"].append("The model weights are now available on Hugging Face.")

        selected, metrics = filter_recently_sent_papers([current], [previous])

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["paper_status_label"], "重要进展")
        self.assertEqual(metrics["reappeared_paper_with_update_count"], 1)

    def test_v10_does_not_treat_newly_extracted_asset_field_as_a_release(self):
        previous = rich_paper(1023)
        previous["url"] = "https://arxiv.org/abs/2608.08888v1"
        current = rich_paper(1024)
        current["url"] = "https://arxiv.org/abs/2608.08888v1"
        current["facts"]["model_weights"] = "https://huggingface.co/example/already-existing"

        selected, metrics = filter_recently_sent_papers([current], [previous])

        self.assertEqual(selected, [])
        self.assertEqual(metrics["paper_repeat_filtered_count"], 1)

    def test_v10_does_not_treat_missing_old_metric_as_a_new_experiment(self):
        previous = rich_paper(1025)
        previous["url"] = "https://arxiv.org/abs/2608.08889v1"
        previous["facts"]["metric_result"] = ""
        current = rich_paper(1026)
        current["url"] = "https://arxiv.org/abs/2608.08889v1"
        current["facts"]["metric_result"] = "Real-robot success rate reached 86% across 120 trials."

        selected, metrics = filter_recently_sent_papers([current], [previous])

        self.assertEqual(selected, [])
        self.assertEqual(metrics["paper_repeat_filtered_count"], 1)

    def test_v10_unversioned_history_requires_update_after_last_send(self):
        previous = rich_paper(1027)
        previous["url"] = "https://arxiv.org/abs/2608.08890"
        previous["_history_created_at"] = "2026-08-12 13:00:00"
        current = rich_paper(1028)
        current["url"] = "https://arxiv.org/abs/2608.08890v3"
        current["publish_date"] = "2026-08-11T12:00:00+00:00"
        current["facts"]["abstract"] = "A substantially rewritten abstract with real-robot experiments."

        selected, metrics = filter_recently_sent_papers([current], [previous])

        self.assertEqual(selected, [])
        self.assertEqual(metrics["paper_repeat_filtered_count"], 1)

        current["publish_date"] = "2026-08-12T14:00:00+00:00"
        selected, metrics = filter_recently_sent_papers([current], [previous])

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["paper_status_label"], "版本更新")

    def test_v10_explicit_old_version_still_requires_post_send_update_time(self):
        previous = rich_paper(1029)
        previous["url"] = "https://arxiv.org/abs/2608.08891v1"
        previous["arxiv_version"] = 1
        previous["_history_created_at"] = "2026-08-12 13:00:00"
        current = rich_paper(1030)
        current["url"] = "https://arxiv.org/abs/2608.08891v2"
        current["arxiv_version"] = 2
        current["publish_date"] = "2026-08-12T12:30:00+00:00"
        current["facts"]["abstract"] = "A substantially rewritten abstract with new robot experiments."

        selected, metrics = filter_recently_sent_papers([current], [previous])

        self.assertEqual(selected, [])
        self.assertEqual(metrics["paper_repeat_filtered_count"], 1)

    def test_v10_reappearance_uses_delivery_time_not_report_creation_time(self):
        previous = rich_paper(1031)
        previous["url"] = "https://arxiv.org/abs/2608.08892v1"
        previous["arxiv_version"] = 1
        previous["_history_created_at"] = "2026-08-12 13:00:00"
        previous["_history_delivery_at"] = "2026-08-12 13:21:00"
        current = rich_paper(1032)
        current["url"] = "https://arxiv.org/abs/2608.08892v2"
        current["arxiv_version"] = 2
        current["facts"]["abstract"] = "A substantially rewritten abstract with new robot experiments."

        current["publish_date"] = "2026-08-12T13:10:00"
        selected, metrics = filter_recently_sent_papers([current], [previous])
        self.assertEqual(selected, [])
        self.assertEqual(metrics["paper_repeat_filtered_count"], 1)

        current["publish_date"] = "2026-08-12T13:22:00"
        selected, metrics = filter_recently_sent_papers([current], [previous])
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["paper_status_label"], "版本更新")

    def test_v10_labels_new_metric_result_as_new_experiment(self):
        previous = rich_paper(1021)
        previous["url"] = "https://arxiv.org/abs/2608.07777v1"
        previous["facts"]["metric_result"] = "Simulation success rate was 72%."
        current = rich_paper(1022)
        current["url"] = "https://arxiv.org/abs/2608.07777v1"
        current["facts"]["metric_result"] = "Real-robot success rate reached 86% across 120 trials."

        selected, metrics = filter_recently_sent_papers([current], [previous])

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["paper_status_label"], "新增实验")
        self.assertIn("86%", selected[0]["paper_change_reason"])
        self.assertEqual(metrics["reappeared_paper_with_update_count"], 1)

    def test_v10_drops_reappeared_update_with_truncated_english_reason(self):
        previous = rich_paper(1023)
        previous["url"] = "https://arxiv.org/abs/2609.00908v1"
        previous["facts"]["metric_result"] = "旧版尚未报告真实任务结果"
        current = rich_paper(1024)
        current["url"] = "https://arxiv.org/abs/2609.00908v2"
        current["arxiv_version"] = 2
        current["facts"]["metric_result"] = (
            "Evaluations on pi-0.5 and X-VLA across RoboTwin 2.0, LIBERO, "
            "and three real-world manipula..."
        )

        selected, metrics = filter_recently_sent_papers([current], [previous])

        self.assertEqual(selected, [])
        self.assertEqual(metrics["paper_repeat_filtered_count"], 1)
        self.assertEqual(metrics["reappeared_paper_with_update_count"], 0)

    def test_v10_domain_quota_preserves_cross_domain_coverage(self):
        papers = []
        for index, domain in enumerate(
            ["world_model"] * 5 + ["physical_ai"] * 6 + ["agent_models"] * 4 + ["infra_open_source"] * 3 + ["other"] * 2,
            start=3000,
        ):
            item = rich_paper(index)
            item["domain_key"] = domain
            item["score"] = 10000 - index
            papers.append(item)
        selected = select_papers_by_domain_quota(
            papers,
            {
                "world_model": {"min": 4, "max": 6},
                "physical_ai": {"min": 5, "max": 8},
                "agent_models": {"min": 3, "max": 5},
                "infra_open_source": {"min": 2, "max": 4},
                "other": {"min": 2, "max": 4},
            },
            25,
        )
        counts = {}
        for item in selected:
            domain = paper_quota_domain(item, {
                "world_model": {},
                "physical_ai": {},
                "agent_models": {},
                "infra_open_source": {},
                "other": {},
            })
            counts[domain] = counts.get(domain, 0) + 1

        self.assertGreaterEqual(counts["world_model"], 4)
        self.assertGreaterEqual(counts["physical_ai"], 5)
        self.assertGreaterEqual(counts["agent_models"], 3)
        self.assertGreaterEqual(counts["infra_open_source"], 2)
        self.assertGreaterEqual(counts["other"], 2)

    def test_v10_minimum_paper_count_can_overflow_a_concentrated_domain(self):
        papers = []
        for index in range(15):
            item = rich_paper(3500 + index)
            item["domain_key"] = "physical_ai"
            item["title"] = f"Distinct robot manipulation method {index}"
            item["title_cn"] = f"机器人操作方法 {index}"
            item["facts"]["target"] = f"机器人操作任务 {index}"
            papers.append(item)

        selected = select_papers_by_domain_quota(
            papers,
            {"physical_ai": {"min": 5, "max": 8}},
            total_limit=25,
            minimum_count=15,
        )

        self.assertEqual(len(selected), 15)

    def test_v10_paper_topic_locks_domain_through_editorial_enrichment(self):
        agent = rich_paper(3991)
        agent.update(
            {
                "topic": "Agent / Models",
                "domain_key": "world_model",
                "title": "Multi-agent path planning with predictive rollouts",
                "summary": "The method coordinates agents with multi-step planning.",
            }
        )
        quotas = {
            "world_model": {"min": 0, "max": 6},
            "physical_ai": {"min": 0, "max": 8},
            "agent_models": {"min": 0, "max": 5},
            "infra_open_source": {"min": 0, "max": 4},
            "other": {"min": 0, "max": 4},
        }

        selected = select_papers_by_domain_quota([agent], quotas, 1)
        decorated = enrich_editorial_fields(selected[0])

        self.assertEqual(paper_quota_domain(agent, quotas), "agent_models")
        self.assertEqual(selected[0]["paper_domain_key"], "agent_models")
        self.assertEqual(decorated["domain_key"], "agent_models")

    def test_v10_other_quota_uses_products_business_display_domain(self):
        item = rich_paper(3992)
        item.update({"topic": "Multimodal / Video", "domain_key": "world_model"})
        quotas = {"other": {"min": 1, "max": 1}}

        selected = select_papers_by_domain_quota([item], quotas, 1)
        decorated = enrich_editorial_fields(selected[0])

        self.assertEqual(selected[0]["paper_domain_key"], "other")
        self.assertEqual(selected[0]["domain_key"], "products_business")
        self.assertEqual(decorated["domain_key"], "products_business")

    def test_v10_domain_quota_gate_blocks_only_maximum_overflow(self):
        papers = []
        for index, domain in enumerate(["world_model"] * 7 + ["physical_ai"] * 5, start=3993):
            item = rich_paper(index)
            item["paper_domain_key"] = domain
            papers.append(item)
        quotas = {
            "world_model": {"min": 4, "max": 6},
            "physical_ai": {"min": 5, "max": 8},
            "agent_models": {"min": 3, "max": 5},
        }

        metrics = evaluate_paper_domain_quotas(papers, quotas)

        self.assertEqual(metrics["paper_domain_quota_status"], "failed")
        self.assertEqual(metrics["paper_domain_quota_exceeded"]["world_model"], {"count": 7, "max": 6})
        self.assertEqual(metrics["paper_domain_quota_underfilled"]["agent_models"], {"count": 0, "min": 3})

        metrics = evaluate_paper_domain_quotas(papers[1:], quotas)
        self.assertEqual(metrics["paper_domain_quota_status"], "passed")
        self.assertIn("agent_models", metrics["paper_domain_quota_underfilled"])

    def test_v10_reappeared_updates_do_not_fill_fresh_domain_quotas(self):
        fresh = rich_paper(39930)
        fresh["paper_domain_key"] = "world_model"
        update = rich_paper(39931)
        update["paper_domain_key"] = "physical_ai"
        update["is_reappeared_update"] = True

        metrics = evaluate_paper_domain_quotas(
            [fresh, update],
            {
                "world_model": {"min": 1, "max": 2},
                "physical_ai": {"min": 1, "max": 2},
            },
        )

        self.assertEqual(metrics["paper_domain_counts"], {"world_model": 1})
        self.assertEqual(metrics["paper_domain_fresh_count"], 1)
        self.assertEqual(metrics["paper_domain_quota_underfilled"]["physical_ai"]["count"], 0)

    def test_v10_final_overlap_metric_requires_strictly_less_than_ten_percent(self):
        current = [rich_paper(index) for index in range(4000, 4010)]
        history = [{**current[0], "_history_report_id": "sent-1"}]

        metrics = finalize_paper_freshness_metrics(current, history, overlap_max=0.10)

        self.assertEqual(metrics["adjacent_report_paper_overlap_rate"], 0.1)
        self.assertEqual(metrics["paper_freshness_status"], "failed")
        metrics = finalize_paper_freshness_metrics(current[:5], history, overlap_max=0.10)
        self.assertEqual(metrics["paper_freshness_status"], "failed")

    def test_v10_freshness_gate_rejects_within_report_duplicate_identity(self):
        first = rich_paper(4100)
        duplicate = {**rich_paper(4101), "url": "https://arxiv.org/abs/4100v2"}

        metrics = finalize_paper_freshness_metrics([first, duplicate], [], overlap_max=0.10)

        self.assertEqual(metrics["paper_within_report_duplicate_count"], 1)
        self.assertEqual(metrics["paper_freshness_status"], "failed")

    def test_v10_three_consecutive_selections_stay_below_ten_percent_overlap(self):
        candidates = []
        domains = (
            ["world_model"] * 18
            + ["physical_ai"] * 24
            + ["agent_models"] * 15
            + ["infra_open_source"] * 12
            + ["other"] * 12
        )
        for index, domain in enumerate(domains, start=5000):
            item = rich_paper(index)
            item["domain_key"] = domain
            item["score"] = 10000 - index
            candidates.append(item)
        quotas = {
            "world_model": {"min": 4, "max": 6},
            "physical_ai": {"min": 5, "max": 8},
            "agent_models": {"min": 3, "max": 5},
            "infra_open_source": {"min": 2, "max": 4},
            "other": {"min": 2, "max": 4},
        }
        history = []
        reports = []
        for report_index in range(3):
            fresh, _ = filter_recently_sent_papers(candidates, history)
            selected = select_papers_by_domain_quota(fresh, quotas, 20)
            metrics = finalize_paper_freshness_metrics(selected, history, overlap_max=0.10)
            self.assertEqual(len(selected), 20)
            self.assertLess(metrics["adjacent_report_paper_overlap_rate"], 0.10)
            self.assertEqual(metrics["paper_freshness_status"], "passed")
            report_id = f"validation-{report_index + 1}"
            history = [
                {**item, "_history_report_id": report_id}
                for item in selected
            ] + history
            reports.append(selected)
        self.assertEqual(
            len({item["url"] for report in reports for item in report}),
            60,
        )

    def test_v10_three_generated_test_emails_render_fresh_paper_counts(self):
        candidates = []
        domains = (
            ["world_model"] * 18
            + ["physical_ai"] * 24
            + ["agent_models"] * 15
            + ["infra_open_source"] * 12
            + ["other"] * 12
        )
        for index, domain in enumerate(domains, start=5200):
            item = rich_paper(index)
            item["domain_key"] = domain
            item["score"] = 10000 - index
            candidates.append(item)
        quotas = {
            "world_model": {"min": 4, "max": 6},
            "physical_ai": {"min": 5, "max": 8},
            "agent_models": {"min": 3, "max": 5},
            "infra_open_source": {"min": 2, "max": 4},
            "other": {"min": 2, "max": 4},
        }
        history = []
        previous_urls = set()
        for report_index in range(3):
            eligible, base_metrics = filter_recently_sent_papers(candidates, history)
            selected = select_papers_by_domain_quota(eligible, quotas, 20)
            metrics = finalize_paper_freshness_metrics(
                selected,
                history,
                base_metrics,
                overlap_max=0.10,
            )
            rendered = [enrich_editorial_fields(item) for item in selected]
            featured = rendered[:10]
            appendix = rendered[10:]
            generator = ReportGenerator(
                design_version="v10-learning-digest",
                report_config={
                    "design_version": "v10-learning-digest",
                    "paper_featured_limit": 10,
                    "paper_appendix_limit": 15,
                    "paper_freshness_metrics": metrics,
                },
            )
            html = generator.generate_html(
                papers=rendered,
                updates=[],
                mixed_items=rendered,
                report_summary={},
                layered_updates={
                    "must_read": [],
                    "physical_ai": [],
                    "watch": [],
                    "featured_papers": featured,
                    "paper_appendix": appendix,
                    "brief": [],
                },
            )

            current_urls = {item["url"] for item in selected}
            self.assertEqual(len(current_urls), 20)
            self.assertLess(metrics["adjacent_report_paper_overlap_rate"], 0.10)
            self.assertEqual(metrics["paper_freshness_status"], "passed")
            self.assertFalse(current_urls & previous_urls)
            self.assertIn("本期新增论文 20 篇", html)
            self.assertIn("今日新增", html)
            self.assertEqual(html.count('class="v10-status">今日新增</span>'), 20)
            for item in selected:
                self.assertIn(item["url"], html)

            report_id = f"generated-validation-{report_index + 1}"
            history = [
                {**item, "_history_report_id": report_id}
                for item in selected
            ] + history
            previous_urls = current_urls

    def test_v10_twelve_fresh_papers_are_not_padded_with_old_history(self):
        fresh_candidates = [rich_paper(index) for index in range(6000, 6012)]
        for index, item in enumerate(fresh_candidates):
            item["domain_key"] = ("world_model", "physical_ai", "agent_models")[index % 3]
        old_history = []
        for index in range(7000, 7030):
            old = rich_paper(index)
            old["_history_report_id"] = "sent-old"
            old_history.append(old)

        eligible, metrics = filter_recently_sent_papers(fresh_candidates + old_history, old_history)
        selected = select_papers_by_domain_quota(eligible, {}, 25)

        self.assertEqual(len(selected), 12)
        self.assertEqual(metrics["paper_repeat_filtered_count"], 30)

    def test_v10_reselects_fresh_papers_when_updates_would_reach_ten_percent(self):
        fresh = [rich_paper(index) for index in range(8000, 8030)]
        updates = [rich_paper(index) for index in range(8100, 8110)]
        for index, item in enumerate(fresh + updates):
            item["domain_key"] = ("world_model", "physical_ai", "agent_models", "infra_open_source", "other")[index % 5]
            item["score"] = 100 - index
        for index, item in enumerate(updates):
            item["is_reappeared_update"] = True
            item["score"] = 1000 - index

        selected, filtered = select_papers_with_strict_freshness(
            fresh + updates,
            {},
            total_limit=20,
            overlap_max=0.10,
        )

        selected_updates = sum(1 for item in selected if item.get("is_reappeared_update"))
        selected_fresh = sum(1 for item in selected if not item.get("is_reappeared_update"))
        self.assertEqual(len(selected), 22)
        self.assertEqual(selected_fresh, 20)
        self.assertEqual(selected_updates, 2)
        self.assertLess(selected_updates / len(selected), 0.10)
        self.assertEqual(filtered, 8)

    def test_v10_update_allowance_stays_strict_across_supported_report_sizes(self):
        expected_updates = {10: 1, 12: 1, 20: 2, 25: 2}
        for total_limit, expected_count in expected_updates.items():
            with self.subTest(total_limit=total_limit):
                fresh = [rich_paper(9000 + index) for index in range(30)]
                updates = [rich_paper(9100 + index) for index in range(5)]
                for index, item in enumerate(fresh + updates):
                    item["domain_key"] = (
                        "world_model",
                        "physical_ai",
                        "agent_models",
                        "infra_open_source",
                        "other",
                    )[index % 5]
                    item["score"] = 100 - index
                for index, item in enumerate(updates):
                    item["is_reappeared_update"] = True
                    item["score"] = 1000 - index

                selected, _ = select_papers_with_strict_freshness(
                    fresh + updates,
                    {},
                    total_limit=total_limit,
                    overlap_max=0.10,
                )

                selected_updates = sum(1 for item in selected if item.get("is_reappeared_update"))
                selected_fresh = sum(1 for item in selected if not item.get("is_reappeared_update"))
                self.assertEqual(selected_fresh, total_limit)
                self.assertEqual(len(selected), total_limit + expected_count)
                self.assertEqual(selected_updates, expected_count)
                self.assertLess(selected_updates / len(selected), 0.10)

    def test_v10_final_visible_reselect_handles_budget_shrink(self):
        papers = [rich_paper(9200 + index) for index in range(20)]
        papers[0]["is_reappeared_update"] = True
        papers[1]["is_reappeared_update"] = True
        layers = {
            "must_read": [],
            "physical_ai": [],
            "watch": [],
            "featured_papers": papers[:10],
            "paper_appendix": papers[10:],
            "brief": [],
        }

        updated, removed = enforce_final_visible_paper_overlap(layers, overlap_max=0.10)
        visible = [
            item
            for section in ("featured_papers", "paper_appendix")
            for item in updated[section]
        ]

        self.assertEqual(removed, 1)
        self.assertEqual(len(visible), 19)
        self.assertEqual(sum(1 for item in visible if item.get("is_reappeared_update")), 1)
        self.assertLess(1 / len(visible), 0.10)

    def test_v10_final_visible_reselect_prefers_shorter_email_to_ten_percent_overlap(self):
        papers = [rich_paper(9300 + index) for index in range(10)]
        papers[0]["is_reappeared_update"] = True
        layers = {
            "must_read": [],
            "physical_ai": [],
            "watch": [],
            "featured_papers": papers,
            "paper_appendix": [],
            "brief": [],
        }

        updated, removed = enforce_final_visible_paper_overlap(layers, overlap_max=0.10)

        self.assertEqual(removed, 1)
        self.assertEqual(len(updated["featured_papers"]), 9)
        self.assertFalse(any(item.get("is_reappeared_update") for item in updated["featured_papers"]))

    def test_v10_rendered_freshness_total_can_include_final_reselect(self):
        metrics = {
            "fresh_paper_count": 9,
            "paper_repeat_filtered_count": 4 + 1,
            "final_overlap_reselected_count": 1,
        }
        generator = ReportGenerator(
            design_version="v10-learning-digest",
            report_config={
                "design_version": "v10-learning-digest",
                "paper_freshness_metrics": metrics,
            },
        )

        html = generator.generate_html(
            papers=[],
            updates=[],
            mixed_items=[],
            report_summary={},
            layered_updates={
                "must_read": [],
                "physical_ai": [],
                "watch": [],
                "featured_papers": [],
                "paper_appendix": [],
                "brief": [],
            },
        )

        self.assertIn("过滤近期及上一封重复 5 篇", html)

    def test_v10_repairs_vla_paper_title_with_acronym_and_chinese_target(self):
        item = rich_paper(1008)
        item["title"] = (
            "RL$^2$-VLA: Adaptive RL Latent Compositional Steering with "
            "Test-Time Scaling for Vision-Language-Action Models"
        )
        item["title_cn"] = "RL^2-VLA：Vision-Language-Action 模型在挑战性和分布外任务上的性能"
        item["facts"]["who"] = "RL^2-VLA"
        item["facts"]["target"] = "Vision-Language-Action (VLA) 模型在挑战性和分布外任务上的性能"

        repaired = repair_v10_paper_title(item)

        self.assertEqual(repaired["title_cn"], "RL^2-VLA：VLA 模型在挑战性和分布外任务上的性能")
        self.assertFalse(mixed_language_title(repaired["title_cn"]))
        self.assertFalse(title_fact_mismatch(repaired, repaired["facts"]))

    def test_v10_renders_full_body_for_updates_beyond_domain_cap(self):
        must_read = [enrich_editorial_fields(rich_news(article_id)) for article_id in range(2000, 2010)]
        watch = [enrich_editorial_fields(rich_news(article_id)) for article_id in range(2010, 2025)]
        generator = ReportGenerator(
            design_version="v10-learning-digest",
            report_config={
                "design_version": "v10-learning-digest",
                "must_read_limit": 10,
                "domain_item_limit": 4,
                "min_visible_information_count": 25,
            },
        )

        html = generator.generate_html(
            papers=[],
            updates=must_read + watch,
            mixed_items=must_read + watch,
            report_summary={},
            layered_updates={
                "must_read": must_read,
                "physical_ai": [],
                "watch": watch,
                "featured_papers": [],
                "paper_appendix": [],
                "brief": [],
            },
        )

        self.assertIn("新闻、博客与访谈", html)
        self.assertEqual(html.count('class="v10-entry"'), 25)

    def test_v10_arxiv_topics_and_candidate_pool_are_active(self):
        collector = ArxivCollector(
            categories=["cs.RO", "cs.AI", "cs.CV", "cs.LG"],
            max_results=56,
            candidate_pool=600,
            topic_limits={
                "Physical AI": 12,
                "World Model": 12,
                "Robotics": 12,
                "Agent / Models": 10,
                "Multimodal / Video": 8,
                "Infra / Efficient AI": 8,
            },
            topic_queries={topic: topic for topic in (
                "Physical AI",
                "World Model",
                "Robotics",
                "Agent / Models",
                "Multimodal / Video",
                "Infra / Efficient AI",
            )},
        )
        requested = []
        collector._fetch_text = lambda url: requested.append(url) or arxiv_list_html()

        collector._fetch_recent_candidates("cs.AI")

        self.assertIn("show=600", requested[0])
        self.assertTrue(collector._matches_topic("Agent / Models", "Tool-use agents", "A reasoning model workflow"))
        self.assertTrue(collector._matches_topic("Multimodal / Video", "Video understanding", "A multimodal benchmark"))
        self.assertTrue(collector._matches_topic("Infra / Efficient AI", "Efficient inference", "Quantization for serving"))

    def test_v10_arxiv_recent_page_retries_smaller_show_count(self):
        collector = ArxivCollector(
            categories=["cs.RO"],
            max_results=4,
            candidate_pool=600,
            topic_limits={"Robotics": 4},
        )
        requested = []

        def fetch(url):
            requested.append(url)
            if "show=600" in url:
                raise RuntimeError("HTTP 400")
            return arxiv_list_html()

        collector._fetch_text = fetch

        self.assertEqual(len(collector._fetch_recent_candidates("cs.RO")), 1)
        self.assertIn("show=600", requested[0])
        self.assertIn("show=500", requested[1])
        self.assertEqual(collector.fetch_diagnostics["request_error_count"], 1)
        self.assertEqual(collector.fetch_diagnostics["successful_page_count"], 1)
        self.assertEqual(
            collector.fetch_diagnostics["retry_paths"],
            [{
                "category": "cs.RO",
                "attempted_show_counts": [600, 500],
                "successful_show_count": 500,
                "result": "success",
            }],
        )

    def test_v10_arxiv_empty_page_is_reported_as_parse_error(self):
        collector = ArxivCollector(
            categories=["cs.RO"],
            max_results=4,
            candidate_pool=500,
            topic_limits={"Robotics": 4},
        )
        collector._fetch_text = lambda url: "<html><body>layout changed</body></html>"

        with self.assertRaisesRegex(ValueError, "arxiv_page_parse_empty"):
            collector._fetch_recent_candidates("cs.RO")
        self.assertEqual(collector.fetch_diagnostics["page_parse_error_count"], 3)

    def test_v10_arxiv_explicit_zero_page_is_not_a_parse_error(self):
        collector = ArxivCollector(
            categories=["cs.RO"],
            max_results=4,
            candidate_pool=500,
            topic_limits={"Robotics": 4},
        )
        requested = []
        collector._fetch_text = lambda url: requested.append(url) or (
            "<html><body><h3>No articles found in cs.RO for this date.</h3></body></html>"
        )

        papers = collector._fetch_recent_candidates("cs.RO")

        self.assertEqual(papers, [])
        self.assertEqual(len(requested), 1)
        self.assertEqual(collector.fetch_diagnostics["request_error_count"], 0)
        self.assertEqual(collector.fetch_diagnostics["page_parse_error_count"], 0)
        self.assertEqual(collector.fetch_diagnostics["successful_page_count"], 1)
        self.assertEqual(collector.fetch_diagnostics["explicit_zero_page_count"], 1)
        self.assertTrue(collector.fetch_diagnostics["true_zero_result"])

    def test_v10_arxiv_successful_page_without_topic_match_is_not_true_zero(self):
        collector = ArxivCollector(
            categories=["cs.RO"],
            max_results=4,
            candidate_pool=500,
            topic_limits={"Robotics": 4},
            topic_queries={"Robotics": "robotics"},
            fallback_days=[1],
        )

        def no_match(*_args):
            collector.fetch_diagnostics["successful_page_count"] += 1
            collector.fetch_diagnostics["parsed_candidate_count"] += 3
            return []

        collector._collect_topic_results = no_match

        self.assertEqual(collector.collect(), [])
        self.assertFalse(collector.fetch_diagnostics["true_zero_result"])
        self.assertTrue(collector.fetch_diagnostics["no_match_result"])

    def test_v10_arxiv_parse_error_recovers_at_smaller_page_size(self):
        collector = ArxivCollector(
            categories=["cs.RO"],
            max_results=4,
            candidate_pool=500,
            topic_limits={"Robotics": 4},
        )
        requested = []

        def fetch(url):
            requested.append(url)
            return arxiv_list_html() if "show=100" in url else "<html>temporary response</html>"

        collector._fetch_text = fetch

        papers = collector._fetch_recent_candidates("cs.RO")

        self.assertEqual(len(papers), 1)
        self.assertEqual(collector.fetch_diagnostics["page_parse_error_count"], 1)
        self.assertEqual(collector.fetch_diagnostics["successful_page_count"], 1)
        self.assertIn("show=100", requested[-1])
        self.assertEqual(collector.fetch_diagnostics["retry_paths"][0]["attempted_show_counts"], [500, 100])
        self.assertEqual(collector.fetch_diagnostics["retry_paths"][0]["successful_show_count"], 100)

    def test_v10_arxiv_http_errors_recover_at_show_50(self):
        collector = ArxivCollector(
            categories=["cs.RO"],
            max_results=4,
            candidate_pool=500,
            topic_limits={"Robotics": 4},
        )
        requested = []

        def fetch(url):
            requested.append(url)
            if not url.endswith("show=50"):
                raise RuntimeError("HTTP 503")
            return arxiv_list_html()

        collector._fetch_text = fetch

        papers = collector._fetch_recent_candidates("cs.RO")

        self.assertEqual(len(papers), 1)
        self.assertEqual([url.rsplit("show=", 1)[-1] for url in requested], ["500", "100", "50"])
        self.assertEqual(collector.fetch_diagnostics["request_error_count"], 2)
        self.assertEqual(collector.fetch_diagnostics["successful_page_count"], 1)
        self.assertEqual(collector.fetch_diagnostics["retry_paths"][0], {
            "category": "cs.RO",
            "attempted_show_counts": [500, 100, 50],
            "successful_show_count": 50,
            "result": "success",
        })

    def test_v10_arxiv_retry_failure_classification_is_local_to_each_category(self):
        collector = ArxivCollector(
            categories=["cs.AI", "cs.RO"],
            max_results=4,
            candidate_pool=500,
            topic_limits={"Robotics": 4},
        )

        collector._fetch_text = lambda url: (_ for _ in ()).throw(RuntimeError("HTTP 503"))
        with self.assertRaises(RuntimeError):
            collector._fetch_recent_candidates("cs.AI")

        collector._fetch_text = lambda url: "<html>temporary response</html>"
        with self.assertRaisesRegex(ValueError, "arxiv_page_parse_empty"):
            collector._fetch_recent_candidates("cs.RO")

        self.assertEqual(collector.fetch_diagnostics["retry_paths"][0]["result"], "http_error")
        self.assertEqual(collector.fetch_diagnostics["retry_paths"][1]["result"], "parse_error")


if __name__ == "__main__":
    unittest.main()
