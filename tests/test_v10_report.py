import unittest
from datetime import datetime, timezone

from main import (
    effective_fresh_paper_minimum,
    evaluate_report_quality,
    evaluate_paper_domain_quotas,
    enforce_final_visible_paper_overlap,
    finalize_paper_freshness_metrics,
    filter_recently_sent_papers,
    item_quality_flags,
    repair_v10_paper_title,
    select_papers_by_domain_quota,
    select_papers_with_strict_freshness,
    paper_quota_domain,
    title_fact_mismatch,
    title_looks_bad,
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
        self.assertIn("15 分钟技术学习版", html)
        self.assertIn("今日编辑判断", html)
        self.assertIn("今日重点情报", html)
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

        self.assertIn("更多技术动态", html)
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
