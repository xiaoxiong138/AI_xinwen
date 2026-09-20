import copy
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import codex_research

from src.collectors.codex_research_inbox_collector import (
    CodexResearchInboxCollector,
    build_codex_research_inbox_collector,
    build_codex_research_readiness_summary,
    evaluate_sent_history_overlap,
)
from main import scan_final_html_quality
from src.database import Database
from src.editorial_engine import attribution_opener_pattern, enrich_editorial_fields
from src.generator import editorial_item_render_key
from src.processors.llm_processor import LLMProcessor
from src.relevance import is_ai_web_content


def test_codex_research_prompt_preserves_v11_candidate_and_evidence_contract():
    prompt = (
        Path(__file__).resolve().parents[1]
        / "prompts"
        / "codex_research_prompt.md"
    ).read_text(encoding="utf-8")

    for requirement in (
        "输出 75-85 条",
        "`news` 至少提交 30 条",
        "`technical` 至少提交 27 条",
        "`paper` 至少提交 20 篇",
        '"discovery_candidates"',
        "至少包含 `news` 50 条、`technical` 50 条、`paper` 40 篇",
        "原子替换",
        "sent_history_overlap_count=0",
        "codex-research-v3",
        '"key_numbers"',
        "新闻 8 条、非论文技术 10 条、论文 10 篇",
        "公司称／团队介绍／发布方披露／官方公告显示",
        "任意两条相似度达到 0.92",
        "unsupported_summary_numeric_examples",
        "key_number_public_copy_missing_examples",
    ):
        assert requirement in prompt


def test_codex_research_cli_json_includes_readiness_summary(monkeypatch, capsys):
    class StaleCollector:
        def __init__(self):
            self.fetch_diagnostics = {
                "fresh": False,
                "age_minutes": 500.0,
                "quality_status": "stale_inbox",
                "submission_quota_status": "not_loaded",
                "submitted_section_counts": {},
            }

        def collect(self):
            return []

    monkeypatch.setattr(
        codex_research,
        "load_config",
        lambda: {
            "sources": {
                "codex_research_inbox": {
                    "minimum_news": 20,
                    "minimum_technical": 20,
                    "minimum_papers": 15,
                }
            }
        },
    )
    monkeypatch.setattr(
        codex_research,
        "build_codex_research_inbox_collector",
        lambda *_args, **_kwargs: StaleCollector(),
    )
    monkeypatch.setattr(sys, "argv", ["codex_research.py", "--json"])

    assert codex_research.main() == 1
    payload = json.loads(capsys.readouterr().out)
    summary = payload["diagnostics"]["readiness_summary"]
    assert summary["status"] == "blocked"
    assert summary["accepted_underfilled"] == [
        "news:0/20",
        "technical:0/20",
        "paper:0/15",
    ]
    assert "stale_inbox" in summary["blockers"]


def test_codex_research_promotes_valid_candidate_atomically(tmp_path, monkeypatch, capsys):
    candidate = tmp_path / "candidate.json"
    target = tmp_path / "latest.json"
    row = _row(1, "news")
    row["facts"]["key_numbers"] = []
    candidate_payload = {
        "schema_version": "codex-research-v3",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "items": [row],
    }
    candidate.write_text(
        json.dumps(candidate_payload, ensure_ascii=False),
        encoding="utf-8",
    )
    target.write_text('{"old": true}', encoding="utf-8")
    database_path = tmp_path / "reports.sqlite3"
    monkeypatch.setattr(
        codex_research,
        "load_config",
        lambda: {
            "sources": {
                "codex_research_inbox": {
                    "path": str(target),
                    "required_schema_version": "codex-research-v3",
                    "minimum_items": 1,
                    "minimum_news": 1,
                    "minimum_technical": 0,
                    "minimum_papers": 0,
                    "technical_category_quotas": {},
                    "technical_category_minimums": {},
                    "paper_domain_quotas": {},
                    "news_format_quotas": {},
                    "minimum_fresh_by_section": {},
                }
            },
            "database": {"path": str(database_path)},
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "codex_research.py",
            "--candidate",
            str(candidate),
            "--promote-on-pass",
            "--json",
        ],
    )

    assert codex_research.main() == 0
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "passed"
    assert output["promotion"]["status"] == "promoted"
    assert output["promotion"]["promoted"] is True
    assert json.loads(target.read_text(encoding="utf-8")) == candidate_payload
    assert candidate.exists()
    assert not list(tmp_path.glob(".latest.json.*.tmp"))


def test_codex_research_failed_candidate_does_not_replace_production(tmp_path, monkeypatch, capsys):
    candidate = tmp_path / "candidate.json"
    target = tmp_path / "latest.json"
    candidate.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "items": [_row(1, "news")],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    original = '{"production": "preserved"}'
    target.write_text(original, encoding="utf-8")
    monkeypatch.setattr(
        codex_research,
        "load_config",
        lambda: {
            "sources": {
                "codex_research_inbox": {
                    "path": str(target),
                    "required_schema_version": "codex-research-v3",
                    "minimum_items": 1,
                    "minimum_news": 1,
                    "minimum_technical": 0,
                    "minimum_papers": 0,
                }
            },
            "database": {"path": str(tmp_path / "reports.sqlite3")},
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "codex_research.py",
            "--candidate",
            str(candidate),
            "--promote-on-pass",
            "--json",
        ],
    )

    assert codex_research.main() == 1
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "failed"
    assert output["promotion"]["status"] == "blocked"
    assert output["promotion"]["promoted"] is False
    assert target.read_text(encoding="utf-8") == original
    assert candidate.exists()


def test_production_inbox_path_respects_environment_override(tmp_path, monkeypatch):
    configured = tmp_path / "configured.json"
    overridden = tmp_path / "overridden.json"
    monkeypatch.setenv("WEB_AGENT_CODEX_RESEARCH_INBOX_PATH", str(overridden))

    assert codex_research.resolve_production_inbox_path(
        {"path": str(configured)}
    ) == overridden


def test_atomic_promotion_rejects_candidate_changed_after_validation(tmp_path):
    candidate = tmp_path / "candidate.json"
    target = tmp_path / "latest.json"
    candidate.write_text('{"new": true}', encoding="utf-8")
    original = '{"production": "preserved"}'
    target.write_text(original, encoding="utf-8")

    result = codex_research.promote_candidate(
        candidate,
        target,
        expected_sha256="0" * 64,
    )

    assert result["status"] == "candidate_changed_after_validation"
    assert result["promoted"] is False
    assert target.read_text(encoding="utf-8") == original
    assert not list(tmp_path.glob(".latest.json.*.tmp"))


def test_collector_hash_matches_raw_crlf_candidate_for_atomic_promotion(tmp_path):
    candidate = tmp_path / "candidate.json"
    target = tmp_path / "latest.json"
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "items": [_row(1, "news")],
    }
    raw_payload = json.dumps(payload, ensure_ascii=False, indent=2).replace(
        "\n", "\r\n"
    ).encode("utf-8")
    candidate.write_bytes(raw_payload)

    collector = CodexResearchInboxCollector(str(candidate))
    collector.collect()
    assert collector.fetch_diagnostics["schema_error_count"] == 0

    result = codex_research.promote_candidate(
        candidate,
        target,
        expected_sha256=collector.fetch_diagnostics["inbox_sha256"],
    )

    assert result["status"] == "promoted"
    assert result["promoted"] is True
    assert target.read_bytes() == raw_payload


def _row(index: int, content_type: str) -> dict:
    primary_section = "paper" if content_type == "paper" else ("technical" if content_type == "project" else "news")
    kind = {"paper": "论文", "technical": "技术", "news": "新闻"}[primary_section]
    summary = (
        "Example AI Lab 发布这项人工智能资料，先说明系统面对的具体问题，再解释模型训练、推理链路和机器人实验如何衔接。"
        "原文给出了可核验的方法步骤、输入输出、实验设置和结果，并交代了适用场景与尚未覆盖的限制，"
        "因此邮件正文可以独立阅读，而不是只复述标题或用抽象判断填充篇幅。"
        "读者不打开链接也能先理解核心变化。"
        "这份整理还保留了发布主体、公开时间和原始证据位置，并把发布方自报数据与第三方验证明确区分开来。"
        "涉及数字时会同时写明测试条件，避免脱离硬件、数据集或基线比较结论。"
    )
    if primary_section == "technical":
        summary += (
            "工程实现还列出了模块接口、部署约束和代码入口，便于继续复现。"
            "技术说明进一步解释了组件之间的数据流、训练或推理阶段的约束，以及相对旧方案究竟替换了哪一个环节。"
        )
    technical_categories = (
        ["embodied_world_model"] * 6
        + ["agent_systems"] * 4
        + ["training_data"] * 4
        + ["inference_deployment"] * 3
        + ["multimodal_architecture"] * 3
    )
    row = {
        "primary_section": primary_section,
        "title": f"AI {kind} source {index}",
        "title_cn": f"人工智能{kind}资料{index}",
        "url": f"https://example.com/ai/{content_type}/{index}",
        "source_detail": "Example AI Lab",
        "platform": "Paper" if content_type == "paper" else "News",
        "content_type": content_type,
        "claim_type": "research_result" if primary_section == "paper" else "verified_fact",
        "publish_date": datetime.now(timezone.utc).date().isoformat(),
        "technical_category": technical_categories[index % 20] if primary_section == "technical" else "",
        "summary": summary,
        "source_excerpt": "原文方法章节给出了完整训练链路、模块输入输出、对照实验与量化结果。",
        "evidence_locator": "Methods 与 Results",
        "summary_preview": "包含具体方法与实验结果",
        "facts": {
            "who": "Example AI Lab",
            "action": "发布",
            "target": f"AI model {primary_section} {index}",
            "evidence": ["原文给出了模型方法和实验结果"],
            "method": "predictive training",
            "metric_result": "benchmark improved",
        },
        "keywords": ["AI", "训练"],
        "score": 8.0,
        "evidence_quality": 0.8,
        "information_density": 0.8,
    }
    if primary_section == "paper":
        row["source_excerpt"] = (
            "实验在动态操作基准上报告任务成功率提高12个百分点，推理延迟增加8毫秒。"
        )
        row["facts"].update({
            "evidence": ["任务成功率提高12个百分点，推理延迟增加8毫秒"],
            "dataset_or_benchmark": "动态操作基准",
            "metric_result": "任务成功率提高12个百分点，推理延迟增加8毫秒",
            "baseline": "不使用预测模块的动作策略",
            "limitation": "尚未覆盖长时间遮挡、多机器人协同和不同硬件平台",
        })
        row["paper_plain_summary"] = (
            "这篇论文研究机器人在动态环境里容易依据过时画面做动作的问题。"
            "它先预测下一时刻的物体状态，再把预测结果交给动作策略，并在统一任务中和原方法比较。"
            "实验中，论文报告任务成功率提高了12个百分点，说明动作前预测确实能减少状态滞后。"
        )
        row["paper_technical_intro"] = (
            "论文采用两阶段约束训练：第一阶段用未来状态监督训练轻量预测模块，第二阶段冻结原有动作策略，"
            "只把预测状态作为额外输入接入控制链路。实验在动态操作基准上以不使用预测模块的策略为基线，"
            "任务成功率提高12个百分点，同时推理延迟增加8毫秒。作者也指出，当前结果尚未覆盖长时间遮挡、"
            "多机器人协同和不同硬件平台，因此工程价值仍要结合跨平台复现判断。"
        )
    elif primary_section == "technical":
        row["facts"].update({
            "architecture": "输入适配层、状态路由层和执行模块",
            "input_output": "输入任务与上下文，输出经过权限校验的执行结果",
            "baseline": "原有单阶段串行处理流程",
            "limitation": "尚未覆盖跨硬件部署和长期生产负载",
            "deployment_context": "单机测试环境与受控任务集",
        })
    return row


def _balanced_papers() -> list[dict]:
    topics = (
        ["World Model"] * 3
        + ["Physical AI / Robotics"] * 4
        + ["Agent / Models"] * 3
        + ["Infra / Open Source"] * 2
        + ["Multimodal Research"] * 3
    )
    papers = [_row(index, "paper") for index in range(15)]
    for paper, topic in zip(papers, topics):
        paper["topic"] = topic
    return papers


def _balanced_news() -> list[dict]:
    news = [_row(index, "news") for index in range(20)]
    for index in range(2):
        news[index]["content_type"] = "interview"
        news[index]["platform"] = "Interview"
        news[index]["claim_type"] = "interview_opinion"
        news[index]["facts"]["claim_type"] = "interview_opinion"
        news[index]["evidence_locator"] = f"文字稿 {12 + index}:40-15:10"
        news[index]["summary"] += "\n\n"
        news[index]["summary"] += (
            "受访者先解释了形成技术判断所依据的实验现象。"
            "随后他说明团队如何排除另一种可能解释。"
            "节目文字稿还记录了这一判断与现有路线的分歧。"
            "最后一段明确列出当前证据仍未覆盖的使用边界。"
            "嘉宾同时给出了团队下一轮验证准备采用的公开基准。"
            "主持人则追问这些结果能否推广到不同部署环境。"
        )
    for index in range(2, 7):
        news[index]["platform"] = "Blog"
    return news


PAPER_DOMAIN_QUOTAS = {
    "world_model": {"min": 3, "max": 5},
    "physical_ai": {"min": 4, "max": 6},
    "agent_models": {"min": 3, "max": 5},
    "infra_open_source": {"min": 2, "max": 4},
    "other": {"min": 1, "max": 3},
}


def _discovery_manifest(items: list[dict]) -> list[dict]:
    manifest = [
        {
            "primary_section": item["primary_section"],
            "url": item["url"],
            "source_detail": item["source_detail"],
            "publish_date": item["publish_date"],
        }
        for item in items
    ]
    existing_counts = {
        section: sum(1 for row in manifest if row["primary_section"] == section)
        for section in ("news", "technical", "paper")
    }
    targets = {"news": 50, "technical": 50, "paper": 60}
    for section, target in targets.items():
        for index in range(existing_counts[section], target):
            manifest.append(
                {
                    "primary_section": section,
                    "url": f"https://discovery.example.com/{section}/{index}",
                    "source_detail": "Discovery Source",
                    "publish_date": datetime.now(timezone.utc).date().isoformat(),
                }
            )
    return manifest


def test_codex_research_inbox_requires_verified_discovery_manifest(tmp_path):
    items = _balanced_news() + [_row(index, "project") for index in range(20)] + _balanced_papers()
    manifest = _discovery_manifest(items)
    path = tmp_path / "latest.json"
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "discovery_candidates": manifest,
        "items": items,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    collector_kwargs = {
        "minimum_items": 55,
        "minimum_papers": 15,
        "minimum_news": 20,
        "minimum_technical": 20,
        "minimum_discovery_candidates": 160,
        "minimum_discovered_by_section": {"news": 50, "technical": 50, "paper": 40},
    }

    collector = CodexResearchInboxCollector(str(path), **collector_kwargs)
    assert len(collector.collect()) == 55
    assert collector.fetch_diagnostics["discovery_quota_status"] == "passed"
    assert collector.fetch_diagnostics["discovery_candidate_count"] == 160
    assert collector.fetch_diagnostics["submitted_not_in_discovery_count"] == 0
    assert len(collector.fetch_diagnostics["inbox_sha256"]) == 64
    assert len(collector.fetch_diagnostics["discovery_manifest_sha256"]) == 64

    payload["discovery_candidates"][0] = {
        **payload["discovery_candidates"][0],
        "url": "https://discovery.example.com/news/replacement",
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    failing_collector = CodexResearchInboxCollector(str(path), **collector_kwargs)
    assert failing_collector.collect() == []
    assert failing_collector.fetch_diagnostics["discovery_quota_status"] == "failed"
    assert failing_collector.fetch_diagnostics["submitted_not_in_discovery_count"] == 1
    assert failing_collector.fetch_diagnostics["submitted_not_in_discovery_examples"] == [
        items[0]["url"]
    ]


def test_shared_collector_factory_applies_all_quality_constraints(tmp_path):
    collector = build_codex_research_inbox_collector(
        {
            "path": "research/latest.json",
            "minimum_items": 55,
            "minimum_papers": 15,
            "minimum_news": 20,
            "minimum_technical": 20,
            "technical_primary_source_ratio_min": 0.8,
            "minimum_discovery_candidates": 160,
            "minimum_discovered_by_section": {"news": 50, "technical": 50, "paper": 40},
            "minimum_submitted_by_section": {"news": 30, "technical": 27, "paper": 20},
            "minimum_key_number_items_by_section": {"news": 8, "technical": 10, "paper": 10},
            "required_schema_version": "codex-research-v3",
            "technical_category_quotas": {"training_data": 4},
            "technical_category_minimums": {"training_data": 2},
            "news_format_quotas": {"interview_or_podcast": 2, "blog": 5},
            "paper_domain_quotas": PAPER_DOMAIN_QUOTAS,
            "minimum_fresh_by_section": {"news": 15, "technical": 15, "paper": 12},
            "supplemental_max_age_hours_by_section": {"news": 168, "technical": 720, "paper": 720},
            "max_attribution_opener_count": 8,
            "max_cross_item_template_repeat_count": 0,
            "cross_item_template_similarity_threshold": 0.92,
        },
        root=tmp_path,
    )

    assert collector.inbox_path == tmp_path / "research/latest.json"
    assert collector.minimum_items == 55
    assert collector.minimum_news == 20
    assert collector.technical_primary_source_ratio_min == 0.8
    assert collector.minimum_discovery_candidates == 160
    assert collector.minimum_discovered_by_section == {
        "news": 50,
        "technical": 50,
        "paper": 40,
    }
    assert collector.minimum_submitted_by_section == {
        "news": 30,
        "technical": 27,
        "paper": 20,
    }
    assert collector.minimum_key_number_items_by_section == {
        "news": 8,
        "technical": 10,
        "paper": 10,
    }
    assert collector.required_schema_version == "codex-research-v3"
    assert collector.technical_category_quotas == {"training_data": 4}
    assert collector.technical_category_minimums == {"training_data": 2}
    assert collector.news_format_quotas == {"interview_or_podcast": 2, "blog": 5}
    assert collector.paper_domain_quotas == PAPER_DOMAIN_QUOTAS
    assert collector.minimum_fresh_by_section == {"news": 15, "technical": 15, "paper": 12}
    assert collector.supplemental_max_age_hours_by_section == {
        "news": 168,
        "technical": 720,
        "paper": 720,
    }
    assert collector.max_attribution_opener_count == 8
    assert collector.max_cross_item_template_repeat_count == 0
    assert collector.cross_item_template_similarity_threshold == 0.92


def test_research_inbox_blocks_cross_item_template_repetition(tmp_path):
    path = tmp_path / "latest.json"
    items = [_row(1, "news"), _row(2, "news")]
    items[1]["facts"]["who"] = "Second AI Lab"
    items[1]["source_detail"] = "Second AI Lab"
    items[1]["summary"] = items[1]["summary"].replace(
        "Example AI Lab",
        "Second AI Lab",
    )
    path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "items": items,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    collector = CodexResearchInboxCollector(
        str(path),
        minimum_items=2,
        minimum_news=2,
        minimum_technical=0,
        minimum_papers=0,
        technical_category_quotas={},
        technical_category_minimums={},
        max_cross_item_template_repeat_count=0,
        cross_item_template_similarity_threshold=0.92,
    )

    assert collector.collect() == []
    assert collector.fetch_diagnostics["cross_item_template_repeat_count"] == 1
    assert collector.fetch_diagnostics["cross_item_template_repeat_examples"][0][
        "similarity"
    ] == 1.0
    readiness = build_codex_research_readiness_summary(
        collector.fetch_diagnostics,
        {
            "minimum_news": 2,
            "minimum_technical": 0,
            "minimum_papers": 0,
            "max_cross_item_template_repeat_count": 0,
        },
    )
    assert "cross_item_template_repetition" in readiness["blockers"]


def test_required_v3_schema_rejects_legacy_inbox_before_ingest(tmp_path):
    path = tmp_path / "latest.json"
    path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "items": [_row(1, "news")],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    collector = CodexResearchInboxCollector(
        str(path),
        required_schema_version="codex-research-v3",
        minimum_items=1,
        minimum_news=1,
        minimum_technical=0,
        minimum_papers=0,
    )

    assert collector.collect() == []
    assert collector.fetch_diagnostics["quality_status"] == "schema_version_mismatch"
    assert collector.fetch_diagnostics["schema_version"] == ""
    assert collector.fetch_diagnostics["required_schema_version"] == "codex-research-v3"
    assert collector.fetch_diagnostics["schema_version_status"] == "failed"
    assert collector.fetch_diagnostics["fresh"] is True

    summary = build_codex_research_readiness_summary(
        collector.fetch_diagnostics,
        {"required_schema_version": "codex-research-v3"},
    )
    assert summary["ready_for_dry_run"] is False
    assert summary["schema_version"] == ""
    assert summary["required_schema_version"] == "codex-research-v3"
    assert summary["schema_version_status"] == "failed"
    assert "schema_version=missing" in summary["blockers"]
    assert "stale_inbox" not in summary["blockers"]
    assert "submission_quota" not in summary["blockers"]
    assert "discovery_quota" not in summary["blockers"]
    assert "accepted_section_minimums" not in summary["blockers"]
    assert "production_ready_status=not_checked" not in summary["blockers"]


def test_shared_collector_factory_honors_isolated_inbox_override(tmp_path, monkeypatch):
    isolated_path = tmp_path / "isolated" / "latest.json"
    monkeypatch.setenv("WEB_AGENT_CODEX_RESEARCH_INBOX_PATH", str(isolated_path))

    collector = build_codex_research_inbox_collector(
        {"path": "production/latest.json"},
        root=tmp_path,
    )

    assert collector.inbox_path == isolated_path


def test_production_inbox_freshness_window_covers_morning_research_lead_time():
    inbox_config = codex_research.load_inbox_config()

    assert 330 <= int(inbox_config["max_age_minutes"]) <= 360


def test_attribution_opener_pattern_ignores_entity_but_keeps_attribution_style():
    assert attribution_opener_pattern("Isaac Lab 团队表示，新版拆分了物理后端。") == "团队表示"
    assert attribution_opener_pattern("Google Research 团队表示，新方法改变检索训练。") == "团队表示"
    assert attribution_opener_pattern("新版先拆分物理后端，团队在说明中列出边界。") == ""


def test_research_inbox_blocks_overused_attribution_openers(tmp_path):
    rows = [_row(1, "news"), _row(2, "news")]
    for index, row in enumerate(rows, start=1):
        row["facts"]["who"] = f"示例实验室{index}"
        row["summary"] = (
            f"示例实验室{index}团队表示，新系统把任务规划、权限检查和工具执行拆成三个可验证阶段。"
            "原文进一步给出模块输入输出、对照流程和部署边界，使读者能够理解具体改动，而不是只看到产品名称。"
            "现有证据来自受控环境，尚未覆盖跨硬件长期负载，因此正文没有把发布方结果写成普遍结论。"
            "发布记录还说明旧流程需要人工在多个工具之间转移上下文，新方案则保留每一步的授权记录与失败原因。"
            "这些材料足以解释本次改动，但没有提供跨组织部署结果，因此没有进一步推断商业收益。"
        )
    path = tmp_path / "candidate.json"
    path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "items": rows,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    collector = CodexResearchInboxCollector(
        str(path),
        minimum_items=2,
        minimum_news=2,
        minimum_technical=0,
        minimum_papers=0,
        technical_category_quotas={},
        technical_category_minimums={},
        max_attribution_opener_count=1,
    )

    assert collector.collect() == []
    assert collector.fetch_diagnostics["attribution_opener_counts"] == {
        "团队表示": 2
    }
    assert collector.fetch_diagnostics["attribution_opener_overuse_count"] == 1
    summary = build_codex_research_readiness_summary(
        collector.fetch_diagnostics,
        {
            "minimum_news": 2,
            "minimum_technical": 0,
            "minimum_papers": 0,
        },
    )
    assert "attribution_opener_overuse" in summary["blockers"]


def test_ai_web_relevance_accepts_substantive_chinese_ai_copy():
    assert is_ai_web_content(
        "具身智能控制系统加入动作前预测",
        "系统使用世界模型预测物体状态，再把预测结果交给机器人策略执行。",
    )


def test_ai_web_relevance_accepts_model_training_system_release():
    assert is_ai_web_content(
        "Megatron Core v0.19.2 ships FP4 overlap support",
        "NVIDIA added Transformer Engine 2.18 and 1F1B all-to-all overlap for model parallel training.",
    )


def test_ai_web_relevance_accepts_language_model_knowledge_engineering():
    assert is_ai_web_content(
        "Poro 2 links medical terminology to a knowledge graph",
        "The language model uses structured clinical concepts to improve retrieval and terminology grounding.",
    )


def test_production_ready_overlap_gate_rejects_recent_sent_urls_and_events():
    items = [
        {"primary_section": "paper", "url": "https://arxiv.org/abs/2609.12345v2"},
        {
            "primary_section": "news",
            "url": "https://example.com/new-source",
            "facts": {"who": "OpenAI", "action": "发布", "target": "Agent Runtime 2"},
        },
    ]
    history = [
        {"primary_section": "paper", "url": "https://arxiv.org/abs/2609.12345v1"},
        {
            "primary_section": "technical",
            "url": "https://example.com/old-source",
            "facts": {"who": "OpenAI", "action": "发布", "target": "Agent Runtime 2"},
        },
    ]

    metrics = evaluate_sent_history_overlap(items, history)

    assert metrics["production_ready_status"] == "failed"
    assert metrics["sent_history_overlap_count"] == 2
    assert metrics["sent_history_overlap_by_section"] == {"paper": 1, "news": 1}


def test_event_identity_normalizes_release_synonyms_and_cosmetic_names():
    news = {
        "primary_section": "news",
        "url": "https://example.com/news/agent-runtime",
        "facts": {
            "who": "OpenAI 团队",
            "action": "推出",
            "target": "全新 Agent Runtime 2.0",
        },
    }
    technical = {
        "primary_section": "technical",
        "url": "https://example.com/docs/agent-runtime",
        "facts": {
            "who": "OpenAI",
            "action": "发布",
            "target": "Agent Runtime 2",
        },
    }

    assert CodexResearchInboxCollector._event_identity(news) == (
        CodexResearchInboxCollector._event_identity(technical)
    )


def test_event_identity_keeps_release_and_later_update_distinct():
    released = {
        "primary_section": "news",
        "facts": {"who": "OpenAI", "action": "发布", "target": "Agent Runtime 2"},
    }
    updated = {
        "primary_section": "technical",
        "facts": {"who": "OpenAI", "action": "升级", "target": "Agent Runtime 2"},
    }

    assert CodexResearchInboxCollector._event_identity(released) != (
        CodexResearchInboxCollector._event_identity(updated)
    )


def test_ai_web_relevance_rejects_unrelated_chinese_news():
    assert not is_ai_web_content(
        "城市公共交通调整周末运行时间",
        "公交部门将根据客流变化调整线路班次，并延长部分站点服务时间。",
    )


def test_ai_web_relevance_rejects_generic_graph_and_software_release():
    assert not is_ai_web_content(
        "Graph editor 2.1 release notes",
        "The desktop application adds new chart colors, keyboard shortcuts, and export options.",
    )


def test_codex_research_inbox_rejects_mechanically_repeated_public_copy(tmp_path):
    path = tmp_path / "latest.json"
    repeated_brand = _row(1, "news")
    repeated_brand["summary"] = (
        "funes 发布 funes 本地记忆项目，并说明数据库结构、检索接口和会话来源。"
        "项目把历史会话写入本地索引，再按任务上下文检索相关记录，同时保留原始出处。"
        "官方文档给出了安装步骤、存储格式和查询示例，也说明当前只验证了单机环境。"
        "这些信息足以判断实现边界，但跨设备同步和长期索引维护仍未提供测试结果。"
    )
    repeated_action = _row(2, "news")
    repeated_action["summary"] = (
        "团队发布产品发布说明，并列出新的权限边界、部署步骤和恢复条件。"
        "系统先校验调用者身份，再按照工具范围分配权限并记录每次操作。"
        "工程文档展示了失败回滚流程和审计日志格式，也限定了当前支持的部署环境。"
        "这些材料能够核对功能范围，但尚未包含跨区域负载下的性能测试。"
    )
    path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "items": [repeated_brand, repeated_action],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    collector = CodexResearchInboxCollector(
        str(path),
        minimum_items=2,
        minimum_papers=0,
        minimum_news=2,
        minimum_technical=0,
        technical_category_quotas={},
    )

    assert collector.collect() == []
    assert collector.fetch_diagnostics["bad_editorial_summary_count"] == 2
    assert collector.fetch_diagnostics["rejection_reason_counts_by_section"]["news"] == {
        "bad_editorial_summary": 2
    }


def test_codex_research_inbox_rejects_truncated_editorial_copy(tmp_path):
    path = tmp_path / "latest.json"
    row = _row(1, "news")
    row["summary"] = row["summary"].rstrip("。") + "，"
    path.write_text(
        json.dumps(
            {"generated_at": datetime.now(timezone.utc).isoformat(), "items": [row]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    collector = CodexResearchInboxCollector(
        str(path),
        minimum_items=1,
        minimum_news=1,
        minimum_papers=0,
        minimum_technical=0,
        technical_category_quotas={},
        news_format_quotas={},
    )

    assert collector.collect() == []
    assert collector.fetch_diagnostics["truncated_editorial_copy_count"] == 1
    assert collector.fetch_diagnostics["truncated_editorial_copy_examples"] == [
        {"title": row["title_cn"], "fields": ["summary"]}
    ]


def test_codex_research_inbox_accepts_fresh_balanced_payload(tmp_path):
    path = tmp_path / "latest.json"
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "items": _balanced_papers()
        + _balanced_news()
        + [_row(index, "project") for index in range(20)],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    collector = CodexResearchInboxCollector(
        str(path), minimum_items=55, minimum_papers=15, minimum_news=20, minimum_technical=20,
        news_format_quotas={"interview_or_podcast": 2, "blog": 5},
        paper_domain_quotas=PAPER_DOMAIN_QUOTAS,
        minimum_fresh_by_section={"news": 15, "technical": 15, "paper": 12},
    )
    items = collector.collect()

    assert len(items) == 55
    assert collector.fetch_diagnostics["quality_status"] == "passed"
    assert items[0]["model_used"] == "codex-automation"
    assert items[0]["analysis_version"] == "codex-research-v3"
    assert items[0]["_codex_research_analysis"]["analysis_version"] == "codex-research-v3"
    assert items[0]["_codex_research_analysis"]["facts"]["who"] == "Example AI Lab"
    assert set(items[0]["facts"]["editorial_source_hashes"]) == {
        "title_cn",
        "summary",
        "source_excerpt",
        "evidence_locator",
        "paper_plain_summary",
        "paper_technical_intro",
    }
    assert collector.fetch_diagnostics["technical_count"] == 20
    assert collector.fetch_diagnostics["submitted_item_count"] == 55
    assert collector.fetch_diagnostics["submitted_section_counts"] == {
        "news": 20,
        "technical": 20,
        "paper": 15,
    }
    assert collector.fetch_diagnostics["rejected_section_counts"] == {
        "news": 0,
        "technical": 0,
        "paper": 0,
        "unknown": 0,
    }
    assert collector.fetch_diagnostics["accepted_section_rates"] == {
        "news": 1.0,
        "technical": 1.0,
        "paper": 1.0,
    }
    assert collector.fetch_diagnostics["rejection_reason_counts_by_section"] == {
        "news": {},
        "technical": {},
        "paper": {},
        "unknown": {},
    }
    assert collector.fetch_diagnostics["news_format_quota_status"] == "passed"
    assert collector.fetch_diagnostics["paper_domain_quota_status"] == "passed"
    assert collector.fetch_diagnostics["freshness_quota_status"] == "passed"
    assert collector.fetch_diagnostics["summary_char_stats_by_section"]["news"]["count"] == 20
    assert collector.fetch_diagnostics["summary_char_stats_by_section"]["technical"]["count"] == 20
    assert collector.fetch_diagnostics["summary_char_stats_by_section"]["paper"]["count"] == 15
    assert collector.fetch_diagnostics["summary_paragraph_stats_by_section"]["news"]["max"] >= 2
    assert collector.fetch_diagnostics["accepted_content_type_counts"]["interview"] == 2
    assert collector.fetch_diagnostics["paper_plain_summary_char_stats"]["count"] == 15
    assert 90 <= collector.fetch_diagnostics["paper_plain_summary_char_stats"]["min"]
    assert collector.fetch_diagnostics["paper_plain_summary_char_stats"]["max"] <= 180
    assert collector.fetch_diagnostics["paper_technical_intro_char_stats"]["count"] == 15
    assert 160 <= collector.fetch_diagnostics["paper_technical_intro_char_stats"]["min"]
    assert collector.fetch_diagnostics["paper_technical_intro_char_stats"]["max"] <= 300
    assert {
        item["paper_domain_key"]
        for item in items
        if item["primary_section"] == "paper"
    } == {"world_model", "physical_ai", "agent_models", "infra_open_source", "other"}


def test_codex_research_inbox_enforces_technical_primary_source_ratio(tmp_path):
    path = tmp_path / "latest.json"
    technical_rows = [_row(index, "project") for index in range(5)]
    for index, row in enumerate(technical_rows):
        if index < 4:
            row["url"] = f"https://github.com/example/project-{index}/releases/tag/v1"
            row["platform"] = "GitHub"
        else:
            row["url"] = "https://techcrunch.com/example-secondary-report"
            row["platform"] = "News"
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "items": technical_rows,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    collector = CodexResearchInboxCollector(
        str(path),
        minimum_items=5,
        minimum_papers=0,
        minimum_news=0,
        minimum_technical=5,
        technical_primary_source_ratio_min=0.8,
        technical_category_quotas={},
    )
    items = collector.collect()

    assert len(items) == 5
    assert collector.fetch_diagnostics["technical_primary_source_count"] == 4
    assert collector.fetch_diagnostics["technical_primary_source_ratio"] == 0.8
    assert collector.fetch_diagnostics["technical_primary_source_status"] == "passed"

    technical_rows[3]["url"] = "https://venturebeat.com/example-secondary-report"
    technical_rows[3]["platform"] = "News"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    failing_collector = CodexResearchInboxCollector(
        str(path),
        minimum_items=5,
        minimum_papers=0,
        minimum_news=0,
        minimum_technical=5,
        technical_primary_source_ratio_min=0.8,
        technical_category_quotas={},
    )

    assert failing_collector.collect() == []
    assert failing_collector.fetch_diagnostics["technical_primary_source_count"] == 3
    assert failing_collector.fetch_diagnostics["technical_primary_source_ratio"] == 0.6
    assert failing_collector.fetch_diagnostics["technical_primary_source_status"] == "failed"
    assert failing_collector.fetch_diagnostics["quality_status"] == "underfilled"


def test_technical_category_targets_can_flex_above_hard_coverage_floor(tmp_path):
    path = tmp_path / "latest.json"
    categories = (
        ["embodied_world_model"] * 5
        + ["agent_systems"] * 5
        + ["training_data"] * 5
        + ["inference_deployment"] * 3
        + ["multimodal_architecture"] * 2
    )
    rows = [_row(index, "project") for index in range(20)]
    for row, category in zip(rows, categories):
        row["technical_category"] = category
    path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "items": rows,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    collector = CodexResearchInboxCollector(
        str(path),
        minimum_items=20,
        minimum_papers=0,
        minimum_news=0,
        minimum_technical=20,
        technical_category_quotas={
            "embodied_world_model": 6,
            "agent_systems": 4,
            "training_data": 4,
            "inference_deployment": 3,
            "multimodal_architecture": 3,
        },
        technical_category_minimums={
            "embodied_world_model": 3,
            "agent_systems": 2,
            "training_data": 2,
            "inference_deployment": 1,
            "multimodal_architecture": 1,
        },
    )

    items = collector.collect()

    assert len(items) == 20
    assert collector.fetch_diagnostics["quality_status"] == "passed"
    assert collector.fetch_diagnostics["technical_quota_status"] == "passed"
    assert collector.fetch_diagnostics["technical_target_status"] == "failed"
    assert set(collector.fetch_diagnostics["technical_category_target_underfilled"]) == {
        "embodied_world_model:5/6",
        "multimodal_architecture:2/3",
    }


def test_codex_editorial_copy_hashes_survive_database_and_render_pipeline(tmp_path):
    path = tmp_path / "latest.json"
    row = _row(1, "news")
    row["summary"] = row["summary"].replace(
        "读者不打开链接也能先理解核心变化。",
        "读者不打开链接也能先理解核心变化。\n\n",
    ).replace(
        "涉及数字时会同时写明测试条件",
        "\n\n涉及数字时会同时写明测试条件",
    )
    path.write_text(
        json.dumps(
            {"generated_at": datetime.now(timezone.utc).isoformat(), "items": [row]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    collector = CodexResearchInboxCollector(
        str(path),
        minimum_items=1,
        minimum_news=1,
        minimum_papers=0,
        minimum_technical=0,
        technical_category_quotas={},
    )
    collected = collector.collect()
    assert len(collected) == 1

    article = collected[0]
    analysis = article["_codex_research_analysis"]
    assert analysis["summary"].count("\n\n") == 2
    assert set(analysis["facts"]["editorial_paragraph_hashes"]) == {"summary"}
    db = Database(str(tmp_path / "fidelity.db"))
    assert db.insert_article(article)
    db.update_article_processing(
        url=article["url"],
        summary=analysis["summary"],
        score=analysis["score"],
        keywords=analysis["keywords"],
        category=analysis["category"],
        title_cn=analysis["title_cn"],
        facts=analysis["facts"],
        evidence_quality=analysis["evidence_quality"],
        information_density=analysis["information_density"],
        model_used=analysis["model_used"],
        analysis_version=analysis["analysis_version"],
        quality_flags=analysis["quality_flags"],
    )
    persisted = db.get_articles_for_run("", processed_only=True)[0]
    rendered_item = enrich_editorial_fields(persisted)
    render_key = editorial_item_render_key(rendered_item)
    rendered_facts = rendered_item.get("facts") or {}
    html = (
        f"<html><body><article data-v11-item-key='{render_key}'>"
        + rendered_item["title_cn"]
        + " "
        + rendered_item["analysis_body"]
        + " "
        + str(rendered_facts.get("source_excerpt") or "")
        + " "
        + str(rendered_facts.get("evidence_locator") or "")
        + "</article></body></html>"
    )
    metrics = scan_final_html_quality(
        html,
        {
            "must_read": [rendered_item],
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

    assert metrics["v11_content_fidelity_missing_count"] == 0
    assert metrics["v11_editorial_source_hash_missing_count"] == 0
    assert metrics["v11_editorial_source_mismatch_count"] == 0, metrics[
        "v11_editorial_source_mismatch_examples"
    ]


def test_codex_research_inbox_preserves_paper_paragraphs(tmp_path):
    path = tmp_path / "latest.json"
    row = _row(1, "paper")
    row["summary"] = row["summary"].replace("读者不打开链接也能先理解核心变化。", "读者不打开链接也能先理解核心变化。\n\n")
    row["paper_plain_summary"] = row["paper_plain_summary"].replace("它先预测", "\n\n它先预测")
    row["paper_technical_intro"] = row["paper_technical_intro"].replace("实验在", "\n\n实验在")
    path.write_text(
        json.dumps(
            {"generated_at": datetime.now(timezone.utc).isoformat(), "items": [row]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    collector = CodexResearchInboxCollector(
        str(path),
        minimum_items=1,
        minimum_news=0,
        minimum_papers=1,
        minimum_technical=0,
        technical_category_quotas={},
        paper_domain_quotas={},
    )

    analysis = collector.collect()[0]["_codex_research_analysis"]

    assert "\n\n" in analysis["summary"]
    assert "\n\n" in analysis["facts"]["paper_plain_summary"]
    assert "\n\n" in analysis["facts"]["paper_technical_intro"]
    assert set(analysis["facts"]["editorial_paragraph_hashes"]) == {
        "summary",
        "paper_plain_summary",
        "paper_technical_intro",
    }


def test_codex_research_inbox_requires_submission_buffer_before_quality_filtering(tmp_path):
    path = tmp_path / "latest.json"
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "items": _balanced_papers()
        + _balanced_news()
        + [_row(index, "project") for index in range(20)],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    collector = CodexResearchInboxCollector(
        str(path),
        minimum_items=55,
        minimum_papers=15,
        minimum_news=20,
        minimum_technical=20,
        minimum_submitted_by_section={"news": 30, "technical": 27, "paper": 20},
        news_format_quotas={"interview_or_podcast": 2, "blog": 5},
        paper_domain_quotas=PAPER_DOMAIN_QUOTAS,
        minimum_fresh_by_section={"news": 15, "technical": 15, "paper": 12},
    )

    assert collector.collect() == []
    assert collector.fetch_diagnostics["quality_status"] == "underfilled"
    assert collector.fetch_diagnostics["submission_quota_status"] == "failed"
    assert collector.fetch_diagnostics["submission_underfilled"] == [
        "news:20/30",
        "technical:20/27",
        "paper:15/20",
    ]


def test_codex_research_inbox_submission_buffer_absorbs_rejected_candidates(tmp_path):
    path = tmp_path / "latest.json"
    accepted_news = _balanced_news()
    accepted_technical = [_row(index, "project") for index in range(20)]
    accepted_papers = _balanced_papers()
    duplicate_news = [copy.deepcopy(item) for item in accepted_news[:10]]
    duplicate_technical = [copy.deepcopy(item) for item in accepted_technical[:7]]
    duplicate_papers = [copy.deepcopy(item) for item in accepted_papers[:5]]
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "items": (
            accepted_news
            + accepted_technical
            + accepted_papers
            + duplicate_news
            + duplicate_technical
            + duplicate_papers
        ),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    collector = CodexResearchInboxCollector(
        str(path),
        minimum_items=55,
        minimum_papers=15,
        minimum_news=20,
        minimum_technical=20,
        minimum_submitted_by_section={"news": 30, "technical": 27, "paper": 20},
        news_format_quotas={"interview_or_podcast": 2, "blog": 5},
        paper_domain_quotas=PAPER_DOMAIN_QUOTAS,
        minimum_fresh_by_section={"news": 15, "technical": 15, "paper": 12},
    )
    items = collector.collect()

    assert len(items) == 55
    assert collector.fetch_diagnostics["quality_status"] == "passed"
    assert collector.fetch_diagnostics["submission_quota_status"] == "passed"
    assert collector.fetch_diagnostics["submitted_section_counts"] == {
        "news": 30,
        "technical": 27,
        "paper": 20,
    }
    assert collector.fetch_diagnostics["rejected_section_counts"] == {
        "news": 10,
        "technical": 7,
        "paper": 5,
        "unknown": 0,
    }
    assert collector.fetch_diagnostics["news_count"] == 20
    assert collector.fetch_diagnostics["technical_count"] == 20
    assert collector.fetch_diagnostics["paper_count"] == 15


def test_codex_research_inbox_enforces_paper_domain_quotas(tmp_path):
    path = tmp_path / "latest.json"
    papers = [_row(index, "paper") for index in range(15)]
    for paper in papers:
        paper["topic"] = "Physical AI / Robotics"
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "items": papers
        + [_row(index, "news") for index in range(20)]
        + [_row(index, "project") for index in range(20)],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    collector = CodexResearchInboxCollector(str(path), paper_domain_quotas=PAPER_DOMAIN_QUOTAS)

    assert collector.collect() == []
    assert collector.fetch_diagnostics["paper_domain_quota_status"] == "failed"
    assert "world_model:0/3" in collector.fetch_diagnostics["paper_domain_underfilled"]
    assert "physical_ai:15/6" in collector.fetch_diagnostics["paper_domain_exceeded"]


def test_codex_research_inbox_enforces_news_format_quotas(tmp_path):
    path = tmp_path / "latest.json"
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "items": [_row(index, "news") for index in range(20)],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    collector = CodexResearchInboxCollector(
        str(path),
        minimum_items=20,
        minimum_news=20,
        minimum_papers=0,
        minimum_technical=0,
        technical_category_quotas={},
        news_format_quotas={"interview_or_podcast": 2, "blog": 5},
    )

    assert collector.collect() == []
    assert collector.fetch_diagnostics["news_format_quota_status"] == "failed"
    assert "interview_or_podcast:0/2" in collector.fetch_diagnostics["news_format_underfilled"]


def test_codex_research_inbox_enforces_minimum_fresh_sources_per_section(tmp_path):
    path = tmp_path / "latest.json"
    groups = {
        "paper": _balanced_papers(),
        "news": [_row(index, "news") for index in range(20)],
        "technical": [_row(index, "project") for index in range(20)],
    }
    old_date = (datetime.now(timezone.utc) - timedelta(days=10)).date().isoformat()
    for section, supplemental_count in (("paper", 4), ("news", 6), ("technical", 6)):
        for item in groups[section][:supplemental_count]:
            item["publish_date"] = old_date
            item["quality_flags"] = ["supplemental_older_source"]
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "items": groups["paper"] + groups["news"] + groups["technical"],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    collector = CodexResearchInboxCollector(
        str(path),
        paper_domain_quotas=PAPER_DOMAIN_QUOTAS,
        minimum_fresh_by_section={"news": 15, "technical": 15, "paper": 12},
    )

    assert collector.collect() == []
    assert collector.fetch_diagnostics["freshness_quota_status"] == "failed"
    assert set(collector.fetch_diagnostics["freshness_underfilled"]) == {
        "news:14/15",
        "technical:14/15",
        "paper:11/12",
    }


def test_codex_research_inbox_rejects_stale_or_underfilled_payload(tmp_path):
    path = tmp_path / "latest.json"
    path.write_text(
        json.dumps(
            {
                "generated_at": (datetime.now(timezone.utc) - timedelta(hours=8)).isoformat(),
                "items": [_row(1, "paper")],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    collector = CodexResearchInboxCollector(str(path), max_age_minutes=240)

    assert collector.collect() == []
    assert collector.fetch_diagnostics["quality_status"] == "stale_inbox"


def test_codex_research_inbox_enforces_technical_direction_quotas(tmp_path):
    path = tmp_path / "latest.json"
    technical = [_row(index, "project") for index in range(20)]
    for item in technical:
        item["technical_category"] = "agent_systems"
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "items": [_row(index, "paper") for index in range(15)]
        + [_row(index, "news") for index in range(20)]
        + technical,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    collector = CodexResearchInboxCollector(str(path))

    assert collector.collect() == []
    assert collector.fetch_diagnostics["technical_quota_status"] == "failed"
    assert "embodied_world_model:0/6" in collector.fetch_diagnostics["technical_category_underfilled"]


def test_codex_research_inbox_requires_fresh_source_or_supplemental_label(tmp_path):
    path = tmp_path / "latest.json"
    old_news = _row(1, "news")
    old_news["publish_date"] = (datetime.now(timezone.utc) - timedelta(days=4)).date().isoformat()
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "items": [old_news],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    collector = CodexResearchInboxCollector(
        str(path), minimum_items=1, minimum_papers=0, minimum_news=1, minimum_technical=0,
        technical_category_quotas={},
    )
    assert collector.collect() == []
    assert collector.fetch_diagnostics["stale_source_count"] == 1
    assert collector.fetch_diagnostics["rejection_reason_counts_by_section"]["news"] == {
        "stale_source": 1
    }

    old_news["quality_flags"] = ["supplemental_older_source"]
    path.write_text(json.dumps({"generated_at": datetime.now(timezone.utc).isoformat(), "items": [old_news]}, ensure_ascii=False), encoding="utf-8")
    collector = CodexResearchInboxCollector(
        str(path), minimum_items=1, minimum_papers=0, minimum_news=1, minimum_technical=0,
        technical_category_quotas={},
    )
    assert len(collector.collect()) == 1


def test_codex_research_inbox_rejects_supplemental_news_older_than_seven_days(tmp_path):
    path = tmp_path / "latest.json"
    old_news = _row(1, "news")
    old_news["publish_date"] = (
        datetime.now(timezone.utc) - timedelta(days=8)
    ).date().isoformat()
    old_news["quality_flags"] = ["supplemental_older_source"]
    path.write_text(
        json.dumps(
            {"generated_at": datetime.now(timezone.utc).isoformat(), "items": [old_news]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    collector = CodexResearchInboxCollector(
        str(path),
        minimum_items=1,
        minimum_papers=0,
        minimum_news=1,
        minimum_technical=0,
        technical_category_quotas={},
    )

    assert collector.collect() == []
    assert collector.fetch_diagnostics["supplemental_too_old_count"] == 1
    assert collector.fetch_diagnostics["rejection_reason_counts_by_section"]["news"] == {
        "supplemental_too_old": 1
    }


def test_codex_research_inbox_rejects_publish_dates_in_the_future(tmp_path):
    path = tmp_path / "latest.json"
    future_news = _row(1, "news")
    future_news["publish_date"] = (
        datetime.now(timezone.utc) + timedelta(days=1)
    ).isoformat()
    path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "items": [future_news],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    collector = CodexResearchInboxCollector(
        str(path),
        minimum_items=1,
        minimum_papers=0,
        minimum_news=1,
        minimum_technical=0,
        technical_category_quotas={},
    )

    assert collector.collect() == []
    assert collector.fetch_diagnostics["future_publish_date_count"] == 1
    assert collector.fetch_diagnostics["rejection_reason_counts_by_section"]["news"] == {
        "future_publish_date": 1
    }


def test_codex_research_inbox_rejects_short_news_and_technical_bodies(tmp_path):
    path = tmp_path / "latest.json"
    short_news = _row(1, "news")
    short_news["summary"] = "新闻只有标题式简介，缺少完整证据、背景、影响和限制。" * 3
    short_technical = _row(2, "project")
    short_technical["summary"] = "技术内容只罗列名词，没有解释机制、输入输出、架构和验证结果。" * 4
    path.write_text(
        json.dumps(
            {"generated_at": datetime.now(timezone.utc).isoformat(), "items": [short_news, short_technical]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    collector = CodexResearchInboxCollector(
        str(path),
        minimum_items=2,
        minimum_papers=0,
        minimum_news=1,
        minimum_technical=1,
        technical_category_quotas={},
    )

    assert collector.collect() == []
    assert collector.fetch_diagnostics["short_summary_count"] == 2


def test_codex_research_inbox_rejects_overlong_news_and_technical_bodies(tmp_path):
    path = tmp_path / "latest.json"
    long_news = _row(1, "news")
    long_news["summary"] += "这段重复扩写没有增加新的事实或证据。" * 8
    long_technical = _row(2, "project")
    long_technical["summary"] += "这段重复扩写没有增加新的机制或实验信息。" * 8
    path.write_text(
        json.dumps(
            {"generated_at": datetime.now(timezone.utc).isoformat(), "items": [long_news, long_technical]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    collector = CodexResearchInboxCollector(
        str(path),
        minimum_items=2,
        minimum_papers=0,
        minimum_news=1,
        minimum_technical=1,
        technical_category_quotas={},
    )

    assert collector.collect() == []
    assert collector.fetch_diagnostics["long_summary_count"] == 2


def test_codex_research_inbox_rejects_english_dominant_editorial_copy(tmp_path):
    collector = CodexResearchInboxCollector(str(tmp_path / "missing.json"))
    row = _row(1, "news")
    row["summary"] = (
        "The AI research team released a new agent runtime with explicit permission boundaries and tool logs. "
        "The system routes each action through a policy layer and records the decision before execution. "
        "Internal tests cover several workflows, but the source does not yet establish broad customer adoption."
    )

    assert collector._normalize_item(row) is None
    assert collector.fetch_diagnostics["non_chinese_summary_count"] == 1


def test_codex_research_inbox_rejects_generic_title_before_content_is_persisted(tmp_path):
    path = tmp_path / "latest.json"
    row = _row(1, "news")
    row["title_cn"] = "AI领域新进展"
    path.write_text(
        json.dumps(
            {"generated_at": datetime.now(timezone.utc).isoformat(), "items": [row]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    collector = CodexResearchInboxCollector(
        str(path),
        minimum_items=1,
        minimum_news=1,
        minimum_papers=0,
        minimum_technical=0,
        technical_category_quotas={},
    )

    assert collector.collect() == []
    assert collector.fetch_diagnostics["bad_editorial_title_count"] == 1
    assert collector.fetch_diagnostics["rejection_reason_counts_by_section"]["news"] == {
        "bad_editorial_title": 1
    }


def test_codex_research_inbox_rejects_repeated_sentence_padding(tmp_path):
    collector = CodexResearchInboxCollector(str(tmp_path / "missing.json"))
    row = _row(2, "news")
    row["summary"] = "团队发布了新的智能体运行框架，并说明权限检查、工具日志和人工审批流程。" * 6

    assert collector._normalize_item(row) is None
    assert collector.fetch_diagnostics["repetitive_summary_count"] == 1


def test_codex_research_inbox_rejects_single_run_on_sentence(tmp_path):
    collector = CodexResearchInboxCollector(str(tmp_path / "missing.json"))
    row = _row(3, "news")
    row["summary"] = (
        "团队发布新的智能体运行框架，系统先检查工具权限，再记录输入输出，随后提交人工审批，"
        "工程文档列出部署参数、失败恢复、日志格式和访问控制，测试覆盖代码审查、数据检索、"
        "客户支持和内部知识库，并给出延迟、成功率、硬件环境和对照版本，但尚未提供长期客户使用结果，"
        "团队还解释了故障回退、权限撤销、任务中断和审计记录之间的关系，"
        "因此当前只能确认公开实现和实验范围，不能把发布方测试直接写成普遍生产效果。"
    )

    assert collector._normalize_item(row) is None
    assert collector.fetch_diagnostics["insufficient_sentence_count"] == 1


def test_codex_research_inbox_rejects_summary_number_without_source_support(tmp_path):
    collector = CodexResearchInboxCollector(str(tmp_path / "missing.json"))
    row = _row(4, "news")
    row["summary"] = row["summary"].replace(
        "读者不打开链接也能先理解核心变化。",
        "发布方还宣称处理延迟降低35%，但邮件必须保留这一数字对应的原文依据。",
    )

    assert collector._normalize_item(row) is None
    assert collector.fetch_diagnostics["unsupported_summary_numeric_count"] == 1
    assert collector.fetch_diagnostics["unsupported_summary_numeric_examples"] == [
        {
            "primary_section": "news",
            "title": row["title_cn"],
            "unsupported_tokens": ["percent:35"],
        }
    ]


def test_codex_research_inbox_normalizes_chinese_and_arabic_metric_claims(tmp_path):
    collector = CodexResearchInboxCollector(str(tmp_path / "missing.json"))

    assert collector._metric_tokens(
        "成功率由65%提高到95%，状态展开快10倍，训练使用20小时数据，共16项任务。"
    ) == collector._metric_tokens(
        "成功率由百分之六十五提高到百分之九十五，状态展开快十倍，"
        "训练使用二十小时数据，共十六项任务。"
    )


def test_codex_research_inbox_normalizes_metric_units_and_currency_scales(tmp_path):
    collector = CodexResearchInboxCollector(str(tmp_path / "missing.json"))

    assert collector._metric_tokens(
        "指标提高12个百分点，增加8ms延迟，吞吐为1,000 tokens/s，预算1万美元。"
    ) == collector._metric_tokens(
        "指标提高十二个百分点，增加八毫秒延迟，吞吐为一千 token/s，预算一万美元。"
    )


def test_codex_research_inbox_accepts_supported_chinese_metric_wording(tmp_path):
    collector = CodexResearchInboxCollector(str(tmp_path / "missing.json"))
    row = _row(5, "news")
    row["source_excerpt"] = "原文报告成功率由65%提高到95%，状态展开速度提升10倍。"
    row["facts"]["evidence"] = [row["source_excerpt"]]
    row["summary"] = row["summary"].replace(
        "读者不打开链接也能先理解核心变化。",
        "原文报告成功率由百分之六十五提高到百分之九十五，状态展开速度提升十倍。",
    )

    assert collector._normalize_item(row) is not None
    assert collector.fetch_diagnostics["unsupported_summary_numeric_count"] == 0


def test_codex_research_v3_requires_key_number_contract_for_supported_metrics(tmp_path):
    collector = CodexResearchInboxCollector(str(tmp_path / "missing.json"))
    collector.require_key_numbers = True
    row = _row(50, "news")
    row["source_excerpt"] = "公告披露新基金规模为5000万美元，并面向三地的早期企业。"
    row["facts"]["evidence"] = [row["source_excerpt"]]

    assert collector._normalize_item(row) is None
    assert collector.fetch_diagnostics["key_number_contract_missing_count"] == 1


def test_codex_research_v3_schema_enables_key_number_contract_during_collection(tmp_path):
    path = tmp_path / "latest.json"
    row = _row(53, "news")
    path.write_text(
        json.dumps(
            {
                "schema_version": "codex-research-v3",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "items": [row],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    collector = CodexResearchInboxCollector(
        str(path),
        minimum_items=1,
        minimum_news=1,
        minimum_technical=0,
        minimum_papers=0,
    )

    assert collector.collect() == []
    assert collector.fetch_diagnostics["schema_version"] == "codex-research-v3"
    assert collector.fetch_diagnostics["key_number_contract_missing_count"] == 1


def test_codex_research_v3_enforces_key_number_coverage_quota(tmp_path):
    path = tmp_path / "latest.json"
    rows = [_row(index, "news") for index in range(2)]
    for row in rows:
        row["facts"]["key_numbers"] = []
    path.write_text(
        json.dumps(
            {
                "schema_version": "codex-research-v3",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "items": rows,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    collector = CodexResearchInboxCollector(
        str(path),
        minimum_items=2,
        minimum_news=2,
        minimum_technical=0,
        minimum_papers=0,
        minimum_key_number_items_by_section={"news": 1},
    )

    assert collector.collect() == []
    assert collector.fetch_diagnostics["key_number_item_counts"] == {
        "news": 0,
        "technical": 0,
        "paper": 0,
    }
    assert collector.fetch_diagnostics["key_number_underfilled"] == ["news:0/1"]
    assert collector.fetch_diagnostics["key_number_quota_status"] == "failed"


def test_codex_research_v3_accepts_complete_key_number_coverage(tmp_path):
    path = tmp_path / "latest.json"
    news_rows = _balanced_news()
    technical_rows = [_row(index, "project") for index in range(20)]
    paper_rows = _balanced_papers()

    for index, row in enumerate(news_rows):
        row["facts"]["key_numbers"] = []
        if 2 <= index < 10:
            evidence = "原始来源报告该项目已覆盖40家公司，并说明统计口径。"
            row["source_excerpt"] = evidence
            row["facts"]["evidence"] = [evidence]
            row["facts"]["key_numbers"] = ["覆盖40家公司"]
            row["summary"] = row["summary"].replace(
                "读者不打开链接也能先理解核心变化。",
                "原始来源报告项目覆盖40家公司，读者无需打开链接也能先理解量级。",
            )

    for index, row in enumerate(technical_rows):
        row["facts"]["key_numbers"] = []
        if index < 10:
            evidence = "工程文档报告受控部署覆盖12个客户，并给出配置边界。"
            row["source_excerpt"] = evidence
            row["facts"]["evidence"] = [evidence]
            row["facts"]["key_numbers"] = ["覆盖12个客户"]
            row["summary"] = row["summary"].replace(
                "读者不打开链接也能先理解核心变化。",
                "工程文档报告受控部署覆盖12个客户，读者无需打开链接也能理解验证量级。",
            )

    for row in paper_rows:
        row["facts"]["key_numbers"] = [
            "任务成功率提高12个百分点",
            "推理延迟增加8毫秒",
        ]

    path.write_text(
        json.dumps(
            {
                "schema_version": "codex-research-v3",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "items": news_rows + technical_rows + paper_rows,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    collector = CodexResearchInboxCollector(
        str(path),
        minimum_items=55,
        minimum_news=20,
        minimum_technical=20,
        minimum_papers=15,
        minimum_key_number_items_by_section={"news": 8, "technical": 10, "paper": 10},
        technical_category_quotas={},
        technical_category_minimums={},
        news_format_quotas={},
        paper_domain_quotas={},
        minimum_fresh_by_section={},
    )

    assert len(collector.collect()) == 55
    assert collector.fetch_diagnostics["key_number_item_counts"] == {
        "news": 8,
        "technical": 10,
        "paper": 15,
    }
    assert collector.fetch_diagnostics["key_number_quota_status"] == "passed"


def test_codex_research_v3_requires_key_numbers_in_public_copy(tmp_path):
    collector = CodexResearchInboxCollector(str(tmp_path / "missing.json"))
    collector.require_key_numbers = True
    row = _row(51, "news")
    row["source_excerpt"] = "公告披露新基金规模为5000万美元，并面向三地的早期企业。"
    row["facts"]["evidence"] = [row["source_excerpt"]]
    row["facts"]["key_numbers"] = ["基金规模5000万美元"]

    assert collector._normalize_item(row) is None
    assert collector.fetch_diagnostics["key_number_public_copy_missing_count"] == 1
    assert collector.fetch_diagnostics["key_number_public_copy_missing_examples"][0][
        "missing_tokens"
    ] == ["usd:50000000"]


def test_codex_research_v3_accepts_key_numbers_with_evidence_and_public_copy(tmp_path):
    collector = CodexResearchInboxCollector(str(tmp_path / "missing.json"))
    collector.require_key_numbers = True
    row = _row(52, "news")
    row["source_excerpt"] = "公告披露新基金规模为5000万美元，并面向三地的早期企业。"
    row["facts"]["evidence"] = [row["source_excerpt"]]
    row["facts"]["key_numbers"] = ["基金规模5000万美元"]
    row["summary"] = row["summary"].replace(
        "读者不打开链接也能先理解核心变化。",
        "公告同时披露基金规模为5000万美元，读者无需打开链接也能先把握投入量级。",
    )

    assert collector._normalize_item(row) is not None
    assert collector.fetch_diagnostics["key_number_public_copy_missing_count"] == 0


def test_codex_research_v3_does_not_count_hidden_paper_summary_as_public_copy(tmp_path):
    collector = CodexResearchInboxCollector(str(tmp_path / "missing.json"))
    collector.require_key_numbers = True
    row = _row(54, "paper")
    row["facts"]["key_numbers"] = ["任务成功率提高12个百分点"]
    row["paper_plain_summary"] = row["paper_plain_summary"].replace(
        "任务成功率提高了12个百分点",
        "任务成功率出现明显提高",
    )
    row["paper_technical_intro"] = row["paper_technical_intro"].replace(
        "任务成功率提高12个百分点，同时推理延迟增加8毫秒",
        "任务成功率出现明显提高，同时推理延迟略有增加",
    )

    assert collector._normalize_item(row) is None
    assert collector.fetch_diagnostics["key_number_public_copy_missing_count"] == 1
    assert collector.fetch_diagnostics["key_number_public_copy_missing_examples"][0][
        "missing_tokens"
    ] == ["percentage_point:12"]


def test_v11_real_key_number_omissions_are_rejected_before_rendering(tmp_path):
    fixture_path = Path(__file__).parent / "fixtures" / "v11_key_number_omission_samples.json"
    samples = json.loads(fixture_path.read_text(encoding="utf-8"))["items"]

    for index, sample in enumerate(samples, start=70):
        collector = CodexResearchInboxCollector(str(tmp_path / f"missing-{index}.json"))
        collector.require_key_numbers = True
        row = _row(index, "news")
        row["title_cn"] = sample["title_cn"]
        row["url"] = sample["url"]
        row["source_excerpt"] = sample["source_excerpt"]
        row["facts"]["evidence"] = [sample["source_excerpt"]]
        row["facts"]["key_numbers"] = sample["key_numbers"]

        assert collector._normalize_item(row) is None
        assert collector.fetch_diagnostics["key_number_public_copy_missing_count"] == 1
        assert collector.fetch_diagnostics["key_number_public_copy_missing_examples"][0][
            "missing_tokens"
        ] == sample["expected_missing_tokens"]


def test_codex_research_inbox_rejects_unsupported_chinese_metric_wording(tmp_path):
    collector = CodexResearchInboxCollector(str(tmp_path / "missing.json"))
    row = _row(6, "news")
    row["source_excerpt"] = "原文的公开测试报告成功率由65%提高到95%，并交代了对应测试条件。"
    row["facts"]["evidence"] = [row["source_excerpt"]]
    row["summary"] = row["summary"].replace(
        "读者不打开链接也能先理解核心变化。",
        "整理稿声称成功率提高到百分之九十九，但原文没有给出这一结果。",
    )

    assert collector._normalize_item(row) is None
    assert collector.fetch_diagnostics["unsupported_summary_numeric_count"] == 1
    assert collector.fetch_diagnostics["unsupported_summary_numeric_examples"][0][
        "unsupported_tokens"
    ] == ["percent:99"]


def test_codex_research_inbox_does_not_treat_chinese_ordinals_as_metrics(tmp_path):
    collector = CodexResearchInboxCollector(str(tmp_path / "missing.json"))

    assert collector._metric_tokens("第一阶段先训练预测模块，第二阶段再冻结策略模型。") == set()


def test_v11_real_production_failures_are_rejected_before_rendering(tmp_path):
    fixture_path = Path(__file__).parent / "fixtures" / "v11_real_rejection_samples.json"
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    path = tmp_path / "latest.json"
    path.write_text(
        json.dumps(
            {"generated_at": datetime.now(timezone.utc).isoformat(), "items": payload["items"]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    collector = CodexResearchInboxCollector(
        str(path),
        minimum_items=4,
        minimum_papers=1,
        minimum_news=2,
        minimum_technical=1,
        technical_category_quotas={},
    )

    assert collector.collect() == []
    assert collector.fetch_diagnostics["short_summary_count"] == 3
    assert collector.fetch_diagnostics["structured_fact_missing_count"] == 1


def test_v11_real_acceptance_samples_survive_inbox_normalization(tmp_path):
    fixture_path = Path(__file__).parent / "fixtures" / "v11_real_acceptance_samples.json"
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    items = copy.deepcopy(payload["items"])
    current_publish_date = datetime.now(timezone.utc).date().isoformat()
    for item in items:
        item["publish_date"] = current_publish_date

    path = tmp_path / "latest.json"
    path.write_text(
        json.dumps(
            {"generated_at": datetime.now(timezone.utc).isoformat(), "items": items},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    collector = CodexResearchInboxCollector(
        str(path),
        minimum_items=4,
        minimum_papers=1,
        minimum_news=2,
        minimum_technical=1,
        minimum_submitted_by_section={},
        technical_category_quotas={},
        news_format_quotas={},
        paper_domain_quotas={},
        minimum_fresh_by_section={},
    )

    collected = collector.collect()

    assert len(collected) == 4
    assert collector.fetch_diagnostics["quality_status"] == "passed"
    by_url = {item["url"]: item for item in collected}
    for source in items:
        normalized = by_url[source["url"]]
        analysis = normalized["_codex_research_analysis"]
        assert analysis["title_cn"] == source["title_cn"]
        assert " ".join(analysis["summary"].split()) == " ".join(source["summary"].split())
        assert analysis["facts"]["source_excerpt"] == source["source_excerpt"]
        assert analysis["facts"]["evidence_locator"] == source["evidence_locator"]
        if source["content_type"] == "paper":
            assert analysis["facts"]["paper_plain_summary"] == source["paper_plain_summary"]
            assert analysis["facts"]["paper_technical_intro"] == source["paper_technical_intro"]

    db = Database(str(tmp_path / "real-fidelity.db"))
    for article in collected:
        analysis = article["_codex_research_analysis"]
        assert db.insert_article(article)
        db.update_article_processing(
            url=article["url"],
            summary=analysis["summary"],
            score=analysis["score"],
            keywords=analysis["keywords"],
            category=analysis["category"],
            title_cn=analysis["title_cn"],
            facts=analysis["facts"],
            evidence_quality=analysis["evidence_quality"],
            information_density=analysis["information_density"],
            model_used=analysis["model_used"],
            analysis_version=analysis["analysis_version"],
            quality_flags=analysis["quality_flags"],
        )
    rendered_items = [
        enrich_editorial_fields(item)
        for item in db.get_articles_for_run("", processed_only=True)
    ]
    layers = {
        "must_read": [
            item for item in rendered_items if item["content_type"] != "paper"
        ],
        "physical_ai": [],
        "watch": [],
        "featured_papers": [],
        "paper_appendix": [
            item for item in rendered_items if item["content_type"] == "paper"
        ],
        "research": [],
        "brief": [],
    }
    html_parts = []
    for item in rendered_items:
        facts = item.get("facts") or {}
        render_key = editorial_item_render_key(item)
        body = (
            f"{item.get('paper_plain_summary', '')} {item.get('paper_technical_intro', '')}"
            if item["content_type"] == "paper"
            else item["analysis_body"]
        )
        html_parts.append(
            f'<article data-v11-item-key="{render_key}">{item["title_cn"]}'
            f'<span class="v11-claim-label">事实类型</span>'
            f'<div class="v11-source-note">{facts.get("source_excerpt", "")} · '
            f'{facts.get("evidence_locator", "")}</div>{body}</article>'
        )
    metrics = scan_final_html_quality(
        "<html><body>" + "".join(html_parts) + "</body></html>",
        layers,
        quality_config={},
        report_config={
            "product_mode": "intelligence_v11_editorial_library",
            "design_version": "v10-learning-digest",
            "min_visible_news_count": 0,
            "min_visible_technical_count": 0,
            "min_visible_paper_count": 0,
            "paper_technical_intro_min_count": 0,
            "total_visible_chars_min": 0,
            "total_visible_chars_max": 100000,
            "source_focus_limit": 10,
            "topic_focus_limit": 10,
        },
    )
    assert metrics["v11_content_fidelity_missing_count"] == 0
    assert metrics["v11_editorial_source_hash_missing_count"] == 0
    assert metrics["v11_editorial_source_mismatch_count"] == 0, metrics[
        "v11_editorial_source_mismatch_examples"
    ]


def test_codex_research_inbox_requires_source_excerpt_and_evidence_locator(tmp_path):
    path = tmp_path / "latest.json"
    missing_excerpt = _row(1, "news")
    missing_excerpt.pop("source_excerpt")
    missing_locator = _row(2, "news")
    missing_locator.pop("evidence_locator")
    path.write_text(
        json.dumps(
            {"generated_at": datetime.now(timezone.utc).isoformat(), "items": [missing_excerpt, missing_locator]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    collector = CodexResearchInboxCollector(
        str(path),
        minimum_items=2,
        minimum_papers=0,
        minimum_news=2,
        minimum_technical=0,
        technical_category_quotas={},
    )

    assert collector.collect() == []
    assert collector.fetch_diagnostics["missing_source_evidence_count"] == 2


def test_codex_research_inbox_rejects_placeholder_source_excerpt(tmp_path):
    path = tmp_path / "latest.json"
    item = _row(1, "news")
    item["source_excerpt"] = "原文有所说明。"
    path.write_text(
        json.dumps(
            {"generated_at": datetime.now(timezone.utc).isoformat(), "items": [item]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    collector = CodexResearchInboxCollector(
        str(path),
        minimum_items=1,
        minimum_news=1,
        minimum_papers=0,
        minimum_technical=0,
        technical_category_quotas={},
    )

    assert collector.collect() == []
    assert collector.fetch_diagnostics["short_source_excerpt_count"] == 1


def test_codex_research_inbox_deduplicates_tracking_and_arxiv_version_variants(tmp_path):
    path = tmp_path / "latest.json"
    first = _row(1, "paper")
    first["url"] = "https://www.arxiv.org/abs/2609.12345v1?utm_source=newsletter#abstract"
    second = _row(2, "paper")
    second["url"] = "https://arxiv.org/abs/2609.12345v2"
    path.write_text(
        json.dumps(
            {"generated_at": datetime.now(timezone.utc).isoformat(), "items": [first, second]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    collector = CodexResearchInboxCollector(
        str(path),
        minimum_items=2,
        minimum_papers=2,
        minimum_news=0,
        minimum_technical=0,
        technical_category_quotas={},
    )

    assert collector.collect() == []
    assert collector.fetch_diagnostics["duplicate_url_count"] == 1
    assert collector.fetch_diagnostics["submitted_section_counts"]["paper"] == 2
    assert collector.fetch_diagnostics["rejected_section_counts"]["paper"] == 1
    assert collector.fetch_diagnostics["accepted_section_rates"]["paper"] == 0.5
    assert collector.fetch_diagnostics["rejection_reason_counts_by_section"]["paper"] == {
        "duplicate_url": 1
    }
    assert collector.fetch_diagnostics["duplicate_url_examples"] == [
        {
            "rejected_section": "paper",
            "rejected_title": second["title_cn"],
            "rejected_url": second["url"],
            "matched_section": "paper",
            "matched_title": first["title_cn"],
            "matched_url": first["url"],
        }
    ]


def test_codex_research_inbox_rejects_items_without_verifiable_publish_date(tmp_path):
    path = tmp_path / "latest.json"
    row = _row(1, "news")
    row["publish_date"] = ""
    path.write_text(
        json.dumps(
            {"generated_at": datetime.now(timezone.utc).isoformat(), "items": [row]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    collector = CodexResearchInboxCollector(
        str(path),
        minimum_items=1,
        minimum_papers=0,
        minimum_news=1,
        minimum_technical=0,
        technical_category_quotas={},
    )

    assert collector.collect() == []
    assert collector.fetch_diagnostics["missing_publish_date_count"] == 1


def test_codex_research_inbox_requires_section_specific_structured_facts(tmp_path):
    path = tmp_path / "latest.json"
    news = _row(1, "news")
    news["facts"]["who"] = ""
    technical = _row(2, "project")
    technical["facts"]["method"] = ""
    paper = _row(3, "paper")
    for key in ("dataset_or_benchmark", "metric_result", "baseline", "limitation", "code_or_project"):
        paper["facts"][key] = ""
    path.write_text(
        json.dumps(
            {"generated_at": datetime.now(timezone.utc).isoformat(), "items": [news, technical, paper]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    collector = CodexResearchInboxCollector(
        str(path),
        minimum_items=3,
        minimum_papers=1,
        minimum_news=1,
        minimum_technical=1,
        technical_category_quotas={},
    )

    assert collector.collect() == []
    assert collector.fetch_diagnostics["structured_fact_missing_count"] == 3


def test_codex_research_inbox_requires_context_for_numeric_technical_results(tmp_path):
    path = tmp_path / "latest.json"
    technical = _row(2, "project")
    technical["facts"]["metric_result"] = "吞吐提升35%"
    technical["facts"].pop("dataset_or_benchmark", None)
    technical["facts"].pop("deployment_context", None)
    technical["facts"].pop("baseline", None)
    technical["facts"]["architecture"] = "路由层、批处理队列和执行池"
    path.write_text(
        json.dumps(
            {"generated_at": datetime.now(timezone.utc).isoformat(), "items": [technical]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    collector = CodexResearchInboxCollector(
        str(path),
        minimum_items=1,
        minimum_papers=0,
        minimum_news=0,
        minimum_technical=1,
        technical_category_quotas={},
    )

    assert collector.collect() == []
    assert collector.fetch_diagnostics["numeric_context_missing_count"] == 1


def test_codex_research_inbox_requires_complete_technical_contract(tmp_path):
    path = tmp_path / "latest.json"
    technical = _row(2, "project")
    technical["facts"]["baseline"] = ""
    technical["facts"]["limitation"] = ""
    path.write_text(
        json.dumps(
            {"generated_at": datetime.now(timezone.utc).isoformat(), "items": [technical]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    collector = CodexResearchInboxCollector(
        str(path),
        minimum_items=1,
        minimum_papers=0,
        minimum_news=0,
        minimum_technical=1,
        technical_category_quotas={},
    )

    assert collector.collect() == []
    assert collector.fetch_diagnostics["technical_contract_missing_count"] == 1
    assert collector.fetch_diagnostics["technical_contract_missing_examples"] == [
        {"title": technical["title_cn"], "missing": ["baseline", "limitation"]}
    ]


def test_codex_research_inbox_requires_complete_paper_contract(tmp_path):
    path = tmp_path / "latest.json"
    paper = _row(2, "paper")
    paper["facts"]["baseline"] = ""
    paper["facts"]["limitation"] = ""
    path.write_text(
        json.dumps(
            {"generated_at": datetime.now(timezone.utc).isoformat(), "items": [paper]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    collector = CodexResearchInboxCollector(
        str(path),
        minimum_items=1,
        minimum_papers=1,
        minimum_news=0,
        minimum_technical=0,
        technical_category_quotas={},
        paper_domain_quotas={},
    )

    assert collector.collect() == []
    assert collector.fetch_diagnostics["paper_contract_missing_count"] == 1
    assert collector.fetch_diagnostics["paper_contract_missing_examples"] == [
        {"title": paper["title_cn"], "missing": ["baseline", "limitation"]}
    ]


def test_codex_research_inbox_enforces_claim_type_for_interviews_and_papers(tmp_path):
    path = tmp_path / "latest.json"
    interview = _row(1, "news")
    interview["content_type"] = "interview"
    interview["claim_type"] = "verified_fact"
    interview["summary"] += "受访者进一步解释了技术判断、公开论据和仍有争议的限制。" * 5
    paper = _row(2, "paper")
    paper["claim_type"] = "official_claim"
    path.write_text(
        json.dumps(
            {"generated_at": datetime.now(timezone.utc).isoformat(), "items": [interview, paper]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    collector = CodexResearchInboxCollector(
        str(path),
        minimum_items=2,
        minimum_papers=1,
        minimum_news=1,
        minimum_technical=0,
        technical_category_quotas={},
    )

    assert collector.collect() == []
    assert collector.fetch_diagnostics["claim_type_error_count"] == 2


def test_codex_research_inbox_requires_claim_calibrated_wording(tmp_path):
    path = tmp_path / "latest.json"
    row = _row(1, "news")
    row["claim_type"] = "official_claim"
    row["facts"]["claim_type"] = "official_claim"
    row["summary"] = row["summary"].replace("发布方自报数据", "性能数据")
    path.write_text(
        json.dumps(
            {"generated_at": datetime.now(timezone.utc).isoformat(), "items": [row]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    collector = CodexResearchInboxCollector(
        str(path),
        minimum_items=1,
        minimum_news=1,
        minimum_papers=0,
        minimum_technical=0,
        technical_category_quotas={},
        news_format_quotas={},
    )

    assert collector.collect() == []
    assert collector.fetch_diagnostics["claim_language_mismatch_count"] == 1
    assert collector.fetch_diagnostics["claim_language_mismatch_examples"] == [
        {
            "primary_section": "news",
            "title": row["title_cn"],
            "claim_type": "official_claim",
            "summary_opening": row["summary"][:120],
            "required_attribution": "公司称／团队介绍／发布方披露／官方公告显示",
        }
    ]


def test_codex_research_inbox_accepts_team_introduction_attribution(tmp_path):
    path = tmp_path / "latest.json"
    row = _row(1, "news")
    row["claim_type"] = "official_claim"
    row["facts"]["claim_type"] = "official_claim"
    row["summary"] = row["summary"].replace("发布方自报数据", "SGLang 团队介绍")
    path.write_text(
        json.dumps(
            {"generated_at": datetime.now(timezone.utc).isoformat(), "items": [row]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    collector = CodexResearchInboxCollector(
        str(path),
        minimum_items=1,
        minimum_news=1,
        minimum_papers=0,
        minimum_technical=0,
        technical_category_quotas={},
        news_format_quotas={},
    )

    assert len(collector.collect()) == 1
    assert collector.fetch_diagnostics["claim_language_mismatch_count"] == 0


def test_codex_research_inbox_requires_precise_interview_locator(tmp_path):
    path = tmp_path / "latest.json"
    interview = _row(0, "news")
    interview["content_type"] = "interview"
    interview["claim_type"] = "interview_opinion"
    interview["facts"]["claim_type"] = "interview_opinion"
    interview["summary"] += "\n\n" + (
        "受访者补充说明了判断所依据的实验现象。"
        "文字稿随后记录了团队排除替代解释的过程。"
        "最后一节列出了目前证据仍未覆盖的部署边界。"
        "嘉宾还说明下一轮准备采用公开基准重新验证结论。"
        "主持人继续追问这一结果能否推广到不同硬件环境。"
    )
    interview["evidence_locator"] = "节目页面"
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "items": [interview],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    collector = CodexResearchInboxCollector(str(path), minimum_items=1)

    assert collector.collect() == []
    assert collector.fetch_diagnostics["interview_locator_missing_count"] == 1

    interview["evidence_locator"] = "文字稿 12:40-15:10"
    payload["items"] = [interview]
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    collector = CodexResearchInboxCollector(
        str(path),
        minimum_items=1,
        minimum_news=1,
        minimum_papers=0,
        minimum_technical=0,
        technical_category_quotas={},
    )

    assert len(collector.collect()) == 1
    assert collector.fetch_diagnostics["interview_locator_missing_count"] == 0


def test_codex_research_inbox_rejects_interview_metadata_as_evidence(tmp_path):
    path = tmp_path / "latest.json"
    interview = _balanced_news()[0]
    metadata = "节目页面显示本期于九月二十日上线，完整节目时长六十分钟。"
    interview["source_excerpt"] = metadata
    interview["facts"]["source_excerpt"] = metadata
    interview["facts"]["evidence"] = [metadata]
    path.write_text(
        json.dumps(
            {"generated_at": datetime.now(timezone.utc).isoformat(), "items": [interview]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    collector = CodexResearchInboxCollector(
        str(path),
        minimum_items=1,
        minimum_news=1,
        minimum_papers=0,
        minimum_technical=0,
        technical_category_quotas={},
        news_format_quotas={},
    )

    assert collector.collect() == []
    assert collector.fetch_diagnostics[
        "interview_evidence_substance_missing_count"
    ] == 1


def test_codex_research_inbox_rejects_unbroken_interview_copy(tmp_path):
    path = tmp_path / "latest.json"
    interview = _balanced_news()[0]
    interview["summary"] = " ".join(interview["summary"].split())
    path.write_text(
        json.dumps(
            {"generated_at": datetime.now(timezone.utc).isoformat(), "items": [interview]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    collector = CodexResearchInboxCollector(
        str(path),
        minimum_items=1,
        minimum_news=1,
        minimum_papers=0,
        minimum_technical=0,
        technical_category_quotas={},
        news_format_quotas={},
    )

    assert collector.collect() == []
    assert collector.fetch_diagnostics["paragraph_structure_missing_count"] == 1


def test_codex_research_inbox_rejects_generic_opening_before_substance(tmp_path):
    path = tmp_path / "latest.json"
    row = _row(1, "news")
    row["summary"] = (
        "本段只是宽泛背景，读者尚不知道具体主体、对象和实际变化，也没有得到可核验的信息。\n\n"
        + row["summary"]
    )
    path.write_text(
        json.dumps(
            {"generated_at": datetime.now(timezone.utc).isoformat(), "items": [row]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    collector = CodexResearchInboxCollector(
        str(path),
        minimum_items=1,
        minimum_news=1,
        minimum_papers=0,
        minimum_technical=0,
        technical_category_quotas={},
        news_format_quotas={},
    )

    assert collector.collect() == []
    assert collector.fetch_diagnostics["opening_substance_missing_count"] == 1


def test_codex_research_inbox_rejects_same_event_across_news_and_technical_sections(tmp_path):
    path = tmp_path / "latest.json"
    news = _row(1, "news")
    technical = _row(2, "project")
    news["facts"].update({"who": "Example AI Lab", "action": "发布", "target": "Orion Agent 2.0"})
    technical["facts"].update({"who": "Example AI Lab", "action": "发布", "target": "Orion Agent 2.0"})
    path.write_text(
        json.dumps(
            {"generated_at": datetime.now(timezone.utc).isoformat(), "items": [news, technical]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    collector = CodexResearchInboxCollector(
        str(path),
        minimum_items=2,
        minimum_papers=0,
        minimum_news=1,
        minimum_technical=1,
        technical_category_quotas={},
    )

    assert collector.collect() == []
    assert collector.fetch_diagnostics["duplicate_event_count"] == 1
    assert collector.fetch_diagnostics["duplicate_event_examples"] == [
        {
            "rejected_section": "technical",
            "rejected_title": technical["title_cn"],
            "rejected_url": technical["url"],
            "matched_section": "news",
            "matched_title": news["title_cn"],
            "matched_url": news["url"],
        }
    ]


def test_codex_research_processor_mode_does_not_require_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    processor = LLMProcessor({"provider": "codex_automation", "model": "Codex Automation", "api_mode": "research_inbox"})

    assert not processor.enabled
    assert processor.health_snapshot()["status"] == "research_inbox"


def test_codex_research_report_item_preserves_curated_text():
    processor = LLMProcessor(
        {"provider": "codex_automation", "model": "Codex Automation", "api_mode": "research_inbox"}
    )
    article = _row(21, "news")
    article.update(
        {
            "title": "funes: Local Memory for Coding Agents, Built on Lance",
            "title_cn": "funes 为编码智能体提供可追溯的本地会话检索",
            "summary": "funes 将编码智能体历史会话写入本地数据库，并保留原始出处。",
            "summary_preview": "编码智能体获得可追溯的本地会话检索",
            "category": "开源生态",
            "model_used": "codex-automation",
        }
    )

    prepared = processor.prepare_report_item(article)

    assert prepared["title_cn"] == article["title_cn"]
    assert prepared["summary"] == article["summary"]
    assert prepared["summary_preview"] == article["summary_preview"]
    assert prepared["facts"] == article["facts"]
    assert prepared["model_used"] == "codex-automation"
