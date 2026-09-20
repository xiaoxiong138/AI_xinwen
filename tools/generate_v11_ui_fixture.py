from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import (
    compact_email_html,
    prepare_v11_email_delivery_volumes,
    scan_v11_delivery_volume_fidelity,
)
from src.editorial_engine import enrich_editorial_fields
from src.generator import ReportGenerator


OUTPUT_DIR = ROOT / "artifacts" / "v11_ui_fixture"
REAL_SAMPLE_PATH = ROOT / "tests" / "fixtures" / "v11_real_acceptance_samples.json"


def load_real_samples() -> list[dict]:
    payload = json.loads(REAL_SAMPLE_PATH.read_text(encoding="utf-8"))
    rows = payload.get("items") if isinstance(payload, dict) else payload
    samples = []
    for index, raw in enumerate(rows or [], start=1):
        item = dict(raw)
        facts = dict(item.get("facts") or {})
        section = str(item.get("primary_section") or facts.get("primary_section") or "news")
        item.setdefault("id", f"real-fixture-{section}-{index}")
        item.setdefault("canonical_url", item.get("url"))
        item.setdefault("publish_date", "2026-09-19T12:00:00+08:00")
        item.setdefault("source_tier", "official" if section != "paper" else "research")
        item.setdefault("model_used", "codex-automation")
        item.setdefault("analysis_version", "codex-research-v2")
        item.setdefault("evidence_quality", 0.86)
        item.setdefault("information_density", 0.84)
        item.setdefault("analysis_body", item.get("summary"))
        facts.setdefault("primary_section", section)
        facts.setdefault("claim_type", item.get("claim_type"))
        facts.setdefault("source_excerpt", item.get("source_excerpt"))
        facts.setdefault("evidence_locator", item.get("evidence_locator"))
        item["facts"] = facts
        samples.append(enrich_editorial_fields(item))
    return samples


def build_update(section: str, index: int) -> dict:
    if section == "news":
        body = (
            f"测试机构{index}发布了一项企业智能体更新，把工具权限、人工审批和运行记录接入同一工作流。"
            "官方说明给出了管理员配置步骤，并展示任务在调用外部系统前如何暂停等待确认。"
            "这次变化的重点不是增加聊天入口，而是把智能体执行过程纳入既有权限体系；目前公开材料仍未提供大规模客户部署数据。"
        )
        content_type = "news"
        title = f"测试机构{index}为企业智能体增加可审计权限控制"
        claim_type = "official_claim"
    else:
        body = (
            f"工程团队{index}把推理服务拆成请求调度、批次编排和缓存复用三个阶段，并公开每个阶段的输入输出关系。"
            "系统先按上下文长度和显存余量组织动态批次，再复用已计算的键值缓存，避免相同前缀反复执行预填充。"
            "基准测试同时报告硬件、并发数和延迟口径，因此可以把收益定位到调度与缓存机制，而不是模型规模变化；跨硬件部署仍需重新校准批次策略。"
        )
        content_type = "project"
        title = f"工程团队{index}公开动态批处理与缓存复用架构"
        claim_type = "verified_fact"
    excerpt = "原始技术文档列出了权限或调度机制、验证条件以及当前尚未覆盖的部署边界。"
    return enrich_editorial_fields({
        "id": f"fixture-{section}-{index}",
        "title": title,
        "title_cn": title,
        "url": f"https://example.com/{section}/{index}",
        "canonical_url": f"https://example.com/{section}/{index}",
        "source_detail": f"测试来源 {index}",
        "source_tier": "official",
        "content_type": content_type,
        "primary_section": section,
        "publish_date": "2026-09-20T02:00:00+08:00",
        "claim_type": claim_type,
        "summary": body,
        "analysis_body": body,
        "model_used": "codex-automation",
        "analysis_version": "codex-research-v2",
        "evidence_quality": 0.86,
        "information_density": 0.84,
        "facts": {
            "primary_section": section,
            "claim_type": claim_type,
            "who": f"测试团队 {index}",
            "action": "发布",
            "target": title,
            "method": "把执行过程拆成可检查阶段，并保留每一步的状态记录",
            "deployment_context": "企业工作流或推理服务",
            "evidence": [excerpt],
            "source_excerpt": excerpt,
            "evidence_locator": "官方技术文档第 2 节",
        },
        "source_excerpt": excerpt,
        "evidence_locator": "官方技术文档第 2 节",
    })


def build_paper(index: int) -> dict:
    plain = (
        "这篇论文研究机器人面对持续移动物体时，旧策略容易依据已经过时的画面做动作。"
        "作者先预测物体下一时刻的位置，再把预测状态交给动作模型，而不是直接从当前画面生成控制指令。"
        "对照实验显示，这个前置预测步骤能够改善动态任务的成功率。"
    )
    technical = (
        "方法采用轻量状态预测模块与冻结动作模型串联，预测模块根据当前图像和语言指令生成未来状态表示。"
        "实验在动态操作基准上与直接动作预测基线比较，平均成功率提高十二个百分点，并通过消融实验确认收益来自未来状态约束。"
        "论文尚未验证长时间遮挡和多机器人协同，因此结论主要适用于当前基准设置。"
    )
    excerpt = "实验表 2 报告动态操作任务平均成功率相对直接动作预测基线提高十二个百分点。"
    title = f"论文{index}提出预测后行动的动态机器人世界模型"
    return enrich_editorial_fields({
        "id": f"fixture-paper-{index}",
        "title": title,
        "title_cn": title,
        "url": f"https://arxiv.org/abs/2609.{index:05d}",
        "canonical_url": f"https://arxiv.org/abs/2609.{index:05d}",
        "source_detail": "arXiv",
        "source_tier": "research",
        "content_type": "paper",
        "primary_section": "paper",
        "publish_date": "2026-09-19T12:00:00+00:00",
        "claim_type": "research_result",
        "summary": plain,
        "paper_plain_summary": plain,
        "paper_technical_intro": technical,
        "model_used": "codex-automation",
        "analysis_version": "codex-research-v2",
        "evidence_quality": 0.88,
        "information_density": 0.87,
        "facts": {
            "primary_section": "paper",
            "claim_type": "research_result",
            "who": f"论文团队 {index}",
            "action": "提出",
            "target": "预测后行动的机器人控制方法",
            "method": "先预测未来状态，再由冻结动作模型选择控制指令",
            "dataset_or_benchmark": "动态操作基准",
            "metric_result": "平均成功率提高十二个百分点",
            "baseline": "直接动作预测基线",
            "limitation": "尚未覆盖长时间遮挡和多机器人协同",
            "evidence": [excerpt],
            "source_excerpt": excerpt,
            "evidence_locator": "论文实验表 2 与消融章节",
            "paper_plain_summary": plain,
            "paper_technical_intro": technical,
        },
        "source_excerpt": excerpt,
        "evidence_locator": "论文实验表 2 与消融章节",
    })


def main() -> int:
    report_config = {
        "product_mode": "intelligence_v11_editorial_library",
        "design_version": "v11-editorial-library",
        "email_split_enabled": True,
        "email_html_max_bytes": 104448,
        "news_section_limit": 24,
        "technical_section_limit": 24,
        "paper_featured_limit": 10,
        "paper_appendix_limit": 15,
        "editorial_decision_limit": 6,
    }
    generator = ReportGenerator(
        design_version="v11-editorial-library",
        report_config=report_config,
    )
    real_samples = load_real_samples()
    real_news = [item for item in real_samples if item.get("primary_section") == "news"]
    real_technical = [item for item in real_samples if item.get("primary_section") == "technical"]
    real_papers = [item for item in real_samples if item.get("primary_section") == "paper"]
    news = real_news[:24] + [
        build_update("news", index)
        for index in range(1, 25 - min(len(real_news), 24))
    ]
    technical = real_technical[:24] + [
        build_update("technical", index)
        for index in range(1, 25 - min(len(real_technical), 24))
    ]
    papers = real_papers[:15] + [
        build_paper(index)
        for index in range(1, 16 - min(len(real_papers), 15))
    ]
    layers = {
        "must_read": news[:6],
        "physical_ai": [],
        "watch": news[6:] + technical,
        "featured_papers": papers[:10],
        "paper_appendix": papers[10:],
        "research": [],
        "brief": [],
    }
    decorated = generator._decorate_layers(layers)
    all_items = news + technical + papers
    report_summary = {"edition_counts": {"news": 24, "technical": 24, "paper": 15}}
    full_html = compact_email_html(generator.generate_html(
        papers=papers,
        updates=news + technical,
        mixed_items=all_items,
        report_summary=report_summary,
        layered_updates=layers,
        title="AI Frontier Intelligence Daily · V11 UI Fixture",
    ))

    def render_volume(index: int, volume: dict) -> str:
        volume_items = volume["items"]
        return generator.generate_html(
            papers=[item for item in volume_items if item.get("content_type") == "paper"],
            updates=[item for item in volume_items if item.get("content_type") != "paper"],
            mixed_items=volume_items,
            report_summary=report_summary,
            layered_updates=volume["layers"],
            title=f"AI Frontier Intelligence Daily · V11 UI Fixture · 第 {index} 卷",
        )

    volumes, split_applied = prepare_v11_email_delivery_volumes(
        full_html,
        decorated,
        report_config,
        render_volume,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, volume in enumerate(volumes, start=1):
        path = OUTPUT_DIR / f"v11_fixture_part{index}.html"
        path.write_text(volume["html"], encoding="utf-8")
        paths.append(path.as_posix())
    fidelity_metrics = scan_v11_delivery_volume_fidelity(
        volumes,
        quality_config={},
        report_config=report_config,
    )
    content_fidelity_missing_counts = fidelity_metrics[
        "email_delivery_volume_content_fidelity_missing_counts"
    ]
    print(json.dumps({
        "split_applied": split_applied,
        "volume_count": len(volumes),
        "volume_sizes": [volume["size_bytes"] for volume in volumes],
        "content_fidelity_missing_counts": content_fidelity_missing_counts,
        "real_sample_count": len(real_samples),
        "real_sample_counts": {
            "news": len(real_news),
            "technical": len(real_technical),
            "paper": len(real_papers),
        },
        "paths": paths,
    }, ensure_ascii=False, indent=2))
    return 1 if any(content_fidelity_missing_counts) else 0


if __name__ == "__main__":
    raise SystemExit(main())
