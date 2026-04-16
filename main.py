from __future__ import annotations

import os
import multiprocessing as mp
import json
import re
import socket
import sys
import copy
from collections import Counter
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import yaml
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.collectors import ArxivCollector, HuggingFaceCollector, RSSCollector, WebSearchCollector
from src.database import Database
from src.generator import ReportGenerator
from src.notifier import EmailNotifier
from src.processors import LLMProcessor
from src.relevance import infer_impact_tag, is_low_signal_update, normalize_text, score_preference_boost, score_update_quality

EVENT_STOPWORDS = {
    "latest",
    "today",
    "breaking",
    "report",
    "reports",
    "news",
    "new",
    "ai",
    "artificial",
    "intelligence",
    "the",
    "for",
    "with",
    "and",
    "from",
}
OFFICIAL_HOST_HINTS = (
    "openai.com",
    "anthropic.com",
    "deepmind.google",
    "google.com",
    "huggingface.co",
    "nvidia.com",
    "aws.amazon.com",
    "microsoft.com",
    "meta.com",
)


def load_config():
    try:
        with open("config.yaml", "r", encoding="utf-8") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print("Error: config.yaml not found.")
        sys.exit(1)


def safe_console_text(text: str) -> str:
    encoding = sys.stdout.encoding or "utf-8"
    return (text or "").encode(encoding, errors="replace").decode(encoding, errors="replace")


def apply_runtime_profile(config: Dict[str, Any], profile: str) -> Dict[str, Any]:
    normalized = str(profile or "").strip().lower()
    if not normalized:
        return config

    runtime_config = copy.deepcopy(config)
    runtime_section = runtime_config.setdefault("runtime", {})
    runtime_section["profile"] = normalized

    if normalized != "validation_fast":
        return runtime_config

    sources = runtime_config.setdefault("sources", {})
    arxiv_config = sources.setdefault("arxiv", {})
    topic_limits = dict(arxiv_config.get("topic_limits", {}) or {})
    if topic_limits:
        first_topic, first_limit = next(iter(topic_limits.items()))
        arxiv_config["topic_limits"] = {first_topic: min(int(first_limit), 2)}
        topic_queries = dict(arxiv_config.get("topic_queries", {}) or {})
        if first_topic in topic_queries:
            arxiv_config["topic_queries"] = {first_topic: topic_queries[first_topic]}
    arxiv_config["candidate_pool"] = min(int(arxiv_config.get("candidate_pool", 160)), 40)
    arxiv_config["days_back"] = 1
    arxiv_config["fallback_days"] = [1]

    rss_config = sources.setdefault("rss", {})
    rss_feeds = list(rss_config.get("feeds", []) or [])
    rss_config["feeds"] = [{**feed, "max_entries": min(int(feed.get("max_entries", 15)), 5)} for feed in rss_feeds[:3]]
    rss_config["days_back"] = 1

    web_search_config = sources.setdefault("web_search", {})
    searches = list(web_search_config.get("searches", []) or [])
    web_search_config["searches"] = [
        {**search, "max_results": min(int(search.get("max_results", 8)), 4)}
        for search in searches[:3]
    ]
    web_search_config["days_back"] = 1
    web_search_config["fallback_days"] = [1]

    report_config = runtime_config.setdefault("report", {})
    report_config["paper_limit"] = min(int(report_config.get("paper_limit", 15)), 4)
    report_config["web_limit"] = min(int(report_config.get("web_limit", 20)), 6)
    report_config["min_web_items"] = min(int(report_config.get("min_web_items", 20)), 4)
    report_config["paper_backfill_hours_ladder"] = []
    report_config["web_backfill_hours_ladder"] = []

    archive_config = runtime_config.setdefault("archive", {})
    archive_config["enabled"] = False
    archive_config["report_dir"] = "archive/validation"

    alerts_config = runtime_config.setdefault("alerts", {})
    alerts_config["enabled"] = False
    alerts_config["send_separate_alert"] = False

    trends_config = runtime_config.setdefault("trends", {})
    trends_config["enabled"] = False

    runtime_section["max_unprocessed_items"] = 8
    runtime_section["max_analysis_backfill_items"] = 0
    runtime_section["skip_paper_enrichment"] = True
    return runtime_config


def resolve_delivery_outcome(
    notification_sent: bool,
    notification_skipped: bool,
    notification_dry_run: bool,
) -> Dict[str, Any]:
    delivery_status = (
        "sent"
        if notification_sent
        else "dry_run"
        if notification_dry_run
        else "skipped"
        if notification_skipped
        else "failed"
    )
    status = (
        "success"
        if notification_sent
        else "dry_run"
        if notification_dry_run
        else "warning"
        if notification_skipped
        else "notification_failed"
    )
    return {
        "success": notification_sent or notification_skipped or notification_dry_run,
        "status": status,
        "retryable": not notification_sent and not notification_skipped and not notification_dry_run,
        "delivery_status": delivery_status,
    }


def build_suppressed_alert_summary(reason: str) -> Dict[str, Any]:
    return {
        "needs_alert": False,
        "issues": [],
        "suppressed": True,
        "reason": reason,
    }


def merge_unique_articles(*groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    seen_urls = set()
    for group in groups:
        for item in group:
            url = item.get("url")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            merged.append(item)
    return merged


def parse_hour_ladder(value: Any, default: List[int]) -> List[int]:
    if isinstance(value, list):
        ladder = []
        for item in value:
            try:
                hours = int(item)
            except (TypeError, ValueError):
                continue
            if hours > 0 and hours not in ladder:
                ladder.append(hours)
        return ladder or default
    try:
        hours = int(value)
        return [hours] if hours > 0 else default
    except (TypeError, ValueError):
        return default


def _collector_worker(collector: Any, queue: Any, network_timeout: int) -> None:
    try:
        socket.setdefaulttimeout(network_timeout)
        queue.put({"items": collector.collect()})
    except Exception as exc:
        queue.put({"error": str(exc)})


def run_collector_with_timeout(
    collector: Any,
    timeout_seconds: int,
    network_timeout: int,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    context = mp.get_context("spawn")
    queue = context.Queue()
    process = context.Process(target=_collector_worker, args=(collector, queue, network_timeout))
    process.start()
    process.join(timeout_seconds)

    if process.is_alive():
        process.terminate()
        process.join(5)
        queue.close()
        return [], f"Timed out after {timeout_seconds} seconds"

    payload: Dict[str, Any] = {}
    if not queue.empty():
        payload = queue.get()
    queue.close()

    if process.exitcode not in (0, None) and not payload:
        return [], f"Collector process exited with code {process.exitcode}"
    if payload.get("error"):
        return [], str(payload["error"])
    return list(payload.get("items") or []), None


def collector_label(collector: Any) -> str:
    return str(getattr(collector, "label", collector.__class__.__name__))


def build_collector_summary(collector_runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    success = [item for item in collector_runs if item.get("status") == "success"]
    fresh_items = sum(int(item.get("inserted_count", 0)) for item in collector_runs)
    timeout_count = sum(1 for item in collector_runs if item.get("status") == "timeout")
    failed_count = sum(1 for item in collector_runs if item.get("status") == "error")
    status_text = (
        f"本轮共运行 {len(collector_runs)} 个采集单元，成功 {len(success)} 个，"
        f"超时 {timeout_count} 个，失败 {failed_count} 个，新入库 {fresh_items} 条。"
    )
    return {
        "status_text": status_text,
        "fresh_items": fresh_items,
        "success_count": len(success),
        "timeout_count": timeout_count,
        "failed_count": failed_count,
        "rows": collector_runs,
    }


def filter_updates_for_report(
    updates: List[Dict[str, Any]],
    target_count: int,
    minimum_count: int,
    source_preferences: Optional[Dict[str, Any]] = None,
    preference_config: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    ranked: List[Dict[str, Any]] = []
    for item in updates:
        quality_score = score_update_quality(
            item.get("title", ""),
            item.get("summary", "") or item.get("content", ""),
            item.get("url", ""),
            item.get("platform", ""),
            item.get("source_detail", ""),
            item.get("category", ""),
            source_preferences=source_preferences,
        )
        preference_score = score_preference_boost(item, preference_config)
        candidate = dict(item)
        candidate["quality_score"] = quality_score
        candidate["preference_score"] = preference_score
        candidate["selection_score"] = (
            float(item.get("score", 0) or 0) * 0.7
            + quality_score * 1.35
            + preference_score
        )
        ranked.append(candidate)

    ranked.sort(
        key=lambda item: (
            float(item.get("selection_score", 0) or 0),
            float(item.get("score", 0) or 0),
            parse_datetime(item.get("publish_date", "")),
        ),
        reverse=True,
    )

    preferred = [
        item
        for item in ranked
        if not is_low_signal_update(
            item.get("title", ""),
            item.get("summary", "") or item.get("content", ""),
            item.get("url", ""),
            item.get("platform", ""),
            item.get("source_detail", ""),
            item.get("category", ""),
            source_preferences=source_preferences,
        )
    ]

    if len(preferred) < minimum_count:
        preferred = ranked

    return preferred[:target_count]


def limit_papers_by_topic(papers: List[Dict[str, Any]], topic_limits: Dict[str, Any], total_limit: int) -> List[Dict[str, Any]]:
    ranked = sorted(
        papers,
        key=lambda item: (float(item.get("score", 0) or 0), item.get("publish_date", "")),
        reverse=True,
    )
    if not topic_limits:
        return ranked[:total_limit]

    normalized_limits = {str(topic): max(0, int(limit)) for topic, limit in topic_limits.items()}
    topic_counts = {topic: 0 for topic in normalized_limits}
    limited: List[Dict[str, Any]] = []
    for item in ranked:
        topic = item.get("topic") or item.get("category") or "Other"
        limit = normalized_limits.get(topic, 0)
        if limit <= 0 or topic_counts.get(topic, 0) >= limit:
            continue
        limited.append(item)
        topic_counts[topic] = topic_counts.get(topic, 0) + 1
        if len(limited) >= total_limit:
            break
    return limited


def apply_preference_scores(items: List[Dict[str, Any]], preference_config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    scored: List[Dict[str, Any]] = []
    for item in items:
        candidate = dict(item)
        candidate["preference_score"] = score_preference_boost(candidate, preference_config)
        candidate["selection_score"] = float(candidate.get("score", 0) or 0) + float(candidate.get("preference_score", 0) or 0)
        scored.append(candidate)
    return scored


def build_trend_summary(
    recent_articles: List[Dict[str, Any]],
    lookback_days: int = 3,
    max_items: int = 5,
    min_occurrences: int = 2,
) -> Dict[str, Any]:
    topic_counter: Counter[str] = Counter()
    keyword_counter: Counter[str] = Counter()

    for article in recent_articles:
        topic = str(article.get("topic_cn") or article.get("topic") or article.get("category") or "").strip()
        if topic:
            topic_counter[topic] += 1
        keywords = article.get("keywords") or ""
        if isinstance(keywords, str):
            keyword_list = [part.strip() for part in keywords.split(",") if part.strip()]
        else:
            keyword_list = [str(part).strip() for part in keywords if str(part).strip()]
        for keyword in keyword_list[:5]:
            keyword_counter[keyword] += 1

    items: List[Dict[str, Any]] = []
    used_labels = set()
    for label, count in topic_counter.most_common(max_items * 2):
        if count < min_occurrences or label in used_labels:
            continue
        used_labels.add(label)
        items.append(
            {
                "label": label,
                "count": count,
                "window": f"近 {lookback_days} 天",
                "summary": f"近 {lookback_days} 天共出现 {count} 次，说明这个方向正在持续升温，不只是单次热点。",
            }
        )
        if len(items) >= max_items:
            break

    if len(items) < max_items:
        for label, count in keyword_counter.most_common(max_items * 3):
            if count < min_occurrences or label in used_labels:
                continue
            used_labels.add(label)
            items.append(
                {
                    "label": label,
                    "count": count,
                    "window": f"近 {lookback_days} 天",
                    "summary": f"这个关键词在近 {lookback_days} 天被反复提及 {count} 次，值得持续跟踪后续产品动作和论文引用。",
                }
            )
            if len(items) >= max_items:
                break

    return {"items": items[:max_items]}


def build_alert_summary(
    collector_runs: List[Dict[str, Any]],
    db: Database,
    updates_count: int,
    alert_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    alert_config = alert_config or {}
    issues: List[str] = []
    arxiv_threshold = int(alert_config.get("arxiv_failure_threshold", 1))
    min_update_count = int(alert_config.get("min_update_count", 24))

    arxiv_runs = [item for item in collector_runs if str(item.get("label", "")).startswith("ArxivCollector[")]
    arxiv_failures_now = [item for item in arxiv_runs if item.get("status") != "success"]
    if arxiv_runs and len(arxiv_failures_now) >= arxiv_threshold:
        recent_arxiv_runs = db.get_recent_collector_runs("ArxivCollector[", limit=max(len(arxiv_runs) * 2, 6))
        consecutive_failed = 0
        for row in recent_arxiv_runs:
            if row.get("status") == "success":
                break
            consecutive_failed += 1
        if consecutive_failed >= arxiv_threshold:
            issues.append(f"Arxiv 采集已连续失败 {consecutive_failed} 次，本轮论文内容更多依赖历史回补。")

    if updates_count < min_update_count:
        issues.append(f"去重后全网动态仅保留 {updates_count} 条，低于预期阈值 {min_update_count} 条。")

    failed_collectors = [item.get("label") for item in collector_runs if item.get("status") == "error"]
    timed_out_collectors = [item.get("label") for item in collector_runs if item.get("status") == "timeout"]
    if timed_out_collectors:
        issues.append("以下采集单元发生超时：" + "、".join(timed_out_collectors[:5]))
    if failed_collectors:
        issues.append("以下采集单元执行失败：" + "、".join(failed_collectors[:5]))

    return {"needs_alert": bool(issues), "issues": issues}


def should_skip_rss_feed(
    db: Database,
    feed_name: str,
    degradation_config: Optional[Dict[str, Any]] = None,
) -> bool:
    degradation_config = degradation_config or {}
    if not degradation_config.get("enabled", False):
        return False

    failure_threshold = int(degradation_config.get("failure_threshold", 2))
    lookback_runs = int(degradation_config.get("lookback_runs", 6))
    recent_runs = db.get_recent_collector_runs(f"RSSCollector[{feed_name}]", limit=lookback_runs)
    consecutive_failures = 0
    for row in recent_runs:
        if row.get("status") == "success":
            break
        consecutive_failures += 1
    return consecutive_failures >= failure_threshold


def hydrate_paper_cache(
    papers: List[Dict[str, Any]],
    db: Database,
    cache_hours: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    cache = db.get_cached_paper_enrichment([str(item.get("url", "")) for item in papers if item.get("url")], max_age_hours=cache_hours)
    hydrated: List[Dict[str, Any]] = []
    missing: List[Dict[str, Any]] = []
    for paper in papers:
        cached = cache.get(str(paper.get("url", "")))
        if not cached:
            missing.append(paper)
            continue
        candidate = dict(paper)
        if cached.get("abstract"):
            candidate["content"] = cached["abstract"]
        if cached.get("author"):
            candidate["author"] = cached["author"]
        if cached.get("publish_date"):
            candidate["publish_date"] = cached["publish_date"]
        hydrated.append(candidate)
    return hydrated, missing


def render_alert_email_html(title: str, run_id: str, issues: List[str]) -> str:
    bullet_html = "".join(f"<li>{issue}</li>" for issue in issues)
    return f"""
    <html lang="zh-CN">
    <body style="font-family:Segoe UI,Microsoft YaHei,sans-serif;background:#f8fafc;padding:24px;color:#16212f;">
        <div style="max-width:720px;margin:0 auto;background:#fff;border:1px solid #e2e8f0;border-radius:16px;padding:24px;">
            <h2 style="margin-top:0;">AI 日报异常提醒</h2>
            <p>本次运行 <strong>{run_id}</strong> 在生成 <strong>{title}</strong> 时检测到以下异常：</p>
            <ul>{bullet_html}</ul>
            <p>建议查看最新日报和采集健康摘要，确认是否需要调整源配置或手动补发。</p>
        </div>
    </body>
    </html>
    """


def build_archive_summary(
    output_html: str = "reports_index.html",
    output_markdown: str = "reports_index.md",
    limit: int = 60,
    report_dir: str = "archive",
) -> Dict[str, Any]:
    root = Path.cwd()
    report_files = list(root.glob("report_*.html"))
    archive_path = root / report_dir
    if archive_path.exists():
        report_files.extend(archive_path.glob("report_*.html"))
    report_files = sorted(report_files, reverse=True)
    entries: List[Dict[str, str]] = []

    for html_path in report_files[:limit]:
        markdown_path = html_path.with_suffix(".md")
        stamp = html_path.stem.replace("report_", "")
        label = stamp
        if len(stamp) >= 13:
            label = f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]} {stamp[9:11]}:{stamp[11:13]}"
        entries.append(
            {
                "label": label,
                "html_path": html_path.relative_to(root).as_posix(),
                "markdown_path": markdown_path.relative_to(root).as_posix() if markdown_path.exists() else "",
            }
        )

    html_items = []
    md_lines = ["# 报告归档", ""]
    for entry in entries:
        html_link = f'<a href="{entry["html_path"]}">HTML</a>' if entry["html_path"] else ""
        md_link = f'<a href="{entry["markdown_path"]}">Markdown</a>' if entry["markdown_path"] else ""
        html_items.append(
            f'<li data-label="{entry["label"]}"><strong>{entry["label"]}</strong> {html_link} {md_link}</li>'
        )
        md_line = f"- {entry['label']}"
        if entry["html_path"]:
            md_line += f" | HTML: {entry['html_path']}"
        if entry["markdown_path"]:
            md_line += f" | Markdown: {entry['markdown_path']}"
        md_lines.append(md_line)

    archive_html = f"""
    <html lang="zh-CN">
    <head>
        <meta charset="utf-8">
        <title>报告归档</title>
        <style>
            body {{
                font-family: Segoe UI, Microsoft YaHei, sans-serif;
                background: #f8fafc;
                padding: 24px;
                color: #16212f;
            }}
            .wrap {{
                max-width: 920px;
                margin: 0 auto;
                background: #fff;
                border: 1px solid #e2e8f0;
                border-radius: 16px;
                padding: 24px;
            }}
            .search {{
                width: 100%;
                padding: 12px 14px;
                border: 1px solid #cbd5e1;
                border-radius: 12px;
                font-size: 14px;
                margin: 8px 0 16px;
            }}
            ul {{
                padding-left: 20px;
            }}
            li {{
                margin: 10px 0;
            }}
            a {{
                display: inline-block;
                margin-left: 8px;
                color: #0d5f5a;
                text-decoration: none;
            }}
        </style>
    </head>
    <body>
        <div class="wrap">
            <h1>报告归档</h1>
            <p>按日期回看历史 HTML / Markdown 报告，也可以直接搜索日期关键字。</p>
            <input id="report-search" class="search" placeholder="搜索日期，例如 2026-03-29 或 22:28" />
            <ul id="report-list">{''.join(html_items)}</ul>
        </div>
        <script>
            const input = document.getElementById("report-search");
            const items = Array.from(document.querySelectorAll("#report-list li"));
            input.addEventListener("input", () => {{
                const keyword = input.value.trim().toLowerCase();
                items.forEach((item) => {{
                    const label = (item.getAttribute("data-label") || "").toLowerCase();
                    item.style.display = !keyword || label.includes(keyword) ? "" : "none";
                }});
            }});
        </script>
    </body>
    </html>
    """
    Path(output_html).write_text(archive_html, encoding="utf-8")
    Path(output_markdown).write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return {"entries": entries, "html_path": output_html, "markdown_path": output_markdown}


def build_trend_summary_v2(
    recent_articles: List[Dict[str, Any]],
    lookback_days: int = 3,
    max_items: int = 5,
    min_occurrences: int = 2,
) -> Dict[str, Any]:
    signal_map: Dict[str, Dict[str, Any]] = {}

    for article in recent_articles:
        label = str(article.get("topic_cn") or article.get("display_topic") or article.get("topic") or article.get("category") or "").strip()
        if not label:
            keywords = article.get("keywords") or ""
            if isinstance(keywords, str):
                keyword_list = [part.strip() for part in keywords.split(",") if part.strip()]
            else:
                keyword_list = [str(part).strip() for part in keywords if str(part).strip()]
            label = keyword_list[0] if keyword_list else ""
        if not label:
            continue

        publish_date = str(article.get("publish_date", "") or "")[:10]
        source = str(article.get("source_detail") or article.get("source") or article.get("platform") or urlparse(str(article.get("url", "") or "")).netloc.lower() or "未知来源")
        entry = signal_map.setdefault(label, {"count": 0, "days": set(), "sources": set(), "examples": []})
        entry["count"] += 1
        if publish_date:
            entry["days"].add(publish_date)
        if source:
            entry["sources"].add(source)
        sample_title = str(article.get("title_cn") or article.get("title") or "").strip()
        if sample_title and sample_title not in entry["examples"]:
            entry["examples"].append(sample_title)

    ranked = sorted(
        signal_map.items(),
        key=lambda pair: (len(pair[1]["days"]), len(pair[1]["sources"]), pair[1]["count"]),
        reverse=True,
    )

    items: List[Dict[str, Any]] = []
    for label, meta in ranked:
        day_count = len(meta["days"])
        source_count = len(meta["sources"])
        if meta["count"] < min_occurrences:
            continue
        if day_count < 2 and source_count < 2:
            continue
        items.append(
            {
                "label": label,
                "count": meta["count"],
                "day_count": day_count,
                "source_count": source_count,
                "window": f"近 {lookback_days} 天",
                "summary": (
                    f"近 {lookback_days} 天里连续出现 {meta['count']} 次，覆盖 {max(day_count, 1)} 天、"
                    f"{max(source_count, 1)} 个来源，说明它不是单点消息，而是在被持续验证。"
                ),
                "examples": meta["examples"][:2],
            }
        )
        if len(items) >= max_items:
            break

    return {"items": items[:max_items]}


def build_alert_summary_v2(
    collector_runs: List[Dict[str, Any]],
    db: Database,
    updates_count: int,
    paper_count: int = 0,
    update_candidate_count: int = 0,
    alert_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    alert_config = alert_config or {}
    issues: List[str] = []
    arxiv_threshold = int(alert_config.get("arxiv_failure_threshold", 1))
    min_update_count = int(alert_config.get("min_update_count", 20))
    min_paper_count = int(alert_config.get("min_paper_count", 12))
    duplicate_ratio_threshold = float(alert_config.get("duplicate_ratio_threshold", 0.55))

    arxiv_runs = [item for item in collector_runs if str(item.get("label", "")).startswith("ArxivCollector[")]
    arxiv_failures_now = [item for item in arxiv_runs if item.get("status") != "success"]
    if arxiv_runs and len(arxiv_failures_now) >= arxiv_threshold:
        recent_arxiv_runs = db.get_recent_collector_runs("ArxivCollector[", limit=max(len(arxiv_runs) * 2, 6))
        consecutive_failed = 0
        for row in recent_arxiv_runs:
            if row.get("status") == "success":
                break
            consecutive_failed += 1
        if consecutive_failed >= arxiv_threshold:
            issues.append(f"Arxiv 采集已连续失败 {consecutive_failed} 次，本轮论文内容更多依赖历史回补。")

    if paper_count and paper_count < min_paper_count:
        issues.append(f"最终论文仅保留 {paper_count} 篇，低于建议阈值 {min_paper_count} 篇。")

    if updates_count < min_update_count:
        issues.append(f"去重后全网动态仅保留 {updates_count} 条，低于建议阈值 {min_update_count} 条。")

    if update_candidate_count:
        duplicate_ratio = 1 - (updates_count / max(update_candidate_count, 1))
        if duplicate_ratio >= duplicate_ratio_threshold:
            issues.append(
                f"动态候选去重比例达到 {duplicate_ratio:.0%}，说明同类消息重复较多，建议继续收紧来源与聚类规则。"
            )

    failed_collectors = [item.get("label") for item in collector_runs if item.get("status") == "error"]
    timed_out_collectors = [item.get("label") for item in collector_runs if item.get("status") == "timeout"]
    critical_failures = [
        label
        for label in failed_collectors + timed_out_collectors
        if any(
            key in str(label)
            for key in [
                "ArxivCollector[",
                "RSSCollector[OpenAI Blog]",
                "RSSCollector[DeepMind Blog]",
                "RSSCollector[Hugging Face Blog]",
            ]
        )
    ]
    if critical_failures:
        issues.append("以下核心采集单元出现异常：" + "、".join(critical_failures[:5]))

    return {"needs_alert": bool(issues), "issues": issues}


def load_archive_manifest(manifest_path: str = "reports_manifest.json") -> Dict[str, Any]:
    path = Path(manifest_path)
    if not path.exists():
        return {"entries": []}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"entries": []}


def is_validation_archive_entry(entry: Dict[str, Any]) -> bool:
    html_path = str(entry.get("html_path") or "")
    markdown_path = str(entry.get("markdown_path") or "")
    normalized = f"{html_path}\n{markdown_path}".replace("\\", "/").lower()
    return "archive/validation/" in normalized


def update_archive_manifest(
    html_filename: str,
    markdown_filename: str,
    report_summary: Dict[str, Any],
    papers: List[Dict[str, Any]],
    updates: List[Dict[str, Any]],
    manifest_path: str = "reports_manifest.json",
) -> None:
    manifest = load_archive_manifest(manifest_path)
    entries = list(manifest.get("entries") or [])

    seen_sources = set()
    sources: List[str] = []
    for item in updates[:10]:
        source = str(item.get("source_detail") or item.get("source") or item.get("platform") or "").strip()
        if source and source not in seen_sources:
            seen_sources.add(source)
            sources.append(source)

    entry = {
        "label": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "html_path": html_filename,
        "markdown_path": markdown_filename,
        "topics": list(report_summary.get("hot_topics") or [])[:4],
        "sources": sources[:4],
        "paper_count": len(papers),
        "update_count": len(updates),
    }

    entries = [
        item
        for item in entries
        if str(item.get("html_path")) != html_filename and str(item.get("markdown_path")) != markdown_filename
    ]
    entries.insert(0, entry)
    Path(manifest_path).write_text(
        json.dumps({"entries": entries[:120]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_archive_summary_v2(
    output_html: str = "reports_index.html",
    output_markdown: str = "reports_index.md",
    limit: int = 60,
    report_dir: str = "archive",
) -> Dict[str, Any]:
    manifest = load_archive_manifest()
    entries = [
        entry
        for entry in list(manifest.get("entries") or [])
        if not is_validation_archive_entry(entry)
    ][:limit]

    if not entries:
        return build_archive_summary(output_html, output_markdown, limit, report_dir=report_dir)

    html_items = []
    md_lines = ["# 报告归档", ""]
    for entry in entries:
        html_link = f'<a href="{entry["html_path"]}">HTML</a>' if entry.get("html_path") else ""
        md_link = f'<a href="{entry["markdown_path"]}">Markdown</a>' if entry.get("markdown_path") else ""
        topics = " ".join(entry.get("topics") or [])
        sources = " ".join(entry.get("sources") or [])
        html_items.append(
            f'<li data-label="{entry["label"]}" data-topics="{topics}" data-sources="{sources}"><strong>{entry["label"]}</strong> '
            f'{html_link} {md_link} '
            f'{"<span style=\"margin-left:8px;color:#64748b;\">主题：" + topics + "</span>" if topics else ""}'
            f'{"<span style=\"margin-left:8px;color:#64748b;\">来源：" + sources + "</span>" if sources else ""}'
            f"</li>"
        )
        md_line = f"- {entry['label']}"
        if entry.get("html_path"):
            md_line += f" | HTML: {entry['html_path']}"
        if entry.get("markdown_path"):
            md_line += f" | Markdown: {entry['markdown_path']}"
        if topics:
            md_line += f" | Topics: {topics}"
        if sources:
            md_line += f" | Sources: {sources}"
        md_lines.append(md_line)

    archive_html = f"""
    <html lang="zh-CN">
    <head>
        <meta charset="utf-8">
        <title>报告归档</title>
        <style>
            body {{
                font-family: Segoe UI, Microsoft YaHei, sans-serif;
                background: #f8fafc;
                padding: 24px;
                color: #16212f;
            }}
            .wrap {{
                max-width: 920px;
                margin: 0 auto;
                background: #fff;
                border: 1px solid #e2e8f0;
                border-radius: 16px;
                padding: 24px;
            }}
            .search {{
                width: 100%;
                padding: 12px 14px;
                border: 1px solid #cbd5e1;
                border-radius: 12px;
                font-size: 14px;
                margin: 8px 0 16px;
            }}
            ul {{
                padding-left: 20px;
            }}
            li {{
                margin: 10px 0;
            }}
            a {{
                display: inline-block;
                margin-left: 8px;
                color: #0d5f5a;
                text-decoration: none;
            }}
        </style>
    </head>
    <body>
        <div class="wrap">
            <h1>报告归档</h1>
            <p>支持按日期、主题关键词和来源摘要回看 HTML / Markdown 报告。</p>
            <input id="report-search" class="search" placeholder="搜索日期、主题或来源，例如 2026-03-29、世界模型、Hugging Face" />
            <ul id="report-list">{''.join(html_items)}</ul>
        </div>
        <script>
            const input = document.getElementById("report-search");
            const items = Array.from(document.querySelectorAll("#report-list li"));
            input.addEventListener("input", () => {{
                const keyword = input.value.trim().toLowerCase();
                items.forEach((item) => {{
                    const label = (item.getAttribute("data-label") || "").toLowerCase();
                    const topics = (item.getAttribute("data-topics") || "").toLowerCase();
                    const sources = (item.getAttribute("data-sources") || "").toLowerCase();
                    const visible = !keyword || label.includes(keyword) || topics.includes(keyword) || sources.includes(keyword);
                    item.style.display = visible ? "" : "none";
                }});
            }});
        </script>
    </body>
    </html>
    """

    Path(output_html).write_text(archive_html, encoding="utf-8")
    Path(output_markdown).write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return {"entries": entries, "html_path": output_html, "markdown_path": output_markdown}


def prepare_report_items(
    items: List[Dict[str, Any]],
    llm_processor: LLMProcessor,
    runtime_results: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    prepared = []
    for item in items:
        prepared.append(llm_processor.prepare_report_item(item, runtime_results.get(item.get("url", ""))))
    return prepared


def clean_title_candidate(text: str, max_len: int = 36) -> str:
    cleaned = re.sub(r"[。！？.!?]+$", "", str(text or "").strip())
    cleaned = re.sub(r"\s+", " ", cleaned).strip("，、；：:; ")
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 1].rstrip("，、；：:; ") + "…"


def title_template_fingerprint(title: str) -> str:
    cleaned = clean_title_candidate(title, max_len=48)
    if not cleaned:
        return ""
    generic_prefixes = (
        "AI产品",
        "AI合作",
        "AI应用",
        "AI行业",
        "AI讨论热点",
        "视频热点背后",
        "世界模型研究",
        "具身智能研究",
        "机器人研究",
        "模型能力和效率路线",
    )
    subject_pattern = (
        r"^[\u4e00-\u9fffA-Za-z0-9·\-\s]{2,18}"
        r"(?=(把|借|继续|开始|正在|尝试|试图|推进|切入|争夺|补齐|放大|折射出|反映出|"
        r"验证|解决|提升|构建|降低|押向))"
    )
    normalized = re.sub(subject_pattern, "<subject>", cleaned)
    for prefix in generic_prefixes:
        if normalized.startswith(prefix):
            normalized = "<generic>" + normalized[len(prefix) :]
            break
    return normalize_text(normalized)


def title_similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, normalize_text(left), normalize_text(right)).ratio()


def build_distinct_title_candidate(item: Dict[str, Any], llm_processor: LLMProcessor) -> List[str]:
    current_title = clean_title_candidate(item.get("title_cn") or item.get("title") or "")
    preview = clean_title_candidate(item.get("summary_preview") or "", max_len=34)
    summary_sentences = llm_processor._split_sentences(item.get("summary", ""))
    summary_title = ""
    if summary_sentences:
        summary_title = llm_processor._refine_title_text(summary_sentences[0], 16, 36) or ""
    subject = llm_processor._extract_subject(item)
    topic = clean_title_candidate(item.get("display_topic") or item.get("topic_cn") or item.get("category") or "", max_len=18)
    source = clean_title_candidate(item.get("source_detail") or item.get("source") or item.get("platform") or "", max_len=16)
    keywords = item.get("keywords") or []
    if isinstance(keywords, str):
        keywords = [part.strip() for part in keywords.split(",") if part.strip()]
    else:
        keywords = [str(part).strip() for part in keywords if str(part).strip()]
    focus = clean_title_candidate(next((keyword for keyword in keywords if keyword and keyword not in current_title), ""), max_len=14)

    candidates: List[str] = []
    for candidate in (preview, summary_title):
        if candidate:
            candidates.append(candidate)
    if focus and subject and subject != "相关机构":
        candidates.append(clean_title_candidate(f"{subject}这次更想解决{focus}", max_len=34))
    if focus and topic:
        candidates.append(clean_title_candidate(f"{topic}这次更值得看{focus}", max_len=34))
    if topic and source:
        candidates.append(clean_title_candidate(f"{source}这次动作落在{topic}", max_len=34))
    if topic and preview:
        candidates.append(clean_title_candidate(f"{topic}：{preview}", max_len=34))

    unique_candidates: List[str] = []
    seen = set()
    for candidate in candidates:
        if not candidate:
            continue
        key = normalize_text(candidate)
        if key in seen or key == normalize_text(current_title):
            continue
        seen.add(key)
        unique_candidates.append(candidate)
    return unique_candidates


def diversify_report_titles(items: List[Dict[str, Any]], llm_processor: LLMProcessor) -> List[Dict[str, Any]]:
    fingerprint_counts = Counter(
        title_template_fingerprint(item.get("title_cn") or item.get("title") or "")
        for item in items
        if title_template_fingerprint(item.get("title_cn") or item.get("title") or "")
    )
    used_titles: List[str] = []
    seen_fingerprints: Counter[str] = Counter()
    diversified: List[Dict[str, Any]] = []

    for item in items:
        candidate = dict(item)
        title = clean_title_candidate(candidate.get("title_cn") or candidate.get("title") or "")
        fingerprint = title_template_fingerprint(title)
        seen_fingerprints[fingerprint] += 1
        needs_rewrite = bool(
            title
            and (
                any(title_similarity(title, used) >= 0.82 for used in used_titles)
                or (fingerprint and fingerprint_counts.get(fingerprint, 0) > 1 and seen_fingerprints[fingerprint] > 1)
            )
        )
        if needs_rewrite:
            for rewritten in build_distinct_title_candidate(candidate, llm_processor):
                if any(title_similarity(rewritten, used) >= 0.82 for used in used_titles):
                    continue
                if fingerprint and title_template_fingerprint(rewritten) == fingerprint:
                    continue
                candidate["title_cn"] = rewritten
                title = rewritten
                break
        if title:
            used_titles.append(title)
        diversified.append(candidate)
    return diversified


def parse_datetime(value: str) -> datetime:
    if not value:
        return datetime.min
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return datetime.min


def normalized_event_title(item: Dict[str, Any]) -> str:
    title = normalize_text(item.get("title", ""), item.get("title_cn", ""))
    return re.sub(r"\b(latest|today|breaking|report|news)\b", " ", title).strip()


def event_tokens(item: Dict[str, Any]) -> set[str]:
    combined = normalize_text(item.get("title", ""), item.get("title_cn", ""), item.get("summary", ""))
    tokens = set()
    for token in re.split(r"\W+", combined):
        if len(token) <= 2 or token in EVENT_STOPWORDS:
            continue
        tokens.add(token)
    return tokens


def is_official_source(item: Dict[str, Any]) -> bool:
    host = urlparse(item.get("url", "")).netloc.lower()
    if any(hint in host for hint in OFFICIAL_HOST_HINTS):
        return True
    platform = (item.get("platform") or "").lower()
    source = (item.get("source") or "").lower()
    return platform in {"blog", "arxiv", "hugging face"} or source == "rss"


def choose_representative(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    left_rank = (
        int(is_official_source(left)),
        float(left.get("selection_score", left.get("score", 0)) or 0),
        float(left.get("score", 0) or 0),
        len(left.get("summary", "") or ""),
        parse_datetime(left.get("publish_date", "")),
    )
    right_rank = (
        int(is_official_source(right)),
        float(right.get("selection_score", right.get("score", 0)) or 0),
        float(right.get("score", 0) or 0),
        len(right.get("summary", "") or ""),
        parse_datetime(right.get("publish_date", "")),
    )
    return left if left_rank >= right_rank else right


def classify_event_match(left: Dict[str, Any], right: Dict[str, Any]) -> Tuple[str, str]:
    left_title = normalized_event_title(left)
    right_title = normalized_event_title(right)
    if not left_title or not right_title:
        return "none", ""

    title_ratio = SequenceMatcher(None, left_title, right_title).ratio()
    overlap = len(event_tokens(left) & event_tokens(right))
    time_gap = abs(parse_datetime(left.get("publish_date", "")) - parse_datetime(right.get("publish_date", "")))
    same_topic = (left.get("display_topic") or left.get("topic_cn")) == (right.get("display_topic") or right.get("topic_cn"))

    if left_title == right_title or title_ratio >= 0.94:
        return "same", "标题几乎一致。"
    if title_ratio >= 0.86 and overlap >= 4 and time_gap <= timedelta(hours=36):
        return "same", "标题和关键词高度重合，且发布时间接近。"
    if title_ratio >= 0.74 and overlap >= 3 and time_gap <= timedelta(hours=48):
        return "possible", "标题与关键词存在明显重合，可能是同一事件。"
    if same_topic and overlap >= 4 and time_gap <= timedelta(hours=24):
        return "possible", "主题一致且关键词重合较高，需要进一步判断。"
    return "none", ""


def dedupe_updates(updates: List[Dict[str, Any]], llm_processor: LLMProcessor, max_llm_checks: int = 12) -> List[Dict[str, Any]]:
    sorted_updates = sorted(
        updates,
        key=lambda item: (
            float(item.get("selection_score", item.get("score", 0)) or 0),
            float(item.get("score", 0) or 0),
            parse_datetime(item.get("publish_date", "")),
        ),
        reverse=True,
    )
    clusters: List[Dict[str, Any]] = []
    llm_checks = 0

    for item in sorted_updates:
        matched_cluster: Optional[Dict[str, Any]] = None
        match_reason = ""
        for cluster in clusters:
            representative = cluster["representative"]
            decision, reason = classify_event_match(representative, item)
            if decision == "none":
                continue
            if decision == "same":
                matched_cluster = cluster
                match_reason = reason
                break
            if decision == "possible" and llm_checks < max_llm_checks:
                verdict = llm_processor.judge_event_similarity(representative, item)
                llm_checks += 1
                if verdict.get("decision") == "same_event":
                    matched_cluster = cluster
                    match_reason = verdict.get("reason", reason)
                    break
        if matched_cluster is None:
            clusters.append(
                {
                    "cluster_id": f"event-{len(clusters) + 1:03d}",
                    "items": [item],
                    "representative": item,
                    "cluster_sources": {item.get("source_detail") or item.get("source") or item.get("platform") or "未知来源"},
                    "dedupe_reason": "unique",
                }
            )
            continue

        matched_cluster["items"].append(item)
        matched_cluster["cluster_sources"].add(item.get("source_detail") or item.get("source") or item.get("platform") or "未知来源")
        matched_cluster["dedupe_reason"] = match_reason or matched_cluster["dedupe_reason"]
        matched_cluster["representative"] = choose_representative(matched_cluster["representative"], item)

    deduped = []
    for cluster in clusters:
        representative = dict(cluster["representative"])
        representative["cluster_id"] = cluster["cluster_id"]
        representative["cluster_size"] = len(cluster["items"])
        representative["cluster_sources"] = sorted(cluster["cluster_sources"])
        representative["dedupe_reason"] = cluster["dedupe_reason"]
        deduped.append(representative)
    return sorted(
        deduped,
        key=lambda item: (
            float(item.get("selection_score", item.get("score", 0)) or 0),
            float(item.get("score", 0) or 0),
            parse_datetime(item.get("publish_date", "")),
        ),
        reverse=True,
    )


def main() -> Dict[str, Any]:
    load_dotenv()
    config = load_config()
    runtime_profile = str(os.getenv("WEB_AGENT_RUN_PROFILE", "") or "").strip()
    config = apply_runtime_profile(config, runtime_profile)
    started_at = datetime.now()
    run_id = started_at.strftime("%Y%m%d_%H%M%S")
    network_timeout = int(config.get("network", {}).get("timeout_seconds", 25))
    collector_timeout = int(config.get("network", {}).get("collector_timeout_seconds", 240))
    arxiv_collector_timeout = int(config.get("network", {}).get("arxiv_collector_timeout_seconds", max(collector_timeout, 180)))
    print(f"Starting AI News Agent - {started_at.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Run ID: {run_id}")
    run_result: Dict[str, Any] = {
        "run_id": run_id,
        "started_at": started_at.isoformat(timespec="seconds"),
        "finished_at": "",
        "success": False,
        "status": "running",
        "retryable": True,
        "delivery_status": "pending",
        "html_report_path": "",
        "markdown_report_path": "",
        "alert_file_path": "",
        "new_articles_count": 0,
        "processed_count": 0,
        "analysis_backfill_count": 0,
        "paper_count": 0,
        "update_count": 0,
        "collector_summary": {},
        "alert_summary": {},
        "runtime_profile": runtime_profile or "default",
    }

    db = Database("ai_news.db")
    print("Database initialized at ai_news.db")

    print("\n=== Step 1: Collecting Data ===")
    collectors = []
    arxiv_config = config["sources"].get("arxiv", {})
    if arxiv_config.get("enabled", False):
        topic_limits = arxiv_config.get("topic_limits", {})
        topic_queries = arxiv_config.get("topic_queries", {})
        for topic, limit in topic_limits.items():
            limit = int(limit)
            if limit <= 0:
                continue
            collector = ArxivCollector(
                categories=arxiv_config.get("categories", []),
                max_results=limit,
                candidate_pool=int(arxiv_config.get("candidate_pool", 160)),
                days_back=int(arxiv_config.get("days_back", 1)),
                topic_limits={str(topic): limit},
                fallback_days=arxiv_config.get("fallback_days", [1, 3, 7, 14]),
                topic_queries={str(topic): topic_queries.get(topic, "")},
            )
            collector.label = f"ArxivCollector[{topic}]"
            collector.timeout_seconds = arxiv_collector_timeout
            collector.run_in_subprocess = False
            collectors.append(collector)
    rss_config = config["sources"].get("rss", {})
    if rss_config.get("enabled", False):
        for feed in rss_config.get("feeds", []):
            if should_skip_rss_feed(db, str(feed.get("name", "feed")), rss_config.get("degradation", {})):
                print(f"Skipping degraded RSS feed: {feed.get('name', 'feed')}")
                continue
            collector = RSSCollector(feeds=[feed], days_back=int(rss_config.get("days_back", 2)))
            collector.label = f"RSSCollector[{feed.get('name', 'feed')}]"
            collectors.append(collector)
    huggingface_config = config["sources"].get("huggingface", {})
    if huggingface_config.get("enabled", False):
        collectors.append(HuggingFaceCollector())
    web_search_config = config["sources"].get("web_search", {})
    if web_search_config.get("enabled", False):
        for search in web_search_config.get("searches", []):
            collector = WebSearchCollector(
                searches=[search],
                locale=web_search_config.get("locale", "US:en"),
                days_back=int(web_search_config.get("days_back", 1)),
                fallback_days=web_search_config.get("fallback_days", [1, 2, 3]),
            )
            collector.label = f"WebSearchCollector[{search.get('name', 'search')}]"
            collectors.append(collector)

    new_articles_count = 0
    collector_runs: List[Dict[str, Any]] = []
    original_socket_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(network_timeout)
    try:
        for collector in collectors:
            label = collector_label(collector)
            started_at = datetime.now()
            try:
                print(f"Running {label}...")
                timeout_seconds = int(getattr(collector, "timeout_seconds", collector_timeout))
                if getattr(collector, "run_in_subprocess", True):
                    items, collector_error = run_collector_with_timeout(collector, timeout_seconds, network_timeout)
                else:
                    items = collector.collect()
                    collector_error = None
                if collector_error:
                    print(f"Error in {label}: {collector_error}")
                    collector_runs.append(
                        {
                            "label": label,
                            "status": "timeout" if "Timed out" in collector_error else "error",
                            "inserted_count": 0,
                            "collected_count": 0,
                            "duration_seconds": round((datetime.now() - started_at).total_seconds(), 1),
                            "error": collector_error,
                        }
                    )
                    continue
                inserted_count = 0
                for item in items:
                    item["run_id"] = run_id
                    if db.insert_article(item):
                        new_articles_count += 1
                        inserted_count += 1
                collector_runs.append(
                    {
                        "label": label,
                        "status": "success",
                        "inserted_count": inserted_count,
                        "collected_count": len(items),
                        "duration_seconds": round((datetime.now() - started_at).total_seconds(), 1),
                        "error": "",
                    }
                )
            except Exception as exc:
                print(f"Error in {label}: {exc}")
                collector_runs.append(
                    {
                        "label": label,
                        "status": "error",
                        "inserted_count": 0,
                        "collected_count": 0,
                        "duration_seconds": round((datetime.now() - started_at).total_seconds(), 1),
                        "error": str(exc),
                    }
                )
    finally:
        socket.setdefaulttimeout(original_socket_timeout)
    print(f"Collection complete. Added {new_articles_count} new items.")
    db.record_collector_runs(run_id, collector_runs)
    collector_summary = build_collector_summary(collector_runs)
    print("Collector health: " + collector_summary["status_text"])
    run_result["new_articles_count"] = new_articles_count
    run_result["collector_summary"] = collector_summary

    print("\n=== Step 2: Processing Data ===")
    run_unprocessed = db.get_unprocessed_articles(run_id=run_id)
    backlog_unprocessed = [item for item in db.get_unprocessed_articles() if item.get("run_id") != run_id]
    unprocessed = merge_unique_articles(run_unprocessed, backlog_unprocessed)
    runtime_limits = dict(config.get("runtime", {}) or {})
    max_unprocessed_items = int(runtime_limits.get("max_unprocessed_items", 0) or 0)
    if max_unprocessed_items > 0 and len(unprocessed) > max_unprocessed_items:
        print(f"Runtime profile capped processing items from {len(unprocessed)} to {max_unprocessed_items}.")
        unprocessed = unprocessed[:max_unprocessed_items]
    print(f"Pending items to process: current run={len(run_unprocessed)}, backlog={len(backlog_unprocessed)}, total={len(unprocessed)}")

    llm_processor = LLMProcessor(config["llm"])
    processed_count = 0
    runtime_results: Dict[str, Dict[str, Any]] = {}
    for index, article in enumerate(unprocessed, start=1):
        short_title = safe_console_text(article.get("title", "")[:80])
        print(f"Processing [{index}/{len(unprocessed)}]: {short_title}")
        result = llm_processor.process_article(article)
        if not result:
            print("  -> Failed to process")
            continue
        db.update_article_processing(
            url=article["url"],
            summary=result.get("summary", ""),
            score=float(result.get("score", 0)),
            keywords=result.get("keywords", []),
            category=result.get("category", "Other"),
            title_cn=result.get("title_cn", ""),
            summary_preview=result.get("summary_preview", ""),
            why_it_matters=result.get("why_it_matters", ""),
            why_now=result.get("why_now", ""),
            expected_effect=result.get("expected_effect", ""),
            future_impact=result.get("future_impact", ""),
        )
        runtime_results[article["url"]] = result
        processed_count += 1
        print(f"  -> Score: {result.get('score')} | Category: {safe_console_text(str(result.get('category')))}")
    print(f"Processing complete. Successfully processed {processed_count} items.")
    run_result["processed_count"] = processed_count

    analysis_backfill_candidates = [
        item for item in db.get_recent_articles_missing_analysis(hours=24, limit=160) if item.get("url") not in runtime_results
    ]
    max_analysis_backfill_items = int(runtime_limits.get("max_analysis_backfill_items", 0) or 0)
    if max_analysis_backfill_items >= 0:
        analysis_backfill_candidates = analysis_backfill_candidates[:max_analysis_backfill_items]
    if analysis_backfill_candidates:
        print(f"Backfilling detailed analysis for {len(analysis_backfill_candidates)} recent processed items...")
    analysis_backfill_count = 0
    for index, article in enumerate(analysis_backfill_candidates, start=1):
        short_title = safe_console_text(article.get("title", "")[:80])
        print(f"Backfill [{index}/{len(analysis_backfill_candidates)}]: {short_title}")
        result = llm_processor.process_article(article)
        if not result:
            print("  -> Failed to backfill")
            continue
        db.update_article_processing(
            url=article["url"],
            summary=result.get("summary", article.get("summary", "")),
            score=float(result.get("score", article.get("score", 0)) or 0),
            keywords=result.get("keywords", article.get("keywords", [])),
            category=result.get("category", article.get("category", "Other")),
            title_cn=result.get("title_cn", article.get("title_cn", "")),
            summary_preview=result.get("summary_preview", article.get("summary_preview", "")),
            why_it_matters=result.get("why_it_matters", article.get("why_it_matters", "")),
            why_now=result.get("why_now", article.get("why_now", "")),
            expected_effect=result.get("expected_effect", article.get("expected_effect", "")),
            future_impact=result.get("future_impact", article.get("future_impact", "")),
        )
        runtime_results[article["url"]] = result
        analysis_backfill_count += 1
        print("  -> Detailed analysis refreshed")
    if analysis_backfill_candidates:
        print(f"Detailed analysis backfill complete. Updated {analysis_backfill_count} items.")
    run_result["analysis_backfill_count"] = analysis_backfill_count

    print("\n=== Step 3: Generating Report ===")
    report_items = db.get_articles_for_run(run_id=run_id, processed_only=True)
    source_preferences = config.get("source_preferences", {})
    preference_config = config.get("preferences", {})
    alert_config = config.get("alerts", {})
    trend_config = config.get("trends", {})
    archive_config = config.get("archive", {})
    paper_topic_limits = arxiv_config.get("topic_limits", {})
    default_paper_limit = sum(int(limit) for limit in paper_topic_limits.values()) if paper_topic_limits else 10
    paper_limit = int(config["report"].get("paper_limit", default_paper_limit))
    paper_cache_hours = int(config["report"].get("paper_enrichment_cache_hours", 168))
    web_limit = int(config["report"].get("web_limit", 20))
    min_web_items = int(config["report"].get("min_web_items", 20))
    paper_backfill_ladder = parse_hour_ladder(
        config["report"].get("paper_backfill_hours_ladder", config["report"].get("paper_backfill_hours", 24 * 7)),
        [24 * 7],
    )
    web_backfill_ladder = parse_hour_ladder(
        config["report"].get("web_backfill_hours_ladder", config["report"].get("web_backfill_hours", 24)),
        [24, 48, 72],
    )

    current_papers = [item for item in report_items if item.get("content_type") == "paper"]
    current_updates = [item for item in report_items if item.get("content_type") != "paper"]
    if paper_backfill_ladder and len(current_papers) < paper_limit:
        needed_papers = paper_limit - len(current_papers)
        for hours in paper_backfill_ladder:
            recent_papers = db.get_recent_processed_articles(
                hours=hours,
                content_type="paper",
                limit=max(paper_limit * 6, needed_papers * 8, 120),
            )
            report_items = merge_unique_articles(report_items, recent_papers)
            print(
                f"Augmented report with recent papers from the last {hours} hours because current run had only {len(current_papers)} papers."
            )
            candidate_papers = [item for item in report_items if item.get("content_type") == "paper"]
            if len(candidate_papers) >= paper_limit:
                break

    if web_backfill_ladder and len(current_updates) < min_web_items:
        needed_updates = min_web_items - len(current_updates)
        for hours in web_backfill_ladder:
            recent_updates = db.get_recent_processed_articles(
                hours=hours,
                limit=max(web_limit * 6, needed_updates * 8, 180),
            )
            report_items = merge_unique_articles(report_items, recent_updates)
            print(
                f"Augmented report with recent processed items from the last {hours} hours because current run had only {len(current_updates)} updates."
            )
            prepared_update_candidates = prepare_report_items(
                [item for item in report_items if item.get("content_type") != "paper"],
                llm_processor,
                runtime_results,
            )
            deduped_candidate_updates = dedupe_updates(prepared_update_candidates, llm_processor)
            if len(deduped_candidate_updates) >= min_web_items:
                break

    prepared_items = prepare_report_items(report_items, llm_processor, runtime_results)
    prepared_items = apply_preference_scores(prepared_items, preference_config)
    papers = limit_papers_by_topic(
        [item for item in prepared_items if item.get("content_type") == "paper"],
        paper_topic_limits,
        paper_limit,
    )

    if papers and arxiv_config.get("enabled", False) and not bool(runtime_limits.get("skip_paper_enrichment", False)):
        print(f"Refreshing full abstracts for final {len(papers)} selected papers...")
        cached_papers, missing_papers = hydrate_paper_cache(papers, db, paper_cache_hours)
        print(f"Paper cache hits: {len(cached_papers)} | misses: {len(missing_papers)}")
        enriched_map = {str(item.get("url", "")): item for item in cached_papers}
        if missing_papers:
            paper_enricher = ArxivCollector(
                categories=arxiv_config.get("categories", []),
                max_results=paper_limit,
                candidate_pool=int(arxiv_config.get("candidate_pool", 160)),
                days_back=int(arxiv_config.get("days_back", 1)),
                topic_limits=paper_topic_limits,
                fallback_days=arxiv_config.get("fallback_days", [1, 3, 7, 14]),
                topic_queries=arxiv_config.get("topic_queries", {}),
            )
            fetched_papers = paper_enricher.enrich_articles(missing_papers, max_items=len(missing_papers))
            db.upsert_paper_enrichment_cache(fetched_papers)
            for paper in fetched_papers:
                enriched_map[str(paper.get("url", ""))] = paper
        enriched_papers = [enriched_map.get(str(paper.get("url", "")), paper) for paper in papers]
        refined_papers: List[Dict[str, Any]] = []
        for paper in enriched_papers:
            result = llm_processor.process_article(paper)
            if result:
                db.update_article_processing(
                    url=paper["url"],
                    summary=result.get("summary", paper.get("summary", "")),
                    score=float(result.get("score", paper.get("score", 0)) or 0),
                    keywords=result.get("keywords", paper.get("keywords", [])),
                    category=result.get("category", paper.get("category", "Other")),
                    title_cn=result.get("title_cn", paper.get("title_cn", "")),
                    summary_preview=result.get("summary_preview", paper.get("summary_preview", "")),
                    why_it_matters=result.get("why_it_matters", paper.get("why_it_matters", "")),
                    why_now=result.get("why_now", paper.get("why_now", "")),
                    expected_effect=result.get("expected_effect", paper.get("expected_effect", "")),
                    future_impact=result.get("future_impact", paper.get("future_impact", "")),
                )
                runtime_results[paper["url"]] = result
                refined_papers.append(llm_processor.prepare_report_item(paper, result))
            else:
                refined_papers.append(paper)
        papers = apply_preference_scores(refined_papers, preference_config)
        papers = limit_papers_by_topic(papers, paper_topic_limits, paper_limit)

    deduped_updates = dedupe_updates(
        [item for item in prepared_items if item.get("content_type") != "paper"],
        llm_processor,
    )
    updates = filter_updates_for_report(
        deduped_updates,
        web_limit,
        min_web_items,
        source_preferences=source_preferences,
        preference_config=preference_config,
    )
    papers = diversify_report_titles(papers, llm_processor)
    updates = diversify_report_titles(updates, llm_processor)
    print(f"Dedupe reduced dynamic items from {len([item for item in prepared_items if item.get('content_type') != 'paper'])} to {len(deduped_updates)} candidates.")
    if len(updates) < min_web_items:
        print(f"Warning: only {len(updates)} web updates available after dedupe, below target {min_web_items}.")
    if not papers and not updates:
        print("No processed items available for the report.")
        run_result.update(
            {
                "status": "no_content",
                "retryable": False,
                "finished_at": datetime.now().isoformat(timespec="seconds"),
                "paper_count": 0,
                "update_count": 0,
                "alert_summary": {},
            }
        )
        return run_result

    for item in papers + updates:
        item["impact_tag"] = infer_impact_tag(
            item.get("title_cn") or item.get("title", ""),
            item.get("summary", ""),
            item.get("why_now", ""),
            item.get("expected_effect", ""),
            item.get("future_impact", ""),
            item.get("display_topic", ""),
        )

    print(f"Report contains {len(papers)} papers and {len(updates)} web updates.")
    run_result["paper_count"] = len(papers)
    run_result["update_count"] = len(updates)
    generator = ReportGenerator()
    mixed_items = generator.build_mixed_items(papers, updates)
    report_summary = llm_processor.summarize_report(papers, updates)
    trend_summary = {"items": []}
    if trend_config.get("enabled", True):
        lookback_days = int(trend_config.get("lookback_days", 3))
        trend_summary = build_trend_summary_v2(
            db.get_recent_processed_articles_since(hours=lookback_days * 24, limit=600),
            lookback_days=lookback_days,
            max_items=int(trend_config.get("max_items", 5)),
            min_occurrences=int(trend_config.get("min_occurrences", 2)),
        )
    if alert_config.get("enabled", True):
        alert_summary = build_alert_summary_v2(
            collector_runs,
            db,
            len(updates),
            paper_count=len(papers),
            update_candidate_count=len(deduped_updates),
            alert_config=alert_config,
        )
    else:
        alert_summary = build_suppressed_alert_summary("disabled_by_runtime_profile")
    run_result["alert_summary"] = alert_summary
    archive_summary = {}
    if archive_config.get("enabled", True):
        archive_summary = build_archive_summary_v2(
            output_html=str(archive_config.get("output_html", "reports_index.html")),
            output_markdown=str(archive_config.get("output_markdown", "reports_index.md")),
            report_dir=str(archive_config.get("report_dir", "archive")),
        )
    report_title = config["report"].get("title", "AI Frontier Intelligence Daily")
    html_report = generator.generate_html(
        papers=papers,
        updates=updates,
        mixed_items=mixed_items,
        report_summary=report_summary,
        title=report_title,
        collector_summary=collector_summary,
        trend_summary=trend_summary,
        alert_summary=alert_summary,
        archive_summary=archive_summary,
    )
    file_stamp = datetime.now().strftime("%Y%m%d_%H%M")
    report_dir = Path(str(archive_config.get("report_dir", "archive")))
    report_dir.mkdir(parents=True, exist_ok=True)
    html_path = report_dir / f"report_{file_stamp}.html"
    with open(html_path, "w", encoding="utf-8") as file:
        file.write(html_report)
    html_filename = html_path.as_posix()
    run_result["html_report_path"] = html_filename
    print(f"HTML Report saved to {html_filename}")
    markdown_report = generator.generate_markdown(
        papers=papers,
        updates=updates,
        mixed_items=mixed_items,
        report_summary=report_summary,
        title=report_title,
        collector_summary=collector_summary,
        trend_summary=trend_summary,
        alert_summary=alert_summary,
        archive_summary=archive_summary,
    )
    markdown_path = report_dir / f"report_{file_stamp}.md"
    with open(markdown_path, "w", encoding="utf-8") as file:
        file.write(markdown_report)
    markdown_filename = markdown_path.as_posix()
    run_result["markdown_report_path"] = markdown_filename
    print(f"Markdown Report saved to {markdown_filename}")
    if archive_config.get("enabled", True):
        update_archive_manifest(
            html_filename=html_filename,
            markdown_filename=markdown_filename,
            report_summary=report_summary,
            papers=papers,
            updates=updates,
        )
        archive_summary = build_archive_summary_v2(
            output_html=str(archive_config.get("output_html", "reports_index.html")),
            output_markdown=str(archive_config.get("output_markdown", "reports_index.md")),
            report_dir=str(archive_config.get("report_dir", "archive")),
        )

    print("\n=== Step 4: Sending Notification ===")
    recipient = os.getenv("EMAIL_RECIPIENT")
    sender = os.getenv("EMAIL_SENDER")
    password = os.getenv("EMAIL_PASSWORD")
    smtp_server = os.getenv("EMAIL_SMTP_SERVER", "smtp.gmail.com")
    smtp_port = os.getenv("EMAIL_SMTP_PORT", "587")
    email_timeout = int(config.get("email", {}).get("timeout_seconds", network_timeout))
    email_max_attempts = int(config.get("email", {}).get("max_attempts", 3))
    email_retry_delay = int(config.get("email", {}).get("retry_delay_seconds", 5))
    notification_sent = False
    notification_skipped = False
    notification_dry_run = False
    email_mode = str(os.getenv("WEB_AGENT_EMAIL_MODE", "send") or "send").strip().lower()
    if email_mode in {"dry-run", "dry_run", "skip", "disabled"}:
        print(f"Skipping email notification because WEB_AGENT_EMAIL_MODE={email_mode}.")
        notification_dry_run = True
    elif recipient and sender and password:
        notifier = EmailNotifier(
            smtp_server=smtp_server,
            smtp_port=int(smtp_port),
            sender_email=sender,
            sender_password=password,
            timeout_seconds=email_timeout,
            max_attempts=email_max_attempts,
            retry_delay_seconds=email_retry_delay,
        )
        subject = f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}] {report_title}"
        success = notifier.send_email(recipient_email=recipient, subject=subject, html_content=html_report)
        if success:
            print("Notification sent successfully.")
            notification_sent = True
            if alert_config.get("enabled", True) and alert_config.get("send_separate_alert", True) and alert_summary.get("needs_alert"):
                alert_subject = f"[异常提醒] {datetime.now().strftime('%Y-%m-%d %H:%M')} {report_title}"
                alert_html = render_alert_email_html(report_title, run_id, alert_summary.get("issues", []))
                alert_sent = notifier.send_email(recipient_email=recipient, subject=alert_subject, html_content=alert_html)
                if alert_sent:
                    print("Alert notification sent successfully.")
                else:
                    print("Failed to send alert notification.")
        else:
            print("Failed to send notification.")
            if alert_summary.get("needs_alert"):
                alert_file = report_dir / f"alert_{file_stamp}.html"
                Path(alert_file).write_text(render_alert_email_html(report_title, run_id, alert_summary.get("issues", [])), encoding="utf-8")
                print(f"Saved alert details to {alert_file}")
                run_result["alert_file_path"] = Path(alert_file).as_posix()
    else:
        print("Skipping email notification (credentials missing in .env).")
        notification_skipped = True
    print("\n=== All Tasks Completed ===")
    run_result.update(resolve_delivery_outcome(notification_sent, notification_skipped, notification_dry_run))
    run_result["finished_at"] = datetime.now().isoformat(timespec="seconds")
    return run_result


if __name__ == "__main__":
    main_result = main()
    if isinstance(main_result, dict) and not main_result.get("success", False):
        sys.exit(1)
