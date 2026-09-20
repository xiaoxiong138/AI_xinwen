from __future__ import annotations

import os
import multiprocessing as mp
import hashlib
import json
import re
import socket
import sys
import copy
import html as html_lib
import unicodedata
from collections import Counter
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import parse_qsl, urlencode, unquote, urlparse, urlsplit, urlunsplit
from urllib.request import urlopen

import yaml
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.collectors import (
    ArxivCollector,
    CodexResearchInboxCollector,
    SUPPORTED_CODEX_RESEARCH_ANALYSIS_VERSIONS,
    build_codex_research_inbox_collector,
    build_codex_research_readiness_summary,
    evaluate_sent_history_overlap,
    HuggingFaceCollector,
    RSSCollector,
    WebSearchCollector,
)
from src.continuity import (
    build_closing_memory,
    build_topic_dossiers,
    build_weekly_digest,
    enrich_continuity,
    reading_queue_context,
)
from src.database import Database, resolve_database_path
from src.editorial_engine import (
    attribution_opener_pattern,
    build_editorial_quality_metrics,
    contains_mojibake,
    enrich_editorial_fields,
    has_field_label_leak,
    has_untranslated_prose,
    mixed_language_title,
    paper_plain_summary_passes,
    paper_technical_intro_passes,
)
from src.generator import ReportGenerator, editorial_item_render_key, editorial_source_identity
from src.notifier import EmailNotifier, resolve_imap_server, verify_email_arrival
from src.ui_audit import run_email_ui_audit

EMAIL_COMMITTED_MARKER = "__SCHEDULER_EMAIL_COMMITTED__="
from src.processors import LLMProcessor
from src.relevance import infer_impact_tag, infer_source_tier, is_low_signal_update, normalize_text, score_preference_boost, score_update_quality

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
PERSISTENT_QUALITY_FLAGS = {"supplemental_older_source"}
GENERIC_SUMMARY_PATTERNS = (
    r"相关机构",
    r"出现了新的动作",
    r"不只是单点更新",
    r"可能影响产品路线",
    r"值得持续关注",
    r"未来可能带来影响",
)
HIGH_STAKES_CLAIM_PATTERNS = (
    r"\b(acquire|acquires|acquired|buy|buys|bought|merger|ipo)\b",
    r"收购|并购|上市|IPO",
    r"\b(raise|raises|raised|funding|series [a-z])\b",
    r"融资|募资",
    r"\$ ?\d+(?:\.\d+)?\s?(?:b|bn|billion|m|million)",
    r"\d+(?:\.\d+)?\s?(?:亿美元|千万美元|百万美元|亿元)",
)
TRUSTED_CLAIM_HOSTS = {
    "techcrunch.com",
    "reuters.com",
    "bloomberg.com",
    "theinformation.com",
    "wsj.com",
    "cnbc.com",
    "fortune.com",
    "forbes.com",
    "businesswire.com",
    "prnewswire.com",
}
TRUSTED_CLAIM_SOURCE_HINTS = (
    "official",
    "press release",
    "techcrunch",
    "reuters",
    "bloomberg",
    "the information",
    "wall street journal",
    "businesswire",
    "pr newswire",
)

V10_DESIGN_VERSION = "v10-learning-digest"
V11_DESIGN_VERSION = "v11-editorial-library"
LEARNING_DIGEST_DESIGNS = {V10_DESIGN_VERSION, V11_DESIGN_VERSION}
CONTINUOUS_READER_DESIGNS = {
    "v8-editorial-reader",
    "v9-continuous-learning",
    *LEARNING_DIGEST_DESIGNS,
}
V11_ACCEPTANCE_CONTRACT_VERSION = 12


def is_continuous_reader_design(value: Any) -> bool:
    if isinstance(value, dict):
        value = value.get("design_version")
    return str(value or "") in CONTINUOUS_READER_DESIGNS


def is_learning_digest_design(value: Any) -> bool:
    if isinstance(value, dict):
        value = value.get("design_version")
    return str(value or "") in LEARNING_DIGEST_DESIGNS


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


def apply_environment_path_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    """Redirect generated artifacts without mutating the production config file."""
    runtime_config = copy.deepcopy(config)
    archive_config = runtime_config.setdefault("archive", {})
    scheduler_config = runtime_config.setdefault("scheduler", {})

    path_overrides = (
        ("WEB_AGENT_REPORT_DIR", archive_config, "report_dir"),
        ("WEB_AGENT_ARCHIVE_OUTPUT_HTML", archive_config, "output_html"),
        ("WEB_AGENT_ARCHIVE_OUTPUT_MARKDOWN", archive_config, "output_markdown"),
        ("WEB_AGENT_UI_AUDIT_OUTPUT_DIR", scheduler_config, "ui_audit_output_dir"),
    )
    for env_name, section, key in path_overrides:
        value = str(os.getenv(env_name, "") or "").strip()
        if value:
            section[key] = value

    archive_enabled = str(os.getenv("WEB_AGENT_ARCHIVE_ENABLED", "") or "").strip().lower()
    if archive_enabled:
        archive_config["enabled"] = archive_enabled in {"1", "true", "yes", "on"}
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
        return ladder
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


def build_collector_failure_record(
    collector: Any,
    label: str,
    error: Any,
    duration_seconds: float,
) -> Dict[str, Any]:
    error_text = str(error or "")
    return {
        "label": label,
        "status": "timeout" if "Timed out" in error_text else "error",
        "inserted_count": 0,
        "collected_count": 0,
        "duration_seconds": round(float(duration_seconds or 0.0), 1),
        "error": error_text,
        "diagnostics": dict(getattr(collector, "fetch_diagnostics", {}) or {}),
    }


def backfill_sent_report_delivery_status(db: Database, slot_dir: Path) -> int:
    sent_records: List[Dict[str, Any]] = []
    if not slot_dir.exists():
        return 0
    for path in slot_dir.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            continue
        if str(payload.get("status", "") or "") != "sent":
            continue
        sent_records.append({
            "run_id": payload.get("run_id", ""),
            "html_report_path": payload.get("html_report_path", ""),
            "delivery_at": payload.get("finished_at", ""),
        })
    return db.mark_report_runs_sent(sent_records)


def build_collector_summary(collector_runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    success = [item for item in collector_runs if item.get("status") == "success"]
    empty = [item for item in collector_runs if item.get("status") == "empty"]
    fresh_items = sum(int(item.get("inserted_count", 0)) for item in collector_runs)
    timeout_count = sum(1 for item in collector_runs if item.get("status") == "timeout")
    failed_count = sum(1 for item in collector_runs if item.get("status") == "error")
    skipped_count = sum(1 for item in collector_runs if item.get("status") == "skipped")
    arxiv_zero_result_warning_count = sum(
        1
        for item in collector_runs
        if str(item.get("label", "")).startswith("ArxivCollector[") and item.get("status") == "empty"
    )
    arxiv_true_zero_result_count = sum(
        1
        for item in collector_runs
        if str(item.get("label", "")).startswith("ArxivCollector[")
        and item.get("status") == "empty"
        and bool((item.get("diagnostics") or {}).get("true_zero_result", False))
    )
    arxiv_no_match_result_count = sum(
        1
        for item in collector_runs
        if str(item.get("label", "")).startswith("ArxivCollector[")
        and item.get("status") == "empty"
        and bool((item.get("diagnostics") or {}).get("no_match_result", False))
    )
    arxiv_rows = [
        item for item in collector_runs
        if str(item.get("label", "")).startswith("ArxivCollector[")
    ]
    arxiv_http_error_count = sum(
        int((item.get("diagnostics") or {}).get("request_error_count", 0) or 0)
        for item in arxiv_rows
    )
    arxiv_parse_error_count = sum(
        int((item.get("diagnostics") or {}).get("page_parse_error_count", 0) or 0)
        for item in arxiv_rows
    )
    arxiv_fallback_recovery_count = sum(
        1
        for item in arxiv_rows
        if (
            int((item.get("diagnostics") or {}).get("request_error_count", 0) or 0)
            or int((item.get("diagnostics") or {}).get("page_parse_error_count", 0) or 0)
        )
        and int((item.get("diagnostics") or {}).get("successful_page_count", 0) or 0)
    )
    arxiv_retry_paths = [
        {
            "source": str(item.get("label", "")),
            **dict(path),
        }
        for item in arxiv_rows
        for path in ((item.get("diagnostics") or {}).get("retry_paths") or [])
        if len(path.get("attempted_show_counts") or []) > 1
    ]
    gpt_search_rows = [
        item for item in collector_runs
        if str(item.get("label", "")).startswith("OpenAIWebSearchCollector")
    ]
    gpt_search_request_count = sum(
        int((item.get("diagnostics") or {}).get("request_count", 0) or 0)
        for item in gpt_search_rows
    )
    gpt_search_success_count = sum(
        int((item.get("diagnostics") or {}).get("success_count", 0) or 0)
        for item in gpt_search_rows
    )
    gpt_search_schema_error_count = sum(
        int((item.get("diagnostics") or {}).get("schema_error_count", 0) or 0)
        for item in gpt_search_rows
    )
    status_text = (
        f"本轮共运行 {len(collector_runs)} 个采集单元，成功 {len(success)} 个，零结果 {len(empty)} 个，"
        f"超时 {timeout_count} 个，失败 {failed_count} 个，跳过 {skipped_count} 个，新入库 {fresh_items} 条。"
    )
    return {
        "status_text": status_text,
        "fresh_items": fresh_items,
        "success_count": len(success),
        "empty_count": len(empty),
        "timeout_count": timeout_count,
        "failed_count": failed_count,
        "skipped_count": skipped_count,
        "arxiv_zero_result_warning_count": arxiv_zero_result_warning_count,
        "arxiv_true_zero_result_count": arxiv_true_zero_result_count,
        "arxiv_no_match_result_count": arxiv_no_match_result_count,
        "arxiv_http_error_count": arxiv_http_error_count,
        "arxiv_parse_error_count": arxiv_parse_error_count,
        "arxiv_fallback_recovery_count": arxiv_fallback_recovery_count,
        "arxiv_retry_paths": arxiv_retry_paths,
        "gpt_search_request_count": gpt_search_request_count,
        "gpt_search_success_count": gpt_search_success_count,
        "gpt_search_schema_error_count": gpt_search_schema_error_count,
        "rows": collector_runs,
    }


def build_source_health_summary(
    collector_runs: List[Dict[str, Any]],
    db: Database,
    history_limit: int = 160,
    active_labels: Optional[set[str]] = None,
) -> Dict[str, Any]:
    recent_rows = db.get_recent_collector_runs(limit=history_limit)
    labels = sorted(
        label
        for label in {
            str(row.get("label", ""))
            for row in recent_rows + collector_runs
            if row.get("label")
        }
        if active_labels is None or label in active_labels
    )
    current_by_label = {str(row.get("label", "")): row for row in collector_runs}
    rows: List[Dict[str, Any]] = []

    for label in labels:
        history = [row for row in recent_rows if str(row.get("label", "")) == label]
        actual_history = [row for row in history if row.get("status") != "skipped"]
        consecutive_failures = 0
        for row in actual_history:
            if row.get("status") in {"success", "empty"}:
                break
            consecutive_failures += 1

        success_rows = [row for row in actual_history if row.get("status") == "success"]
        failure_rows = [row for row in actual_history if row.get("status") not in {"success", "empty", "skipped"}]
        empty_success_rows = [
            row
            for row in actual_history
            if row.get("status") == "empty"
            or (
                row.get("status") == "success"
                and int(row.get("collected_count", 0) or 0) == 0
                and int(row.get("inserted_count", 0) or 0) == 0
            )
        ]
        current = current_by_label.get(label, {})
        rows.append(
            {
                "label": label,
                "current_status": current.get("status", ""),
                "current_inserted_count": int(current.get("inserted_count", 0) or 0),
                "current_collected_count": int(current.get("collected_count", 0) or 0),
                "recent_runs": len(history),
                "recent_success_count": len(success_rows),
                "recent_failure_count": len(failure_rows),
                "recent_empty_success_count": len(empty_success_rows),
                "consecutive_failures": consecutive_failures,
                "last_success_at": str(success_rows[0].get("created_at", "")) if success_rows else "",
                "last_status": str(history[0].get("status", "")) if history else current.get("status", ""),
                "last_error": str(history[0].get("error", "")) if history else current.get("error", ""),
            }
        )

    risky_rows = [
        row
        for row in rows
        if row["consecutive_failures"] > 0 or row["current_status"] in {"error", "timeout", "skipped"}
    ]
    unstable_rows = [
        row
        for row in rows
        if row["consecutive_failures"] >= 2
        or (
            row["recent_failure_count"] >= 2
            and row["recent_failure_count"] / max(1, row["recent_runs"]) >= 0.5
        )
        or row["recent_empty_success_count"] >= 5
    ]
    return {
        "history_limit": history_limit,
        "source_count": len(rows),
        "risky_source_count": len(risky_rows),
        "unstable_source_count": len(unstable_rows),
        "rows": rows,
        "risky_rows": risky_rows[:10],
        "unstable_rows": unstable_rows[:10],
    }


def build_quality_diagnostics(
    *,
    current_papers_count: int,
    current_updates_count: int,
    report_items_count: int,
    prepared_items_count: int,
    paper_candidate_count: int,
    update_candidate_count: int,
    deduped_update_count: int,
    selected_paper_count: int,
    selected_update_count: int,
    paper_limit: int,
    min_paper_count: int,
    web_limit: int,
    min_web_items: int,
    paper_backfill_hours_used: List[int],
    web_backfill_hours_used: List[int],
    collector_summary: Dict[str, Any],
    generic_summary_count: int = 0,
    low_evidence_count: int = 0,
    aggregator_demoted_count: int = 0,
    bad_title_count: int = 0,
    design_version: str = "",
    source_health: Optional[Dict[str, Any]] = None,
    source_weight_adjustments: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    warnings: List[str] = []
    if selected_paper_count < min_paper_count:
        warnings.append(f"selected_papers_below_minimum:{selected_paper_count}/{min_paper_count}")
    if selected_update_count < min_web_items:
        warnings.append(f"selected_updates_below_minimum:{selected_update_count}/{min_web_items}")
    if update_candidate_count and deduped_update_count < update_candidate_count:
        dropped = update_candidate_count - deduped_update_count
        warnings.append(f"dedupe_removed_updates:{dropped}")
    if int(collector_summary.get("failed_count", 0) or 0) or int(collector_summary.get("timeout_count", 0) or 0):
        warnings.append("collector_failures_present")
    if generic_summary_count:
        warnings.append(f"generic_summaries_present:{generic_summary_count}")
    if low_evidence_count:
        warnings.append(f"low_evidence_items_present:{low_evidence_count}")
    if aggregator_demoted_count:
        warnings.append(f"aggregator_items_demoted:{aggregator_demoted_count}")
    if bad_title_count:
        warnings.append(f"bad_titles_present:{bad_title_count}")

    source_health = source_health or {}
    source_weight_adjustments = source_weight_adjustments or {}
    return {
        "report_design_version": design_version or getattr(ReportGenerator, "DESIGN_VERSION", ""),
        "targets": {
            "paper_limit": paper_limit,
            "min_paper_count": min_paper_count,
            "web_limit": web_limit,
            "min_web_items": min_web_items,
        },
        "collection": {
            "current_papers": current_papers_count,
            "current_updates": current_updates_count,
            "fresh_items": int(collector_summary.get("fresh_items", 0) or 0),
            "collector_success_count": int(collector_summary.get("success_count", 0) or 0),
            "collector_empty_count": int(collector_summary.get("empty_count", 0) or 0),
            "collector_failed_count": int(collector_summary.get("failed_count", 0) or 0),
            "collector_timeout_count": int(collector_summary.get("timeout_count", 0) or 0),
            "collector_skipped_count": int(collector_summary.get("skipped_count", 0) or 0),
            "arxiv_zero_result_warning_count": int(
                collector_summary.get("arxiv_zero_result_warning_count", 0) or 0
            ),
            "arxiv_true_zero_result_count": int(
                collector_summary.get("arxiv_true_zero_result_count", 0) or 0
            ),
            "arxiv_no_match_result_count": int(
                collector_summary.get("arxiv_no_match_result_count", 0) or 0
            ),
            "arxiv_http_error_count": int(collector_summary.get("arxiv_http_error_count", 0) or 0),
            "arxiv_parse_error_count": int(collector_summary.get("arxiv_parse_error_count", 0) or 0),
            "arxiv_fallback_recovery_count": int(
                collector_summary.get("arxiv_fallback_recovery_count", 0) or 0
            ),
            "arxiv_retry_paths": list(collector_summary.get("arxiv_retry_paths") or []),
            "gpt_search_request_count": int(collector_summary.get("gpt_search_request_count", 0) or 0),
            "gpt_search_success_count": int(collector_summary.get("gpt_search_success_count", 0) or 0),
            "gpt_search_schema_error_count": int(
                collector_summary.get("gpt_search_schema_error_count", 0) or 0
            ),
        },
        "selection": {
            "report_items": report_items_count,
            "prepared_items": prepared_items_count,
            "paper_candidates": paper_candidate_count,
            "update_candidates_before_dedupe": update_candidate_count,
            "update_candidates_after_dedupe": deduped_update_count,
            "selected_papers": selected_paper_count,
            "selected_updates": selected_update_count,
        },
        "backfill": {
            "paper_hours_used": paper_backfill_hours_used,
            "web_hours_used": web_backfill_hours_used,
        },
        "content_quality": {
            "generic_summary_count": generic_summary_count,
            "low_evidence_count": low_evidence_count,
            "aggregator_demoted_count": aggregator_demoted_count,
            "bad_title_count": bad_title_count,
        },
        "source_health": {
            "source_count": int(source_health.get("source_count", 0) or 0),
            "risky_source_count": int(source_health.get("risky_source_count", 0) or 0),
            "unstable_source_count": int(source_health.get("unstable_source_count", 0) or 0),
            "unstable_rows": list(source_health.get("unstable_rows", []) or [])[:10],
        },
        "source_weight_adjustments": source_weight_adjustments,
        "warnings": warnings,
    }


def summary_looks_generic(text: str) -> bool:
    return any(re.search(pattern, str(text or "")) for pattern in GENERIC_SUMMARY_PATTERNS)


def evidence_quality_value(item: Dict[str, Any]) -> float:
    try:
        return float(item.get("evidence_quality", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def information_density_value(item: Dict[str, Any]) -> float:
    try:
        value = item.get("information_density", item.get("evidence_quality", 0.0))
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def is_google_news_aggregator(item: Dict[str, Any]) -> bool:
    host = urlparse(str(item.get("url", "") or "")).netloc.lower()
    source_detail = str(item.get("source_detail", "") or "").lower()
    return host == "news.google.com" or "google news" in source_detail


def title_uses_truncated_source_prefix(item: Dict[str, Any], title: str) -> bool:
    source_title = re.sub(r"\s+", " ", str(item.get("title") or "")).strip()
    prefix_match = re.match(r"^([^\u4e00-\u9fff]+)(?=[\u4e00-\u9fff])", str(title or ""))
    if not source_title or not prefix_match:
        return False
    prefix = prefix_match.group(1).rstrip()
    if not prefix or not source_title.lower().startswith(prefix.lower()) or len(prefix) >= len(source_title):
        return False
    return bool(re.match(r"[A-Za-z0-9]", source_title[len(prefix) :]))


def title_looks_bad(item: Dict[str, Any]) -> bool:
    title = str(item.get("title_cn") or item.get("title") or "").strip()
    if not title:
        return True
    if len(title) > 56 or any(marker in title for marker in ("…", "...")):
        return True
    if title_uses_truncated_source_prefix(item, title):
        return True
    if has_untranslated_prose(title):
        return True
    if mixed_language_title(title):
        return True
    if re.search(r"[\u4e00-\u9fff]\s+[\u4e00-\u9fff]", title):
        return True
    if re.search(
        r"(?:增加|减少|提升|降低|扩展|优化|改进|支持|引入|采用|实现|构建|发布|更新|训练|部署)，(?:也|并|但|同时|仍)",
        title,
    ):
        return True
    if any(
        phrase in title
        for phrase in (
            "行业资源配置",
            "开始出现新变化",
            "正在重排资源",
            "AI领域新进展",
            "据把AI能力推进",
            "发布产品更新",
            "更新行业动态",
            "产品或技术变化",
            "这项进展",
            "要解决的是",
            "核心动作是",
            "机器人的新方法与实验",
        )
    ):
        return True
    if len(title) >= 18:
        repeated_bigrams = Counter(
            title[index : index + 2]
            for index in range(0, max(0, len(title) - 1))
            if re.search(r"[\u4e00-\u9fff]{2}", title[index : index + 2])
        )
        if any(count >= 3 for count in repeated_bigrams.values()):
            return True
        for size in (4, 5, 6):
            chunks = [title[index : index + size] for index in range(0, max(0, len(title) - size + 1))]
            meaningful = [chunk for chunk in chunks if len(set(chunk)) > 1 and re.search(r"[\u4e00-\u9fff]", chunk)]
            if len(meaningful) != len(set(meaningful)):
                return True
    ascii_letters = len(re.findall(r"[A-Za-z]", title))
    cjk_chars = len(re.findall(r"[\u4e00-\u9fff]", title))
    has_chinese_action = bool(re.search(r"发布|推出|部署|合作|融资|提出|展示|表示|报道|披露|开源|更新", title))
    if ascii_letters >= 18 and cjk_chars < 4 and not has_chinese_action:
        return True
    if re.search(r"\b(this paper|demonstr|propos|publish|launch)\b", title, re.IGNORECASE) and cjk_chars < 8:
        return True
    return False


def build_source_grounded_news_brief(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Build a no-inference news card from the source title and source excerpt."""
    if str(item.get("content_type") or "news").lower() == "paper":
        return None
    source_tier = str(item.get("source_tier") or source_tier_for_item(item)).lower()
    if source_tier in {"aggregator", "low_signal"} or is_google_news_aggregator(item):
        return None

    def clean_source_text(value: Any) -> str:
        text = html_lib.unescape(str(value or ""))
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\[?&#?8230;?\]?", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+The post .+? appeared first on .+?[.]?$", "", text, flags=re.IGNORECASE)
        return re.sub(r"\s+", " ", text).strip(" -|·")

    source_title = clean_source_text(item.get("title"))
    source_body = clean_source_text(item.get("content"))
    if not source_title or contains_mojibake(source_title) or len(source_title) < 12:
        return None
    if not source_body or contains_mojibake(source_body):
        return None

    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?。！？])\s+", source_body)
        if sentence.strip()
    ]
    evidence_sentences: List[str] = []
    normalized_title = normalize_text(source_title)
    for sentence in sentences:
        if len(sentence) < 28:
            continue
        if SequenceMatcher(None, normalized_title, normalize_text(sentence)).ratio() >= 0.88:
            continue
        evidence_sentences.append(sentence)
        if len(" ".join(evidence_sentences)) >= 150 or len(evidence_sentences) >= 2:
            break
    source_excerpt = " ".join(evidence_sentences) or source_body
    if len(source_excerpt) < 36 or len(re.findall(r"[A-Za-z\u4e00-\u9fff]", source_excerpt)) < 24:
        return None
    if len(source_excerpt) > 260:
        clipped = source_excerpt[:260]
        boundary = max(clipped.rfind(". "), clipped.rfind("。"), clipped.rfind("; "), clipped.rfind("；"))
        source_excerpt = clipped[: boundary + 1] if boundary >= 120 else re.sub(r"\s+\S*$", "", clipped).rstrip(" ,;:") + "…"

    candidate = dict(item)
    candidate.update(
        {
            "source_tier": source_tier,
            "source_grounded_brief": True,
            "source_display_title": source_title[:180],
            "source_excerpt": source_excerpt,
            "brief_line": source_excerpt,
            "summary_quality_tier": "source_brief",
            "quality_tier": "brief",
            "report_section": "brief",
        }
    )
    return candidate


def build_content_quality_counts(
    selected_items: List[Dict[str, Any]],
    update_candidates: List[Dict[str, Any]],
    source_preferences: Optional[Dict[str, Any]] = None,
) -> Dict[str, int]:
    generic_summary_count = sum(1 for item in selected_items if summary_looks_generic(item.get("summary", "")))
    low_evidence_count = sum(1 for item in selected_items if evidence_quality_value(item) < 0.35)
    bad_title_count = sum(
        1
        for item in selected_items
        if not item.get("source_grounded_brief") and title_looks_bad(item)
    )
    aggregator_demoted_count = sum(
        1
        for item in update_candidates
        if is_google_news_aggregator(item)
        and is_low_signal_update(
            item.get("title", ""),
            item.get("summary", "") or item.get("content", ""),
            item.get("url", ""),
            item.get("platform", ""),
            item.get("source_detail", ""),
            item.get("category", ""),
            source_preferences=source_preferences,
        )
    )
    return {
        "generic_summary_count": generic_summary_count,
        "low_evidence_count": low_evidence_count,
        "aggregator_demoted_count": aggregator_demoted_count,
        "bad_title_count": bad_title_count,
    }


def source_tier_for_item(item: Dict[str, Any], source_preferences: Optional[Dict[str, Any]] = None) -> str:
    return infer_source_tier(item, source_preferences)


def item_host(item: Dict[str, Any]) -> str:
    host = urlparse(str(item.get("canonical_url") or item.get("url", "") or "")).netloc.lower()
    return host[4:] if host.startswith("www.") else host


def has_high_stakes_claim(item: Dict[str, Any]) -> bool:
    facts = item.get("facts") if isinstance(item.get("facts"), dict) else {}
    evidence = facts.get("evidence") if isinstance(facts, dict) else []
    if isinstance(evidence, str):
        evidence = [evidence]
    text = normalize_text(
        item.get("title", ""),
        item.get("title_cn", ""),
        item.get("summary", ""),
        item.get("summary_preview", ""),
        facts.get("action", "") if isinstance(facts, dict) else "",
        facts.get("target", "") if isinstance(facts, dict) else "",
        " ".join(str(point or "") for point in (evidence or [])),
    )
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in HIGH_STAKES_CLAIM_PATTERNS)


def claim_source_is_trusted(item: Dict[str, Any]) -> bool:
    source_tier = str(item.get("source_tier") or "").lower()
    if source_tier in {"official", "research", "primary"}:
        return True
    host = item_host(item)
    if host in TRUSTED_CLAIM_HOSTS or any(host.endswith(f".{trusted}") for trusted in TRUSTED_CLAIM_HOSTS):
        return True
    source_text = normalize_text(item.get("source_detail", ""), item.get("platform", ""))
    return any(hint in source_text for hint in TRUSTED_CLAIM_SOURCE_HINTS)


def suspicious_claim(item: Dict[str, Any]) -> bool:
    if not has_high_stakes_claim(item):
        return False
    if claim_source_is_trusted(item):
        return False
    if item.get("content_type") == "paper":
        return False
    return True


def item_quality_flags(item: Dict[str, Any]) -> List[str]:
    flags = [
        str(flag)
        for flag in (item.get("quality_flags") or [])
        if str(flag) in PERSISTENT_QUALITY_FLAGS
    ]
    if item.get("source_grounded_brief"):
        if is_google_news_aggregator(item):
            flags.append("aggregator_source")
        if contains_mojibake(str(item.get("source_display_title") or "")) or contains_mojibake(
            str(item.get("source_excerpt") or "")
        ):
            flags.append("mojibake_suspect")
        return sorted(set(flags))
    facts = item.get("facts") or {}
    if not isinstance(facts, dict) or not facts.get("who") or not facts.get("action") or not facts.get("target"):
        flags.append("missing_facts")
    if evidence_quality_value(item) < 0.45:
        flags.append("low_evidence")
    if information_density_value(item) < 0.45:
        flags.append("low_density")
    if summary_looks_generic(str(item.get("summary", "") or "")):
        flags.append("generic_summary")
    if is_google_news_aggregator(item):
        flags.append("aggregator_source")
    if not item.get("_codex_research_validated") and title_fact_mismatch(
        item, facts if isinstance(facts, dict) else {}
    ):
        flags.append("title_fact_mismatch")
    if title_looks_bad(item):
        flags.append("bad_title")
    if mixed_language_title(item.get("editorial_title") or item.get("title_cn") or item.get("title")):
        flags.append("mixed_language_title")
    if has_field_label_leak(" ".join(str(item.get(key, "") or "") for key in ("editorial_title", "editorial_lead", "analysis_body", "evidence_line", "paper_technical_intro"))):
        flags.append("field_label_leak")
    for editorial_flag in item.get("editorial_flags") or []:
        if editorial_flag in {"title_fact_mismatch", "mixed_language_title", "bad_title"}:
            continue
        if editorial_flag not in flags:
            flags.append(str(editorial_flag))
    if suspicious_claim(item):
        flags.append("suspicious_claim")
    if item.get("report_section") in {"must_read", "physical_ai", "watch", "featured_papers"} and not learning_card_complete(item):
        flags.append("learning_card_missing")
    return sorted(set(flags))


def title_fact_mismatch(item: Dict[str, Any], facts: Dict[str, Any]) -> bool:
    title_text = normalize_text(
        str(item.get("title", "") or ""),
        str(item.get("title_cn", "") or ""),
        str(item.get("editorial_title", "") or ""),
    )
    summary_text = normalize_text(str(item.get("summary", "") or ""))
    if not title_text or not facts:
        return False
    fact_parts = [
        str(facts.get("who", "") or ""),
        str(facts.get("action", "") or ""),
        str(facts.get("target", "") or ""),
    ]
    fact_parts = [normalize_text(part) for part in fact_parts if len(normalize_text(part)) >= 3]
    if item.get("content_type") == "paper":
        who = normalize_text(str(facts.get("who", "") or ""))
        who_short_text = re.sub(
            r"[（(](?:论文)?(?:方法|模型|系统)[）)]$",
            "",
            _compact_entity_piece(facts.get("who", "")),
        ).strip()
        who_short = normalize_text(who_short_text)
        target_clause = re.split(r"[，,；;。]", str(facts.get("target", "") or ""), maxsplit=1)[0]
        target = normalize_text(target_clause)
        target_short = normalize_text(_compact_paper_target(target_clause))
        target_prefix = normalize_text(
            _compact_paper_target(target_clause, max_len=12).rstrip("，,；;：:")
        )
        target_matches = bool(
            target in title_text
            or (target_short and target_short in title_text)
            or (target_prefix and target_prefix in title_text)
        )
        if not target_matches:
            target_cjk = "".join(re.findall(r"[\u4e00-\u9fff]", target))
            title_cjk = "".join(re.findall(r"[\u4e00-\u9fff]", title_text))
            target_bigrams = {
                target_cjk[index : index + 2]
                for index in range(max(0, len(target_cjk) - 1))
            }
            matched_bigrams = sum(1 for token in target_bigrams if token in title_cjk)
            target_matches = bool(
                target_bigrams
                and matched_bigrams >= max(2, int(len(target_bigrams) * 0.45))
            )
        if target_matches:
            return False
        if (
            who
            and target
            and (who in title_text or who_short in title_text)
            and target_matches
        ):
            return False
    if fact_parts:
        matched_parts = sum(1 for part in fact_parts if part in title_text)
        if matched_parts >= min(2, len(fact_parts)):
            return False
    title_tokens = [
        token
        for token in re.split(r"\W+", title_text)
        if len(token) >= 4 and token not in EVENT_STOPWORDS
    ]
    if len(title_tokens) < 4:
        return False
    evidence = facts.get("evidence") or []
    if isinstance(evidence, list):
        evidence_text = " ".join(str(part or "") for part in evidence[:4])
    else:
        evidence_text = str(evidence or "")
    fact_text = normalize_text(" ".join(fact_parts), evidence_text, summary_text)
    matched_tokens = sum(1 for token in title_tokens[:8] if token in fact_text)
    return matched_tokens < 2


def _compact_title_piece(value: Any, max_len: int = 18) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", text)
    text = re.sub(r"[。！？.!?].*$", "", text).strip()
    return text[:max_len]


def _compact_entity_piece(value: Any, max_len: int = 24) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    acronym = re.match(r"^([A-Za-z][A-Za-z0-9$^+._-]{1,20})\s*\(", text)
    if acronym:
        return acronym.group(1)
    if len(text) <= max_len:
        return text
    candidate = text[:max_len]
    if max_len < len(text) and re.match(r"[A-Za-z0-9]", text[max_len]) and re.search(r"[A-Za-z0-9]$", candidate):
        candidate = re.sub(r"[A-Za-z0-9$^+._-]+$", "", candidate).rstrip(" -_(")
    return candidate or text.split(" ", 1)[0][:max_len]


def _compact_paper_target(value: Any, max_len: int = 24) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    text = re.sub(
        r"[A-Za-z][A-Za-z\- ]{4,}\s*\(([A-Z][A-Z0-9-]{1,10})\)",
        lambda match: match.group(1),
        text,
    )
    text = text.replace("在通用", "").replace("中的", "").replace("方面的", "")
    return _compact_title_piece(text, max_len)


def _topic_title_piece(value: Any, max_len: int = 18) -> str:
    text = _compact_title_piece(value, max_len)
    mapping = {
        "Open Source": "开源生态",
        "Product Release": "产品发布",
        "Infrastructure": "基础设施",
        "Industry": "行业动态",
        "Partnership": "企业合作",
        "Physical AI": "具身智能",
        "World Model": "世界模型",
        "Robotics": "机器人",
    }
    return mapping.get(text, text)


def _title_action_cn(action: str, target: str = "") -> str:
    normalized = normalize_text(action)
    target_normalized = normalize_text(target)
    if any(token in normalized for token in ("launch", "release", "发布", "推出")):
        return "发布"
    if any(token in normalized for token in ("propose", "提出")):
        return "提出"
    if any(token in normalized for token in ("partner", "合作")):
        return "合作推进"
    if any(token in normalized for token in ("raise", "fund", "融资")):
        return "完成融资"
    if any(token in normalized for token in ("replace", "layoff", "裁员", "替代")):
        if any(token in target_normalized for token in ("agent", "员工", "employee", "岗位")):
            return "用AI Agent调整岗位"
        return "调整"
    if any(token in normalized for token in ("open source", "开源")):
        return "开源"
    return _compact_title_piece(action, 8) or "更新"


def fact_based_title(item: Dict[str, Any]) -> str:
    facts = item.get("facts_cn") or item.get("facts") or {}
    if not isinstance(facts, dict):
        return ""
    raw_target = re.sub(r"\s+", " ", str(facts.get("target", "") or "")).strip()
    raw_action = str(facts.get("action", "") or "")
    english_target_words = re.findall(r"[A-Za-z][A-Za-z0-9$^+._-]*", raw_target)
    explicit_agent_workforce_change = (
        "agent" in normalize_text(raw_target)
        and any(token in normalize_text(raw_target) for token in ("employee", "worker", "job"))
        and any(token in normalize_text(raw_action) for token in ("replace", "layoff", "replac", "裁员", "替代"))
    )
    if (
        not re.search(r"[\u4e00-\u9fff]", raw_target)
        and len(english_target_words) >= 3
        and not explicit_agent_workforce_change
    ):
        return ""
    who = _compact_entity_piece(facts.get("who"), 24)
    if item.get("content_type") == "paper":
        who = re.sub(r"[（(](?:论文)?(?:方法|模型|系统)[）)]$", "", who).strip()
        target = _compact_paper_target(raw_target, 24)
        if not who or not target:
            return ""
        return f"{who}：{target}"
    action = _title_action_cn(raw_action, raw_target)
    target = _compact_title_piece(raw_target, 24)
    if not who or not target:
        return ""
    target_normalized = normalize_text(target)
    if "ai agent" in target_normalized and "岗位" in action:
        target = "部分岗位"
    title = f"{who}{action}{target}"
    return title if len(title) <= 52 else f"{who}{action}{target[: max(8, 52 - len(who) - len(action))]}"


def repair_title_fact_mismatch(item: Dict[str, Any]) -> Dict[str, Any]:
    facts = item.get("facts") or {}
    if not isinstance(facts, dict):
        return item
    if (
        str(item.get("model_used") or "") == "codex-automation"
        and str(item.get("title_cn") or "").strip()
        and not title_looks_bad(item)
    ):
        return item
    display_title = normalize_text(str(item.get("title_cn", "") or item.get("title", "") or ""))
    fact_parts = [
        normalize_text(str(facts.get("who", "") or "")),
        normalize_text(str(facts.get("action", "") or "")),
        normalize_text(str(facts.get("target", "") or "")),
    ]
    fact_parts = [part for part in fact_parts if len(part) >= 3]
    display_matches = sum(1 for part in fact_parts if part in display_title)
    display_mismatch = bool(fact_parts) and display_matches < min(2, len(fact_parts))
    clear_fact_mismatch = title_fact_mismatch(item, facts)
    if not clear_fact_mismatch and not display_mismatch:
        return item
    repaired_title = fact_based_title(item)
    if not repaired_title:
        return item
    if title_looks_bad({"title_cn": repaired_title}) and not title_looks_bad(item) and not clear_fact_mismatch:
        return item
    updated = dict(item)
    updated["title_cn"] = repaired_title
    return updated


def _trim_repaired_title(text: Any, limit: int = 34) -> str:
    text = re.sub(r"\s+", " ", str(text or "").replace("\n", " ")).strip(" .,;:|-/。；，、")
    return text if len(text) <= limit else text[: limit - 1].rstrip(" .,;:|-/。；，、") + "…"


def _looks_like_natural_chinese_title(text: str) -> bool:
    if not text:
        return False
    cjk_count = len(re.findall(r"[\u4e00-\u9fff]", text))
    if cjk_count < 6:
        return False
    if text.startswith(("关键看", "后续", "下一步", "值得关注", "需关注")):
        return False
    return not any(phrase in text for phrase in ("后续需", "后续要", "值得持续关注", "需要观察", "观察其"))


def _title_from_summary_text(text: Any, limit: int = 34) -> str:
    text = re.sub(r"\s+", " ", str(text or "").replace("\n", " ")).strip()
    if not text or "论文来源" in text or re.match(r"^[A-Za-z0-9\- ()]+论文针对", text):
        return ""
    for marker in ("。", "；", ";"):
        if marker in text:
            text = text.split(marker, 1)[0]
            break
    comma_head = text.split("，", 1)[0].strip() if "，" in text else text
    if len(comma_head) >= 12 and _looks_like_natural_chinese_title(comma_head):
        text = comma_head
    return _trim_repaired_title(text, limit) if _looks_like_natural_chinese_title(text) else ""


def _compressed_title_from_summary(item: Dict[str, Any]) -> str:
    summary = str(item.get("summary", "") or "")
    facts = item.get("facts") if isinstance(item.get("facts"), dict) else {}
    who = str(facts.get("who") or "").strip()
    if "AgentCore payments" in summary or "payments功能" in summary:
        prefix = "Amazon Bedrock AgentCore" if "Amazon Bedrock" in summary or "Amazon Bedrock" in who else who
        return _trim_repaired_title(f"{prefix or 'AgentCore'}发布支付功能预览")
    if "ElevenLabs替代方案" in summary:
        return _trim_repaired_title(f"{who or 'OmniVoice Studio'}发布本地开源语音工具")
    return ""


def suggest_repaired_title_with_source(item: Dict[str, Any]) -> Dict[str, str]:
    compressed = _compressed_title_from_summary(item)
    if compressed and not title_looks_bad({"title_cn": compressed}):
        return {"title": compressed, "source": "domain_compression"}
    fact_title = fact_based_title(item)
    if fact_title and not title_looks_bad({**item, "title_cn": fact_title}):
        return {"title": fact_title, "source": "facts"}
    summary_title = _title_from_summary_text(item.get("summary"))
    if summary_title and not title_looks_bad({**item, "title_cn": summary_title}):
        return {"title": summary_title, "source": "summary"}
    preview_title = _title_from_summary_text(item.get("summary_preview"))
    if preview_title and not title_looks_bad({**item, "title_cn": preview_title}):
        return {"title": preview_title, "source": "summary_preview"}
    facts = item.get("facts") if isinstance(item.get("facts"), dict) else {}
    fact_who = _compact_title_piece(facts.get("who"), 16)
    content_type_for_fact = str(item.get("content_type", "") or "")
    display_topic_for_fact = str(item.get("display_topic") or item.get("topic_cn") or item.get("category") or "")
    if fact_who and fact_who.lower() != "this paper":
        if "论文" in display_topic_for_fact or content_type_for_fact == "paper":
            return {"title": _trim_repaired_title(f"{fact_who}更新AI研究方法与实验结果"), "source": "facts"}
        topic = _topic_title_piece(display_topic_for_fact or "AI产品动态", 12)
        return {"title": _trim_repaired_title(f"{fact_who}更新{topic}"), "source": "facts"}
    title = str(item.get("title") or item.get("title_cn") or "").replace(":", " ").replace("|", " ")
    original_title = _trim_repaired_title(title)
    if original_title and not title_looks_bad({"title_cn": original_title}):
        return {"title": original_title, "source": "original_title"}
    entity_match = re.search(r"\b[A-Z][A-Za-z0-9.-]{2,}\b", title)
    entity = entity_match.group(0) if entity_match else ""
    if not entity:
        facts = item.get("facts") if isinstance(item.get("facts"), dict) else {}
        fact_who = str(facts.get("who") or "").strip()
        if re.match(r"^[A-Z][A-Za-z0-9.-]{2,}$", fact_who):
            entity = fact_who
    content_type = str(item.get("content_type", "") or "")
    display_topic = str(item.get("display_topic") or item.get("topic_cn") or item.get("category") or "")
    if entity and content_type == "paper":
        return {"title": _trim_repaired_title(f"{entity}更新AI研究方法与实验结果"), "source": "safe_fallback"}
    if entity:
        return {"title": _trim_repaired_title(f"{entity}更新AI产品动态"), "source": "safe_fallback"}
    if "论文" in display_topic or content_type == "paper":
        return {"title": "AI论文更新方法与实验结果", "source": "safe_fallback"}
    return {"title": "AI行业动态更新", "source": "safe_fallback"}


def suggest_repaired_title(item: Dict[str, Any]) -> str:
    return suggest_repaired_title_with_source(item)["title"]


REPORT_SECTION_ORDER = ("must_read", "physical_ai", "watch", "featured_papers", "paper_appendix", "brief")
LEARNING_DOMAIN_ORDER = (
    "world_model",
    "physical_ai",
    "agent_models",
    "infra_open_source",
    "products_business",
)
LEARNING_DOMAIN_LABELS = {
    "world_model": "World Model",
    "physical_ai": "Physical AI / Robotics",
    "agent_models": "Agent / Models",
    "infra_open_source": "Infra / Open Source",
    "products_business": "Products / Business",
}


PHYSICAL_AI_TERMS = (
    "physical ai",
    "embodied ai",
    "embodied intelligence",
    "robot",
    "robotics",
    "humanoid",
    "vla",
    "vision-language-action",
    "manipulation",
    "locomotion",
    "warehouse automation",
    "industrial automation",
    "具身",
    "物理ai",
    "物理 ai",
    "机器人",
    "人形机器人",
    "机械臂",
    "操作任务",
    "真实环境",
)


def polish_report_title(title: str) -> str:
    polished = str(title or "").strip()
    if not polished:
        return polished
    replacements = {
        "尝试提升": "提升",
        "尝试降低": "降低",
        "尝试走向": "走向",
        "尝试进入": "进入",
        "尝试推出": "推出",
        "尝试发布": "发布",
        "尝试验证": "验证",
        "尝试解决": "解决",
    }
    for old, new in replacements.items():
        polished = polished.replace(old, new)
    if len(polished) > 10:
        polished = re.sub(r"(?<!不)尝试", "", polished)
    return re.sub(r"\s+", " ", polished).strip()


def is_physical_ai_item(item: Dict[str, Any]) -> bool:
    facts = item.get("facts") if isinstance(item.get("facts"), dict) else {}
    evidence = facts.get("evidence") if isinstance(facts, dict) else []
    if isinstance(evidence, str):
        evidence = [evidence]
    text_parts = [
        item.get("title_cn", ""),
        item.get("title", ""),
        item.get("summary_preview", ""),
        item.get("summary", ""),
        item.get("display_topic", ""),
        item.get("topic_cn", ""),
        item.get("category", ""),
        facts.get("who", "") if isinstance(facts, dict) else "",
        facts.get("action", "") if isinstance(facts, dict) else "",
        facts.get("target", "") if isinstance(facts, dict) else "",
        " ".join(str(point) for point in (evidence or [])),
    ]
    text = " ".join(str(part or "") for part in text_parts).lower()
    return any(term in text for term in PHYSICAL_AI_TERMS)


def item_learning_text(item: Dict[str, Any]) -> str:
    facts = item.get("facts") if isinstance(item.get("facts"), dict) else {}
    evidence = facts.get("evidence") if isinstance(facts, dict) else []
    if isinstance(evidence, str):
        evidence = [evidence]
    return " ".join(
        str(part or "")
        for part in (
            item.get("title_cn", ""),
            item.get("title", ""),
            item.get("summary_preview", ""),
            item.get("summary", ""),
            item.get("display_topic", ""),
            item.get("topic_cn", ""),
            item.get("category", ""),
            item.get("platform", ""),
            facts.get("who", "") if isinstance(facts, dict) else "",
            facts.get("action", "") if isinstance(facts, dict) else "",
            facts.get("target", "") if isinstance(facts, dict) else "",
            " ".join(str(point) for point in (evidence or [])),
        )
    ).lower()


def learning_domain_key(item: Dict[str, Any]) -> str:
    facts = item.get("facts") if isinstance(item.get("facts"), dict) else {}
    paper_domain_key = str(
        item.get("paper_domain_key") or facts.get("paper_domain_key") or ""
    ).strip()
    if item.get("content_type") == "paper" and paper_domain_key in {
        "world_model",
        "physical_ai",
        "agent_models",
        "infra_open_source",
        "other",
    }:
        return paper_domain_key if paper_domain_key != "other" else "products_business"
    text = item_learning_text(item)
    if any(token in text for token in ("world model", "world models", "世界模型", "latent dynamics", "jepa", "video prediction", "predictive model", "rollout")):
        return "world_model"
    if is_physical_ai_item(item) or any(token in text for token in ("robot", "robotics", "humanoid", "manipulation", "locomotion", "具身", "机器人", "机械臂")):
        return "physical_ai"
    if any(token in text for token in ("agent", "workflow", "reasoning", "llm", "gpt", "claude", "gemini", "模型", "智能体", "工作流")):
        return "agent_models"
    if any(token in text for token in ("gpu", "chip", "inference", "datacenter", "open source", "github", "license", "算力", "芯片", "推理", "开源", "基础设施")):
        return "infra_open_source"
    return "products_business"


def paper_quota_domain(item: Dict[str, Any], quotas: Optional[Dict[str, Any]] = None) -> str:
    quota_keys = set((quotas or {}).keys()) or {
        "world_model",
        "physical_ai",
        "agent_models",
        "infra_open_source",
        "other",
    }
    facts = item.get("facts") if isinstance(item.get("facts"), dict) else {}
    locked = str(
        item.get("paper_domain_key") or facts.get("paper_domain_key") or ""
    ).strip()
    if locked in quota_keys:
        return locked
    topic = " ".join(
        str(item.get(key) or "").strip().lower()
        for key in ("topic", "display_topic", "topic_cn")
    )
    if "world model" in topic or "世界模型" in topic:
        domain = "world_model"
    elif any(token in topic for token in ("physical ai", "robotics", "具身", "机器人")):
        domain = "physical_ai"
    elif any(token in topic for token in ("agent / models", "agent/models", "agent", "模型")):
        domain = "agent_models"
    elif any(token in topic for token in ("infra", "efficient ai", "open source", "基础设施", "开源")):
        domain = "infra_open_source"
    elif any(token in topic for token in ("multimodal", "video", "多模态", "视频")):
        domain = "other"
    else:
        domain = str(item.get("domain_key") or learning_domain_key(item))
    return domain if domain in quota_keys else "other"


def _compact_learning_text(text: Any, limit: int = 88) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    cleaned = re.sub(r"^[：:，,\-\s]+", "", cleaned)
    cleaned = re.sub(r"[。！？!?.]+$", "", cleaned)
    if not cleaned:
        return ""
    return cleaned if len(cleaned) <= limit else cleaned[: limit - 1].rstrip("，、；： ") + "…"


def build_learning_fields(item: Dict[str, Any]) -> Dict[str, str]:
    facts = item.get("facts") if isinstance(item.get("facts"), dict) else {}
    evidence = facts.get("evidence") if isinstance(facts, dict) else []
    if isinstance(evidence, str):
        evidence = [evidence]
    evidence = [str(point).strip() for point in (evidence or []) if str(point).strip()]
    who = _compact_learning_text(facts.get("who") if isinstance(facts, dict) else "", 24)
    action = _compact_learning_text(facts.get("action") if isinstance(facts, dict) else "", 24)
    target = _compact_learning_text(facts.get("target") if isinstance(facts, dict) else "", 46)
    title = _compact_learning_text(item.get("title_cn") or item.get("title"), 54)
    domain = learning_domain_key(item)
    subject = who if who and who.lower() not in {"unknown", "this paper", "researchers"} else title or "这条内容"
    if action and target:
        takeaway = f"{subject}{action}{target}"
    elif target:
        takeaway = f"{subject}围绕{target}给出新线索"
    else:
        takeaway = item.get("summary_preview") or title
    takeaway = _compact_learning_text(takeaway, 74)

    text = item_learning_text(item)
    if item.get("content_type") == "paper":
        technical = _compact_learning_text(item.get("summary") or item.get("summary_preview") or target, 118)
        if not re.search(r"方法|模型|训练|实验|评测|基准|参数|成功率|框架|策略|预测|控制|生成|规划", technical, re.IGNORECASE):
            technical = f"这篇论文需要重点看方法、实验设置和结果指标：{_compact_learning_text(target or title, 62)}"
    elif domain == "world_model":
        technical = f"学习重点是它如何把预测、潜空间动态或仿真结果接入规划与决策，而不是只看生成效果。"
    elif domain == "physical_ai":
        technical = f"学习重点是它是否把感知、规划和动作闭环连到真实机器人任务，并给出可核验的实机或部署证据。"
    elif domain == "agent_models":
        technical = f"学习重点是它把模型能力推进到哪类任务执行、推理流程或开发者工作流。"
    elif domain == "infra_open_source":
        technical = f"学习重点是它如何改变算力、推理成本、部署栈或开源生态的默认选择。"
    else:
        technical = f"学习重点是这条变化先影响用户入口、商业化路径、客户采用还是行业资源配置。"
    if evidence:
        evidence_hint = "；".join(_compact_learning_text(point, 48) for point in evidence[:2])
        background = f"事实依据：{evidence_hint}"
    elif evidence_quality_value(item) < 0.35:
        background = "事实依据不足：当前只能作为待确认线索。"
    else:
        background = "事实依据来自标题、来源和正文摘要，仍建议点开原文核对细节。"
    if item.get("content_type") == "paper":
        deep_dive = "继续深挖：先看方法图、实验表格、对照基线、失败案例和是否开源代码。"
    elif "interview" in text or "访谈" in text or "观点" in text:
        deep_dive = "继续深挖：优先看原访谈上下文，区分事实、判断和个人路线偏好。"
    elif "github" in text or "open source" in text or "开源" in text:
        deep_dive = "继续深挖：看许可证、代码活跃度、复现文档和社区采用。"
    else:
        deep_dive = "继续深挖：看原文里的数字、客户、实验或产品细节是否支撑结论。"
    return {
        "domain_key": domain,
        "domain_label": LEARNING_DOMAIN_LABELS.get(domain, LEARNING_DOMAIN_LABELS["products_business"]),
        "learning_takeaway": takeaway,
        "technical_context": _compact_learning_text(technical, 132),
        "background_context": _compact_learning_text(background, 118),
        "deep_dive_prompt": _compact_learning_text(deep_dive, 112),
    }


def enrich_learning_fields(item: Dict[str, Any]) -> Dict[str, Any]:
    enriched = dict(item)
    for key, value in build_learning_fields(enriched).items():
        enriched.setdefault(key, value)
        if not enriched.get(key):
            enriched[key] = value
    return enriched


def learning_card_complete(item: Dict[str, Any]) -> bool:
    facts = item.get("facts") if isinstance(item.get("facts"), dict) else {}
    required = (
        item.get("learning_takeaway"),
        item.get("technical_context"),
        item.get("background_context"),
        item.get("deep_dive_prompt"),
    )
    return bool(facts and all(str(value or "").strip() for value in required))


def report_source_key(item: Dict[str, Any]) -> str:
    facts = item.get("facts") if isinstance(item.get("facts"), dict) else {}
    who = str(facts.get("who", "") if isinstance(facts, dict) else "").strip().lower()
    if who and who not in {"this paper", "unknown", "researchers"}:
        return f"entity:{who}"
    source = str(item.get("source_detail") or item.get("platform") or "").strip().lower()
    if source:
        return f"source:{source}"
    host = urlparse(str(item.get("canonical_url") or item.get("url") or "")).netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    return f"host:{host}" if host else "unknown"


def paper_description_english_leak_count(items: List[Dict[str, Any]]) -> int:
    generator = ReportGenerator()
    count = 0
    for item in items:
        if item.get("content_type") != "paper":
            continue
        description = generator._paper_substantive_description(item)
        long_english_phrases = re.findall(r"\b[A-Za-z][A-Za-z0-9./_-]*(?:\s+[A-Za-z][A-Za-z0-9./_-]*){3,}\b", description)
        if long_english_phrases:
            count += 1
    return count


def paper_substantive_description_fail_count(items: List[Dict[str, Any]]) -> int:
    generator = ReportGenerator()
    failures = 0
    for item in items:
        if item.get("content_type") != "paper":
            continue
        if (
            paper_plain_summary_passes(item.get("paper_plain_summary"))
            and paper_technical_intro_passes(item.get("paper_technical_intro"))
        ):
            continue
        description = generator._paper_substantive_description(item)
        if not generator._paper_description_is_substantive(description):
            failures += 1
    return failures


def build_report_structure_diagnostics(layers: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    must_read = layers.get("must_read", [])
    featured_papers = layers.get("featured_papers", [])
    paper_appendix = layers.get("paper_appendix", [])
    research_fallback = layers.get("research", [])
    visible_papers = featured_papers + paper_appendix
    if not visible_papers:
        visible_papers = research_fallback
    source_counts: Dict[str, int] = {}
    for item in must_read:
        key = report_source_key(item)
        source_counts[key] = source_counts.get(key, 0) + 1
    source_concentration = max(source_counts.values(), default=0)
    all_items = flatten_report_layers(layers)
    if not any(key in layers for key in ("featured_papers", "paper_appendix")):
        all_items = all_items + list(layers.get("research", []))
    tracking_candidates = (
        len(layers.get("must_read", []))
        + len(layers.get("physical_ai", []))
        + len(layers.get("watch", []))
        + len(featured_papers)
        + len(paper_appendix)
        + (0 if featured_papers or paper_appendix else len(research_fallback))
    )
    english_leak_count = paper_description_english_leak_count(all_items)
    paper_description_fail_count = paper_substantive_description_fail_count(visible_papers)
    paper_pass_count = sum(1 for item in visible_papers if paper_core_summary_passes(item))
    paper_fail_count = max(0, len(visible_papers) - paper_pass_count)
    paper_plain_pass_count = sum(
        1 for item in featured_papers if paper_plain_summary_passes(item.get("paper_plain_summary"))
    )
    paper_plain_fail_count = max(0, len(featured_papers) - paper_plain_pass_count)
    title_only_item_count = sum(
        1
        for item in visible_papers
        if not str(
            item.get("paper_plain_summary")
            or item.get("paper_compact_summary")
            or item.get("paper_technical_intro")
            or item.get("analysis_body")
            or item.get("summary")
            or ""
        ).strip()
    )
    visible_v10_items = [item for item in all_items if str(item.get("report_section") or "") != "brief"]
    primary_source_count = sum(
        1
        for item in visible_v10_items
        if infer_source_tier(item).lower() in {"official", "research", "primary"}
    )
    primary_source_ratio = round(primary_source_count / len(visible_v10_items), 3) if visible_v10_items else 1.0
    visible_technical_items = [
        item for item in visible_v10_items if report_primary_section(item) == "technical"
    ]
    technical_primary_source_count = sum(
        1
        for item in visible_technical_items
        if infer_source_tier(item).lower() in {"official", "research", "primary"}
    )
    technical_primary_source_ratio = (
        round(technical_primary_source_count / len(visible_technical_items), 3)
        if visible_technical_items
        else 1.0
    )
    event_keys = [
        normalized_event_title(item) or str(item.get("canonical_url") or item.get("url") or "")
        for item in visible_v10_items
    ]
    event_keys = [key for key in event_keys if key]
    duplicate_event_rate = round(1 - len(set(event_keys)) / len(event_keys), 3) if event_keys else 0.0
    suspicious_count = sum(1 for item in all_items if "suspicious_claim" in set(item.get("quality_flags") or item_quality_flags(item)))
    high_evidence_count = sum(1 for item in all_items if evidence_quality_value(item) >= 0.45)
    high_evidence_warning = bool(all_items and high_evidence_count == len(all_items) and any(item.get("report_section") == "brief" for item in all_items))
    domain_counts: Dict[str, int] = {}
    for item in all_items:
        domain_key = str(item.get("domain_key") or learning_domain_key(item))
        domain_counts[domain_key] = domain_counts.get(domain_key, 0) + 1
    domain_coverage_warning_count = sum(1 for key in LEARNING_DOMAIN_ORDER if domain_counts.get(key, 0) == 0)
    learning_card_missing_count = sum(
        1
        for item in all_items
        if item.get("report_section") in {"must_read", "physical_ai", "watch", "featured_papers"}
        and not learning_card_complete(item)
    )
    technical_context_missing_count = sum(
        1
        for item in all_items
        if item.get("report_section") in {"must_read", "physical_ai", "watch", "featured_papers"}
        and not str(item.get("technical_context") or "").strip()
    )
    editorial_metrics = build_editorial_quality_metrics(all_items)
    physical_ai_count = max(
        len(layers.get("physical_ai", [])),
        int(editorial_metrics.get("physical_ai_featured_count", 0) or 0),
    )
    paper_technical_intro_missing_count = int(editorial_metrics.get("paper_technical_intro_fail_count", 0) or 0)
    visible_information_count = sum(
        1
        for item in (
            list(layers.get("must_read", []))
            + list(layers.get("physical_ai", []))
            + list(layers.get("watch", []))
        )
        if item.get("content_type") != "paper" and str(item.get("quality_tier") or "") != "brief"
    )
    source_news_brief_count = sum(
        1
        for item in layers.get("brief", [])
        if item.get("content_type") != "paper" and item.get("source_grounded_brief")
    )
    primary_section_counts = Counter(report_primary_section(item) for item in all_items)
    unique_section_keys = {
        str(item.get("canonical_url") or item.get("url") or item.get("title_cn") or "")
        for item in all_items
    }
    return {
        "physical_ai_count": physical_ai_count,
        "physical_ai_item_count": physical_ai_count,
        "must_read_source_concentration": source_concentration,
        "must_read_source_counts": source_counts,
        "paper_description_english_leak_count": english_leak_count,
        "paper_plain_summary_english_leak_count": english_leak_count,
        "paper_substantive_description_fail_count": paper_description_fail_count,
        "paper_core_summary_pass_count": paper_pass_count,
        "paper_core_summary_fail_count": paper_fail_count,
        "paper_plain_summary_pass_count": paper_plain_pass_count,
        "paper_plain_summary_fail_count": paper_plain_fail_count,
        "paper_plain_summary_pass_rate": round(paper_plain_pass_count / len(featured_papers), 3) if featured_papers else 1.0,
        "title_only_item_count": title_only_item_count,
        "low_value_module_count": 0,
        "unsupported_claim_count": suspicious_count,
        "primary_source_ratio": primary_source_ratio,
        "technical_primary_source_count": technical_primary_source_count,
        "technical_primary_source_ratio": technical_primary_source_ratio,
        "duplicate_event_rate": duplicate_event_rate,
        "paper_selected_count": len(featured_papers) if featured_papers else min(len(research_fallback), 6),
        "paper_appendix_count": len(paper_appendix),
        "visible_paper_count": len(visible_papers),
        "visible_information_count": visible_information_count,
        "source_news_brief_count": source_news_brief_count,
        "visible_news_count": int(primary_section_counts.get("news", 0)),
        "visible_technical_count": int(primary_section_counts.get("technical", 0)),
        "cross_section_duplicate_count": len(all_items) - len(unique_section_keys),
        "memory_card_count": len(must_read),
        "suspicious_claim_count": suspicious_count,
        "high_evidence_calibration_warning": high_evidence_warning,
        "tracking_question_count": min(5, tracking_candidates),
        "domain_counts": domain_counts,
        "domain_coverage_warning_count": domain_coverage_warning_count,
        "learning_card_missing_count": learning_card_missing_count,
        "technical_context_missing_count": technical_context_missing_count,
        "paper_technical_intro_missing_count": paper_technical_intro_missing_count,
        "editorial_quality_status": editorial_metrics.get("editorial_quality_status", "unknown"),
        "deepseek_schema_valid_count": editorial_metrics.get("deepseek_schema_valid_count", 0),
        "deepseek_empty_facts_count": editorial_metrics.get("deepseek_empty_facts_count", 0),
        "deepseek_key_field_missing_count": editorial_metrics.get("deepseek_key_field_missing_count", 0),
        "deepseek_health_hint": editorial_metrics.get("deepseek_health_hint", ""),
        "gpt_schema_valid_count": editorial_metrics.get("gpt_schema_valid_count", 0),
        "gpt_empty_facts_count": editorial_metrics.get("gpt_empty_facts_count", 0),
        "gpt_key_field_missing_count": editorial_metrics.get("gpt_key_field_missing_count", 0),
        "gpt_health_hint": editorial_metrics.get("gpt_health_hint", ""),
        "llm_schema_valid_count": editorial_metrics.get("llm_schema_valid_count", 0),
        "llm_empty_facts_count": editorial_metrics.get("llm_empty_facts_count", 0),
        "llm_key_field_missing_count": editorial_metrics.get("llm_key_field_missing_count", 0),
        "llm_health_hint": editorial_metrics.get("llm_health_hint", ""),
        "mixed_language_title_count": editorial_metrics.get("mixed_language_title_count", 0),
        "field_label_leak_count": editorial_metrics.get("field_label_leak_count", 0),
        "low_info_expanded_count": editorial_metrics.get("low_info_expanded_count", 0),
        "mojibake_suspect_count": editorial_metrics.get("mojibake_suspect_count", 0),
        "paper_technical_intro_pass_count": editorial_metrics.get("paper_technical_intro_pass_count", 0),
        "paper_technical_intro_fail_count": editorial_metrics.get("paper_technical_intro_fail_count", 0),
        "paper_plain_summary_editorial_pass_count": editorial_metrics.get("paper_plain_summary_pass_count", 0),
        "paper_plain_summary_editorial_fail_count": editorial_metrics.get("paper_plain_summary_fail_count", 0),
        "physical_ai_featured_count": editorial_metrics.get("physical_ai_featured_count", 0),
    }


def repair_bad_titles_in_layers(layers: Dict[str, List[Dict[str, Any]]]) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, int]]:
    repaired_count = 0
    unresolved_count = 0
    examples: List[Dict[str, Any]] = []
    updated: Dict[str, List[Dict[str, Any]]] = {}
    for key, items in layers.items():
        updated[key] = []
        for item in items:
            candidate = dict(item)
            if candidate.get("source_grounded_brief"):
                candidate["quality_flags"] = item_quality_flags(candidate)
                updated[key].append(candidate)
                continue
            raw_title = str(candidate.get("title_cn") or candidate.get("title") or "").strip()
            candidate["title_cn"] = (
                raw_title
                if str(candidate.get("model_used") or "") == "codex-automation"
                else polish_report_title(raw_title)
            )
            candidate = enrich_learning_fields(candidate)
            candidate = enrich_editorial_fields(candidate)
            if title_looks_bad(candidate):
                old_title = str(candidate.get("title_cn") or candidate.get("title") or "")
                title_suggestion = suggest_repaired_title_with_source(candidate)
                new_title = polish_report_title(title_suggestion["title"])
                if new_title and new_title != old_title:
                    candidate["title_cn"] = new_title
                    repaired_count += 1
                    if len(examples) < 10:
                        examples.append(
                            {
                                "rank": int(candidate.get("report_rank", 0) or 0),
                                "section": str(candidate.get("report_section") or key or ""),
                                "article_id": int(candidate.get("id", 0) or 0),
                                "old_title": old_title,
                                "new_title": new_title,
                                "source": title_suggestion.get("source", ""),
                            }
                        )
            if title_looks_bad(candidate):
                unresolved_count += 1
            candidate["quality_flags"] = item_quality_flags(candidate)
            updated[key].append(candidate)
    rank = 1
    for key in REPORT_SECTION_ORDER:
        for item in updated.get(key, []):
            item["report_rank"] = rank
            rank += 1
    return updated, {
        "bad_title_repaired_count": repaired_count,
        "bad_title_unresolved_count": unresolved_count,
        "examples": examples,
    }


def is_focus_quality_item(item: Dict[str, Any]) -> bool:
    flags = set(item_quality_flags(item))
    return not (
        {
            "missing_facts",
            "low_evidence",
            "low_density",
            "generic_summary",
            "aggregator_source",
            "title_fact_mismatch",
            "bad_title",
            "suspicious_claim",
            "untranslated_fact",
            "field_label_leak",
            "bad_public_phrase",
            "mojibake_suspect",
            "template_fallback",
            "unsupported_evidence",
            "unsupported_numeric_claim",
            "generic_method",
            "invalid_result",
        }
        & flags
    )


def v8_has_concrete_news_evidence(item: Dict[str, Any]) -> bool:
    facts = item.get("facts_cn") or item.get("facts") or {}
    if not isinstance(facts, dict):
        return False
    if any(
        str(facts.get(key) or "").strip()
        for key in ("method", "metric_result", "code_or_project", "deployment_context")
    ):
        return True
    evidence = facts.get("evidence") or []
    if isinstance(evidence, str):
        evidence = [evidence]
    evidence_text = " ".join(str(point or "") for point in evidence[:3])
    return bool(
        re.search(
            r"\d|客户|用户|上线|协议|合同|代码|开源|基准|实验|成功率|准确率|成本|延迟|吞吐|部署",
            evidence_text,
            re.IGNORECASE,
        )
    )


def v10_has_concrete_news_evidence(item: Dict[str, Any]) -> bool:
    facts = item.get("facts_cn") or item.get("facts") or {}
    if not isinstance(facts, dict):
        return False
    evidence = facts.get("evidence") or []
    if isinstance(evidence, str):
        evidence = [evidence]
    evidence = [str(point or "").strip() for point in evidence if str(point or "").strip()]
    combined = " ".join(
        str(facts.get(key) or "")
        for key in (
            "method",
            "core_method",
            "metric_result",
            "code_or_project",
            "deployment_context",
            "baseline",
        )
    ) + " " + " ".join(evidence)
    has_numeric = bool(re.search(r"\d+(?:\.\d+)?\s?(?:%|倍|x|X|亿|万|million|billion|个|座|项)", combined))
    method = str(facts.get("core_method") or facts.get("method") or "").strip()
    target = str(facts.get("target") or "").strip()
    has_substantive_method = (
        len(method) >= 16
        and method != target
        and bool(re.search(r"通过|先|再|结合|采用|训练|预测|连接|提取|优化|编码|解码|检索|控制|约束", method))
    )
    code_or_project = str(facts.get("code_or_project") or "").strip()
    has_code = (
        len(code_or_project) >= 5
        and code_or_project.lower() not in {"open-source", "open source", "github", "开源"}
    )
    text = normalize_text(item.get("title", ""), item.get("title_cn", ""), item.get("summary", ""))
    is_viewpoint = any(token in text for token in ("interview", "opinion", "essay", "访谈", "观点", "专栏", "演讲"))
    has_supported_viewpoint = is_viewpoint and len(evidence) >= 2
    has_specific_features = len(evidence) >= 2 and bool(
        re.search(
            r"集成|开源|协议|许可证|访问控制|连接器|加速|降低|提升|客户|部署|成功率|吞吐|延迟|融资|收购|交易",
            " ".join(evidence),
            re.IGNORECASE,
        )
    )
    return has_numeric or has_substantive_method or has_code or has_supported_viewpoint or has_specific_features


def v10_paper_fallback_topic(item: Dict[str, Any]) -> str:
    title_text = str(item.get("title") or "").lower()
    if "multi-robot" in title_text or "multi robot" in title_text:
        return "多机器人协作方法与实验"
    if "safe" in title_text or "safety" in title_text or "attack" in title_text:
        return "机器人安全方法与评测"
    if "navigation" in title_text:
        if "subarctic" in title_text or "forest" in title_text:
            return "亚寒带森林机器人导航挑战与评测"
        return "机器人导航方法与评测"
    if "vision-language-action" in title_text or re.search(r"\bvla\b", title_text):
        if "pre-train" in title_text or "pretrain" in title_text:
            return "视觉语言动作模型继续预训练方法"
        return "视觉语言动作模型训练与评测"
    if any(term in title_text for term in ("grasp", "manipulation", "dexterous")):
        return "机器人操作与抓取方法"
    if "planning" in title_text:
        return "机器人规划方法与评测"
    if "humanoid" in title_text or "locomotion" in title_text:
        return "人形机器人运动控制方法"
    if "multimodal" in title_text or "vision-language" in title_text:
        return "多模态模型训练与评测"
    if any(term in title_text for term in ("world model", "world modeling", "video prediction")):
        return "世界模型训练与评测"
    return {
        "world_model": "世界模型训练与评测",
        "physical_ai": "机器人控制与操作研究",
        "agent_models": "智能体方法与评测",
        "infra_open_source": "模型训练与推理研究",
    }.get(str(item.get("domain_key") or learning_domain_key(item)), "人工智能方法与实验")


def repair_v10_paper_title(item: Dict[str, Any]) -> Dict[str, Any]:
    candidate = dict(item)
    if candidate.get("title_cn"):
        candidate["title_cn"] = str(candidate.get("title_cn") or "").rstrip("，,；;：:、 ")
    current_curated_title = str(candidate.get("title_cn") or "").strip()
    if (
        str(candidate.get("model_used") or "") == "codex-automation"
        and current_curated_title
        and len(current_curated_title) <= 72
        and not title_looks_bad(candidate)
        and not has_untranslated_prose(current_curated_title)
        and not any(marker in current_curated_title for marker in ("…", "..."))
    ):
        candidate["editorial_title"] = current_curated_title
        return candidate
    facts = candidate.get("facts_cn") or candidate.get("facts") or {}
    if not isinstance(facts, dict):
        facts = {}
    original_title = str(candidate.get("title") or "").strip()
    entity_match = re.match(r"^([A-Za-z][A-Za-z0-9._+\- ]{1,24})(?::|\s+-\s+)", original_title)
    entity = entity_match.group(1).strip() if entity_match else ""
    if not entity:
        current_title = str(candidate.get("title_cn") or "").strip()
        current_entity_match = re.match(r"^([A-Za-z][A-Za-z0-9._+\- ]{1,24})：", current_title)
        entity = current_entity_match.group(1).strip() if current_entity_match else ""
    if not entity:
        fact_who = str(facts.get("who") or "").strip()
        if fact_who.lower() not in {"", "this paper", "researchers", "研究者", "研究团队"}:
            entity = fact_who
    generic_targets = {
        "人工智能", "模型", "方法", "框架", "机器人", "具身智能", "世界模型",
        "智能体", "基础设施", "模型训练", "模型推理", "研究任务",
    }
    if normalize_text(entity) in {normalize_text(value) for value in generic_targets}:
        entity = ""
    if re.search(r"要解决的是|核心动作|研究问题|实验结果", entity):
        entity = ""
    target = re.split(r"[，；;。]", str(facts.get("target") or "").strip(), maxsplit=1)[0].strip()
    target = _compact_paper_target(target, 38)
    target = re.sub(r"[（(][A-Za-z][A-Za-z0-9 /_-]{2,}[）)]", "", target).strip()
    if normalize_text(target) in {normalize_text(value) for value in generic_targets}:
        target = ""
    normalized_entity = re.sub(r"[^a-z0-9]+", "", entity.lower())
    normalized_target = re.sub(r"[^a-z0-9]+", "", target.lower())
    target_repeats_entity = bool(
        normalized_entity
        and normalized_target
        and min(len(normalized_entity), len(normalized_target)) >= 4
        and (
            normalized_entity.startswith(normalized_target)
            or normalized_target.startswith(normalized_entity)
        )
    )
    if target_repeats_entity:
        target = ""
    if entity and target:
        repaired_title = f"{entity}：{target}".strip("，；：: ")
        if len(repaired_title) <= 56 and not title_looks_bad({**candidate, "title_cn": repaired_title}):
            candidate["title_cn"] = repaired_title
            candidate["editorial_title"] = repaired_title
            return candidate

    current_title = str(candidate.get("title_cn") or candidate.get("title") or "").strip()
    generic_title_pattern = re.compile(
        r"(?:的新方法与实验|方法与实验研究|研究方法与实验结果)$|"
        r"^(?:人工智能|模型|方法|框架|机器人|具身智能|世界模型|智能体|研究任务)$"
    )
    repeated_title = bool(re.fullmatch(r"(.{2,12})\1", current_title))
    if (
        not target_repeats_entity
        and not title_looks_bad(candidate)
        and not generic_title_pattern.search(current_title)
        and not repeated_title
    ):
        candidate["editorial_title"] = str(candidate.get("title_cn") or candidate.get("title") or "").strip()
        return candidate

    suggestion = suggest_repaired_title_with_source(candidate).get("title", "")
    suggestion = str(suggestion or "").rstrip("，；：:（( ")
    suggestion_is_generic = bool(
        generic_title_pattern.search(suggestion)
        or re.fullmatch(r"(.{2,12})\1", suggestion)
        or
        re.fullmatch(
            r"(?:研究团队|研究者|论文|本文)?[：:]?(?:人工智能|模型|方法|框架|机器人|具身智能|世界模型|智能体|研究任务)",
            suggestion,
        )
    )
    if (
        suggestion
        and not target_repeats_entity
        and not suggestion_is_generic
        and not title_looks_bad({**candidate, "title_cn": suggestion})
    ):
        candidate["title_cn"] = suggestion
        candidate["editorial_title"] = suggestion
        return candidate

    meaningful_tokens = [
        token
        for token in re.findall(r"[A-Za-z][A-Za-z0-9._+\-]*", original_title)
        if token.lower() not in {"a", "an", "the", "this", "via", "for", "with", "from", "using", "of", "in", "on"}
    ]
    fallback_entity = entity or " ".join(meaningful_tokens[:2])
    fallback_topic = v10_paper_fallback_topic(candidate)
    fallback_title = _trim_repaired_title(
        f"{fallback_entity}：{fallback_topic}" if fallback_entity else fallback_topic,
        56,
    )
    if fallback_title and not title_looks_bad({**candidate, "title_cn": fallback_title}):
        candidate["title_cn"] = fallback_title
        candidate["editorial_title"] = fallback_title
    return candidate


def v10_paper_is_ai_relevant(item: Dict[str, Any]) -> bool:
    topic = normalize_text(
        item.get("paper_domain_key", ""),
        item.get("domain_key", ""),
        item.get("topic", ""),
        item.get("category", ""),
        item.get("source_detail", ""),
    )
    text = normalize_text(
        item.get("title", ""),
        item.get("content", ""),
        item.get("summary", ""),
    )
    if "world model" in topic or "world_model" in topic:
        return any(
            token in text
            for token in (
                "machine learning", "artificial intelligence", "neural", "robot", "agent",
                "policy", "planning", "control", "video", "prediction", "predictive",
                "latent", "generative", "reinforcement", "benchmark", "dataset",
                "世界模型", "机器学习", "人工智能", "神经网络", "机器人", "智能体",
                "策略", "规划", "控制", "视频生成", "预测", "潜变量", "生成模型",
                "强化学习", "基准", "数据集", "自动驾驶",
            )
        )
    return True


def v10_paper_has_display_content(item: Dict[str, Any]) -> bool:
    if not v10_paper_is_ai_relevant(item):
        return False
    candidate = enrich_editorial_fields({**dict(item), "_v8_force_fact_title": True})
    candidate = repair_v10_paper_title(candidate)
    if title_looks_bad(candidate):
        return False
    if str(candidate.get("summary_quality_tier") or "") == "index_only":
        return True
    return bool(
        str(
            candidate.get("paper_compact_summary")
            or candidate.get("paper_plain_summary")
            or candidate.get("paper_technical_intro")
            or ""
        ).strip()
    )


def v10_paper_has_editorial_summary(item: Dict[str, Any]) -> bool:
    candidate = enrich_editorial_fields({**dict(item), "_v8_force_fact_title": True})
    return bool(
        str(candidate.get("summary_quality_tier") or "") == "editorial_ready"
        and paper_plain_summary_passes(candidate.get("paper_plain_summary"))
        and paper_technical_intro_passes(candidate.get("paper_technical_intro"))
    )


def should_preserve_existing_paper_analysis(
    paper: Dict[str, Any],
    result: Optional[Dict[str, Any]],
) -> bool:
    return bool(
        result
        and str(result.get("model_used") or "") == "template_fallback"
        and v10_paper_has_display_content(paper)
    )


def repair_v10_paper_titles_in_layers(
    layers: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, List[Dict[str, Any]]]:
    def safe_chinese_title(item: Dict[str, Any]) -> str:
        facts = item.get("facts_cn") or item.get("facts") or {}
        if not isinstance(facts, dict):
            facts = {}
        target = str(facts.get("target") or facts.get("research_problem") or "").strip()
        target = re.sub(r"[A-Za-z0-9$^+._/\\-]+", " ", target)
        target = re.sub(r"[（）()\[\]【】]", " ", target)
        target = re.sub(r"\s+", "", target).strip("，；：、。 ")
        target = _compact_title_piece(target, 44)
        generic_safe_parts = {
            "人工智能", "模型", "方法", "框架", "机器人", "具身智能", "世界模型", "智能体",
        }
        if normalize_text(target) in {normalize_text(value) for value in generic_safe_parts}:
            target = ""
        if len(re.findall(r"[\u4e00-\u9fff]", target)) >= 6:
            return target
        method = str(facts.get("core_method") or facts.get("method") or "").strip()
        method = re.sub(r"[A-Za-z0-9$^+._/\\-]+", " ", method)
        method = re.sub(r"[（）()\[\]【】]", " ", method)
        method = re.sub(r"\s+", "", method).strip("，；：、。 ")
        method = _compact_title_piece(method, 24)
        if normalize_text(method) in {normalize_text(value) for value in generic_safe_parts}:
            method = ""
        combined = _compact_title_piece(f"{target}{method}", 44)
        if (
            len(re.findall(r"[\u4e00-\u9fff]", combined)) >= 6
            and not re.fullmatch(r"(.{2,12})\1", combined)
        ):
            return combined
        domain = str(item.get("domain_key") or learning_domain_key(item))
        return v10_paper_fallback_topic({**item, "domain_key": domain})

    repaired: Dict[str, List[Dict[str, Any]]] = {}
    for key, items in layers.items():
        repaired[key] = []
        for item in items:
            candidate = repair_v10_paper_title(item) if item.get("content_type") == "paper" else dict(item)
            candidate_title = str(candidate.get("title_cn") or candidate.get("title") or "")
            if item.get("content_type") == "paper" and (
                title_looks_bad(candidate)
                or mixed_language_title(candidate_title)
                or has_untranslated_prose(candidate_title)
                or len(re.findall(r"[A-Za-z][A-Za-z-]{3,}", candidate_title)) >= 2
                or any(marker in candidate_title for marker in ("…", "..."))
            ):
                candidate["title_cn"] = safe_chinese_title(candidate)
                candidate["editorial_title"] = candidate["title_cn"]
            if item.get("content_type") == "paper":
                candidate["_v8_force_fact_title"] = False
            candidate = enrich_editorial_fields(candidate)
            if item.get("content_type") == "paper":
                candidate = repair_v10_paper_title(candidate)
            candidate["quality_flags"] = item_quality_flags(candidate)
            repaired[key].append(candidate)
    return repaired


def apply_feedback_preference_scores(items: List[Dict[str, Any]], feedback_weights: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
    adjusted: List[Dict[str, Any]] = []
    source_weights = feedback_weights.get("source", {})
    topic_weights = feedback_weights.get("topic", {})
    entity_weights = feedback_weights.get("entity", {})
    for item in items:
        candidate = dict(item)
        facts = candidate.get("facts") or {}
        boost = 0.0
        boost += source_weights.get(str(candidate.get("source_detail", "") or ""), 0.0)
        boost += source_weights.get(str(candidate.get("platform", "") or ""), 0.0)
        boost += topic_weights.get(str(candidate.get("topic", "") or ""), 0.0)
        boost += topic_weights.get(str(candidate.get("category", "") or ""), 0.0)
        if isinstance(facts, dict):
            boost += entity_weights.get(str(facts.get("who", "") or ""), 0.0)
        candidate["feedback_preference_score"] = round(boost, 3)
        candidate["selection_score"] = float(candidate.get("selection_score", candidate.get("score", 0)) or 0) + boost
        adjusted.append(candidate)
    return adjusted


def build_feedback_links(item: Dict[str, Any], report_id: str, feedback_config: Dict[str, Any]) -> Dict[str, str]:
    if (
        not feedback_config.get("enabled", False)
        or feedback_config.get("runtime_healthy") is False
        or not item.get("id")
    ):
        return {}
    host = str(feedback_config.get("host", "127.0.0.1") or "127.0.0.1")
    port = int(feedback_config.get("port", 8765) or 8765)
    base = f"http://{host}:{port}/feedback?report_id={report_id}&item_id={int(item['id'])}"
    return {
        "useful": f"{base}&signal=useful",
        "not_useful": f"{base}&signal=not_useful",
        "track": f"{base}&signal=track",
        "mute_similar": f"{base}&signal=mute_similar",
        "too_shallow": f"{base}&signal=too_shallow",
        "paper_too_shallow": f"{base}&signal=paper_too_shallow",
        "paper_unclear": f"{base}&signal=paper_unclear",
        "too_long": f"{base}&signal=too_long",
        "too_generic": f"{base}&signal=too_generic",
        "source_suspicious": f"{base}&signal=source_suspicious",
        "not_memorable": f"{base}&signal=not_memorable",
    }


def feedback_server_is_healthy(feedback_config: Dict[str, Any]) -> bool:
    if not feedback_config.get("enabled", False):
        return False
    host = str(feedback_config.get("host", "127.0.0.1") or "127.0.0.1")
    port = int(feedback_config.get("port", 8765) or 8765)
    try:
        with urlopen(f"http://{host}:{port}/health", timeout=1.0) as response:
            return int(getattr(response, "status", 0) or 0) == 200
    except Exception:
        return False


def paper_core_summary_status(item: Dict[str, Any]) -> Dict[str, Any]:
    if item.get("content_type") != "paper":
        return {"passed": True, "missing": []}
    facts = item.get("facts") if isinstance(item.get("facts"), dict) else {}
    evidence = facts.get("evidence") if isinstance(facts, dict) else []
    if isinstance(evidence, str):
        evidence = [evidence]
    evidence_text = " ".join(str(point or "") for point in (evidence or []))
    text = " ".join(
        str(item.get(field, "") or "")
        for field in (
            "title_cn",
            "title",
            "summary_preview",
            "summary",
            "why_it_matters",
            "expected_effect",
            "future_impact",
        )
    )
    method_ok = bool(str(facts.get("action") or "").strip() and str(facts.get("target") or "").strip())
    method_ok = method_ok or bool(re.search(r"提出|引入|采用|训练|微调|框架|方法|模型|算法|benchmark|dataset", text, re.IGNORECASE))
    result_ok = bool(evidence)
    result_ok = result_ok and bool(
        re.search(r"\d|%|提升|降低|达到|优于|超过|成功率|准确率|基准|实验|评测|对比|result|benchmark|outperform|success", evidence_text + " " + text, re.IGNORECASE)
    )
    meaning_ok = bool(str(facts.get("audience") or "").strip())
    topic_text = str(item.get("display_topic") or item.get("topic_cn") or item.get("category") or item.get("topic") or "")
    if re.search(r"具身|机器人|世界模型|Physical AI|Robotics|World Model", topic_text, re.IGNORECASE):
        meaning_ok = True
    meaning_ok = meaning_ok or bool(
        re.search(r"影响|意义|复现|部署|应用|落地|成本|试错|开发者|研究者|团队|客户|场景|value|impact", text, re.IGNORECASE)
    )
    missing = []
    if not method_ok:
        missing.append("method")
    if not result_ok:
        missing.append("result")
    if not meaning_ok:
        missing.append("meaning")
    return {"passed": not missing, "missing": missing}


def paper_core_summary_passes(item: Dict[str, Any]) -> bool:
    return bool(paper_core_summary_status(item).get("passed"))


def classify_report_layers(
    papers: List[Dict[str, Any]],
    updates: List[Dict[str, Any]],
    *,
    report_id: str,
    feedback_config: Dict[str, Any],
    source_preferences: Optional[Dict[str, Any]] = None,
    report_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    report_config = report_config or {}
    research_limit = max(1, int(report_config.get("research_limit", 10) or 10))
    paper_featured_limit = max(1, int(report_config.get("paper_featured_limit", 6) or 6))
    paper_appendix_limit = max(0, int(report_config.get("paper_appendix_limit", research_limit) or 0))
    paper_technical_intro_min_count = max(0, int(report_config.get("paper_technical_intro_min_count", 12)))
    v3_paper_limits_configured = "paper_featured_limit" in report_config or "paper_appendix_limit" in report_config
    visible_paper_limit = (
        max(research_limit, paper_featured_limit + paper_appendix_limit)
        if v3_paper_limits_configured
        else research_limit
    )
    physical_ai_limit = max(1, int(report_config.get("physical_ai_min_items", 4) or 4))
    physical_ai_featured_min_count = max(1, int(report_config.get("physical_ai_featured_min_count", physical_ai_limit) or physical_ai_limit))
    all_items = sorted(
        [dict(item) for item in papers + updates],
        key=lambda item: (
            float(item.get("selection_score", item.get("score", 0)) or 0),
            float(item.get("score", 0) or 0),
            parse_datetime(item.get("publish_date", "")),
        ),
        reverse=True,
    )
    for index, item in enumerate(all_items):
        item["canonical_url"] = item.get("canonical_url") or item.get("url", "")
        item["source_tier"] = item.get("source_tier") or source_tier_for_item(item, source_preferences)
        item = repair_title_fact_mismatch(item)
        item["title_cn"] = polish_report_title(str(item.get("title_cn") or item.get("title") or ""))
        item = enrich_learning_fields(item)
        item = enrich_editorial_fields(
            item,
            physical_ai_min_items=physical_ai_featured_min_count,
            paper_technical_intro_min_count=paper_technical_intro_min_count,
        )
        all_items[index] = item
        item["quality_flags"] = item_quality_flags(item)
        item["feedback_links"] = build_feedback_links(item, report_id, feedback_config)

    used_urls: set[str] = set()
    physical_ai: List[Dict[str, Any]] = []
    for item in all_items:
        if len(physical_ai) >= physical_ai_limit:
            break
        url = str(item.get("url", ""))
        if url in used_urls:
            continue
        if item.get("content_type") == "paper":
            continue
        if not is_physical_ai_item(item):
            continue
        if not is_focus_quality_item(item):
            continue
        item["report_section"] = "physical_ai"
        physical_ai.append(item)
        used_urls.add(url)

    must_read: List[Dict[str, Any]] = []
    must_read_source_counts: Dict[str, int] = {}
    for item in all_items:
        if len(must_read) >= 8:
            break
        if str(item.get("url", "")) in used_urls:
            continue
        if item.get("content_type") == "paper":
            continue
        if is_focus_quality_item(item):
            source_key = report_source_key(item)
            if must_read_source_counts.get(source_key, 0) >= 2:
                continue
            item["report_section"] = "must_read"
            must_read.append(item)
            used_urls.add(str(item.get("url", "")))
            must_read_source_counts[source_key] = must_read_source_counts.get(source_key, 0) + 1

    for item in all_items:
        if len(must_read) >= 8:
            break
        if str(item.get("url", "")) in used_urls:
            continue
        if item.get("content_type") == "paper":
            continue
        if is_focus_quality_item(item):
            item["report_section"] = "must_read"
            must_read.append(item)
            used_urls.add(str(item.get("url", "")))

    research: List[Dict[str, Any]] = []
    watch: List[Dict[str, Any]] = []
    brief: List[Dict[str, Any]] = []
    for item in all_items:
        url = str(item.get("url", ""))
        if url in used_urls:
            continue
        if item.get("content_type") == "paper" and evidence_quality_value(item) >= 0.35:
            item["report_section"] = "research"
            research.append(item)
        elif (
            evidence_quality_value(item) >= 0.35
            and information_density_value(item) >= 0.35
            and not summary_looks_generic(item.get("summary", ""))
            and "suspicious_claim" not in set(item.get("quality_flags") or item_quality_flags(item))
        ):
            item["report_section"] = "watch"
            watch.append(item)
        else:
            item["report_section"] = "brief"
            brief.append(item)

    research_focus = research[:research_limit]
    visible_research = research[:visible_paper_limit]
    featured_papers: List[Dict[str, Any]] = []
    paper_appendix: List[Dict[str, Any]] = []
    featured_urls: set[str] = set()
    for domain_key in LEARNING_DOMAIN_ORDER:
        if len(featured_papers) >= paper_featured_limit:
            break
        for item in visible_research:
            url = str(item.get("url", ""))
            if url in featured_urls or item.get("domain_key") != domain_key:
                continue
            paper_status = paper_core_summary_status(item)
            item["paper_core_summary_status"] = paper_status
            if paper_status.get("passed"):
                item["report_section"] = "featured_papers"
                featured_papers.append(item)
                featured_urls.add(url)
                break
    for item in visible_research:
        if str(item.get("url", "")) in featured_urls:
            continue
        paper_status = paper_core_summary_status(item)
        item["paper_core_summary_status"] = paper_status
        if len(featured_papers) < paper_featured_limit and paper_status.get("passed"):
            item["report_section"] = "featured_papers"
            featured_papers.append(item)
            featured_urls.add(str(item.get("url", "")))
        elif len(paper_appendix) < paper_appendix_limit:
            item["report_section"] = "paper_appendix"
            paper_appendix.append(item)
        else:
            item["report_section"] = "brief"
            brief.append(item)
    for item in research[visible_paper_limit:]:
        item["report_section"] = "brief"
    brief = research[visible_paper_limit:] + brief

    v8_mode = is_continuous_reader_design(report_config)
    layers = {
        "must_read": must_read[:8],
        "physical_ai": physical_ai[:physical_ai_limit],
        "watch": watch if v8_mode else watch[:6],
        "featured_papers": featured_papers,
        "paper_appendix": paper_appendix,
        "research": research_focus,
        "brief": brief if v8_mode else brief[:20],
    }
    layers = suppress_repeated_analysis_fields(layers)
    rank = 1
    for key in REPORT_SECTION_ORDER:
        for item in layers.get(key, []):
            item["report_rank"] = rank
            rank += 1
    return layers


def flatten_report_layers(layers: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for key in REPORT_SECTION_ORDER:
        result.extend(layers.get(key, []))
    return result


def report_primary_section(item: Dict[str, Any]) -> str:
    facts = item.get("facts") if isinstance(item.get("facts"), dict) else {}
    section = str(item.get("primary_section") or facts.get("primary_section") or "").strip().lower()
    if section in {"news", "technical", "paper"}:
        return section
    if str(item.get("content_type") or "").strip().lower() == "paper":
        return "paper"
    if str(item.get("content_type") or "").strip().lower() in {"project", "open_source", "opensource"}:
        return "technical"
    return "news"


def v11_item_contract_failures(item: Dict[str, Any]) -> List[str]:
    """Return production contract failures for a Codex-researched V11 item."""
    candidate = enrich_editorial_fields(enrich_learning_fields(dict(item)))
    model_used = str(candidate.get("model_used") or "").strip().lower()
    analysis_version = str(candidate.get("analysis_version") or "").strip().lower()
    is_codex_research = model_used == "codex-automation" or analysis_version.startswith(
        "codex-research"
    )
    if not is_codex_research:
        return []

    failures: List[str] = []
    section = report_primary_section(candidate)
    facts = candidate.get("facts") if isinstance(candidate.get("facts"), dict) else {}
    source_excerpt = str(
        candidate.get("source_excerpt") or facts.get("source_excerpt") or ""
    ).strip()
    evidence_locator = str(
        candidate.get("evidence_locator") or facts.get("evidence_locator") or ""
    ).strip()
    claim_type = str(candidate.get("claim_type") or facts.get("claim_type") or "").strip()
    if analysis_version not in SUPPORTED_CODEX_RESEARCH_ANALYSIS_VERSIONS:
        failures.append("analysis_version")
    if not str(candidate.get("publish_date") or "").strip():
        failures.append("publish_date")
    if not source_excerpt:
        failures.append("source_excerpt")
    if not evidence_locator:
        failures.append("evidence_locator")
    if not claim_type:
        failures.append("claim_type")
    if evidence_quality_value(candidate) < 0.45:
        failures.append("evidence_quality")
    if information_density_value(candidate) < 0.45:
        failures.append("information_density")
    if title_looks_bad(candidate):
        failures.append("title")

    blocked_flags = {
        "missing_facts",
        "low_evidence",
        "low_density",
        "generic_summary",
        "title_fact_mismatch",
        "bad_title",
        "mixed_language_title",
        "field_label_leak",
        "mojibake_suspect",
        "bad_public_phrase",
        "low_info_expanded",
        "suspicious_claim",
        "untranslated_fact",
        "unsupported_numeric_claim",
    }
    observed_flags = set(candidate.get("editorial_flags") or []) | set(
        item_quality_flags(candidate)
    )
    failures.extend(sorted(blocked_flags & observed_flags))
    if str(candidate.get("quality_tier") or "") == "brief":
        failures.append("quality_tier")

    if section == "paper":
        if not paper_plain_summary_passes(candidate.get("paper_plain_summary")):
            failures.append("paper_plain_summary")
        if not paper_technical_intro_passes(candidate.get("paper_technical_intro")):
            failures.append("paper_technical_intro")
        if not str(facts.get("method") or facts.get("core_method") or "").strip():
            failures.append("paper_method")
        if not any(
            str(facts.get(key) or "").strip()
            for key in ("metric_result", "dataset_or_benchmark", "baseline")
        ):
            failures.append("paper_result_context")
    else:
        body = re.sub(
            r"\s+",
            "",
            str(candidate.get("analysis_body") or candidate.get("summary") or ""),
        )
        content_type = str(candidate.get("content_type") or "").strip().lower()
        minimum_body_chars = (
            300
            if content_type in {"interview", "podcast", "video"}
            else 220
            if section == "technical"
            else 180
        )
        if len(body) < minimum_body_chars:
            failures.append("body_under_min")
    return sorted(set(failures))


def select_v11_update_candidates(
    candidates: List[Dict[str, Any]],
    report_config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Preserve independent news and technical buffers before editorial ranking."""
    targets = {
        "news": max(
            int(report_config.get("min_visible_news_count", 20) or 20),
            int(report_config.get("news_section_limit", 24) or 24),
            int(report_config.get("news_candidate_pool_limit", 30) or 30),
        ),
        "technical": max(
            int(report_config.get("min_visible_technical_count", 20) or 20),
            int(report_config.get("technical_section_limit", 24) or 24),
            int(report_config.get("technical_candidate_pool_limit", 30) or 30),
        ),
    }
    selected: List[Dict[str, Any]] = []
    for section in ("news", "technical"):
        section_candidates = [
            item
            for item in candidates
            if report_primary_section(item) == section
        ]
        selected.extend(section_candidates[:targets[section]])
    return selected


def apply_v9_continuity(
    layers: Dict[str, List[Dict[str, Any]]],
    previous_items: List[Dict[str, Any]],
    suppress_repeated_focus: bool = True,
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, int]]:
    enriched = enrich_continuity(flatten_report_layers(layers), previous_items)
    by_identity: Dict[str, Dict[str, Any]] = {}
    for item in enriched:
        identity = str(item.get("canonical_url") or item.get("url") or f"article:{item.get('id')}")
        by_identity[identity] = item

    result = {key: [] for key in REPORT_SECTION_ORDER}
    repeated_focus_count = 0
    new_count = 0
    updated_count = 0
    for section in REPORT_SECTION_ORDER:
        for original in layers.get(section, []):
            identity = str(original.get("canonical_url") or original.get("url") or f"article:{original.get('id')}")
            item = dict(by_identity.get(identity, original))
            status = str(item.get("continuity_status") or "new")
            if status == "new":
                new_count += 1
            elif status == "updated":
                updated_count += 1
            if (
                suppress_repeated_focus
                and status == "repeated"
                and section in {"must_read", "physical_ai", "watch", "featured_papers"}
            ):
                repeated_focus_count += 1
                if item.get("content_type") == "paper":
                    item["report_section"] = "paper_appendix"
                    result["paper_appendix"].append(item)
                else:
                    item["report_section"] = "brief"
                    result["brief"].append(item)
                continue
            item["report_section"] = section
            result[section].append(item)

    seen: set[str] = set()
    for section in REPORT_SECTION_ORDER:
        unique: List[Dict[str, Any]] = []
        for item in result[section]:
            identity = str(item.get("canonical_url") or item.get("url") or f"article:{item.get('id')}")
            if identity in seen:
                continue
            seen.add(identity)
            unique.append(item)
        result[section] = unique

    rank = 1
    for section in REPORT_SECTION_ORDER:
        for item in result[section]:
            item["report_rank"] = rank
            rank += 1
    metrics = {
        "new_item_count": new_count,
        "updated_item_count": updated_count,
        "repeated_focus_demoted_count": repeated_focus_count,
    }
    return result, metrics


def downgrade_failed_focus_items(layers: Dict[str, List[Dict[str, Any]]], failed_urls: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    failed_set = {str(url) for url in failed_urls if url}
    if not failed_set:
        return layers
    updated = {key: list(value) for key, value in layers.items()}
    downgraded: List[Dict[str, Any]] = []
    for key in ("must_read", "physical_ai", "watch", "featured_papers", "paper_appendix", "research"):
        retained: List[Dict[str, Any]] = []
        for item in updated.get(key, []):
            if str(item.get("url", "")) in failed_set:
                item = dict(item)
                item["report_section"] = "brief"
                item.setdefault("quality_flags", item_quality_flags(item))
                downgraded.append(item)
            else:
                retained.append(item)
        updated[key] = retained
    updated["brief"] = downgraded + updated.get("brief", [])
    rank = 1
    for key in REPORT_SECTION_ORDER:
        for item in updated.get(key, []):
            item["report_rank"] = rank
            rank += 1
    return updated


def apply_v8_reading_budget(
    layers: Dict[str, List[Dict[str, Any]]],
    report_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    report_config = report_config or {}
    if not is_continuous_reader_design(report_config):
        return layers

    must_limit = max(1, int(report_config.get("must_read_limit", 5) or 5))
    domain_limit = max(1, int(report_config.get("domain_item_limit", 2) or 2))
    featured_limit = max(1, int(report_config.get("paper_featured_limit", 8) or 8))
    appendix_limit = max(0, int(report_config.get("paper_appendix_limit", 12) or 12))
    brief_limit = max(0, int(report_config.get("brief_limit", 8) or 8))
    paper_body_min = max(0, int(report_config.get("paper_body_char_min", 0) or 0))
    paper_body_max = max(paper_body_min, int(report_config.get("paper_body_char_limit", 220) or 220))
    source_limit = max(1, int(report_config.get("source_focus_limit", 2) or 2))
    topic_limit = max(1, int(report_config.get("topic_focus_limit", 3) or 3))

    def unique_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []
        seen_urls: set[str] = set()
        seen_events: set[str] = set()
        for item in items:
            key = str(item.get("canonical_url") or item.get("url") or item.get("title_cn") or item.get("title") or "")
            event_key = normalized_event_title(item)
            dedupe_by_event = str(item.get("content_type") or "").lower() != "paper"
            if not key or key in seen_urls or (dedupe_by_event and event_key and event_key in seen_events):
                continue
            seen_urls.add(key)
            if dedupe_by_event and event_key:
                seen_events.add(event_key)
            result.append(dict(item))
        return result

    def sort_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        source_rank = {"official": 4, "research": 4, "primary": 3, "media": 2, "aggregator": 0, "low_signal": 0}
        return sorted(
            items,
            key=lambda item: (
                1 if item.get("content_type") != "paper" or v10_paper_has_editorial_summary(item) else 0,
                source_rank.get(str(item.get("source_tier") or "").lower(), 1),
                float(item.get("selection_score", item.get("score", 0)) or 0),
                evidence_quality_value(item),
                information_density_value(item),
                parse_datetime(item.get("publish_date", "")),
            ),
            reverse=True,
        )

    def source_key(item: Dict[str, Any]) -> str:
        host = item_host(item)
        source_detail = normalize_text(item.get("source_detail", ""), item.get("platform", ""))
        if (
            str(report_config.get("product_mode") or "") == "intelligence_v11_editorial_library"
            and host == "github.com"
            and source_detail
        ):
            return f"github:{source_detail}"
        return host or source_detail or "unknown"

    def topic_key(item: Dict[str, Any]) -> str:
        if (
            str(report_config.get("product_mode") or "") == "intelligence_v11_editorial_library"
            and report_primary_section(item) == "technical"
        ):
            facts = item.get("facts") if isinstance(item.get("facts"), dict) else {}
            category = str(
                item.get("technical_category") or facts.get("technical_category") or "other"
            ).strip().lower()
            return f"technical:{category}"
        return str(item.get("domain_key") or learning_domain_key(item))

    if str(report_config.get("product_mode") or "") == "intelligence_v11_editorial_library":
        all_candidates = unique_items(
            list(layers.get("must_read", []))
            + list(layers.get("physical_ai", []))
            + list(layers.get("watch", []))
            + list(layers.get("featured_papers", []))
            + list(layers.get("paper_appendix", []))
            + list(layers.get("research", []))
            + list(layers.get("brief", []))
        )
        all_candidates = [
            enrich_editorial_fields({**dict(item), "_v8_force_fact_title": False})
            for item in all_candidates
        ]
        contract_rejections: List[Dict[str, Any]] = []
        contract_ready_candidates: List[Dict[str, Any]] = []
        for item in all_candidates:
            failure_reasons = v11_item_contract_failures(item)
            if failure_reasons:
                contract_rejections.append(
                    {
                        "section": report_primary_section(item),
                        "url": str(item.get("canonical_url") or item.get("url") or ""),
                        "reasons": failure_reasons,
                    }
                )
            else:
                contract_ready_candidates.append(item)
        all_candidates = contract_ready_candidates
        report_config["_v11_selection_contract_rejections"] = contract_rejections
        nonpaper = [item for item in all_candidates if report_primary_section(item) != "paper"]
        papers = [
            item
            for item in all_candidates
            if report_primary_section(item) == "paper"
            and paper_plain_summary_passes(item.get("paper_plain_summary"))
            and paper_technical_intro_passes(item.get("paper_technical_intro"))
        ]
        ordered_nonpaper = sort_items(nonpaper)
        supplemental_limits = {
            str(section): max(0, int(limit or 0))
            for section, limit in dict(
                report_config.get("supplemental_visible_max_by_section")
                or {"news": 5, "technical": 5, "paper": 3}
            ).items()
        }

        def select_with_supplemental_limit(
            candidates: List[Dict[str, Any]],
            *,
            section: str,
            limit: int,
            enforce_diversity: bool = False,
        ) -> List[Dict[str, Any]]:
            selected: List[Dict[str, Any]] = []
            selected_urls: set[str] = set()
            selected_sources: Counter[str] = Counter()
            selected_topics: Counter[str] = Counter()
            supplemental_count = 0
            supplemental_limit = supplemental_limits.get(section, 0)

            def add_candidates(*, enforce_caps: bool) -> None:
                nonlocal supplemental_count
                for item in candidates:
                    if len(selected) >= limit:
                        return
                    url = str(item.get("canonical_url") or item.get("url") or "")
                    if not url or url in selected_urls:
                        continue
                    supplemental = "supplemental_older_source" in set(
                        item.get("quality_flags") or []
                    )
                    if supplemental and supplemental_count >= supplemental_limit:
                        continue
                    source = source_key(item)
                    topic = topic_key(item)
                    if enforce_caps and (
                        selected_sources[source] >= source_limit
                        or selected_topics[topic] >= topic_limit
                    ):
                        continue
                    selected.append(item)
                    selected_urls.add(url)
                    selected_sources[source] += 1
                    selected_topics[topic] += 1
                    supplemental_count += int(supplemental)

            add_candidates(enforce_caps=enforce_diversity)
            if len(selected) < limit:
                add_candidates(enforce_caps=False)
            return selected

        news_limit = max(
            int(report_config.get("min_visible_news_count", 20) or 20),
            int(report_config.get("news_section_limit", 24) or 24),
        )
        technical_limit = max(
            int(report_config.get("min_visible_technical_count", 20) or 20),
            int(report_config.get("technical_section_limit", 24) or 24),
        )
        selected_nonpaper = (
            select_with_supplemental_limit(
                [item for item in ordered_nonpaper if report_primary_section(item) == "news"],
                section="news",
                limit=news_limit,
                enforce_diversity=True,
            )
            + select_with_supplemental_limit(
                [item for item in ordered_nonpaper if report_primary_section(item) == "technical"],
                section="technical",
                limit=technical_limit,
                enforce_diversity=True,
            )
        )
        ordered_nonpaper = sort_items(selected_nonpaper)

        def select_diverse_must_read(
            candidates: List[Dict[str, Any]],
            limit: int,
        ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
            source_cap = max(
                1,
                int(report_config.get("must_read_source_limit", 2) or 2),
            )
            available_sections = {
                report_primary_section(item) for item in candidates
            }
            section_cap = (
                max(1, (limit + 1) // 2)
                if {"news", "technical"}.issubset(available_sections)
                else limit
            )
            selected: List[Dict[str, Any]] = []
            selected_urls: set[str] = set()
            source_counts: Counter[str] = Counter()
            section_counts: Counter[str] = Counter()

            def item_url(item: Dict[str, Any]) -> str:
                return str(
                    item.get("canonical_url")
                    or item.get("url")
                    or item.get("title_cn")
                    or item.get("title")
                    or ""
                )

            def add_pass(*, enforce_section_cap: bool, enforce_diversity: bool) -> None:
                for item in candidates:
                    if len(selected) >= limit:
                        return
                    url = item_url(item)
                    if not url or url in selected_urls:
                        continue
                    source = report_source_key(item)
                    section = report_primary_section(item)
                    if enforce_diversity and source_counts[source] >= source_cap:
                        continue
                    if enforce_section_cap and section_counts[section] >= section_cap:
                        continue
                    selected.append(dict(item))
                    selected_urls.add(url)
                    source_counts[source] += 1
                    section_counts[section] += 1

            add_pass(enforce_section_cap=True, enforce_diversity=True)
            add_pass(enforce_section_cap=False, enforce_diversity=True)
            add_pass(enforce_section_cap=False, enforce_diversity=False)
            remaining = [
                dict(item) for item in candidates if item_url(item) not in selected_urls
            ]
            return selected, remaining

        must_read, watch = select_diverse_must_read(
            ordered_nonpaper,
            must_limit,
        )
        selected_papers = select_with_supplemental_limit(
            papers,
            section="paper",
            limit=featured_limit + appendix_limit,
        )
        featured_papers = [dict(item) for item in selected_papers[:featured_limit]]
        paper_appendix = [dict(item) for item in selected_papers[featured_limit:]]
        result = {
            "must_read": must_read,
            "physical_ai": [],
            "watch": watch,
            "featured_papers": featured_papers,
            "paper_appendix": paper_appendix,
            "research": [],
            "brief": [],
        }
        rank = 1
        for key in REPORT_SECTION_ORDER:
            for item in result.get(key, []):
                item["report_section"] = key
                item["report_rank"] = rank
                rank += 1
        return result

    news_candidates = unique_items(
        list(layers.get("must_read", []))
        + list(layers.get("physical_ai", []))
        + list(layers.get("watch", []))
        + list(layers.get("brief", []))
    )
    normalized_news_candidates = [
        enrich_editorial_fields({**dict(item), "_v8_force_fact_title": True})
        for item in sort_items(news_candidates)
    ]
    source_news_candidate_pool = list(normalized_news_candidates)
    is_v10_design = is_learning_digest_design(report_config)
    strict_v11_papers = str(report_config.get("product_mode") or "") == "intelligence_v11_editorial_library"
    if is_v10_design:
        brief_limit = max(
            brief_limit,
            int(report_config.get("source_news_brief_limit", 12) or 12),
        )
    news_candidates = [
        item
        for item in normalized_news_candidates
        if item.get("content_type") != "paper"
        and item.get("quality_tier") != "brief"
        and is_focus_quality_item(item)
        and (
            v10_has_concrete_news_evidence(item)
            if is_v10_design
            else v8_has_concrete_news_evidence(item)
        )
    ]
    source_counts: Counter[str] = Counter()
    topic_counts: Counter[str] = Counter()
    selected_urls: set[str] = set()

    def can_select(item: Dict[str, Any]) -> bool:
        source = source_key(item)
        topic = str(item.get("domain_key") or learning_domain_key(item))
        return source_counts[source] < source_limit and topic_counts[topic] < topic_limit

    def mark_selected(item: Dict[str, Any], section: str) -> Dict[str, Any]:
        candidate = dict(item)
        candidate["report_section"] = section
        published = parse_datetime(candidate.get("publish_date", ""))
        age = datetime.now() - published if published != datetime.min else timedelta(days=999)
        candidate["freshness_label"] = "今日新增" if age <= timedelta(hours=36) else "历史补位"
        source_counts[source_key(candidate)] += 1
        topic_counts[str(candidate.get("domain_key") or learning_domain_key(candidate))] += 1
        selected_urls.add(str(candidate.get("canonical_url") or candidate.get("url") or ""))
        return candidate

    must_read: List[Dict[str, Any]] = []
    for source_pass in range(source_limit):
        for item in news_candidates:
            if len(must_read) >= must_limit:
                break
            url = str(item.get("canonical_url") or item.get("url") or "")
            if url in selected_urls or source_counts[source_key(item)] != source_pass:
                continue
            if can_select(item):
                must_read.append(mark_selected(item, "must_read"))
        if len(must_read) >= must_limit:
            break

    domain_items: List[Dict[str, Any]] = []
    for domain_key in LEARNING_DOMAIN_ORDER:
        count = 0
        for source_pass in range(source_limit):
            for item in news_candidates:
                url = str(item.get("canonical_url") or item.get("url") or "")
                if url in selected_urls or str(item.get("domain_key") or learning_domain_key(item)) != domain_key:
                    continue
                if source_counts[source_key(item)] != source_pass or not can_select(item):
                    continue
                domain_items.append(mark_selected(item, "watch"))
                count += 1
                if count >= domain_limit:
                    break
            if count >= domain_limit:
                break

    if is_v10_design:
        minimum_information_count = max(
            must_limit,
            int(report_config.get("min_visible_information_count", 25) or 25),
        )
        for item in news_candidates:
            if len(must_read) + len(domain_items) >= minimum_information_count:
                break
            url = str(item.get("canonical_url") or item.get("url") or "")
            if url in selected_urls or not can_select(item):
                continue
            domain_items.append(mark_selected(item, "watch"))

        section_minimums = {
            "news": int(report_config.get("min_visible_news_count", 20) or 20),
            "technical": int(report_config.get("min_visible_technical_count", 20) or 20),
        }
        for primary_section, minimum in section_minimums.items():
            selected_count = sum(
                1
                for selected in must_read + domain_items
                if report_primary_section(selected) == primary_section
            )
            if selected_count >= minimum:
                continue
            for item in news_candidates:
                if selected_count >= minimum:
                    break
                url = str(item.get("canonical_url") or item.get("url") or "")
                if url in selected_urls or report_primary_section(item) != primary_section:
                    continue
                domain_items.append(mark_selected(item, "watch"))
                selected_count += 1

    paper_candidates = unique_items(
        list(layers.get("featured_papers", []))
        + list(layers.get("paper_appendix", []))
        + list(layers.get("research", []))
        + [item for item in layers.get("brief", []) if item.get("content_type") == "paper"]
    )
    featured_papers: List[Dict[str, Any]] = []
    paper_appendix: List[Dict[str, Any]] = []
    for item in sort_items(paper_candidates):
        candidate = enrich_editorial_fields({**dict(item), "_v8_force_fact_title": True})
        if is_v10_design:
            candidate = repair_v10_paper_title(candidate)
        if title_looks_bad(candidate):
            continue
        facts_cn = candidate.get("facts_cn") if isinstance(candidate.get("facts_cn"), dict) else {}
        has_mechanism = bool(str(facts_cn.get("method") or "").strip())
        evidence_points = facts_cn.get("evidence") or []
        if isinstance(evidence_points, str):
            evidence_points = [evidence_points]
        has_result = bool(str(facts_cn.get("metric_result") or "").strip()) or any(
            re.search(r"\d|实验|结果|成功率|准确率|提升|降低|优于|超过|对比", str(point or ""), re.IGNORECASE)
            for point in evidence_points
        )
        intro = str(candidate.get("paper_technical_intro") or "")
        intro_is_complete = not re.search(r"…|\.\.\.", intro)
        intro_length_ok = paper_body_min <= len(intro) <= paper_body_max
        focus_quality_ok = is_focus_quality_item(candidate)
        if (
            len(featured_papers) < featured_limit
            and focus_quality_ok
            and has_mechanism
            and has_result
            and intro_is_complete
            and intro_length_ok
            and paper_technical_intro_passes(intro)
            and (
                not is_learning_digest_design(report_config)
                or paper_plain_summary_passes(candidate.get("paper_plain_summary"))
            )
        ):
            candidate["report_section"] = "featured_papers"
            featured_papers.append(candidate)
        elif (
            len(paper_appendix) < appendix_limit
            and (
                not is_learning_digest_design(report_config)
                or (
                    strict_v11_papers
                    and paper_plain_summary_passes(candidate.get("paper_plain_summary"))
                    and paper_technical_intro_passes(candidate.get("paper_technical_intro"))
                )
                or (
                    not strict_v11_papers
                    and (
                        str(candidate.get("summary_quality_tier") or "") == "index_only"
                        or bool(
                            str(
                                candidate.get("paper_compact_summary")
                                or candidate.get("paper_plain_summary")
                                or candidate.get("paper_technical_intro")
                                or ""
                            ).strip()
                        )
                    )
                )
            )
        ):
            candidate["report_section"] = "paper_appendix"
            paper_appendix.append(candidate)

    used = {
        str(item.get("canonical_url") or item.get("url") or "")
        for item in must_read + domain_items + featured_papers + paper_appendix
    }
    brief_candidates = unique_items(
        [
            item
            for item in list(layers.get("brief", [])) + source_news_candidate_pool
            if str(item.get("canonical_url") or item.get("url") or "") not in used
            and item.get("content_type") != "paper"
        ]
    )
    brief: List[Dict[str, Any]] = []
    for item in sort_items(brief_candidates):
        if is_v10_design:
            candidate = build_source_grounded_news_brief(item)
            if not candidate:
                continue
        else:
            candidate = enrich_editorial_fields({**dict(item), "_v8_force_fact_title": True})
            if title_looks_bad(candidate) or has_untranslated_prose(candidate.get("brief_line")):
                continue
        candidate["report_section"] = "brief"
        candidate["freshness_label"] = "今日新增" if datetime.now() - parse_datetime(candidate.get("publish_date", "")) <= timedelta(hours=36) else "历史补位"
        brief.append(candidate)
        if len(brief) >= brief_limit:
            break

    result = {
        "must_read": must_read,
        "physical_ai": [],
        "watch": domain_items,
        "featured_papers": featured_papers,
        "paper_appendix": paper_appendix,
        "research": [],
        "brief": brief,
    }
    rank = 1
    for key in REPORT_SECTION_ORDER:
        for item in result.get(key, []):
            item["report_rank"] = rank
            rank += 1
    return result


def v8_backfill_pool_ready(items: List[Dict[str, Any]], report_config: Optional[Dict[str, Any]] = None) -> bool:
    report_config = report_config or {}
    if not is_continuous_reader_design(report_config):
        return True
    eligible: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        key = str(item.get("canonical_url") or item.get("url") or "")
        if not key or key in seen or item.get("content_type") == "paper":
            continue
        seen.add(key)
        candidate = enrich_editorial_fields({**dict(item), "_v8_force_fact_title": True})
        if candidate.get("quality_tier") == "brief":
            continue
        if not is_focus_quality_item(candidate) or not v8_has_concrete_news_evidence(candidate):
            continue
        eligible.append(candidate)
    provisional_layers = {
        "must_read": eligible,
        "physical_ai": [],
        "watch": [],
        "featured_papers": [],
        "paper_appendix": [],
        "research": [],
        "brief": [],
    }
    selected_layers = apply_v8_reading_budget(provisional_layers, report_config)
    selected = list(selected_layers.get("must_read", [])) + list(selected_layers.get("watch", []))
    source_count = len(
        {
            item_host(item) or normalize_text(item.get("source_detail", ""), item.get("platform", ""))
            for item in selected
        }
    )
    domain_count = len({str(item.get("domain_key") or learning_domain_key(item)) for item in selected})
    return (
        len(selected) >= int(report_config.get("v8_focus_candidate_target", 12) or 12)
        and source_count >= int(report_config.get("v8_focus_source_target", 6) or 6)
        and domain_count >= int(report_config.get("v8_focus_domain_target", 4) or 4)
    )


def compact_email_html(html_content: str) -> str:
    compacted = re.sub(r"<!--(?!\[if)[\s\S]*?-->", "", str(html_content or ""))
    compacted = re.sub(r">\r?\n[ \t]*<", "><", compacted)
    compacted = re.sub(r"\r?\n[ \t]*\r?\n+", "\n", compacted)
    return compacted.strip()


def build_v11_email_volume_layers(
    layers: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    volume_definitions = (
        ("新闻、博客与访谈", "news"),
        ("技术方法与工程实践", "technical"),
        ("论文精读", "paper"),
    )
    volumes: List[Dict[str, Any]] = []
    for label, primary_section in volume_definitions:
        volume_layers = {
            key: [
                dict(item)
                for item in layers.get(key, [])
                if report_primary_section(item) == primary_section
            ]
            for key in REPORT_SECTION_ORDER
        }
        items = flatten_report_layers(volume_layers)
        if not items:
            continue
        volumes.append(
            {
                "label": label,
                "primary_section": primary_section,
                "layers": volume_layers,
                "items": items,
            }
        )
    return volumes


def prepare_v11_email_delivery_volumes(
    html_report: str,
    layers: Dict[str, List[Dict[str, Any]]],
    report_config: Dict[str, Any],
    render_volume: Callable[[int, Dict[str, Any]], str],
) -> Tuple[List[Dict[str, Any]], bool]:
    original = [{
        "label": "完整日报",
        "html": html_report,
        "primary_section": "all",
        "item_count": len(flatten_report_layers(layers)),
        "items": flatten_report_layers(layers),
        "layers": layers,
    }]
    max_bytes = int(report_config.get("email_html_max_bytes", 104448) or 104448)
    if (
        str(report_config.get("product_mode") or "") != "intelligence_v11_editorial_library"
        or not bool(report_config.get("email_split_enabled", True))
        or len(html_report.encode("utf-8")) <= max_bytes
    ):
        return original, False

    def sliced_volume(
        volume: Dict[str, Any],
        selected_items: List[Dict[str, Any]],
        part_index: int = 1,
        part_count: int = 1,
    ) -> Dict[str, Any]:
        selected_keys = {editorial_item_render_key(item) for item in selected_items}
        volume_layers = {
            key: [
                item
                for item in volume["layers"].get(key, [])
                if editorial_item_render_key(item) in selected_keys
            ]
            for key in REPORT_SECTION_ORDER
        }
        label = str(volume["label"])
        if part_count > 1:
            label = f"{label}（{part_index}/{part_count}）"
        return {
            "label": label,
            "primary_section": volume["primary_section"],
            "layers": volume_layers,
            "items": selected_items,
        }

    def render_candidate(index: int, volume: Dict[str, Any]) -> Dict[str, Any]:
        volume_html = compact_email_html(render_volume(index, volume))
        return {
            **volume,
            "html": volume_html,
            "item_count": len(volume["items"]),
            "size_bytes": len(volume_html.encode("utf-8")),
        }

    rendered: List[Dict[str, Any]] = []
    for base_volume in build_v11_email_volume_layers(layers):
        candidate = render_candidate(
            len(rendered) + 1,
            sliced_volume(base_volume, list(base_volume["items"])),
        )
        if int(candidate["size_bytes"]) <= max_bytes:
            rendered.append(candidate)
            continue

        items = list(base_volume["items"])
        split_candidates: List[Dict[str, Any]] = []
        for part_count in range(2, len(items) + 1):
            base_size, remainder = divmod(len(items), part_count)
            cursor = 0
            trial: List[Dict[str, Any]] = []
            for part_index in range(1, part_count + 1):
                chunk_size = base_size + (1 if part_index <= remainder else 0)
                chunk = items[cursor : cursor + chunk_size]
                cursor += chunk_size
                volume = sliced_volume(base_volume, chunk, part_index, part_count)
                trial.append(render_candidate(len(rendered) + part_index, volume))
            if trial and all(int(volume["size_bytes"]) <= max_bytes for volume in trial):
                split_candidates = trial
                break
        if not split_candidates:
            return original, False
        rendered.extend(split_candidates)

    if rendered:
        return rendered, True
    return original, False


def send_email_delivery_volumes(
    notifier: Any,
    recipient: str,
    base_subject: str,
    volumes: List[Dict[str, Any]],
) -> Dict[str, Any]:
    volume_count = len(volumes)
    subjects = [
        (
            base_subject
            if volume_count == 1
            else f"{base_subject} [{index}/{volume_count}] {volume['label']}"
        )
        for index, volume in enumerate(volumes, start=1)
    ]
    sent_count = 0
    for subject, volume in zip(subjects, volumes):
        if not notifier.send_email(
            recipient_email=recipient,
            subject=subject,
            html_content=str(volume["html"]),
        ):
            return {
                "success": False,
                "subjects": subjects,
                "sent_count": sent_count,
                "volume_count": volume_count,
            }
        sent_count += 1
    return {
        "success": True,
        "subjects": subjects,
        "sent_count": sent_count,
        "volume_count": volume_count,
    }


class _V11RenderedItemTextParser(HTMLParser):
    """Collect visible text from each V11 item container in the final email HTML."""

    _VOID_TAGS = {
        "area", "base", "br", "col", "embed", "hr", "img", "input",
        "link", "meta", "param", "source", "track", "wbr",
    }
    _BLOCK_TAGS = {
        "article", "blockquote", "div", "h1", "h2", "h3", "h4", "h5",
        "h6", "li", "p", "section", "table", "td", "th", "tr",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._tag_stack: List[str] = []
        self._captures: List[Dict[str, Any]] = []
        self.item_text: Dict[str, str] = {}
        self.item_structured_text: Dict[str, str] = {}

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        normalized_tag = str(tag or "").lower()
        if normalized_tag == "br" or normalized_tag in self._BLOCK_TAGS:
            for capture in self._captures:
                capture["parts"].append("\n")
        if normalized_tag not in self._VOID_TAGS:
            self._tag_stack.append(normalized_tag)
        item_key = dict(attrs).get("data-v11-item-key")
        if item_key:
            self._captures.append(
                {
                    "key": str(item_key),
                    "tag": normalized_tag,
                    "depth": len(self._tag_stack),
                    "parts": [],
                }
            )

    def handle_data(self, data: str) -> None:
        if not data:
            return
        for capture in self._captures:
            capture["parts"].append(data)

    def handle_endtag(self, tag: str) -> None:
        normalized_tag = str(tag or "").lower()
        if normalized_tag in self._BLOCK_TAGS:
            for capture in self._captures:
                capture["parts"].append("\n")
        for capture in list(self._captures):
            if (
                capture["tag"] == normalized_tag
                and capture["depth"] == len(self._tag_stack)
            ):
                raw_text = "".join(capture["parts"])
                normalized_text = re.sub(r"\s+", " ", raw_text).strip()
                self.item_text[capture["key"]] = normalized_text
                self.item_structured_text[capture["key"]] = (
                    _normalize_editorial_paragraphs(raw_text)
                )
                self._captures.remove(capture)
        if self._tag_stack:
            if self._tag_stack[-1] == normalized_tag:
                self._tag_stack.pop()
            elif normalized_tag in self._tag_stack:
                reverse_index = self._tag_stack[::-1].index(normalized_tag)
                del self._tag_stack[len(self._tag_stack) - reverse_index - 1 :]


def _normalize_editorial_paragraphs(value: Any) -> str:
    paragraphs = [
        re.sub(r"\s+", " ", html_lib.unescape(part)).strip()
        for part in re.split(r"(?:\r\n|\r|\n)+", str(value or ""))
        if re.sub(r"\s+", " ", html_lib.unescape(part)).strip()
    ]
    return "\n".join(paragraphs)


def _editorial_paragraph_digest(value: Any) -> str:
    normalized = _normalize_editorial_paragraphs(value)
    if normalized.count("\n") < 1:
        return ""
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _v11_rendered_item_text(html_content: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    parser = _V11RenderedItemTextParser()
    parser.feed(html_content)
    parser.close()
    return parser.item_text, parser.item_structured_text


def scan_final_html_quality(
    html_content: str,
    layers: Dict[str, List[Dict[str, Any]]],
    quality_config: Optional[Dict[str, Any]] = None,
    report_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    quality_config = quality_config or {}
    report_config = report_config or {}
    v10_mode = is_learning_digest_design(report_config)
    v11_mode = str(report_config.get("product_mode") or "") == "intelligence_v11_editorial_library"
    items = flatten_report_layers(layers)
    focus_items = list(layers.get("must_read", [])) + list(layers.get("watch", [])) + list(layers.get("featured_papers", []))
    paper_items = list(layers.get("featured_papers", []))
    fresh_paper_items = [item for item in paper_items if not item.get("is_reappeared_update")]
    visible_paper_items = paper_items + list(layers.get("paper_appendix", []))
    visible_fresh_paper_items = [item for item in visible_paper_items if not item.get("is_reappeared_update")]
    public_texts: List[str] = []
    opening_counts: Counter[str] = Counter()
    attribution_opening_counts: Counter[str] = Counter()
    attribution_opening_run = 0
    max_attribution_opening_run = 0
    previous_attribution_opening = ""
    sentence_counts: Counter[str] = Counter()
    displayed_body_items = focus_items
    for item in displayed_body_items:
        body = str(
            item.get("paper_plain_summary") or item.get("paper_technical_intro")
            if item.get("content_type") == "paper"
            else item.get("analysis_body") or item.get("summary") or ""
        )
        evidence = str(item.get("evidence_line") or "")
        public_texts.extend([str(item.get("editorial_title") or item.get("title_cn") or ""), body, evidence])
        opening = re.sub(r"\W+", "", body)[:12]
        if len(opening) >= 6:
            opening_counts[opening] += 1
        attribution_opening = attribution_opener_pattern(body)
        if attribution_opening:
            attribution_opening_counts[attribution_opening] += 1
        if attribution_opening and attribution_opening == previous_attribution_opening:
            attribution_opening_run += 1
        elif attribution_opening:
            attribution_opening_run = 1
        else:
            attribution_opening_run = 0
        previous_attribution_opening = attribution_opening
        max_attribution_opening_run = max(
            max_attribution_opening_run,
            attribution_opening_run,
        )
        for sentence in re.split(r"[。！？.!?]+", body):
            key = re.sub(r"\W+", "", normalize_text(sentence))[:100]
            if len(key) >= 18:
                sentence_counts[key] += 1

    title_items = focus_items + list(layers.get("paper_appendix", []))
    bad_title_examples: List[str] = []
    for item in title_items:
        title = str(item.get("title_cn") or item.get("title") or "").strip()
        token_leak = any(token in title for token in ("...", "…", "Product Release", "Industry Update"))
        if title_looks_bad(item) or token_leak:
            bad_title_examples.append(title)
    bad_title_count = len(bad_title_examples)
    untranslated_count = sum(
        1
        for item in focus_items
        if "untranslated_fact" in set(item.get("editorial_flags") or [])
        or any(has_untranslated_prose(text) for text in (
            item.get("editorial_title"),
            item.get("analysis_body"),
            item.get("paper_technical_intro"),
            item.get("evidence_line"),
        ))
    )
    paper_mechanism_missing_count = sum(
        1
        for item in paper_items
        if not str((item.get("facts_cn") or {}).get("method") or "").strip()
    )
    paper_result_context_missing_count = sum(
        1
        for item in paper_items
        if not (
            str((item.get("facts_cn") or {}).get("metric_result") or "").strip()
            or any(
                re.search(r"\d|实验|结果|成功率|准确率|提升|降低|优于|超过|对比", str(point or ""), re.IGNORECASE)
                for point in (
                    [(item.get("facts_cn") or {}).get("evidence")]
                    if isinstance((item.get("facts_cn") or {}).get("evidence"), str)
                    else ((item.get("facts_cn") or {}).get("evidence") or [])
                )
            )
            or (
                v11_mode
                and paper_technical_intro_passes(item.get("paper_technical_intro"))
            )
        )
    )
    paper_body_min = max(0, int(report_config.get("paper_body_char_min", 0) or 0))
    paper_body_max = max(paper_body_min, int(report_config.get("paper_body_char_limit", 220) or 220))
    paper_intro_length_fail_count = sum(
        1
        for item in paper_items
        if not paper_body_min <= len(str(item.get("paper_technical_intro") or "")) <= paper_body_max
    )
    paper_plain_summary_fail_count = sum(
        1 for item in paper_items if not paper_plain_summary_passes(item.get("paper_plain_summary"))
    )
    featured_urls = {str(item.get("canonical_url") or item.get("url") or "") for item in paper_items}
    appendix_urls = {str(item.get("canonical_url") or item.get("url") or "") for item in layers.get("paper_appendix", [])}
    exact_duplicate_count = sum(1 for count in sentence_counts.values() if count > 1)
    truncated_focus_text_count = sum(
        1
        for item in focus_items
        if any(
            re.search(r"…|\.\.\.", str(value or ""))
            for value in (
                item.get("editorial_title") or item.get("title_cn"),
                item.get("analysis_body"),
                item.get("paper_technical_intro"),
                item.get("evidence_line"),
            )
        )
    )
    html_without_assets = re.sub(r"<(style|script)\b[^>]*>.*?</\1>", "", html_content, flags=re.IGNORECASE | re.DOTALL)
    visible_text = html_lib.unescape(re.sub(r"<[^>]+>", " ", html_without_assets))
    visible_text = re.sub(r"\s+", " ", visible_text).strip()
    if v11_mode:
        v11_rendered_item_text, v11_rendered_item_structured_text = (
            _v11_rendered_item_text(html_content)
        )
    else:
        v11_rendered_item_text, v11_rendered_item_structured_text = {}, {}
    v11_content_fidelity_missing_examples: List[Dict[str, Any]] = []
    v11_key_number_fidelity_missing_examples: List[Dict[str, Any]] = []
    v11_editorial_source_hash_missing_examples: List[Dict[str, Any]] = []
    v11_editorial_source_mismatch_examples: List[Dict[str, Any]] = []
    if v11_mode:
        for item in items:
            render_key = editorial_item_render_key(item)
            rendered_item_text = v11_rendered_item_text.get(render_key, "") if render_key else ""
            rendered_item_structured_text = (
                v11_rendered_item_structured_text.get(render_key, "")
                if render_key
                else ""
            )
            item_facts = item.get("facts") if isinstance(item.get("facts"), dict) else {}
            expected_fragments = {
                "title": item.get("title_cn") or item.get("title"),
                "source_excerpt": item_facts.get("source_excerpt")
                or item.get("source_excerpt"),
                "evidence_locator": item_facts.get("evidence_locator")
                or item.get("evidence_locator"),
            }
            if report_primary_section(item) == "paper":
                expected_fragments.update(
                    {
                        "paper_plain_summary": item.get("paper_plain_summary"),
                        "paper_technical_intro": item.get("paper_technical_intro"),
                    }
                )
            else:
                expected_fragments["analysis_body"] = (
                    item.get("analysis_body") or item.get("summary")
                )
            missing_fragments = []
            for field, value in expected_fragments.items():
                normalized_value = re.sub(
                    r"\s+", " ", html_lib.unescape(str(value or ""))
                ).strip()
                if normalized_value and normalized_value not in rendered_item_text:
                    missing_fragments.append(field)
                    continue
                structured_value = _normalize_editorial_paragraphs(value)
                if (
                    structured_value.count("\n") >= 1
                    and structured_value not in rendered_item_structured_text
                ):
                    missing_fragments.append(f"{field}:paragraphs")
            if missing_fragments:
                v11_content_fidelity_missing_examples.append(
                    {
                        "url": str(item.get("canonical_url") or item.get("url") or ""),
                        "fields": missing_fragments,
                        "container_missing": not bool(rendered_item_text),
                    }
                )
            facts_cn = item.get("facts_cn") if isinstance(item.get("facts_cn"), dict) else {}
            key_numbers = item_facts.get("key_numbers")
            if not isinstance(key_numbers, list):
                key_numbers = facts_cn.get("key_numbers")
            if isinstance(key_numbers, list) and key_numbers:
                expected_key_tokens = set().union(
                    *(
                        CodexResearchInboxCollector._metric_tokens(value)
                        for value in key_numbers
                    )
                )
                if report_primary_section(item) == "paper":
                    public_key_text = " ".join(
                        [
                            str(item.get("paper_plain_summary") or ""),
                            str(item.get("paper_technical_intro") or ""),
                        ]
                    )
                else:
                    public_key_text = str(
                        item.get("analysis_body") or item.get("summary") or ""
                    )
                public_key_tokens = CodexResearchInboxCollector._metric_tokens(
                    public_key_text
                )
                missing_key_tokens = expected_key_tokens - public_key_tokens
                if missing_key_tokens:
                    v11_key_number_fidelity_missing_examples.append(
                        {
                            "url": str(item.get("canonical_url") or item.get("url") or ""),
                            "missing_tokens": sorted(missing_key_tokens),
                        }
                    )
            if str(item.get("analysis_version") or "") not in SUPPORTED_CODEX_RESEARCH_ANALYSIS_VERSIONS:
                continue
            facts = item_facts
            source_hashes = facts.get("editorial_source_hashes")
            if not isinstance(source_hashes, dict):
                source_hashes = facts_cn.get("editorial_source_hashes")
            if not isinstance(source_hashes, dict) or not source_hashes:
                v11_editorial_source_hash_missing_examples.append(
                    {"url": str(item.get("canonical_url") or item.get("url") or "")}
                )
                continue
            paper_item = report_primary_section(item) == "paper"
            current_values = {
                "title_cn": item.get("title_cn") or item.get("title"),
                "summary": (
                    item.get("summary")
                    if paper_item
                    else item.get("analysis_body") or item.get("summary")
                ),
                "source_excerpt": facts.get("source_excerpt"),
                "evidence_locator": facts.get("evidence_locator"),
            }
            if isinstance(facts.get("key_numbers"), list):
                current_values["key_numbers"] = json.dumps(
                    facts.get("key_numbers"),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            if paper_item:
                current_values.update(
                    {
                        "paper_plain_summary": item.get("paper_plain_summary"),
                        "paper_technical_intro": item.get("paper_technical_intro"),
                    }
                )
            mismatched_fields = []
            for field, expected_hash in source_hashes.items():
                if field not in current_values or not str(expected_hash or "").strip():
                    continue
                normalized = re.sub(r"\s+", " ", str(current_values[field] or "")).strip()
                current_hash = (
                    hashlib.sha256(normalized.encode("utf-8")).hexdigest()
                    if normalized
                    else ""
                )
                if current_hash != str(expected_hash):
                    mismatched_fields.append(field)
            if mismatched_fields:
                v11_editorial_source_mismatch_examples.append(
                    {
                        "url": str(item.get("canonical_url") or item.get("url") or ""),
                        "fields": mismatched_fields,
                    }
                )
            paragraph_hashes = facts.get("editorial_paragraph_hashes")
            if not isinstance(paragraph_hashes, dict):
                paragraph_hashes = facts_cn.get("editorial_paragraph_hashes")
            if isinstance(paragraph_hashes, dict):
                paragraph_mismatches = []
                for field, expected_hash in paragraph_hashes.items():
                    if field not in current_values or not str(expected_hash or "").strip():
                        continue
                    if _editorial_paragraph_digest(current_values[field]) != str(expected_hash):
                        paragraph_mismatches.append(f"{field}:paragraphs")
                if paragraph_mismatches:
                    v11_editorial_source_mismatch_examples.append(
                        {
                            "url": str(item.get("canonical_url") or item.get("url") or ""),
                            "fields": paragraph_mismatches,
                        }
                    )
    html_size_bytes = len(html_content.encode("utf-8"))
    html_size_warning_bytes = max(
        1,
        int(report_config.get("email_html_warning_bytes", 97280) or 97280),
    )
    html_size_max_bytes = max(
        html_size_warning_bytes,
        int(report_config.get("email_html_max_bytes", 104448) or 104448),
    )
    memory_texts = [
        re.sub(r"^0\d\s*", "", re.sub(r"\s+", " ", html_lib.unescape(re.sub(r"<[^>]+>", " ", row))).strip())
        for row in re.findall(
            r'<div\s+class="v8-memory-row"[^>]*>(.*?)</div>',
            html_content,
            flags=re.IGNORECASE | re.DOTALL,
        )
    ]
    def focus_source_key(item: Dict[str, Any]) -> str:
        host = item_host(item)
        source_detail = normalize_text(item.get("source_detail", ""), item.get("platform", ""))
        if v11_mode and host == "github.com" and source_detail:
            return f"github:{source_detail}"
        return host or source_detail or "unknown"

    def focus_topic_key(item: Dict[str, Any]) -> str:
        if v11_mode and report_primary_section(item) == "technical":
            facts = item.get("facts") if isinstance(item.get("facts"), dict) else {}
            category = str(
                item.get("technical_category") or facts.get("technical_category") or "other"
            ).strip().lower()
            return f"technical:{category}"
        return str(item.get("domain_key") or learning_domain_key(item))

    source_counts = Counter(
        focus_source_key(item)
        for item in focus_items
        if item.get("content_type") != "paper"
    )
    topic_counts = Counter(
        focus_topic_key(item)
        for item in focus_items
        if item.get("content_type") != "paper"
    )
    source_concentration = max(source_counts.values(), default=0) / max(1, sum(source_counts.values()))
    source_news_brief_count = sum(
        1
        for item in layers.get("brief", [])
        if item.get("content_type") != "paper" and item.get("source_grounded_brief")
    )
    v11_rendered_counts = {"news": 0, "technical": 0, "paper": 0}
    v11_supplemental_expected_count = sum(
        1
        for item in items
        if "supplemental_older_source" in set(item.get("quality_flags") or [])
    )
    v11_supplemental_counts_by_section = Counter(
        report_primary_section(item)
        for item in items
        if "supplemental_older_source" in set(item.get("quality_flags") or [])
    )
    supplemental_visible_limits = {
        str(section): max(0, int(limit or 0))
        for section, limit in dict(
            report_config.get("supplemental_visible_max_by_section")
            or {"news": 5, "technical": 5, "paper": 3}
        ).items()
    }
    v11_supplemental_limit_exceeded = {
        section: {
            "count": int(v11_supplemental_counts_by_section.get(section, 0)),
            "max": limit,
        }
        for section, limit in supplemental_visible_limits.items()
        if int(v11_supplemental_counts_by_section.get(section, 0)) > limit
    }
    supplemental_max_age_hours = {
        str(section): max(0, int(hours or 0))
        for section, hours in dict(
            report_config.get("supplemental_max_age_hours_by_section")
            or {"news": 168, "technical": 720, "paper": 720}
        ).items()
    }
    v11_supplemental_age_violation_examples: List[Dict[str, Any]] = []
    now = datetime.now()
    for item in items:
        if "supplemental_older_source" not in set(item.get("quality_flags") or []):
            continue
        section = report_primary_section(item)
        maximum_hours = supplemental_max_age_hours.get(section, 0)
        published_at = parse_datetime(str(item.get("publish_date") or ""))
        if maximum_hours <= 0 or published_at == datetime.min:
            continue
        age_hours = (now - published_at).total_seconds() / 3600
        if age_hours > maximum_hours:
            v11_supplemental_age_violation_examples.append({
                "section": section,
                "url": str(item.get("canonical_url") or item.get("url") or ""),
                "publish_date": str(item.get("publish_date") or ""),
                "age_hours": round(age_hours, 1),
                "max_age_hours": maximum_hours,
            })
    v11_supplemental_label_count = html_content.count("补充阅读 ·")
    valid_claim_types = {
        "verified_fact",
        "official_claim",
        "interview_opinion",
        "analysis",
        "research_result",
    }
    v11_claim_label_expected_count = sum(
        1
        for item in items
        if str(
            item.get("claim_type")
            or (item.get("facts") or {}).get("claim_type")
            or ""
        ).strip().lower() in valid_claim_types
    )
    v11_claim_label_visible_count = html_content.count('class="v11-claim-label"')
    if v11_mode:
        for raw_count, raw_label in re.findall(
            r'<div\s+class="v11-count-value"[^>]*>(\d+)</div><div\s+class="v11-count-label"[^>]*>([^<]+)</div>',
            html_content,
            flags=re.IGNORECASE,
        ):
            label = re.sub(r"^本期", "", html_lib.unescape(raw_label).strip())
            if label.startswith("新闻"):
                v11_rendered_counts["news"] = int(raw_count)
            elif label.startswith("非论文技术"):
                v11_rendered_counts["technical"] = int(raw_count)
            elif label.startswith("论文"):
                v11_rendered_counts["paper"] = int(raw_count)

    def display_body_limit_for_item(item: Dict[str, Any]) -> int:
        primary_section = report_primary_section(item)
        content_type = str(item.get("content_type") or "").strip().lower()
        if primary_section == "paper":
            return int(report_config.get("paper_body_char_limit", 300) or 300)
        if content_type in {"interview", "podcast", "video"}:
            return int(report_config.get("interview_body_char_limit", 500) or 500)
        if primary_section == "technical":
            return int(report_config.get("technical_body_char_limit", 400) or 400)
        return int(report_config.get("news_body_char_limit", 180) or 180)

    display_body_over_limit_count = 0
    for item in displayed_body_items:
        body = str(
            item.get("paper_plain_summary") or item.get("paper_technical_intro") or ""
            if report_primary_section(item) == "paper"
            else item.get("analysis_body") or item.get("summary") or ""
        )
        if len(body) > display_body_limit_for_item(item):
            display_body_over_limit_count += 1
    low_value_labels = (
        "相较上次",
        "可信度：",
        "长期技术档案",
        "技术谱系：",
        "相对前序：",
        "本期相关论文",
        "论文索引",
        "读完后留下这些",
        "快讯 / 待确认",
    )
    low_value_module_count = sum(visible_text.count(label) for label in low_value_labels)
    if v11_mode:
        title_only_item_count = sum(
            1
            for item in visible_paper_items
            if not paper_plain_summary_passes(item.get("paper_plain_summary"))
            or not paper_technical_intro_passes(item.get("paper_technical_intro"))
        )
    elif v10_mode:
        appendix_blocks = re.findall(
            r'<div\s+class="v10-more-paper"[^>]*>(.*?)</div>\s*</div>?',
            html_content,
            flags=re.IGNORECASE | re.DOTALL,
        )
        title_only_item_count = sum(
            1 for block in appendix_blocks if 'class="v10-more-summary"' not in block
        )
    else:
        title_only_item_count = sum(
            1
            for item in paper_items + list(layers.get("paper_appendix", []))
            if not str(
                item.get("paper_plain_summary")
                or item.get("paper_compact_summary")
                or item.get("paper_technical_intro")
                or ""
            ).strip()
        )
    metrics = {
        "final_html_quality_status": "passed",
        "final_html_bad_title_count": bad_title_count,
        "final_html_bad_title_examples": bad_title_examples[:5],
        "untranslated_fact_count": untranslated_count,
        "exact_duplicate_sentence_count": exact_duplicate_count,
        "paper_mechanism_missing_count": paper_mechanism_missing_count,
        "paper_result_context_missing_count": paper_result_context_missing_count,
        "paper_intro_length_fail_count": paper_intro_length_fail_count,
        "paper_plain_summary_fail_count": paper_plain_summary_fail_count,
        "paper_plain_summary_pass_rate": round(
            (len(paper_items) - paper_plain_summary_fail_count) / len(paper_items),
            3,
        ) if paper_items else 1.0,
        "low_value_module_count": low_value_module_count,
        "title_only_item_count": title_only_item_count,
        "display_body_over_limit_count": display_body_over_limit_count,
        "appendix_body_overlap_count": len(featured_urls & appendix_urls),
        "truncated_focus_text_count": truncated_focus_text_count,
        "visible_text_chars": len(visible_text),
        "html_size_bytes": html_size_bytes,
        "html_size_kb": round(html_size_bytes / 1024, 1),
        "email_clipping_warning": html_size_bytes > html_size_warning_bytes,
        "email_clipping_risk": html_size_bytes > html_size_max_bytes,
        "reading_budget_underfilled": len(visible_text) < int(report_config.get("total_visible_chars_min", 5000) or 5000),
        "reading_budget_exceeded": len(visible_text) > int(report_config.get("total_visible_chars_max", 9000) or 9000),
        "max_opening_repeat_count": max(opening_counts.values(), default=0),
        "attribution_opener_counts": dict(attribution_opening_counts),
        "max_attribution_opener_repeat_count": max(
            attribution_opening_counts.values(),
            default=0,
        ),
        "max_attribution_opener_run": max_attribution_opening_run,
        "focus_source_concentration": round(source_concentration, 3),
        "details_element_count": len(re.findall(r"<details\b", html_content, flags=re.IGNORECASE)),
        "score_label_count": len(re.findall(r"\bScore\s+\d", html_content, flags=re.IGNORECASE)),
        "feedback_link_count": html_content.count("/feedback?"),
        "featured_paper_count": len(paper_items),
        "featured_fresh_paper_count": len(fresh_paper_items),
        "visible_paper_count": len(visible_paper_items),
        "visible_fresh_paper_count": len(visible_fresh_paper_items),
        "v11_source_note_count": html_content.count('class="v11-source-note"'),
        "v11_paper_plain_visible_count": (
            html_content.count('class="v10-paper-plain"')
            + html_content.count('class="v10-more-plain"')
        ),
        "v11_paper_technical_visible_count": (
            html_content.count('class="v10-paper-tech"')
            + html_content.count('class="v10-more-tech"')
        ),
        "v11_rendered_news_count": v11_rendered_counts["news"],
        "v11_rendered_technical_count": v11_rendered_counts["technical"],
        "v11_rendered_paper_count": v11_rendered_counts["paper"],
        "v11_supplemental_expected_count": v11_supplemental_expected_count,
        "v11_supplemental_label_count": v11_supplemental_label_count,
        "v11_supplemental_counts_by_section": dict(v11_supplemental_counts_by_section),
        "v11_supplemental_limit_exceeded": v11_supplemental_limit_exceeded,
        "v11_supplemental_age_violation_count": len(
            v11_supplemental_age_violation_examples
        ),
        "v11_supplemental_age_violation_examples": (
            v11_supplemental_age_violation_examples[:5]
        ),
        "v11_claim_label_expected_count": v11_claim_label_expected_count,
        "v11_claim_label_visible_count": v11_claim_label_visible_count,
        "v11_content_fidelity_missing_count": len(
            v11_content_fidelity_missing_examples
        ),
        "v11_content_fidelity_missing_examples": v11_content_fidelity_missing_examples[:5],
        "v11_key_number_fidelity_missing_count": len(
            v11_key_number_fidelity_missing_examples
        ),
        "v11_key_number_fidelity_missing_examples": (
            v11_key_number_fidelity_missing_examples[:5]
        ),
        "v11_editorial_source_hash_missing_count": len(
            v11_editorial_source_hash_missing_examples
        ),
        "v11_editorial_source_hash_missing_examples": (
            v11_editorial_source_hash_missing_examples[:5]
        ),
        "v11_editorial_source_mismatch_count": len(
            v11_editorial_source_mismatch_examples
        ),
        "v11_editorial_source_mismatch_examples": (
            v11_editorial_source_mismatch_examples[:5]
        ),
        "v11_inline_style_count": len(
            re.findall(r'<(?:body|div|table|h1|h2|h3|article|a)\b[^>]*\sstyle="', html_content, flags=re.IGNORECASE)
        ),
        "visible_information_count": max(
            0,
            len(re.findall(r'class="v10-entry"', html_content)) - len(paper_items),
        ) if v10_mode else sum(1 for item in focus_items if item.get("content_type") != "paper"),
        "source_news_brief_count": source_news_brief_count,
        "memory_item_count": len(re.findall(r'class="v8-memory-row"', html_content)),
        "memory_total_chars": sum(len(text) for text in memory_texts),
        "memory_budget_exceeded": sum(len(text) for text in memory_texts)
        > int(report_config.get("memory_total_char_limit", 240) or 240),
        "focus_source_item_count": sum(source_counts.values()),
        "focus_source_max_count": max(source_counts.values(), default=0),
        "focus_topic_max_count": max(topic_counts.values(), default=0),
        "focus_brief_item_count": sum(1 for item in focus_items if item.get("quality_tier") == "brief"),
    }
    metrics["visible_news_count"] = metrics["visible_information_count"] + source_news_brief_count
    if v11_mode:
        metrics["visible_news_count"] = v11_rendered_counts["news"]
        metrics["visible_technical_count"] = v11_rendered_counts["technical"]
        metrics["visible_paper_count"] = v11_rendered_counts["paper"]
        metrics["visible_information_count"] = (
            v11_rendered_counts["news"] + v11_rendered_counts["technical"]
        )
    thresholds = {
        "final_html_bad_title_count": int(quality_config.get("final_html_bad_title_count", 0) or 0),
        "untranslated_fact_count": int(quality_config.get("untranslated_fact_count", 0) or 0),
        "exact_duplicate_sentence_count": int(quality_config.get("exact_duplicate_sentence_count", 1) or 1),
        "paper_mechanism_missing_count": int(quality_config.get("paper_mechanism_missing_count", 0) or 0),
        "paper_result_context_missing_count": int(quality_config.get("paper_result_context_missing_count", 0) or 0),
        "appendix_body_overlap_count": int(quality_config.get("appendix_body_overlap_count", 0) or 0),
        "truncated_focus_text_count": int(quality_config.get("truncated_focus_text_count", 0) or 0),
        "paper_intro_length_fail_count": int(quality_config.get("paper_intro_length_fail_count", 0) or 0),
    }
    if (
        any(int(metrics[key]) > limit for key, limit in thresholds.items())
        or metrics["reading_budget_underfilled"]
        or metrics["reading_budget_exceeded"]
        or metrics["email_clipping_risk"]
        or metrics["max_opening_repeat_count"]
        > int(quality_config.get("max_opening_repeat_count", 2) or 2)
        or metrics["max_attribution_opener_repeat_count"]
        > int(quality_config.get("max_attribution_opener_repeat_count", 8) or 8)
        or metrics["max_attribution_opener_run"]
        > int(quality_config.get("max_attribution_opener_run", 1) or 1)
        or (
            metrics["focus_source_item_count"] >= 8
            and metrics["focus_source_concentration"]
            > float(quality_config.get("focus_source_concentration_max", 0.25) or 0.25)
        )
        or metrics["focus_source_max_count"] > int(report_config.get("source_focus_limit", 2) or 2)
        or metrics["focus_topic_max_count"] > int(report_config.get("topic_focus_limit", 3) or 3)
        or metrics["display_body_over_limit_count"] > 0
        or (
            v11_mode
            and (
                metrics["v11_source_note_count"] < len(items)
                or metrics["v11_paper_plain_visible_count"] < len(visible_paper_items)
                or metrics["v11_paper_technical_visible_count"] < len(visible_paper_items)
                or metrics["v11_rendered_news_count"]
                < int(report_config.get("min_visible_news_count", 20) or 20)
                or metrics["v11_rendered_technical_count"]
                < int(report_config.get("min_visible_technical_count", 20) or 20)
                or metrics["v11_rendered_paper_count"]
                < int(report_config.get("min_visible_paper_count", 15) or 15)
                or metrics["v11_supplemental_label_count"]
                < metrics["v11_supplemental_expected_count"]
                or bool(metrics["v11_supplemental_limit_exceeded"])
                or metrics["v11_supplemental_age_violation_count"] > 0
                or metrics["v11_claim_label_expected_count"] != len(items)
                or metrics["v11_claim_label_visible_count"]
                != metrics["v11_claim_label_expected_count"]
                or metrics["v11_content_fidelity_missing_count"] > 0
                or metrics["v11_key_number_fidelity_missing_count"] > 0
                or metrics["v11_editorial_source_hash_missing_count"] > 0
                or metrics["v11_editorial_source_mismatch_count"] > 0
                or metrics["v11_inline_style_count"] < len(items) * 3 + 10
            )
        )
        or (not v10_mode and metrics["memory_budget_exceeded"])
        or metrics["focus_brief_item_count"] > 0
        or (
            v10_mode
            and not int(report_config.get("paper_technical_intro_min_count", 10))
            <= metrics["featured_fresh_paper_count"]
            <= int(report_config.get("paper_featured_limit", 12) or 12)
        )
        or (v10_mode and metrics["paper_plain_summary_fail_count"] > 0)
        or (v10_mode and metrics["low_value_module_count"] > 0)
        or (v10_mode and metrics["title_only_item_count"] > 0)
        or (
            v10_mode
            and metrics["visible_fresh_paper_count"] < int(report_config.get("min_visible_paper_count", 28))
        )
        or (
            v10_mode
            and metrics["visible_information_count"]
            < int(report_config.get("min_visible_information_count", 25) or 25)
        )
        or (
            v10_mode
            and metrics["visible_news_count"]
            < int(quality_config.get("min_visible_news_count", 8) or 8)
        )
        or (not v10_mode and is_continuous_reader_design(report_config) and not 6 <= metrics["featured_paper_count"] <= 8)
        or (not v10_mode and is_continuous_reader_design(report_config) and metrics["memory_item_count"] != 3)
        or (is_continuous_reader_design(report_config) and metrics["details_element_count"] > 0)
    ):
        metrics["final_html_quality_status"] = "failed"
    return metrics


def repeated_sentence_count(items: List[Dict[str, Any]]) -> int:
    counts: Dict[str, int] = {}
    repeated = 0
    for item in items:
        text = " ".join(
            str(item.get(field, "") or "")
            for field in ("summary", "why_it_matters", "why_now", "expected_effect", "future_impact")
        )
        for sentence in re.split(r"(?<=[。！？.!?])\s*", text):
            key = re.sub(r"\W+", "", normalize_text(sentence))[:80]
            if len(key) < 18:
                continue
            counts[key] = counts.get(key, 0) + 1
            if counts[key] == 3:
                repeated += 1
    return repeated


def _analysis_sentence_keys(text: str) -> List[str]:
    keys: List[str] = []
    for sentence in re.split(r"(?<=[。！？.!?])\s*", str(text or "")):
        key = re.sub(r"\W+", "", normalize_text(sentence))[:80]
        if len(key) >= 18:
            keys.append(key)
    return keys


def suppress_repeated_analysis_fields(
    layers: Dict[str, List[Dict[str, Any]]],
    *,
    max_repeats: int = 2,
) -> Dict[str, List[Dict[str, Any]]]:
    counts: Dict[str, int] = {}
    for item in flatten_report_layers(layers):
        for field in ("why_now", "expected_effect", "future_impact"):
            keys = _analysis_sentence_keys(str(item.get(field, "") or ""))
            if not keys:
                continue
            if any(counts.get(key, 0) >= max_repeats for key in keys):
                item[field] = ""
                continue
            for key in keys:
                counts[key] = counts.get(key, 0) + 1
    return layers


def evaluate_report_quality(layers: Dict[str, List[Dict[str, Any]]], quality_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    quality_config = quality_config or {}
    strict_v8 = bool(quality_config.get("v8_enabled", False))
    strict_v10 = bool(quality_config.get("v10_enabled", False))
    strict_v11 = bool(quality_config.get("v11_enabled", False))
    for section_key, section_items in list(layers.items()):
        layers[section_key] = [
            dict(item)
            if item.get("_final_render_item")
            else enrich_editorial_fields(enrich_learning_fields(item))
            for item in section_items
        ]
    must_read = layers.get("must_read", [])
    featured_papers = layers.get("featured_papers", [])
    fresh_featured_papers = [item for item in featured_papers if not item.get("is_reappeared_update")]
    focus_items = must_read + layers.get("physical_ai", []) + featured_papers
    if strict_v8:
        focus_items = must_read + layers.get("physical_ai", []) + layers.get("watch", []) + featured_papers
    all_items = flatten_report_layers(layers)
    failed_items: List[Dict[str, Any]] = []
    generic_count = 0
    low_focus_count = 0
    paper_pass_count = 0
    paper_fail_count = 0
    title_mismatch_count = 0
    bad_title_count = 0
    suspicious_claim_count = 0
    paper_description_fail_count = 0
    paper_numeric_parse_error_count = 0
    learning_card_missing_count = 0
    technical_context_missing_count = 0
    paper_technical_intro_missing_count = 0
    paper_plain_summary_missing_count = 0
    for item in all_items:
        flags = item_quality_flags(item)
        if not strict_v8:
            flags = [
                flag
                for flag in flags
                if flag not in {
                    "learning_card_missing",
                    "paper_technical_intro_fail",
                    "paper_mechanism_missing",
                    "paper_result_context_missing",
                    "untranslated_fact",
                }
            ]
        if strict_v8 and item in focus_items and not learning_card_complete(item):
            learning_card_missing_count += 1
            flags = sorted(set(flags + ["learning_card_missing"]))
        if strict_v8 and item in focus_items and not str(item.get("technical_context") or "").strip():
            technical_context_missing_count += 1
            flags = sorted(set(flags + ["technical_context_missing"]))
        if item.get("content_type") == "paper":
            paper_status = paper_core_summary_status(item)
            item["paper_core_summary_status"] = paper_status
            description = ReportGenerator()._paper_substantive_description(item)
            technical_intro = str(item.get("paper_technical_intro") or "")
            if not ReportGenerator()._paper_description_is_substantive(description):
                paper_description_fail_count += 1
                if not strict_v8:
                    flags = sorted(set(flags + ["paper_substantive_description_incomplete"]))
            if strict_v8 and item in focus_items and not paper_technical_intro_passes(technical_intro):
                paper_technical_intro_missing_count += 1
                flags = sorted(set(flags + ["paper_technical_intro_missing"]))
            if strict_v10 and item in featured_papers and not paper_plain_summary_passes(item.get("paper_plain_summary")):
                paper_plain_summary_missing_count += 1
                flags = sorted(set(flags + ["paper_plain_summary_missing"]))
            if "等量化证据" in description or re.search(r"论文给出了\d+(?:\.\d+)?(?:\s|$)", description):
                paper_numeric_parse_error_count += 1
                if not strict_v8:
                    flags = sorted(set(flags + ["paper_numeric_parse_error"]))
            if paper_status.get("passed"):
                paper_pass_count += 1
            else:
                paper_fail_count += 1
                if not strict_v8:
                    flags = sorted(set(flags + ["paper_core_summary_incomplete"]))
        item["quality_flags"] = flags
        if "generic_summary" in flags:
            generic_count += 1
        if "title_fact_mismatch" in flags:
            title_mismatch_count += 1
        if "bad_title" in flags:
            bad_title_count += 1
        if "suspicious_claim" in flags:
            suspicious_claim_count += 1
        blocking_flags = {
            "missing_facts", "low_evidence", "low_density", "generic_summary", "title_fact_mismatch",
            "bad_title", "mixed_language_title", "field_label_leak", "mojibake_suspect",
            "bad_public_phrase", "low_info_expanded", "suspicious_claim", "learning_card_missing",
            "technical_context_missing", "paper_technical_intro_missing",
            "paper_plain_summary_missing",
        }
        if not strict_v8:
            blocking_flags.update({"paper_core_summary_incomplete", "paper_substantive_description_incomplete", "paper_numeric_parse_error"})
        if item in focus_items and (blocking_flags & set(flags)):
            low_focus_count += 1
            failed_items.append(item)
        elif (not strict_v8 and ({"generic_summary", "missing_facts", "bad_title", "suspicious_claim"} & set(flags))) or (
            strict_v8 and "bad_title" in flags
        ):
            failed_items.append(item)
    repeated_count = repeated_sentence_count(all_items)
    high_evidence_count = sum(1 for item in all_items if evidence_quality_value(item) >= 0.45)
    high_evidence_warning = bool(
        all_items
        and high_evidence_count == len(all_items)
        and any(
            str(item.get("quality_tier") or "") == "brief"
            or str(item.get("report_section") or "") == "brief"
            for item in all_items
        )
    )
    status = "passed"
    domain_counts: Dict[str, int] = {}
    for item in all_items:
        domain_key = str(item.get("domain_key") or learning_domain_key(item))
        domain_counts[domain_key] = domain_counts.get(domain_key, 0) + 1
    domain_coverage_warning_count = sum(1 for key in LEARNING_DOMAIN_ORDER if domain_counts.get(key, 0) == 0)
    if strict_v8:
        if low_focus_count or bad_title_count or repeated_count > int(quality_config.get("max_repeated_sentence_count", 1)):
            status = "failed"
    elif generic_count or low_focus_count or bad_title_count or suspicious_claim_count or repeated_count > int(quality_config.get("max_repeated_sentence_count", 1)):
        status = "failed"
    legacy_status = status
    editorial_metrics = build_editorial_quality_metrics(
        all_items,
        physical_ai_min_items=int(quality_config.get("physical_ai_featured_min_count", 4) or 4),
        paper_technical_intro_min_count=int(quality_config.get("paper_technical_intro_min_count", 12)),
    )
    if quality_config.get("editorial_enabled", False):
        if editorial_metrics.get("editorial_quality_status") == "failed":
            status = "failed"
            failed_urls = set(editorial_metrics.get("failed_editorial_urls") or [])
            for item in all_items:
                if str(item.get("url") or "") in failed_urls and item not in failed_items:
                    failed_items.append(item)
        elif status != "failed":
            status = "passed"
    title_only_item_count = sum(
        1
        for item in featured_papers + layers.get("paper_appendix", [])
        if not str(
            item.get("paper_plain_summary")
            or item.get("paper_compact_summary")
            or item.get("paper_technical_intro")
            or ""
        ).strip()
    )
    visible_papers = featured_papers + list(layers.get("paper_appendix", []))
    visible_paper_count = len(visible_papers)
    visible_fresh_paper_count = sum(1 for item in visible_papers if not item.get("is_reappeared_update"))
    visible_information_count = sum(
        1
        for item in (
            list(layers.get("must_read", []))
            + list(layers.get("physical_ai", []))
            + list(layers.get("watch", []))
        )
        if item.get("content_type") != "paper" and str(item.get("quality_tier") or "") != "brief"
    )
    source_news_brief_count = sum(
        1
        for item in layers.get("brief", [])
        if item.get("content_type") != "paper" and item.get("source_grounded_brief")
    )
    visible_news_count = visible_information_count + source_news_brief_count
    visible_technical_count = 0
    if strict_v11:
        visible_news_count = sum(1 for item in all_items if report_primary_section(item) == "news")
        visible_technical_count = sum(1 for item in all_items if report_primary_section(item) == "technical")
        visible_information_count = visible_news_count + visible_technical_count
    visible_v10_items = [item for item in all_items if str(item.get("report_section") or "") != "brief"]
    primary_source_count = sum(
        1
        for item in visible_v10_items
        if infer_source_tier(item).lower() in {"official", "research", "primary"}
    )
    primary_source_ratio = round(primary_source_count / len(visible_v10_items), 3) if visible_v10_items else 1.0
    visible_technical_items = [
        item for item in visible_v10_items if report_primary_section(item) == "technical"
    ]
    technical_primary_source_count = sum(
        1
        for item in visible_technical_items
        if infer_source_tier(item).lower() in {"official", "research", "primary"}
    )
    technical_primary_source_ratio = (
        round(technical_primary_source_count / len(visible_technical_items), 3)
        if visible_technical_items
        else 1.0
    )
    event_keys = [
        normalized_event_title(item) or str(item.get("canonical_url") or item.get("url") or "")
        for item in visible_v10_items
    ]
    event_keys = [key for key in event_keys if key]
    duplicate_event_rate = round(1 - len(set(event_keys)) / len(event_keys), 3) if event_keys else 0.0
    paper_plain_pass_rate = (
        round((len(featured_papers) - paper_plain_summary_missing_count) / len(featured_papers), 3)
        if featured_papers
        else 1.0
    )
    if strict_v10 and (
        paper_plain_summary_missing_count
        or title_only_item_count
        or generic_count > int(quality_config.get("generic_phrase_count", 0) or 0)
        or suspicious_claim_count > int(quality_config.get("unsupported_claim_count", 0) or 0)
        or duplicate_event_rate > float(quality_config.get("duplicate_event_ratio_max", 0.08) or 0.08)
        or (
            not strict_v11
            and primary_source_ratio
            < float(quality_config.get("primary_source_ratio_min", 0.70) or 0.70)
        )
        or (
            strict_v11
            and technical_primary_source_ratio
            < float(quality_config.get("technical_primary_source_ratio_min", 0.80) or 0.80)
        )
        or paper_plain_pass_rate < float(quality_config.get("paper_plain_summary_pass_rate_min", 1.0) or 1.0)
        or visible_fresh_paper_count < int(quality_config.get("min_visible_paper_count", 28))
        or visible_information_count < int(quality_config.get("min_visible_information_count", 25) or 25)
        or visible_news_count < int(quality_config.get("min_visible_news_count", 8) or 8)
    ):
        status = "failed"
    if strict_v11 and technical_primary_source_ratio < float(
        quality_config.get("technical_primary_source_ratio_min", 0.80) or 0.80
    ):
        status = "failed"
    return {
        "status": status,
        "v3_quality_status": legacy_status,
        "v4_quality_status": legacy_status,
        "v5_quality_status": legacy_status,
        "generic_summary_count": generic_count,
        "generic_phrase_count": generic_count,
        "low_focus_quality_count": low_focus_count,
        "title_fact_mismatch_count": title_mismatch_count,
        "bad_title_count": bad_title_count,
        "suspicious_claim_count": suspicious_claim_count,
        "repeated_sentence_count": repeated_count,
        "duplicate_expression_count": repeated_count,
        "paper_core_summary_pass_count": paper_pass_count,
        "paper_core_summary_fail_count": paper_fail_count,
        "paper_substantive_description_fail_count": paper_description_fail_count,
        "paper_numeric_parse_error_count": paper_numeric_parse_error_count,
        "learning_card_missing_count": learning_card_missing_count,
        "technical_context_missing_count": technical_context_missing_count,
        "paper_technical_intro_missing_count": paper_technical_intro_missing_count,
        "paper_plain_summary_missing_count": paper_plain_summary_missing_count,
        "paper_plain_summary_pass_rate": paper_plain_pass_rate,
        "title_only_item_count": title_only_item_count,
        "visible_paper_count": visible_paper_count,
        "visible_fresh_paper_count": visible_fresh_paper_count,
        "featured_fresh_paper_count": len(fresh_featured_papers),
        "visible_information_count": visible_information_count,
        "source_news_brief_count": source_news_brief_count,
        "visible_news_count": visible_news_count,
        "visible_technical_count": visible_technical_count,
        "low_value_module_count": 0,
        "unsupported_claim_count": suspicious_claim_count,
        "primary_source_ratio": primary_source_ratio,
        "technical_primary_source_count": technical_primary_source_count,
        "technical_primary_source_ratio": technical_primary_source_ratio,
        "duplicate_event_rate": duplicate_event_rate,
        "domain_coverage_warning_count": domain_coverage_warning_count,
        "domain_counts": domain_counts,
        "physical_ai_item_count": len(layers.get("physical_ai", [])),
        "paper_selected_count": len(layers.get("featured_papers", [])),
        "paper_appendix_count": len(layers.get("paper_appendix", [])),
        "memory_card_count": len(must_read),
        "high_evidence_calibration_warning": high_evidence_warning,
        "failed_item_urls": [str(item.get("url", "")) for item in failed_items if item.get("url")],
        "failed_items": failed_items,
        **{key: value for key, value in editorial_metrics.items() if key != "failed_editorial_urls"},
    }


def model_path_breakdown(items: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for item in items:
        model_used = str(item.get("model_used", "") or "").strip()
        analysis_version = str(item.get("analysis_version", "") or "").strip()
        if model_used:
            key = model_used
        elif analysis_version == "v2":
            key = "fallback_v2"
        else:
            key = "legacy_pending_backfill"
        counts[key] = counts.get(key, 0) + 1
    return counts


def refresh_missing_model_path_items(
    items: List[Dict[str, Any]],
    *,
    db: Database,
    llm_processor: LLMProcessor,
    max_items: int,
) -> Dict[str, Dict[str, Any]]:
    if max_items <= 0:
        return {}
    refreshed: Dict[str, Dict[str, Any]] = {}
    candidates = [
        item for item in items
        if item.get("url") and not str(item.get("model_used", "") or "").strip()
    ][:max_items]
    for item in candidates:
        source_article = db.get_article_by_id(int(item.get("id"))) if item.get("id") else None
        source_article = source_article or item
        result = llm_processor.process_article(source_article)
        if not result:
            continue
        rewrite_attempts = int(source_article.get("rewrite_attempts", 0) or 0)
        db.update_article_processing(
            url=source_article["url"],
            summary=result.get("summary", source_article.get("summary", "")),
            score=float(result.get("score", source_article.get("score", 0)) or 0),
            keywords=result.get("keywords", source_article.get("keywords", [])),
            category=result.get("category", source_article.get("category", "Other")),
            title_cn=result.get("title_cn", source_article.get("title_cn", "")),
            summary_preview=result.get("summary_preview", source_article.get("summary_preview", "")),
            why_it_matters=result.get("why_it_matters", source_article.get("why_it_matters", "")),
            why_now=result.get("why_now", source_article.get("why_now", "")),
            expected_effect=result.get("expected_effect", source_article.get("expected_effect", "")),
            future_impact=result.get("future_impact", source_article.get("future_impact", "")),
            facts=result.get("facts", source_article.get("facts", {})),
            evidence_quality=float(result.get("evidence_quality", source_article.get("evidence_quality", 0.0)) or 0.0),
            information_density=float(result.get("information_density", source_article.get("information_density", 0.0)) or 0.0),
            model_used=result.get("model_used", source_article.get("model_used", "")),
            analysis_version="v2",
            quality_flags=item_quality_flags(result),
            rewrite_attempts=rewrite_attempts,
        )
        refreshed_item = llm_processor.prepare_report_item(source_article, result)
        refreshed_item["rewrite_attempts"] = rewrite_attempts
        refreshed[str(source_article.get("url", ""))] = refreshed_item
    return refreshed


def filter_updates_for_report(
    updates: List[Dict[str, Any]],
    target_count: int,
    minimum_count: int,
    source_preferences: Optional[Dict[str, Any]] = None,
    preference_config: Optional[Dict[str, Any]] = None,
    diversify_sources: bool = False,
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
        evidence_quality = evidence_quality_value(item)
        information_density = information_density_value(item)
        evidence_penalty = -0.9 if evidence_quality < 0.35 else 0.0
        density_penalty = -0.8 if information_density < 0.35 else 0.0
        editorial_penalty = -100.0 if str(item.get("model_used") or "") == "template_fallback" else 0.0
        candidate = dict(item)
        candidate["quality_score"] = quality_score
        candidate["preference_score"] = preference_score
        candidate["evidence_quality"] = evidence_quality
        candidate["information_density"] = information_density
        candidate["selection_score"] = (
            float(item.get("score", 0) or 0) * 0.7
            + quality_score * 1.35
            + evidence_quality * 0.8
            + information_density * 1.1
            + preference_score
            + evidence_penalty
            + density_penalty
            + editorial_penalty
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

    if not diversify_sources:
        return preferred[:target_count]

    qualified = [
        item
        for item in preferred
        if evidence_quality_value(item) >= 0.35
        and information_density_value(item) >= 0.35
        and not summary_looks_generic(item.get("summary", ""))
    ]
    selected: List[Dict[str, Any]] = []
    selected_urls: set[str] = set()
    source_counts: Counter[str] = Counter()

    def selection_source(item: Dict[str, Any]) -> str:
        return item_host(item) or normalize_text(item.get("source_detail", ""), item.get("platform", "")) or "unknown"

    for source_pass in range(4):
        for item in qualified:
            if len(selected) >= target_count:
                break
            url = str(item.get("canonical_url") or item.get("url") or "")
            source = selection_source(item)
            if not url or url in selected_urls or source_counts[source] != source_pass:
                continue
            selected.append(item)
            selected_urls.add(url)
            source_counts[source] += 1
        if len(selected) >= target_count:
            break
    if len(selected) < target_count:
        for item in preferred:
            url = str(item.get("canonical_url") or item.get("url") or "")
            if not url or url in selected_urls:
                continue
            selected.append(item)
            selected_urls.add(url)
            if len(selected) >= target_count:
                break
    return selected


def build_source_health_weights(source_health: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    config = config or {}
    if not config.get("enabled", True):
        return {}
    penalty_per_failure = float(config.get("penalty_per_consecutive_failure", -0.6))
    empty_success_penalty = float(config.get("empty_success_penalty", -0.15))
    max_penalty = abs(float(config.get("max_penalty", 2.0)))
    bonus_recent_success = float(config.get("bonus_recent_success", 0.15))
    weights: Dict[str, float] = {}
    for row in source_health.get("rows", []) or []:
        label = str(row.get("label", "") or "")
        if not label:
            continue
        consecutive_failures = int(row.get("consecutive_failures", 0) or 0)
        empty_success_count = int(row.get("recent_empty_success_count", 0) or 0)
        success_count = int(row.get("recent_success_count", 0) or 0)
        weight = consecutive_failures * penalty_per_failure
        weight += min(empty_success_count, 5) * empty_success_penalty
        if success_count >= 3 and consecutive_failures == 0:
            weight += bonus_recent_success
        weight = max(-max_penalty, min(max_penalty, weight))
        if abs(weight) >= 0.01:
            weights[label] = round(weight, 2)
    return weights


def apply_source_health_adjustments(
    source_preferences: Dict[str, Any],
    source_health: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    adjusted = copy.deepcopy(source_preferences or {})
    dynamic_weights = build_source_health_weights(source_health, config)
    if not dynamic_weights:
        return adjusted, {"enabled": bool((config or {}).get("enabled", True)), "weights": {}}
    source_weights = dict(adjusted.get("source_weights") or {})
    for label, weight in dynamic_weights.items():
        source_weights[label] = round(float(source_weights.get(label, 0) or 0) + weight, 2)
    adjusted["source_weights"] = source_weights
    return adjusted, {"enabled": True, "weights": dynamic_weights}


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


def normalize_paper_url(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    try:
        parsed = urlsplit(raw)
    except ValueError:
        return raw.rstrip("/").lower()
    if not parsed.scheme or not parsed.netloc:
        return raw.rstrip("/").lower()
    tracking_names = {
        "fbclid",
        "gclid",
        "mc_cid",
        "mc_eid",
        "ref",
        "source",
    }
    query = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if not key.lower().startswith("utm_") and key.lower() not in tracking_names
    ]
    scheme = "https" if parsed.scheme.lower() in {"http", "https"} else parsed.scheme.lower()
    host = parsed.hostname.lower() if parsed.hostname else parsed.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    port = parsed.port
    netloc = host if port in {None, 80 if scheme == "http" else 443} else f"{host}:{port}"
    path = re.sub(r"/{2,}", "/", unquote(parsed.path or "/")).rstrip("/") or "/"
    return urlunsplit((scheme, netloc, path, urlencode(sorted(query)), "")).lower()


def normalize_doi(value: Any) -> str:
    text = unquote(str(value or "")).strip().lower()
    match = re.search(r"(?:doi\.org/|doi:\s*)?(10\.\d{4,9}/[^?#\s]+)", text)
    if not match:
        return ""
    doi = match.group(1).rstrip(".,;:")
    for opening, closing in (("(", ")"), ("[", "]"), ("{", "}"), ("<", ">")):
        while doi.endswith(closing) and doi.count(closing) > doi.count(opening):
            doi = doi[:-1]
    return doi


def extract_arxiv_id(value: Any) -> str:
    url = unquote(str(value or "")).strip().lower()
    patterns = (
        r"(?:export\.)?arxiv\.org/(?:abs|pdf|html)/((?:\d{4}\.\d{4,5}|[a-z-]+(?:\.[a-z]{2})?/\d{7})(?:v\d+)?)(?:\.pdf)?",
        r"(?:www\.)?ar5iv(?:\.labs\.arxiv)?\.org/html/((?:\d{4}\.\d{4,5}|[a-z-]+(?:\.[a-z]{2})?/\d{7})(?:v\d+)?)",
    )
    for pattern in patterns:
        match = re.search(pattern, url, re.IGNORECASE)
        if match:
            return re.sub(r"v\d+$", "", match.group(1), flags=re.IGNORECASE)
    fallback = re.search(
        r"(?:export\.)?arxiv\.org/(?:abs|pdf|html)/([^/?#]+?)(?:\.pdf)?(?:[?#]|$)",
        url,
        re.IGNORECASE,
    )
    if fallback:
        return re.sub(r"v\d+$", "", fallback.group(1), flags=re.IGNORECASE)
    return ""


def paper_title_fingerprint(value: Any) -> str:
    title = html_lib.unescape(str(value or "")).strip().lower()
    if not title:
        return ""
    title = unicodedata.normalize("NFKC", title)
    title = re.sub(r"^\s*(?:arxiv\s*:\s*)?\d{4}\.\d{4,5}(?:v\d+)?\s*[-:|]\s*", "", title)
    title = re.sub(
        r"\s*(?:[-|]\s*)?(?:arxiv|papers with code|hugging face papers)\s*$",
        "",
        title,
        flags=re.IGNORECASE,
    )
    fingerprint = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", title)
    if len(fingerprint) < 18 or fingerprint in {
        "untitledpaper",
        "researchpaper",
        "newresearchpaper",
        "machinelearningpaper",
    }:
        return ""
    return fingerprint[:240]


def paper_history_keys(item: Dict[str, Any]) -> set[str]:
    keys: set[str] = set()
    explicit_arxiv_id = normalize_text(item.get("arxiv_id", "")).lower()
    if explicit_arxiv_id:
        explicit_arxiv_id = re.sub(r"^(?:arxiv:)?", "", explicit_arxiv_id)
        explicit_arxiv_id = re.sub(r"v\d+$", "", explicit_arxiv_id)
        if explicit_arxiv_id:
            keys.add(f"arxiv:{explicit_arxiv_id}")
    urls = {
        normalize_paper_url(item.get("canonical_url")),
        normalize_paper_url(item.get("url")),
    }
    for url in urls:
        arxiv_id = extract_arxiv_id(url)
        if arxiv_id:
            keys.add(f"arxiv:{arxiv_id}")
    explicit_doi = normalize_text(item.get("doi", ""), _paper_fact_value(item, "doi")).lower()
    for value in [explicit_doi, *urls]:
        doi = normalize_doi(value)
        if doi:
            keys.add(f"doi:{doi}")
    keys.update(url.rstrip("/") for url in urls if url)
    title_fingerprint = paper_title_fingerprint(item.get("title"))
    if title_fingerprint:
        keys.add(f"title:{title_fingerprint}")
    if not keys:
        title = normalize_text(item.get("title", ""), item.get("title_cn", ""))
        if title:
            keys.add(title)
    return keys


def paper_history_key(item: Dict[str, Any]) -> str:
    keys = paper_history_keys(item)
    for prefix in ("arxiv:", "doi:"):
        preferred = sorted(key for key in keys if key.startswith(prefix))
        if preferred:
            return preferred[0]
    return sorted(keys)[0] if keys else ""


def paper_identity_matches(left: Dict[str, Any], right: Dict[str, Any]) -> bool:
    left_keys = paper_history_keys(left)
    right_keys = paper_history_keys(right)
    if not left_keys or not right_keys:
        return False

    for prefix in ("arxiv:", "doi:"):
        left_strong = {key for key in left_keys if key.startswith(prefix)}
        right_strong = {key for key in right_keys if key.startswith(prefix)}
        if left_strong and right_strong and left_strong.isdisjoint(right_strong):
            return False
    return bool(left_keys & right_keys)


def paper_arxiv_version(item: Dict[str, Any]) -> int:
    versions = [int(item.get("arxiv_version", 1) or 1)]
    for value in (item.get("url"), item.get("canonical_url")):
        match = re.search(r"arxiv\.org/(?:abs|pdf)/[^/?#]+v(\d+)", str(value or "").lower())
        if match:
            versions.append(int(match.group(1)))
    return max(versions)


def _paper_fact_value(item: Dict[str, Any], key: str) -> str:
    for facts in (item.get("facts_cn"), item.get("facts")):
        if not isinstance(facts, dict):
            continue
        value = facts.get(key)
        if isinstance(value, list):
            value = " ".join(str(part) for part in value if part)
        normalized = normalize_text(value)
        if normalized:
            return normalized
    return ""


def detect_reappeared_paper_update(
    item: Dict[str, Any],
    previous_items: List[Dict[str, Any]],
) -> Tuple[str, str]:
    if not previous_items:
        return "今日新增", ""

    def substantively_new(current: str, previous_values: List[str], threshold: float = 0.86) -> bool:
        current = normalize_text(current).strip("。；;，, .").lower()
        normalized_previous = [
            normalize_text(value).strip("。；;，, .").lower()
            for value in previous_values
            if normalize_text(value)
        ]
        if not current:
            return False
        if not normalized_previous:
            return True
        return all(SequenceMatcher(None, current, previous).ratio() < threshold for previous in normalized_previous)

    def has_explicit_arxiv_version(previous: Dict[str, Any]) -> bool:
        try:
            if int(previous.get("arxiv_version", 0) or 0) > 0:
                return True
        except (TypeError, ValueError):
            pass
        return any(
            re.search(r"arxiv\.org/(?:abs|pdf)/[^/?#]+v\d+", str(value or ""), re.IGNORECASE)
            for value in (previous.get("url"), previous.get("canonical_url"))
        )

    def update_is_newer_than_history() -> bool:
        current_updated_at = parse_datetime(str(item.get("publish_date") or ""))
        history_delivery_at = max(
            (
                parse_datetime(
                    str(
                        previous.get("_history_delivery_at")
                        or previous.get("_history_created_at")
                        or ""
                    )
                )
                for previous in previous_items
            ),
            default=datetime.min,
        )
        return (
            current_updated_at != datetime.min
            and history_delivery_at != datetime.min
            and current_updated_at > history_delivery_at
        )

    history_has_send_time = any(
        parse_datetime(
            str(previous.get("_history_delivery_at") or previous.get("_history_created_at") or "")
        ) != datetime.min
        for previous in previous_items
    )

    def explicit_change_evidence(value: str = "") -> bool:
        actions: List[str] = []
        evidence: List[str] = []
        for facts in (item.get("facts_cn"), item.get("facts")):
            if not isinstance(facts, dict):
                continue
            if facts.get("action"):
                actions.append(str(facts.get("action")))
            raw_evidence = facts.get("evidence") or []
            if isinstance(raw_evidence, str):
                raw_evidence = [raw_evidence]
            evidence.extend(str(point or "") for point in raw_evidence if point)
        context = normalize_text(
            value,
            " ".join(actions),
            " ".join(evidence),
        )
        return bool(
            re.search(
                r"新增|新版本|首次|现已|刚刚|发布|释出|开源|上线|更新|"
                r"\b(?:newly|new version|released|open[- ]sourced|now available|updated|added)\b",
                context,
                re.IGNORECASE,
            )
        )

    latest = previous_items[0]
    current_version = paper_arxiv_version(item)
    previous_version = max(paper_arxiv_version(previous) for previous in previous_items)
    comparison_fields = (
        "abstract",
        "method",
        "core_method",
        "metric_result",
        "dataset_or_benchmark",
        "baseline",
        "limitation",
    )
    changed_fields = []
    for key in comparison_fields:
        current_value = (
            _paper_fact_value(item, key)
            or (normalize_text(item.get("content", "")) if key == "abstract" else "")
        )
        previous_value = (
            _paper_fact_value(latest, key)
            or (normalize_text(latest.get("content", "")) if key == "abstract" else "")
        )
        if not current_value:
            continue
        similarity = SequenceMatcher(None, current_value.lower(), previous_value.lower()).ratio() if previous_value else 0.0
        if not previous_value or similarity < 0.86:
            changed_fields.append(key)
    version_change_is_proven = (
        update_is_newer_than_history()
        if history_has_send_time
        else any(has_explicit_arxiv_version(previous) for previous in previous_items)
    )
    if current_version > previous_version and changed_fields and version_change_is_proven:
        return "版本更新", "新版本更新了" + "、".join(changed_fields[:3])
    asset_fields = ("code_or_project", "code_repository", "project_page", "model_weights", "dataset_release")
    for asset_key in asset_fields:
        current_asset = _paper_fact_value(item, asset_key)
        previous_assets = [_paper_fact_value(previous, asset_key) for previous in previous_items]
        if (
            current_asset
            and substantively_new(current_asset, previous_assets, threshold=0.92)
            and (any(normalize_text(value) for value in previous_assets) or explicit_change_evidence(current_asset))
        ):
            label = "代码已发布" if asset_key in {"code_or_project", "code_repository"} else "重要进展"
            return label, f"新增代码、项目或研究资产：{current_asset[:80]}"
    current_metric = _paper_fact_value(item, "metric_result")
    previous_metrics = [_paper_fact_value(previous, "metric_result") for previous in previous_items]
    current_deployment = _paper_fact_value(item, "deployment_context")
    previous_deployments = [_paper_fact_value(previous, "deployment_context") for previous in previous_items]
    previous_deployment = " ".join(previous_deployments)
    real_world_pattern = r"真实|实机|real[- ]world|robot deployment|部署"
    if (
        current_metric
        and substantively_new(current_metric, previous_metrics)
        and re.search(r"\d|提升|降低|超过|优于|成功率", current_metric)
        and (any(normalize_text(value) for value in previous_metrics) or update_is_newer_than_history())
    ) or (
        current_deployment
        and re.search(real_world_pattern, current_deployment, re.IGNORECASE)
        and substantively_new(current_deployment, previous_deployments)
        and not re.search(real_world_pattern, previous_deployment, re.IGNORECASE)
        and (any(normalize_text(value) for value in previous_deployments) or update_is_newer_than_history() or explicit_change_evidence(current_deployment))
    ):
        return "新增实验", (current_metric or current_deployment)[:100]
    adoption = _paper_fact_value(item, "adoption_or_citation") or _paper_fact_value(item, "deployment_context")
    previous_adoptions = [
        _paper_fact_value(previous, "adoption_or_citation") or _paper_fact_value(previous, "deployment_context")
        for previous in previous_items
    ]
    if (
        adoption
        and substantively_new(adoption, previous_adoptions)
        and re.search(r"采用|集成|引用|official|product|产品", adoption, re.IGNORECASE)
        and (any(normalize_text(value) for value in previous_adoptions) or update_is_newer_than_history() or explicit_change_evidence(adoption))
    ):
        return "重要进展", adoption[:100]
    return "", ""


def filter_recently_sent_papers(
    papers: List[Dict[str, Any]],
    history_items: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    history_by_key: Dict[str, List[Dict[str, Any]]] = {}
    for history_item in history_items:
        if history_item.get("content_type") != "paper":
            continue
        for key in paper_history_keys(history_item):
            history_by_key.setdefault(key, []).append(history_item)
    fresh: List[Dict[str, Any]] = []
    filtered_count = 0
    reappeared_update_count = 0
    for item in papers:
        candidate = dict(item)
        matching_history = []
        seen_history_ids = set()
        for key in paper_history_keys(candidate):
            for history_item in history_by_key.get(key, []):
                if not paper_identity_matches(candidate, history_item):
                    continue
                marker = id(history_item)
                if marker not in seen_history_ids:
                    matching_history.append(history_item)
                    seen_history_ids.add(marker)
        if matching_history:
            status_label, reason = detect_reappeared_paper_update(candidate, matching_history)
            if not status_label:
                filtered_count += 1
                continue
            reason = normalize_paper_change_reason(status_label, reason, candidate)
            if not reason:
                filtered_count += 1
                continue
            candidate["paper_status_label"] = status_label
            candidate["paper_change_reason"] = reason
            candidate["is_reappeared_update"] = True
            reappeared_update_count += 1
        else:
            candidate["paper_status_label"] = "今日新增"
            candidate["paper_change_reason"] = ""
            candidate["is_reappeared_update"] = False
        fresh.append(candidate)
    return fresh, {
        "paper_candidate_count_before_freshness": len(papers),
        "paper_fresh_candidate_count": len(fresh),
        "paper_repeat_filtered_count": filtered_count,
        "paper_history_unique_count": len({paper_history_key(item) for item in history_items if paper_history_key(item)}),
        "fresh_paper_count": len(fresh) - reappeared_update_count,
        "reappeared_paper_with_update_count": reappeared_update_count,
    }


def normalize_paper_change_reason(
    status_label: str,
    reason: Any,
    item: Optional[Dict[str, Any]] = None,
) -> str:
    """Keep reappeared-paper explanations public-ready or suppress the update."""
    text = re.sub(r"\s+", " ", str(reason or "")).strip()
    if not text or contains_mojibake(text) or any(marker in text for marker in ("...", "…")):
        return ""
    field_labels = {
        "abstract": "摘要",
        "method": "方法",
        "core_method": "核心方法",
        "metric_result": "实验结果",
        "dataset_or_benchmark": "数据集或基准",
        "baseline": "对照基线",
        "limitation": "局限",
    }
    referenced_fields = [key for key in field_labels if key in text]
    for key in referenced_fields:
        raw_value = _paper_fact_value(item or {}, key)
        if raw_value and (
            contains_mojibake(raw_value)
            or any(marker in raw_value for marker in ("...", "…"))
        ):
            return ""
        text = text.replace(key, field_labels[key])
    if not has_untranslated_prose(text):
        return text

    percent = re.search(r"\b\d+(?:\.\d+)?\s*%", text)
    lowered = text.lower()
    if percent and "success" in lowered:
        setting = "真实机器人实验" if re.search(r"real[- ](?:robot|world)", lowered) else "实验"
        comparison = "，并优于对照基线" if re.search(r"outperform|improv|better|higher", lowered) else ""
        return f"新增{setting}，成功率达到 {percent.group(0)}{comparison}。"
    if str(status_label or "") == "版本更新" and re.search(r"abstract|method|experiment|result", lowered):
        return "论文发布了新版本，并更新了摘要、方法或实验内容。"
    return ""


def select_papers_by_domain_quota(
    papers: List[Dict[str, Any]],
    quotas: Optional[Dict[str, Any]],
    total_limit: int,
    minimum_count: int = 0,
) -> List[Dict[str, Any]]:
    ranked = sorted(
        [dict(item) for item in papers],
        key=lambda item: (
            2 if v10_paper_has_editorial_summary(item) else (1 if v10_paper_has_display_content(item) else 0),
            float(item.get("selection_score", item.get("score", 0)) or 0),
            evidence_quality_value(item),
            information_density_value(item),
            parse_datetime(item.get("publish_date", "")),
        ),
        reverse=True,
    )
    quotas = dict(quotas or {})
    if not quotas:
        return ranked[:total_limit]

    def quota_domain(item: Dict[str, Any]) -> str:
        return paper_quota_domain(item, quotas)

    normalized = {
        str(domain): {
            "min": max(0, int((values or {}).get("min", 0) or 0)),
            "max": max(0, int((values or {}).get("max", total_limit) or total_limit)),
        }
        for domain, values in quotas.items()
    }
    selected: List[Dict[str, Any]] = []
    counts: Counter[str] = Counter()

    def append_from(domain: str, limit: int) -> None:
        for item in ranked:
            if len(selected) >= total_limit or counts[domain] >= limit:
                break
            if any(paper_identity_matches(item, existing) for existing in selected) or quota_domain(item) != domain:
                continue
            item["paper_domain_key"] = domain
            item["domain_key"] = domain if domain != "other" else "products_business"
            selected.append(item)
            counts[domain] += 1

    for domain, limits in normalized.items():
        append_from(domain, min(limits["min"], limits["max"]))
    for domain, limits in normalized.items():
        append_from(domain, limits["max"])
    minimum_count = min(total_limit, max(0, int(minimum_count or 0)))
    if len(selected) < minimum_count:
        for item in ranked:
            if len(selected) >= minimum_count:
                break
            if any(paper_identity_matches(item, existing) for existing in selected):
                continue
            domain = quota_domain(item)
            item["paper_domain_key"] = domain
            item["domain_key"] = domain if domain != "other" else "products_business"
            selected.append(item)
            counts[domain] += 1
    return selected[:total_limit]


def select_papers_with_strict_freshness(
    papers: List[Dict[str, Any]],
    quotas: Optional[Dict[str, Any]],
    total_limit: int,
    overlap_max: float = 0.10,
    minimum_count: int = 0,
) -> Tuple[List[Dict[str, Any]], int]:
    def rank(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return sorted(
            [dict(item) for item in items],
            key=lambda item: (
                2 if v10_paper_has_editorial_summary(item) else (1 if v10_paper_has_display_content(item) else 0),
                float(item.get("selection_score", item.get("score", 0)) or 0),
                evidence_quality_value(item),
                information_density_value(item),
                parse_datetime(item.get("publish_date", "")),
            ),
            reverse=True,
        )

    fresh = rank([item for item in papers if not item.get("is_reappeared_update")])
    updates = rank([item for item in papers if item.get("is_reappeared_update")])
    selected_fresh = select_papers_by_domain_quota(
        fresh,
        quotas,
        total_limit,
        minimum_count=minimum_count,
    )
    if not updates or not selected_fresh:
        return selected_fresh, len(updates)

    for update_limit in range(len(updates), -1, -1):
        kept_updates = updates[:update_limit]
        selected = rank(selected_fresh + kept_updates)
        if len(kept_updates) / len(selected) < overlap_max:
            return selected, len(updates) - len(kept_updates)
    return selected_fresh, len(updates)


def enforce_final_visible_paper_overlap(
    layers: Dict[str, List[Dict[str, Any]]],
    overlap_max: float = 0.10,
) -> Tuple[Dict[str, List[Dict[str, Any]]], int]:
    updated = {key: [dict(item) for item in value] for key, value in layers.items()}

    def visible_papers() -> List[Dict[str, Any]]:
        return [
            item
            for item in flatten_report_layers(updated)
            if item.get("content_type") == "paper"
        ]

    removed_count = 0
    while True:
        papers = visible_papers()
        reappeared = [item for item in papers if item.get("is_reappeared_update")]
        if not papers or not reappeared or len(reappeared) / len(papers) < overlap_max:
            break
        removable = reappeared[-1]
        removed = False
        for section in reversed(REPORT_SECTION_ORDER):
            retained = []
            for item in updated.get(section, []):
                if not removed and item is removable:
                    removed = True
                    removed_count += 1
                    continue
                retained.append(item)
            updated[section] = retained
            if removed:
                break
        if not removed:
            break

    rank = 1
    for section in REPORT_SECTION_ORDER:
        for item in updated.get(section, []):
            item["report_section"] = section
            item["report_rank"] = rank
            rank += 1
    return updated, removed_count


def finalize_paper_freshness_metrics(
    selected_papers: List[Dict[str, Any]],
    history_items: List[Dict[str, Any]],
    base_metrics: Optional[Dict[str, Any]] = None,
    overlap_max: float = 0.10,
) -> Dict[str, Any]:
    metrics = dict(base_metrics or {})
    within_report_duplicate_count = 0
    for index, item in enumerate(selected_papers):
        if any(paper_identity_matches(item, previous) for previous in selected_papers[:index]):
            within_report_duplicate_count += 1
    adjacent_report_id = ""
    adjacent_items: List[Dict[str, Any]] = []
    for item in history_items:
        if item.get("content_type") != "paper":
            continue
        report_id = str(item.get("_history_report_id") or "")
        if report_id and not adjacent_report_id:
            adjacent_report_id = report_id
        if report_id == adjacent_report_id:
            adjacent_items.append(item)
    overlap_count = sum(
        1
        for item in selected_papers
        if any(paper_identity_matches(item, previous) for previous in adjacent_items)
    )
    overlap_rate_raw = overlap_count / max(1, len(selected_papers))
    overlap_rate = round(overlap_rate_raw, 3)
    reappeared_count = sum(1 for item in selected_papers if item.get("is_reappeared_update"))
    metrics.update({
        "adjacent_report_id": adjacent_report_id,
        "adjacent_report_paper_overlap_count": overlap_count,
        "adjacent_report_paper_overlap_rate": overlap_rate,
        "paper_within_report_duplicate_count": within_report_duplicate_count,
        "fresh_paper_count": len(selected_papers) - reappeared_count,
        "reappeared_paper_with_update_count": reappeared_count,
        "paper_freshness_status": (
            "passed"
            if overlap_rate_raw < overlap_max and within_report_duplicate_count == 0
            else "failed"
        ),
    })
    return metrics


def scan_v11_delivery_volume_fidelity(
    delivery_volumes: List[Dict[str, Any]],
    quality_config: Dict[str, Any],
    report_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Verify approved copy inside every HTML volume that will actually be sent."""
    missing_counts: List[int] = []
    missing_examples: List[Dict[str, Any]] = []
    key_number_missing_counts: List[int] = []
    key_number_missing_examples: List[Dict[str, Any]] = []
    for volume_index, volume in enumerate(delivery_volumes, start=1):
        volume_html = str(volume.get("html") or "")
        volume_scan_config = dict(report_config)
        volume_scan_config.update(
            {
                "min_visible_news_count": 0,
                "min_visible_technical_count": 0,
                "min_visible_paper_count": 0,
                "paper_technical_intro_min_count": 0,
                "total_visible_chars_min": 0,
                "total_visible_chars_max": max(100000, len(volume_html)),
            }
        )
        volume_metrics = scan_final_html_quality(
            volume_html,
            dict(volume.get("layers") or {}),
            quality_config=quality_config,
            report_config=volume_scan_config,
        )
        missing_count = int(
            volume_metrics.get("v11_content_fidelity_missing_count", 0) or 0
        )
        missing_counts.append(missing_count)
        for example in volume_metrics.get("v11_content_fidelity_missing_examples", []) or []:
            missing_examples.append({"volume": volume_index, **dict(example)})
        key_number_missing_count = int(
            volume_metrics.get("v11_key_number_fidelity_missing_count", 0) or 0
        )
        key_number_missing_counts.append(key_number_missing_count)
        for example in volume_metrics.get("v11_key_number_fidelity_missing_examples", []) or []:
            key_number_missing_examples.append(
                {"volume": volume_index, **dict(example)}
            )
    return {
        "email_delivery_volume_content_fidelity_missing_counts": missing_counts,
        "email_delivery_volume_content_fidelity_missing_count": sum(missing_counts),
        "email_delivery_volume_content_fidelity_missing_examples": missing_examples[:5],
        "email_delivery_volume_key_number_fidelity_missing_counts": (
            key_number_missing_counts
        ),
        "email_delivery_volume_key_number_fidelity_missing_count": sum(
            key_number_missing_counts
        ),
        "email_delivery_volume_key_number_fidelity_missing_examples": (
            key_number_missing_examples[:5]
        ),
    }


def evaluate_paper_domain_quotas(
    selected_papers: List[Dict[str, Any]],
    quotas: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    normalized_quotas = {
        str(domain): {
            "min": max(0, int((limits or {}).get("min", 0) or 0)),
            "max": max(0, int((limits or {}).get("max", len(selected_papers)) or len(selected_papers))),
        }
        for domain, limits in dict(quotas or {}).items()
    }
    fresh_papers = [item for item in selected_papers if not item.get("is_reappeared_update")]
    counts = Counter(
        paper_quota_domain(item, normalized_quotas)
        for item in fresh_papers
    )
    exceeded = {
        domain: {"count": int(counts.get(domain, 0)), "max": limits["max"]}
        for domain, limits in normalized_quotas.items()
        if int(counts.get(domain, 0)) > limits["max"]
    }
    underfilled = {
        domain: {"count": int(counts.get(domain, 0)), "min": limits["min"]}
        for domain, limits in normalized_quotas.items()
        if int(counts.get(domain, 0)) < limits["min"]
    }
    return {
        "paper_domain_counts": dict(counts),
        "paper_domain_fresh_count": len(fresh_papers),
        "paper_domain_quota_status": "failed" if exceeded else "passed",
        "paper_domain_quota_exceeded": exceeded,
        "paper_domain_quota_underfilled": underfilled,
    }


def should_block_report_send(
    quality_gate_config: Dict[str, Any],
    quality_status: str,
    email_mode: str,
    quality_diagnostics: Optional[Dict[str, Any]] = None,
) -> bool:
    if not quality_gate_config.get("block_send_on_final_failure", False):
        return False
    if str(email_mode or "").strip().lower() in {"dry-run", "dry_run", "skip", "disabled"}:
        return False
    blocking_reasons = (
        report_send_blocking_reasons(quality_gate_config, quality_diagnostics)
        if quality_diagnostics
        else []
    )
    if blocking_reasons:
        return True
    if str(quality_status or "") == "passed":
        return False
    if not quality_gate_config.get("allow_degraded_send_on_soft_failure", False):
        return True
    if not quality_diagnostics:
        return True
    return False


def report_send_blocking_reasons(
    quality_gate_config: Dict[str, Any],
    quality_diagnostics: Dict[str, Any],
) -> List[str]:
    diagnostics = dict(quality_diagnostics or {})
    quality = diagnostics.get("quality_gate")
    if isinstance(quality, dict):
        diagnostics = quality

    count_thresholds = {
        "final_html_bad_title_count": int(quality_gate_config.get("final_html_bad_title_count", 0) or 0),
        "untranslated_fact_count": int(quality_gate_config.get("untranslated_fact_count", 0) or 0),
        "exact_duplicate_sentence_count": int(quality_gate_config.get("exact_duplicate_sentence_count", 1) or 1),
        "paper_mechanism_missing_count": int(quality_gate_config.get("paper_mechanism_missing_count", 0) or 0),
        "paper_result_context_missing_count": int(quality_gate_config.get("paper_result_context_missing_count", 0) or 0),
        "appendix_body_overlap_count": int(quality_gate_config.get("appendix_body_overlap_count", 0) or 0),
        "truncated_focus_text_count": int(quality_gate_config.get("truncated_focus_text_count", 0) or 0),
        "paper_intro_length_fail_count": int(quality_gate_config.get("paper_intro_length_fail_count", 0) or 0),
        "paper_plain_summary_fail_count": 0,
        "paper_technical_intro_fail_count": 0,
        "paper_technical_intro_missing_count": 0,
        "title_only_item_count": int(quality_gate_config.get("title_only_item_count", 0) or 0),
        "low_value_module_count": int(quality_gate_config.get("low_value_module_count", 0) or 0),
        "generic_phrase_count": int(quality_gate_config.get("generic_phrase_count", 0) or 0),
        "low_focus_quality_count": 0,
        "bad_title_count": 0,
        "suspicious_claim_count": int(quality_gate_config.get("unsupported_claim_count", 0) or 0),
        "unsupported_claim_count": int(quality_gate_config.get("unsupported_claim_count", 0) or 0),
        "mixed_language_title_count": 0,
        "field_label_leak_count": 0,
        "low_info_expanded_count": 0,
        "mojibake_suspect_count": 0,
    }
    reasons = [
        key
        for key, limit in count_thresholds.items()
        if int(diagnostics.get(key, 0) or 0) > limit
    ]
    if int(diagnostics.get("paper_within_report_duplicate_count", 0) or 0) > 0:
        reasons.append("paper_within_report_duplicate_count")
    if float(diagnostics.get("adjacent_report_paper_overlap_rate", 0.0) or 0.0) > float(
        quality_gate_config.get("adjacent_report_paper_overlap_max", 0.10) or 0.10
    ):
        reasons.append("adjacent_report_paper_overlap_rate")
    if str(diagnostics.get("paper_freshness_status") or "passed") == "failed":
        reasons.append("paper_freshness_status")
    if str(diagnostics.get("paper_domain_quota_status") or "passed") == "failed":
        reasons.append("paper_domain_quota_status")
    if bool(quality_gate_config.get("hard_min_visible_paper_count", False)) and int(
        diagnostics.get("visible_paper_count", 0) or 0
    ) < int(quality_gate_config.get("min_visible_paper_count", 15) or 15):
        reasons.append("visible_paper_count")
    if "visible_news_count" in diagnostics and int(diagnostics.get("visible_news_count", 0) or 0) < int(
        quality_gate_config.get("min_visible_news_count", 20) or 20
    ):
        reasons.append("visible_news_count")
    if "visible_technical_count" in diagnostics and int(diagnostics.get("visible_technical_count", 0) or 0) < int(
        quality_gate_config.get("min_visible_technical_count", 20) or 20
    ):
        reasons.append("visible_technical_count")
    if "technical_primary_source_ratio" in diagnostics and float(
        diagnostics.get("technical_primary_source_ratio", 0.0) or 0.0
    ) < float(
        quality_gate_config.get("technical_primary_source_ratio_min", 0.80) or 0.80
    ):
        reasons.append("technical_primary_source_ratio")
    if int(diagnostics.get("cross_section_duplicate_count", 0) or 0) > 0:
        reasons.append("cross_section_duplicate_count")
    if int(diagnostics.get("cross_section_event_duplicate_count", 0) or 0) > 0:
        reasons.append("cross_section_event_duplicate_count")
    if dict(diagnostics.get("v11_supplemental_limit_exceeded") or {}):
        reasons.append("v11_supplemental_limit_exceeded")
    if int(diagnostics.get("v11_supplemental_age_violation_count", 0) or 0) > 0:
        reasons.append("v11_supplemental_age_violation_count")
    for key in (
        "body_under_min_count",
        "publish_date_missing_count",
        "source_evidence_missing_count",
        "claim_type_missing_count",
        "paper_full_text_missing_count",
        "analysis_version_mismatch_count",
        "v11_external_item_count",
        "v11_content_fidelity_missing_count",
        "email_delivery_volume_content_fidelity_missing_count",
        "v11_key_number_fidelity_missing_count",
        "email_delivery_volume_key_number_fidelity_missing_count",
        "v11_editorial_source_hash_missing_count",
        "v11_editorial_source_mismatch_count",
    ):
        if int(diagnostics.get(key, 0) or 0) > 0:
            reasons.append(key)
    if "editorial_decision_count" in diagnostics:
        editorial_decision_count = int(diagnostics.get("editorial_decision_count", 0) or 0)
        editorial_decision_min = int(quality_gate_config.get("editorial_decision_min_count", 5) or 5)
        editorial_decision_max = int(quality_gate_config.get("editorial_decision_max_count", 7) or 7)
        if not editorial_decision_min <= editorial_decision_count <= editorial_decision_max:
            reasons.append("editorial_decision_count")
    for key in (
        "editorial_decision_source_missing_count",
        "editorial_decision_duplicate_source_count",
    ):
        if int(diagnostics.get(key, 0) or 0) > 0:
            reasons.append(key)
    if bool(diagnostics.get("email_clipping_risk", False)):
        reasons.append("email_clipping_risk")
    if (
        "ui_audit_status" in diagnostics
        and str(diagnostics.get("ui_audit_status") or "unknown") != "passed"
    ):
        reasons.append("ui_audit_status")
    if (
        bool(quality_gate_config.get("require_codex_research_inbox", False))
        and str(diagnostics.get("codex_research_inbox_status") or "not_run") != "success"
    ):
        reasons.append("codex_research_inbox_status")
    return sorted(set(reasons))


def should_persist_collector_history(report_only: bool, runtime_profile: str) -> bool:
    return not report_only and str(runtime_profile or "").strip().lower() != "validation_fast"


def effective_fresh_paper_minimum(configured_minimum: int, available_count: int) -> int:
    configured = max(1, int(configured_minimum or 1))
    available = max(0, int(available_count or 0))
    return min(configured, available) if available else 1


def persist_committed_send_slot(
    slot_dir: Path,
    slot_id: str,
    payload: Dict[str, Any],
) -> str:
    normalized_slot_id = str(slot_id or "").strip()
    if not re.fullmatch(r"\d{8}_\d{4}", normalized_slot_id):
        return ""
    resolved_dir = slot_dir if slot_dir.is_absolute() else Path(__file__).resolve().parent / slot_dir
    resolved_dir.mkdir(parents=True, exist_ok=True)
    slot_path = resolved_dir / f"{normalized_slot_id}.json"
    existing: Dict[str, Any] = {}
    if slot_path.exists():
        try:
            existing = dict(json.loads(slot_path.read_text(encoding="utf-8-sig")) or {})
        except (OSError, json.JSONDecodeError, TypeError):
            existing = {}
    committed = {
        **existing,
        **payload,
        "slot_id": normalized_slot_id,
        "status": "sent",
        "finished_at": str(payload.get("committed_at") or datetime.now().isoformat(timespec="seconds")),
    }
    temporary_path = slot_path.with_suffix(f"{slot_path.suffix}.{os.getpid()}.tmp")
    temporary_path.write_text(json.dumps(committed, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    temporary_path.replace(slot_path)
    return slot_path.as_posix()


def record_email_commit(
    db: Database,
    *,
    report_id: str,
    run_id: str,
    subject: str,
    html_report_path: str,
    markdown_report_path: str,
    quality_status: str,
    send_slot_id: str = "",
    send_slot_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    payload = {
        "run_id": run_id,
        "report_id": report_id,
        "email_subject": subject,
        "html_report_path": html_report_path,
        "markdown_report_path": markdown_report_path,
        "quality_status": quality_status,
        "committed_at": datetime.now().isoformat(timespec="seconds"),
    }
    persistence_errors: List[str] = []
    if send_slot_dir is not None:
        try:
            payload["send_slot_path"] = persist_committed_send_slot(send_slot_dir, send_slot_id, payload)
        except Exception as exc:
            payload["send_slot_path"] = ""
            persistence_errors.append(f"send_slot:{exc}")
    try:
        db.update_report_delivery_status(report_id, "sent", payload["committed_at"])
    except Exception as exc:
        persistence_errors.append(f"report_run:{exc}")
    if persistence_errors:
        payload["persistence_errors"] = persistence_errors
    return payload


def apply_preference_scores(items: List[Dict[str, Any]], preference_config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    scored: List[Dict[str, Any]] = []
    for item in items:
        candidate = dict(item)
        candidate["preference_score"] = score_preference_boost(candidate, preference_config)
        evidence_quality = evidence_quality_value(candidate)
        information_density = information_density_value(candidate)
        candidate["evidence_quality"] = evidence_quality
        candidate["information_density"] = information_density
        candidate["selection_score"] = (
            float(candidate.get("score", 0) or 0)
            + float(candidate.get("preference_score", 0) or 0)
            + evidence_quality * 0.5
            + information_density * 0.7
            - (0.6 if evidence_quality < 0.35 else 0.0)
            - (0.5 if information_density < 0.35 else 0.0)
        )
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
    arxiv_failures_now = [item for item in arxiv_runs if item.get("status") in {"error", "timeout"}]
    if arxiv_runs and len(arxiv_failures_now) >= arxiv_threshold:
        recent_arxiv_runs = db.get_recent_collector_runs("ArxivCollector[", limit=max(len(arxiv_runs) * 2, 6))
        consecutive_failed = 0
        for row in recent_arxiv_runs:
            if row.get("status") in {"success", "empty"}:
                break
            consecutive_failed += 1
        if consecutive_failed >= arxiv_threshold:
            issues.append(
                f"Arxiv 采集已连续失败 {consecutive_failed} 次；"
                "本轮如继续生成，只使用近期已入库但尚未发送的新论文候选，不使用旧论文补位。"
            )

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
    recovery_interval_hours = int(degradation_config.get("recovery_interval_hours", 12))
    recent_runs = db.get_recent_collector_runs(f"RSSCollector[{feed_name}]", limit=lookback_runs)
    actual_runs = [row for row in recent_runs if row.get("status") != "skipped"]
    if not actual_runs:
        return False

    consecutive_failures = 0
    for row in actual_runs:
        if row.get("status") == "success":
            break
        consecutive_failures += 1
    if consecutive_failures < failure_threshold:
        return False

    if recovery_interval_hours <= 0:
        return True

    latest_actual_run = actual_runs[0]
    created_at = str(latest_actual_run.get("created_at", "") or "").strip()
    try:
        latest_seen = datetime.fromisoformat(created_at)
    except ValueError:
        try:
            latest_seen = datetime.strptime(created_at, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return True
    current_utc = datetime.now(timezone.utc).replace(tzinfo=None)
    return current_utc - latest_seen < timedelta(hours=recovery_interval_hours)


def should_skip_empty_collector(
    db: Database,
    label: str,
    degradation_config: Optional[Dict[str, Any]] = None,
) -> bool:
    degradation_config = degradation_config or {}
    if not degradation_config.get("enabled", False):
        return False
    threshold = max(1, int(degradation_config.get("empty_success_threshold", 5) or 5))
    lookback_runs = max(threshold, int(degradation_config.get("lookback_runs", threshold) or threshold))
    recovery_interval_hours = max(0, int(degradation_config.get("recovery_interval_hours", 24) or 24))
    recent_runs = [
        row
        for row in db.get_recent_collector_runs(label, limit=lookback_runs)
        if row.get("status") != "skipped"
    ]
    if len(recent_runs) < threshold:
        return False
    consecutive_empty = 0
    for row in recent_runs:
        if row.get("status") == "success" and int(row.get("collected_count", 0) or 0) == 0:
            consecutive_empty += 1
            continue
        break
    if consecutive_empty < threshold:
        return False
    if recovery_interval_hours <= 0:
        return True
    created_at = str(recent_runs[0].get("created_at", "") or "").strip()
    try:
        latest_seen = datetime.fromisoformat(created_at)
    except ValueError:
        return True
    current_utc = datetime.now(timezone.utc).replace(tzinfo=None)
    return current_utc - latest_seen < timedelta(hours=recovery_interval_hours)


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
    arxiv_failures_now = [item for item in arxiv_runs if item.get("status") in {"error", "timeout"}]
    if arxiv_runs and len(arxiv_failures_now) >= arxiv_threshold:
        recent_arxiv_runs = db.get_recent_collector_runs("ArxivCollector[", limit=max(len(arxiv_runs) * 2, 6))
        consecutive_failed = 0
        for row in recent_arxiv_runs:
            if row.get("status") in {"success", "empty"}:
                break
            consecutive_failed += 1
        if consecutive_failed >= arxiv_threshold:
            issues.append(
                f"Arxiv 采集已连续失败 {consecutive_failed} 次；"
                "本轮如继续生成，只使用近期已入库但尚未发送的新论文候选，不使用旧论文补位。"
            )

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
        trusted_curated_title = (
            str(candidate.get("model_used") or "") == "codex-automation"
            and bool(title)
            and not title_looks_bad(candidate)
            and not has_untranslated_prose(title)
            and not any(marker in title for marker in ("…", "..."))
        )
        needs_rewrite = bool(
            not trusted_curated_title
            and
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
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is not None:
            parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
        return parsed
    except ValueError:
        return datetime.min


def normalized_event_title(item: Dict[str, Any]) -> str:
    facts = item.get("facts_cn") or item.get("facts") or {}
    if isinstance(facts, dict):
        fact_parts = []
        for key in ("who", "action", "target"):
            value = str(facts.get(key) or "").strip()
            if value.lower() not in {
                "",
                "unknown",
                "this paper",
                "researchers",
                "研究者",
                "研究团队",
                "相关机构",
            }:
                fact_parts.append(value)
        if len(fact_parts) >= 2:
            return re.sub(r"\W+", "", normalize_text(*fact_parts))[:120]
    title = normalize_text(
        item.get("editorial_title")
        or item.get("title_cn")
        or item.get("title")
        or ""
    )
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
    llm_provider = str((config.get("llm") or {}).get("provider") or "").strip().lower()
    codex_research_mode = llm_provider == "codex_automation"
    runtime_profile = str(os.getenv("WEB_AGENT_RUN_PROFILE", "") or "").strip()
    report_only = str(os.getenv("WEB_AGENT_REPORT_ONLY", "") or "").strip().lower() in {"1", "true", "yes", "on"}
    config = apply_runtime_profile(config, runtime_profile)
    config = apply_environment_path_overrides(config)
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
        "delivery_verification": {},
        "source_weight_adjustments": {},
        "runtime_profile": runtime_profile or "default",
        "report_only": report_only,
    }

    database_path = resolve_database_path(config, Path(__file__).resolve().parent)
    Path(database_path).parent.mkdir(parents=True, exist_ok=True)
    db = Database(database_path)
    print(f"Database initialized at {database_path}")
    scheduler_config = dict(config.get("scheduler") or {})
    sent_backfill_count = backfill_sent_report_delivery_status(
        db,
        Path(str(scheduler_config.get("send_slot_dir", "logs/send_slots"))),
    )
    if sent_backfill_count:
        print(f"Backfilled delivery_status=sent for {sent_backfill_count} confirmed report run(s).")

    print("\n=== Step 1: Collecting Data ===")
    collectors = []
    collector_runs: List[Dict[str, Any]] = []
    arxiv_config = config["sources"].get("arxiv", {})
    if arxiv_config.get("enabled", False) and not report_only and not codex_research_mode:
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
    if rss_config.get("enabled", False) and not report_only and not codex_research_mode:
        for feed in rss_config.get("feeds", []):
            if should_skip_rss_feed(db, str(feed.get("name", "feed")), rss_config.get("degradation", {})):
                feed_name = str(feed.get("name", "feed"))
                print(f"Skipping degraded RSS feed: {feed_name}")
                collector_runs.append(
                    {
                        "label": f"RSSCollector[{feed_name}]",
                        "status": "skipped",
                        "inserted_count": 0,
                        "collected_count": 0,
                        "duration_seconds": 0,
                        "error": "degraded feed cooldown active; recovery probe will retry after cooldown",
                    }
                )
                continue
            collector = RSSCollector(feeds=[feed], days_back=int(rss_config.get("days_back", 2)))
            collector.label = f"RSSCollector[{feed.get('name', 'feed')}]"
            collectors.append(collector)
    huggingface_config = config["sources"].get("huggingface", {})
    if huggingface_config.get("enabled", False) and not report_only and not codex_research_mode:
        collectors.append(HuggingFaceCollector())
    codex_research_config = config["sources"].get("codex_research_inbox", {})
    if codex_research_config.get("enabled", False) and not report_only:
        collectors.append(
            build_codex_research_inbox_collector(
                codex_research_config,
                root=Path(__file__).resolve().parent,
            )
        )
    web_search_config = config["sources"].get("web_search", {})
    if web_search_config.get("enabled", False) and not report_only and not codex_research_mode:
        for search in web_search_config.get("searches", []):
            search_name = str(search.get("name", "search"))
            search_label = f"WebSearchCollector[{search_name}]"
            if should_skip_empty_collector(db, search_label, web_search_config.get("degradation", {})):
                print(f"Skipping repeatedly empty web search: {search_name}")
                collector_runs.append(
                    {
                        "label": search_label,
                        "status": "skipped",
                        "inserted_count": 0,
                        "collected_count": 0,
                        "duration_seconds": 0,
                        "error": "five consecutive empty runs; recovery probe will retry after cooldown",
                    }
                )
                continue
            collector = WebSearchCollector(
                searches=[search],
                locale=web_search_config.get("locale", "US:en"),
                days_back=int(web_search_config.get("days_back", 1)),
                fallback_days=web_search_config.get("fallback_days", [1, 2, 3]),
            )
            collector.label = search_label
            collectors.append(collector)

    new_articles_count = 0
    codex_research_urls: List[str] = []
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
                if label == "CodexResearchInboxCollector" and items:
                    history_metrics = evaluate_sent_history_overlap(
                        items,
                        db.get_recent_report_items(
                            days=int(codex_research_config.get("history_dedupe_days", 7) or 7),
                            limit=int(codex_research_config.get("history_dedupe_limit", 5000) or 5000),
                            sent_only=True,
                        ),
                    )
                    collector.fetch_diagnostics.update(history_metrics)
                    if history_metrics.get("production_ready_status") != "passed":
                        collector_error = "sent_history_overlap"
                if label == "CodexResearchInboxCollector":
                    collector.fetch_diagnostics["readiness_summary"] = (
                        build_codex_research_readiness_summary(
                            collector.fetch_diagnostics,
                            codex_research_config,
                        )
                    )
                if collector_error:
                    print(f"Error in {label}: {collector_error}")
                    collector_runs.append(
                        build_collector_failure_record(
                            collector,
                            label,
                            collector_error,
                            (datetime.now() - started_at).total_seconds(),
                        )
                    )
                    continue
                inserted_count = 0
                for item in items:
                    if label == "CodexResearchInboxCollector":
                        codex_research_urls.append(str(item.get("url") or ""))
                    item["run_id"] = run_id
                    item["canonical_url"] = item.get("canonical_url") or item.get("url", "")
                    item["source_tier"] = source_tier_for_item(item, config.get("source_preferences", {}))
                    inserted = db.insert_article(item)
                    if inserted:
                        new_articles_count += 1
                        inserted_count += 1
                    if label == "CodexResearchInboxCollector":
                        db.update_article_source_snapshot(item["url"], item)
                    web_analysis = item.get("_codex_research_analysis")
                    if isinstance(web_analysis, dict):
                        db.update_article_processing(
                            url=item["url"],
                            summary=str(web_analysis.get("summary") or ""),
                            score=float(web_analysis.get("score", 0.0) or 0.0),
                            keywords=list(web_analysis.get("keywords") or []),
                            category=str(web_analysis.get("category") or "Other"),
                            title_cn=str(web_analysis.get("title_cn") or ""),
                            summary_preview=str(web_analysis.get("summary_preview") or ""),
                            why_it_matters=str(web_analysis.get("why_it_matters") or ""),
                            why_now=str(web_analysis.get("why_now") or ""),
                            expected_effect=str(web_analysis.get("expected_effect") or ""),
                            future_impact=str(web_analysis.get("future_impact") or ""),
                            facts=dict(web_analysis.get("facts") or {}),
                            evidence_quality=float(web_analysis.get("evidence_quality", 0.0) or 0.0),
                            information_density=float(web_analysis.get("information_density", 0.0) or 0.0),
                            model_used=str(web_analysis.get("model_used") or "codex-automation"),
                            analysis_version=str(web_analysis.get("analysis_version") or "codex-research-v3"),
                            quality_flags=list(web_analysis.get("quality_flags") or []),
                            rewrite_attempts=0,
                        )
                collector_diagnostics = dict(getattr(collector, "fetch_diagnostics", {}) or {})
                collector_status = "success"
                collector_error = ""
                if label.startswith("ArxivCollector[") and not items:
                    if int(collector_diagnostics.get("page_parse_error_count", 0) or 0):
                        collector_status = "error"
                        collector_error = "arxiv_page_structure_error"
                    elif (
                        int(collector_diagnostics.get("request_error_count", 0) or 0)
                        and not int(collector_diagnostics.get("successful_page_count", 0) or 0)
                    ):
                        collector_status = "error"
                        collector_error = "arxiv_http_fetch_error"
                    else:
                        collector_status = "empty"
                        collector_error = (
                            "arxiv_true_zero_results"
                            if bool(collector_diagnostics.get("true_zero_result", False))
                            else "arxiv_no_matching_results"
                        )
                elif label == "CodexResearchInboxCollector" and not items:
                    collector_status = "error"
                    collector_error = str(collector_diagnostics.get("quality_status") or "codex_research_inbox_empty")
                collector_runs.append(
                    {
                        "label": label,
                        "status": collector_status,
                        "inserted_count": inserted_count,
                        "collected_count": len(items),
                        "duration_seconds": round((datetime.now() - started_at).total_seconds(), 1),
                        "error": collector_error,
                        "diagnostics": collector_diagnostics,
                    }
                )
            except Exception as exc:
                print(f"Error in {label}: {exc}")
                collector_runs.append(
                    build_collector_failure_record(
                        collector,
                        label,
                        exc,
                        (datetime.now() - started_at).total_seconds(),
                    )
                )
    finally:
        socket.setdefaulttimeout(original_socket_timeout)
    validation_profile = str(runtime_profile or "").strip().lower() == "validation_fast"
    if report_only:
        print("Report-only mode: skipped collection and preserved source-health history.")
    elif validation_profile:
        print("Validation profile: collector results were not added to production source-health history.")
    if should_persist_collector_history(report_only, runtime_profile):
        print(f"Collection complete. Added {new_articles_count} new items.")
        db.record_collector_runs(run_id, collector_runs)
    collector_summary = build_collector_summary(collector_runs)
    codex_research_run = next(
        (row for row in collector_runs if str(row.get("label") or "") == "CodexResearchInboxCollector"),
        {},
    )
    codex_research_inbox_status = str(codex_research_run.get("status") or "not_run")
    codex_research_inbox_quality_status = str(
        (codex_research_run.get("diagnostics") or {}).get("quality_status")
        or codex_research_run.get("error")
        or codex_research_inbox_status
    )
    observability_config = dict(config.get("observability", {}) or {})
    if report_only:
        latest_report = db.get_latest_report_run(exclude_validation=True)
        latest_diagnostics = dict(latest_report.get("quality_diagnostics") or {})
        source_health = dict(latest_diagnostics.get("source_health") or {})
    elif validation_profile:
        source_health = {
            "history_limit": 0,
            "source_count": len(collector_runs),
            "risky_source_count": 0,
            "unstable_source_count": 0,
            "risky_rows": [],
            "unstable_rows": [],
            "validation_only": True,
        }
    else:
        source_health = build_source_health_summary(
            collector_runs,
            db,
            history_limit=int(observability_config.get("source_health_history_limit", 160)),
            active_labels={"CodexResearchInboxCollector"} if codex_research_mode else None,
        )
    print("Collector health: " + collector_summary["status_text"])
    run_result["new_articles_count"] = new_articles_count
    run_result["collector_summary"] = collector_summary
    run_result["source_health"] = source_health
    if codex_research_mode and not report_only:
        codex_runs = [item for item in collector_runs if item.get("label") == "CodexResearchInboxCollector"]
        if not codex_runs or not any(int(item.get("collected_count", 0) or 0) > 0 for item in codex_runs):
            reason = str((codex_runs[-1] if codex_runs else {}).get("error") or "missing_codex_research")
            raise RuntimeError(f"Codex research is unavailable: {reason}")

    print("\n=== Step 2: Processing Data ===")
    run_unprocessed = [] if report_only else db.get_unprocessed_articles(run_id=run_id)
    backlog_unprocessed = [] if report_only else [item for item in db.get_unprocessed_articles() if item.get("run_id") != run_id]
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
            facts=result.get("facts", {}),
            evidence_quality=float(result.get("evidence_quality", 0.0) or 0.0),
            information_density=float(result.get("information_density", 0.0) or 0.0),
            model_used=result.get("model_used", ""),
            analysis_version="v2",
            quality_flags=item_quality_flags(result),
            rewrite_attempts=int(article.get("rewrite_attempts", 0) or 0),
        )
        runtime_results[article["url"]] = result
        processed_count += 1
        print(f"  -> Score: {result.get('score')} | Category: {safe_console_text(str(result.get('category')))}")
    print(f"Processing complete. Successfully processed {processed_count} items.")
    run_result["processed_count"] = processed_count

    analysis_backfill_candidates = [] if (report_only or codex_research_mode) else [
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
            facts=result.get("facts", article.get("facts", {})),
            evidence_quality=float(result.get("evidence_quality", article.get("evidence_quality", 0.0)) or 0.0),
            information_density=float(result.get("information_density", article.get("information_density", 0.0)) or 0.0),
            model_used=result.get("model_used", article.get("model_used", "")),
            analysis_version="v2",
            quality_flags=item_quality_flags(result),
            rewrite_attempts=int(article.get("rewrite_attempts", 0) or 0),
        )
        runtime_results[article["url"]] = result
        analysis_backfill_count += 1
        print("  -> Detailed analysis refreshed")
    if analysis_backfill_candidates:
        print(f"Detailed analysis backfill complete. Updated {analysis_backfill_count} items.")
    run_result["analysis_backfill_count"] = analysis_backfill_count

    print("\n=== Step 3: Generating Report ===")
    report_items = db.get_articles_for_run(run_id=run_id, processed_only=True)
    current_research_items: List[Dict[str, Any]] = []
    if codex_research_urls:
        current_research_items = db.get_articles_by_urls(codex_research_urls)
        report_items = merge_unique_articles(current_research_items, report_items)
        print(f"Added {len(current_research_items)} verified Codex research item(s) to this report batch.")
    if str(runtime_limits.get("profile") or "") == "validation_fast" and runtime_results:
        validation_batch = db.get_articles_by_urls(list(runtime_results))
        report_items = merge_unique_articles(report_items, validation_batch)
        print(f"Validation profile added {len(validation_batch)} item(s) from this processing batch.")
    source_preferences = config.get("source_preferences", {})
    source_preferences, source_weight_adjustments = apply_source_health_adjustments(
        source_preferences,
        source_health,
        observability_config.get("source_health_weighting", {}),
    )
    run_result["source_weight_adjustments"] = source_weight_adjustments
    preference_config = config.get("preferences", {})
    feedback_config = dict(config.get("feedback", {}) or {})
    feedback_config["runtime_healthy"] = feedback_server_is_healthy(feedback_config)
    run_result["feedback_server_status"] = "healthy" if feedback_config["runtime_healthy"] else "unhealthy"
    report_config = dict(config.get("report", {}) or {})
    llm_config = dict(config.get("llm", {}) or {})
    report_config["model_path_label"] = (
        "Codex 网页研究 + 本地质量整理"
        if codex_research_mode
        else f"{llm_config.get('model', 'GPT')}（OpenAI Responses API + Web Search）"
    )
    quality_gate_config = dict(config.get("quality_gate", {}) or {})
    quality_gate_config.setdefault(
        "paper_technical_intro_min_count",
        int(report_config.get("paper_technical_intro_min_count", 12) or 12),
    )
    quality_gate_config.setdefault(
        "physical_ai_featured_min_count",
        int(report_config.get("physical_ai_featured_min_count", report_config.get("physical_ai_min_items", 4)) or 4),
    )
    alert_config = dict(config.get("alerts", {}) or {})
    trend_config = config.get("trends", {})
    archive_config = config.get("archive", {})
    paper_topic_limits = arxiv_config.get("topic_limits", {})
    default_paper_limit = sum(int(limit) for limit in paper_topic_limits.values()) if paper_topic_limits else 10
    paper_limit = int(report_config.get("paper_limit", default_paper_limit))
    paper_cache_hours = int(report_config.get("paper_enrichment_cache_hours", 168))
    web_limit = int(report_config.get("web_limit", 20))
    min_web_items = int(report_config.get("min_web_items", 20))
    paper_backfill_ladder = parse_hour_ladder(
        report_config.get("paper_backfill_hours_ladder", report_config.get("paper_backfill_hours", 24 * 7)),
        [24 * 7],
    )
    web_backfill_ladder = parse_hour_ladder(
        report_config.get("web_backfill_hours_ladder", report_config.get("web_backfill_hours", 24)),
        [24, 48, 72],
    )

    current_papers = [item for item in report_items if item.get("content_type") == "paper"]
    current_updates = [item for item in report_items if item.get("content_type") != "paper"]
    is_v10_design = is_learning_digest_design(report_config)
    strict_v11_mode = str(report_config.get("product_mode") or "") == "intelligence_v11_editorial_library"
    paper_history: List[Dict[str, Any]] = []
    cooldown_history: List[Dict[str, Any]] = []
    adjacent_history: List[Dict[str, Any]] = []
    if is_v10_design and bool(report_config.get("paper_freshness_enabled", True)):
        cooldown_history = db.get_recent_report_items(
            days=int(report_config.get("paper_repeat_cooldown_days", 7) or 7),
            limit=int(report_config.get("paper_repeat_history_limit", 1600) or 1600),
            sent_only=True,
            content_type="paper",
        )
        adjacent_history = db.get_latest_sent_report_items(content_type="paper")
        seen_history_rows: set[Tuple[str, Any]] = set()
        for history_item in adjacent_history + cooldown_history:
            marker = (
                str(history_item.get("_history_report_id", "") or ""),
                history_item.get("id"),
            )
            if marker in seen_history_rows:
                continue
            seen_history_rows.add(marker)
            paper_history.append(history_item)
    paper_backfill_hours_used: List[int] = []
    web_backfill_hours_used: List[int] = []
    if not strict_v11_mode and paper_backfill_ladder and (len(current_papers) < paper_limit or is_v10_design):
        needed_papers = max(0, paper_limit - len(current_papers))
        paper_candidate_ladder = (
            parse_hour_ladder(
                report_config.get(
                    "paper_fresh_candidate_hours_ladder",
                    report_config.get("paper_fresh_candidate_hours", 72),
                ),
                [72, 168, 336, 720],
            )
            if is_v10_design else paper_backfill_ladder
        )
        for hours in paper_candidate_ladder:
            paper_backfill_hours_used.append(int(hours))
            recent_papers = db.get_recent_processed_articles(
                hours=hours,
                content_type="paper",
                limit=max(
                    int(report_config.get("paper_candidate_query_limit", 1000) or 1000),
                    paper_limit * 12,
                    needed_papers * 12,
                    240,
                ),
            )
            report_items = merge_unique_articles(report_items, recent_papers)
            print(
                f"Added unsent paper candidates discovered in the last {hours} hours; current run had {len(current_papers)} papers."
            )
            candidate_papers = [
                item
                for item in report_items
                if item.get("content_type") == "paper"
                and (not is_v10_design or v10_paper_is_ai_relevant(item))
            ]
            eligible_papers = candidate_papers
            if is_v10_design and paper_history:
                eligible_papers, _ = filter_recently_sent_papers(candidate_papers, paper_history)
            if is_v10_design:
                minimum_visible_papers = int(report_config.get("min_visible_paper_count", 15) or 15)
                preview_papers, _ = select_papers_with_strict_freshness(
                    eligible_papers,
                    report_config.get("paper_domain_quotas", {}),
                    min(paper_limit, int(report_config.get("paper_target_max_count", 25) or 25)),
                    float(quality_gate_config.get("adjacent_report_paper_overlap_max", 0.10) or 0.10),
                    minimum_count=minimum_visible_papers,
                )
                summarized_count = sum(v10_paper_has_editorial_summary(item) for item in preview_papers)
                if len(preview_papers) >= minimum_visible_papers and summarized_count >= minimum_visible_papers:
                    break
            elif len(eligible_papers) >= paper_limit:
                break

    min_visible_information_count = max(
        min_web_items,
        int(report_config.get("min_visible_information_count", min_web_items) or min_web_items),
    )

    def v10_eligible_update_count(items: List[Dict[str, Any]]) -> int:
        if not is_v10_design:
            return len([item for item in items if item.get("content_type") != "paper"])
        eligible_urls: set[str] = set()
        for raw_item in items:
            if raw_item.get("content_type") == "paper":
                continue
            candidate = enrich_editorial_fields({**dict(raw_item), "_v8_force_fact_title": True})
            if (
                candidate.get("quality_tier") != "brief"
                and is_focus_quality_item(candidate)
                and v10_has_concrete_news_evidence(candidate)
            ):
                key = str(candidate.get("canonical_url") or candidate.get("url") or candidate.get("title_cn") or "")
                if key:
                    eligible_urls.add(key)
        return len(eligible_urls)

    current_eligible_update_count = v10_eligible_update_count(report_items)
    current_research_section_counts = Counter(
        report_primary_section(item) for item in current_research_items
    )
    v11_research_pool_ready = (
        strict_v11_mode
        and current_research_section_counts.get("news", 0)
        >= int(report_config.get("min_visible_news_count", 20) or 20)
        and current_research_section_counts.get("technical", 0)
        >= int(report_config.get("min_visible_technical_count", 20) or 20)
        and current_research_section_counts.get("paper", 0)
        >= int(report_config.get("min_visible_paper_count", 15) or 15)
    )
    needs_web_backfill = not v11_research_pool_ready and (
        len(current_updates) < min_web_items
        or (is_v10_design and current_eligible_update_count < min_visible_information_count)
    )
    if web_backfill_ladder and needs_web_backfill:
        needed_updates = max(
            min_web_items - len(current_updates),
            min_visible_information_count - current_eligible_update_count if is_v10_design else 0,
        )
        for hours in web_backfill_ladder:
            web_backfill_hours_used.append(int(hours))
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
            deduped_candidate_updates = dedupe_updates(
                prepared_update_candidates,
                llm_processor,
                max_llm_checks=0 if report_only else 12,
            )
            preview_updates = filter_updates_for_report(
                deduped_candidate_updates,
                web_limit,
                min_web_items,
                source_preferences=source_preferences,
                preference_config=preference_config,
                diversify_sources=is_continuous_reader_design(report_config),
            )
            eligible_update_count = v10_eligible_update_count(deduped_candidate_updates)
            enough_candidates = (
                eligible_update_count >= min_visible_information_count
                if is_v10_design
                else len(deduped_candidate_updates) >= min_web_items
            )
            if enough_candidates and v8_backfill_pool_ready(
                preview_updates,
                report_config,
            ):
                break

    prepared_items = prepare_report_items(report_items, llm_processor, runtime_results)
    prepared_items = apply_preference_scores(prepared_items, preference_config)
    prepared_items = apply_feedback_preference_scores(prepared_items, db.get_preference_weights())
    research_order = {
        str(url): index for index, url in enumerate(codex_research_urls)
    }
    if strict_v11_mode and research_order:
        prepared_items = [
            item
            for item in prepared_items
            if str(item.get("canonical_url") or item.get("url") or "") in research_order
            or str(item.get("url") or "") in research_order
        ]
        prepared_items.sort(
            key=lambda item: research_order.get(
                str(item.get("canonical_url") or item.get("url") or ""),
                research_order.get(str(item.get("url") or ""), len(research_order)),
            )
        )
    paper_candidates = [
        item
        for item in prepared_items
        if item.get("content_type") == "paper"
        and (not is_v10_design or v10_paper_is_ai_relevant(item))
    ]
    update_candidates = [item for item in prepared_items if item.get("content_type") != "paper"]
    paper_freshness_metrics = {
        "paper_candidate_count_before_freshness": len(paper_candidates),
        "paper_fresh_candidate_count": len(paper_candidates),
        "paper_repeat_filtered_count": 0,
        "paper_history_unique_count": 0,
    }
    if is_v10_design and bool(report_config.get("paper_freshness_enabled", True)):
        paper_freshness_metrics["adjacent_report_history_count"] = len(adjacent_history)
        paper_freshness_metrics["cooldown_history_count"] = len(cooldown_history)
        paper_candidates, paper_freshness_metrics = filter_recently_sent_papers(
            paper_candidates,
            paper_history,
        )
        paper_freshness_metrics["adjacent_report_history_count"] = len(adjacent_history)
        paper_freshness_metrics["cooldown_history_count"] = len(cooldown_history)
        if paper_freshness_metrics["paper_repeat_filtered_count"]:
            print(
                "Filtered "
                f"{paper_freshness_metrics['paper_repeat_filtered_count']} paper(s) already sent "
                f"within {int(report_config.get('paper_repeat_cooldown_days', 7) or 7)} day(s)."
            )
    if strict_v11_mode:
        papers = list(paper_candidates[:paper_limit])
        overlap_filtered_count = 0
    elif is_v10_design:
        papers, overlap_filtered_count = select_papers_with_strict_freshness(
            paper_candidates,
            report_config.get("paper_domain_quotas", {}),
            min(paper_limit, int(report_config.get("paper_target_max_count", 25) or 25)),
            float(quality_gate_config.get("adjacent_report_paper_overlap_max", 0.10) or 0.10),
            minimum_count=int(report_config.get("min_visible_paper_count", 15) or 15),
        )
        paper_freshness_metrics["paper_repeat_filtered_count"] += overlap_filtered_count
    else:
        papers = limit_papers_by_topic(paper_candidates, paper_topic_limits, paper_limit)

    if (
        papers
        and arxiv_config.get("enabled", False)
        and not strict_v11_mode
        and not report_only
        and not bool(runtime_limits.get("skip_paper_enrichment", False))
    ):
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
            if should_preserve_existing_paper_analysis(paper, result):
                print(
                    "Preserved existing paper analysis after the live LLM path fell back to heuristics: "
                    f"{paper.get('title', '')[:90]}"
                )
                refined_papers.append(paper)
                continue
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
                    facts=result.get("facts", paper.get("facts", {})),
                    evidence_quality=float(result.get("evidence_quality", paper.get("evidence_quality", 0.0)) or 0.0),
                    information_density=float(result.get("information_density", paper.get("information_density", 0.0)) or 0.0),
                    model_used=result.get("model_used", paper.get("model_used", "")),
                    analysis_version="v2",
                    quality_flags=item_quality_flags(result),
                    rewrite_attempts=int(paper.get("rewrite_attempts", 0) or 0),
                )
                runtime_results[paper["url"]] = result
                refined_papers.append(llm_processor.prepare_report_item(paper, result))
            else:
                refined_papers.append(paper)
        papers = apply_preference_scores(refined_papers, preference_config)
        if is_v10_design:
            papers, post_enrichment_filtered_count = select_papers_with_strict_freshness(
                papers,
                report_config.get("paper_domain_quotas", {}),
                min(paper_limit, int(report_config.get("paper_target_max_count", 25) or 25)),
                float(quality_gate_config.get("adjacent_report_paper_overlap_max", 0.10) or 0.10),
                minimum_count=int(report_config.get("min_visible_paper_count", 15) or 15),
            )
            paper_freshness_metrics["paper_repeat_filtered_count"] += post_enrichment_filtered_count
        else:
            papers = limit_papers_by_topic(papers, paper_topic_limits, paper_limit)

    deduped_updates = (
        list(update_candidates)
        if strict_v11_mode
        else dedupe_updates(
            update_candidates,
            llm_processor,
            max_llm_checks=0 if report_only else 12,
        )
    )
    if strict_v11_mode:
        updates = select_v11_update_candidates(deduped_updates, report_config)
    elif is_v10_design:
        concrete_updates: List[Dict[str, Any]] = []
        fallback_updates: List[Dict[str, Any]] = []
        for item in deduped_updates:
            candidate = enrich_editorial_fields({**dict(item), "_v8_force_fact_title": True})
            if (
                candidate.get("quality_tier") != "brief"
                and is_focus_quality_item(candidate)
                and v10_has_concrete_news_evidence(candidate)
            ):
                concrete_updates.append(candidate)
            else:
                fallback_updates.append(item)
        updates = filter_updates_for_report(
            concrete_updates,
            min(web_limit, len(concrete_updates)),
            min(min_web_items, len(concrete_updates)),
            source_preferences=source_preferences,
            preference_config=preference_config,
            diversify_sources=True,
        )
        selected_update_urls = {
            str(item.get("canonical_url") or item.get("url") or "")
            for item in updates
        }
        if len(updates) < web_limit:
            fallback_selection = filter_updates_for_report(
                fallback_updates,
                web_limit - len(updates),
                0,
                source_preferences=source_preferences,
                preference_config=preference_config,
                diversify_sources=True,
            )
            updates.extend(
                item
                for item in fallback_selection
                if str(item.get("canonical_url") or item.get("url") or "") not in selected_update_urls
            )
    else:
        updates = filter_updates_for_report(
            deduped_updates,
            web_limit,
            min_web_items,
            source_preferences=source_preferences,
            preference_config=preference_config,
            diversify_sources=is_continuous_reader_design(report_config),
        )
    if not strict_v11_mode:
        papers = diversify_report_titles(papers, llm_processor)
        updates = diversify_report_titles(updates, llm_processor)
    max_model_path_backfill_items = int(quality_gate_config.get("max_model_path_backfill_items", 4) or 4)
    refreshed_model_path_items = (
        {}
        if strict_v11_mode
        else refresh_missing_model_path_items(
            papers + updates,
            db=db,
            llm_processor=llm_processor,
            max_items=max_model_path_backfill_items,
        )
    )
    if refreshed_model_path_items:
        print(f"Refreshed model path metadata for {len(refreshed_model_path_items)} selected item(s).")
        papers = [refreshed_model_path_items.get(str(item.get("url", "")), item) for item in papers]
        updates = [refreshed_model_path_items.get(str(item.get("url", "")), item) for item in updates]
        papers = apply_preference_scores(papers, preference_config)
        updates = apply_preference_scores(updates, preference_config)
        papers = apply_feedback_preference_scores(papers, db.get_preference_weights())
        updates = apply_feedback_preference_scores(updates, db.get_preference_weights())
        papers = diversify_report_titles(papers, llm_processor)
        updates = diversify_report_titles(updates, llm_processor)
    if is_v10_design and bool(report_config.get("paper_freshness_enabled", True)):
        configured_visible_min = int(report_config.get("min_visible_paper_count", 10))
        configured_featured_min = int(report_config.get("paper_technical_intro_min_count", 10))
        selected_fresh_paper_count = sum(1 for item in papers if not item.get("is_reappeared_update"))
        effective_visible_min = (
            configured_visible_min
            if bool(quality_gate_config.get("hard_min_visible_paper_count", False))
            else effective_fresh_paper_minimum(configured_visible_min, selected_fresh_paper_count)
        )
        effective_featured_min = min(
            max(0, configured_featured_min),
            selected_fresh_paper_count,
            int(report_config.get("paper_featured_limit", 12) or 12),
        )
        target_min = int(report_config.get("paper_target_min_count", 15) or 15)
        report_config["min_visible_paper_count"] = effective_visible_min
        report_config["paper_technical_intro_min_count"] = effective_featured_min
        quality_gate_config["min_visible_paper_count"] = effective_visible_min
        quality_gate_config["paper_technical_intro_min_count"] = effective_featured_min
        alert_config["min_paper_count"] = effective_visible_min
        paper_freshness_metrics.update(
            {
                "configured_min_visible_paper_count": configured_visible_min,
                "effective_min_visible_paper_count": effective_visible_min,
                "effective_featured_paper_min_count": effective_featured_min,
                "configured_target_min_paper_count": target_min,
                "configured_target_max_paper_count": int(report_config.get("paper_target_max_count", 25) or 25),
                "paper_target_underfilled": selected_fresh_paper_count < target_min,
            }
        )
        if selected_fresh_paper_count < target_min:
            print(
                f"Paper freshness note: {selected_fresh_paper_count} genuinely new papers available, below the ideal target {target_min}; "
                "the report will stay short instead of reusing old papers."
            )
    print(f"Dedupe reduced dynamic items from {len([item for item in prepared_items if item.get('content_type') != 'paper'])} to {len(deduped_updates)} candidates.")
    if len(updates) < min_web_items:
        print(f"Warning: only {len(updates)} web updates available after dedupe, below target {min_web_items}.")
    if not papers and not updates:
        print("No processed items available for the report.")
        run_result["quality_diagnostics"] = build_quality_diagnostics(
            current_papers_count=len(current_papers),
            current_updates_count=len(current_updates),
            report_items_count=len(report_items),
            prepared_items_count=len(prepared_items),
            paper_candidate_count=len(paper_candidates),
            update_candidate_count=len(update_candidates),
            deduped_update_count=len(deduped_updates),
            selected_paper_count=0,
            selected_update_count=0,
            paper_limit=paper_limit,
            min_paper_count=int(alert_config.get("min_paper_count", min(10, paper_limit))),
            web_limit=web_limit,
            min_web_items=min_web_items,
            paper_backfill_hours_used=paper_backfill_hours_used,
            web_backfill_hours_used=web_backfill_hours_used,
            collector_summary=collector_summary,
            design_version=str(config.get("report", {}).get("design_version") or ReportGenerator.DESIGN_VERSION),
            source_health=source_health,
            source_weight_adjustments=source_weight_adjustments,
        )
        run_result["quality_diagnostics"]["report_product_mode"] = str(
            report_config.get("product_mode") or ""
        )
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

    file_stamp = datetime.now().strftime("%Y%m%d_%H%M")
    report_id = f"{file_stamp}_{run_id}"
    run_result["report_id"] = report_id
    report_layers = classify_report_layers(
        papers,
        updates,
        report_id=report_id,
        feedback_config=feedback_config,
        source_preferences=source_preferences,
        report_config=report_config,
    )
    title_repair_summary = {"bad_title_repaired_count": 0, "bad_title_unresolved_count": 0, "examples": []}
    report_layers, current_title_repair = repair_bad_titles_in_layers(report_layers)
    title_repair_summary["bad_title_repaired_count"] += current_title_repair["bad_title_repaired_count"]
    title_repair_summary["bad_title_unresolved_count"] = current_title_repair["bad_title_unresolved_count"]
    title_repair_summary["examples"] = (title_repair_summary["examples"] + current_title_repair.get("examples", []))[:10]
    report_layers = apply_v8_reading_budget(report_layers, report_config)
    quality_gate_result = evaluate_report_quality(report_layers, quality_gate_config)
    auto_rewrite_attempted_count = 0
    auto_rewrite_success_count = 0
    if (
        quality_gate_config.get("enabled", True)
        and quality_gate_config.get("auto_rewrite_once", True)
        and not report_only
        and llm_processor.live_generation_available
        and quality_gate_result.get("status") == "failed"
    ):
        failed_items = list(quality_gate_result.get("failed_items") or [])
        max_rewrite_items = int(quality_gate_config.get("max_auto_rewrite_items", 6) or 6)
        rewrite_candidates = [
            item for item in failed_items
            if item.get("url") and int(item.get("rewrite_attempts", 0) or 0) < 1
        ][:max_rewrite_items]
        if rewrite_candidates:
            print(f"Quality gate failed; auto-rewriting {len(rewrite_candidates)} weak item(s) once...")
        refreshed_by_url: Dict[str, Dict[str, Any]] = {}
        for item in rewrite_candidates:
            auto_rewrite_attempted_count += 1
            source_article = db.get_article_by_id(int(item.get("id"))) if item.get("id") else None
            source_article = source_article or item
            result = llm_processor.process_article(source_article)
            if not result:
                continue
            rewrite_attempts = int(source_article.get("rewrite_attempts", 0) or 0) + 1
            db.update_article_processing(
                url=source_article["url"],
                summary=result.get("summary", source_article.get("summary", "")),
                score=float(result.get("score", source_article.get("score", 0)) or 0),
                keywords=result.get("keywords", source_article.get("keywords", [])),
                category=result.get("category", source_article.get("category", "Other")),
                title_cn=result.get("title_cn", source_article.get("title_cn", "")),
                summary_preview=result.get("summary_preview", source_article.get("summary_preview", "")),
                why_it_matters=result.get("why_it_matters", source_article.get("why_it_matters", "")),
                why_now=result.get("why_now", source_article.get("why_now", "")),
                expected_effect=result.get("expected_effect", source_article.get("expected_effect", "")),
                future_impact=result.get("future_impact", source_article.get("future_impact", "")),
                facts=result.get("facts", source_article.get("facts", {})),
                evidence_quality=float(result.get("evidence_quality", source_article.get("evidence_quality", 0.0)) or 0.0),
                information_density=float(result.get("information_density", source_article.get("information_density", 0.0)) or 0.0),
                model_used=result.get("model_used", source_article.get("model_used", "")),
                analysis_version="v2",
                quality_flags=item_quality_flags(result),
                rewrite_attempts=rewrite_attempts,
            )
            refreshed_item = llm_processor.prepare_report_item(source_article, result)
            refreshed_item["rewrite_attempts"] = rewrite_attempts
            refreshed_by_url[str(source_article.get("url", ""))] = refreshed_item
            if is_focus_quality_item(refreshed_item):
                auto_rewrite_success_count += 1
        if refreshed_by_url:
            papers = [refreshed_by_url.get(str(item.get("url", "")), item) for item in papers]
            updates = [refreshed_by_url.get(str(item.get("url", "")), item) for item in updates]
            papers = apply_preference_scores(papers, preference_config)
            updates = apply_preference_scores(updates, preference_config)
            papers = apply_feedback_preference_scores(papers, db.get_preference_weights())
            updates = apply_feedback_preference_scores(updates, db.get_preference_weights())
            papers = diversify_report_titles(papers, llm_processor)
            updates = diversify_report_titles(updates, llm_processor)
            for item in papers + updates:
                item["impact_tag"] = infer_impact_tag(
                    item.get("title_cn") or item.get("title", ""),
                    item.get("summary", ""),
                    item.get("why_now", ""),
                    item.get("expected_effect", ""),
                    item.get("future_impact", ""),
                    item.get("display_topic", ""),
                )
            report_layers = classify_report_layers(
                papers,
                updates,
                report_id=report_id,
                feedback_config=feedback_config,
                source_preferences=source_preferences,
                report_config=report_config,
            )
            report_layers, current_title_repair = repair_bad_titles_in_layers(report_layers)
            title_repair_summary["bad_title_repaired_count"] += current_title_repair["bad_title_repaired_count"]
            title_repair_summary["bad_title_unresolved_count"] = current_title_repair["bad_title_unresolved_count"]
            title_repair_summary["examples"] = (title_repair_summary["examples"] + current_title_repair.get("examples", []))[:10]
            report_layers = apply_v8_reading_budget(report_layers, report_config)
            quality_gate_result = evaluate_report_quality(report_layers, quality_gate_config)

    if quality_gate_result.get("status") == "failed":
        report_layers = downgrade_failed_focus_items(
            report_layers,
            list(quality_gate_result.get("failed_item_urls") or []),
        )
        report_layers = apply_v8_reading_budget(report_layers, report_config)
        quality_gate_result = evaluate_report_quality(report_layers, quality_gate_config)
    continuity_metrics: Dict[str, int] = {}
    continuity_history: List[Dict[str, Any]] = []
    if str(report_config.get("design_version") or "") == "v9-continuous-learning":
        continuity_history = db.get_recent_report_items(
            days=int(report_config.get("continuity_lookback_days", 14) or 14),
            limit=int(report_config.get("continuity_history_limit", 800) or 800),
            exclude_report_id=report_id,
            sent_only=True,
        )
        report_layers, continuity_metrics = apply_v9_continuity(
            report_layers,
            continuity_history,
            suppress_repeated_focus=bool(report_config.get("suppress_repeated_focus", True)) and not report_only,
        )
        quality_gate_result = evaluate_report_quality(report_layers, quality_gate_config)
    run_result["quality_status"] = quality_gate_result.get("status", "unknown")
    run_result["auto_rewrite_attempted_count"] = auto_rewrite_attempted_count
    run_result["auto_rewrite_success_count"] = auto_rewrite_success_count
    run_result["model_path_breakdown"] = model_path_breakdown(flatten_report_layers(report_layers))
    run_result["llm_health"] = llm_processor.health_snapshot()

    print(f"Report contains {len(papers)} papers and {len(updates)} web updates.")
    run_result["paper_count"] = len(papers)
    run_result["fresh_paper_count"] = sum(1 for item in papers if not item.get("is_reappeared_update"))
    run_result["reappeared_paper_with_update_count"] = sum(
        1 for item in papers if item.get("is_reappeared_update")
    )
    run_result["update_count"] = len(updates)
    layered_items_for_quality = flatten_report_layers(report_layers)
    content_quality_counts = build_content_quality_counts(
        selected_items=layered_items_for_quality,
        update_candidates=deduped_updates,
        source_preferences=source_preferences,
    )
    run_result["quality_diagnostics"] = build_quality_diagnostics(
        current_papers_count=len(current_papers),
        current_updates_count=len(current_updates),
        report_items_count=len(report_items),
        prepared_items_count=len(prepared_items),
        paper_candidate_count=len(paper_candidates),
        update_candidate_count=len(update_candidates),
        deduped_update_count=len(deduped_updates),
        selected_paper_count=len(papers),
        selected_update_count=len(updates),
        paper_limit=paper_limit,
        min_paper_count=int(alert_config.get("min_paper_count", min(10, paper_limit))),
        web_limit=web_limit,
        min_web_items=min_web_items,
        paper_backfill_hours_used=paper_backfill_hours_used,
        web_backfill_hours_used=web_backfill_hours_used,
        collector_summary=collector_summary,
        design_version=str(config.get("report", {}).get("design_version") or ReportGenerator.DESIGN_VERSION),
        source_health=source_health,
        source_weight_adjustments=source_weight_adjustments,
        **content_quality_counts,
    )
    run_result["quality_diagnostics"]["report_product_mode"] = str(
        report_config.get("product_mode") or ""
    )
    if strict_v11_mode:
        run_result["quality_diagnostics"]["v11_acceptance_contract_version"] = (
            V11_ACCEPTANCE_CONTRACT_VERSION
        )
    run_result["quality_diagnostics"]["quality_gate"] = {
        key: value
        for key, value in quality_gate_result.items()
        if key != "failed_items"
    }
    run_result["quality_diagnostics"]["quality_gate"].update({
        "codex_research_inbox_status": codex_research_inbox_status,
        "codex_research_inbox_quality_status": codex_research_inbox_quality_status,
    })
    run_result["quality_diagnostics"]["codex_research_inbox"] = dict(
        codex_research_run.get("diagnostics") or {}
    )
    run_result["quality_diagnostics"]["auto_rewrite_attempted_count"] = auto_rewrite_attempted_count
    run_result["quality_diagnostics"]["auto_rewrite_success_count"] = auto_rewrite_success_count
    run_result["quality_diagnostics"]["title_repair"] = title_repair_summary
    run_result["quality_diagnostics"]["model_path_breakdown"] = run_result["model_path_breakdown"]
    run_result["quality_diagnostics"]["llm_health"] = run_result["llm_health"]
    run_result["quality_diagnostics"]["report_structure"] = build_report_structure_diagnostics(report_layers)
    run_result["quality_diagnostics"]["paper_freshness"] = paper_freshness_metrics
    if continuity_metrics:
        run_result["quality_diagnostics"]["continuity"] = continuity_metrics
    generator = ReportGenerator(
        design_version=str(config.get("report", {}).get("design_version") or ReportGenerator.DESIGN_VERSION),
        report_config=report_config,
    )
    final_overlap_reselected_count = 0
    if is_v10_design and bool(report_config.get("paper_freshness_enabled", True)):
        report_layers, final_overlap_reselected_count = enforce_final_visible_paper_overlap(
            report_layers,
            float(quality_gate_config.get("adjacent_report_paper_overlap_max", 0.10) or 0.10),
        )
        if final_overlap_reselected_count:
            print(
                "Final visible-paper reselect removed "
                f"{final_overlap_reselected_count} reappeared update(s) to keep overlap strictly below 10%."
            )
            quality_gate_result = evaluate_report_quality(report_layers, quality_gate_config)
        post_reselect_papers = [
            item
            for item in flatten_report_layers(report_layers)
            if item.get("content_type") == "paper"
        ]
        post_reselect_paper_count = len(post_reselect_papers)
        post_reselect_fresh_paper_count = sum(
            1 for item in post_reselect_papers if not item.get("is_reappeared_update")
        )
        configured_visible_min = int(report_config.get("min_visible_paper_count", 10) or 10)
        effective_visible_min = (
            configured_visible_min
            if bool(quality_gate_config.get("hard_min_visible_paper_count", False))
            else effective_fresh_paper_minimum(
                configured_visible_min,
                post_reselect_fresh_paper_count,
            )
        )
        effective_featured_min = min(
            int(report_config.get("paper_technical_intro_min_count", 10) or 10),
            post_reselect_fresh_paper_count,
        )
        report_config["min_visible_paper_count"] = effective_visible_min
        report_config["paper_technical_intro_min_count"] = effective_featured_min
        quality_gate_config["min_visible_paper_count"] = effective_visible_min
        quality_gate_config["paper_technical_intro_min_count"] = effective_featured_min
        alert_config["min_paper_count"] = effective_visible_min
        generator.report_config["min_visible_paper_count"] = effective_visible_min
        generator.report_config["paper_technical_intro_min_count"] = effective_featured_min
        run_result["quality_diagnostics"]["paper_freshness"][
            "final_overlap_reselected_count"
        ] = final_overlap_reselected_count
        run_result["quality_diagnostics"]["paper_freshness"][
            "post_reselect_paper_count"
        ] = post_reselect_paper_count
        run_result["quality_diagnostics"]["paper_freshness"][
            "post_reselect_fresh_paper_count"
        ] = post_reselect_fresh_paper_count
        run_result["quality_diagnostics"]["paper_freshness"][
            "effective_min_visible_paper_count"
        ] = effective_visible_min
    final_report_layers = generator._decorate_layers(report_layers)
    final_report_layers, final_title_repair = repair_bad_titles_in_layers(final_report_layers)
    # V11 inbox copy has already passed source-hash validation and is immutable.
    if is_learning_digest_design(report_config) and not strict_v11_mode:
        final_report_layers = repair_v10_paper_titles_in_layers(final_report_layers)
    title_repair_summary["bad_title_repaired_count"] += final_title_repair["bad_title_repaired_count"]
    title_repair_summary["bad_title_unresolved_count"] = final_title_repair["bad_title_unresolved_count"]
    title_repair_summary["examples"] = (
        title_repair_summary["examples"] + final_title_repair.get("examples", [])
    )[:10]
    run_result["quality_diagnostics"]["title_repair"] = title_repair_summary
    for item in flatten_report_layers(final_report_layers):
        item["_final_render_item"] = True
    final_quality_result = evaluate_report_quality(final_report_layers, quality_gate_config)
    for item in flatten_report_layers(final_report_layers):
        item["_final_render_item"] = True
    final_report_items = flatten_report_layers(final_report_layers)
    if is_v10_design and bool(report_config.get("paper_freshness_enabled", True)):
        final_visible_papers = [item for item in final_report_items if item.get("content_type") == "paper"]
        paper_freshness_metrics = finalize_paper_freshness_metrics(
            final_visible_papers,
            paper_history,
            paper_freshness_metrics,
            overlap_max=float(quality_gate_config.get("adjacent_report_paper_overlap_max", 0.10) or 0.10),
        )
        paper_freshness_metrics["paper_repeat_filtered_count"] = int(
            paper_freshness_metrics.get("paper_repeat_filtered_count", 0) or 0
        ) + final_overlap_reselected_count
        paper_freshness_metrics["final_overlap_reselected_count"] = final_overlap_reselected_count
        paper_domain_quotas = dict(report_config.get("paper_domain_quotas") or {})
        paper_domain_metrics = evaluate_paper_domain_quotas(final_visible_papers, paper_domain_quotas)
        paper_freshness_metrics.update(paper_domain_metrics)
        target_min = int(report_config.get("paper_target_min_count", 15) or 15)
        final_fresh_paper_count = sum(
            1 for item in final_visible_papers if not item.get("is_reappeared_update")
        )
        if final_fresh_paper_count < target_min:
            warning = f"fresh_paper_target_underfilled:{final_fresh_paper_count}/{target_min}"
            if warning not in run_result["quality_diagnostics"].setdefault("warnings", []):
                run_result["quality_diagnostics"]["warnings"].append(warning)
        run_result["paper_count"] = len(final_visible_papers)
        run_result["fresh_paper_count"] = final_fresh_paper_count
        run_result["reappeared_paper_with_update_count"] = (
            len(final_visible_papers) - final_fresh_paper_count
        )
        report_config["paper_freshness_metrics"] = paper_freshness_metrics
        generator.report_config["paper_freshness_metrics"] = paper_freshness_metrics
        run_result["quality_diagnostics"]["paper_freshness"] = paper_freshness_metrics
        run_result["quality_diagnostics"]["quality_gate"].update({
            "adjacent_report_paper_overlap_rate": paper_freshness_metrics["adjacent_report_paper_overlap_rate"],
            "paper_within_report_duplicate_count": paper_freshness_metrics["paper_within_report_duplicate_count"],
            "final_overlap_reselected_count": final_overlap_reselected_count,
            "paper_repeat_filtered_count": paper_freshness_metrics["paper_repeat_filtered_count"],
            "fresh_paper_count": paper_freshness_metrics["fresh_paper_count"],
            "reappeared_paper_with_update_count": paper_freshness_metrics["reappeared_paper_with_update_count"],
            "arxiv_zero_result_warning_count": int(collector_summary.get("arxiv_zero_result_warning_count", 0) or 0),
            "paper_freshness_status": paper_freshness_metrics["paper_freshness_status"],
            "paper_domain_quota_status": paper_freshness_metrics["paper_domain_quota_status"],
            "paper_domain_quota_exceeded": paper_freshness_metrics["paper_domain_quota_exceeded"],
            "paper_domain_quota_underfilled": paper_freshness_metrics["paper_domain_quota_underfilled"],
        })
    if str(report_config.get("design_version") or "") == "v9-continuous-learning":
        dossiers = build_topic_dossiers(final_report_items)
        queue_context = reading_queue_context(
            db.get_reading_queue(statuses=["tracked"], limit=int(report_config.get("reading_queue_limit", 5) or 5)),
            final_report_items,
            limit=int(report_config.get("reading_queue_limit", 5) or 5),
        )
        weekly_items = continuity_history + final_report_items
        editorial_weights = db.get_preference_weights().get("editorial_style", {})
        v9_context = {
            "topic_dossiers": dossiers,
            "reading_queue": queue_context,
            "closing_memory": build_closing_memory(final_report_items),
            "weekly_digest": build_weekly_digest(weekly_items),
            "show_weekly_digest": datetime.now().weekday() == int(report_config.get("weekly_digest_weekday", 0) or 0),
            "editorial_profile": {
                "mechanism_depth": float(editorial_weights.get("mechanism_depth", 0.0) or 0.0),
                "paper_depth": float(editorial_weights.get("paper_depth", 0.0) or 0.0),
                "concision": float(editorial_weights.get("concision", 0.0) or 0.0),
            },
        }
        generator.report_config["v9_context"] = v9_context
        report_config["v9_context"] = v9_context
        run_result["quality_diagnostics"]["continuity"] = {
            **continuity_metrics,
            "topic_dossier_count": len(dossiers),
            "reading_queue_count": len(queue_context),
            "paper_context_count": sum(
                1 for item in final_report_items if item.get("content_type") == "paper" and item.get("technical_lineage")
            ),
        }
    final_editorial_metrics = build_editorial_quality_metrics(
        final_report_items,
        physical_ai_min_items=int(quality_gate_config.get("physical_ai_featured_min_count", 2) or 2),
        paper_technical_intro_min_count=int(quality_gate_config.get("paper_technical_intro_min_count", 6)),
    )
    final_section_counts = Counter(report_primary_section(item) for item in final_report_items)
    final_event_identities = [
        CodexResearchInboxCollector._event_identity(item)
        for item in final_report_items
        if CodexResearchInboxCollector._event_identity(item)
    ]
    final_section_metrics = {
        "visible_news_count": int(final_section_counts.get("news", 0)),
        "visible_technical_count": int(final_section_counts.get("technical", 0)),
        "visible_paper_count": int(final_section_counts.get("paper", 0)),
        "visible_information_count": int(final_section_counts.get("news", 0) + final_section_counts.get("technical", 0)),
        "cross_section_duplicate_count": len(final_report_items)
        - len({str(item.get("canonical_url") or item.get("url") or item.get("title_cn") or "") for item in final_report_items}),
        "cross_section_event_duplicate_count": len(final_event_identities) - len(set(final_event_identities)),
    }
    if str(report_config.get("product_mode") or "") == "intelligence_v11_editorial_library":
        contract_rejections = list(
            report_config.get("_v11_selection_contract_rejections") or []
        )
        contract_rejection_sections = Counter(
            str(row.get("section") or "unknown")
            for row in contract_rejections
            if isinstance(row, dict)
        )
        contract_rejection_reasons = Counter(
            str(reason)
            for row in contract_rejections
            if isinstance(row, dict)
            for reason in (row.get("reasons") or [])
        )
        current_research_url_set = {
            CodexResearchInboxCollector._url_identity(value)
            for value in codex_research_urls
            if CodexResearchInboxCollector._url_identity(value)
        }
        for research_item in current_research_items:
            for value in (research_item.get("url"), research_item.get("canonical_url")):
                identity = CodexResearchInboxCollector._url_identity(value)
                if identity:
                    current_research_url_set.add(identity)
        external_items = []
        if not report_only:
            for item in final_report_items:
                item_identities = {
                    CodexResearchInboxCollector._url_identity(value)
                    for value in (item.get("url"), item.get("canonical_url"))
                    if CodexResearchInboxCollector._url_identity(value)
                }
                if not item_identities.intersection(current_research_url_set):
                    external_items.append(item)
        final_section_metrics.update({
            "v11_selection_contract_rejected_count": len(contract_rejections),
            "v11_selection_contract_rejected_by_section": dict(
                contract_rejection_sections
            ),
            "v11_selection_contract_rejection_reasons": dict(
                contract_rejection_reasons
            ),
            "v11_external_item_count": len(external_items),
            "v11_external_item_examples": [
                {
                    "title": str(item.get("title_cn") or item.get("title") or "")[:120],
                    "url": str(item.get("canonical_url") or item.get("url") or ""),
                }
                for item in external_items[:5]
            ],
        })
        body_under_min_count = 0
        publish_date_missing_count = 0
        source_evidence_missing_count = 0
        claim_type_missing_count = 0
        paper_full_text_missing_count = 0
        analysis_version_mismatch_count = 0
        for item in final_report_items:
            primary_section = report_primary_section(item)
            facts = item.get("facts") if isinstance(item.get("facts"), dict) else {}
            if (
                str(item.get("model_used") or "") == "codex-automation"
                and str(item.get("analysis_version") or "") not in SUPPORTED_CODEX_RESEARCH_ANALYSIS_VERSIONS
            ):
                analysis_version_mismatch_count += 1
            if not str(item.get("publish_date") or "").strip():
                publish_date_missing_count += 1
            if not str(item.get("claim_type") or facts.get("claim_type") or "").strip():
                claim_type_missing_count += 1
            if not (
                str(item.get("source_excerpt") or facts.get("source_excerpt") or "").strip()
                and str(item.get("evidence_locator") or facts.get("evidence_locator") or "").strip()
            ):
                source_evidence_missing_count += 1
            if primary_section == "paper":
                if not (
                    paper_plain_summary_passes(item.get("paper_plain_summary"))
                    and paper_technical_intro_passes(item.get("paper_technical_intro"))
                ):
                    paper_full_text_missing_count += 1
                continue
            body = re.sub(r"\s+", "", str(item.get("analysis_body") or item.get("summary") or ""))
            content_type = str(item.get("content_type") or "").strip().lower()
            minimum_body_chars = (
                300
                if content_type in {"interview", "podcast", "video"}
                else 220
                if primary_section == "technical"
                else 180
            )
            if len(body) < minimum_body_chars:
                body_under_min_count += 1
        final_section_metrics.update({
            "body_under_min_count": body_under_min_count,
            "publish_date_missing_count": publish_date_missing_count,
            "source_evidence_missing_count": source_evidence_missing_count,
            "claim_type_missing_count": claim_type_missing_count,
            "paper_full_text_missing_count": paper_full_text_missing_count,
            "analysis_version_mismatch_count": analysis_version_mismatch_count,
        })
    if is_v10_design:
        final_reader_context = generator._v10_reader_context(final_report_layers)
        final_editorial_decisions = list(final_reader_context.get("editorial_decisions", []))
        final_decision_source_identities = [
            str(item.get("source_identity") or "").strip()
            for item in final_editorial_decisions
        ]
        final_item_identities = {
            editorial_source_identity(item)
            for item in final_report_items
            if editorial_source_identity(item)
        }
        final_section_metrics.update({
            "editorial_decision_count": len(final_editorial_decisions),
            "editorial_decision_source_missing_count": sum(
                1
                for identity in final_decision_source_identities
                if not identity or identity not in final_item_identities
            ),
            "editorial_decision_duplicate_source_count": len(final_decision_source_identities)
            - len(set(final_decision_source_identities)),
        })
    editorial_decision_count = int(final_section_metrics.get("editorial_decision_count", 0) or 0)
    editorial_decision_min = int(report_config.get("editorial_decision_min_count", 5) or 5)
    editorial_decision_max = int(report_config.get("editorial_decision_max_count", 7) or 7)
    section_minimum_failed = (
        final_section_metrics["visible_news_count"] < int(report_config.get("min_visible_news_count", 20) or 20)
        or final_section_metrics["visible_technical_count"] < int(report_config.get("min_visible_technical_count", 20) or 20)
        or final_section_metrics["visible_paper_count"] < int(report_config.get("min_visible_paper_count", 15) or 15)
        or final_section_metrics["cross_section_duplicate_count"] > 0
        or final_section_metrics["cross_section_event_duplicate_count"] > 0
        or any(
            int(final_section_metrics.get(key, 0) or 0) > 0
            for key in (
                "body_under_min_count",
                "publish_date_missing_count",
                "source_evidence_missing_count",
                "claim_type_missing_count",
                "paper_full_text_missing_count",
                "analysis_version_mismatch_count",
                "v11_external_item_count",
            )
        )
        or (
            is_v10_design
            and not editorial_decision_min <= editorial_decision_count <= editorial_decision_max
        )
        or int(final_section_metrics.get("editorial_decision_source_missing_count", 0) or 0) > 0
        or int(final_section_metrics.get("editorial_decision_duplicate_source_count", 0) or 0) > 0
    )
    if (
        final_quality_result.get("status") != "passed"
        or final_editorial_metrics.get("editorial_quality_status") != "passed"
        or paper_freshness_metrics.get("paper_freshness_status") == "failed"
        or paper_freshness_metrics.get("paper_domain_quota_status") == "failed"
        or section_minimum_failed
        or (
            not report_only
            and bool(quality_gate_config.get("require_codex_research_inbox", False))
            and codex_research_inbox_status != "success"
        )
    ):
        run_result["quality_status"] = "failed"
    else:
        run_result["quality_status"] = "passed"
    run_result["quality_diagnostics"]["quality_gate"].update(
        {key: value for key, value in final_quality_result.items() if key != "failed_items"}
    )
    run_result["quality_diagnostics"]["quality_gate"].update(final_editorial_metrics)
    run_result["quality_diagnostics"]["quality_gate"].update(final_section_metrics)
    run_result["quality_diagnostics"]["report_structure"] = build_report_structure_diagnostics(final_report_layers)
    layered_items = final_report_items
    mixed_items = layered_items or generator.build_mixed_items(papers, updates)
    if is_continuous_reader_design(report_config):
        report_summary = {
            "lead_summary": "",
            "hot_topics": [LEARNING_DOMAIN_LABELS[key] for key in LEARNING_DOMAIN_ORDER],
            "key_takeaways": [],
            "watchlist": [],
        }
    else:
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
        layered_updates=copy.deepcopy(final_report_layers),
        archive_summary=archive_summary,
    )
    html_report = compact_email_html(html_report)
    full_html_size_bytes = len(html_report.encode("utf-8"))
    edition_counts = {
        section: sum(1 for item in final_report_items if report_primary_section(item) == section)
        for section in ("news", "technical", "paper")
    }
    def render_delivery_volume(index: int, volume: Dict[str, Any]) -> str:
        volume_items = list(volume["items"])
        volume_papers = [item for item in volume_items if item.get("content_type") == "paper"]
        volume_updates = [item for item in volume_items if item.get("content_type") != "paper"]
        volume_report_summary = dict(report_summary)
        volume_report_summary["edition_counts"] = edition_counts
        return generator.generate_html(
            papers=volume_papers,
            updates=volume_updates,
            mixed_items=volume_items,
            report_summary=volume_report_summary,
            title=f"{report_title} · 第 {index} 卷：{volume['label']}",
            collector_summary=collector_summary,
            trend_summary=trend_summary,
            alert_summary=alert_summary,
            layered_updates=copy.deepcopy(volume["layers"]),
            archive_summary=archive_summary,
        )

    delivery_volumes, email_split_applied = prepare_v11_email_delivery_volumes(
        html_report,
        final_report_layers,
        report_config,
        render_delivery_volume,
    )
    scan_report_config = dict(report_config)
    if email_split_applied:
        relaxed_limit = full_html_size_bytes + 1
        scan_report_config["email_html_warning_bytes"] = relaxed_limit
        scan_report_config["email_html_max_bytes"] = relaxed_limit
    final_html_metrics = scan_final_html_quality(
        html_report,
        final_report_layers,
        quality_config=quality_gate_config,
        report_config=scan_report_config,
    )
    final_html_metrics.update(
        {
            "email_split_applied": email_split_applied,
            "email_delivery_volume_count": len(delivery_volumes),
            "email_delivery_volume_sizes": [
                int(volume.get("size_bytes") or len(str(volume.get("html") or "").encode("utf-8")))
                for volume in delivery_volumes
            ],
            "full_html_size_bytes": full_html_size_bytes,
        }
    )
    volume_editorial_decision_counts = [
        str(volume.get("html") or "").count('class="v10-decision"')
        for volume in delivery_volumes
    ]
    volume_editorial_decision_visible_source_counts = [
        str(volume.get("html") or "").count('data-source-key="')
        for volume in delivery_volumes
    ]
    volume_editorial_decision_source_missing_counts: List[int] = []
    volume_editorial_decision_duplicate_source_counts: List[int] = []
    for volume in delivery_volumes:
        volume_context = generator._v10_reader_context(dict(volume.get("layers") or {}))
        volume_decisions = list(volume_context.get("editorial_decisions", []))
        decision_identities = [
            str(item.get("source_identity") or "").strip()
            for item in volume_decisions
        ]
        item_identities = {
            editorial_source_identity(item)
            for item in list(volume.get("items") or [])
            if editorial_source_identity(item)
        }
        volume_editorial_decision_source_missing_counts.append(
            sum(1 for identity in decision_identities if not identity or identity not in item_identities)
        )
        volume_editorial_decision_duplicate_source_counts.append(
            len(decision_identities) - len(set(decision_identities))
        )
    volume_claim_label_counts = [
        str(volume.get("html") or "").count('class="v11-claim-label"')
        for volume in delivery_volumes
    ]
    volume_item_counts = [
        int(volume.get("item_count", 0) or 0)
        for volume in delivery_volumes
    ]
    volume_fidelity_metrics = scan_v11_delivery_volume_fidelity(
        delivery_volumes,
        quality_gate_config,
        scan_report_config,
    )
    volume_content_fidelity_missing_counts = list(
        volume_fidelity_metrics[
            "email_delivery_volume_content_fidelity_missing_counts"
        ]
    )
    volume_nav_mismatch_count = 0
    expected_preheader = (
        f"AI 前沿日报：{edition_counts['news']} 条新闻与观点、"
        f"{edition_counts['technical']} 条技术内容、{edition_counts['paper']} 篇论文"
    )
    volume_preheader_mismatch_count = sum(
        expected_preheader not in str(volume.get("html") or "")
        for volume in delivery_volumes
    )
    if email_split_applied:
        section_anchors = {
            "news": "news-and-voices",
            "technical": "technical-trends",
            "paper": "paper-deep-reads",
        }
        for volume in delivery_volumes:
            volume_html = str(volume.get("html") or "")
            expected_anchor = section_anchors.get(str(volume.get("primary_section") or ""))
            live_anchors = {
                anchor
                for anchor in section_anchors.values()
                if f'href="#{anchor}"' in volume_html
            }
            if expected_anchor is None or live_anchors != {expected_anchor}:
                volume_nav_mismatch_count += 1
    final_html_metrics.update(
        {
            "email_delivery_volume_editorial_decision_counts": volume_editorial_decision_counts,
            "email_delivery_volume_editorial_decision_visible_source_counts": volume_editorial_decision_visible_source_counts,
            "email_delivery_volume_editorial_decision_source_missing_counts": volume_editorial_decision_source_missing_counts,
            "email_delivery_volume_editorial_decision_duplicate_source_counts": volume_editorial_decision_duplicate_source_counts,
            "email_delivery_volume_claim_label_counts": volume_claim_label_counts,
            "email_delivery_volume_item_counts": volume_item_counts,
            **volume_fidelity_metrics,
            "email_delivery_volume_nav_mismatch_count": volume_nav_mismatch_count,
            "email_delivery_volume_preheader_mismatch_count": volume_preheader_mismatch_count,
        }
    )
    if (
        strict_v11_mode
        and (
            any(not 5 <= count <= 7 for count in volume_editorial_decision_counts)
            or volume_editorial_decision_visible_source_counts != volume_editorial_decision_counts
            or any(volume_editorial_decision_source_missing_counts)
            or any(volume_editorial_decision_duplicate_source_counts)
            or volume_claim_label_counts != volume_item_counts
            or any(volume_content_fidelity_missing_counts)
            or volume_nav_mismatch_count
            or volume_preheader_mismatch_count
        )
    ):
        final_html_metrics["final_html_quality_status"] = "failed"
    run_result["quality_diagnostics"]["quality_gate"].update(final_html_metrics)
    run_result["quality_diagnostics"]["quality_gate"].update(final_section_metrics)
    if section_minimum_failed:
        run_result["quality_status"] = "failed"
    if final_html_metrics.get("final_html_quality_status") != "passed":
        run_result["quality_status"] = "failed"
    run_result["quality_diagnostics"]["quality_gate"]["status"] = run_result["quality_status"]
    report_dir = Path(str(archive_config.get("report_dir", "archive")))
    report_dir.mkdir(parents=True, exist_ok=True)
    html_path = report_dir / f"report_{file_stamp}.html"
    with open(html_path, "w", encoding="utf-8") as file:
        file.write(html_report)
    html_filename = html_path.as_posix()
    run_result["html_report_path"] = html_filename
    print(f"HTML Report saved to {html_filename}")
    email_volume_paths: List[str] = []
    if email_split_applied:
        for index, volume in enumerate(delivery_volumes, start=1):
            volume_path = report_dir / f"report_{file_stamp}_part{index}.html"
            with volume_path.open("w", encoding="utf-8", newline="") as file:
                file.write(str(volume["html"]))
            volume["path"] = volume_path.as_posix()
            email_volume_paths.append(volume_path.as_posix())
        print(f"Email report split into {len(email_volume_paths)} delivery volumes.")
    else:
        delivery_volumes[0]["path"] = html_filename
        email_volume_paths.append(html_filename)
    run_result["email_volume_paths"] = email_volume_paths
    run_result["quality_diagnostics"]["quality_gate"][
        "email_delivery_volume_paths"
    ] = email_volume_paths
    if bool(scheduler_config.get("ui_audit_enabled", True)):
        ui_audit_root = Path(
            str(scheduler_config.get("ui_audit_output_dir") or "artifacts/v11_production_ui_audit")
        )
        if not ui_audit_root.is_absolute():
            ui_audit_root = Path(__file__).resolve().parent / ui_audit_root
        ui_audit_result = run_email_ui_audit(
            email_volume_paths,
            output_dir=ui_audit_root / report_id,
            root=Path(__file__).resolve().parent,
            timeout_seconds=int(scheduler_config.get("ui_audit_timeout_seconds", 180) or 180),
        )
    else:
        ui_audit_result = {"status": "disabled", "passed": False, "failed_render_count": 0}
    run_result["ui_audit"] = ui_audit_result
    run_result["quality_diagnostics"]["ui_audit"] = ui_audit_result
    run_result["quality_diagnostics"]["quality_gate"].update(
        {
            "ui_audit_status": str(ui_audit_result.get("status") or "unknown"),
            "ui_audit_render_count": int(ui_audit_result.get("render_count", 0) or 0),
            "ui_audit_failed_render_count": int(
                ui_audit_result.get("failed_render_count", 0) or 0
            ),
        }
    )
    if bool(scheduler_config.get("ui_audit_enabled", True)) and not ui_audit_result.get("passed", False):
        run_result["quality_status"] = "failed"
        run_result["quality_diagnostics"]["quality_gate"]["status"] = "failed"
    markdown_report = generator.generate_markdown(
        papers=papers,
        updates=updates,
        mixed_items=mixed_items,
        report_summary=report_summary,
        title=report_title,
        collector_summary=collector_summary,
        trend_summary=trend_summary,
        alert_summary=alert_summary,
        layered_updates=copy.deepcopy(final_report_layers),
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
    email_mode = str(os.getenv("WEB_AGENT_EMAIL_MODE", "send") or "send").strip().lower()
    runtime_slot_id = str(os.getenv("WEB_AGENT_SEND_SLOT_ID", "") or "").strip()
    if report_only and not runtime_slot_id:
        runtime_slot_id = (
            "__report_only__"
            if email_mode in {"dry-run", "dry_run", "skip", "disabled"}
            else "__report_only_send__"
        )
    elif email_mode in {"dry-run", "dry_run", "skip", "disabled"}:
        runtime_slot_id = "__dry_run__"
    elif not runtime_slot_id:
        runtime_slot_id = "__production__"
    send_block_reasons = report_send_blocking_reasons(
        quality_gate_config,
        run_result.get("quality_diagnostics", {}),
    )
    quality_blocked = should_block_report_send(
        quality_gate_config,
        str(run_result.get("quality_status") or ""),
        email_mode,
        run_result.get("quality_diagnostics", {}),
    )
    run_result.setdefault("quality_diagnostics", {})["delivery_gate"] = {
        "status": (
            "blocked"
            if quality_blocked
            else "degraded_send"
            if run_result.get("quality_status") != "passed"
            else "passed"
        ),
        "hard_block_reasons": send_block_reasons,
        "soft_failure_allowed": bool(
            run_result.get("quality_status") != "passed" and not quality_blocked
        ),
    }
    preliminary_delivery_status = (
        "preview"
        if email_mode in {"dry-run", "dry_run", "skip", "disabled"}
        else "blocked"
        if quality_blocked
        else "pending"
    )
    db.record_report_run(
        report_id=report_id,
        run_id=run_id,
        slot_id=runtime_slot_id,
        html_report_path=html_filename,
        markdown_report_path=markdown_filename,
        quality_status=str(run_result.get("quality_status", "")),
        quality_diagnostics=run_result.get("quality_diagnostics", {}),
        delivery_status=preliminary_delivery_status,
    )
    db.record_report_items(report_id, final_report_items)
    if str(report_config.get("design_version") or "") == "v9-continuous-learning":
        db.record_topic_events(report_id, final_report_items)
        db.upsert_topic_dossiers(
            report_id,
            list((report_config.get("v9_context") or {}).get("topic_dossiers") or []),
        )

    if quality_blocked:
        diagnostic_path = report_dir / f"quality_diagnostic_{file_stamp}.json"
        diagnostic_path.write_text(
            json.dumps(run_result.get("quality_diagnostics", {}), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        run_result.update(
            {
                "success": False,
                "status": "quality_blocked",
                "retryable": False,
                "delivery_status": "blocked",
                "quality_diagnostic_path": diagnostic_path.as_posix(),
                "finished_at": datetime.now().isoformat(timespec="seconds"),
            }
        )
        print(f"Email send blocked by final quality gate. Diagnostics saved to {diagnostic_path.as_posix()}")
        return run_result

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
        base_subject = f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}] {report_title}"
        delivery_result = send_email_delivery_volumes(
            notifier,
            recipient,
            base_subject,
            delivery_volumes,
        )
        volume_count = int(delivery_result["volume_count"])
        email_subjects = list(delivery_result["subjects"])
        run_result["email_subject"] = email_subjects[0]
        run_result["email_subjects"] = email_subjects
        sent_volume_count = int(delivery_result["sent_count"])
        success = bool(delivery_result["success"])
        run_result["email_volume_sent_count"] = sent_volume_count
        if success:
            print(f"Notification sent successfully ({sent_volume_count}/{volume_count} volume(s)).")
            notification_sent = True
            commit_payload = record_email_commit(
                db,
                report_id=report_id,
                run_id=run_id,
                subject=email_subjects[0],
                html_report_path=html_filename,
                markdown_report_path=markdown_filename,
                quality_status=str(run_result.get("quality_status", "")),
                send_slot_id=runtime_slot_id,
                send_slot_dir=Path(str(scheduler_config.get("send_slot_dir", "logs/send_slots"))),
            )
            print(
                EMAIL_COMMITTED_MARKER
                + json.dumps(commit_payload, ensure_ascii=False),
                flush=True,
            )
            arrival_config = dict(config.get("email", {}).get("arrival_check", {}) or {})
            if arrival_config.get("enabled", False):
                imap_server = resolve_imap_server(
                    smtp_server=os.getenv("EMAIL_SMTP_SERVER", smtp_server),
                    configured_imap_server=os.getenv("EMAIL_IMAP_SERVER", str(arrival_config.get("imap_server", ""))),
                )
                volume_verifications = [
                    verify_email_arrival(
                        imap_server=imap_server,
                        imap_port=int(os.getenv("EMAIL_IMAP_PORT", str(arrival_config.get("imap_port", 993)))),
                        username=os.getenv("EMAIL_IMAP_USERNAME", os.getenv("EMAIL_SENDER", "")),
                        password=os.getenv("EMAIL_IMAP_PASSWORD", os.getenv("EMAIL_PASSWORD", "")),
                        subject_contains=subject,
                        since_minutes=int(arrival_config.get("since_minutes", 30)),
                        mailbox=arrival_config.get("mailboxes") or str(arrival_config.get("mailbox", "INBOX")),
                        timeout_seconds=int(arrival_config.get("timeout_seconds", config.get("network", {}).get("timeout_seconds", 25))),
                        expected_sender=sender,
                        retry_attempts=int(arrival_config.get("retry_attempts", 1)),
                        retry_delay_seconds=int(arrival_config.get("retry_delay_seconds", 5)),
                    )
                    for subject in email_subjects
                ]
                run_result["delivery_verification"] = (
                    volume_verifications[0]
                    if len(volume_verifications) == 1
                    else {
                        "status": (
                            "found"
                            if all(result.get("status") == "found" for result in volume_verifications)
                            else "not_found"
                            if any(result.get("status") == "not_found" for result in volume_verifications)
                            else "error"
                        ),
                        "volume_count": len(volume_verifications),
                        "volumes": volume_verifications,
                    }
                )
                print(f"Arrival verification status: {run_result['delivery_verification'].get('status')}")
            if alert_config.get("enabled", True) and alert_config.get("send_separate_alert", False) and alert_summary.get("needs_alert"):
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
    db.update_report_delivery_status(report_id, str(run_result.get("delivery_status", "") or ""))
    run_result["finished_at"] = datetime.now().isoformat(timespec="seconds")
    return run_result


if __name__ == "__main__":
    main_result = main()
    if isinstance(main_result, dict) and not main_result.get("success", False):
        sys.exit(1)
