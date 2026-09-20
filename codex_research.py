from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict

import yaml

from src.collectors import (
    CodexResearchInboxCollector,
    build_codex_research_inbox_collector,
    build_codex_research_readiness_summary,
    evaluate_sent_history_overlap,
)
from src.database import Database


ROOT = Path(__file__).resolve().parent


def load_config() -> Dict[str, Any]:
    return yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8")) or {}


def load_inbox_config() -> Dict[str, Any]:
    return dict((load_config().get("sources") or {}).get("codex_research_inbox") or {})


def build_collector(inbox_path_override: Path | None = None) -> CodexResearchInboxCollector:
    config = load_inbox_config()
    return build_codex_research_inbox_collector(
        config,
        root=ROOT,
        inbox_path_override=inbox_path_override,
    )


def resolve_inbox_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def resolve_production_inbox_path(config: Dict[str, Any]) -> Path:
    return resolve_inbox_path(
        os.getenv("WEB_AGENT_CODEX_RESEARCH_INBOX_PATH", "")
        or str(config.get("path") or "data/codex_research/latest.json")
    )


def promote_candidate(
    candidate_path: Path,
    target_path: Path,
    *,
    expected_sha256: str = "",
) -> Dict[str, Any]:
    candidate_path = candidate_path.resolve()
    target_path = target_path.resolve()
    payload = candidate_path.read_bytes()
    payload_sha256 = hashlib.sha256(payload).hexdigest()
    if expected_sha256 and payload_sha256 != expected_sha256:
        return {
            "status": "candidate_changed_after_validation",
            "promoted": False,
            "candidate_path": candidate_path.as_posix(),
            "target_path": target_path.as_posix(),
            "expected_sha256": expected_sha256,
            "actual_sha256": payload_sha256,
            "size_bytes": len(payload),
        }
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{target_path.name}.",
            suffix=".tmp",
            dir=target_path.parent,
            delete=False,
        ) as temporary:
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, target_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return {
        "status": "promoted",
        "promoted": True,
        "candidate_path": candidate_path.as_posix(),
        "target_path": target_path.as_posix(),
        "sha256": payload_sha256,
        "size_bytes": len(payload),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the scheduled Codex research inbox.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable diagnostics.")
    parser.add_argument(
        "--candidate",
        help="Validate this candidate JSON instead of the configured production inbox.",
    )
    parser.add_argument(
        "--promote-on-pass",
        action="store_true",
        help="Atomically replace the configured production inbox after validation passes.",
    )
    args = parser.parse_args()
    if args.promote_on_pass and not args.candidate:
        parser.error("--promote-on-pass requires --candidate")

    full_config = load_config()
    inbox_config = dict((full_config.get("sources") or {}).get("codex_research_inbox") or {})
    candidate_path = resolve_inbox_path(args.candidate) if args.candidate else None
    collector = build_codex_research_inbox_collector(
        inbox_config,
        root=ROOT,
        inbox_path_override=candidate_path,
    )
    items = collector.collect()
    history_metrics = {
        "sent_history_overlap_count": 0,
        "sent_history_overlap_by_section": {},
        "sent_history_overlap_examples": [],
        "production_ready_status": "not_checked",
    }
    if items:
        database_path = Path(str((full_config.get("database") or {}).get("path") or "data/ai_news.db"))
        if not database_path.is_absolute():
            database_path = ROOT / database_path
        database = Database(str(database_path))
        history = database.get_recent_report_items(
            days=int(inbox_config.get("history_dedupe_days", 7) or 7),
            limit=int(inbox_config.get("history_dedupe_limit", 5000) or 5000),
            sent_only=True,
        )
        history_metrics = evaluate_sent_history_overlap(items, history)
    collector.fetch_diagnostics.update(history_metrics)
    readiness_summary = build_codex_research_readiness_summary(
        collector.fetch_diagnostics,
        inbox_config,
    )
    collector.fetch_diagnostics["readiness_summary"] = readiness_summary
    passed = (
        collector.fetch_diagnostics.get("quality_status") == "passed"
        and history_metrics.get("production_ready_status") == "passed"
        and bool(readiness_summary.get("ready_for_dry_run", False))
    )
    promotion: Dict[str, Any] = {
        "status": "not_requested",
        "promoted": False,
    }
    if args.promote_on_pass:
        target_path = resolve_production_inbox_path(inbox_config)
        if passed:
            promotion = promote_candidate(
                candidate_path,
                target_path,
                expected_sha256=str(
                    collector.fetch_diagnostics.get("inbox_sha256") or ""
                ),
            )
            passed = bool(promotion.get("promoted", False))
        else:
            promotion = {
                "status": "blocked",
                "promoted": False,
                "candidate_path": candidate_path.as_posix(),
                "target_path": target_path.as_posix(),
            }
    payload = {
        "status": "passed" if passed else "failed",
        "item_count": len(items),
        "diagnostics": collector.fetch_diagnostics,
        "promotion": promotion,
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(
            f"Codex research validation {payload['status']}: items={len(items)}, "
            f"papers={collector.fetch_diagnostics.get('paper_count', 0)}, "
            f"news={collector.fetch_diagnostics.get('news_count', 0)}"
            f", technical={collector.fetch_diagnostics.get('technical_count', 0)}, "
            f"promotion={promotion.get('status')}"
        )
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
