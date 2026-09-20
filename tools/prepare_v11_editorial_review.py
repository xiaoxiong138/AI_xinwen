from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scheduler_runner import (  # noqa: E402
    _v11_editorial_review_result,
    _v11_primary_section,
    load_runtime_config,
    read_json,
    resolve_database_path,
    write_json,
)


def _load_report(
    slot_id: str,
) -> tuple[Dict[str, Any], list[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    config, scheduler_config = load_runtime_config()
    slot_path = Path(str(scheduler_config["send_slot_dir"])) / f"{slot_id}.json"
    slot = read_json(slot_path)
    if str(slot.get("status") or "") != "sent":
        raise RuntimeError(f"Sent slot not found: {slot_path}")
    database_path = resolve_database_path(config, ROOT)
    conn = sqlite3.connect(str(database_path))
    conn.row_factory = sqlite3.Row
    try:
        run_id = str(slot.get("run_id") or "")
        html_path = str(slot.get("html_report_path") or "")
        report_row = (
            conn.execute(
                "SELECT * FROM report_runs WHERE run_id = ? ORDER BY created_at DESC LIMIT 1",
                (run_id,),
            ).fetchone()
            if run_id
            else conn.execute(
                "SELECT * FROM report_runs WHERE html_report_path = ? ORDER BY created_at DESC LIMIT 1",
                (html_path,),
            ).fetchone()
        )
        if report_row is None:
            raise RuntimeError(f"Report run not found for slot {slot_id}")
        report = dict(report_row)
        item_rows = conn.execute(
            "SELECT rank, section, snapshot_json FROM report_items "
            "WHERE report_id = ? ORDER BY rank",
            (str(report.get("report_id") or ""),),
        ).fetchall()
    finally:
        conn.close()
    items = []
    for row in item_rows:
        try:
            item = json.loads(row["snapshot_json"] or "{}")
        except (TypeError, json.JSONDecodeError):
            continue
        item.setdefault("report_rank", row["rank"])
        item.setdefault("report_section", row["section"])
        items.append(item)
    return report, items, scheduler_config, slot


def _review_delivery_context(
    report: Dict[str, Any], slot: Dict[str, Any]
) -> Dict[str, Any]:
    raw_diagnostics = report.get("quality_diagnostics") or {}
    if isinstance(raw_diagnostics, str):
        try:
            diagnostics = json.loads(raw_diagnostics)
        except json.JSONDecodeError:
            diagnostics = {}
    else:
        diagnostics = dict(raw_diagnostics) if isinstance(raw_diagnostics, dict) else {}
    quality_gate = dict(diagnostics.get("quality_gate") or {})
    ui_audit = dict(diagnostics.get("ui_audit") or slot.get("ui_audit") or {})
    delivery_verification = dict(slot.get("delivery_verification") or {})
    verification_rows = list(delivery_verification.get("volumes") or [])
    if not verification_rows and delivery_verification:
        verification_rows = [delivery_verification]
    subjects = [str(value or "") for value in (slot.get("email_subjects") or [])]
    archive_paths = [
        str(value or "")
        for value in (
            quality_gate.get("email_delivery_volume_paths")
            or slot.get("email_volume_paths")
            or []
        )
        if str(value or "").strip()
    ]
    expected_volume_count = int(
        quality_gate.get("email_delivery_volume_count")
        or len(subjects)
        or len(archive_paths)
        or 1
    )
    return {
        "delivery_status": str(report.get("delivery_status") or slot.get("status") or ""),
        "html_report_path": str(report.get("html_report_path") or ""),
        "expected_volume_count": expected_volume_count,
        "sent_volume_count": int(slot.get("email_volume_sent_count", 0) or 0),
        "subjects": subjects,
        "archive_paths": archive_paths,
        "arrival_status": str(delivery_verification.get("status") or "missing"),
        "matched_subjects": [
            str(row.get("matched_subject") or "")
            for row in verification_rows
            if isinstance(row, dict)
        ],
        "ui_audit_status": str(ui_audit.get("status") or "missing"),
        "ui_audit_metrics_path": str(ui_audit.get("metrics_path") or ""),
    }


def _sample_payload(item: Dict[str, Any], section: str) -> Dict[str, Any]:
    facts = item.get("facts") if isinstance(item.get("facts"), dict) else {}
    return {
        "section": section,
        "article_id": item.get("id"),
        "title": str(item.get("title_cn") or item.get("title") or ""),
        "url": str(item.get("canonical_url") or item.get("url") or ""),
        "publish_date": str(item.get("publish_date") or ""),
        "claim_type": str(item.get("claim_type") or facts.get("claim_type") or ""),
        "body": str(
            item.get("paper_plain_summary")
            if section == "paper"
            else item.get("analysis_body") or item.get("summary") or ""
        ),
        "technical_detail": str(item.get("paper_technical_intro") or ""),
        "source_excerpt": str(item.get("source_excerpt") or facts.get("source_excerpt") or ""),
        "evidence_locator": str(item.get("evidence_locator") or facts.get("evidence_locator") or ""),
        "accuracy": None,
        "specificity": None,
        "readability": None,
        "notes": "",
    }


def prepare_review(slot_id: str, *, sample_size: int = 3) -> Dict[str, Any]:
    report, items, scheduler_config, slot = _load_report(slot_id)
    samples = []
    for section in ("news", "technical", "paper"):
        section_items = [item for item in items if _v11_primary_section(item) == section]
        samples.extend(_sample_payload(item, section) for item in section_items[:sample_size])

    review_dir = Path(str(scheduler_config["v11_editorial_review_dir"]))
    review_path = review_dir / f"{slot_id}.json"
    existing = read_json(review_path)
    existing_by_key = {
        (str(sample.get("section") or ""), str(sample.get("url") or "")): sample
        for sample in (existing.get("samples") or [])
        if isinstance(sample, dict)
    }
    for sample in samples:
        previous = existing_by_key.get((sample["section"], sample["url"]))
        if previous:
            for key in ("accuracy", "specificity", "readability", "notes"):
                sample[key] = previous.get(key)

    same_report = str(existing.get("report_id") or "") == str(report.get("report_id") or "")
    existing_client_rendering = (
        dict(existing.get("client_rendering") or {}) if same_report else {}
    )
    payload = {
        "schema_version": 2,
        "slot_id": slot_id,
        "report_id": str(report.get("report_id") or ""),
        "status": str(existing.get("status") or "pending") if same_report else "pending",
        "reviewer": str(existing.get("reviewer") or "") if same_report else "",
        "reviewed_at": str(existing.get("reviewed_at") or "") if same_report else "",
        "instructions": (
            "Open every source and judge the email copy for accuracy, specificity, and readability. "
            "Set all three booleans and add a concrete note of at least eight characters. "
            "Also open the delivered email in QQ desktop and mobile clients, then complete the "
            "client_rendering checks and notes."
        ),
        "email_delivery": _review_delivery_context(report, slot),
        "client_rendering": {
            "qq_desktop": existing_client_rendering.get("qq_desktop"),
            "qq_mobile": existing_client_rendering.get("qq_mobile"),
            "no_clipping": existing_client_rendering.get("no_clipping"),
            "spacing_readable": existing_client_rendering.get("spacing_readable"),
            "notes": str(existing_client_rendering.get("notes") or ""),
        },
        "samples": samples,
    }
    write_json(review_path, payload)
    return {"review_path": review_path.as_posix(), "review": payload}


def check_review(slot_id: str, *, sample_size: int = 3) -> Dict[str, Any]:
    report, items, scheduler_config, _slot = _load_report(slot_id)
    expected = []
    for section in ("news", "technical", "paper"):
        section_items = [item for item in items if _v11_primary_section(item) == section]
        expected.extend(
            {
                "section": section,
                "url": str(item.get("canonical_url") or item.get("url") or ""),
            }
            for item in section_items[:sample_size]
        )
    review_path = Path(str(scheduler_config["v11_editorial_review_dir"])) / f"{slot_id}.json"
    result = _v11_editorial_review_result(
        read_json(review_path),
        report_id=str(report.get("report_id") or ""),
        slot_id=slot_id,
        expected_samples=expected,
        require_client_rendering=bool(
            scheduler_config.get("v11_client_render_review_required", False)
        ),
    )
    result["review_path"] = review_path.as_posix()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare or validate a V11 editorial review packet.")
    parser.add_argument("--slot-id", required=True)
    parser.add_argument("--sample-size", type=int, default=3)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = (
        check_review(args.slot_id, sample_size=max(1, args.sample_size))
        if args.check
        else prepare_review(args.slot_id, sample_size=max(1, args.sample_size))
    )
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    encoding = sys.stdout.encoding or "utf-8"
    print(payload.encode(encoding, errors="replace").decode(encoding, errors="replace"))
    return 0 if not args.check or result.get("status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
