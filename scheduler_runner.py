from __future__ import annotations

import contextlib
import ctypes
import importlib.util
import json
import locale
import os
import queue
import re
import shutil
import socket
import sqlite3
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml
from dotenv import load_dotenv

from src.collectors import (
    CodexResearchInboxCollector,
    SUPPORTED_CODEX_RESEARCH_ANALYSIS_VERSIONS,
    build_codex_research_inbox_collector,
    build_codex_research_readiness_summary,
    evaluate_sent_history_overlap,
)
from src.database import Database, resolve_database_path
from src.editorial_engine import (
    build_editorial_quality_metrics,
    contains_mojibake,
    paper_plain_summary_passes,
    paper_technical_intro_passes,
)
from src.notifier import EmailNotifier, resolve_imap_server, verify_email_arrival
from src.relevance import infer_source_tier
from src.ui_audit import run_email_ui_audit as execute_email_ui_audit

ROOT = Path(__file__).resolve().parent
RESULT_MARKER = "__SCHEDULER_RESULT__="
EMAIL_COMMITTED_MARKER = "__SCHEDULER_EMAIL_COMMITTED__="
TASK_FIELD_ALIASES: Dict[str, tuple[str, ...]] = {
    "task_name": ("任务名", "TaskName"),
    "next_run_time": ("下次运行时间", "Next Run Time"),
    "status": ("模式", "Status"),
    "logon_mode": ("登录状态", "Logon Mode"),
    "last_run_time": ("上次运行时间", "Last Run Time"),
    "last_result": ("上次结果", "Last Result"),
    "scheduled_task_state": ("计划任务状态", "Scheduled Task State"),
}
TASK_RESULT_HINTS: Dict[int, str] = {
    0: "success",
    1: "failed",
    267008: "ready",
    267009: "running",
    267010: "disabled",
    267011: "never_ran",
    0xC000013A: "interrupted",
}
TASK_RESULT_MESSAGES: Dict[int, str] = {
    0: "The operation completed successfully.",
    1: "The scheduled process exited with code 1; inspect the application run log.",
    267008: "Task is ready to run at the next scheduled time.",
    267009: "Task is currently running.",
    267010: "Task is disabled.",
    267011: "Task has not run yet.",
    0xC000013A: "The scheduled process was interrupted (0xC000013A), commonly by console closure or cancellation.",
}
DOCTOR_PASSING_HINTS = {"success", "ready", "running", "never_ran"}
QUALITY_INFO_WARNING_PREFIXES = (
    "dedupe_removed_updates",
    "fresh_paper_target_underfilled",
)
FOCUS_REPORT_SECTIONS = {"must_read", "physical_ai", "watch", "featured_papers", "research"}
LEARNING_DIGEST_DESIGN_VERSIONS = {"v10-learning-digest", "v11-editorial-library"}


def is_learning_digest_design(value: Any) -> bool:
    return str(value or "") in LEARNING_DIGEST_DESIGN_VERSIONS


DEFAULT_SCHEDULER_CONFIG: Dict[str, Any] = {
    "log_dir": "logs",
    "status_file": "logs/last_run.json",
    "last_success_file": "logs/last_success.json",
    "validation_status_file": "logs/last_validation_run.json",
    "lock_file": "logs/scheduler.lock",
    "validation_report_dir": "archive/validation",
    "doctor_status_file": "logs/doctor_latest.json",
    "doctor_history_file": "logs/doctor_history.json",
    "send_calendar_dir": "logs",
    "log_archive_dir": "logs/archive",
    "task_backup_dir": "logs/task_backups",
    "task_setup_script": "setup_scheduled_tasks.ps1",
    "offline_task_setup_script": "setup_offline_tasks.ps1",
    "repair_task_script": "repair_scheduled_tasks.ps1",
    "task_names": ["Web_Agent_Send_1300_v2", "Web_Agent_Send_2100_v2"],
    "legacy_task_names": ["Web_Agent_Send_1200", "Web_Agent_Send_1200_v2", "Web_Agent_Send_2100"],
    "monitor_auxiliary_tasks": True,
    "require_offline_tasks": False,
    "run_as_user_env": "WEB_AGENT_RUNAS_USER",
    "run_as_password_env": "WEB_AGENT_RUNAS_PASSWORD",
    "send_slot_dir": "logs/send_slots",
    "ui_audit_enabled": True,
    "ui_audit_output_dir": "artifacts/v11_production_ui_audit",
    "ui_audit_timeout_seconds": 180,
    "v11_editorial_review_required": True,
    "v11_editorial_review_dir": "data/v11_editorial_reviews",
    "send_slot_stale_seconds": 10800,
    "send_slots": [
        {"id": "1300", "time": "13:00"},
        {"id": "2100", "time": "21:00"},
    ],
    "send_window_before_minutes": 10,
    "send_window_after_minutes": 180,
    "doctor_task_name": "Web_Agent_Doctor_0900",
    "preflight_task_name": "Web_Agent_Preflight_2030",
    "log_archive_after_days": 14,
    "task_backup_retention_days": 30,
    "task_backup_keep_count": 20,
    "max_run_seconds": 2700,
    "validation_max_run_seconds": 900,
    "validation_report_retention_days": 7,
    "max_attempts": 2,
    "retry_delay_seconds": 60,
    "stale_lock_seconds": 10800,
    "send_failure_email": False,
    "failure_email_subject_prefix": "[AI日报调度失败]",
    "send_doctor_alert_email": False,
    "doctor_warn_streak_threshold": 2,
    "doctor_alert_subject_prefix": "[AI日报健康检查告警]",
    "minimum_free_disk_mb": 512,
    "critical_free_disk_mb": 8,
}


def configure_utf8_stdio() -> None:
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("PYTHONUTF8", "1")
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


class Tee:
    def __init__(self, *streams: Any):
        self.streams = streams

    def write(self, data: str) -> None:
        for stream in self.streams:
            try:
                stream.write(data)
            except UnicodeEncodeError:
                encoding = getattr(stream, "encoding", None) or "utf-8"
                safe_data = data.encode(encoding, errors="replace").decode(encoding, errors="replace")
                stream.write(safe_data)
            stream.flush()

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def load_runtime_config() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    config_path = ROOT / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    scheduler_config = dict(DEFAULT_SCHEDULER_CONFIG)
    scheduler_config.update(config.get("scheduler", {}) or {})

    for key in (
        "log_dir",
        "status_file",
        "last_success_file",
        "validation_status_file",
        "lock_file",
        "validation_report_dir",
        "doctor_status_file",
        "doctor_history_file",
        "send_calendar_dir",
        "log_archive_dir",
        "task_backup_dir",
        "send_slot_dir",
        "ui_audit_output_dir",
        "v11_editorial_review_dir",
    ):
        scheduler_config[key] = str((ROOT / str(scheduler_config[key])).resolve())
    return config, scheduler_config


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8-sig")


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}


def build_status_snapshot(status: Dict[str, Any]) -> Dict[str, Any]:
    snapshot = dict(status)
    collector_summary = dict(snapshot.get("collector_summary") or {})
    rows = list(collector_summary.get("rows") or [])
    collector_summary.pop("rows", None)
    collector_summary["failure_rows"] = [
        {
            "label": row.get("label", ""),
            "status": row.get("status", ""),
            "error": row.get("error", ""),
            "duration_seconds": row.get("duration_seconds", 0),
        }
        for row in rows
        if row.get("status") != "success"
    ]
    snapshot["collector_summary"] = collector_summary
    source_health = dict(snapshot.get("source_health") or {})
    source_health.pop("rows", None)
    if source_health:
        snapshot["source_health"] = source_health
    return snapshot


def _pick_first(data: Dict[str, str], *keys: str) -> str:
    for key in keys:
        value = str(data.get(key, "")).strip()
        if value:
            return value
    return ""


def describe_task_result(raw_value: Any) -> Dict[str, str]:
    raw_text = str(raw_value or "").strip()
    if not raw_text:
        return {"code": "", "hex": "", "hint": "unknown", "message": ""}
    try:
        code = int(raw_text)
    except (TypeError, ValueError):
        return {"code": raw_text, "hex": "", "hint": "unknown", "message": ""}

    unsigned_code = code & 0xFFFFFFFF
    win32_code = unsigned_code & 0xFFFF
    hint = TASK_RESULT_HINTS.get(code) or TASK_RESULT_HINTS.get(unsigned_code, "unknown")
    message = (TASK_RESULT_MESSAGES.get(code) or TASK_RESULT_MESSAGES.get(unsigned_code, "")).strip()
    if not message:
        raw_message = ctypes.FormatError(win32_code).strip()
        if raw_message and raw_message != "<no description>":
            message = raw_message.rstrip(".")

    return {
        "code": str(code),
        "hex": f"0x{unsigned_code:08X}",
        "hint": hint,
        "message": message,
    }


def parse_schtasks_list_output(output: str) -> Dict[str, str]:
    raw: Dict[str, str] = {}
    for line in output.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        raw[key.strip()] = value.strip()

    parsed = {
        field_name: _pick_first(raw, *aliases)
        for field_name, aliases in TASK_FIELD_ALIASES.items()
    }
    result_info = describe_task_result(parsed.get("last_result", ""))
    parsed["last_result_hex"] = result_info["hex"]
    parsed["last_result_hint"] = result_info["hint"]
    parsed["last_result_message"] = result_info["message"]
    return parsed


def query_scheduled_task(task_name: str) -> Dict[str, Any]:
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    completed = subprocess.run(
        ["schtasks", "/Query", "/TN", task_name, "/FO", "LIST", "/V"],
        capture_output=True,
        text=True,
        encoding=locale.getpreferredencoding(False),
        errors="replace",
        creationflags=creationflags,
        check=False,
    )
    if completed.returncode != 0:
        return {
            "task_name": task_name,
            "available": False,
            "error": (completed.stderr or completed.stdout or "").strip(),
        }

    parsed = parse_schtasks_list_output(completed.stdout)
    parsed["task_name"] = parsed.get("task_name") or task_name
    parsed["available"] = True
    return parsed


def is_interactive_task(task: Dict[str, Any]) -> bool:
    logon_mode = str(task.get("logon_mode", "") or "").lower()
    if "background" in logon_mode or "后台" in logon_mode or "s4u" in logon_mode:
        return False
    return "interactive" in logon_mode or "交互" in logon_mode


def build_monitored_task_names(scheduler_config: Dict[str, Any]) -> list[str]:
    names = [str(name) for name in (scheduler_config.get("task_names") or []) if str(name).strip()]
    if scheduler_config.get("monitor_auxiliary_tasks", True):
        for key in ("doctor_task_name", "preflight_task_name"):
            name = str(scheduler_config.get(key, "") or "").strip()
            if name and name not in names:
                names.append(name)
    return names


def build_legacy_task_names(scheduler_config: Dict[str, Any]) -> list[str]:
    monitored = {name.lstrip("\\") for name in build_monitored_task_names(scheduler_config)}
    names: list[str] = []
    for raw_name in scheduler_config.get("legacy_task_names") or []:
        name = str(raw_name).strip()
        if name and name.lstrip("\\") not in monitored and name not in names:
            names.append(name)
    return names


def is_task_enabled(task: Dict[str, Any]) -> bool:
    state = str(task.get("scheduled_task_state", "") or "").lower()
    if "disabled" in state or "禁用" in state or "已禁用" in state:
        return False
    return bool(task.get("available", False))


def build_status_text(
    last_status: Dict[str, Any],
    task_infos: list[Dict[str, Any]],
    lock_info: Dict[str, Any],
    validation_status: Optional[Dict[str, Any]] = None,
    last_success: Optional[Dict[str, Any]] = None,
) -> str:
    lines = [
        f"Current time: {datetime.now().isoformat(timespec='seconds')}",
        f"Recent status: {last_status.get('status', 'unknown') or 'unknown'}",
        f"Delivery status: {last_status.get('delivery_status', 'unknown') or 'unknown'}",
        f"Recent finished: {last_status.get('finished_at', '') or 'N/A'}",
        f"Run log: {last_status.get('log_file', '') or 'N/A'}",
    ]

    if last_status.get("html_report_path") or last_status.get("markdown_report_path"):
        lines.append(f"HTML report: {last_status.get('html_report_path', '') or 'N/A'}")
        lines.append(f"Markdown report: {last_status.get('markdown_report_path', '') or 'N/A'}")

    last_success = dict(last_success or {})
    if last_success:
        lines.append(f"Last successful send: {last_success.get('finished_at', '') or 'N/A'}")
        lines.append(f"Last success HTML: {last_success.get('html_report_path', '') or 'N/A'}")

    previous_success = dict(last_status.get("previous_success") or {})
    if previous_success:
        lines.append(f"Previous success: {previous_success.get('finished_at', '') or 'N/A'}")
        lines.append(f"Previous success HTML: {previous_success.get('html_report_path', '') or 'N/A'}")

    validation_status = dict(validation_status or {})
    if validation_status:
        lines.append(f"Recent validation status: {validation_status.get('status', 'unknown') or 'unknown'}")
        lines.append(f"Validation delivery: {validation_status.get('delivery_status', 'unknown') or 'unknown'}")
        lines.append(f"Validation finished: {validation_status.get('finished_at', '') or 'N/A'}")
        lines.append(f"Validation log: {validation_status.get('log_file', '') or 'N/A'}")

    cleanup_summary = dict(last_status.get("cleanup_summary") or {})
    removed_logs = list(cleanup_summary.get("removed_logs") or [])
    removed_validation_reports = list(cleanup_summary.get("removed_validation_reports") or [])
    removed_task_backups = list(cleanup_summary.get("removed_task_backups") or [])
    lines.append(f"Removed logs this run: {len(removed_logs)}")
    lines.append(f"Removed validation reports this run: {len(removed_validation_reports)}")
    lines.append(f"Removed task backups this run: {len(removed_task_backups)}")

    if lock_info:
        lock_pid = int(lock_info.get("pid", 0) or 0)
        lock_is_recent = False
        acquired_at = str(lock_info.get("acquired_at", "") or "")
        try:
            acquired_dt = datetime.fromisoformat(acquired_at)
            stale_seconds = int(DEFAULT_CONFIG.get("scheduler", {}).get("stale_lock_seconds", 10800) or 10800)
            lock_is_recent = (datetime.now() - acquired_dt).total_seconds() <= stale_seconds
        except Exception:
            lock_is_recent = False
        lock_state = "active" if is_pid_running(lock_pid) and lock_is_recent else "stale"
        lines.append(
            "Current lock: "
            f"{lock_state}, "
            f"pid={lock_info.get('pid', '') or 'N/A'}, "
            f"acquired_at={lock_info.get('acquired_at', '') or 'N/A'}"
        )
    else:
        lines.append("Current lock: idle")

    lines.append("")
    lines.append("Scheduled tasks:")
    for task in task_infos:
        if not task.get("available", False):
            lines.append(f"- {task.get('task_name', 'unknown')}: unavailable ({task.get('error', 'query failed')})")
            continue
        lines.append(
            f"- {task.get('task_name', 'unknown')}: {task.get('status', 'unknown') or 'unknown'}, "
            f"next {task.get('next_run_time', 'N/A') or 'N/A'}, "
            f"logon {task.get('logon_mode', 'N/A') or 'N/A'}, "
            f"last result {task.get('last_result', 'N/A') or 'N/A'} "
            f"({task.get('last_result_hex', '') or 'N/A'}, {task.get('last_result_hint', 'unknown')})"
        )
        if task.get("last_result_message"):
            lines.append(f"  message: {task.get('last_result_message')}")
    return "\n".join(lines)


def build_scheduler_status_payload() -> Dict[str, Any]:
    _, scheduler_config = load_runtime_config()
    status_path = Path(scheduler_config["status_file"])
    last_success_path = Path(scheduler_config["last_success_file"])
    validation_status_path = Path(scheduler_config["validation_status_file"])
    lock_path = Path(scheduler_config["lock_file"])
    last_status = read_json(status_path)
    last_success = read_json(last_success_path)
    validation_status = read_json(validation_status_path)
    lock_info = read_json(lock_path) if lock_path.exists() else {}
    lock_pid = int(lock_info.get("pid", 0) or 0) if lock_info else 0
    lock_summary = {
        "state": "idle",
        "pid": lock_info.get("pid", ""),
        "acquired_at": lock_info.get("acquired_at", ""),
    }
    if lock_info:
        lock_summary["state"] = "active" if is_pid_running(lock_pid) else "stale"
    task_names = build_monitored_task_names(scheduler_config)
    task_infos = [query_scheduled_task(task_name) for task_name in task_names]
    legacy_task_names = build_legacy_task_names(scheduler_config)
    legacy_task_infos = [query_scheduled_task(task_name) for task_name in legacy_task_names]
    return {
        "current_time": datetime.now().isoformat(timespec="seconds"),
        "last_run": last_status,
        "last_success": last_success,
        "last_validation_run": validation_status,
        "lock": lock_summary,
        "tasks": task_infos,
        "legacy_tasks": legacy_task_infos,
    }


def print_scheduler_status(as_json: bool = False) -> int:
    payload = build_scheduler_status_payload()
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    last_status = dict(payload.get("last_run") or {})
    last_success = dict(payload.get("last_success") or {})
    validation_status = dict(payload.get("last_validation_run") or {})
    lock_summary = dict(payload.get("lock") or {})
    task_infos = list(payload.get("tasks") or [])
    if lock_summary.get("state") == "idle":
        lock_info = {}
    else:
        lock_info = {
            "pid": lock_summary.get("pid", ""),
            "acquired_at": lock_summary.get("acquired_at", ""),
        }
    print(
        build_status_text(
            last_status,
            task_infos,
            lock_info,
            validation_status=validation_status,
            last_success=last_success,
        )
    )
    return 0


def _doctor_item(name: str, level: str, detail: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    payload = {
        "name": name,
        "level": level,
        "detail": detail,
    }
    if data:
        payload["data"] = data
    return payload


def format_arxiv_retry_paths(paths: list[Dict[str, Any]], limit: int = 5) -> str:
    if not paths:
        return ""
    return (
        "retry paths "
        + "; ".join(
            f"{path.get('source', 'arXiv')}:{'->'.join(str(value) for value in path.get('attempted_show_counts', []))}"
            + (
                f" (recovered at {path.get('successful_show_count')})"
                if path.get("successful_show_count")
                else f" ({path.get('result', 'failed')})"
            )
            for path in paths[: max(1, int(limit or 5))]
        )
        + "; "
    )


def build_scheduler_launcher_status(log_dir: Path) -> Dict[str, Any]:
    status_path = log_dir / "scheduler_launcher_latest.json"
    if not status_path.exists():
        return {"available": False, "status": "not_recorded", "status_file": status_path.as_posix()}
    payload = read_json(status_path)
    status = str(payload.get("status", "unknown") or "unknown")
    pid = int(payload.get("pid", 0) or 0)
    if status == "running" and not is_pid_running(pid):
        status = "interrupted"
    return {
        **payload,
        "available": True,
        "status": status,
        "status_file": status_path.as_posix(),
        "pid_running": is_pid_running(pid) if pid else False,
    }


def check_feedback_server_status(config: Dict[str, Any]) -> Dict[str, Any]:
    feedback_config = dict(config.get("feedback") or {})
    enabled = bool(feedback_config.get("enabled", False))
    host = str(feedback_config.get("host", "127.0.0.1") or "127.0.0.1")
    port = int(feedback_config.get("port", 8765) or 8765)
    status = {"enabled": enabled, "host": host, "port": port, "reachable": False, "healthy": False, "error": ""}
    if not enabled:
        return status
    try:
        with socket.create_connection((host, port), timeout=1.0):
            status["reachable"] = True
    except OSError as exc:
        status["error"] = str(exc)
        return status
    errors: List[str] = []
    for path in ("/health", "/"):
        try:
            with urllib.request.urlopen(f"http://{host}:{port}{path}", timeout=2) as response:
                body = response.read(4096).decode("utf-8", errors="replace")
                status["healthy"] = "WEB_AGENT_FEEDBACK_OK" in body or "AI 日报反馈服务正在运行" in body
                if status["healthy"]:
                    status["error"] = ""
                    return status
                errors.append(f"{path}: unexpected response")
        except Exception as exc:
            errors.append(f"{path}: {exc}")
    status["error"] = "; ".join(errors)
    return status


def delivery_verification_passed(verification: Dict[str, Any]) -> bool:
    if bool(verification.get("verified", False)):
        return True
    if str(verification.get("status") or "") != "found":
        return False
    volumes = list(verification.get("volumes") or [])
    if not volumes:
        return bool(verification.get("matched_subject"))
    return all(
        str(volume.get("status") or "") == "found"
        and (
            bool(volume.get("verified", False))
            or bool(volume.get("matched_subject"))
        )
        for volume in volumes
    )


def build_latest_report_snapshot_metrics(report_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
    from main import evaluate_paper_domain_quotas

    if not report_id:
        return {}
    report_config = dict(config.get("report") or {})
    conn = None
    try:
        conn = sqlite3.connect(resolve_database_path(config, ROOT))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT section, snapshot_json FROM report_items WHERE report_id = ? ORDER BY rank",
            (report_id,),
        ).fetchall()
    except Exception:
        return {}
    finally:
        if conn is not None:
            conn.close()
    items = []
    for row in rows:
        try:
            item = json.loads(row["snapshot_json"] or "{}")
        except Exception:
            continue
        item.setdefault("report_section", row["section"])
        items.append(item)
    if not items:
        return {}
    metrics = build_editorial_quality_metrics(
        items,
        physical_ai_min_items=int(report_config.get("physical_ai_featured_min_count", 4) or 4),
        paper_technical_intro_min_count=int(report_config.get("paper_technical_intro_min_count", 12) or 12),
    )
    metrics.update(
        evaluate_paper_domain_quotas(
            [item for item in items if str(item.get("content_type") or "").lower() == "paper"],
            report_config.get("paper_domain_quotas", {}),
        )
    )
    return metrics


def build_v8_production_acceptance(
    scheduler_config: Dict[str, Any],
    *,
    required_count: int = 3,
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    required_count = max(1, int(required_count or 3))
    slot_dir = Path(str(scheduler_config.get("send_slot_dir", ROOT / "logs/send_slots")))
    if not slot_dir.is_absolute():
        slot_dir = ROOT / slot_dir
    sent_slots = []
    for path in slot_dir.glob("*.json") if slot_dir.exists() else []:
        payload = read_json(path)
        if str(payload.get("status", "")) != "sent":
            continue
        finished_at = _parse_status_datetime(payload.get("finished_at"))
        if finished_at is None:
            continue
        payload["_finished_at"] = finished_at
        sent_slots.append(payload)
    sent_slots.sort(key=lambda item: item["_finished_at"], reverse=True)

    conn = None
    rows = []
    try:
        conn = sqlite3.connect(str(db_path or resolve_database_path(None, ROOT)))
        conn.row_factory = sqlite3.Row
        for slot in sent_slots:
            run_id = str(slot.get("run_id", "") or "")
            html_path = str(slot.get("html_report_path", "") or "")
            if run_id:
                row = conn.execute(
                    "SELECT * FROM report_runs WHERE run_id = ? ORDER BY created_at DESC LIMIT 1",
                    (run_id,),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT * FROM report_runs WHERE html_report_path = ? ORDER BY created_at DESC LIMIT 1",
                    (html_path,),
                ).fetchone()
            if row is None:
                continue
            report = dict(row)
            try:
                diagnostics = json.loads(report.get("quality_diagnostics") or "{}")
            except (TypeError, json.JSONDecodeError):
                diagnostics = {}
            if str(diagnostics.get("report_design_version", "")) != "v8-editorial-reader":
                continue
            gate = dict(diagnostics.get("quality_gate") or {})
            issues = []
            required_zero = {
                "final_html_bad_title_count": int(gate.get("final_html_bad_title_count", 0) or 0),
                "untranslated_fact_count": int(gate.get("untranslated_fact_count", 0) or 0),
                "paper_mechanism_missing_count": int(gate.get("paper_mechanism_missing_count", 0) or 0),
                "paper_result_context_missing_count": int(gate.get("paper_result_context_missing_count", 0) or 0),
                "appendix_body_overlap_count": int(gate.get("appendix_body_overlap_count", 0) or 0),
                "truncated_focus_text_count": int(gate.get("truncated_focus_text_count", 0) or 0),
                "paper_intro_length_fail_count": int(gate.get("paper_intro_length_fail_count", 0) or 0),
                "display_body_over_limit_count": int(gate.get("display_body_over_limit_count", 0) or 0),
            }
            for name, value in required_zero.items():
                if value:
                    issues.append(f"{name}={value}")
            duplicate_count = int(gate.get("exact_duplicate_sentence_count", 0) or 0)
            opening_repeat = int(gate.get("max_opening_repeat_count", 0) or 0)
            source_concentration = float(gate.get("focus_source_concentration", 0.0) or 0.0)
            if duplicate_count > 1:
                issues.append(f"exact_duplicate_sentence_count={duplicate_count}")
            if opening_repeat > 2:
                issues.append(f"max_opening_repeat_count={opening_repeat}")
            if source_concentration > 0.25:
                issues.append(f"focus_source_concentration={source_concentration:.3f}")
            source_max = int(gate.get("focus_source_max_count", 0) or 0)
            topic_max = int(gate.get("focus_topic_max_count", 0) or 0)
            if source_max > 2:
                issues.append(f"focus_source_max_count={source_max}")
            if topic_max > 3:
                issues.append(f"focus_topic_max_count={topic_max}")
            if bool(gate.get("memory_budget_exceeded", False)):
                issues.append("memory_budget_exceeded")
            if str(gate.get("final_html_quality_status", "")) != "passed":
                issues.append("final_html_quality_status_not_passed")
            rows.append(
                {
                    "slot_id": slot.get("slot_id", ""),
                    "finished_at": slot.get("finished_at", ""),
                    "report_id": report.get("report_id", ""),
                    "html_report_path": report.get("html_report_path", ""),
                    "passed": not issues,
                    "issues": issues,
                }
            )
            if len(rows) >= required_count:
                break
    except (OSError, sqlite3.Error) as exc:
        return {
            "status": "pending",
            "required_count": required_count,
            "verified_count": 0,
            "passed_count": 0,
            "reports": [],
            "error": str(exc),
        }
    finally:
        if conn is not None:
            conn.close()

    passed_count = sum(1 for row in rows if row.get("passed"))
    if any(not row.get("passed") for row in rows):
        status = "failed"
    elif len(rows) >= required_count:
        status = "passed"
    else:
        status = "pending"
    return {
        "status": status,
        "required_count": required_count,
        "verified_count": len(rows),
        "passed_count": passed_count,
        "reports": rows,
        "error": "",
    }


def _paper_snapshot_keys(item: Dict[str, Any]) -> set[str]:
    from main import paper_history_keys

    return paper_history_keys(item)


def _paper_snapshot_key(item: Dict[str, Any]) -> str:
    from main import paper_history_key

    return paper_history_key(item)


def _paper_snapshots_match(left: Dict[str, Any], right: Dict[str, Any]) -> bool:
    from main import paper_identity_matches

    return paper_identity_matches(left, right)


def _paper_reappearance_is_supported(item: Dict[str, Any], previous_items: list[Dict[str, Any]]) -> bool:
    label = str(item.get("paper_status_label") or "")
    reason = str(item.get("paper_change_reason") or "").strip()
    if not item.get("is_reappeared_update") or not reason or not previous_items:
        return False

    from main import detect_reappeared_paper_update

    detected_label, detected_reason = detect_reappeared_paper_update(item, previous_items)
    return label == detected_label and bool(detected_reason)


def build_paper_freshness_production_acceptance(
    scheduler_config: Dict[str, Any],
    *,
    required_days: int = 3,
    overlap_max: float = 0.10,
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    required_days = max(1, int(required_days or 3))
    slot_dir = Path(str(scheduler_config.get("send_slot_dir", ROOT / "logs/send_slots")))
    if not slot_dir.is_absolute():
        slot_dir = ROOT / slot_dir
    sent_slots = []
    for path in slot_dir.glob("*.json") if slot_dir.exists() else []:
        payload = read_json(path)
        if str(payload.get("status", "")) != "sent":
            continue
        finished_at = _parse_status_datetime(payload.get("finished_at"))
        if finished_at is None:
            continue
        payload["_finished_at"] = finished_at
        sent_slots.append(payload)
    sent_slots.sort(key=lambda item: item["_finished_at"])

    conn = None
    reports = []
    seen_report_ids: set[str] = set()
    try:
        conn = sqlite3.connect(str(db_path or resolve_database_path(None, ROOT)))
        conn.row_factory = sqlite3.Row
        for slot in sent_slots:
            run_id = str(slot.get("run_id", "") or "")
            html_path = str(slot.get("html_report_path", "") or "")
            row = (
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
            if row is None:
                continue
            report = dict(row)
            report_id = str(report.get("report_id", "") or "")
            if (
                not report_id
                or report_id in seen_report_ids
                or str(report.get("delivery_status", "") or "") != "sent"
            ):
                continue
            seen_report_ids.add(report_id)
            try:
                diagnostics = json.loads(report.get("quality_diagnostics") or "{}")
            except (TypeError, json.JSONDecodeError):
                diagnostics = {}
            report_design_version = str(diagnostics.get("report_design_version", ""))
            gate = dict(diagnostics.get("quality_gate") or {})
            freshness = dict(diagnostics.get("paper_freshness") or {})
            report["_freshness_enabled"] = is_learning_digest_design(report_design_version) and (
                bool(freshness) or bool(gate.get("paper_freshness_status"))
            )
            report["_paper_freshness_status"] = str(
                freshness.get("paper_freshness_status")
                or gate.get("paper_freshness_status")
                or "unknown"
            )
            report["_paper_domain_quota_status"] = str(
                freshness.get("paper_domain_quota_status")
                or gate.get("paper_domain_quota_status")
                or "unknown"
            )
            report["_minimum_paper_count"] = max(
                1,
                int(
                    freshness.get("effective_min_visible_paper_count")
                    or freshness.get("configured_min_visible_paper_count")
                    or 10
                ),
            )
            report["_recommended_minimum_paper_count"] = max(
                report["_minimum_paper_count"],
                int(
                    freshness.get("configured_target_min_paper_count")
                    or freshness.get("configured_min_visible_paper_count")
                    or 10
                ),
            )
            report["_maximum_paper_count"] = max(
                report["_minimum_paper_count"],
                int(freshness.get("configured_target_max_paper_count") or 25),
            )
            item_rows = conn.execute(
                "SELECT snapshot_json FROM report_items WHERE report_id = ? ORDER BY rank",
                (report_id,),
            ).fetchall()
            papers = []
            for item_row in item_rows:
                try:
                    item = json.loads(item_row["snapshot_json"] or "{}")
                except (TypeError, json.JSONDecodeError):
                    continue
                if str(item.get("content_type", "")).lower() == "paper":
                    item["_history_created_at"] = str(report.get("created_at") or "")
                    item["_history_delivery_at"] = str(report.get("delivery_at") or "")
                    papers.append(item)
            report["_finished_at"] = slot["_finished_at"]
            report["_papers"] = papers
            report["_fresh_papers"] = [item for item in papers if not item.get("is_reappeared_update")]
            report["_keys"] = {_paper_snapshot_key(item) for item in papers if _paper_snapshot_key(item)}
            report["_aliases"] = set().union(*(_paper_snapshot_keys(item) for item in papers)) if papers else set()
            reports.append(report)
    except (OSError, sqlite3.Error) as exc:
        return {
            "status": "pending",
            "required_days": required_days,
            "verified_days": 0,
            "verified_report_count": 0,
            "reports": [],
            "error": str(exc),
        }
    finally:
        if conn is not None:
            conn.close()

    available_days = sorted(
        {report["_finished_at"].date() for report in reports if report.get("_freshness_enabled")},
        reverse=True,
    )
    consecutive_days = []
    if available_days:
        available_day_set = set(available_days)
        expected_day = available_days[0]
        while expected_day in available_day_set and len(consecutive_days) < required_days:
            consecutive_days.append(expected_day)
            expected_day -= timedelta(days=1)
    selected_days = {day.isoformat() for day in consecutive_days}
    rows = []
    for index, report in enumerate(reports):
        report_day = report["_finished_at"].date().isoformat()
        if report_day not in selected_days or not report.get("_freshness_enabled"):
            continue
        current_keys = report["_keys"]
        fresh_paper_count = len(report.get("_fresh_papers") or [])
        within_report_duplicates = []
        for paper_index, item in enumerate(report["_papers"]):
            if any(
                _paper_snapshots_match(item, previous_item)
                for previous_item in report["_papers"][:paper_index]
            ):
                within_report_duplicates.append(_paper_snapshot_key(item))
        previous = reports[index - 1] if index else None
        previous_papers = previous["_papers"] if previous else []
        overlap_items = [
            item
            for item in report["_papers"]
            if any(_paper_snapshots_match(item, prior_item) for prior_item in previous_papers)
        ]
        overlap_keys = sorted({_paper_snapshot_key(item) for item in overlap_items if _paper_snapshot_key(item)})
        overlap_count = len(overlap_items)
        overlap_rate = overlap_count / max(1, len(current_keys))
        cooldown_items: list[Dict[str, Any]] = []
        for prior in reports[:index]:
            if report["_finished_at"] - prior["_finished_at"] <= timedelta(days=7):
                cooldown_items.extend(prior["_papers"])
        unjustified = []
        for item in report["_papers"]:
            previous_items = [
                previous_item
                for previous_item in cooldown_items
                if _paper_snapshots_match(item, previous_item)
            ]
            if not previous_items:
                continue
            if not _paper_reappearance_is_supported(item, previous_items):
                unjustified.append(_paper_snapshot_key(item))
        issues = []
        report_quality_status = str(report.get("quality_status", "unknown") or "unknown")
        freshness_status = str(report.get("_paper_freshness_status", "unknown") or "unknown")
        domain_quota_status = str(report.get("_paper_domain_quota_status", "unknown") or "unknown")
        minimum_paper_count = int(report.get("_minimum_paper_count", 10) or 10)
        recommended_minimum_paper_count = int(report.get("_recommended_minimum_paper_count", 10) or 10)
        maximum_paper_count = int(report.get("_maximum_paper_count", 25) or 25)
        if report_quality_status != "passed":
            issues.append(f"report_quality_status={report_quality_status}")
        if freshness_status != "passed":
            issues.append(f"paper_freshness_status={freshness_status}")
        if domain_quota_status not in {"unknown", "passed"}:
            issues.append(f"paper_domain_quota_status={domain_quota_status}")
        if fresh_paper_count < minimum_paper_count:
            issues.append(f"fresh_paper_count_below_min={fresh_paper_count}/{minimum_paper_count}")
        if fresh_paper_count > maximum_paper_count:
            issues.append(f"fresh_paper_count_above_max={fresh_paper_count}/{maximum_paper_count}")
        if overlap_rate >= overlap_max:
            issues.append(f"adjacent_overlap={overlap_rate:.3f}")
        if within_report_duplicates:
            issues.append(f"within_report_duplicates={len(within_report_duplicates)}")
        if unjustified:
            issues.append(f"unjustified_7d_repeats={len(set(unjustified))}")
        rows.append({
            "date": report_day,
            "finished_at": report["_finished_at"].isoformat(),
            "report_id": report.get("report_id", ""),
            "paper_count": len(current_keys),
            "fresh_paper_count": fresh_paper_count,
            "reappeared_paper_with_update_count": len(current_keys) - fresh_paper_count,
            "minimum_paper_count": minimum_paper_count,
            "recommended_minimum_paper_count": recommended_minimum_paper_count,
            "paper_target_underfilled": fresh_paper_count < recommended_minimum_paper_count,
            "maximum_paper_count": maximum_paper_count,
            "report_quality_status": report_quality_status,
            "paper_freshness_status": freshness_status,
            "paper_domain_quota_status": domain_quota_status,
            "adjacent_report_paper_overlap_count": overlap_count,
            "adjacent_report_paper_overlap_rate": round(overlap_rate, 3),
            "adjacent_report_paper_overlap_keys": overlap_keys[:10],
            "paper_within_report_duplicate_count": len(within_report_duplicates),
            "paper_within_report_duplicate_keys": sorted(set(within_report_duplicates))[:10],
            "unjustified_7d_repeat_count": len(set(unjustified)),
            "unjustified_7d_repeat_keys": sorted(set(unjustified))[:10],
            "passed": not issues,
            "issues": issues,
        })
    rows.sort(key=lambda row: row["finished_at"], reverse=True)
    verified_days = len({row["date"] for row in rows})
    if any(not row["passed"] for row in rows):
        status = "failed"
    elif verified_days >= required_days:
        status = "passed"
    else:
        status = "pending"
    return {
        "status": status,
        "required_days": required_days,
        "verified_days": verified_days,
        "verified_report_count": len(rows),
        "passed_report_count": sum(1 for row in rows if row["passed"]),
        "reports": rows,
        "error": "",
    }


V11_PRODUCT_MODE = "intelligence_v11_editorial_library"
V11_DESIGN_VERSION = "v11-editorial-library"
V11_ACCEPTANCE_CONTRACT_VERSION = 12
V11_GENERIC_PHRASES = (
    "相关机构",
    "出现了新的动作",
    "不只是单点更新",
    "可能影响产品路线",
    "值得持续关注",
    "未来可能带来影响",
    "学习重点是",
    "技术上，它主要围绕",
    "需要回看原文确认",
    "当前摘要还缺少",
)


def _v11_primary_section(item: Dict[str, Any]) -> str:
    facts = item.get("facts") if isinstance(item.get("facts"), dict) else {}
    section = str(item.get("primary_section") or facts.get("primary_section") or "").strip().lower()
    if section in {"news", "technical", "paper"}:
        return section
    content_type = str(item.get("content_type") or "").strip().lower()
    if content_type == "paper":
        return "paper"
    if content_type in {"project", "open_source", "opensource"}:
        return "technical"
    return "news"


def _v11_item_identity(item: Dict[str, Any]) -> str:
    url = str(item.get("canonical_url") or item.get("url") or "").strip()
    if url:
        return CodexResearchInboxCollector._url_identity(url)
    return str(item.get("title_cn") or item.get("title") or f"article:{item.get('id', '')}").strip().lower()


def _v11_sample_quality(item: Dict[str, Any], section: str) -> list[str]:
    facts = item.get("facts") if isinstance(item.get("facts"), dict) else {}
    evidence = facts.get("evidence") or item.get("evidence_points") or []
    if isinstance(evidence, str):
        evidence = [evidence]
    source_excerpt = str(item.get("source_excerpt") or facts.get("source_excerpt") or "").strip()
    evidence_locator = str(
        item.get("evidence_locator") or facts.get("evidence_locator") or ""
    ).strip()
    publish_date = str(item.get("publish_date") or "").strip()
    claim_type = str(item.get("claim_type") or facts.get("claim_type") or "").strip()
    title = str(item.get("title_cn") or item.get("title") or "").strip()
    issues = []
    if str(item.get("analysis_version") or "") not in SUPPORTED_CODEX_RESEARCH_ANALYSIS_VERSIONS:
        issues.append("analysis_version_outdated")
    if not title:
        issues.append("title_missing")
    elif contains_mojibake(title) or "…" in title or "..." in title:
        issues.append("title_not_publishable")
    if not re.match(r"^\d{4}-\d{2}-\d{2}", publish_date):
        issues.append("publish_date_missing")
    if claim_type not in {
        "verified_fact",
        "official_claim",
        "interview_opinion",
        "analysis",
        "research_result",
    }:
        issues.append("claim_type_missing")
    if not all(str(facts.get(key) or "").strip() for key in ("who", "action", "target")):
        issues.append("structured_facts_missing")
    if not source_excerpt:
        issues.append("source_excerpt_missing")
    if not evidence_locator:
        issues.append("evidence_locator_missing")
    try:
        evidence_quality = float(item.get("evidence_quality", 0) or 0)
        information_density = float(
            item.get("information_density", item.get("evidence_quality", 0)) or 0
        )
    except (TypeError, ValueError):
        evidence_quality = 0.0
        information_density = 0.0
    if evidence_quality < 0.45:
        issues.append("evidence_quality_low")
    if information_density < 0.45:
        issues.append("information_density_low")
    if section == "paper":
        body = " ".join(
            str(item.get(key) or "").strip()
            for key in ("paper_plain_summary", "paper_technical_intro")
            if str(item.get(key) or "").strip()
        )
        minimum_chars = 160
        maximum_chars = 500
        if not paper_plain_summary_passes(item.get("paper_plain_summary")):
            issues.append("paper_plain_summary_failed")
        if not paper_technical_intro_passes(item.get("paper_technical_intro")):
            issues.append("paper_technical_intro_failed")
    else:
        body = str(item.get("analysis_body") or item.get("summary") or "").strip()
        content_type = str(item.get("content_type") or "").strip().lower()
        minimum_chars = 300 if content_type in {"interview", "podcast", "video"} else (220 if section == "technical" else 180)
        maximum_chars = 500 if content_type in {"interview", "podcast", "video"} else (380 if section == "technical" else 300)
        if (
            content_type in {"interview", "podcast", "video"}
            and not re.search(r"\d{1,2}:\d{2}|文字稿|transcript|章节", evidence_locator, re.IGNORECASE)
        ):
            issues.append("interview_locator_imprecise")
    compact_body = re.sub(r"\s+", "", body)
    if len(compact_body) < minimum_chars:
        issues.append(f"body_too_short={len(compact_body)}/{minimum_chars}")
    if len(compact_body) > maximum_chars:
        issues.append(f"body_too_long={len(compact_body)}/{maximum_chars}")
    if not evidence:
        issues.append("evidence_points_missing")
    if len(re.findall(r"[。！？.!?]", body)) < 2:
        issues.append("body_lacks_sentence_structure")
    matched_phrases = [phrase for phrase in V11_GENERIC_PHRASES if phrase in body]
    if matched_phrases:
        issues.append(f"generic_phrase={matched_phrases[0]}")
    return issues


def _v11_editorial_review_result(
    review: Dict[str, Any],
    *,
    report_id: str,
    slot_id: str,
    expected_samples: list[Dict[str, Any]],
    require_client_rendering: bool = False,
) -> Dict[str, Any]:
    issues: list[str] = []
    review_samples = list(review.get("samples") or []) if isinstance(review, dict) else []
    if not review:
        issues.append("editorial_review_missing")
    if int(review.get("schema_version", 0) or 0) < 1:
        issues.append("editorial_review_schema_invalid")
    if str(review.get("report_id") or "") != report_id:
        issues.append("editorial_review_report_mismatch")
    if str(review.get("slot_id") or "") != slot_id:
        issues.append("editorial_review_slot_mismatch")
    if str(review.get("status") or "") != "passed":
        issues.append(f"editorial_review_status={review.get('status') or 'missing'}")
    if not str(review.get("reviewer") or "").strip():
        issues.append("editorial_review_reviewer_missing")
    reviewed_at = str(review.get("reviewed_at") or "").strip()
    if _parse_status_datetime(reviewed_at) is None:
        issues.append("editorial_review_timestamp_missing")

    client_rendering = (
        dict(review.get("client_rendering") or {})
        if isinstance(review, dict)
        else {}
    )
    if require_client_rendering:
        failed_client_checks = [
            key
            for key in ("qq_desktop", "qq_mobile", "no_clipping", "spacing_readable")
            if client_rendering.get(key) is not True
        ]
        if failed_client_checks:
            issues.append("client_rendering_failed=" + ",".join(failed_client_checks))
        if len(str(client_rendering.get("notes") or "").strip()) < 8:
            issues.append("client_rendering_notes_missing")

    review_by_key = {
        (
            str(sample.get("section") or "").strip(),
            str(sample.get("url") or "").strip(),
        ): sample
        for sample in review_samples
        if isinstance(sample, dict)
    }
    section_counts = {section: 0 for section in ("news", "technical", "paper")}
    passed_sample_count = 0
    for expected in expected_samples:
        section = str(expected.get("section") or "").strip()
        url = str(expected.get("url") or "").strip()
        sample = review_by_key.get((section, url))
        if not sample:
            issues.append(f"editorial_review_sample_missing={section}:{url}")
            continue
        section_counts[section] = section_counts.get(section, 0) + 1
        failed_dimensions = [
            dimension
            for dimension in ("accuracy", "specificity", "readability")
            if sample.get(dimension) is not True
        ]
        if failed_dimensions:
            issues.append(
                f"editorial_review_sample_failed={section}:{url}:"
                + ",".join(failed_dimensions)
            )
            continue
        if len(str(sample.get("notes") or "").strip()) < 8:
            issues.append(f"editorial_review_notes_missing={section}:{url}")
            continue
        passed_sample_count += 1

    return {
        "status": "passed" if not issues else "failed",
        "reviewer": str(review.get("reviewer") or ""),
        "reviewed_at": reviewed_at,
        "sample_count": len(review_samples),
        "expected_sample_count": len(expected_samples),
        "passed_sample_count": passed_sample_count,
        "section_counts": section_counts,
        "client_rendering_required": require_client_rendering,
        "client_rendering_passed": not require_client_rendering or not any(
            issue.startswith("client_rendering_") for issue in issues
        ),
        "issues": issues,
    }


def build_v11_production_acceptance(
    scheduler_config: Dict[str, Any],
    *,
    required_days: int = 3,
    reports_per_day: int = 2,
    minimum_news: int = 20,
    minimum_technical: int = 20,
    minimum_papers: int = 15,
    minimum_discovery_candidates: int = 160,
    minimum_discovered_by_section: Optional[Dict[str, int]] = None,
    technical_primary_source_ratio_min: float = 0.80,
    overlap_max: float = 0.10,
    sample_size: int = 3,
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    required_days = max(1, int(required_days or 3))
    reports_per_day = max(1, int(reports_per_day or 2))
    sample_size = max(1, int(sample_size or 3))
    technical_primary_source_ratio_min = max(
        0.0,
        min(1.0, float(technical_primary_source_ratio_min or 0.80)),
    )
    minimum_discovery_candidates = max(0, int(minimum_discovery_candidates or 0))
    minimum_discovered_by_section = {
        section: max(0, int(minimum or 0))
        for section, minimum in dict(
            minimum_discovered_by_section
            or {"news": 50, "technical": 50, "paper": 40}
        ).items()
        if section in {"news", "technical", "paper"}
    }
    required_report_count = required_days * reports_per_day
    slot_dir = Path(str(scheduler_config.get("send_slot_dir", ROOT / "logs/send_slots")))
    if not slot_dir.is_absolute():
        slot_dir = ROOT / slot_dir
    configured_slot_ids = [
        str(slot.get("id") or "").strip()
        for slot in scheduler_config.get("send_slots", DEFAULT_SCHEDULER_CONFIG["send_slots"])
        if str(slot.get("id") or "").strip()
    ][:reports_per_day]
    if len(configured_slot_ids) < reports_per_day:
        configured_slot_ids = ["1300", "2100"][:reports_per_day]

    sent_slots = []
    for path in slot_dir.glob("*.json") if slot_dir.exists() else []:
        payload = read_json(path)
        slot_id = str(payload.get("slot_id") or path.stem or "").strip()
        if str(payload.get("status") or "") != "sent" or slot_id in {"", "__report_only__", "__dry_run__"}:
            continue
        finished_at = _parse_status_datetime(payload.get("finished_at"))
        if finished_at is None:
            continue
        payload["_finished_at"] = finished_at
        payload["_slot_id"] = slot_id
        sent_slots.append(payload)
    sent_slots.sort(key=lambda item: item["_finished_at"])

    reports: list[Dict[str, Any]] = []
    seen_report_ids: set[str] = set()
    conn = None
    try:
        conn = sqlite3.connect(str(db_path or resolve_database_path(None, ROOT)))
        conn.row_factory = sqlite3.Row
        for slot in sent_slots:
            run_id = str(slot.get("run_id") or "")
            html_path = str(slot.get("html_report_path") or "")
            row = (
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
            if row is None:
                continue
            report = dict(row)
            report_id = str(report.get("report_id") or "")
            report_html = str(report.get("html_report_path") or "").replace("\\", "/")
            if (
                not report_id
                or report_id in seen_report_ids
                or str(report.get("delivery_status") or "") != "sent"
                or "/validation/" in report_html
            ):
                continue
            try:
                diagnostics = json.loads(report.get("quality_diagnostics") or "{}")
            except (TypeError, json.JSONDecodeError):
                diagnostics = {}
            if str(diagnostics.get("report_product_mode") or "") != V11_PRODUCT_MODE:
                continue
            if str(diagnostics.get("report_design_version") or "") != V11_DESIGN_VERSION:
                continue
            item_rows = conn.execute(
                "SELECT rank, section, snapshot_json FROM report_items WHERE report_id = ? ORDER BY rank",
                (report_id,),
            ).fetchall()
            items = []
            for item_row in item_rows:
                try:
                    item = json.loads(item_row["snapshot_json"] or "{}")
                except (TypeError, json.JSONDecodeError):
                    continue
                item.setdefault("report_rank", item_row["rank"])
                item.setdefault("report_section", item_row["section"])
                items.append(item)
            report["_items"] = items
            report["_diagnostics"] = diagnostics
            report["_finished_at"] = slot["_finished_at"]
            report["_slot_id"] = str(report.get("slot_id") or slot["_slot_id"])
            report["_slot_payload"] = slot
            reports.append(report)
            seen_report_ids.add(report_id)
    except (OSError, sqlite3.Error) as exc:
        return {
            "status": "pending",
            "required_days": required_days,
            "required_report_count": required_report_count,
            "verified_days": 0,
            "verified_report_count": 0,
            "reports": [],
            "error": str(exc),
        }
    finally:
        if conn is not None:
            conn.close()

    contract_start_index = next(
        (
            index
            for index, report in enumerate(reports)
            if int(
                report.get("_diagnostics", {}).get("v11_acceptance_contract_version", 0)
                or 0
            ) >= V11_ACCEPTANCE_CONTRACT_VERSION
        ),
        None,
    )
    pre_contract_report_count = (
        len(reports) if contract_start_index is None else contract_start_index
    )
    contract_started_at = ""
    if contract_start_index is None:
        reports = []
    else:
        reports = reports[contract_start_index:]
        contract_started_at = reports[0]["_finished_at"].isoformat()

    available_days = sorted({report["_finished_at"].date() for report in reports}, reverse=True)
    consecutive_days = []
    if available_days:
        available_set = set(available_days)
        day = available_days[0]
        while day in available_set and len(consecutive_days) < required_days:
            consecutive_days.append(day)
            day -= timedelta(days=1)
    selected_day_strings = {day.isoformat() for day in consecutive_days}
    selected_reports = [
        report for report in reports if report["_finished_at"].date().isoformat() in selected_day_strings
    ]
    selected_reports.sort(key=lambda report: report["_finished_at"])

    rows = []
    previous_papers: list[Dict[str, Any]] = []
    paper_history: list[tuple[datetime, Dict[str, Any]]] = []
    for report in selected_reports:
        items = report["_items"]
        by_section = {
            section: [item for item in items if _v11_primary_section(item) == section]
            for section in ("news", "technical", "paper")
        }
        technical_primary_source_count = sum(
            1
            for item in by_section["technical"]
            if infer_source_tier(item).lower()
            in {"official", "research", "primary"}
        )
        technical_primary_source_ratio = (
            round(
                technical_primary_source_count / len(by_section["technical"]),
                3,
            )
            if by_section["technical"]
            else 0.0
        )
        identities = [_v11_item_identity(item) for item in items if _v11_item_identity(item)]
        duplicate_count = len(identities) - len(set(identities))
        event_identities = [
            CodexResearchInboxCollector._event_identity(item)
            for item in items
            if CodexResearchInboxCollector._event_identity(item)
        ]
        duplicate_event_count = len(event_identities) - len(set(event_identities))
        within_report_paper_duplicates = []
        for index, paper in enumerate(by_section["paper"]):
            if any(_paper_snapshots_match(paper, prior) for prior in by_section["paper"][:index]):
                within_report_paper_duplicates.append(_paper_snapshot_key(paper))
        overlap_items = [
            paper
            for paper in by_section["paper"]
            if any(_paper_snapshots_match(paper, prior) for prior in previous_papers)
        ]
        overlap_rate = len(overlap_items) / max(1, len(by_section["paper"]))
        cooldown_items = [
            prior
            for timestamp, prior in paper_history
            if report["_finished_at"] - timestamp <= timedelta(days=7)
        ]
        unjustified = []
        for paper in by_section["paper"]:
            matches = [prior for prior in cooldown_items if _paper_snapshots_match(paper, prior)]
            if matches and not _paper_reappearance_is_supported(paper, matches):
                unjustified.append(_paper_snapshot_key(paper))

        sample_rows = []
        for section in ("news", "technical", "paper"):
            for item in by_section[section][:sample_size]:
                sample_issues = _v11_sample_quality(item, section)
                sample_rows.append({
                    "article_id": item.get("id"),
                    "section": section,
                    "title": str(item.get("title_cn") or item.get("title") or ""),
                    "url": str(item.get("canonical_url") or item.get("url") or ""),
                    "publish_date": str(item.get("publish_date") or ""),
                    "claim_type": str(
                        item.get("claim_type")
                        or (item.get("facts") or {}).get("claim_type")
                        or ""
                    ),
                    "body_chars": len(
                        re.sub(
                            r"\s+",
                            "",
                            " ".join(
                                str(item.get(key) or "")
                                for key in (
                                    ("paper_plain_summary", "paper_technical_intro")
                                    if section == "paper"
                                    else ("analysis_body",)
                                )
                            ),
                        )
                    ),
                    "source_evidence_present": bool(
                        str(
                            item.get("source_excerpt")
                            or (item.get("facts") or {}).get("source_excerpt")
                            or ""
                        ).strip()
                        and str(
                            item.get("evidence_locator")
                            or (item.get("facts") or {}).get("evidence_locator")
                            or ""
                        ).strip()
                    ),
                    "passed": not sample_issues,
                    "issues": sample_issues,
                })
        sample_counts = {
            section: sum(1 for sample in sample_rows if sample["section"] == section)
            for section in ("news", "technical", "paper")
        }
        issues = []
        minimums = {"news": minimum_news, "technical": minimum_technical, "paper": minimum_papers}
        for section, minimum in minimums.items():
            if len(by_section[section]) < minimum:
                issues.append(f"{section}_count_below_min={len(by_section[section])}/{minimum}")
            if sample_counts[section] < sample_size:
                issues.append(f"{section}_sample_underfilled={sample_counts[section]}/{sample_size}")
        if duplicate_count:
            issues.append(f"cross_section_duplicates={duplicate_count}")
        if duplicate_event_count:
            issues.append(f"cross_section_event_duplicates={duplicate_event_count}")
        if within_report_paper_duplicates:
            issues.append(f"paper_within_report_duplicates={len(within_report_paper_duplicates)}")
        if previous_papers and overlap_rate >= overlap_max:
            issues.append(f"adjacent_paper_overlap={overlap_rate:.3f}")
        if unjustified:
            issues.append(f"unjustified_7d_repeats={len(set(unjustified))}")
        failed_samples = [sample for sample in sample_rows if not sample["passed"]]
        if failed_samples:
            issues.append(f"sample_quality_failures={len(failed_samples)}")
        if technical_primary_source_ratio < technical_primary_source_ratio_min:
            issues.append(
                "technical_primary_source_ratio="
                f"{technical_primary_source_ratio:.3f}/{technical_primary_source_ratio_min:.3f}"
            )
        editorial_review_required = bool(
            scheduler_config.get("v11_editorial_review_required", False)
        )
        editorial_review_path = ""
        editorial_review_result = {
            "status": "disabled",
            "sample_count": 0,
            "expected_sample_count": len(sample_rows),
            "passed_sample_count": 0,
            "section_counts": {},
            "issues": [],
        }
        if editorial_review_required:
            review_dir = Path(
                str(
                    scheduler_config.get("v11_editorial_review_dir")
                    or ROOT / "data/v11_editorial_reviews"
                )
            )
            if not review_dir.is_absolute():
                review_dir = ROOT / review_dir
            review_path = review_dir / f"{report['_slot_id']}.json"
            editorial_review_path = review_path.as_posix()
            editorial_review_result = _v11_editorial_review_result(
                read_json(review_path),
                report_id=str(report.get("report_id") or ""),
                slot_id=str(report.get("_slot_id") or ""),
                expected_samples=sample_rows,
                require_client_rendering=bool(
                    scheduler_config.get("v11_client_render_review_required", False)
                ),
            )
            issues.extend(editorial_review_result["issues"])
        if str(report.get("quality_status") or "") != "passed":
            issues.append(f"report_quality_status={report.get('quality_status') or 'unknown'}")
        quality_gate = dict(report.get("_diagnostics", {}).get("quality_gate") or {})
        final_html_quality_status = str(
            quality_gate.get("final_html_quality_status") or "missing"
        )
        inline_style_count = int(quality_gate.get("v11_inline_style_count", 0) or 0)
        inline_style_required_count = len(items) * 3 + 10
        generic_phrase_count = int(quality_gate.get("generic_phrase_count", 0) or 0)
        mixed_language_title_count = int(
            quality_gate.get("mixed_language_title_count", 0) or 0
        )
        attribution_opener_repeat_count = int(
            quality_gate.get("max_attribution_opener_repeat_count", 0) or 0
        )
        attribution_opener_run = int(
            quality_gate.get("max_attribution_opener_run", 0) or 0
        )
        supplemental_expected_count = max(
            int(quality_gate.get("v11_supplemental_expected_count", 0) or 0),
            sum(
                1
                for item in items
                if "supplemental_older_source" in set(item.get("quality_flags") or [])
            ),
        )
        supplemental_label_count = int(
            quality_gate.get("v11_supplemental_label_count", 0) or 0
        )
        supplemental_items = [
            item
            for item in items
            if "supplemental_older_source" in set(item.get("quality_flags") or [])
        ]
        supplemental_counts_by_section: Dict[str, int] = {}
        for item in supplemental_items:
            section = _v11_primary_section(item)
            supplemental_counts_by_section[section] = (
                supplemental_counts_by_section.get(section, 0) + 1
            )
        supplemental_limits = {"news": 5, "technical": 5, "paper": 3}
        supplemental_limit_exceeded = {
            section: {"count": count, "max": supplemental_limits[section]}
            for section, count in supplemental_counts_by_section.items()
            if section in supplemental_limits and count > supplemental_limits[section]
        }
        supplemental_max_age_days = {"news": 7, "technical": 30, "paper": 30}
        supplemental_age_violations = []
        for item in supplemental_items:
            section = _v11_primary_section(item)
            published_at = _parse_status_datetime(item.get("publish_date"))
            if published_at is None or section not in supplemental_max_age_days:
                continue
            age_days = (report["_finished_at"] - published_at).total_seconds() / 86400
            if age_days > supplemental_max_age_days[section]:
                supplemental_age_violations.append({
                    "section": section,
                    "url": str(item.get("canonical_url") or item.get("url") or ""),
                    "publish_date": str(item.get("publish_date") or ""),
                    "age_days": round(age_days, 2),
                    "max_age_days": supplemental_max_age_days[section],
                })
        content_fidelity_missing_count = int(
            quality_gate.get("v11_content_fidelity_missing_count", 0) or 0
        )
        claim_label_expected_count = int(
            quality_gate.get("v11_claim_label_expected_count", 0) or 0
        )
        claim_label_visible_count = int(
            quality_gate.get("v11_claim_label_visible_count", 0) or 0
        )
        if final_html_quality_status != "passed":
            issues.append(f"final_html_quality_status={final_html_quality_status}")
        if inline_style_count < inline_style_required_count:
            issues.append(
                f"v11_inline_style_count={inline_style_count}/{inline_style_required_count}"
            )
        if generic_phrase_count:
            issues.append(f"generic_phrase_count={generic_phrase_count}")
        if mixed_language_title_count:
            issues.append(f"mixed_language_title_count={mixed_language_title_count}")
        if "max_attribution_opener_repeat_count" not in quality_gate:
            issues.append("max_attribution_opener_repeat_count=missing")
        elif attribution_opener_repeat_count > 8:
            issues.append(
                "max_attribution_opener_repeat_count="
                f"{attribution_opener_repeat_count}/8"
            )
        if "max_attribution_opener_run" not in quality_gate:
            issues.append("max_attribution_opener_run=missing")
        elif attribution_opener_run > 1:
            issues.append(f"max_attribution_opener_run={attribution_opener_run}/1")
        if supplemental_label_count < supplemental_expected_count:
            issues.append(
                "v11_supplemental_label_count="
                f"{supplemental_label_count}/{supplemental_expected_count}"
            )
        if supplemental_limit_exceeded:
            issues.append("v11_supplemental_limit_exceeded")
        if supplemental_age_violations:
            issues.append(
                "v11_supplemental_age_violation_count="
                f"{len(supplemental_age_violations)}"
            )
        if content_fidelity_missing_count:
            issues.append(
                f"v11_content_fidelity_missing_count={content_fidelity_missing_count}"
            )
        if claim_label_expected_count != len(items):
            issues.append(
                f"v11_claim_label_expected_count={claim_label_expected_count}/{len(items)}"
            )
        if claim_label_visible_count != claim_label_expected_count:
            issues.append(
                "v11_claim_label_visible_count="
                f"{claim_label_visible_count}/{claim_label_expected_count}"
            )
        research_inbox = dict(report.get("_diagnostics", {}).get("codex_research_inbox") or {})
        research_schema_version = str(research_inbox.get("schema_version") or "")
        if research_schema_version != "codex-research-v3":
            issues.append(
                f"codex_research_schema_version={research_schema_version or 'missing'}"
            )
        sent_history_overlap_count = int(
            research_inbox.get("sent_history_overlap_count", 0) or 0
        )
        sent_history_overlap_by_section = dict(
            research_inbox.get("sent_history_overlap_by_section") or {}
        )
        if str(research_inbox.get("quality_status") or "missing") != "passed":
            issues.append(f"codex_research_quality_status={research_inbox.get('quality_status') or 'missing'}")
        if "attribution_opener_overuse_count" not in research_inbox:
            issues.append("codex_research_attribution_opener_overuse_count=missing")
        elif int(research_inbox.get("attribution_opener_overuse_count", 0) or 0):
            issues.append(
                "codex_research_attribution_opener_overuse_count="
                f"{int(research_inbox.get('attribution_opener_overuse_count', 0) or 0)}"
            )
        if "cross_item_template_repeat_count" not in research_inbox:
            issues.append("codex_research_cross_item_template_repeat_count=missing")
        elif int(research_inbox.get("cross_item_template_repeat_count", 0) or 0):
            issues.append(
                "codex_research_cross_item_template_repeat_count="
                f"{int(research_inbox.get('cross_item_template_repeat_count', 0) or 0)}"
            )
        if sent_history_overlap_count:
            issues.append(
                f"sent_history_overlap_count={sent_history_overlap_count}"
            )
        for metric in (
            "discovery_quota_status",
            "submission_quota_status",
            "technical_quota_status",
            "news_format_quota_status",
            "paper_domain_quota_status",
            "freshness_quota_status",
            "key_number_quota_status",
        ):
            if str(research_inbox.get(metric) or "missing") != "passed":
                issues.append(f"codex_research_{metric}={research_inbox.get(metric) or 'missing'}")
        discovery_candidate_count = int(
            research_inbox.get("discovery_candidate_count", 0) or 0
        )
        discovery_section_counts = dict(
            research_inbox.get("discovery_section_counts") or {}
        )
        inbox_sha256 = str(research_inbox.get("inbox_sha256") or "")
        discovery_manifest_sha256 = str(
            research_inbox.get("discovery_manifest_sha256") or ""
        )
        if not re.fullmatch(r"[0-9a-f]{64}", inbox_sha256):
            issues.append("codex_research_inbox_sha256_missing")
        if not re.fullmatch(r"[0-9a-f]{64}", discovery_manifest_sha256):
            issues.append("codex_research_discovery_manifest_sha256_missing")
        if discovery_candidate_count < minimum_discovery_candidates:
            issues.append(
                "codex_research_discovery_candidate_count="
                f"{discovery_candidate_count}/{minimum_discovery_candidates}"
            )
        for section, minimum in minimum_discovered_by_section.items():
            count = int(discovery_section_counts.get(section, 0) or 0)
            if count < minimum:
                issues.append(
                    f"codex_research_discovery_{section}_count={count}/{minimum}"
                )
        for metric in (
            "discovery_duplicate_url_count",
            "discovery_invalid_row_count",
            "submitted_not_in_discovery_count",
            "key_number_contract_missing_count",
            "key_number_evidence_missing_count",
            "key_number_public_copy_missing_count",
        ):
            count = int(research_inbox.get(metric, 0) or 0)
            if count:
                issues.append(f"codex_research_{metric}={count}")
        research_technical_primary_ratio = float(
            research_inbox.get("technical_primary_source_ratio", 0.0) or 0.0
        )
        if research_technical_primary_ratio < technical_primary_source_ratio_min:
            issues.append(
                "codex_research_technical_primary_source_ratio="
                f"{research_technical_primary_ratio:.3f}/{technical_primary_source_ratio_min:.3f}"
            )
        if bool(quality_gate.get("email_clipping_risk", False)):
            issues.append(f"email_clipping_risk={int(quality_gate.get('html_size_bytes', 0) or 0)}")
        slot_payload = dict(report.get("_slot_payload") or {})
        email_split_applied = bool(quality_gate.get("email_split_applied", False))
        email_volume_count = int(quality_gate.get("email_delivery_volume_count", 1) or 1)
        email_volume_sizes = [
            int(value or 0) for value in (quality_gate.get("email_delivery_volume_sizes") or [])
        ]
        volume_editorial_decision_counts = [
            int(value or 0)
            for value in (
                quality_gate.get("email_delivery_volume_editorial_decision_counts") or []
            )
        ]
        volume_editorial_decision_visible_source_counts = [
            int(value or 0)
            for value in (
                quality_gate.get("email_delivery_volume_editorial_decision_visible_source_counts") or []
            )
        ]
        volume_editorial_decision_source_missing_counts = [
            int(value or 0)
            for value in (
                quality_gate.get("email_delivery_volume_editorial_decision_source_missing_counts") or []
            )
        ]
        volume_editorial_decision_duplicate_source_counts = [
            int(value or 0)
            for value in (
                quality_gate.get("email_delivery_volume_editorial_decision_duplicate_source_counts") or []
            )
        ]
        volume_claim_label_counts = [
            int(value or 0)
            for value in (
                quality_gate.get("email_delivery_volume_claim_label_counts") or []
            )
        ]
        volume_item_counts = [
            int(value or 0)
            for value in (
                quality_gate.get("email_delivery_volume_item_counts") or []
            )
        ]
        volume_nav_mismatch_count = int(
            quality_gate.get("email_delivery_volume_nav_mismatch_count", 0) or 0
        )
        volume_preheader_mismatch_count = int(
            quality_gate.get("email_delivery_volume_preheader_mismatch_count", 0) or 0
        )
        volume_content_fidelity_missing_counts = [
            int(value or 0)
            for value in (
                quality_gate.get(
                    "email_delivery_volume_content_fidelity_missing_counts"
                )
                or []
            )
        ]
        volume_content_fidelity_missing_count = int(
            quality_gate.get(
                "email_delivery_volume_content_fidelity_missing_count",
                sum(volume_content_fidelity_missing_counts),
            )
            or 0
        )
        volume_key_number_fidelity_missing_counts = [
            int(value or 0)
            for value in (
                quality_gate.get(
                    "email_delivery_volume_key_number_fidelity_missing_counts"
                )
                or []
            )
        ]
        volume_key_number_fidelity_missing_count = int(
            quality_gate.get(
                "email_delivery_volume_key_number_fidelity_missing_count",
                sum(volume_key_number_fidelity_missing_counts),
            )
            or 0
        )
        email_volume_paths = [
            str(value or "").strip()
            for value in (
                quality_gate.get("email_delivery_volume_paths")
                or slot_payload.get("email_volume_paths")
                or []
            )
            if str(value or "").strip()
        ]
        resolved_volume_paths = [
            path if path.is_absolute() else ROOT / path
            for path in (Path(value) for value in email_volume_paths)
        ]
        archived_volume_sizes = [
            path.stat().st_size if path.exists() and path.is_file() else 0
            for path in resolved_volume_paths
        ]
        if email_split_applied and email_volume_count < 2:
            issues.append(f"email_split_volume_count={email_volume_count}")
        if any(size > 104448 for size in email_volume_sizes):
            issues.append(f"email_volume_oversize={max(email_volume_sizes)}")
        if len(email_volume_paths) != email_volume_count:
            issues.append(
                f"email_volume_archive_count={len(email_volume_paths)}/{email_volume_count}"
            )
        elif any(size <= 0 for size in archived_volume_sizes):
            issues.append("email_volume_archive_missing")
        elif (
            len(email_volume_sizes) == email_volume_count
            and archived_volume_sizes != email_volume_sizes
        ):
            issues.append(
                "email_volume_archive_size_mismatch="
                + ",".join(str(size) for size in archived_volume_sizes)
                + "/"
                + ",".join(str(size) for size in email_volume_sizes)
            )
        if len(volume_editorial_decision_counts) != email_volume_count:
            issues.append(
                "email_volume_editorial_decision_count_rows="
                f"{len(volume_editorial_decision_counts)}/{email_volume_count}"
            )
        elif any(not 5 <= count <= 7 for count in volume_editorial_decision_counts):
            issues.append(
                "email_volume_editorial_decision_counts="
                + ",".join(str(count) for count in volume_editorial_decision_counts)
            )
        if volume_editorial_decision_visible_source_counts != volume_editorial_decision_counts:
            issues.append(
                "email_volume_editorial_decision_visible_sources="
                + ",".join(str(count) for count in volume_editorial_decision_visible_source_counts)
                + "/"
                + ",".join(str(count) for count in volume_editorial_decision_counts)
            )
        if any(volume_editorial_decision_source_missing_counts):
            issues.append(
                "email_volume_editorial_decision_source_missing_counts="
                + ",".join(str(count) for count in volume_editorial_decision_source_missing_counts)
            )
        if any(volume_editorial_decision_duplicate_source_counts):
            issues.append(
                "email_volume_editorial_decision_duplicate_source_counts="
                + ",".join(str(count) for count in volume_editorial_decision_duplicate_source_counts)
            )
        if (
            len(volume_claim_label_counts) != email_volume_count
            or len(volume_item_counts) != email_volume_count
        ):
            issues.append(
                "email_volume_claim_label_rows="
                f"{len(volume_claim_label_counts)}/{len(volume_item_counts)}/{email_volume_count}"
            )
        elif volume_claim_label_counts != volume_item_counts:
            issues.append(
                "email_volume_claim_label_counts="
                + ",".join(str(count) for count in volume_claim_label_counts)
                + "/"
                + ",".join(str(count) for count in volume_item_counts)
            )
        if volume_nav_mismatch_count:
            issues.append(f"email_delivery_volume_nav_mismatch_count={volume_nav_mismatch_count}")
        if volume_preheader_mismatch_count:
            issues.append(
                "email_delivery_volume_preheader_mismatch_count="
                f"{volume_preheader_mismatch_count}"
            )
        if volume_content_fidelity_missing_count:
            issues.append(
                "email_delivery_volume_content_fidelity_missing_count="
                f"{volume_content_fidelity_missing_count}"
            )
        if volume_key_number_fidelity_missing_count:
            issues.append(
                "email_delivery_volume_key_number_fidelity_missing_count="
                f"{volume_key_number_fidelity_missing_count}"
            )
        sent_volume_count = int(slot_payload.get("email_volume_sent_count", 0) or 0)
        expected_subjects = [str(value or "") for value in (slot_payload.get("email_subjects") or [])]
        delivery_verification = dict(slot_payload.get("delivery_verification") or {})
        delivery_arrival_status = str(delivery_verification.get("status") or "missing")
        if sent_volume_count != email_volume_count:
            issues.append(f"email_volume_sent_count={sent_volume_count}/{email_volume_count}")
        if delivery_arrival_status != "found":
            issues.append(f"delivery_arrival_status={delivery_arrival_status}")
        verification_rows = list(delivery_verification.get("volumes") or [])
        if not verification_rows and delivery_verification:
            verification_rows = [delivery_verification]
        matched_subjects = [str(value.get("matched_subject") or "") for value in verification_rows]
        if len(verification_rows) != email_volume_count:
            issues.append(f"delivery_verified_volume_count={len(verification_rows)}/{email_volume_count}")
        elif expected_subjects and (
            len(set(matched_subjects)) != email_volume_count
            or set(matched_subjects) != set(expected_subjects)
        ):
            issues.append("delivery_volume_subject_mismatch")
        post_send_scan = dict(slot_payload.get("post_send_quality_scan") or {})
        post_send_focus_issue_count = int(post_send_scan.get("focus_issue_count", 0) or 0)
        if post_send_focus_issue_count:
            issues.append(f"post_send_focus_issue_count={post_send_focus_issue_count}")
        ui_audit = dict(slot_payload.get("ui_audit") or {})
        ui_audit_status = str(ui_audit.get("status") or "missing")
        ui_audit_render_count = int(ui_audit.get("render_count", 0) or 0)
        ui_audit_failed_render_count = int(ui_audit.get("failed_render_count", 0) or 0)
        expected_ui_audit_render_count = email_volume_count * 4
        if ui_audit_status != "passed":
            issues.append(f"ui_audit_status={ui_audit_status}")
        if ui_audit_render_count != expected_ui_audit_render_count:
            issues.append(
                f"ui_audit_render_count={ui_audit_render_count}/{expected_ui_audit_render_count}"
            )
        if ui_audit_failed_render_count:
            issues.append(f"ui_audit_failed_render_count={ui_audit_failed_render_count}")
        if str(quality_gate.get("paper_domain_quota_status") or "passed") == "failed":
            issues.append("paper_domain_quota_status=failed")
        editorial_decision_count = int(quality_gate.get("editorial_decision_count", 0) or 0)
        if not 5 <= editorial_decision_count <= 7:
            issues.append(f"editorial_decision_count={editorial_decision_count}/5-7")
        for metric in (
            "editorial_decision_source_missing_count",
            "editorial_decision_duplicate_source_count",
        ):
            if int(quality_gate.get(metric, 0) or 0) > 0:
                issues.append(f"{metric}={int(quality_gate.get(metric, 0) or 0)}")
        for metric in (
            "body_under_min_count",
            "publish_date_missing_count",
            "source_evidence_missing_count",
            "claim_type_missing_count",
            "paper_full_text_missing_count",
            "analysis_version_mismatch_count",
            "v11_external_item_count",
            "v11_key_number_fidelity_missing_count",
            "v11_editorial_source_hash_missing_count",
            "v11_editorial_source_mismatch_count",
        ):
            if int(quality_gate.get(metric, 0) or 0) > 0:
                issues.append(f"{metric}={int(quality_gate.get(metric, 0) or 0)}")
        rows.append({
            "date": report["_finished_at"].date().isoformat(),
            "slot_id": report["_slot_id"],
            "email_split_applied": email_split_applied,
            "email_delivery_volume_count": email_volume_count,
            "email_delivery_volume_sizes": email_volume_sizes,
            "email_delivery_volume_paths": email_volume_paths,
            "email_delivery_volume_archive_sizes": archived_volume_sizes,
            "email_delivery_volume_editorial_decision_counts": volume_editorial_decision_counts,
            "email_delivery_volume_editorial_decision_visible_source_counts": volume_editorial_decision_visible_source_counts,
            "email_delivery_volume_editorial_decision_source_missing_counts": volume_editorial_decision_source_missing_counts,
            "email_delivery_volume_editorial_decision_duplicate_source_counts": volume_editorial_decision_duplicate_source_counts,
            "email_delivery_volume_claim_label_counts": volume_claim_label_counts,
            "email_delivery_volume_item_counts": volume_item_counts,
            "email_delivery_volume_nav_mismatch_count": volume_nav_mismatch_count,
            "email_delivery_volume_preheader_mismatch_count": volume_preheader_mismatch_count,
            "email_delivery_volume_content_fidelity_missing_counts": (
                volume_content_fidelity_missing_counts
            ),
            "email_delivery_volume_content_fidelity_missing_count": (
                volume_content_fidelity_missing_count
            ),
            "email_delivery_volume_key_number_fidelity_missing_counts": (
                volume_key_number_fidelity_missing_counts
            ),
            "email_delivery_volume_key_number_fidelity_missing_count": (
                volume_key_number_fidelity_missing_count
            ),
            "email_volume_sent_count": sent_volume_count,
            "delivery_arrival_status": delivery_arrival_status,
            "delivery_verified_volume_count": len(verification_rows),
            "post_send_focus_issue_count": post_send_focus_issue_count,
            "ui_audit_status": ui_audit_status,
            "ui_audit_render_count": ui_audit_render_count,
            "ui_audit_expected_render_count": expected_ui_audit_render_count,
            "ui_audit_failed_render_count": ui_audit_failed_render_count,
            "finished_at": report["_finished_at"].isoformat(),
            "report_id": report.get("report_id", ""),
            "html_report_path": report.get("html_report_path", ""),
            "news_count": len(by_section["news"]),
            "technical_count": len(by_section["technical"]),
            "technical_primary_source_count": technical_primary_source_count,
            "technical_primary_source_ratio": technical_primary_source_ratio,
            "paper_count": len(by_section["paper"]),
            "cross_section_duplicate_count": duplicate_count,
            "cross_section_event_duplicate_count": duplicate_event_count,
            "paper_within_report_duplicate_count": len(within_report_paper_duplicates),
            "adjacent_report_paper_overlap_rate": round(overlap_rate, 3),
            "unjustified_7d_repeat_count": len(set(unjustified)),
            "html_size_bytes": int(quality_gate.get("html_size_bytes", 0) or 0),
            "email_clipping_warning": bool(quality_gate.get("email_clipping_warning", False)),
            "email_clipping_risk": bool(quality_gate.get("email_clipping_risk", False)),
            "final_html_quality_status": final_html_quality_status,
            "v11_inline_style_count": inline_style_count,
            "v11_inline_style_required_count": inline_style_required_count,
            "generic_phrase_count": generic_phrase_count,
            "mixed_language_title_count": mixed_language_title_count,
            "v11_supplemental_expected_count": supplemental_expected_count,
            "v11_supplemental_label_count": supplemental_label_count,
            "v11_supplemental_counts_by_section": supplemental_counts_by_section,
            "v11_supplemental_limit_exceeded": supplemental_limit_exceeded,
            "v11_supplemental_age_violation_count": len(supplemental_age_violations),
            "v11_supplemental_age_violation_examples": supplemental_age_violations[:5],
            "v11_content_fidelity_missing_count": content_fidelity_missing_count,
            "v11_claim_label_expected_count": claim_label_expected_count,
            "v11_claim_label_visible_count": claim_label_visible_count,
            "v11_external_item_count": int(
                quality_gate.get("v11_external_item_count", 0) or 0
            ),
            "v11_editorial_source_hash_missing_count": int(
                quality_gate.get("v11_editorial_source_hash_missing_count", 0) or 0
            ),
            "v11_editorial_source_mismatch_count": int(
                quality_gate.get("v11_editorial_source_mismatch_count", 0) or 0
            ),
            "sample_quality_pass_count": sum(1 for sample in sample_rows if sample["passed"]),
            "sample_quality_required_count": sample_size * 3,
            "sample_quality": sample_rows,
            "editorial_review_required": editorial_review_required,
            "editorial_review_path": editorial_review_path,
            "editorial_review_status": editorial_review_result["status"],
            "editorial_review_pass_count": editorial_review_result["passed_sample_count"],
            "editorial_review_required_count": editorial_review_result["expected_sample_count"],
            "editorial_review": editorial_review_result,
            "codex_research_quality_status": str(research_inbox.get("quality_status") or "missing"),
            "codex_research_discovery_quota_status": str(
                research_inbox.get("discovery_quota_status") or "missing"
            ),
            "codex_research_discovery_candidate_count": discovery_candidate_count,
            "codex_research_discovery_section_counts": discovery_section_counts,
            "codex_research_inbox_sha256": inbox_sha256,
            "codex_research_discovery_manifest_sha256": discovery_manifest_sha256,
            "codex_research_discovery_duplicate_url_count": int(
                research_inbox.get("discovery_duplicate_url_count", 0) or 0
            ),
            "codex_research_discovery_invalid_row_count": int(
                research_inbox.get("discovery_invalid_row_count", 0) or 0
            ),
            "codex_research_submitted_not_in_discovery_count": int(
                research_inbox.get("submitted_not_in_discovery_count", 0) or 0
            ),
            "codex_research_submission_quota_status": str(
                research_inbox.get("submission_quota_status") or "missing"
            ),
            "codex_research_submitted_section_counts": dict(
                research_inbox.get("submitted_section_counts") or {}
            ),
            "codex_research_rejected_section_counts": dict(
                research_inbox.get("rejected_section_counts") or {}
            ),
            "codex_research_rejection_reason_counts_by_section": dict(
                research_inbox.get("rejection_reason_counts_by_section") or {}
            ),
            "codex_research_accepted_section_rates": dict(
                research_inbox.get("accepted_section_rates") or {}
            ),
            "codex_research_fresh_counts": dict(research_inbox.get("fresh_source_counts") or {}),
            "codex_research_news_format_counts": dict(research_inbox.get("news_format_counts") or {}),
            "codex_research_schema_version": research_schema_version,
            "codex_research_key_number_contract_missing_count": int(
                research_inbox.get("key_number_contract_missing_count", 0) or 0
            ),
            "codex_research_key_number_evidence_missing_count": int(
                research_inbox.get("key_number_evidence_missing_count", 0) or 0
            ),
            "codex_research_key_number_public_copy_missing_count": int(
                research_inbox.get("key_number_public_copy_missing_count", 0) or 0
            ),
            "codex_research_key_number_item_counts": dict(
                research_inbox.get("key_number_item_counts") or {}
            ),
            "codex_research_key_number_underfilled": list(
                research_inbox.get("key_number_underfilled") or []
            ),
            "sent_history_overlap_count": sent_history_overlap_count,
            "sent_history_overlap_by_section": sent_history_overlap_by_section,
            "codex_research_technical_primary_source_count": int(
                research_inbox.get("technical_primary_source_count", 0) or 0
            ),
            "codex_research_technical_primary_source_ratio": research_technical_primary_ratio,
            "passed": not issues,
            "issues": issues,
        })
        previous_papers = by_section["paper"]
        paper_history.extend((report["_finished_at"], paper) for paper in by_section["paper"])

    slots_by_day: Dict[str, set[str]] = {}
    for row in rows:
        day_slots = slots_by_day.setdefault(row["date"], set())
        for slot_name in configured_slot_ids:
            if row["slot_id"].endswith(f"_{slot_name}"):
                day_slots.add(slot_name)
    missing_slots = {
        day: [slot_name for slot_name in configured_slot_ids if slot_name not in slots_by_day.get(day, set())]
        for day in sorted(selected_day_strings)
    }
    missing_slots = {day: values for day, values in missing_slots.items() if values}
    verified_days = len({row["date"] for row in rows})
    if any(not row["passed"] for row in rows):
        status = "failed"
    elif verified_days >= required_days and len(rows) >= required_report_count and not missing_slots:
        status = "passed"
    else:
        status = "pending"
    return {
        "status": status,
        "required_days": required_days,
        "reports_per_day": reports_per_day,
        "required_report_count": required_report_count,
        "technical_primary_source_ratio_min": technical_primary_source_ratio_min,
        "minimum_discovery_candidates": minimum_discovery_candidates,
        "minimum_discovered_by_section": minimum_discovered_by_section,
        "verified_days": verified_days,
        "verified_report_count": len(rows),
        "passed_report_count": sum(1 for row in rows if row["passed"]),
        "expected_slot_ids": configured_slot_ids,
        "acceptance_contract_version": V11_ACCEPTANCE_CONTRACT_VERSION,
        "contract_started_at": contract_started_at,
        "pre_contract_report_count": pre_contract_report_count,
        "missing_slots": missing_slots,
        "reports": sorted(rows, key=lambda row: row["finished_at"], reverse=True),
        "error": "",
    }


def build_product_diagnostics(
    config: Dict[str, Any],
    last_run: Dict[str, Any],
    last_success: Dict[str, Any],
    scheduler_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    database_path = resolve_database_path(config, ROOT)
    db = Database(database_path)
    latest_report = db.get_latest_report_run(exclude_validation=True)
    latest_quality = dict(latest_report.get("quality_diagnostics") or {})
    latest_research_inbox = dict(latest_quality.get("codex_research_inbox") or {})
    history_overlap_recorded = "sent_history_overlap_count" in latest_research_inbox
    sent_history_overlap_count = (
        int(latest_research_inbox.get("sent_history_overlap_count", 0) or 0)
        if history_overlap_recorded
        else None
    )
    latest_collection_report = db.get_latest_report_run_with_collection()
    latest_collection_quality = dict(latest_collection_report.get("quality_diagnostics") or {})
    latest_collection = dict(latest_collection_quality.get("collection") or {})
    collection_created_at = str(latest_collection_report.get("created_at") or "")
    collection_age_hours = None
    if collection_created_at:
        try:
            collection_time = datetime.fromisoformat(collection_created_at)
            if collection_time.tzinfo is None:
                collection_time = collection_time.replace(tzinfo=timezone.utc)
            collection_age_hours = round(
                max(0.0, (datetime.now(timezone.utc) - collection_time.astimezone(timezone.utc)).total_seconds() / 3600),
                1,
            )
        except ValueError:
            collection_age_hours = None
    report_quality = latest_report.get("quality_status") or last_run.get("quality_status") or last_success.get("quality_status") or "unknown"
    post_send_quality_scan = dict(last_run.get("post_send_quality_scan") or last_success.get("post_send_quality_scan") or {})
    ui_audit = dict(last_run.get("ui_audit") or last_success.get("ui_audit") or {})
    raw_model_path_breakdown = dict(
        latest_quality.get("model_path_breakdown")
        or last_run.get("model_path_breakdown")
        or last_success.get("model_path_breakdown")
        or {}
    )
    llm_health = dict(
        latest_quality.get("llm_health")
        or last_run.get("llm_health")
        or last_success.get("llm_health")
        or {}
    )
    configured_design_version = str((config.get("report") or {}).get("design_version") or "")
    latest_design_version = str(latest_quality.get("report_design_version") or "")
    effective_design_version = latest_design_version or configured_design_version
    uses_editorial_quality = effective_design_version in {
        "v6-editorial-learning",
        "v7-classic-briefing",
        "v8-editorial-reader",
        "v9-continuous-learning",
        "v10-learning-digest",
        "v11-editorial-library",
    }
    report_structure = dict(latest_quality.get("report_structure") or {})
    quality_gate = dict(latest_quality.get("quality_gate") or {})
    continuity = dict(latest_quality.get("continuity") or {})
    paper_freshness = dict(latest_quality.get("paper_freshness") or {})
    latest_report_id = str(latest_report.get("report_id") or "")
    if "adjacent_report_history_count" not in paper_freshness:
        paper_freshness["adjacent_report_history_count"] = len(
            db.get_latest_sent_report_items(
                content_type="paper",
                exclude_report_id=latest_report_id,
            )
        )
    if "cooldown_history_count" not in paper_freshness:
        paper_freshness["cooldown_history_count"] = len(
            db.get_recent_report_items(
                days=int((config.get("report") or {}).get("paper_repeat_cooldown_days", 7) or 7),
                limit=int((config.get("report") or {}).get("paper_repeat_history_limit", 1600) or 1600),
                exclude_report_id=latest_report_id,
                sent_only=True,
                content_type="paper",
            )
        )
    snapshot_metrics = (
        build_latest_report_snapshot_metrics(latest_report_id, config)
        if uses_editorial_quality
        else {}
    )
    if snapshot_metrics:
        quality_gate.update(snapshot_metrics)
        report_structure.update(snapshot_metrics)
        paper_freshness.update(
            {
                key: snapshot_metrics[key]
                for key in (
                    "paper_domain_counts",
                    "paper_domain_quota_status",
                    "paper_domain_quota_exceeded",
                    "paper_domain_quota_underfilled",
                )
                if key in snapshot_metrics
            }
        )
        if (
            snapshot_metrics.get("editorial_quality_status") == "passed"
            and report_quality in {"failed", "degraded"}
            and not quality_gate.get("final_html_quality_status")
        ):
            report_quality = "passed"
    if uses_editorial_quality:
        gate_status = str(
            quality_gate.get("final_html_quality_status")
            or quality_gate.get("editorial_quality_status")
            or quality_gate.get("status")
            or ""
        )
    else:
        gate_status = str(
            quality_gate.get("v5_quality_status")
            or quality_gate.get("v4_quality_status")
            or quality_gate.get("v3_quality_status")
            or quality_gate.get("status")
            or ""
        )
    if gate_status in {"failed", "degraded"}:
        report_quality = gate_status
    report_config = dict(config.get("report") or {})
    fresh_paper_count = int(
        quality_gate.get("fresh_paper_count", paper_freshness.get("fresh_paper_count", 0)) or 0
    )
    configured_minimum_paper_count = max(
        1,
        int(
            paper_freshness.get("configured_min_visible_paper_count")
            or report_config.get("min_visible_paper_count", 10)
            or 10
        ),
    )
    minimum_paper_count = max(
        1,
        int(
            paper_freshness.get("effective_min_visible_paper_count")
            or configured_minimum_paper_count
        ),
    )
    configured_target_min_paper_count = max(
        minimum_paper_count,
        int(
            paper_freshness.get("configured_target_min_paper_count")
            or report_config.get("paper_target_min_count", 15)
            or 15
        ),
    )
    return {
        "report_design_version": latest_design_version or configured_design_version,
        "report_product_mode": str(
            latest_quality.get("report_product_mode")
            or report_config.get("product_mode")
            or ""
        ),
        "latest_report_design_version": latest_design_version,
        "configured_report_design_version": configured_design_version,
        "source_health": dict(latest_quality.get("source_health") or last_run.get("source_health") or last_success.get("source_health") or {}),
        "source_weight_adjustments": dict(
            latest_quality.get("source_weight_adjustments")
            or last_run.get("source_weight_adjustments")
            or last_success.get("source_weight_adjustments")
            or {}
        ),
        "feedback_server_status": check_feedback_server_status(config),
        "last_report_quality_status": report_quality,
        "v3_quality_status": quality_gate.get("v3_quality_status") or quality_gate.get("status") or report_quality,
        "v4_quality_status": quality_gate.get("v4_quality_status") or quality_gate.get("status") or report_quality,
        "v5_quality_status": quality_gate.get("v5_quality_status") or quality_gate.get("status") or report_quality,
        "v10_quality_status": quality_gate.get("status") or report_quality,
        "adjacent_report_paper_overlap_rate": float(
            quality_gate.get(
                "adjacent_report_paper_overlap_rate",
                paper_freshness.get("adjacent_report_paper_overlap_rate", 0.0),
            )
            or 0.0
        ),
        "paper_within_report_duplicate_count": int(
            paper_freshness.get(
                "paper_within_report_duplicate_count",
                quality_gate.get("paper_within_report_duplicate_count", 0),
            )
            or 0
        ),
        "sent_history_overlap_count": sent_history_overlap_count,
        "sent_history_overlap_status": (
            "not_recorded"
            if not history_overlap_recorded
            else "passed"
            if sent_history_overlap_count == 0
            else "failed"
        ),
        "sent_history_overlap_by_section": dict(
            latest_research_inbox.get("sent_history_overlap_by_section") or {}
        ),
        "adjacent_report_history_count": int(
            paper_freshness.get("adjacent_report_history_count", 0) or 0
        ),
        "cooldown_history_count": int(
            paper_freshness.get("cooldown_history_count", 0) or 0
        ),
        "final_overlap_reselected_count": int(
            paper_freshness.get(
                "final_overlap_reselected_count",
                quality_gate.get("final_overlap_reselected_count", 0),
            )
            or 0
        ),
        "paper_repeat_filtered_count": int(
            quality_gate.get(
                "paper_repeat_filtered_count",
                paper_freshness.get("paper_repeat_filtered_count", 0),
            )
            or 0
        ),
        "fresh_paper_count": fresh_paper_count,
        "configured_minimum_paper_count": configured_minimum_paper_count,
        "minimum_paper_count": minimum_paper_count,
        "configured_target_min_paper_count": configured_target_min_paper_count,
        "recommended_minimum_paper_count": configured_target_min_paper_count,
        "paper_target_underfilled": fresh_paper_count < configured_target_min_paper_count,
        "reappeared_paper_with_update_count": int(
            quality_gate.get(
                "reappeared_paper_with_update_count",
                paper_freshness.get("reappeared_paper_with_update_count", 0),
            )
            or 0
        ),
        "arxiv_zero_result_warning_count": int(latest_collection.get("arxiv_zero_result_warning_count", 0) or 0),
        "arxiv_true_zero_result_count": int(latest_collection.get("arxiv_true_zero_result_count", 0) or 0),
        "arxiv_no_match_result_count": int(latest_collection.get("arxiv_no_match_result_count", 0) or 0),
        "arxiv_http_error_count": int(latest_collection.get("arxiv_http_error_count", 0) or 0),
        "arxiv_parse_error_count": int(latest_collection.get("arxiv_parse_error_count", 0) or 0),
        "arxiv_fallback_recovery_count": int(
            latest_collection.get("arxiv_fallback_recovery_count", 0) or 0
        ),
        "arxiv_retry_paths": list(latest_collection.get("arxiv_retry_paths") or []),
        "gpt_search_request_count": int(latest_collection.get("gpt_search_request_count", 0) or 0),
        "gpt_search_success_count": int(latest_collection.get("gpt_search_success_count", 0) or 0),
        "gpt_search_schema_error_count": int(
            latest_collection.get("gpt_search_schema_error_count", 0) or 0
        ),
        "arxiv_collection_report_id": str(latest_collection_report.get("report_id") or ""),
        "arxiv_collection_created_at": collection_created_at,
        "arxiv_collection_age_hours": collection_age_hours,
        "paper_freshness_status": str(
            quality_gate.get(
                "paper_freshness_status",
                paper_freshness.get("paper_freshness_status", "unknown"),
            )
            or "unknown"
        ),
        "paper_domain_counts": dict(paper_freshness.get("paper_domain_counts") or {}),
        "paper_domain_quota_status": str(
            quality_gate.get(
                "paper_domain_quota_status",
                paper_freshness.get("paper_domain_quota_status", "unknown"),
            )
            or "unknown"
        ),
        "paper_domain_quota_exceeded": dict(
            quality_gate.get(
                "paper_domain_quota_exceeded",
                paper_freshness.get("paper_domain_quota_exceeded", {}),
            )
            or {}
        ),
        "paper_domain_quota_underfilled": dict(
            quality_gate.get(
                "paper_domain_quota_underfilled",
                paper_freshness.get("paper_domain_quota_underfilled", {}),
            )
            or {}
        ),
        "memory_card_count": int(quality_gate.get("memory_card_count", report_structure.get("memory_card_count", 0)) or 0),
        "learning_card_missing_count": int(
            quality_gate.get("learning_card_missing_count", report_structure.get("learning_card_missing_count", 0)) or 0
        ),
        "technical_context_missing_count": int(
            quality_gate.get("technical_context_missing_count", report_structure.get("technical_context_missing_count", 0)) or 0
        ),
        "domain_coverage_warning_count": int(
            quality_gate.get("domain_coverage_warning_count", report_structure.get("domain_coverage_warning_count", 0)) or 0
        ),
        "paper_technical_intro_missing_count": int(
            quality_gate.get(
                "paper_technical_intro_missing_count",
                report_structure.get("paper_technical_intro_missing_count", 0),
            )
            or 0
        ),
        "editorial_quality_status": quality_gate.get("editorial_quality_status")
        or report_structure.get("editorial_quality_status")
        or quality_gate.get("status")
        or report_quality,
        "final_html_quality_status": quality_gate.get("final_html_quality_status") or "unknown",
        "final_html_bad_title_count": int(quality_gate.get("final_html_bad_title_count", 0) or 0),
        "untranslated_fact_count": int(quality_gate.get("untranslated_fact_count", 0) or 0),
        "exact_duplicate_sentence_count": int(quality_gate.get("exact_duplicate_sentence_count", 0) or 0),
        "paper_mechanism_missing_count": int(quality_gate.get("paper_mechanism_missing_count", 0) or 0),
        "paper_result_context_missing_count": int(quality_gate.get("paper_result_context_missing_count", 0) or 0),
        "paper_intro_length_fail_count": int(quality_gate.get("paper_intro_length_fail_count", 0) or 0),
        "display_body_over_limit_count": int(quality_gate.get("display_body_over_limit_count", 0) or 0),
        "memory_total_chars": int(quality_gate.get("memory_total_chars", 0) or 0),
        "memory_budget_exceeded": bool(quality_gate.get("memory_budget_exceeded", False)),
        "focus_source_max_count": int(quality_gate.get("focus_source_max_count", 0) or 0),
        "focus_topic_max_count": int(quality_gate.get("focus_topic_max_count", 0) or 0),
        "appendix_body_overlap_count": int(quality_gate.get("appendix_body_overlap_count", 0) or 0),
        "truncated_focus_text_count": int(quality_gate.get("truncated_focus_text_count", 0) or 0),
        "visible_text_chars": int(quality_gate.get("visible_text_chars", 0) or 0),
        "html_size_bytes": int(quality_gate.get("html_size_bytes", 0) or 0),
        "html_size_kb": float(quality_gate.get("html_size_kb", 0.0) or 0.0),
        "email_clipping_warning": bool(quality_gate.get("email_clipping_warning", False)),
        "email_clipping_risk": bool(quality_gate.get("email_clipping_risk", False)),
        "focus_source_concentration": float(quality_gate.get("focus_source_concentration", 0.0) or 0.0),
        "memory_item_count": int(quality_gate.get("memory_item_count", 0) or 0),
        "featured_paper_count": int(quality_gate.get("featured_paper_count", 0) or 0),
        "deepseek_schema_valid_count": int(
            quality_gate.get("deepseek_schema_valid_count", report_structure.get("deepseek_schema_valid_count", 0)) or 0
        ),
        "deepseek_empty_facts_count": int(
            quality_gate.get("deepseek_empty_facts_count", report_structure.get("deepseek_empty_facts_count", 0)) or 0
        ),
        "deepseek_key_field_missing_count": int(
            quality_gate.get("deepseek_key_field_missing_count", report_structure.get("deepseek_key_field_missing_count", 0)) or 0
        ),
        "deepseek_health_hint": quality_gate.get("deepseek_health_hint") or report_structure.get("deepseek_health_hint") or "",
        "gpt_schema_valid_count": int(
            quality_gate.get("gpt_schema_valid_count", report_structure.get("gpt_schema_valid_count", 0)) or 0
        ),
        "gpt_empty_facts_count": int(
            quality_gate.get("gpt_empty_facts_count", report_structure.get("gpt_empty_facts_count", 0)) or 0
        ),
        "gpt_key_field_missing_count": int(
            quality_gate.get("gpt_key_field_missing_count", report_structure.get("gpt_key_field_missing_count", 0)) or 0
        ),
        "gpt_health_hint": quality_gate.get("gpt_health_hint") or report_structure.get("gpt_health_hint") or "",
        "llm_schema_valid_count": int(
            quality_gate.get("llm_schema_valid_count", report_structure.get("llm_schema_valid_count", 0)) or 0
        ),
        "llm_health_hint": quality_gate.get("llm_health_hint") or report_structure.get("llm_health_hint") or "",
        "llm_health": llm_health,
        "llm_health_status": str(llm_health.get("status") or "unknown"),
        "template_fallback_count": int(
            quality_gate.get("template_fallback_count", report_structure.get("template_fallback_count", 0)) or 0
        ),
        "index_only_paper_count": int(
            quality_gate.get("index_only_paper_count", report_structure.get("index_only_paper_count", 0)) or 0
        ),
        "generic_fact_bundle_count": int(
            quality_gate.get("generic_fact_bundle_count", report_structure.get("generic_fact_bundle_count", 0)) or 0
        ),
        "unsupported_numeric_claim_count": int(
            quality_gate.get("unsupported_numeric_claim_count", report_structure.get("unsupported_numeric_claim_count", 0)) or 0
        ),
        "mixed_language_title_count": int(
            quality_gate.get("mixed_language_title_count", report_structure.get("mixed_language_title_count", 0)) or 0
        ),
        "field_label_leak_count": int(
            quality_gate.get("field_label_leak_count", report_structure.get("field_label_leak_count", 0)) or 0
        ),
        "low_info_expanded_count": int(
            quality_gate.get("low_info_expanded_count", report_structure.get("low_info_expanded_count", 0)) or 0
        ),
        "mojibake_suspect_count": int(
            quality_gate.get("mojibake_suspect_count", report_structure.get("mojibake_suspect_count", 0)) or 0
        ),
        "paper_technical_intro_pass_count": int(
            quality_gate.get("paper_technical_intro_pass_count", report_structure.get("paper_technical_intro_pass_count", 0)) or 0
        ),
        "paper_technical_intro_fail_count": int(
            quality_gate.get("paper_technical_intro_fail_count", report_structure.get("paper_technical_intro_fail_count", 0)) or 0
        ),
        "paper_plain_summary_pass_count": int(
            quality_gate.get("paper_plain_summary_pass_count", report_structure.get("paper_plain_summary_pass_count", 0)) or 0
        ),
        "paper_plain_summary_fail_count": int(
            quality_gate.get(
                "paper_plain_summary_fail_count",
                quality_gate.get(
                    "paper_plain_summary_missing_count",
                    report_structure.get("paper_plain_summary_fail_count", 0),
                ),
            )
            or 0
        ),
        "paper_plain_summary_pass_rate": float(
            quality_gate.get("paper_plain_summary_pass_rate", report_structure.get("paper_plain_summary_pass_rate", 0.0))
            or 0.0
        ),
        "low_value_module_count": int(
            quality_gate.get("low_value_module_count", report_structure.get("low_value_module_count", 0)) or 0
        ),
        "title_only_item_count": int(
            quality_gate.get("title_only_item_count", report_structure.get("title_only_item_count", 0)) or 0
        ),
        "unsupported_claim_count": int(
            quality_gate.get("unsupported_claim_count", report_structure.get("unsupported_claim_count", 0)) or 0
        ),
        "primary_source_ratio": float(
            quality_gate.get("primary_source_ratio", report_structure.get("primary_source_ratio", 0.0)) or 0.0
        ),
        "technical_primary_source_count": int(
            quality_gate.get(
                "technical_primary_source_count",
                report_structure.get("technical_primary_source_count", 0),
            )
            or 0
        ),
        "technical_primary_source_ratio": float(
            quality_gate.get(
                "technical_primary_source_ratio",
                report_structure.get("technical_primary_source_ratio", 0.0),
            )
            or 0.0
        ),
        "duplicate_event_rate": float(
            quality_gate.get("duplicate_event_rate", report_structure.get("duplicate_event_rate", 0.0)) or 0.0
        ),
        "physical_ai_featured_count": int(
            quality_gate.get("physical_ai_featured_count", report_structure.get("physical_ai_featured_count", 0)) or 0
        ),
        "generic_phrase_count": int(quality_gate.get("generic_phrase_count", quality_gate.get("generic_summary_count", 0)) or 0),
        "duplicate_expression_count": int(quality_gate.get("duplicate_expression_count", quality_gate.get("repeated_sentence_count", 0)) or 0),
        "suspicious_claim_count": int(quality_gate.get("suspicious_claim_count", report_structure.get("suspicious_claim_count", 0)) or 0),
        "paper_substantive_description_fail_count": int(
            quality_gate.get(
                "paper_substantive_description_fail_count",
                report_structure.get("paper_substantive_description_fail_count", 0),
            )
            or 0
        ),
        "paper_numeric_parse_error_count": int(
            quality_gate.get(
                "paper_numeric_parse_error_count",
                report_structure.get("paper_numeric_parse_error_count", 0),
            )
            or 0
        ),
        "paper_core_summary_pass_count": int(
            quality_gate.get(
                "paper_core_summary_pass_count",
                report_structure.get("paper_core_summary_pass_count", 0),
            )
            or 0
        ),
        "paper_core_summary_fail_count": int(
            quality_gate.get(
                "paper_core_summary_fail_count",
                report_structure.get("paper_core_summary_fail_count", 0),
            )
            or 0
        ),
        "physical_ai_item_count": int(
            quality_gate.get("physical_ai_item_count", report_structure.get("physical_ai_item_count", report_structure.get("physical_ai_count", 0))) or 0
        ),
        "paper_selected_count": int(quality_gate.get("paper_selected_count", report_structure.get("paper_selected_count", 0)) or 0),
        "paper_appendix_count": int(quality_gate.get("paper_appendix_count", report_structure.get("paper_appendix_count", 0)) or 0),
        "visible_paper_count": int(
            quality_gate.get("visible_paper_count", report_structure.get("visible_paper_count", 0)) or 0
        ),
        "visible_information_count": int(
            quality_gate.get(
                "visible_information_count",
                report_structure.get("visible_information_count", 0),
            )
            or 0
        ),
        "source_news_brief_count": int(
            quality_gate.get(
                "source_news_brief_count",
                report_structure.get("source_news_brief_count", 0),
            )
            or 0
        ),
        "visible_news_count": int(
            quality_gate.get(
                "visible_news_count",
                report_structure.get("visible_news_count", 0),
            )
            or 0
        ),
        "visible_technical_count": int(
            quality_gate.get(
                "visible_technical_count",
                report_structure.get("visible_technical_count", 0),
            )
            or 0
        ),
        "cross_section_duplicate_count": int(
            quality_gate.get(
                "cross_section_duplicate_count",
                report_structure.get("cross_section_duplicate_count", 0),
            )
            or 0
        ),
        "cross_section_event_duplicate_count": int(
            quality_gate.get(
                "cross_section_event_duplicate_count",
                report_structure.get("cross_section_event_duplicate_count", 0),
            )
            or 0
        ),
        "high_evidence_calibration_warning": bool(
            quality_gate.get(
                "high_evidence_calibration_warning",
                report_structure.get("high_evidence_calibration_warning", False),
            )
        ),
        "post_send_quality_scan": post_send_quality_scan,
        "ui_audit": ui_audit,
        "auto_rewrite_attempted_count": int(
            latest_quality.get("auto_rewrite_attempted_count", last_run.get("auto_rewrite_attempted_count", 0)) or 0
        ),
        "auto_rewrite_success_count": int(
            latest_quality.get("auto_rewrite_success_count", last_run.get("auto_rewrite_success_count", 0)) or 0
        ),
        "model_path_breakdown": normalize_model_path_breakdown(raw_model_path_breakdown),
        "title_repair": dict(latest_quality.get("title_repair") or {}),
        "report_structure": report_structure,
        "feedback_count_7d": db.get_feedback_count(days=7),
        "continuity_new_item_count": int(continuity.get("new_item_count", 0) or 0),
        "continuity_updated_item_count": int(continuity.get("updated_item_count", 0) or 0),
        "repeated_focus_demoted_count": int(continuity.get("repeated_focus_demoted_count", 0) or 0),
        "topic_dossier_count": int(continuity.get("topic_dossier_count", 0) or 0),
        "reading_queue_count": int(continuity.get("reading_queue_count", 0) or 0),
        "paper_context_count": int(continuity.get("paper_context_count", 0) or 0),
        "v8_production_acceptance": build_v8_production_acceptance(
            dict(scheduler_config or config.get("scheduler") or DEFAULT_SCHEDULER_CONFIG),
            db_path=database_path,
        ),
        "paper_freshness_production_acceptance": build_paper_freshness_production_acceptance(
            dict(scheduler_config or config.get("scheduler") or DEFAULT_SCHEDULER_CONFIG),
            overlap_max=float((config.get("quality_gate") or {}).get("adjacent_report_paper_overlap_max", 0.10) or 0.10),
            db_path=database_path,
        ),
        "v11_production_acceptance": build_v11_production_acceptance(
            dict(scheduler_config or config.get("scheduler") or DEFAULT_SCHEDULER_CONFIG),
            minimum_news=int(report_config.get("min_visible_news_count", 20) or 20),
            minimum_technical=int(report_config.get("min_visible_technical_count", 20) or 20),
            minimum_papers=int(report_config.get("min_visible_paper_count", 15) or 15),
            minimum_discovery_candidates=int(
                ((config.get("sources") or {}).get("codex_research_inbox") or {}).get(
                    "minimum_discovery_candidates",
                    160,
                )
                or 160
            ),
            minimum_discovered_by_section=dict(
                ((config.get("sources") or {}).get("codex_research_inbox") or {}).get(
                    "minimum_discovered_by_section"
                )
                or {"news": 50, "technical": 50, "paper": 40}
            ),
            technical_primary_source_ratio_min=float(
                (config.get("quality_gate") or {}).get(
                    "technical_primary_source_ratio_min",
                    0.80,
                )
                or 0.80
            ),
            overlap_max=float((config.get("quality_gate") or {}).get("adjacent_report_paper_overlap_max", 0.10) or 0.10),
            db_path=database_path,
        ),
        "latest_report": latest_report,
    }


def normalize_model_path_breakdown(breakdown: Dict[str, Any]) -> Dict[str, int]:
    normalized: Dict[str, int] = {}
    for key, value in (breakdown or {}).items():
        normalized_key = "fallback_v2" if str(key) == "v2" else str(key)
        normalized[normalized_key] = normalized.get(normalized_key, 0) + int(value or 0)
    return normalized


def _report_item_uses_legacy_model_path(item: Dict[str, Any], include_v2_fallback: bool = False) -> bool:
    model_used = str(item.get("model_used", "") or "").strip()
    analysis_version = str(item.get("analysis_version", "") or "").strip()
    if not model_used and not analysis_version:
        return True
    return bool(include_v2_fallback and not model_used and analysis_version == "v2")


def get_report_model_path_backfill_candidates(
    db: Database,
    report_id: str,
    limit: int = 10,
    include_v2_fallback: bool = False,
) -> list[Dict[str, Any]]:
    conn = db._get_conn()
    try:
        rows = conn.execute(
            """
            SELECT article_id, rank, section, snapshot_json
            FROM report_items
            WHERE report_id = ?
            ORDER BY rank
            """,
            (report_id,),
        ).fetchall()
    finally:
        conn.close()
    candidates: list[Dict[str, Any]] = []
    for row in rows:
        try:
            snapshot = json.loads(row["snapshot_json"] or "{}")
        except json.JSONDecodeError:
            snapshot = {}
        if not row["article_id"] or not _report_item_uses_legacy_model_path(snapshot, include_v2_fallback=include_v2_fallback):
            continue
        candidates.append(
            {
                "article_id": int(row["article_id"]),
                "rank": int(row["rank"] or 0),
                "section": str(row["section"] or ""),
                "title": snapshot.get("title_cn") or snapshot.get("title") or "",
                "url": snapshot.get("url", ""),
                "snapshot": snapshot,
            }
        )
        if len(candidates) >= max(1, int(limit)):
            break
    return candidates


def _get_report_run_for_command(db: Database, report_id: str = "") -> Dict[str, Any]:
    report_id = str(report_id or "").strip()
    return db.get_report_run(report_id) if report_id else db.get_latest_report_run()


def _load_report_item_snapshots(db: Database, report_id: str) -> list[Dict[str, Any]]:
    conn = db._get_conn()
    try:
        rows = conn.execute(
            """
            SELECT id, article_id, rank, section, snapshot_json
            FROM report_items
            WHERE report_id = ?
            ORDER BY rank
            """,
            (report_id,),
        ).fetchall()
    finally:
        conn.close()
    items: list[Dict[str, Any]] = []
    for row in rows:
        try:
            snapshot = json.loads(row["snapshot_json"] or "{}")
        except json.JSONDecodeError:
            snapshot = {}
        snapshot.setdefault("id", row["article_id"])
        snapshot.setdefault("report_rank", int(row["rank"] or 0))
        snapshot.setdefault("report_section", str(row["section"] or ""))
        snapshot["_report_item_id"] = int(row["id"])
        snapshot["_article_id"] = int(row["article_id"] or 0)
        items.append(snapshot)
    return items


def scan_report_quality(
    report_id: str = "",
    limit: int = 10,
    persist: bool = False,
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    from main import (
        build_content_quality_counts,
        evidence_quality_value,
        information_density_value,
        summary_looks_generic,
        title_fact_mismatch,
        title_looks_bad,
    )

    if db_path:
        resolved_db_path = str(db_path)
    else:
        runtime_config, _ = load_runtime_config()
        resolved_db_path = resolve_database_path(runtime_config, ROOT)
    db = Database(resolved_db_path)
    report_run = _get_report_run_for_command(db, report_id)
    resolved_report_id = str(report_run.get("report_id", "") or report_id or "")
    if not resolved_report_id or not report_run:
        return {"report_id": resolved_report_id, "counts": {}, "issues": [], "error": "No matching report_runs snapshot exists."}
    items = _load_report_item_snapshots(db, resolved_report_id)
    counts = build_content_quality_counts(selected_items=items, update_candidates=[], source_preferences={})
    counts["low_density_count"] = sum(1 for item in items if information_density_value(item) < 0.35)
    counts["title_fact_mismatch_count"] = sum(
        1
        for item in items
        if not item.get("_codex_research_validated")
        and title_fact_mismatch(item, item.get("facts") if isinstance(item.get("facts"), dict) else {})
    )
    issues: list[Dict[str, Any]] = []
    for item in items:
        issue_types: list[str] = []
        if title_looks_bad(item):
            issue_types.append("bad_title")
        if summary_looks_generic(item.get("summary", "")):
            issue_types.append("generic_summary")
        if evidence_quality_value(item) < 0.35:
            issue_types.append("low_evidence")
        if information_density_value(item) < 0.35:
            issue_types.append("low_density")
        facts = item.get("facts") if isinstance(item.get("facts"), dict) else {}
        if not item.get("_codex_research_validated") and title_fact_mismatch(item, facts):
            issue_types.append("title_fact_mismatch")
        if not issue_types:
            continue
        issues.append(
            {
                "rank": int(item.get("report_rank", 0) or 0),
                "section": str(item.get("report_section", "") or ""),
                "article_id": int(item.get("_article_id", item.get("id", 0)) or 0),
                "title": item.get("title_cn") or item.get("title") or "",
                "issue_types": issue_types,
                "evidence_quality": evidence_quality_value(item),
                "information_density": information_density_value(item),
            }
        )
    issues.sort(key=lambda row: (int(row.get("rank", 0) or 0), row.get("title", "")))
    section_issue_counts: Dict[str, int] = {}
    for issue in issues:
        section = str(issue.get("section", "") or "unknown")
        section_issue_counts[section] = section_issue_counts.get(section, 0) + 1
    focus_issue_count = sum(
        1
        for issue in issues
        if str(issue.get("section", "") or "") in FOCUS_REPORT_SECTIONS
    )
    brief_issue_count = section_issue_counts.get("brief", 0)
    result = {
        "report_id": resolved_report_id,
        "scanned": len(items),
        "counts": counts,
        "issues": issues[: max(1, int(limit))],
        "issue_count": len(issues),
        "focus_issue_count": focus_issue_count,
        "brief_issue_count": brief_issue_count,
        "section_issue_counts": section_issue_counts,
        "error": "",
    }
    if persist:
        report_run = db.get_report_run(resolved_report_id)
        diagnostics = dict(report_run.get("quality_diagnostics") or {})
        diagnostics["content_quality"] = {
            **dict(diagnostics.get("content_quality") or {}),
            **counts,
        }
        quality_gate = dict(diagnostics.get("quality_gate") or {})
        quality_gate["bad_title_count"] = int(counts.get("bad_title_count", 0) or 0)
        final_html_failed = str(quality_gate.get("final_html_quality_status") or "") == "failed"
        if not issues and not final_html_failed:
            quality_gate["failed_item_urls"] = []
            quality_gate["status"] = "passed"
        diagnostics["quality_gate"] = quality_gate
        title_repair = dict(diagnostics.get("title_repair") or {})
        title_repair["bad_title_unresolved_count"] = int(counts.get("bad_title_count", 0) or 0)
        diagnostics["title_repair"] = title_repair
        if int(counts.get("bad_title_count", 0) or 0) == 0:
            diagnostics["warnings"] = [
                warning
                for warning in list(diagnostics.get("warnings") or [])
                if not str(warning).startswith("bad_titles_present:")
            ]
        diagnostics["last_report_quality_scan"] = {
            "report_id": resolved_report_id,
            "scanned": len(items),
            "issue_count": len(issues),
            "focus_issue_count": focus_issue_count,
            "brief_issue_count": brief_issue_count,
            "section_issue_counts": section_issue_counts,
            "scanned_at": datetime.now().isoformat(timespec="seconds"),
        }
        conn = db._get_conn()
        try:
            existing_status = str(report_run.get("quality_status") or "unknown")
            if final_html_failed:
                quality_status = "failed"
                quality_gate["status"] = "failed"
            elif focus_issue_count == 0:
                quality_status = "passed"
            elif existing_status in {"failed", "degraded"}:
                quality_status = existing_status
            else:
                quality_status = "degraded"
            conn.execute(
                "UPDATE report_runs SET quality_status = ?, quality_diagnostics = ? WHERE report_id = ?",
                (quality_status, json.dumps(diagnostics, ensure_ascii=False, sort_keys=True, default=str), resolved_report_id),
            )
            conn.commit()
        finally:
            conn.close()
    return result


def refresh_report_quality_after_run(
    status: Dict[str, Any],
    *,
    limit: int = 10,
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    report_id = str(status.get("report_id", "") or "").strip()
    if not report_id:
        return status

    try:
        scan_result = scan_report_quality(report_id=report_id, limit=limit, persist=True, db_path=db_path)
        status["post_send_quality_scan"] = {
            "report_id": scan_result.get("report_id", report_id),
            "scanned": int(scan_result.get("scanned", 0) or 0),
            "issue_count": int(scan_result.get("issue_count", 0) or 0),
            "focus_issue_count": int(scan_result.get("focus_issue_count", 0) or 0),
            "brief_issue_count": int(scan_result.get("brief_issue_count", 0) or 0),
            "section_issue_counts": dict(scan_result.get("section_issue_counts") or {}),
            "error": str(scan_result.get("error", "") or ""),
        }
        if scan_result.get("error"):
            return status

        db = Database(str(db_path or resolve_database_path(None, ROOT)))
        report_run = db.get_report_run(report_id)
        if report_run:
            status["quality_status"] = report_run.get("quality_status") or status.get("quality_status", "")
            status["quality_diagnostics"] = report_run.get("quality_diagnostics") or status.get("quality_diagnostics", {})
    except Exception as exc:
        status["post_send_quality_scan"] = {
            "report_id": report_id,
            "scanned": 0,
            "issue_count": 0,
            "focus_issue_count": 0,
            "brief_issue_count": 0,
            "section_issue_counts": {},
            "error": str(exc),
        }
    return status


def run_email_ui_audit(
    volume_paths: list[str],
    *,
    output_dir: Path,
    timeout_seconds: int = 180,
) -> Dict[str, Any]:
    return execute_email_ui_audit(
        volume_paths,
        output_dir=output_dir,
        root=ROOT,
        timeout_seconds=timeout_seconds,
    )


def refresh_email_ui_audit_after_run(
    status: Dict[str, Any],
    scheduler_config: Dict[str, Any],
    *,
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    if not bool(scheduler_config.get("ui_audit_enabled", True)):
        status["ui_audit"] = {"status": "disabled", "passed": False}
        return status
    volume_paths = [str(value) for value in (status.get("email_volume_paths") or []) if str(value or "").strip()]
    report_id = str(status.get("report_id") or "").strip()
    slot_id = str((status.get("send_slot") or {}).get("slot_id") or report_id or "report")
    safe_slot_id = re.sub(r"[^0-9A-Za-z_.-]+", "-", slot_id).strip("-") or "report"
    output_root = Path(str(scheduler_config.get("ui_audit_output_dir") or ROOT / "artifacts/v11_production_ui_audit"))
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    existing_result = dict(status.get("ui_audit") or {})
    if str(existing_result.get("status") or "") in {"passed", "failed", "error"}:
        result = existing_result
    else:
        result = run_email_ui_audit(
            volume_paths,
            output_dir=output_root / safe_slot_id,
            timeout_seconds=int(scheduler_config.get("ui_audit_timeout_seconds", 180) or 180),
        )
    status["ui_audit"] = result
    if not report_id:
        return status
    db = Database(str(db_path or resolve_database_path(None, ROOT)))
    report_run = db.get_report_run(report_id)
    if not report_run:
        return status
    diagnostics = dict(report_run.get("quality_diagnostics") or {})
    diagnostics["ui_audit"] = result
    quality_gate = dict(diagnostics.get("quality_gate") or {})
    quality_gate["ui_audit_status"] = result.get("status")
    quality_gate["ui_audit_failed_render_count"] = int(result.get("failed_render_count", 0) or 0)
    diagnostics["quality_gate"] = quality_gate
    quality_status = str(report_run.get("quality_status") or status.get("quality_status") or "unknown")
    if not result.get("passed", False):
        quality_status = "failed"
        quality_gate["status"] = "failed"
    conn = db._get_conn()
    try:
        conn.execute(
            "UPDATE report_runs SET quality_status = ?, quality_diagnostics = ? WHERE report_id = ?",
            (quality_status, json.dumps(diagnostics, ensure_ascii=False, sort_keys=True, default=str), report_id),
        )
        conn.commit()
    finally:
        conn.close()
    status["quality_status"] = quality_status
    status["quality_diagnostics"] = diagnostics
    return status


def _trim_title_fragment(text: Any, limit: int = 18) -> str:
    text = " ".join(str(text or "").replace("\n", " ").split())
    text = text.strip(" .,;:|-/。；，、")
    return text if len(text) <= limit else text[: limit - 1].rstrip(" .,;:|-/") + "…"


def _looks_like_chinese_title_candidate(text: str) -> bool:
    if not text:
        return False
    cjk_count = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
    if cjk_count < 6:
        return False
    weak_starts = ("关键看", "后续", "下一步", "值得关注", "需关注")
    if text.startswith(weak_starts):
        return False
    weak_phrases = ("后续需", "后续要", "值得持续关注", "需要观察", "观察其")
    return not any(phrase in text for phrase in weak_phrases)


def _title_from_chinese_text(text: Any, limit: int = 34) -> str:
    text = " ".join(str(text or "").replace("\n", " ").split())
    text = text.replace("...", "…").strip()
    if not text:
        return ""
    if "论文来源" in text or re.match(r"^[A-Za-z0-9\- ()]+论文针对", text):
        return ""
    for marker in ("。", "；", ";"):
        if marker in text:
            text = text.split(marker, 1)[0]
            break
    comma_head = text.split("，", 1)[0].strip() if "，" in text else text
    if len(comma_head) >= 12 and _looks_like_chinese_title_candidate(comma_head):
        text = comma_head
    return _trim_title_fragment(text, limit) if _looks_like_chinese_title_candidate(text) else ""


def _compressed_domain_title_from_summary(item: Dict[str, Any]) -> str:
    summary = str(item.get("summary", "") or "")
    facts = item.get("facts") if isinstance(item.get("facts"), dict) else {}
    who = str(facts.get("who") or "").strip()
    if "AgentCore payments" in summary or "payments功能" in summary:
        prefix = "Amazon Bedrock AgentCore" if "Amazon Bedrock" in summary or "Amazon Bedrock" in who else who
        return _trim_title_fragment(f"{prefix or 'AgentCore'}发布支付功能预览", 34)
    if "ElevenLabs替代方案" in summary:
        prefix = who or "OmniVoice Studio"
        return _trim_title_fragment(f"{prefix}发布本地开源语音工具", 34)
    return ""


def suggest_title_from_snapshot_with_source(item: Dict[str, Any]) -> Dict[str, str]:
    from main import title_looks_bad

    compressed_title = _compressed_domain_title_from_summary(item)
    if compressed_title and not title_looks_bad({"title_cn": compressed_title}):
        return {"title": compressed_title, "source": "domain_compression"}
    summary_title = _title_from_chinese_text(item.get("summary"), 34)
    if summary_title and not title_looks_bad({"title_cn": summary_title}):
        return {"title": summary_title, "source": "summary"}
    preview_title = _title_from_chinese_text(item.get("summary_preview"), 34)
    if preview_title and not title_looks_bad({"title_cn": preview_title}):
        return {"title": preview_title, "source": "summary_preview"}
    facts = item.get("facts") if isinstance(item.get("facts"), dict) else {}
    who = _trim_title_fragment(facts.get("who") or "", 18)
    target = _trim_title_fragment(facts.get("target") or "", 20)
    action = str(facts.get("action") or "").strip().lower()
    action_map = {
        "launch": "发布",
        "launches": "发布",
        "release": "发布",
        "releases": "发布",
        "announce": "宣布",
        "announces": "宣布",
        "propose": "提出",
        "proposes": "提出",
        "demonstrate": "展示",
        "demonstrates": "展示",
        "publish": "发布",
        "publishes": "发布",
    }
    action_cn = action_map.get(action, facts.get("action") or "更新")
    if who and who.lower() != "this paper" and target:
        fact_title = _trim_title_fragment(f"{who}{action_cn}{target}", 34)
        if fact_title and not title_looks_bad({"title_cn": fact_title}):
            return {"title": fact_title, "source": "facts"}
        if fact_title:
            content_type_for_fact = str(item.get("content_type", "") or "")
            display_topic_for_fact = str(item.get("display_topic") or item.get("topic_cn") or item.get("category") or "")
            if "论文" in display_topic_for_fact or content_type_for_fact == "paper":
                return {"title": _trim_title_fragment(f"{who}更新AI研究方法与实验结果", 34), "source": "facts"}
            return {"title": _trim_title_fragment(f"{who}发布AI领域新进展", 34), "source": "facts"}
    title = str(item.get("title") or item.get("title_cn") or "").strip()
    title = title.replace(":", " ").replace("|", " ")
    title = " ".join(title.split())
    original_title = _trim_title_fragment(title, 34)
    if original_title and not title_looks_bad({"title_cn": original_title}):
        return {"title": original_title, "source": "original_title"}
    entity_match = re.search(r"\b[A-Z][A-Za-z0-9.-]{2,}\b", title)
    entity = entity_match.group(0) if entity_match else ""
    if not entity:
        fact_who = str(facts.get("who") or "").strip() if isinstance(facts, dict) else ""
        if re.match(r"^[A-Z][A-Za-z0-9.-]{2,}$", fact_who):
            entity = fact_who
    content_type = str(item.get("content_type", "") or "")
    display_topic = str(item.get("display_topic") or item.get("topic_cn") or item.get("category") or "")
    if entity and content_type == "paper":
        return {"title": _trim_title_fragment(f"{entity}更新AI研究方法与实验结果", 34), "source": "safe_fallback"}
    if entity:
        return {"title": _trim_title_fragment(f"{entity}发布AI领域新进展", 34), "source": "safe_fallback"}
    if "论文" in display_topic or content_type == "paper":
        return {"title": "AI论文更新方法与实验结果", "source": "safe_fallback"}
    return {"title": "AI行业动态更新", "source": "safe_fallback"}


def suggest_title_from_snapshot(item: Dict[str, Any]) -> str:
    return suggest_title_from_snapshot_with_source(item)["title"]


def fix_report_bad_titles(report_id: str = "", limit: int = 10, dry_run: bool = False) -> Dict[str, Any]:
    from main import title_looks_bad

    config, _ = load_runtime_config()
    db = Database(resolve_database_path(config, ROOT))
    report_run = _get_report_run_for_command(db, report_id)
    resolved_report_id = str(report_run.get("report_id", "") or report_id or "")
    if not resolved_report_id or not report_run:
        return {"report_id": resolved_report_id, "attempted": 0, "updated": 0, "dry_run": bool(dry_run), "items": [], "error": "No matching report_runs snapshot exists."}
    items = [item for item in _load_report_item_snapshots(db, resolved_report_id) if title_looks_bad(item)]
    items = items[: max(1, int(limit))]
    updates: list[Dict[str, Any]] = []
    for item in items:
        title_suggestion = suggest_title_from_snapshot_with_source(item)
        new_title = title_suggestion["title"]
        updates.append(
            {
                "report_item_id": int(item.get("_report_item_id", 0) or 0),
                "article_id": int(item.get("_article_id", 0) or 0),
                "rank": int(item.get("report_rank", 0) or 0),
                "section": str(item.get("report_section", "") or ""),
                "old_title": item.get("title_cn") or item.get("title") or "",
                "new_title": new_title,
                "source": title_suggestion.get("source", ""),
            }
        )
    if dry_run:
        return {"report_id": resolved_report_id, "attempted": len(updates), "updated": 0, "dry_run": True, "items": updates, "error": ""}
    conn = db._get_conn()
    try:
        for update in updates:
            row = conn.execute("SELECT snapshot_json FROM report_items WHERE id = ?", (update["report_item_id"],)).fetchone()
            try:
                snapshot = json.loads(row["snapshot_json"] or "{}") if row else {}
            except json.JSONDecodeError:
                snapshot = {}
            snapshot["title_cn"] = update["new_title"]
            conn.execute(
                "UPDATE report_items SET snapshot_json = ? WHERE id = ?",
                (json.dumps(snapshot, ensure_ascii=False, sort_keys=True, default=str), update["report_item_id"]),
            )
            if update["article_id"]:
                conn.execute("UPDATE articles SET title_cn = ? WHERE id = ?", (update["new_title"], update["article_id"]))
        conn.commit()
    finally:
        conn.close()
    scan_result = scan_report_quality(report_id=resolved_report_id, limit=10, persist=True)
    report_run = db.get_report_run(resolved_report_id)
    diagnostics = dict(report_run.get("quality_diagnostics") or {})
    title_repair = dict(diagnostics.get("title_repair") or {})
    prior_examples = list(title_repair.get("examples") or [])
    title_repair["bad_title_repaired_count"] = int(title_repair.get("bad_title_repaired_count", 0) or 0) + len(updates)
    title_repair["bad_title_unresolved_count"] = int((scan_result.get("counts") or {}).get("bad_title_count", 0) or 0)
    title_repair["examples"] = (prior_examples + updates)[:10]
    diagnostics["title_repair"] = title_repair
    conn = db._get_conn()
    try:
        conn.execute(
            "UPDATE report_runs SET quality_diagnostics = ? WHERE report_id = ?",
            (json.dumps(diagnostics, ensure_ascii=False, sort_keys=True, default=str), resolved_report_id),
        )
        conn.commit()
    finally:
        conn.close()
    return {
        "report_id": resolved_report_id,
        "attempted": len(updates),
        "updated": len(updates),
        "dry_run": False,
        "items": updates,
        "post_scan_counts": scan_result.get("counts", {}),
        "title_repair": title_repair,
        "error": "",
    }


def _merge_backfilled_snapshot(
    snapshot: Dict[str, Any],
    result: Dict[str, Any],
    preserve_display_fields: bool = False,
) -> Dict[str, Any]:
    updated = dict(snapshot)
    display_fields = {"title_cn", "summary", "summary_preview", "why_it_matters", "why_now", "expected_effect", "future_impact"}
    for field in (
        "title_cn",
        "summary",
        "summary_preview",
        "why_it_matters",
        "why_now",
        "expected_effect",
        "future_impact",
        "category",
        "topic_cn",
        "score",
        "keywords",
        "facts",
        "evidence_quality",
        "information_density",
        "model_used",
    ):
        if preserve_display_fields and field in display_fields:
            continue
        if field in result:
            updated[field] = result.get(field)
    updated["analysis_version"] = "v2"
    return updated


def _refresh_report_model_path_diagnostics(
    db: Database,
    report_id: str,
    refresh_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, int]:
    from main import model_path_breakdown

    conn = db._get_conn()
    try:
        rows = conn.execute(
            "SELECT snapshot_json FROM report_items WHERE report_id = ? ORDER BY rank",
            (report_id,),
        ).fetchall()
        items = []
        for row in rows:
            try:
                items.append(json.loads(row["snapshot_json"] or "{}"))
            except json.JSONDecodeError:
                items.append({})
        breakdown = normalize_model_path_breakdown(model_path_breakdown(items))
        report_run = db.get_report_run(report_id) or db.get_latest_report_run()
        diagnostics = dict(report_run.get("quality_diagnostics") or {})
        diagnostics["model_path_breakdown"] = breakdown
        if refresh_summary:
            diagnostics["last_model_path_refresh"] = {
                **refresh_summary,
                "refreshed_at": datetime.now().isoformat(timespec="seconds"),
            }
        conn.execute(
            "UPDATE report_runs SET quality_diagnostics = ? WHERE report_id = ?",
            (json.dumps(diagnostics, ensure_ascii=False, sort_keys=True, default=str), report_id),
        )
        conn.commit()
        return breakdown
    finally:
        conn.close()


def backfill_latest_report_model_paths(
    limit: int = 10,
    dry_run: bool = False,
    include_v2_fallback: bool = False,
    report_id: str = "",
) -> Dict[str, Any]:
    from main import item_quality_flags, load_config
    from src.processors.llm_processor import LLMProcessor

    config, _ = load_runtime_config()
    db = Database(resolve_database_path(config, ROOT))
    report_id = str(report_id or "").strip()
    report_run = db.get_report_run(report_id) if report_id else db.get_latest_report_run()
    report_id = str(report_run.get("report_id", "") or report_id)
    if not report_id or not report_run:
        return {
            "report_id": report_id,
            "attempted": 0,
            "updated": 0,
            "failed": 0,
            "dry_run": bool(dry_run),
            "items": [],
            "model_path_breakdown": {},
            "error": "No matching report_runs snapshot exists.",
        }

    candidates = get_report_model_path_backfill_candidates(
        db,
        report_id,
        limit=limit,
        include_v2_fallback=include_v2_fallback,
    )
    if dry_run:
        return {
            "report_id": report_id,
            "attempted": 0,
            "updated": 0,
            "failed": 0,
            "dry_run": True,
            "items": candidates,
            "model_path_breakdown": normalize_model_path_breakdown(
                dict((report_run.get("quality_diagnostics") or {}).get("model_path_breakdown") or {})
            ),
            "error": "",
        }

    processor = LLMProcessor(config.get("llm", {}))
    updated_items: list[Dict[str, Any]] = []
    failed_items: list[Dict[str, Any]] = []
    for candidate in candidates:
        article = db.get_article_by_id(int(candidate["article_id"]))
        if not article:
            failed_items.append({**candidate, "error": "article_not_found"})
            continue
        result = processor.process_article(article)
        if not result:
            failed_items.append({**candidate, "error": "analysis_failed"})
            continue
        rewrite_attempts = int(article.get("rewrite_attempts", 0) or 0)
        if include_v2_fallback:
            existing_display = article
        else:
            existing_display = result
        db.update_article_processing(
            url=article["url"],
            summary=existing_display.get("summary", article.get("summary", "")),
            score=float(result.get("score", article.get("score", 0)) or 0),
            keywords=result.get("keywords", article.get("keywords", [])),
            category=result.get("category", article.get("category", "Other")),
            title_cn=existing_display.get("title_cn", article.get("title_cn", "")),
            summary_preview=existing_display.get("summary_preview", article.get("summary_preview", "")),
            why_it_matters=existing_display.get("why_it_matters", article.get("why_it_matters", "")),
            why_now=existing_display.get("why_now", article.get("why_now", "")),
            expected_effect=existing_display.get("expected_effect", article.get("expected_effect", "")),
            future_impact=existing_display.get("future_impact", article.get("future_impact", "")),
            facts=result.get("facts", article.get("facts", {})),
            evidence_quality=float(result.get("evidence_quality", article.get("evidence_quality", 0.0)) or 0.0),
            information_density=float(result.get("information_density", article.get("information_density", 0.0)) or 0.0),
            model_used=result.get("model_used", article.get("model_used", "")),
            analysis_version="v2",
            quality_flags=item_quality_flags(result),
            rewrite_attempts=rewrite_attempts,
        )
        snapshot = _merge_backfilled_snapshot(
            candidate["snapshot"],
            result,
            preserve_display_fields=include_v2_fallback,
        )
        conn = db._get_conn()
        try:
            conn.execute(
                """
                UPDATE report_items
                SET snapshot_json = ?
                WHERE report_id = ? AND article_id = ?
                """,
                (
                    json.dumps(snapshot, ensure_ascii=False, sort_keys=True, default=str),
                    report_id,
                    int(candidate["article_id"]),
                ),
            )
            conn.commit()
        finally:
            conn.close()
        updated_items.append(
            {
                "article_id": int(candidate["article_id"]),
                "rank": candidate["rank"],
                "section": candidate["section"],
                "title": snapshot.get("title_cn") or result.get("title_cn") or candidate.get("title", ""),
                "model_used": result.get("model_used", ""),
                "evidence_quality": result.get("evidence_quality", 0.0),
                "information_density": result.get("information_density", 0.0),
            }
        )

    refresh_summary = {
        "mode": "refresh_fallback_model_path" if include_v2_fallback else "backfill_legacy_model_path",
        "report_id": report_id,
        "attempted": len(candidates),
        "updated": len(updated_items),
        "failed": len(failed_items),
    }
    breakdown = _refresh_report_model_path_diagnostics(db, report_id, refresh_summary=refresh_summary)
    return {
        "report_id": report_id,
        "attempted": len(candidates),
        "updated": len(updated_items),
        "failed": len(failed_items),
        "dry_run": False,
        "items": updated_items,
        "failed_items": failed_items,
        "model_path_breakdown": breakdown,
        "error": "",
    }


def refresh_latest_report_fallback_model_paths(limit: int = 10, dry_run: bool = False, report_id: str = "") -> Dict[str, Any]:
    result = backfill_latest_report_model_paths(limit=limit, dry_run=dry_run, include_v2_fallback=True, report_id=report_id)
    result["mode"] = "refresh_fallback_model_path"
    return result


def _quote_command_arg(value: Any) -> str:
    text = str(value)
    if not text:
        return '""'
    if any(char.isspace() for char in text) or '"' in text:
        return '"' + text.replace('"', '\\"') + '"'
    return text


def build_task_repair_commands(scheduler_config: Dict[str, Any]) -> list[Dict[str, str]]:
    legacy_task_names = [str(name).lstrip("\\") for name in (scheduler_config.get("legacy_task_names") or [])]
    setup_script = Path(str(scheduler_config.get("task_setup_script", ROOT / "setup_scheduled_tasks.ps1")))
    if not setup_script.is_absolute():
        setup_script = (ROOT / setup_script).resolve()
    offline_script = Path(str(scheduler_config.get("offline_task_setup_script", ROOT / "setup_offline_tasks.ps1")))
    if not offline_script.is_absolute():
        offline_script = (ROOT / offline_script).resolve()
    repair_script = Path(str(scheduler_config.get("repair_task_script", ROOT / "repair_scheduled_tasks.ps1")))
    if not repair_script.is_absolute():
        repair_script = (ROOT / repair_script).resolve()
    doctor_command = [
        sys.executable,
        str(ROOT / "scheduler_runner.py"),
        "--doctor",
        "--record",
    ]

    commands: list[Dict[str, str]] = []
    require_offline_tasks = bool(scheduler_config.get("require_offline_tasks", False))
    if require_offline_tasks:
        commands.append(
            {
                "name": "repair_all_scheduled_tasks",
                "purpose": "Delete legacy tasks, rebuild offline-capable tasks, and record a fresh doctor snapshot.",
                "command": (
                    "powershell -NoProfile -ExecutionPolicy Bypass -File "
                    f"{_quote_command_arg(repair_script)}"
                ),
            }
        )
        commands.append(
            {
                "name": "repair_all_scheduled_tasks_s4u",
                "purpose": "Delete legacy tasks and rebuild S4U/background tasks without storing a Windows password.",
                "command": (
                    "powershell -NoProfile -ExecutionPolicy Bypass -File "
                    f"{_quote_command_arg(repair_script)} -UseS4U -NoPrompt"
                ),
            }
        )
    else:
        commands.append(
            {
                "name": "rebuild_interactive_send_tasks",
                "purpose": "Recreate 13:00 and 21:00 interactive send tasks for logged-in desktop use.",
                "command": (
                    "powershell -NoProfile -ExecutionPolicy Bypass -File "
                    f"{_quote_command_arg(setup_script)}"
                ),
            }
        )
    for task_name in legacy_task_names:
        if task_name:
            commands.append(
                {
                    "name": f"delete_legacy_{task_name}",
                    "purpose": "Remove an old scheduled task that can duplicate sends.",
                    "command": f"schtasks /Delete /TN {task_name} /F",
                }
            )
    if require_offline_tasks:
        commands.append(
            {
                "name": "rebuild_offline_tasks",
                "purpose": "Recreate 13:00 and 21:00 tasks with offline-capable Windows credentials.",
                "command": (
                    "powershell -NoProfile -ExecutionPolicy Bypass -File "
                    f"{_quote_command_arg(offline_script)}"
                ),
            }
        )
        commands.append(
            {
                "name": "rebuild_s4u_background_tasks",
                "purpose": "Recreate 13:00 and 21:00 tasks in S4U/background mode without storing a Windows password.",
                "command": (
                    "powershell -NoProfile -ExecutionPolicy Bypass -File "
                    f"{_quote_command_arg(offline_script)} -UseS4U"
                ),
            }
        )
    commands.append(
        {
            "name": "record_doctor_after_repair",
            "purpose": "Record a fresh health snapshot after task repair.",
            "command": " ".join(_quote_command_arg(part) for part in doctor_command),
        }
    )
    feedback_script = ROOT / "setup_feedback_server_task.ps1"
    commands.append(
        {
            "name": "install_feedback_server_task",
            "purpose": "Start the local email feedback service at Windows logon.",
            "command": " ".join(
                _quote_command_arg(part)
                for part in [
                    "powershell",
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(feedback_script),
                ]
            ),
        }
    )
    commands.append(
        {
            "name": "backfill_latest_report_model_path",
            "purpose": "Refresh legacy model-path items in the latest report snapshot without rerunning collection.",
            "command": " ".join(
                _quote_command_arg(part)
                for part in [
                    sys.executable,
                    str(ROOT / "scheduler_runner.py"),
                    "--backfill-model-path",
                    "--limit",
                    "10",
                ]
            ),
        }
    )
    commands.append(
        {
            "name": "refresh_fallback_model_path",
            "purpose": "Reprocess ambiguous fallback_v2 report snapshot items so future diagnostics can distinguish template and LLM fallback paths.",
            "command": " ".join(
                _quote_command_arg(part)
                for part in [
                    sys.executable,
                    str(ROOT / "scheduler_runner.py"),
                    "--refresh-fallback-model-path",
                    "--limit",
                    "10",
                ]
            ),
        }
    )
    return commands


def classify_quality_warnings(warnings: list[Any]) -> Dict[str, list[str]]:
    informational: list[str] = []
    actionable: list[str] = []
    for raw_warning in warnings:
        warning = str(raw_warning)
        if any(warning.startswith(prefix) for prefix in QUALITY_INFO_WARNING_PREFIXES):
            informational.append(warning)
        else:
            actionable.append(warning)
    return {"info": informational, "warn": actionable}


def build_codex_research_candidate_check(
    config: Dict[str, Any],
    inbox_config: Dict[str, Any],
    *,
    root: Path = ROOT,
    sent_history: Optional[list[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    configured_paths = dict(
        inbox_config.get("candidate_paths")
        or {
            "1300": "data/codex_research/candidate_1300.json",
            "2100": "data/codex_research/candidate_2100.json",
        }
    )
    production_collector = build_codex_research_inbox_collector(
        inbox_config,
        root=root,
    )
    production_path = production_collector.inbox_path
    production_mtime_ns = (
        production_path.stat().st_mtime_ns if production_path.exists() else 0
    )
    rows: list[Dict[str, Any]] = []
    pending_rows: list[Dict[str, Any]] = []
    history = sent_history

    for slot_id, raw_path in configured_paths.items():
        candidate_path = Path(str(raw_path))
        if not candidate_path.is_absolute():
            candidate_path = root / candidate_path
        row: Dict[str, Any] = {
            "slot_id": str(slot_id),
            "path": candidate_path.as_posix(),
            "exists": candidate_path.exists(),
            "newer_than_production": False,
            "status": "missing",
        }
        if not candidate_path.exists():
            rows.append(row)
            continue

        stat = candidate_path.stat()
        row.update(
            {
                "size_bytes": stat.st_size,
                "modified_at": datetime.fromtimestamp(
                    stat.st_mtime,
                    tz=timezone.utc,
                ).astimezone().isoformat(timespec="seconds"),
                "newer_than_production": stat.st_mtime_ns > production_mtime_ns,
            }
        )
        if not row["newer_than_production"]:
            row["status"] = "superseded"
            rows.append(row)
            continue

        candidate_collector = build_codex_research_inbox_collector(
            inbox_config,
            root=root,
            inbox_path_override=candidate_path,
        )
        items = candidate_collector.collect()
        if items:
            if history is None:
                history = Database(resolve_database_path(config, root)).get_recent_report_items(
                    days=int(inbox_config.get("history_dedupe_days", 7) or 7),
                    limit=int(inbox_config.get("history_dedupe_limit", 5000) or 5000),
                    sent_only=True,
                )
            candidate_collector.fetch_diagnostics.update(
                evaluate_sent_history_overlap(items, history)
            )
        readiness = build_codex_research_readiness_summary(
            candidate_collector.fetch_diagnostics,
            inbox_config,
        )
        candidate_diagnostics = candidate_collector.fetch_diagnostics
        row.update(
            {
                "status": (
                    "ready_not_promoted"
                    if readiness.get("ready_for_dry_run")
                    else "blocked"
                ),
                "quality_status": str(
                    candidate_collector.fetch_diagnostics.get("quality_status")
                    or "unknown"
                ),
                "schema_version": str(
                    candidate_collector.fetch_diagnostics.get("schema_version") or ""
                ),
                "inbox_sha256": str(
                    candidate_collector.fetch_diagnostics.get("inbox_sha256") or ""
                ),
                "accepted_counts": dict(readiness.get("accepted_counts") or {}),
                "blockers": list(readiness.get("blockers") or []),
                "top_rejection_reasons": list(
                    readiness.get("top_rejection_reasons") or []
                ),
                "discovery_candidate_count": int(
                    candidate_diagnostics.get("discovery_candidate_count", 0) or 0
                ),
                "discovery_section_counts": dict(
                    candidate_diagnostics.get("discovery_section_counts") or {}
                ),
                "submitted_section_counts": dict(
                    candidate_diagnostics.get("submitted_section_counts") or {}
                ),
                "attribution_opener_counts": dict(
                    candidate_diagnostics.get("attribution_opener_counts") or {}
                ),
                "attribution_opener_overuse_count": int(
                    candidate_diagnostics.get("attribution_opener_overuse_count", 0)
                    or 0
                ),
                "attribution_opener_overuse_examples": list(
                    candidate_diagnostics.get("attribution_opener_overuse_examples")
                    or []
                ),
                "cross_item_template_repeat_count": int(
                    candidate_diagnostics.get("cross_item_template_repeat_count", 0)
                    or 0
                ),
                "cross_item_template_repeat_examples": list(
                    candidate_diagnostics.get("cross_item_template_repeat_examples")
                    or []
                ),
            }
        )
        rows.append(row)
        pending_rows.append(row)

    if not pending_rows:
        return _doctor_item(
            "codex_research_candidates",
            "ok",
            "No unpromoted Codex research candidate is newer than the production inbox.",
            {
                "production_path": production_path.as_posix(),
                "candidates": rows,
            },
        )

    status_text = ", ".join(
        f"{row['slot_id']}={row['status']}" for row in pending_rows
    )
    return _doctor_item(
        "codex_research_candidates",
        "warn",
        f"Unpromoted Codex research candidate(s) found: {status_text}.",
        {
            "production_path": production_path.as_posix(),
            "candidates": rows,
        },
    )


def build_doctor_payload() -> Dict[str, Any]:
    config, scheduler_config = load_runtime_config()
    load_dotenv(ROOT / ".env")
    status_payload = build_scheduler_status_payload()
    repair_commands = build_task_repair_commands(scheduler_config)
    checks: list[Dict[str, Any]] = []

    configured_slots = list(scheduler_config.get("send_slots") or [])
    configured_slot_times = [str(slot.get("time", "") or "") for slot in configured_slots]
    expected_slot_times = {"13:00", "21:00"}
    if set(configured_slot_times) == expected_slot_times and len(configured_slot_times) == 2:
        checks.append(
            _doctor_item(
                "send_schedule_config",
                "ok",
                "Configured send windows are exactly 13:00 and 21:00.",
                {"send_slots": configured_slots},
            )
        )
    else:
        checks.append(
            _doctor_item(
                "send_schedule_config",
                "warn",
                f"Expected exactly 13:00 and 21:00 send windows, got: {', '.join(configured_slot_times) or 'none'}",
                {"send_slots": configured_slots},
            )
        )

    email_vars = {
        "EMAIL_RECIPIENT": bool(os.getenv("EMAIL_RECIPIENT")),
        "EMAIL_SENDER": bool(os.getenv("EMAIL_SENDER")),
        "EMAIL_PASSWORD": bool(os.getenv("EMAIL_PASSWORD")),
    }
    missing_email_vars = [name for name, present in email_vars.items() if not present]
    if missing_email_vars:
        checks.append(
            _doctor_item(
                "email_env",
                "fail",
                f"Missing required email settings: {', '.join(missing_email_vars)}",
                {"present": email_vars},
            )
        )

    else:
        checks.append(
            _doctor_item(
                "email_env",
                "ok",
                "Required email settings are present.",
                {"present": email_vars},
            )
        )

    llm_config = dict(config.get("llm") or {})
    if llm_config:
        llm_key_env = str(llm_config.get("api_key_env") or "OPENAI_API_KEY")
        llm_provider = str(llm_config.get("provider") or "openai")
        llm_model = str(llm_config.get("model") or "")
        if llm_provider == "codex_automation":
            inbox_config = dict((config.get("sources") or {}).get("codex_research_inbox") or {})
            collector = build_codex_research_inbox_collector(inbox_config, root=ROOT)
            items = collector.collect()
            if items:
                history_metrics = evaluate_sent_history_overlap(
                    items,
                    Database(resolve_database_path(config, ROOT)).get_recent_report_items(
                        days=int(inbox_config.get("history_dedupe_days", 7) or 7),
                        limit=int(inbox_config.get("history_dedupe_limit", 5000) or 5000),
                        sent_only=True,
                    ),
                )
                collector.fetch_diagnostics.update(history_metrics)
            inbox_status = str(collector.fetch_diagnostics.get("quality_status") or "unknown")
            inbox_ok = (
                inbox_status == "passed"
                and str(collector.fetch_diagnostics.get("production_ready_status") or "missing") == "passed"
            )
            inbox_failure = (
                inbox_status
                if inbox_status != "passed"
                else f"production_{collector.fetch_diagnostics.get('production_ready_status') or 'missing'}"
            )
            inbox_diagnostics = {
                "provider": llm_provider,
                "model": llm_model,
                "api_mode": str(llm_config.get("api_mode") or ""),
                **collector.fetch_diagnostics,
            }
            inbox_diagnostics["readiness_summary"] = build_codex_research_readiness_summary(
                inbox_diagnostics,
                inbox_config,
            )
            checks.append(
                _doctor_item(
                    "codex_research_inbox",
                    "ok" if inbox_ok else "fail",
                    (
                        "Fresh Codex research is ready for report generation."
                        if inbox_ok
                        else f"Codex research inbox is not ready: {inbox_failure}."
                    ),
                    inbox_diagnostics,
                )
            )
            checks.append(
                build_codex_research_candidate_check(
                    config,
                    inbox_config,
                )
            )
        else:
            llm_key_present = bool(os.getenv(llm_key_env))
            checks.append(
                _doctor_item(
                    "llm_env",
                    "ok" if llm_key_present else "fail",
                    (
                        f"{llm_provider} model {llm_model} is configured and {llm_key_env} is present."
                        if llm_key_present
                        else f"{llm_provider} model {llm_model} is configured but {llm_key_env} is missing."
                    ),
                    {
                        "provider": llm_provider,
                        "model": llm_model,
                        "api_mode": str(llm_config.get("api_mode") or ""),
                        "api_key_env": llm_key_env,
                        "api_key_present": llm_key_present,
                    },
                )
            )

    log_dir = Path(scheduler_config["log_dir"])
    if log_dir.exists():
        checks.append(
            _doctor_item(
                "log_dir",
                "ok",
                f"Log directory is available at {log_dir.as_posix()}",
            )
        )
    else:
        checks.append(
            _doctor_item(
                "log_dir",
                "fail",
                f"Log directory is missing: {log_dir.as_posix()}",
            )
        )

    archive_dir = ROOT / str(config.get("archive", {}).get("report_dir", "archive"))
    if archive_dir.exists():
        checks.append(
            _doctor_item(
                "archive_dir",
                "ok",
                f"Archive directory is available at {archive_dir.as_posix()}",
            )
        )
    else:
        checks.append(
            _doctor_item(
                "archive_dir",
                "warn",
                f"Archive directory does not exist yet: {archive_dir.as_posix()}",
            )
        )

    manifest_path = ROOT / "reports_manifest.json"
    if manifest_path.exists():
        checks.append(
            _doctor_item(
                "manifest",
                "ok",
                f"Archive manifest exists at {manifest_path.as_posix()}",
            )
        )
    else:
        checks.append(
            _doctor_item(
                "manifest",
                "warn",
                f"Archive manifest is missing: {manifest_path.as_posix()}",
            )
        )

    lock_summary = dict(status_payload.get("lock") or {})
    lock_state = str(lock_summary.get("state", "idle") or "idle")
    if lock_state == "stale":
        checks.append(
            _doctor_item(
                "lock",
                "fail",
                f"Scheduler lock is stale (pid={lock_summary.get('pid', '') or 'N/A'}).",
                lock_summary,
            )
        )
    elif lock_state == "active":
        checks.append(
            _doctor_item(
                "lock",
                "warn",
                f"Scheduler is currently running (pid={lock_summary.get('pid', '') or 'N/A'}).",
                lock_summary,
            )
        )
    else:
        checks.append(_doctor_item("lock", "ok", "Scheduler lock is idle.", lock_summary))

    launcher_status = build_scheduler_launcher_status(log_dir)
    if launcher_status.get("available"):
        launcher_state = str(launcher_status.get("status", "unknown") or "unknown")
        launcher_level = "ok" if launcher_state == "success" else ("warn" if launcher_state == "running" else "fail")
        checks.append(
            _doctor_item(
                "scheduler_launcher",
                launcher_level,
                f"Latest scheduler launcher status={launcher_state}, exit_code={launcher_status.get('exit_code', '')}.",
                launcher_status,
            )
        )

    disk_usage = shutil.disk_usage(ROOT)
    free_disk_mb = round(disk_usage.free / (1024 * 1024), 1)
    minimum_free_disk_mb = max(1, int(scheduler_config.get("minimum_free_disk_mb", 512)))
    critical_free_disk_mb = max(1, int(scheduler_config.get("critical_free_disk_mb", 32)))
    disk_level = "fail" if free_disk_mb < critical_free_disk_mb else "warn" if free_disk_mb < minimum_free_disk_mb else "ok"
    checks.append(
        _doctor_item(
            "disk_space",
            disk_level,
            (
                f"Workspace disk has {free_disk_mb:.1f} MB free; "
                f"warning threshold={minimum_free_disk_mb} MB, critical threshold={critical_free_disk_mb} MB."
            ),
            {
                "path": ROOT.as_posix(),
                "free_mb": free_disk_mb,
                "total_mb": round(disk_usage.total / (1024 * 1024), 1),
                "warning_threshold_mb": minimum_free_disk_mb,
                "critical_threshold_mb": critical_free_disk_mb,
            },
        )
    )

    last_run = dict(status_payload.get("last_run") or {})
    last_success = dict(status_payload.get("last_success") or {})
    if not last_run:
        checks.append(_doctor_item("last_run", "warn", "No previous run status file found yet."))
    else:
        last_status = str(last_run.get("status", "unknown") or "unknown")
        delivery_status = str(last_run.get("delivery_status", "unknown") or "unknown")
        if last_run.get("success", False) and delivery_status in {"sent", "skipped"}:
            checks.append(
                _doctor_item(
                    "last_run",
                    "ok",
                    f"Last run finished with status={last_status}, delivery={delivery_status}.",
                    {
                        "finished_at": last_run.get("finished_at", ""),
                        "log_file": last_run.get("log_file", ""),
                    },
                )
            )
        else:
            checks.append(
                _doctor_item(
                    "last_run",
                    "fail",
                    f"Last run finished with status={last_status}, delivery={delivery_status}.",
                    {
                        "finished_at": last_run.get("finished_at", ""),
                        "log_file": last_run.get("log_file", ""),
                    },
                )
            )

    if last_success:
        checks.append(
            _doctor_item(
                "last_success",
                "ok",
                f"Last successful email finished at {last_success.get('finished_at', '') or 'unknown'}.",
                {
                    "html_report_path": last_success.get("html_report_path", ""),
                    "markdown_report_path": last_success.get("markdown_report_path", ""),
                    "log_file": last_success.get("log_file", ""),
                },
            )
        )
    else:
        checks.append(_doctor_item("last_success", "warn", "No successful email snapshot has been recorded yet."))

    product_diagnostics = build_product_diagnostics(config, last_run, last_success, scheduler_config)
    feedback_status = dict(product_diagnostics.get("feedback_server_status") or {})
    if feedback_status.get("enabled"):
        checks.append(
            _doctor_item(
                "feedback_server",
                "ok" if feedback_status.get("healthy") else "warn",
                (
                    f"Feedback server is healthy at {feedback_status.get('host')}:{feedback_status.get('port')}."
                    if feedback_status.get("healthy")
                    else f"Feedback server is enabled but unhealthy at {feedback_status.get('host')}:{feedback_status.get('port')}: {feedback_status.get('error', '')}"
                ),
                feedback_status,
            )
        )
    checks.append(
        _doctor_item(
            "last_report_quality",
            "warn" if product_diagnostics.get("last_report_quality_status") == "failed" else "ok",
            f"Last report quality status: {product_diagnostics.get('last_report_quality_status')}",
            {
                "auto_rewrite_attempted_count": product_diagnostics.get("auto_rewrite_attempted_count", 0),
                "auto_rewrite_success_count": product_diagnostics.get("auto_rewrite_success_count", 0),
                "report_design_version": product_diagnostics.get("report_design_version", ""),
                "model_path_breakdown": product_diagnostics.get("model_path_breakdown", {}),
                "feedback_count_7d": product_diagnostics.get("feedback_count_7d", 0),
                "v3_quality_status": product_diagnostics.get("v3_quality_status", ""),
                "editorial_quality_status": product_diagnostics.get("editorial_quality_status", ""),
                "final_html_quality_status": product_diagnostics.get("final_html_quality_status", ""),
                "final_html_bad_title_count": product_diagnostics.get("final_html_bad_title_count", 0),
                "untranslated_fact_count": product_diagnostics.get("untranslated_fact_count", 0),
                "exact_duplicate_sentence_count": product_diagnostics.get("exact_duplicate_sentence_count", 0),
                "paper_mechanism_missing_count": product_diagnostics.get("paper_mechanism_missing_count", 0),
                "paper_result_context_missing_count": product_diagnostics.get("paper_result_context_missing_count", 0),
                "paper_intro_length_fail_count": product_diagnostics.get("paper_intro_length_fail_count", 0),
                "display_body_over_limit_count": product_diagnostics.get("display_body_over_limit_count", 0),
                "memory_total_chars": product_diagnostics.get("memory_total_chars", 0),
                "memory_budget_exceeded": product_diagnostics.get("memory_budget_exceeded", False),
                "focus_source_max_count": product_diagnostics.get("focus_source_max_count", 0),
                "focus_topic_max_count": product_diagnostics.get("focus_topic_max_count", 0),
                "appendix_body_overlap_count": product_diagnostics.get("appendix_body_overlap_count", 0),
                "truncated_focus_text_count": product_diagnostics.get("truncated_focus_text_count", 0),
                "visible_text_chars": product_diagnostics.get("visible_text_chars", 0),
                "html_size_bytes": product_diagnostics.get("html_size_bytes", 0),
                "html_size_kb": product_diagnostics.get("html_size_kb", 0.0),
                "email_clipping_warning": product_diagnostics.get("email_clipping_warning", False),
                "email_clipping_risk": product_diagnostics.get("email_clipping_risk", False),
                "focus_source_concentration": product_diagnostics.get("focus_source_concentration", 0.0),
                "memory_item_count": product_diagnostics.get("memory_item_count", 0),
                "featured_paper_count": product_diagnostics.get("featured_paper_count", 0),
                "deepseek_health_hint": product_diagnostics.get("deepseek_health_hint", ""),
                "deepseek_schema_valid_count": product_diagnostics.get("deepseek_schema_valid_count", 0),
                "gpt_health_hint": product_diagnostics.get("gpt_health_hint", ""),
                "gpt_schema_valid_count": product_diagnostics.get("gpt_schema_valid_count", 0),
                "gpt_empty_facts_count": product_diagnostics.get("gpt_empty_facts_count", 0),
                "gpt_key_field_missing_count": product_diagnostics.get("gpt_key_field_missing_count", 0),
                "gpt_search_request_count": product_diagnostics.get("gpt_search_request_count", 0),
                "gpt_search_success_count": product_diagnostics.get("gpt_search_success_count", 0),
                "gpt_search_schema_error_count": product_diagnostics.get("gpt_search_schema_error_count", 0),
                "llm_health_hint": product_diagnostics.get("llm_health_hint", ""),
                "llm_schema_valid_count": product_diagnostics.get("llm_schema_valid_count", 0),
                "mixed_language_title_count": product_diagnostics.get("mixed_language_title_count", 0),
                "field_label_leak_count": product_diagnostics.get("field_label_leak_count", 0),
                "low_info_expanded_count": product_diagnostics.get("low_info_expanded_count", 0),
                "mojibake_suspect_count": product_diagnostics.get("mojibake_suspect_count", 0),
                "generic_phrase_count": product_diagnostics.get("generic_phrase_count", 0),
                "duplicate_expression_count": product_diagnostics.get("duplicate_expression_count", 0),
                "paper_technical_intro_pass_count": product_diagnostics.get("paper_technical_intro_pass_count", 0),
                "paper_technical_intro_fail_count": product_diagnostics.get("paper_technical_intro_fail_count", 0),
                "paper_core_summary_pass_count": product_diagnostics.get("paper_core_summary_pass_count", 0),
                "paper_core_summary_fail_count": product_diagnostics.get("paper_core_summary_fail_count", 0),
                "physical_ai_featured_count": product_diagnostics.get("physical_ai_featured_count", 0),
                "physical_ai_item_count": product_diagnostics.get("physical_ai_item_count", 0),
                "paper_selected_count": product_diagnostics.get("paper_selected_count", 0),
                "paper_appendix_count": product_diagnostics.get("paper_appendix_count", 0),
                "title_repair": product_diagnostics.get("title_repair", {}),
                "report_structure": product_diagnostics.get("report_structure", {}),
                "source_weight_adjustments": product_diagnostics.get("source_weight_adjustments", {}),
                "latest_report_design_version": product_diagnostics.get("latest_report_design_version", ""),
                "configured_report_design_version": product_diagnostics.get("configured_report_design_version", ""),
                "continuity_new_item_count": product_diagnostics.get("continuity_new_item_count", 0),
                "continuity_updated_item_count": product_diagnostics.get("continuity_updated_item_count", 0),
                "repeated_focus_demoted_count": product_diagnostics.get("repeated_focus_demoted_count", 0),
                "topic_dossier_count": product_diagnostics.get("topic_dossier_count", 0),
                "reading_queue_count": product_diagnostics.get("reading_queue_count", 0),
                "paper_context_count": product_diagnostics.get("paper_context_count", 0),
                "adjacent_report_paper_overlap_rate": product_diagnostics.get("adjacent_report_paper_overlap_rate", 0.0),
                "paper_within_report_duplicate_count": product_diagnostics.get("paper_within_report_duplicate_count", 0),
                "adjacent_report_history_count": product_diagnostics.get("adjacent_report_history_count", 0),
                "cooldown_history_count": product_diagnostics.get("cooldown_history_count", 0),
                "final_overlap_reselected_count": product_diagnostics.get("final_overlap_reselected_count", 0),
                "paper_repeat_filtered_count": product_diagnostics.get("paper_repeat_filtered_count", 0),
                "fresh_paper_count": product_diagnostics.get("fresh_paper_count", 0),
                "reappeared_paper_with_update_count": product_diagnostics.get("reappeared_paper_with_update_count", 0),
                "arxiv_zero_result_warning_count": product_diagnostics.get("arxiv_zero_result_warning_count", 0),
                "paper_freshness_status": product_diagnostics.get("paper_freshness_status", "unknown"),
                "paper_domain_counts": product_diagnostics.get("paper_domain_counts", {}),
                "paper_domain_quota_status": product_diagnostics.get("paper_domain_quota_status", "unknown"),
                "paper_domain_quota_exceeded": product_diagnostics.get("paper_domain_quota_exceeded", {}),
                "paper_domain_quota_underfilled": product_diagnostics.get("paper_domain_quota_underfilled", {}),
            },
        )
    )
    paper_freshness_status = str(product_diagnostics.get("paper_freshness_status", "unknown") or "unknown")
    overlap_rate = float(product_diagnostics.get("adjacent_report_paper_overlap_rate", 0.0) or 0.0)
    within_report_duplicates = int(product_diagnostics.get("paper_within_report_duplicate_count", 0) or 0)
    if paper_freshness_status != "unknown":
        freshness_ok = (
            paper_freshness_status == "passed"
            and overlap_rate < 0.10
            and within_report_duplicates == 0
        )
        checks.append(
            _doctor_item(
                "paper_freshness",
                "ok" if freshness_ok else "warn",
                (
                    f"Paper freshness passed: adjacent overlap {overlap_rate:.1%}, "
                    f"within-report duplicates {within_report_duplicates}, "
                    f"fresh {product_diagnostics.get('fresh_paper_count', 0)}, "
                    f"filtered {product_diagnostics.get('paper_repeat_filtered_count', 0)}."
                    if freshness_ok
                    else (
                        f"Paper freshness needs attention: status={paper_freshness_status}, "
                        f"adjacent overlap={overlap_rate:.1%}, "
                        f"within-report duplicates={within_report_duplicates}."
                    )
                ),
                {
                    "adjacent_report_paper_overlap_rate": overlap_rate,
                    "paper_within_report_duplicate_count": within_report_duplicates,
                    "adjacent_report_history_count": product_diagnostics.get("adjacent_report_history_count", 0),
                    "cooldown_history_count": product_diagnostics.get("cooldown_history_count", 0),
                    "final_overlap_reselected_count": product_diagnostics.get("final_overlap_reselected_count", 0),
                    "paper_repeat_filtered_count": product_diagnostics.get("paper_repeat_filtered_count", 0),
                    "fresh_paper_count": product_diagnostics.get("fresh_paper_count", 0),
                    "reappeared_paper_with_update_count": product_diagnostics.get("reappeared_paper_with_update_count", 0),
                    "arxiv_zero_result_warning_count": product_diagnostics.get("arxiv_zero_result_warning_count", 0),
                    "arxiv_http_error_count": product_diagnostics.get("arxiv_http_error_count", 0),
                    "arxiv_parse_error_count": product_diagnostics.get("arxiv_parse_error_count", 0),
                    "arxiv_fallback_recovery_count": product_diagnostics.get("arxiv_fallback_recovery_count", 0),
                    "paper_freshness_status": paper_freshness_status,
                },
            )
        )
    paper_domain_quota_status = str(product_diagnostics.get("paper_domain_quota_status", "unknown") or "unknown")
    if paper_domain_quota_status != "unknown":
        exceeded = dict(product_diagnostics.get("paper_domain_quota_exceeded") or {})
        underfilled = dict(product_diagnostics.get("paper_domain_quota_underfilled") or {})
        checks.append(
            _doctor_item(
                "paper_domain_quotas",
                "fail" if exceeded or paper_domain_quota_status == "failed" else ("warn" if underfilled else "ok"),
                (
                    f"Paper domain quotas exceeded: {', '.join(exceeded) or 'unknown'}."
                    if exceeded or paper_domain_quota_status == "failed"
                    else f"Paper domain quotas passed; underfilled fresh domains: {', '.join(underfilled) or 'none'}."
                ),
                {
                    "status": paper_domain_quota_status,
                    "counts": product_diagnostics.get("paper_domain_counts", {}),
                    "exceeded": exceeded,
                    "underfilled": underfilled,
                },
            )
        )
    arxiv_http_errors = int(product_diagnostics.get("arxiv_http_error_count", 0) or 0)
    arxiv_parse_errors = int(product_diagnostics.get("arxiv_parse_error_count", 0) or 0)
    arxiv_zero_warnings = int(product_diagnostics.get("arxiv_zero_result_warning_count", 0) or 0)
    arxiv_true_zero = int(product_diagnostics.get("arxiv_true_zero_result_count", 0) or 0)
    arxiv_no_match = int(product_diagnostics.get("arxiv_no_match_result_count", 0) or 0)
    arxiv_recovered = int(product_diagnostics.get("arxiv_fallback_recovery_count", 0) or 0)
    arxiv_retry_paths = list(product_diagnostics.get("arxiv_retry_paths") or [])
    arxiv_collection_age = product_diagnostics.get("arxiv_collection_age_hours")
    arxiv_stale_hours = float((config.get("observability") or {}).get("arxiv_collection_stale_hours", 36) or 36)
    arxiv_stale = arxiv_collection_age is None or float(arxiv_collection_age) > arxiv_stale_hours
    arxiv_retry_path_text = format_arxiv_retry_paths(arxiv_retry_paths)
    if is_learning_digest_design(product_diagnostics.get("report_design_version")):
        arxiv_level = "fail" if arxiv_parse_errors else (
            "warn" if arxiv_http_errors or arxiv_zero_warnings or arxiv_stale else "ok"
        )
        checks.append(
            _doctor_item(
                "arxiv_collection_health",
                arxiv_level,
                (
                    f"arXiv collection: HTTP errors {arxiv_http_errors}, parse errors {arxiv_parse_errors}, "
                    f"explicit-zero sources {arxiv_true_zero}, no-match sources {arxiv_no_match}, "
                    f"fallback recoveries {arxiv_recovered}; "
                    f"{arxiv_retry_path_text}"
                    f"source report age {arxiv_collection_age if arxiv_collection_age is not None else 'unknown'}h."
                ),
                {
                    "arxiv_http_error_count": arxiv_http_errors,
                    "arxiv_parse_error_count": arxiv_parse_errors,
                    "arxiv_zero_result_warning_count": arxiv_zero_warnings,
                    "arxiv_true_zero_result_count": arxiv_true_zero,
                    "arxiv_no_match_result_count": arxiv_no_match,
                    "arxiv_fallback_recovery_count": arxiv_recovered,
                    "arxiv_retry_paths": arxiv_retry_paths,
                    "collection_report_id": product_diagnostics.get("arxiv_collection_report_id", ""),
                    "collection_created_at": product_diagnostics.get("arxiv_collection_created_at", ""),
                    "collection_age_hours": arxiv_collection_age,
                    "stale_after_hours": arxiv_stale_hours,
                    "stale": arxiv_stale,
                },
            )
        )
    freshness_acceptance = dict(product_diagnostics.get("paper_freshness_production_acceptance") or {})
    if (
        freshness_acceptance
        and is_learning_digest_design(product_diagnostics.get("report_design_version"))
        and str(product_diagnostics.get("report_product_mode") or "") != V11_PRODUCT_MODE
    ):
        acceptance_status = str(freshness_acceptance.get("status", "pending") or "pending")
        verified_days = int(freshness_acceptance.get("verified_days", 0) or 0)
        required_days = int(freshness_acceptance.get("required_days", 3) or 3)
        checks.append(
            _doctor_item(
                "paper_freshness_production_acceptance",
                "ok" if acceptance_status == "passed" else "warn",
                (
                    f"Paper freshness production acceptance passed for {verified_days}/{required_days} days."
                    if acceptance_status == "passed"
                    else (
                        "Paper freshness production acceptance failed; inspect report overlap and cooldown issues."
                        if acceptance_status == "failed"
                        else f"Paper freshness production acceptance is pending: {verified_days}/{required_days} days verified."
                    )
                ),
                freshness_acceptance,
            )
        )
    v8_acceptance = dict(product_diagnostics.get("v8_production_acceptance") or {})
    if v8_acceptance and str(product_diagnostics.get("report_design_version", "")) == "v8-editorial-reader":
        acceptance_status = str(v8_acceptance.get("status", "pending") or "pending")
        verified_count = int(v8_acceptance.get("verified_count", 0) or 0)
        required_count = int(v8_acceptance.get("required_count", 3) or 3)
        checks.append(
            _doctor_item(
                "v8_production_acceptance",
                "ok" if acceptance_status == "passed" else "warn",
                (
                    f"V8 production acceptance passed for {verified_count}/{required_count} consecutive sent reports."
                    if acceptance_status == "passed"
                    else (
                        f"V8 production acceptance failed; inspect the per-report issues."
                        if acceptance_status == "failed"
                        else f"V8 production acceptance is pending: {verified_count}/{required_count} sent reports verified."
                    )
                ),
                v8_acceptance,
            )
        )
    v11_acceptance = dict(product_diagnostics.get("v11_production_acceptance") or {})
    if v11_acceptance and str(product_diagnostics.get("report_product_mode") or "") == V11_PRODUCT_MODE:
        acceptance_status = str(v11_acceptance.get("status", "pending") or "pending")
        verified_days = int(v11_acceptance.get("verified_days", 0) or 0)
        required_days = int(v11_acceptance.get("required_days", 3) or 3)
        verified_reports = int(v11_acceptance.get("verified_report_count", 0) or 0)
        required_reports = int(v11_acceptance.get("required_report_count", 6) or 6)
        checks.append(
            _doctor_item(
                "v11_production_acceptance",
                "ok" if acceptance_status == "passed" else "warn",
                (
                    f"V11 production acceptance passed: {verified_days}/{required_days} days, "
                    f"{verified_reports}/{required_reports} sent reports."
                    if acceptance_status == "passed"
                    else (
                        "V11 production acceptance failed; inspect per-report quota, freshness, duplicate, and sample-quality issues."
                        if acceptance_status == "failed"
                        else (
                            f"V11 production acceptance is pending: {verified_days}/{required_days} days, "
                            f"{verified_reports}/{required_reports} sent reports."
                        )
                    )
                ),
                v11_acceptance,
            )
        )
    report_structure = dict(product_diagnostics.get("report_structure") or {})
    if report_structure:
        v8_report = str(product_diagnostics.get("report_design_version") or "") == "v8-editorial-reader"
        physical_ai_count = int(
            report_structure.get(
                "physical_ai_featured_count",
                report_structure.get("physical_ai_count", 0),
            )
            or 0
        )
        paper_selected_count = int(report_structure.get("paper_selected_count", 0) or 0)
        paper_appendix_count = int(report_structure.get("paper_appendix_count", 0) or 0)
        paper_core_fail_count = int(report_structure.get("paper_core_summary_fail_count", 0) or 0)
        concentration = int(report_structure.get("must_read_source_concentration", 0) or 0)
        english_leaks = int(
            report_structure.get(
                "paper_description_english_leak_count",
                report_structure.get("paper_plain_summary_english_leak_count", 0),
            )
            or 0
        )
        tracking_questions = int(report_structure.get("tracking_question_count", 0) or 0)
        editorial_status = str(report_structure.get("editorial_quality_status", "") or "")
        mixed_titles = int(report_structure.get("mixed_language_title_count", 0) or 0)
        field_leaks = int(report_structure.get("field_label_leak_count", 0) or 0)
        low_info_expanded = int(report_structure.get("low_info_expanded_count", 0) or 0)
        paper_technical_fail = int(report_structure.get("paper_technical_intro_fail_count", 0) or 0)
        level = "ok"
        problems = []
        if editorial_status == "failed":
            level = "warn"
            problems.append("editorial quality failed")
        if physical_ai_count < 4:
            level = "warn"
            problems.append(f"Physical AI only {physical_ai_count}")
        if mixed_titles:
            level = "warn"
            problems.append(f"mixed-language titles {mixed_titles}")
        if field_leaks:
            level = "warn"
            problems.append(f"field label leaks {field_leaks}")
        if low_info_expanded:
            level = "warn"
            problems.append(f"low-info expanded items {low_info_expanded}")
        if paper_technical_fail:
            level = "warn"
            problems.append(f"paper technical-intro failures {paper_technical_fail}")
        if paper_selected_count < 4:
            level = "warn"
            problems.append(f"featured papers only {paper_selected_count}")
        if paper_appendix_count < 4:
            level = "warn"
            problems.append(f"paper appendix only {paper_appendix_count}")
        if paper_core_fail_count and not v8_report:
            level = "warn"
            problems.append(f"paper core-summary failures {paper_core_fail_count}")
        if concentration > 2:
            level = "warn"
            problems.append(f"must-read source concentration {concentration}")
        if english_leaks and not v8_report:
            level = "warn"
            problems.append(f"paper description English leaks {english_leaks}")
        if tracking_questions < 3 and not v8_report:
            level = "warn"
            problems.append(f"tracking questions only {tracking_questions}")
        checks.append(
            _doctor_item(
                "report_structure",
                level,
                "Report structure looks balanced." if not problems else "Report structure needs attention: " + "; ".join(problems),
                report_structure,
            )
        )
    report_design_version = str(product_diagnostics.get("report_design_version") or "")
    if product_diagnostics.get("latest_report"):
        checks.append(
            _doctor_item(
                "report_design_version",
                "ok" if report_design_version else "warn",
                f"Latest report design version: {report_design_version or 'unknown'}",
                {
                    "report_design_version": report_design_version,
                    "latest_report_design_version": product_diagnostics.get("latest_report_design_version", ""),
                    "configured_report_design_version": product_diagnostics.get("configured_report_design_version", ""),
                },
            )
        )
    latest_source_health = dict(product_diagnostics.get("source_health") or {})
    if latest_source_health:
        unstable_source_count = int(latest_source_health.get("unstable_source_count", 0) or 0)
        checks.append(
            _doctor_item(
                "report_source_health",
                "warn" if unstable_source_count else "ok",
                f"Latest report source health: {unstable_source_count} unstable source(s).",
                latest_source_health,
            )
        )
    post_send_quality_scan = dict(product_diagnostics.get("post_send_quality_scan") or {})
    if post_send_quality_scan:
        post_scan_error = str(post_send_quality_scan.get("error", "") or "")
        post_scan_issue_count = int(post_send_quality_scan.get("issue_count", 0) or 0)
        post_scan_focus_issue_count = int(post_send_quality_scan.get("focus_issue_count", post_scan_issue_count) or 0)
        post_scan_brief_issue_count = int(post_send_quality_scan.get("brief_issue_count", 0) or 0)
        post_scan_scanned = int(post_send_quality_scan.get("scanned", 0) or 0)
        checks.append(
            _doctor_item(
                "post_send_quality_scan",
                "warn" if post_scan_error or post_scan_focus_issue_count else "ok",
                (
                    f"Post-send quality scan checked {post_scan_scanned} item(s), "
                    f"found {post_scan_focus_issue_count} focus issue(s), "
                    f"{post_scan_brief_issue_count} brief issue(s), {post_scan_issue_count} total issue(s)."
                    if not post_scan_error
                    else f"Post-send quality scan failed: {post_scan_error}"
                ),
                post_send_quality_scan,
            )
        )
    ui_audit = dict(product_diagnostics.get("ui_audit") or {})
    if ui_audit:
        ui_audit_status = str(ui_audit.get("status") or "unknown")
        ui_audit_render_count = int(ui_audit.get("render_count", 0) or 0)
        ui_audit_failed_render_count = int(ui_audit.get("failed_render_count", 0) or 0)
        checks.append(
            _doctor_item(
                "email_ui_audit",
                "ok" if ui_audit_status == "passed" and ui_audit_failed_render_count == 0 else "warn",
                (
                    f"Email UI audit passed all {ui_audit_render_count} render(s)."
                    if ui_audit_status == "passed" and ui_audit_failed_render_count == 0
                    else (
                        f"Email UI audit status={ui_audit_status}, "
                        f"failed renders={ui_audit_failed_render_count}/{ui_audit_render_count}."
                    )
                ),
                ui_audit,
            )
        )
    model_path_breakdown = dict(product_diagnostics.get("model_path_breakdown") or {})
    legacy_pending_count = int(model_path_breakdown.get("legacy_pending_backfill", 0) or 0)
    if legacy_pending_count:
        backfill_command = next(
            (command for command in repair_commands if command.get("name") == "backfill_latest_report_model_path"),
            {},
        )
        checks.append(
            _doctor_item(
                "model_path_backfill",
                "warn",
                f"Latest report has {legacy_pending_count} item(s) still using legacy analysis metadata.",
                {
                    "legacy_pending_backfill": legacy_pending_count,
                    "command": backfill_command.get("command", ""),
                },
            )
        )

    status_current_time = _parse_status_datetime(status_payload.get("current_time")) or datetime.now()
    send_calendar = build_send_calendar_payload(
        scheduler_config,
        target_datetime=status_current_time,
        last_run=last_run,
        last_success=last_success,
    )
    missed_slots = [
        slot for slot in list(send_calendar.get("slots") or [])
        if str(slot.get("health_status", "") or "") == "missed"
    ]
    checks.append(
        _doctor_item(
            "send_calendar",
            "warn" if missed_slots else "ok",
            (
                "Missed send slots detected today: "
                + ", ".join(f"{slot.get('time')} ({slot.get('slot_id')})" for slot in missed_slots)
                if missed_slots
                else "No missed send slots detected for today."
            ),
            {"calendar": send_calendar, "missed_slots": missed_slots},
        )
    )

    observability_config = dict(config.get("observability") or {})
    if observability_config.get("require_production_diagnostics", False) and last_success:
        if last_success.get("quality_diagnostics") and last_success.get("source_health"):
            checks.append(_doctor_item("production_diagnostics", "ok", "Last successful production send includes diagnostics."))
        else:
            checks.append(
                _doctor_item(
                    "production_diagnostics",
                    "warn",
                    "Last successful production send does not include quality/source diagnostics yet. The next real send will populate them.",
                    {
                        "last_success_finished_at": last_success.get("finished_at", ""),
                        "html_report_path": last_success.get("html_report_path", ""),
                    },
                )
            )

    arrival_config = dict(config.get("email", {}).get("arrival_check", {}) or {})
    if arrival_config.get("enabled", False):
        resolved_imap_server = resolve_imap_server(
            smtp_server=os.getenv("EMAIL_SMTP_SERVER", "smtp.gmail.com"),
            configured_imap_server=os.getenv("EMAIL_IMAP_SERVER", str(arrival_config.get("imap_server", ""))),
        )
        imap_config = {
            "imap_server": bool(resolved_imap_server),
            "username": bool(os.getenv("EMAIL_IMAP_USERNAME") or os.getenv("EMAIL_SENDER")),
            "password": bool(os.getenv("EMAIL_IMAP_PASSWORD") or os.getenv("EMAIL_PASSWORD")),
        }
        missing_imap = [name for name, present in imap_config.items() if not present]
        if missing_imap:
            checks.append(
                _doctor_item(
                    "delivery_arrival_config",
                    "warn",
                    "Mailbox arrival verification is enabled but IMAP settings are incomplete: "
                    + ", ".join(missing_imap),
                    {"present": imap_config},
                )
            )
        elif last_success:
            verification = dict(last_success.get("delivery_verification") or {})
            if delivery_verification_passed(verification):
                checks.append(_doctor_item("delivery_arrival", "ok", "Mailbox arrival was verified for the last successful send.", verification))
            else:
                checks.append(
                    _doctor_item(
                        "delivery_arrival",
                        "warn",
                        f"Mailbox arrival is enabled but not verified yet: {verification.get('status', 'missing_verification')}",
                        verification,
                    )
                )
        else:
            checks.append(_doctor_item("delivery_arrival", "warn", "Mailbox arrival verification is enabled, but no successful send exists yet."))

    quality_diagnostics = dict(last_run.get("quality_diagnostics") or last_success.get("quality_diagnostics") or {})
    quality_warnings = list(quality_diagnostics.get("warnings") or [])
    if quality_diagnostics:
        classified_quality_warnings = classify_quality_warnings(quality_warnings)
        actionable_quality_warnings = classified_quality_warnings["warn"]
        informational_quality_warnings = classified_quality_warnings["info"]
        quality_payload = dict(quality_diagnostics)
        quality_payload["warning_levels"] = classified_quality_warnings
        checks.append(
            _doctor_item(
                "quality_diagnostics",
                "warn" if actionable_quality_warnings else "ok",
                (
                    "Quality diagnostics recorded actionable warnings: " + ", ".join(actionable_quality_warnings[:5])
                    if actionable_quality_warnings
                    else (
                        "Quality diagnostics only recorded informational notes: "
                        + ", ".join(informational_quality_warnings[:5])
                        if informational_quality_warnings
                        else "Quality diagnostics are present and no selection warnings were recorded."
                    )
                ),
                quality_payload,
            )
        )

    source_health = (
        {}
        if latest_source_health
        else dict(last_run.get("source_health") or last_success.get("source_health") or {})
    )
    risky_source_count = int(source_health.get("risky_source_count", 0) or 0)
    if source_health:
        checks.append(
            _doctor_item(
                "source_health",
                "warn" if risky_source_count else "ok",
                (
                    f"Source health has {risky_source_count} risky source(s)."
                    if risky_source_count
                    else "Source health has no risky sources."
                ),
                {
                    "source_count": source_health.get("source_count", 0),
                    "risky_source_count": risky_source_count,
                    "risky_rows": source_health.get("risky_rows", []),
                },
            )
        )

    legacy_task_infos = list(status_payload.get("legacy_tasks") or [])
    active_legacy_tasks = [task for task in legacy_task_infos if task.get("available", False) and is_task_enabled(task)]
    if active_legacy_tasks:
        checks.append(
            _doctor_item(
                "duplicate_legacy_tasks",
                "warn",
                "Legacy Web_Agent scheduled tasks are still enabled and may duplicate sends: "
                + ", ".join(str(task.get("task_name", "")) for task in active_legacy_tasks),
                {"tasks": active_legacy_tasks},
            )
        )
    elif legacy_task_infos:
        checks.append(
            _doctor_item(
                "duplicate_legacy_tasks",
                "ok",
                "No enabled legacy Web_Agent send tasks were found.",
                {"tasks": legacy_task_infos},
            )
        )

    task_infos = list(status_payload.get("tasks") or [])
    primary_task_names = {str(name) for name in (scheduler_config.get("task_names") or [])}
    require_offline_tasks = bool(scheduler_config.get("require_offline_tasks", False))
    permission_denied_tasks = [
        task
        for task in task_infos
        if str(task.get("last_result_hex", "") or "") == "0x800710E0"
        or str(task.get("last_result", "") or "") == "-2147020576"
    ]
    if active_legacy_tasks or permission_denied_tasks:
        repair_issue_names = []
        if active_legacy_tasks:
            repair_issue_names.append("enabled legacy tasks")
        if permission_denied_tasks:
            repair_issue_names.append("Windows permission denied tasks")
        checks.append(
            _doctor_item(
                "task_repair_commands",
                "warn",
                "Repair commands are available for: " + ", ".join(repair_issue_names),
                {
                    "permission_denied_tasks": permission_denied_tasks,
                    "active_legacy_tasks": active_legacy_tasks,
                    "commands": repair_commands,
                },
            )
        )
    for task in task_infos:
        task_name = str(task.get("task_name", "unknown") or "unknown")
        if not task.get("available", False):
            checks.append(
                _doctor_item(
                    f"task:{task_name}",
                    "fail",
                    f"Scheduled task is unavailable: {task.get('error', 'query failed')}",
                    task,
                )
            )
            continue

        result_hint = str(task.get("last_result_hint", "unknown") or "unknown")
        result_message = str(task.get("last_result_message", "") or "")
        offline_issue = require_offline_tasks and task_name.lstrip("\\") in primary_task_names and is_interactive_task(task)
        if result_hint in DOCTOR_PASSING_HINTS:
            level = "warn" if offline_issue else "ok"
            detail = (
                f"Task status={task.get('status', 'unknown')}, "
                f"next_run={task.get('next_run_time', 'N/A')}, "
                f"last_result={task.get('last_result', 'N/A')} ({result_hint})"
            )
            if offline_issue:
                detail = f"{detail}. Offline mode required but this task still uses an interactive logon mode."
            checks.append(
                _doctor_item(
                    f"task:{task_name}",
                    level,
                    detail,
                    task,
                )
            )
        else:
            detail = (
                f"Task status={task.get('status', 'unknown')}, "
                f"next_run={task.get('next_run_time', 'N/A')}, "
                f"last_result={task.get('last_result', 'N/A')} ({result_hint})"
            )
            if result_message:
                detail = f"{detail}, message={result_message}"
            checks.append(_doctor_item(f"task:{task_name}", "warn", detail, task))

    counts = {
        "ok": sum(1 for item in checks if item["level"] == "ok"),
        "warn": sum(1 for item in checks if item["level"] == "warn"),
        "fail": sum(1 for item in checks if item["level"] == "fail"),
    }
    overall = "fail" if counts["fail"] else "warn" if counts["warn"] else "ok"
    return {
        "current_time": datetime.now().isoformat(timespec="seconds"),
        "overall": overall,
        "counts": counts,
        "checks": checks,
        "status": status_payload,
        "repair_commands": repair_commands,
        **product_diagnostics,
    }


def write_doctor_snapshot(payload: Dict[str, Any], scheduler_config: Optional[Dict[str, Any]] = None) -> Path:
    if scheduler_config is None:
        _, scheduler_config = load_runtime_config()
    snapshot_path = Path(str(scheduler_config["doctor_status_file"]))
    write_json(snapshot_path, payload)
    return snapshot_path


def _doctor_issue_signature(payload: Dict[str, Any]) -> str:
    parts = []
    for item in payload.get("checks", []):
        level = str(item.get("level", "") or "")
        if level == "ok":
            continue
        parts.append(f"{item.get('name', '')}:{level}:{item.get('detail', '')}")
    return "|".join(parts)


def update_doctor_history(payload: Dict[str, Any], scheduler_config: Dict[str, Any]) -> Dict[str, Any]:
    history_path = Path(str(scheduler_config["doctor_history_file"]))
    history = read_json(history_path)
    entries = list(history.get("entries") or [])
    current_entry = {
        "checked_at": payload.get("current_time", datetime.now().isoformat(timespec="seconds")),
        "overall": payload.get("overall", "unknown"),
        "counts": dict(payload.get("counts") or {}),
        "signature": _doctor_issue_signature(payload),
    }
    entries.append(current_entry)
    entries = entries[-20:]

    warn_streak = 0
    for entry in reversed(entries):
        if str(entry.get("overall", "")) == "warn":
            warn_streak += 1
            continue
        break

    if str(payload.get("overall", "")) == "ok":
        last_alert_signature = ""
        last_alert_overall = ""
    else:
        last_alert_signature = str(history.get("last_alert_signature", "") or "")
        last_alert_overall = str(history.get("last_alert_overall", "") or "")

    updated_history = {
        "entries": entries,
        "warn_streak": warn_streak,
        "last_alert_signature": last_alert_signature,
        "last_alert_overall": last_alert_overall,
        "last_alert_sent_at": str(history.get("last_alert_sent_at", "") or ""),
    }
    write_json(history_path, updated_history)
    return updated_history


def build_doctor_alert_html(payload: Dict[str, Any], history: Dict[str, Any]) -> str:
    non_ok_items = [
        item for item in payload.get("checks", [])
        if str(item.get("level", "")) != "ok"
    ]
    status_payload = dict(payload.get("status") or {})
    last_success = dict(status_payload.get("last_success") or {})
    top_issues = non_ok_items[:5]
    issues_html = "".join(
        f"<li><strong>{item.get('name', 'unknown')}</strong>: {item.get('detail', '')}</li>"
        for item in top_issues
    ) or "<li>No specific non-ok checks were recorded.</li>"
    extra_count = max(0, len(non_ok_items) - len(top_issues))
    extra_html = f"<p>另有 {extra_count} 项非 OK，请查看 doctor_latest.json。</p>" if extra_count else ""
    recent_success_html = (
        f"<li>最近成功：{last_success.get('finished_at', 'N/A')} | {last_success.get('html_report_path', 'N/A')}</li>"
        if last_success
        else "<li>最近成功：未记录</li>"
    )
    repair_commands = list(payload.get("repair_commands") or [])
    repair_command_text = "\n".join(str(command.get("command", "") or "") for command in repair_commands)
    if "setup_offline_tasks.ps1" in repair_command_text or "-UseS4U" in repair_command_text:
        repair_hint_html = (
            "先运行 <code>python D:\\Web_Agent\\scheduler_runner.py --doctor --self-heal</code>；"
            "如果是离线模式告警，请执行 <code>setup_offline_tasks.ps1</code> 并输入 Windows 密码。"
        )
    else:
        repair_hint_html = (
            "先运行 <code>python D:\\Web_Agent\\scheduler_runner.py --doctor --self-heal</code>；"
            "如需重建 13:00/21:00 发送任务，请执行 <code>setup_scheduled_tasks.ps1</code>。"
        )
    return f"""
    <html lang="zh-CN">
    <body style="font-family:Segoe UI,Microsoft YaHei,sans-serif;color:#16212f;">
        <h2>AI 日报需要处理：{payload.get("overall", "unknown")}</h2>
        <ul>
            <li>检查时间：{payload.get("current_time", "")}</li>
            <li>ok / warn / fail：{payload.get("counts", {}).get("ok", 0)} / {payload.get("counts", {}).get("warn", 0)} / {payload.get("counts", {}).get("fail", 0)}</li>
            <li>连续 warn：{history.get("warn_streak", 0)}</li>
            {recent_success_html}
        </ul>
        <p><strong>优先处理：</strong></p>
        <ul>{issues_html}</ul>
        {extra_html}
        <p>建议动作：{repair_hint_html}</p>
    </body>
    </html>
    """


def collect_task_self_heal_candidates(
    payload: Dict[str, Any],
    scheduler_config: Optional[Dict[str, Any]] = None,
) -> list[Dict[str, Any]]:
    candidates: list[Dict[str, Any]] = []
    status_payload = dict(payload.get("status") or {})
    scheduler_config = scheduler_config or {}
    primary_task_names = {str(name) for name in (scheduler_config.get("task_names") or [])}
    require_offline_tasks = bool(scheduler_config.get("require_offline_tasks", False))
    for task in list(status_payload.get("legacy_tasks") or []):
        task_name = str(task.get("task_name", "unknown") or "unknown")
        if task.get("available", False) and is_task_enabled(task):
            candidates.append(
                {
                    "task_name": task_name,
                    "reason": "enabled_legacy_task",
                    "detail": "Legacy scheduled task is still enabled and may duplicate sends.",
                }
            )

    for task in list(status_payload.get("tasks") or []):
        task_name = str(task.get("task_name", "unknown") or "unknown")
        if not task.get("available", False):
            candidates.append(
                {
                    "task_name": task_name,
                    "reason": "unavailable",
                    "detail": str(task.get("error", "query failed") or "query failed"),
                }
            )
            continue

        if (
            require_offline_tasks
            and task_name.lstrip("\\") in primary_task_names
            and is_interactive_task(task)
        ):
            candidates.append(
                {
                    "task_name": task_name,
                    "reason": "offline_interactive_required",
                    "detail": "Offline mode is required but the task still uses an interactive logon mode.",
                }
            )
            continue

        result_hint = str(task.get("last_result_hint", "unknown") or "unknown")
        if result_hint in DOCTOR_PASSING_HINTS:
            continue
        candidates.append(
            {
                "task_name": task_name,
                "reason": "bad_last_result",
                "detail": (
                    f"last_result={task.get('last_result', 'N/A')} "
                    f"({task.get('last_result_hex', 'N/A')}, {result_hint}) "
                    f"{task.get('last_result_message', '') or ''}"
                ).strip(),
            }
        )
    return candidates


def export_scheduled_task_xml(task_name: str, backup_dir: Path) -> Dict[str, Any]:
    clean_name = str(task_name or "").lstrip("\\")
    safe_name = clean_name.replace("\\", "_").replace("/", "_")
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / f"{safe_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xml"
    command = ["schtasks", "/Query", "/TN", clean_name, "/XML"]
    result: Dict[str, Any] = {
        "task_name": clean_name,
        "backup_path": backup_path.as_posix(),
        "command": command,
        "success": False,
        "returncode": None,
        "stdout": "",
        "stderr": "",
    }
    if not clean_name:
        result["stderr"] = "Task name is empty."
        return result

    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    completed = subprocess.run(
        command,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=creationflags,
        check=False,
    )
    result["returncode"] = completed.returncode
    result["stdout"] = (completed.stdout or "").strip()
    result["stderr"] = (completed.stderr or "").strip()
    result["success"] = completed.returncode == 0 and bool(result["stdout"])
    if result["success"]:
        backup_path.write_text(str(result["stdout"]), encoding="utf-8")
        result["stdout"] = ""
    return result


def delete_scheduled_task(task_name: str, backup_dir: Optional[Path] = None) -> Dict[str, Any]:
    clean_name = str(task_name or "").lstrip("\\")
    command = ["schtasks", "/Delete", "/TN", clean_name, "/F"]
    result: Dict[str, Any] = {
        "task_name": clean_name,
        "command": command,
        "backup": {},
        "success": False,
        "returncode": None,
        "stdout": "",
        "stderr": "",
    }
    if not clean_name:
        result["stderr"] = "Task name is empty."
        return result
    if backup_dir is not None:
        backup_result = export_scheduled_task_xml(clean_name, backup_dir)
        result["backup"] = backup_result
        if not backup_result.get("success", False):
            result["stderr"] = "Scheduled task backup failed; deletion was not attempted."
            return result

    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    completed = subprocess.run(
        command,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=creationflags,
        check=False,
    )
    result["returncode"] = completed.returncode
    result["stdout"] = (completed.stdout or "").strip()
    result["stderr"] = (completed.stderr or "").strip()
    result["success"] = completed.returncode == 0
    return result


def run_task_self_heal(
    scheduler_config: Dict[str, Any],
    candidates: list[Dict[str, Any]],
    dry_run: bool = False,
) -> Dict[str, Any]:
    legacy_candidates = [
        candidate for candidate in candidates
        if str(candidate.get("reason", "") or "") == "enabled_legacy_task"
    ]
    repair_candidates = [
        candidate for candidate in candidates
        if str(candidate.get("reason", "") or "") != "enabled_legacy_task"
    ]
    primary_task_names = {str(name) for name in (scheduler_config.get("task_names") or [])}
    require_offline_tasks = bool(scheduler_config.get("require_offline_tasks", False))
    needs_offline_rebuild = False
    for candidate in repair_candidates:
        reason = str(candidate.get("reason", "") or "")
        task_name = str(candidate.get("task_name", "") or "").lstrip("\\")
        if reason == "offline_interactive_required":
            needs_offline_rebuild = True
            break
        if require_offline_tasks and task_name in primary_task_names and reason in {"bad_last_result", "unavailable"}:
            needs_offline_rebuild = True
            break
    script_key = "offline_task_setup_script" if needs_offline_rebuild else "task_setup_script"
    default_script = "setup_offline_tasks.ps1" if needs_offline_rebuild else "setup_scheduled_tasks.ps1"
    script_path = Path(str(scheduler_config.get(script_key, ROOT / default_script)))
    if not script_path.is_absolute():
        script_path = (ROOT / script_path).resolve()

    command = [
        "powershell",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(script_path),
    ]
    display_command = list(command)
    if needs_offline_rebuild:
        user_env = str(scheduler_config.get("run_as_user_env", "WEB_AGENT_RUNAS_USER") or "WEB_AGENT_RUNAS_USER")
        password_env = str(scheduler_config.get("run_as_password_env", "WEB_AGENT_RUNAS_PASSWORD") or "WEB_AGENT_RUNAS_PASSWORD")
        run_as_user = os.getenv(user_env, "").strip()
        run_as_password = os.getenv(password_env, "")
        if run_as_user:
            command.extend(["-RunAsUser", run_as_user])
            display_command.extend(["-RunAsUser", run_as_user])
        if run_as_password:
            command.extend(["-RunAsPassword", run_as_password])
            display_command.extend(["-RunAsPassword", "<redacted>"])

    result: Dict[str, Any] = {
        "attempted": bool(candidates),
        "dry_run": dry_run,
        "candidates": candidates,
        "script_path": script_path.as_posix(),
        "command": display_command,
        "success": False,
        "returncode": None,
        "stdout": "",
        "stderr": "",
        "message": "",
        "legacy_cleanup": {
            "attempted": bool(legacy_candidates),
            "success": False,
            "candidates": legacy_candidates,
            "results": [],
        },
    }
    if not candidates:
        result["message"] = "No scheduled task repair candidates were found."
        return result
    if dry_run:
        result["legacy_cleanup"]["success"] = bool(legacy_candidates)
        result["success"] = True
        result["message"] = "Dry run only. No changes were applied."
        return result

    if legacy_candidates:
        backup_dir = Path(str(scheduler_config.get("task_backup_dir", ROOT / "logs/task_backups")))
        cleanup_results = [
            delete_scheduled_task(str(candidate.get("task_name", "") or ""), backup_dir=backup_dir)
            for candidate in legacy_candidates
        ]
        result["legacy_cleanup"]["results"] = cleanup_results
        result["legacy_cleanup"]["success"] = all(item.get("success", False) for item in cleanup_results)

    if not repair_candidates:
        result["success"] = bool(result["legacy_cleanup"].get("success", False))
        result["message"] = (
            "Legacy scheduled task cleanup completed."
            if result["success"]
            else "Legacy scheduled task cleanup failed."
        )
        return result

    if not script_path.exists():
        result["message"] = f"Task setup script is missing: {script_path.as_posix()}"
        return result
    if needs_offline_rebuild and "-RunAsPassword" not in command:
        password_env = str(scheduler_config.get("run_as_password_env", "WEB_AGENT_RUNAS_PASSWORD") or "WEB_AGENT_RUNAS_PASSWORD")
        result["message"] = (
            "Offline scheduled-task repair requires a Windows password. "
            f"Set {password_env} for unattended repair or run setup_offline_tasks.ps1 interactively."
        )
        return result

    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    completed = subprocess.run(
        command,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=creationflags,
        check=False,
    )
    result["returncode"] = completed.returncode
    result["stdout"] = (completed.stdout or "").strip()
    result["stderr"] = (completed.stderr or "").strip()
    legacy_cleanup_ok = (
        not result["legacy_cleanup"].get("attempted", False)
        or bool(result["legacy_cleanup"].get("success", False))
    )
    result["success"] = completed.returncode == 0 and legacy_cleanup_ok
    result["message"] = "Scheduled task repair completed." if result["success"] else "Scheduled task repair failed."
    return result


def build_repair_plan(payload: Dict[str, Any], scheduler_config: Dict[str, Any]) -> Dict[str, Any]:
    candidates = collect_task_self_heal_candidates(payload, scheduler_config)
    legacy_candidates = [
        candidate for candidate in candidates
        if str(candidate.get("reason", "") or "") == "enabled_legacy_task"
    ]
    repair_candidates = [
        candidate for candidate in candidates
        if str(candidate.get("reason", "") or "") != "enabled_legacy_task"
    ]
    password_env = str(scheduler_config.get("run_as_password_env", "WEB_AGENT_RUNAS_PASSWORD") or "WEB_AGENT_RUNAS_PASSWORD")
    password_present = bool(os.getenv(password_env, ""))
    require_offline_tasks = bool(scheduler_config.get("require_offline_tasks", False))
    runner_command = " ".join(
        _quote_command_arg(part)
        for part in [sys.executable, str(ROOT / "scheduler_runner.py"), "--doctor", "--self-heal"]
    )
    dry_run_command = f"{runner_command} --dry-run"
    legacy_cleanup_command = " ".join(
        _quote_command_arg(part)
        for part in [sys.executable, str(ROOT / "scheduler_runner.py"), "--cleanup-legacy-tasks", "--confirm"]
    )
    repair_script = Path(str(scheduler_config.get("repair_task_script", ROOT / "repair_scheduled_tasks.ps1")))
    if not repair_script.is_absolute():
        repair_script = (ROOT / repair_script).resolve()
    setup_script = Path(str(scheduler_config.get("task_setup_script", ROOT / "setup_scheduled_tasks.ps1")))
    if not setup_script.is_absolute():
        setup_script = (ROOT / setup_script).resolve()
    repair_script_command = (
        "powershell -NoProfile -ExecutionPolicy Bypass -File "
        f"{_quote_command_arg(repair_script)}"
    )
    s4u_repair_script_command = f"{repair_script_command} -UseS4U -NoPrompt"
    interactive_setup_command = (
        "powershell -NoProfile -ExecutionPolicy Bypass -File "
        f"{_quote_command_arg(setup_script)}"
    )

    action_items: list[Dict[str, Any]] = [
        {
            "step": 1,
            "name": "preview_self_heal",
            "description": "Preview the cleanup and rebuild candidates without changing scheduled tasks.",
            "command": dry_run_command,
            "destructive": False,
            "requires_password": False,
            "blocked": False,
        }
    ]
    if legacy_candidates:
        backup_dir = Path(str(scheduler_config.get("task_backup_dir", ROOT / "logs/task_backups")))
        action_items.append(
            {
                "step": len(action_items) + 1,
                "name": "delete_legacy_tasks",
                "description": "Back up and delete legacy scheduled tasks that can duplicate noon/evening sends.",
                "command": legacy_cleanup_command,
                "destructive": True,
                "requires_password": False,
                "blocked": False,
                "backup_dir": backup_dir.as_posix(),
                "tasks": [str(candidate.get("task_name", "") or "") for candidate in legacy_candidates],
            }
        )
    if repair_candidates:
        if not require_offline_tasks:
            action_items.append(
                {
                    "step": len(action_items) + 1,
                    "name": "rebuild_interactive_send_tasks",
                    "description": "Rebuild the 13:00 and 21:00 interactive send tasks for logged-in desktop use.",
                    "command": interactive_setup_command,
                    "destructive": True,
                    "requires_password": False,
                    "blocked": False,
                    "tasks": [str(candidate.get("task_name", "") or "") for candidate in repair_candidates],
                }
            )
        elif not password_present:
            action_items.append(
                {
                    "step": len(action_items) + 1,
                    "name": "rebuild_s4u_background_tasks",
                    "description": (
                        "Rebuild send, doctor, and preflight tasks in S4U/background mode without storing a password. "
                        "This avoids InteractiveToken-only failures but may have fewer network privileges than password mode."
                    ),
                    "command": s4u_repair_script_command,
                    "destructive": True,
                    "requires_password": False,
                    "blocked": False,
                    "tasks": [str(candidate.get("task_name", "") or "") for candidate in repair_candidates],
                }
            )
        if require_offline_tasks:
            action_items.append(
                {
                    "step": len(action_items) + 1,
                    "name": "rebuild_offline_tasks",
                    "description": "Rebuild offline-capable send, doctor, and preflight tasks.",
                    "command": repair_script_command,
                    "destructive": True,
                    "requires_password": True,
                    "blocked": not password_present,
                    "password_env": password_env,
                    "tasks": [str(candidate.get("task_name", "") or "") for candidate in repair_candidates],
                }
            )
    action_items.append(
        {
            "step": len(action_items) + 1,
            "name": "record_doctor",
            "description": "Record a fresh health snapshot after repair.",
            "command": " ".join(
                _quote_command_arg(part)
                for part in [sys.executable, str(ROOT / "scheduler_runner.py"), "--doctor", "--record"]
            ),
            "destructive": False,
            "requires_password": False,
            "blocked": False,
        }
    )

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "overall": payload.get("overall", "unknown"),
        "candidate_count": len(candidates),
        "legacy_cleanup_needed": bool(legacy_candidates),
        "offline_rebuild_needed": bool(repair_candidates),
        "password_env": password_env,
        "password_present": password_present,
        "candidates": candidates,
        "action_items": action_items,
    }


def build_repair_plan_text(plan: Dict[str, Any]) -> str:
    lines = [
        f"Repair plan generated at: {plan.get('generated_at', '') or 'unknown'}",
        f"Doctor overall: {plan.get('overall', 'unknown')}",
        f"Candidates: {plan.get('candidate_count', 0)}",
        f"Windows password env: {plan.get('password_env', 'WEB_AGENT_RUNAS_PASSWORD')} "
        f"({'present' if plan.get('password_present', False) else 'missing'})",
        "",
        "Actions:",
    ]
    for item in list(plan.get("action_items") or []):
        flags = []
        if item.get("destructive", False):
            flags.append("destructive")
        if item.get("requires_password", False):
            flags.append("requires_password")
        if item.get("blocked", False):
            flags.append("blocked")
        flag_text = f" [{' / '.join(flags)}]" if flags else ""
        lines.append(f"{item.get('step', '?')}. {item.get('name', 'action')}{flag_text}: {item.get('description', '')}")
        tasks = list(item.get("tasks") or [])
        if tasks:
            lines.append("   tasks: " + ", ".join(tasks))
        if item.get("backup_dir"):
            lines.append(f"   backup_dir: {item.get('backup_dir')}")
        lines.append(f"   command: {item.get('command', '')}")
    return "\n".join(lines)


def print_repair_plan(as_json: bool = False) -> int:
    payload = build_doctor_payload()
    _, scheduler_config = load_runtime_config()
    plan = build_repair_plan(payload, scheduler_config)
    if as_json:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
    else:
        print(build_repair_plan_text(plan))
    return 0


def build_legacy_cleanup_payload(
    payload: Dict[str, Any],
    scheduler_config: Dict[str, Any],
    confirm: bool = False,
) -> Dict[str, Any]:
    candidates = [
        candidate for candidate in collect_task_self_heal_candidates(payload, scheduler_config)
        if str(candidate.get("reason", "") or "") == "enabled_legacy_task"
    ]
    results: list[Dict[str, Any]] = []
    backup_dir = Path(str(scheduler_config.get("task_backup_dir", ROOT / "logs/task_backups")))
    if confirm:
        results = [
            delete_scheduled_task(str(candidate.get("task_name", "") or ""), backup_dir=backup_dir)
            for candidate in candidates
        ]
    success = bool(candidates) if not confirm else all(result.get("success", False) for result in results)
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "confirmed": confirm,
        "destructive": confirm,
        "candidate_count": len(candidates),
        "backup_dir": backup_dir.as_posix(),
        "candidates": candidates,
        "results": results,
        "success": success,
        "status": "completed" if confirm else "preview",
        "message": (
            "No enabled legacy scheduled tasks were found."
            if not candidates
            else (
                "Legacy scheduled task cleanup completed."
                if confirm and success
                else "Legacy scheduled task cleanup failed."
                if confirm
                else "Preview only. Add --confirm to delete these legacy scheduled tasks."
            )
        ),
    }


def build_legacy_cleanup_text(payload: Dict[str, Any]) -> str:
    lines = [
        f"Legacy cleanup status: {payload.get('status', 'unknown')}",
        f"Confirmed: {payload.get('confirmed', False)}",
        f"Candidates: {payload.get('candidate_count', 0)}",
        f"Backup dir: {payload.get('backup_dir', '') or 'N/A'}",
        f"Message: {payload.get('message', '')}",
    ]
    candidates = list(payload.get("candidates") or [])
    if candidates:
        lines.append("")
        lines.append("Legacy tasks:")
        for candidate in candidates:
            lines.append(f"- {candidate.get('task_name', 'unknown')}: {candidate.get('detail', '')}")
    results = list(payload.get("results") or [])
    if results:
        lines.append("")
        lines.append("Delete results:")
        for result in results:
            lines.append(
                f"- {result.get('task_name', 'unknown')}: "
                f"{'success' if result.get('success', False) else 'failed'} "
                f"(returncode={result.get('returncode')})"
            )
            backup = dict(result.get("backup") or {})
            if backup:
                lines.append(f"   backup: {backup.get('backup_path', '') or 'N/A'} ({'ok' if backup.get('success', False) else 'failed'})")
    return "\n".join(lines)


def print_legacy_cleanup_report(as_json: bool = False, confirm: bool = False) -> int:
    doctor_payload = build_doctor_payload()
    _, scheduler_config = load_runtime_config()
    payload = build_legacy_cleanup_payload(doctor_payload, scheduler_config, confirm=confirm)
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(build_legacy_cleanup_text(payload))
    if confirm:
        return 0 if payload.get("success", False) else 1
    return 0


def _infer_task_name_from_backup(path: Path) -> str:
    stem = path.stem
    parts = stem.rsplit("_", 2)
    if len(parts) == 3 and parts[-2].isdigit() and parts[-1].isdigit():
        return parts[0]
    return stem


def build_task_backups_payload(scheduler_config: Dict[str, Any]) -> Dict[str, Any]:
    backup_dir = Path(str(scheduler_config.get("task_backup_dir", ROOT / "logs/task_backups")))
    backups: list[Dict[str, Any]] = []
    if backup_dir.exists():
        for path in sorted(backup_dir.glob("*.xml"), key=lambda item: item.stat().st_mtime, reverse=True):
            try:
                stat = path.stat()
            except OSError:
                continue
            task_name = _infer_task_name_from_backup(path)
            preview_command = " ".join(
                _quote_command_arg(part)
                for part in [sys.executable, str(ROOT / "scheduler_runner.py"), "--restore-task-backup", str(path)]
            )
            restore_command = f"{preview_command} --confirm"
            backups.append(
                {
                    "task_name": task_name,
                    "backup_path": path.as_posix(),
                    "size_bytes": stat.st_size,
                    "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
                    "preview_command": preview_command,
                    "restore_command": restore_command,
                    "raw_schtasks_restore_command": (
                        "schtasks /Create /TN "
                        f"{_quote_command_arg(task_name)} /XML {_quote_command_arg(path)} /F"
                    ),
                }
            )
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "backup_dir": backup_dir.as_posix(),
        "backup_count": len(backups),
        "backups": backups,
    }


def build_task_backups_text(payload: Dict[str, Any]) -> str:
    lines = [
        f"Task backups: {payload.get('backup_count', 0)}",
        f"Backup dir: {payload.get('backup_dir', '') or 'N/A'}",
    ]
    backups = list(payload.get("backups") or [])
    if not backups:
        lines.append("No task backup XML files found.")
        return "\n".join(lines)

    lines.append("")
    lines.append("Backups:")
    for backup in backups:
        lines.append(
            f"- {backup.get('task_name', 'unknown')} | "
            f"{backup.get('modified_at', 'unknown')} | "
            f"{backup.get('backup_path', '')}"
        )
        lines.append(f"  preview: {backup.get('preview_command', '')}")
        lines.append(f"  restore: {backup.get('restore_command', '')}")
        lines.append(f"  raw_schtasks_restore: {backup.get('raw_schtasks_restore_command', '')}")
    return "\n".join(lines)


def print_task_backups_report(as_json: bool = False) -> int:
    _, scheduler_config = load_runtime_config()
    payload = build_task_backups_payload(scheduler_config)
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(build_task_backups_text(payload))
    return 0


def build_task_restore_payload(backup_path_text: str, confirm: bool = False, task_name: str = "") -> Dict[str, Any]:
    backup_path_raw = str(backup_path_text or "").strip()
    backup_path = Path(backup_path_raw) if backup_path_raw else Path()
    if backup_path_raw and not backup_path.is_absolute():
        backup_path = (ROOT / backup_path).resolve()

    inferred_task_name = task_name or (_infer_task_name_from_backup(backup_path) if backup_path_raw else "")
    command = ["schtasks", "/Create", "/TN", inferred_task_name, "/XML", str(backup_path), "/F"]
    payload: Dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "confirmed": confirm,
        "destructive": confirm,
        "backup_path": backup_path.as_posix() if backup_path_raw else "",
        "task_name": inferred_task_name,
        "command": command,
        "success": False,
        "status": "preview",
        "returncode": None,
        "stdout": "",
        "stderr": "",
        "message": "",
    }

    if not backup_path_raw:
        payload["status"] = "missing_argument"
        payload["message"] = "Provide a backup XML path after --restore-task-backup."
        return payload
    if not backup_path.exists():
        payload["status"] = "missing_backup"
        payload["message"] = f"Backup XML was not found: {backup_path}"
        return payload
    if backup_path.suffix.lower() != ".xml":
        payload["status"] = "invalid_backup_type"
        payload["message"] = "Only .xml task backups can be restored."
        return payload
    if not inferred_task_name:
        payload["status"] = "missing_task_name"
        payload["message"] = "Unable to infer task name from backup file name."
        return payload
    if not confirm:
        payload["success"] = True
        payload["message"] = "Preview only. Add --confirm to restore this scheduled task."
        return payload

    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding=locale.getpreferredencoding(False),
        errors="replace",
        creationflags=creationflags,
        check=False,
    )
    payload["returncode"] = completed.returncode
    payload["stdout"] = (completed.stdout or "").strip()
    payload["stderr"] = (completed.stderr or "").strip()
    payload["success"] = completed.returncode == 0
    payload["status"] = "completed" if payload["success"] else "failed"
    payload["message"] = "Scheduled task restored from backup." if payload["success"] else "Scheduled task restore failed."
    return payload


def build_task_restore_text(payload: Dict[str, Any]) -> str:
    lines = [
        f"Restore status: {payload.get('status', 'unknown')}",
        f"Confirmed: {payload.get('confirmed', False)}",
        f"Task name: {payload.get('task_name', '') or 'N/A'}",
        f"Backup path: {payload.get('backup_path', '') or 'N/A'}",
        "Command: " + " ".join(_quote_command_arg(part) for part in payload.get("command", [])),
        f"Message: {payload.get('message', '')}",
    ]
    if payload.get("returncode") is not None:
        lines.append(f"Return code: {payload.get('returncode')}")
    if payload.get("stdout"):
        lines.append(f"Stdout: {payload.get('stdout')}")
    if payload.get("stderr"):
        lines.append(f"Stderr: {payload.get('stderr')}")
    return "\n".join(lines)


def print_task_restore_report(backup_path_text: str, as_json: bool = False, confirm: bool = False) -> int:
    payload = build_task_restore_payload(backup_path_text, confirm=confirm)
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(build_task_restore_text(payload))
    return 0 if payload.get("success", False) else 1


def should_send_doctor_alert(payload: Dict[str, Any], history: Dict[str, Any], scheduler_config: Dict[str, Any]) -> bool:
    if not scheduler_config.get("send_doctor_alert_email", True):
        return False

    overall = str(payload.get("overall", "unknown") or "unknown")
    signature = _doctor_issue_signature(payload)
    last_alert_signature = str(history.get("last_alert_signature", "") or "")
    last_alert_overall = str(history.get("last_alert_overall", "") or "")
    warn_streak = int(history.get("warn_streak", 0) or 0)
    warn_threshold = max(1, int(scheduler_config.get("doctor_warn_streak_threshold", 2)))

    if overall == "fail":
        return signature != last_alert_signature or last_alert_overall != "fail"
    if overall == "warn" and warn_streak >= warn_threshold:
        return signature != last_alert_signature or last_alert_overall != "warn"
    return False


def send_doctor_alert_email(config: Dict[str, Any], scheduler_config: Dict[str, Any], payload: Dict[str, Any], history: Dict[str, Any]) -> bool:
    if not scheduler_config.get("send_doctor_alert_email", True):
        return False

    load_dotenv(ROOT / ".env")
    recipient_env = str(config.get("alerts", {}).get("recipients_env", "EMAIL_RECIPIENT"))
    recipient = os.getenv(recipient_env) or os.getenv("EMAIL_RECIPIENT")
    sender = os.getenv("EMAIL_SENDER")
    password = os.getenv("EMAIL_PASSWORD")
    smtp_server = os.getenv("EMAIL_SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.getenv("EMAIL_SMTP_PORT", "587"))

    if not recipient or not sender or not password:
        print("Skipping doctor alert email because email credentials are incomplete.")
        return False

    notifier = EmailNotifier(
        smtp_server=smtp_server,
        smtp_port=smtp_port,
        sender_email=sender,
        sender_password=password,
        timeout_seconds=int(config.get("email", {}).get("timeout_seconds", config.get("network", {}).get("timeout_seconds", 25))),
        max_attempts=int(config.get("email", {}).get("max_attempts", 2)),
        retry_delay_seconds=int(config.get("email", {}).get("retry_delay_seconds", 5)),
    )
    subject_prefix = str(scheduler_config.get("doctor_alert_subject_prefix", "[AI日报健康检查告警]"))
    subject = f"{subject_prefix} {datetime.now().strftime('%Y-%m-%d %H:%M')} {payload.get('overall', 'unknown')}"
    return notifier.send_email(
        recipient_email=recipient,
        subject=subject,
        html_content=build_doctor_alert_html(payload, history),
    )


def build_email_arrival_check(config: Dict[str, Any], scheduler_config: Dict[str, Any], last_success: Dict[str, Any]) -> Dict[str, Any]:
    load_dotenv(ROOT / ".env")
    arrival_config = dict(config.get("email", {}).get("arrival_check", {}) or {})
    subject = str(last_success.get("email_subject", "") or "").strip()
    if not subject:
        return {
            "enabled": bool(arrival_config.get("enabled", False)),
            "verified": False,
            "status": "skipped_missing_subject",
            "matched_subject": "",
            "matched_date": "",
            "error": "Last successful send does not include email_subject yet. The next send will record it.",
        }

    smtp_server = os.getenv("EMAIL_SMTP_SERVER", "smtp.gmail.com")
    imap_server = resolve_imap_server(
        smtp_server=smtp_server,
        configured_imap_server=os.getenv("EMAIL_IMAP_SERVER", str(arrival_config.get("imap_server", ""))),
    )
    since_minutes = int(arrival_config.get("since_minutes", 30))
    finished_at = str(last_success.get("finished_at", "") or "").strip()
    if finished_at:
        try:
            finished_dt = datetime.fromisoformat(finished_at)
            elapsed_minutes = int((datetime.now() - finished_dt).total_seconds() // 60) + 15
            since_minutes = max(since_minutes, elapsed_minutes)
        except ValueError:
            pass
    return verify_email_arrival(
        imap_server=imap_server,
        imap_port=int(os.getenv("EMAIL_IMAP_PORT", str(arrival_config.get("imap_port", 993)))),
        username=os.getenv("EMAIL_IMAP_USERNAME", os.getenv("EMAIL_SENDER", "")),
        password=os.getenv("EMAIL_IMAP_PASSWORD", os.getenv("EMAIL_PASSWORD", "")),
        subject_contains=subject,
        since_minutes=since_minutes,
        mailbox=arrival_config.get("mailboxes") or str(arrival_config.get("mailbox", "INBOX")),
        timeout_seconds=int(arrival_config.get("timeout_seconds", config.get("network", {}).get("timeout_seconds", 25))),
        expected_sender=os.getenv("EMAIL_SENDER", ""),
        retry_attempts=int(arrival_config.get("retry_attempts", 1)),
        retry_delay_seconds=int(arrival_config.get("retry_delay_seconds", 5)),
    )


def print_email_arrival_report(as_json: bool = False) -> int:
    config, scheduler_config = load_runtime_config()
    last_success_path = Path(str(scheduler_config["last_success_file"]))
    last_success = read_json(last_success_path)
    if not last_success:
        result = {
            "enabled": bool(config.get("email", {}).get("arrival_check", {}).get("enabled", False)),
            "verified": False,
            "status": "skipped_missing_last_success",
            "matched_subject": "",
            "matched_date": "",
            "error": "No last_success.json snapshot exists yet.",
        }
    else:
        result = build_email_arrival_check(config, scheduler_config, last_success)
        last_success["delivery_verification"] = result
        last_success["arrival_checked_at"] = datetime.now().isoformat(timespec="seconds")
        write_json(last_success_path, last_success)

    if as_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"Arrival check status: {result.get('status', 'unknown')}")
        print(f"Verified: {result.get('verified', False)}")
        if result.get("matched_subject"):
            print(f"Matched subject: {result.get('matched_subject')}")
        if result.get("matched_date"):
            print(f"Matched date: {result.get('matched_date')}")
        if result.get("error"):
            print(f"Error: {result.get('error')}")
    return 0 if result.get("verified", False) else 1


def _parse_int_arg(raw_args: list[str], name: str, default: int) -> int:
    try:
        index = raw_args.index(name)
        return int(raw_args[index + 1])
    except (ValueError, IndexError, TypeError):
        return default


def _parse_str_arg(raw_args: list[str], name: str, default: str = "") -> str:
    try:
        index = raw_args.index(name)
        return str(raw_args[index + 1])
    except (ValueError, IndexError, TypeError):
        return default


def print_model_path_backfill_report(
    *,
    limit: int = 10,
    dry_run: bool = False,
    as_json: bool = False,
    include_v2_fallback: bool = False,
    report_id: str = "",
) -> int:
    result = backfill_latest_report_model_paths(
        limit=limit,
        dry_run=dry_run,
        include_v2_fallback=include_v2_fallback,
        report_id=report_id,
    )
    result["mode"] = "refresh_fallback_model_path" if include_v2_fallback else "backfill_legacy_model_path"
    if as_json:
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    else:
        print(f"Mode: {result.get('mode', '')}")
        print(f"Report ID: {result.get('report_id', '')}")
        if result.get("error"):
            print(f"Error: {result.get('error')}")
        print(f"Dry run: {result.get('dry_run', False)}")
        print(f"Attempted: {result.get('attempted', 0)}")
        print(f"Updated: {result.get('updated', 0)}")
        print(f"Failed: {result.get('failed', 0)}")
        print(f"Model path breakdown: {json.dumps(result.get('model_path_breakdown', {}), ensure_ascii=False, sort_keys=True)}")
        for item in result.get("items", [])[:20]:
            print(f"- #{item.get('rank', '')} {item.get('section', '')}: {item.get('title', '')}")
    return 1 if result.get("error") or result.get("failed", 0) else 0


def print_report_quality_scan(
    *,
    report_id: str = "",
    limit: int = 10,
    as_json: bool = False,
    persist: bool = False,
) -> int:
    result = scan_report_quality(report_id=report_id, limit=limit, persist=persist)
    if as_json:
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    else:
        print(f"Report ID: {result.get('report_id', '')}")
        if result.get("error"):
            print(f"Error: {result.get('error')}")
        print(f"Scanned: {result.get('scanned', 0)}")
        print(f"Issue count: {result.get('issue_count', 0)}")
        print(f"Counts: {json.dumps(result.get('counts', {}), ensure_ascii=False, sort_keys=True)}")
        for item in result.get("issues", [])[:limit]:
            print(f"- #{item.get('rank', '')} {item.get('section', '')}: {item.get('title', '')} [{', '.join(item.get('issue_types', []))}]")
    return 1 if result.get("error") else 0


def print_bad_title_fix_report(
    *,
    report_id: str = "",
    limit: int = 10,
    dry_run: bool = False,
    as_json: bool = False,
) -> int:
    result = fix_report_bad_titles(report_id=report_id, limit=limit, dry_run=dry_run)
    if as_json:
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    else:
        print(f"Report ID: {result.get('report_id', '')}")
        if result.get("error"):
            print(f"Error: {result.get('error')}")
        print(f"Dry run: {result.get('dry_run', False)}")
        print(f"Attempted: {result.get('attempted', 0)}")
        print(f"Updated: {result.get('updated', 0)}")
        for item in result.get("items", [])[:limit]:
            print(f"- #{item.get('rank', '')}: {item.get('old_title', '')} -> {item.get('new_title', '')}")
    return 1 if result.get("error") else 0


def print_doctor_report(
    as_json: bool = False,
    self_heal: bool = False,
    dry_run: bool = False,
    persist_history: bool = False,
) -> int:
    payload = build_doctor_payload()
    config, scheduler_config = load_runtime_config()

    self_heal_result: Dict[str, Any] = {
        "attempted": False,
        "dry_run": dry_run,
        "candidates": [],
        "success": False,
        "message": "Task self-heal not requested.",
    }
    if self_heal:
        candidates = collect_task_self_heal_candidates(payload, scheduler_config)
        self_heal_result = run_task_self_heal(scheduler_config, candidates, dry_run=dry_run)
        if self_heal_result.get("attempted", False) and self_heal_result.get("success", False) and not dry_run:
            pre_heal_summary = {
                "overall": payload.get("overall", "unknown"),
                "counts": payload.get("counts", {}),
                "candidates": candidates,
            }
            payload = build_doctor_payload()
            payload["pre_heal_summary"] = pre_heal_summary
        payload["self_heal"] = self_heal_result

    snapshot_path = write_doctor_snapshot(payload, scheduler_config)
    history_path = Path(str(scheduler_config["doctor_history_file"]))
    history = (
        update_doctor_history(payload, scheduler_config)
        if persist_history
        else read_json(history_path)
    )
    payload["history_summary"] = {
        "warn_streak": int(history.get("warn_streak", 0) or 0),
        "last_alert_sent_at": str(history.get("last_alert_sent_at", "") or ""),
        "last_alert_overall": str(history.get("last_alert_overall", "") or ""),
        "recorded": persist_history,
    }
    write_doctor_snapshot(payload, scheduler_config)

    doctor_alert_sent = False
    if persist_history and should_send_doctor_alert(payload, history, scheduler_config):
        doctor_alert_sent = send_doctor_alert_email(config, scheduler_config, payload, history)
        if doctor_alert_sent:
            history["last_alert_signature"] = _doctor_issue_signature(payload)
            history["last_alert_overall"] = str(payload.get("overall", "unknown") or "unknown")
            history["last_alert_sent_at"] = datetime.now().isoformat(timespec="seconds")
            write_json(Path(str(scheduler_config["doctor_history_file"])), history)
            payload["history_summary"] = {
                "warn_streak": history.get("warn_streak", 0),
                "last_alert_sent_at": history.get("last_alert_sent_at", ""),
                "last_alert_overall": history.get("last_alert_overall", ""),
                "recorded": persist_history,
            }
            write_doctor_snapshot(payload, scheduler_config)

    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"Doctor overall: {payload.get('overall', 'unknown')}")
        print(
            "Counts: "
            f"ok={payload.get('counts', {}).get('ok', 0)} "
            f"warn={payload.get('counts', {}).get('warn', 0)} "
            f"fail={payload.get('counts', {}).get('fail', 0)}"
        )
        print("")
        for item in payload.get("checks", []):
            print(f"- [{item.get('level', 'unknown')}] {item.get('name', 'unknown')}: {item.get('detail', '')}")
        print("")
        repair_commands = list(payload.get("repair_commands") or [])
        if repair_commands:
            print("Repair commands:")
            for command in repair_commands:
                print(f"- {command.get('name', 'command')}: {command.get('command', '')}")
            print("")
        print(f"History recorded: {persist_history}")
        print(f"Warn streak: {history.get('warn_streak', 0)}")
        print(f"Doctor alert sent: {doctor_alert_sent}")
        if self_heal:
            print(f"Task self-heal attempted: {self_heal_result.get('attempted', False)}")
            print(f"Task self-heal success: {self_heal_result.get('success', False)}")
            print(f"Task self-heal message: {self_heal_result.get('message', '')}")
        print(f"Snapshot file: {snapshot_path.as_posix()}")
    if self_heal and self_heal_result.get("attempted", False) and not self_heal_result.get("success", False):
        return 1
    return 0 if payload.get("overall") != "fail" else 1


def archive_old_logs(
    log_dir: Path,
    archive_dir: Path,
    archive_after_days: int,
    keep_paths: Optional[set[Path]] = None,
) -> list[str]:
    keep_paths = {path.resolve() for path in (keep_paths or set())}
    if archive_after_days <= 0 or not log_dir.exists():
        return []

    archive_dir.mkdir(parents=True, exist_ok=True)
    cutoff = datetime.now() - timedelta(days=archive_after_days)
    archived: list[str] = []
    for path in sorted(log_dir.glob("*.log")):
        try:
            resolved_path = path.resolve()
        except OSError:
            continue
        if resolved_path in keep_paths:
            continue
        try:
            modified_at = datetime.fromtimestamp(path.stat().st_mtime)
        except OSError:
            continue
        if modified_at >= cutoff:
            continue

        target = archive_dir / path.name
        if target.exists():
            timestamp = modified_at.strftime("%Y%m%d_%H%M%S")
            target = archive_dir / f"{path.stem}_{timestamp}{path.suffix}"
        try:
            shutil.move(str(path), str(target))
            archived.append(target.relative_to(log_dir).as_posix() if target.is_relative_to(log_dir) else target.name)
        except OSError:
            continue
    return archived


def cleanup_old_logs(log_dir: Path, retention_days: int, keep_paths: Optional[set[Path]] = None) -> list[str]:
    keep_paths = keep_paths or set()
    if retention_days <= 0 or not log_dir.exists():
        return []

    cutoff = datetime.now() - timedelta(days=retention_days)
    removed: list[str] = []
    for path in log_dir.glob("*.log"):
        if path in keep_paths:
            continue
        try:
            modified_at = datetime.fromtimestamp(path.stat().st_mtime)
        except OSError:
            continue
        if modified_at >= cutoff:
            continue
        try:
            path.unlink()
            removed.append(path.name)
        except OSError:
            continue
    return removed


def cleanup_validation_reports(validation_dir: Path, retention_days: int) -> list[str]:
    if retention_days <= 0 or not validation_dir.exists():
        return []

    cutoff = datetime.now() - timedelta(days=retention_days)
    removed: list[str] = []
    candidates = sorted(validation_dir.glob("report_*.*"))
    for path in candidates:
        if path.suffix.lower() not in {".html", ".md"}:
            continue
        try:
            modified_at = datetime.fromtimestamp(path.stat().st_mtime)
        except OSError:
            continue
        if modified_at >= cutoff:
            continue
        try:
            path.unlink()
            removed.append(path.name)
        except OSError:
            continue
    return removed


def cleanup_task_backups(backup_dir: Path, retention_days: int, keep_count: int) -> list[str]:
    if not backup_dir.exists():
        return []

    cutoff = datetime.now() - timedelta(days=retention_days) if retention_days > 0 else None
    keep_count = max(0, keep_count)
    removed: list[str] = []
    candidates: list[tuple[float, Path]] = []
    for path in backup_dir.glob("*.xml"):
        try:
            candidates.append((path.stat().st_mtime, path))
        except OSError:
            continue

    for index, (modified_ts, path) in enumerate(sorted(candidates, key=lambda item: item[0], reverse=True)):
        modified_at = datetime.fromtimestamp(modified_ts)
        if keep_count > 0 and index < keep_count:
            continue
        exceeds_count = keep_count > 0 and index >= keep_count
        expired = cutoff is not None and modified_at < cutoff
        if not exceeds_count and not expired:
            continue
        try:
            path.unlink()
            removed.append(path.name)
        except OSError:
            continue
    return removed


def parse_time_of_day(value: str) -> tuple[int, int]:
    hour_text, minute_text = str(value).strip().split(":", 1)
    hour = int(hour_text)
    minute = int(minute_text)
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        raise ValueError(f"Invalid time of day: {value}")
    return hour, minute


def resolve_send_slot(
    now: Optional[datetime] = None,
    scheduler_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    now = now or datetime.now()
    scheduler_config = scheduler_config or DEFAULT_SCHEDULER_CONFIG
    before_minutes = max(0, int(scheduler_config.get("send_window_before_minutes", 10)))
    after_minutes = max(0, int(scheduler_config.get("send_window_after_minutes", 180)))
    slots = list(scheduler_config.get("send_slots") or DEFAULT_SCHEDULER_CONFIG["send_slots"])

    candidates: list[Dict[str, Any]] = []
    schedule_dates = [now.date() - timedelta(days=1), now.date(), now.date() + timedelta(days=1)]
    for slot in slots:
        slot_id = str(slot.get("id", "") or "").strip()
        slot_time = str(slot.get("time", "") or "").strip()
        if not slot_id or not slot_time:
            continue
        try:
            hour, minute = parse_time_of_day(slot_time)
        except (TypeError, ValueError):
            continue
        for schedule_date in schedule_dates:
            scheduled_at = datetime.combine(schedule_date, datetime.min.time()).replace(hour=hour, minute=minute)
            window_start = scheduled_at - timedelta(minutes=before_minutes)
            window_end = scheduled_at + timedelta(minutes=after_minutes)
            if window_start <= now <= window_end:
                candidates.append(
                    {
                        "id": slot_id,
                        "time": slot_time,
                        "slot_id": f"{scheduled_at.strftime('%Y%m%d')}_{slot_id}",
                        "scheduled_at": scheduled_at.isoformat(timespec="seconds"),
                        "window_start": window_start.isoformat(timespec="seconds"),
                        "window_end": window_end.isoformat(timespec="seconds"),
                        "distance_seconds": abs(int((now - scheduled_at).total_seconds())),
                    }
                )
    if not candidates:
        return {
            "allowed": False,
            "reason": "outside_send_window",
            "checked_at": now.isoformat(timespec="seconds"),
            "configured_slots": slots,
            "window_before_minutes": before_minutes,
            "window_after_minutes": after_minutes,
        }
    candidates.sort(key=lambda item: int(item["distance_seconds"]))
    selected = dict(candidates[0])
    selected["allowed"] = True
    return selected


def build_send_slot_id(
    now: Optional[datetime] = None,
    scheduler_config: Optional[Dict[str, Any]] = None,
) -> str:
    slot = resolve_send_slot(now, scheduler_config)
    if not slot.get("allowed", False):
        return ""
    return str(slot["slot_id"])


def acquire_send_slot(
    slot_dir: Path,
    slot_id: str,
    stale_seconds: int,
    dead_pid_grace_seconds: int = 60,
) -> Tuple[bool, Dict[str, Any]]:
    slot_dir.mkdir(parents=True, exist_ok=True)
    slot_path = slot_dir / f"{slot_id}.json"
    now = datetime.now()
    payload = {
        "slot_id": slot_id,
        "pid": os.getpid(),
        "status": "in_progress",
        "acquired_at": now.isoformat(timespec="seconds"),
    }
    if slot_path.exists():
        existing = read_json(slot_path)
        acquired_at = str(existing.get("acquired_at", "") or existing.get("finished_at", ""))
        try:
            acquired_dt = datetime.fromisoformat(acquired_at)
            age_seconds = max(0, int((now - acquired_dt).total_seconds()))
        except ValueError:
            age_seconds = stale_seconds + 1
        existing["age_seconds"] = age_seconds
        existing_pid = int(existing.get("pid", 0) or 0)
        pid_running = is_pid_running(existing_pid) if existing_pid > 0 else False
        existing["pid_running"] = pid_running
        if str(existing.get("status", "")) == "sent" or pid_running or age_seconds <= dead_pid_grace_seconds:
            existing["slot_path"] = slot_path.as_posix()
            return False, existing
        slot_path.unlink(missing_ok=True)

    try:
        with slot_path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
    except FileExistsError:
        existing = read_json(slot_path)
        existing["slot_path"] = slot_path.as_posix()
        return False, existing
    payload["slot_path"] = slot_path.as_posix()
    return True, payload


def finalize_send_slot(slot_info: Dict[str, Any], status: Dict[str, Any]) -> None:
    slot_path_text = str(slot_info.get("slot_path", "") or "")
    if not slot_path_text:
        return
    slot_path = Path(slot_path_text)
    if not slot_path.exists():
        return
    if status.get("success", False) and status.get("delivery_status") == "sent":
        payload = dict(slot_info)
        payload.update(
            {
                "status": "sent",
                "finished_at": status.get("finished_at", datetime.now().isoformat(timespec="seconds")),
                "run_id": status.get("run_id", ""),
                "html_report_path": status.get("html_report_path", ""),
                "markdown_report_path": status.get("markdown_report_path", ""),
                "quality_status": status.get("quality_status", ""),
                "email_subjects": list(status.get("email_subjects") or []),
                "email_volume_sent_count": int(status.get("email_volume_sent_count", 0) or 0),
                "email_volume_paths": list(status.get("email_volume_paths") or []),
                "delivery_verification": dict(status.get("delivery_verification") or {}),
                "post_send_quality_scan": dict(status.get("post_send_quality_scan") or {}),
                "ui_audit": dict(status.get("ui_audit") or {}),
            }
        )
        write_json(slot_path, payload)
        return
    slot_path.unlink(missing_ok=True)


def recover_committed_send_slot(slot_info: Dict[str, Any], started_at: Optional[datetime] = None) -> Dict[str, Any]:
    slot_path_text = str(slot_info.get("slot_path", "") or "")
    if not slot_path_text:
        return {}
    persisted = read_json(Path(slot_path_text))
    if str(persisted.get("status", "") or "") != "sent":
        return {}
    now = datetime.now()
    return {
        **persisted,
        "success": True,
        "status": "sent_commit_recovered",
        "retryable": False,
        "delivery_status": "sent",
        "started_at": (started_at or now).isoformat(timespec="seconds"),
        "finished_at": str(persisted.get("finished_at") or now.isoformat(timespec="seconds")),
    }


def _parse_status_datetime(value: Any) -> Optional[datetime]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone().replace(tzinfo=None)
    return parsed


def build_send_calendar_payload(
    scheduler_config: Dict[str, Any],
    target_datetime: Optional[datetime] = None,
    last_run: Optional[Dict[str, Any]] = None,
    last_success: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    target_datetime = target_datetime or datetime.now()
    target_date = target_datetime.date()
    slot_dir = Path(str(scheduler_config.get("send_slot_dir", ROOT / "logs/send_slots")))
    slots = list(scheduler_config.get("send_slots") or DEFAULT_SCHEDULER_CONFIG["send_slots"])
    before_minutes = max(0, int(scheduler_config.get("send_window_before_minutes", 10)))
    after_minutes = max(0, int(scheduler_config.get("send_window_after_minutes", 180)))
    slot_stale_seconds = max(
        60,
        int(
            scheduler_config.get(
                "send_slot_stale_seconds",
                scheduler_config.get("stale_lock_seconds", 10800),
            )
        ),
    )
    last_run = dict(last_run or {})
    last_success = dict(last_success or {})
    rows: list[Dict[str, Any]] = []

    for slot in slots:
        slot_id_suffix = str(slot.get("id", "") or "").strip()
        slot_time = str(slot.get("time", "") or "").strip()
        if not slot_id_suffix or not slot_time:
            continue
        try:
            hour, minute = parse_time_of_day(slot_time)
            scheduled_at = datetime.combine(target_date, datetime.min.time()).replace(hour=hour, minute=minute)
        except (TypeError, ValueError):
            scheduled_at = datetime.combine(target_date, datetime.min.time())
        window_start = scheduled_at - timedelta(minutes=before_minutes)
        window_end = scheduled_at + timedelta(minutes=after_minutes)
        daily_slot_id = f"{target_date.strftime('%Y%m%d')}_{slot_id_suffix}"
        slot_payload = read_json(slot_dir / f"{daily_slot_id}.json")
        raw_status = str(slot_payload.get("status", "missing") or "missing")
        if raw_status == "in_progress":
            acquired_at = _parse_status_datetime(slot_payload.get("acquired_at"))
            slot_pid = int(slot_payload.get("pid", 0) or 0)
            pid_running = is_pid_running(slot_pid) if slot_pid > 0 else False
            age_seconds = (target_datetime - acquired_at).total_seconds() if acquired_at else slot_stale_seconds + 1
            if acquired_at is None or age_seconds > slot_stale_seconds or (not pid_running and age_seconds > 60):
                raw_status = "stale"
        health_status = raw_status
        if raw_status in {"missing", "stale"}:
            if target_datetime < window_start:
                health_status = "upcoming"
            elif target_datetime <= window_end:
                health_status = "pending"
            else:
                health_status = "missed"
        row = {
            "slot_id": daily_slot_id,
            "time": slot_time,
            "status": raw_status,
            "health_status": health_status,
            "scheduled_at": scheduled_at.isoformat(timespec="seconds"),
            "window_start": window_start.isoformat(timespec="seconds"),
            "window_end": window_end.isoformat(timespec="seconds"),
            "run_id": slot_payload.get("run_id", ""),
            "finished_at": slot_payload.get("finished_at", ""),
            "html_report_path": slot_payload.get("html_report_path", ""),
            "markdown_report_path": slot_payload.get("markdown_report_path", ""),
        }
        if str(last_run.get("send_slot", {}).get("slot_id", "") or "") == daily_slot_id:
            row["last_run_status"] = last_run.get("status", "")
            row["last_run_finished_at"] = last_run.get("finished_at", "")
            if not row["finished_at"]:
                row["finished_at"] = last_run.get("finished_at", "")
            if not row["html_report_path"]:
                row["html_report_path"] = last_run.get("html_report_path", "")
            if not row["markdown_report_path"]:
                row["markdown_report_path"] = last_run.get("markdown_report_path", "")
        rows.append(row)

    return {
        "date": target_date.isoformat(),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "configured_slots": [
            {"id": str(slot.get("id", "") or ""), "time": str(slot.get("time", "") or "")}
            for slot in slots
        ],
        "slots": rows,
        "last_run": {
            "status": last_run.get("status", ""),
            "delivery_status": last_run.get("delivery_status", ""),
            "finished_at": last_run.get("finished_at", ""),
            "html_report_path": last_run.get("html_report_path", ""),
        },
        "last_success": {
            "finished_at": last_success.get("finished_at", ""),
            "html_report_path": last_success.get("html_report_path", ""),
        },
    }


def write_send_calendar(scheduler_config: Dict[str, Any], last_run: Dict[str, Any]) -> Optional[Path]:
    if str(last_run.get("run_mode", "scheduler") or "scheduler") != "scheduler":
        return None
    finished_at = _parse_status_datetime(last_run.get("finished_at")) or datetime.now()
    calendar_dir = Path(str(scheduler_config.get("send_calendar_dir", scheduler_config.get("log_dir", ROOT / "logs"))))
    calendar_path = calendar_dir / f"send_calendar_{finished_at.strftime('%Y%m%d')}.json"
    last_success = read_json(Path(str(scheduler_config.get("last_success_file", ""))))
    payload = build_send_calendar_payload(
        scheduler_config,
        target_datetime=finished_at,
        last_run=last_run,
        last_success=last_success,
    )
    write_json(calendar_path, payload)
    return calendar_path


def build_send_calendar_text(payload: Dict[str, Any]) -> str:
    lines = [
        f"Send calendar: {payload.get('date', '') or 'unknown'}",
        f"Generated at: {payload.get('generated_at', '') or 'unknown'}",
        "",
        "Slots:",
    ]
    for slot in list(payload.get("slots") or []):
        report_path = slot.get("html_report_path", "") or "N/A"
        finished_at = slot.get("finished_at", "") or slot.get("last_run_finished_at", "") or "N/A"
        last_run_status = slot.get("last_run_status", "")
        suffix = f", last_run={last_run_status}" if last_run_status else ""
        lines.append(
            f"- {slot.get('time', 'N/A')} {slot.get('slot_id', 'unknown')}: "
            f"{slot.get('health_status', slot.get('status', 'missing'))}, "
            f"raw={slot.get('status', 'missing')}, finished={finished_at}, report={report_path}{suffix}"
        )

    last_success = dict(payload.get("last_success") or {})
    lines.append("")
    lines.append(
        "Last success: "
        f"{last_success.get('finished_at', '') or 'N/A'} | "
        f"{last_success.get('html_report_path', '') or 'N/A'}"
    )
    return "\n".join(lines)


def print_send_calendar_report(as_json: bool = False) -> int:
    _, scheduler_config = load_runtime_config()
    today = datetime.now()
    payload = build_send_calendar_payload(
        scheduler_config,
        target_datetime=today,
        last_run=read_json(Path(str(scheduler_config["status_file"]))),
        last_success=read_json(Path(str(scheduler_config["last_success_file"]))),
    )
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(build_send_calendar_text(payload))
    return 0


def write_scheduler_status(status_path: Path, status: Dict[str, Any], scheduler_config: Dict[str, Any]) -> Dict[str, Any]:
    snapshot = build_status_snapshot(status)
    calendar_path = write_send_calendar(scheduler_config, snapshot)
    if calendar_path:
        snapshot["send_calendar_file"] = calendar_path.as_posix()
    write_json(status_path, snapshot)
    return snapshot


def should_skip_for_idempotency(
    status_path: Path,
    window_minutes: int,
    last_success_path: Optional[Path] = None,
) -> Tuple[bool, Dict[str, Any]]:
    if window_minutes <= 0:
        return False, {}

    last_status = read_json(status_path)
    last_success = read_json(last_success_path) if last_success_path else {}
    if last_success:
        last_success["success"] = True
        last_success["delivery_status"] = "sent"
        last_status = last_status or last_success
        effective_status = last_success
    else:
        effective_status = last_status
    if not last_status:
        return False, {}
    if not last_success and str(last_status.get("status", "")) == "skipped_recent_success":
        previous_success = last_status.get("previous_success") or {}
        if previous_success.get("finished_at"):
            effective_status = {
                **previous_success,
                "success": True,
                "delivery_status": "sent",
            }

    if not effective_status.get("success", False):
        return False, last_status
    if str(effective_status.get("delivery_status", "")) != "sent":
        return False, last_status

    finished_at = str(effective_status.get("finished_at", "") or "")
    if not finished_at:
        return False, last_status
    try:
        finished_dt = datetime.fromisoformat(finished_at)
    except ValueError:
        return False, last_status

    if datetime.now() - finished_dt <= timedelta(minutes=window_minutes):
        return True, effective_status
    return False, last_status


def build_last_success_snapshot(status: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "success": True,
        "status": status.get("status", "success"),
        "delivery_status": "sent",
        "finished_at": status.get("finished_at", ""),
        "started_at": status.get("started_at", ""),
        "run_id": status.get("run_id", ""),
        "run_mode": status.get("run_mode", "scheduler"),
        "html_report_path": status.get("html_report_path", ""),
        "markdown_report_path": status.get("markdown_report_path", ""),
        "paper_count": status.get("paper_count", 0),
        "fresh_paper_count": status.get("fresh_paper_count", 0),
        "reappeared_paper_with_update_count": status.get("reappeared_paper_with_update_count", 0),
        "update_count": status.get("update_count", 0),
        "new_articles_count": status.get("new_articles_count", 0),
        "quality_status": status.get("quality_status", ""),
        "post_send_quality_scan": status.get("post_send_quality_scan", {}),
        "ui_audit": status.get("ui_audit", {}),
        "quality_diagnostics": status.get("quality_diagnostics", {}),
        "source_health": status.get("source_health", {}),
        "source_weight_adjustments": status.get("source_weight_adjustments", {}),
        "email_subject": status.get("email_subject", ""),
        "delivery_verification": status.get("delivery_verification", {}),
        "log_file": status.get("log_file", ""),
    }


def is_pid_running(pid: int) -> bool:
    if pid <= 0:
        return False
    process_query_limited_information = 0x1000
    handle = ctypes.windll.kernel32.OpenProcess(process_query_limited_information, False, pid)
    if handle == 0:
        return False
    ctypes.windll.kernel32.CloseHandle(handle)
    return True


def acquire_run_lock(
    lock_path: Path,
    stale_lock_seconds: int,
    dead_pid_grace_seconds: int = 60,
) -> Tuple[bool, Dict[str, Any]]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    payload = {
        "pid": os.getpid(),
        "acquired_at": now.isoformat(timespec="seconds"),
        "hostname": os.environ.get("COMPUTERNAME", ""),
    }

    if lock_path.exists():
        existing = read_json(lock_path)
        existing_pid = int(existing.get("pid", 0) or 0)
        acquired_at = existing.get("acquired_at", "")
        pid_running = False
        try:
            acquired_dt = datetime.fromisoformat(acquired_at)
            age_seconds = max(0, int((now - acquired_dt).total_seconds()))
        except ValueError:
            age_seconds = stale_lock_seconds + 1

        if existing_pid:
            pid_running = is_pid_running(existing_pid)
        existing["age_seconds"] = age_seconds
        existing["pid_running"] = pid_running
        if pid_running or age_seconds <= max(0, int(dead_pid_grace_seconds)):
            return False, existing
        lock_path.unlink(missing_ok=True)

    try:
        with lock_path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
    except FileExistsError:
        return False, read_json(lock_path)
    return True, payload


def release_run_lock(lock_path: Path) -> None:
    existing = read_json(lock_path)
    existing_pid = int(existing.get("pid", 0) or 0)
    if existing_pid and existing_pid != os.getpid():
        return
    lock_path.unlink(missing_ok=True)


def _build_exception_result(error: str, started_at: datetime, trace: str = "") -> Dict[str, Any]:
    return {
        "success": False,
        "status": "exception",
        "retryable": True,
        "delivery_status": "failed",
        "error": error,
        "traceback": trace,
        "started_at": started_at.isoformat(timespec="seconds"),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }


def parse_worker_result_line(line: str) -> Optional[Dict[str, Any]]:
    stripped = line.strip()
    if not stripped.startswith(RESULT_MARKER):
        return None
    payload = stripped[len(RESULT_MARKER) :]
    try:
        return dict(json.loads(payload))
    except Exception:
        return None


def parse_email_committed_line(line: str) -> Optional[Dict[str, Any]]:
    stripped = line.strip()
    if not stripped.startswith(EMAIL_COMMITTED_MARKER):
        return None
    payload = stripped[len(EMAIL_COMMITTED_MARKER) :]
    try:
        return dict(json.loads(payload))
    except Exception:
        return None


def load_application_main(main_path: Optional[Path] = None) -> Any:
    resolved_path = (main_path or (ROOT / "main.py")).resolve()
    if not resolved_path.is_file():
        raise RuntimeError(f"Application entrypoint not found: {resolved_path}")
    # Windows multiprocessing uses spawn and re-imports the worker target by module
    # name. The production entrypoint therefore needs the stable, importable name
    # `main`; temporary test entrypoints stay isolated from the real application.
    module_name = "main" if resolved_path == (ROOT / "main.py").resolve() else f"web_agent_application_main_{os.getpid()}"
    spec = importlib.util.spec_from_file_location(module_name, resolved_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load application entrypoint: {resolved_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    entrypoint = getattr(module, "main", None)
    if not callable(entrypoint):
        sys.modules.pop(module_name, None)
        raise RuntimeError(f"Application entrypoint has no callable main(): {resolved_path}")
    return entrypoint


def _run_main_worker() -> int:
    configure_utf8_stdio()
    started_at = datetime.now()
    try:
        result = dict(load_application_main()() or {})
    except BaseException as exc:  # pragma: no cover - exercised through runner integration
        result = _build_exception_result(str(exc), started_at, traceback.format_exc())

    print(f"{RESULT_MARKER}{json.dumps(result, ensure_ascii=False)}", flush=True)
    return 0 if result.get("success", False) else 1


def _stream_subprocess_output(stream: Any, output_queue: "queue.Queue[str]") -> None:
    try:
        for line in iter(stream.readline, ""):
            output_queue.put(line)
    finally:
        stream.close()


def _build_timeout_result(max_run_seconds: int, started_at: datetime) -> Dict[str, Any]:
    return {
        "success": False,
        "status": "timeout",
        "retryable": True,
        "delivery_status": "failed",
        "error": f"main.py exceeded timeout of {max_run_seconds} seconds",
        "started_at": started_at.isoformat(timespec="seconds"),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }


def _build_committed_email_result(
    committed: Dict[str, Any],
    started_at: datetime,
    *,
    status: str,
    error: str,
) -> Dict[str, Any]:
    return {
        **dict(committed),
        "success": True,
        "status": status,
        "retryable": False,
        "delivery_status": "sent",
        "error": error,
        "started_at": started_at.isoformat(timespec="seconds"),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }


def run_main_once(
    max_run_seconds: int,
    email_mode: str = "send",
    run_profile: str = "",
    report_only: bool = False,
    send_slot_id: str = "",
) -> Dict[str, Any]:
    started_at = datetime.now()
    started_monotonic = time.monotonic()
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["WEB_AGENT_EMAIL_MODE"] = email_mode
    if send_slot_id:
        env["WEB_AGENT_SEND_SLOT_ID"] = send_slot_id
    else:
        env.pop("WEB_AGENT_SEND_SLOT_ID", None)
    if run_profile:
        env["WEB_AGENT_RUN_PROFILE"] = run_profile
    if report_only:
        env["WEB_AGENT_REPORT_ONLY"] = "1"
    command = [sys.executable, str(ROOT / "scheduler_runner.py"), "--worker"]
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    process = subprocess.Popen(
        command,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
        creationflags=creationflags,
    )

    output_queue: "queue.Queue[str]" = queue.Queue()
    stdout_thread = threading.Thread(
        target=_stream_subprocess_output,
        args=(process.stdout, output_queue),
        daemon=True,
    )
    stdout_thread.start()

    result: Dict[str, Any] = {}
    email_committed: Dict[str, Any] = {}
    while True:
        try:
            line = output_queue.get(timeout=0.5)
            parsed = parse_worker_result_line(line)
            if parsed is not None:
                result = parsed
                continue
            committed = parse_email_committed_line(line)
            if committed is not None:
                email_committed = committed
                print("Email submission committed; retries are now disabled.")
                continue
            print(line, end="")
        except queue.Empty:
            pass

        if process.poll() is not None and output_queue.empty():
            break

        if time.monotonic() - started_monotonic > max_run_seconds:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
            stdout_thread.join(timeout=2)
            if email_committed:
                return _build_committed_email_result(
                    email_committed,
                    started_at,
                    status="sent_followup_timeout",
                    error=f"Email was submitted, but main.py exceeded timeout of {max_run_seconds} seconds during follow-up.",
                )
            return _build_timeout_result(max_run_seconds, started_at)

    stdout_thread.join(timeout=2)

    if email_committed and (not result or not result.get("success", False)):
        return _build_committed_email_result(
            email_committed,
            started_at,
            status="sent_followup_failed",
            error=str(result.get("error") or f"main.py exited with code {process.returncode} after email submission"),
        )

    if not result:
        return _build_exception_result(
            f"main.py exited with code {process.returncode}",
            started_at,
            "",
        )

    if not result.get("finished_at"):
        result["finished_at"] = datetime.now().isoformat(timespec="seconds")
    if process.returncode not in (0, None) and result.get("success", False):
        result["status"] = "warning"
    return result


def build_failure_email_html(status: Dict[str, Any], log_file: Path) -> str:
    html_report_path = status.get("html_report_path") or "N/A"
    markdown_report_path = status.get("markdown_report_path") or "N/A"
    error = status.get("error") or status.get("status") or "unknown"
    traceback_text = status.get("traceback", "")
    traceback_block = (
        f"<pre style='white-space:pre-wrap;background:#f8fafc;padding:12px;border-radius:8px;'>"
        f"{traceback_text}</pre>"
        if traceback_text
        else ""
    )
    return f"""
    <html lang="zh-CN">
    <body style="font-family:Segoe UI,Microsoft YaHei,sans-serif;color:#16212f;">
        <h2>AI 日报调度失败</h2>
        <p>本次自动发送未成功完成，请查看以下信息：</p>
        <ul>
            <li>状态：{status.get("status", "unknown")}</li>
            <li>开始时间：{status.get("started_at", "")}</li>
            <li>结束时间：{status.get("finished_at", "")}</li>
            <li>报告 HTML：{html_report_path}</li>
            <li>报告 Markdown：{markdown_report_path}</li>
            <li>日志文件：{log_file.as_posix()}</li>
            <li>错误：{error}</li>
        </ul>
        {traceback_block}
    </body>
    </html>
    """


def send_failure_email(config: Dict[str, Any], scheduler_config: Dict[str, Any], status: Dict[str, Any], log_file: Path) -> bool:
    if not scheduler_config.get("send_failure_email", True):
        return False

    load_dotenv(ROOT / ".env")
    recipient_env = str(config.get("alerts", {}).get("recipients_env", "EMAIL_RECIPIENT"))
    recipient = os.getenv(recipient_env) or os.getenv("EMAIL_RECIPIENT")
    sender = os.getenv("EMAIL_SENDER")
    password = os.getenv("EMAIL_PASSWORD")
    smtp_server = os.getenv("EMAIL_SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.getenv("EMAIL_SMTP_PORT", "587"))

    if not recipient or not sender or not password:
        print("Skipping failure email because email credentials are incomplete.")
        return False

    notifier = EmailNotifier(
        smtp_server=smtp_server,
        smtp_port=smtp_port,
        sender_email=sender,
        sender_password=password,
        timeout_seconds=int(config.get("email", {}).get("timeout_seconds", config.get("network", {}).get("timeout_seconds", 25))),
        max_attempts=int(config.get("email", {}).get("max_attempts", 2)),
        retry_delay_seconds=int(config.get("email", {}).get("retry_delay_seconds", 5)),
    )
    subject_prefix = str(scheduler_config.get("failure_email_subject_prefix", "[AI日报调度失败]"))
    subject = f"{subject_prefix} {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    return notifier.send_email(
        recipient_email=recipient,
        subject=subject,
        html_content=build_failure_email_html(status, log_file),
    )


def main(validate_run: bool = False, dry_run: bool = False, report_only: bool = False) -> int:
    configure_utf8_stdio()
    os.chdir(ROOT)
    config, scheduler_config = load_runtime_config()
    non_sending_run = bool(validate_run or dry_run or report_only)
    log_dir = Path(scheduler_config["log_dir"])
    status_path = Path(
        scheduler_config["validation_status_file"] if non_sending_run else scheduler_config["status_file"]
    )
    last_success_path = Path(scheduler_config["last_success_file"])
    lock_path = Path(scheduler_config["lock_file"])
    log_dir.mkdir(parents=True, exist_ok=True)
    run_label = "validation" if validate_run else "rerender" if report_only else "dry-run" if dry_run else "scheduler"
    log_file = log_dir / f"{run_label}_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    with log_file.open("w", encoding="utf-8", buffering=1) as log_handle:
        tee = Tee(sys.__stdout__, log_handle)
        with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
            print(f"[{datetime.now().isoformat(timespec='seconds')}] {run_label.capitalize()} runner started.")
            archived_logs = archive_old_logs(
                log_dir,
                Path(str(scheduler_config.get("log_archive_dir", log_dir / "archive"))),
                int(scheduler_config.get("log_archive_after_days", 14)),
                keep_paths={log_file},
            )
            removed_logs = cleanup_old_logs(
                log_dir,
                int(scheduler_config.get("log_retention_days", 30)),
                keep_paths={log_file},
            )
            removed_archived_logs = cleanup_old_logs(
                Path(str(scheduler_config.get("log_archive_dir", log_dir / "archive"))),
                int(scheduler_config.get("log_retention_days", 30)),
            )
            removed_validation_reports = cleanup_validation_reports(
                Path(scheduler_config["validation_report_dir"]),
                int(scheduler_config.get("validation_report_retention_days", 7)),
            )
            removed_task_backups = cleanup_task_backups(
                Path(str(scheduler_config.get("task_backup_dir", log_dir / "task_backups"))),
                int(scheduler_config.get("task_backup_retention_days", 30)),
                int(scheduler_config.get("task_backup_keep_count", 20)),
            )
            cleanup_summary = {
                "archived_logs": archived_logs,
                "removed_logs": removed_logs,
                "removed_archived_logs": removed_archived_logs,
                "removed_validation_reports": removed_validation_reports,
                "removed_task_backups": removed_task_backups,
            }
            if archived_logs:
                print(f"Archived {len(archived_logs)} old log files.")
            if removed_logs:
                print(f"Cleaned up {len(removed_logs)} expired log files.")
            if removed_archived_logs:
                print(f"Cleaned up {len(removed_archived_logs)} expired archived log files.")
            if removed_validation_reports:
                print(f"Cleaned up {len(removed_validation_reports)} expired validation report files.")
            if removed_task_backups:
                print(f"Cleaned up {len(removed_task_backups)} expired task backup files.")
            acquired, lock_info = acquire_run_lock(lock_path, int(scheduler_config["stale_lock_seconds"]))
            if not acquired:
                status = {
                    "success": False,
                    "status": "skipped_locked",
                    "retryable": False,
                    "delivery_status": "skipped",
                    "run_mode": run_label,
                    "lock_info": lock_info,
                    "started_at": datetime.now().isoformat(timespec="seconds"),
                    "finished_at": datetime.now().isoformat(timespec="seconds"),
                    "log_file": log_file.as_posix(),
                    "attempts_used": 0,
                    "cleanup_summary": cleanup_summary,
                }
                write_scheduler_status(status_path, status, scheduler_config)
                print("Another scheduler run is still active. Skipping this trigger.")
                return 0

            try:
                skip_for_window = False
                last_status: Dict[str, Any] = {}
                if not non_sending_run:
                    skip_for_window, last_status = should_skip_for_idempotency(
                        status_path,
                        int(scheduler_config.get("idempotency_window_minutes", 90)),
                        last_success_path,
                    )
                if skip_for_window:
                    status = {
                        "success": True,
                        "status": "skipped_recent_success",
                        "retryable": False,
                        "delivery_status": "skipped",
                        "run_mode": run_label,
                        "started_at": datetime.now().isoformat(timespec="seconds"),
                        "finished_at": datetime.now().isoformat(timespec="seconds"),
                        "log_file": log_file.as_posix(),
                        "attempts_used": 0,
                        "cleanup_summary": cleanup_summary,
                        "previous_success": {
                            "finished_at": last_status.get("finished_at", ""),
                            "html_report_path": last_status.get("html_report_path", ""),
                            "markdown_report_path": last_status.get("markdown_report_path", ""),
                        },
                    }
                    write_scheduler_status(status_path, status, scheduler_config)
                    print("Skipping this trigger because a recent successful send already covered the current window.")
                    return 0

                send_slot_info: Dict[str, Any] = {}
                if not non_sending_run:
                    resolved_send_slot = resolve_send_slot(scheduler_config=scheduler_config)
                    if not resolved_send_slot.get("allowed", False):
                        status = {
                            "success": True,
                            "status": "skipped_outside_send_window",
                            "retryable": False,
                            "delivery_status": "skipped",
                            "run_mode": run_label,
                            "send_slot": resolved_send_slot,
                            "started_at": datetime.now().isoformat(timespec="seconds"),
                            "finished_at": datetime.now().isoformat(timespec="seconds"),
                            "log_file": log_file.as_posix(),
                            "attempts_used": 0,
                            "cleanup_summary": cleanup_summary,
                        }
                        write_scheduler_status(status_path, status, scheduler_config)
                        print("Skipping this trigger because it is outside the configured send windows.")
                        return 0
                    send_slot_id = str(resolved_send_slot["slot_id"])
                    send_slot_acquired, send_slot_info = acquire_send_slot(
                        Path(str(scheduler_config["send_slot_dir"])),
                        send_slot_id,
                        int(scheduler_config.get("send_slot_stale_seconds", scheduler_config["stale_lock_seconds"])),
                    )
                    send_slot_info.update(
                        {
                            "scheduled_at": resolved_send_slot.get("scheduled_at", ""),
                            "window_start": resolved_send_slot.get("window_start", ""),
                            "window_end": resolved_send_slot.get("window_end", ""),
                        }
                    )
                    if not send_slot_acquired:
                        status = {
                            "success": True,
                            "status": "skipped_duplicate_slot",
                            "retryable": False,
                            "delivery_status": "skipped",
                            "run_mode": run_label,
                            "send_slot": send_slot_info,
                            "started_at": datetime.now().isoformat(timespec="seconds"),
                            "finished_at": datetime.now().isoformat(timespec="seconds"),
                            "log_file": log_file.as_posix(),
                            "attempts_used": 0,
                            "cleanup_summary": cleanup_summary,
                        }
                        write_scheduler_status(status_path, status, scheduler_config)
                        print(f"Skipping this trigger because send slot {send_slot_id} is already claimed.")
                        return 0

                final_status: Dict[str, Any] = {}
                failure_email_sent = False
                max_attempts = max(1, int(scheduler_config["max_attempts"]))
                retry_delay_seconds = max(0, int(scheduler_config["retry_delay_seconds"]))
                max_run_seconds = max(
                    60,
                    int(
                        scheduler_config["validation_max_run_seconds"]
                        if validate_run
                        else scheduler_config["max_run_seconds"]
                    ),
                )

                for attempt in range(1, max_attempts + 1):
                    print(f"Starting {run_label} attempt {attempt}/{max_attempts}.")
                    run_kwargs = {
                        "email_mode": "dry-run" if non_sending_run else "send",
                        "run_profile": "validation_fast" if validate_run else "",
                    }
                    if send_slot_info:
                        run_kwargs["send_slot_id"] = str(send_slot_info.get("slot_id", "") or "")
                    if report_only:
                        run_kwargs["report_only"] = True
                    attempt_status = run_main_once(max_run_seconds, **run_kwargs)
                    attempt_status["attempts_used"] = attempt
                    attempt_status["log_file"] = log_file.as_posix()
                    attempt_status["run_mode"] = run_label
                    final_status = attempt_status

                    retryable = bool(attempt_status.get("retryable", False))
                    delivery_failed = attempt_status.get("delivery_status") == "failed"
                    if attempt_status.get("success", False) and not delivery_failed:
                        print(f"{run_label.capitalize()} attempt {attempt} succeeded.")
                        break
                    attempt_started_at = _parse_status_datetime(attempt_status.get("started_at")) or datetime.now()
                    committed_status = recover_committed_send_slot(send_slot_info, started_at=attempt_started_at)
                    if committed_status:
                        committed_status["attempts_used"] = attempt
                        committed_status["log_file"] = log_file.as_posix()
                        committed_status["run_mode"] = run_label
                        final_status = committed_status
                        print("Recovered a durable sent commit from the send slot; further retries are disabled.")
                        break
                    if attempt >= max_attempts or not retryable:
                        print(f"{run_label.capitalize()} attempt {attempt} finished without a retry.")
                        break
                    print(f"{run_label.capitalize()} attempt {attempt} failed. Waiting {retry_delay_seconds} seconds before retry.")
                    time.sleep(retry_delay_seconds)

                if not final_status.get("success", False) and not non_sending_run:
                    failure_email_sent = send_failure_email(config, scheduler_config, final_status, log_file)
                    if failure_email_sent:
                        print("Failure alert email sent successfully.")
                    else:
                        print("Failure alert email was not sent.")

                final_status["failure_email_sent"] = failure_email_sent
                final_status["log_file"] = log_file.as_posix()
                if send_slot_info:
                    final_status["send_slot"] = send_slot_info
                final_status["cleanup_summary"] = cleanup_summary
                if final_status.get("success", False):
                    final_status = refresh_report_quality_after_run(final_status)
                    final_status = refresh_email_ui_audit_after_run(final_status, scheduler_config)
                if send_slot_info:
                    finalize_send_slot(send_slot_info, final_status)
                write_scheduler_status(status_path, final_status, scheduler_config)
                if (
                    not non_sending_run
                    and final_status.get("success", False)
                    and final_status.get("delivery_status") == "sent"
                ):
                    write_json(last_success_path, build_last_success_snapshot(final_status))
                print(
                    f"[{datetime.now().isoformat(timespec='seconds')}] "
                    f"{run_label.capitalize()} runner finished with status {final_status.get('status')}."
                )
                return 0 if final_status.get("success", False) else 1
            finally:
                release_run_lock(lock_path)


if __name__ == "__main__":
    configure_utf8_stdio()
    raw_args = sys.argv[1:]
    args = set(raw_args)
    if "--worker" in args:
        sys.exit(_run_main_worker())
    if "--cleanup-legacy-tasks" in args:
        sys.exit(print_legacy_cleanup_report(as_json="--json" in args, confirm="--confirm" in args))
    if "--restore-task-backup" in args:
        try:
            backup_arg = raw_args[raw_args.index("--restore-task-backup") + 1]
        except (ValueError, IndexError):
            backup_arg = ""
        sys.exit(print_task_restore_report(backup_arg, as_json="--json" in args, confirm="--confirm" in args))
    if "--task-backups" in args:
        sys.exit(print_task_backups_report(as_json="--json" in args))
    if "--repair-plan" in args:
        sys.exit(print_repair_plan(as_json="--json" in args))
    if "--doctor" in args:
        sys.exit(
            print_doctor_report(
                as_json="--json" in args,
                self_heal="--self-heal" in args,
                dry_run="--dry-run" in args,
                persist_history="--record" in args,
            )
        )
    if "--status" in args:
        sys.exit(print_scheduler_status(as_json="--json" in args))
    if "--send-calendar" in args:
        sys.exit(print_send_calendar_report(as_json="--json" in args))
    if "--check-arrival" in args:
        sys.exit(print_email_arrival_report(as_json="--json" in args))
    if "--backfill-model-path" in args:
        sys.exit(
            print_model_path_backfill_report(
                limit=_parse_int_arg(raw_args, "--limit", 10),
                dry_run="--dry-run" in args,
                as_json="--json" in args,
                report_id=_parse_str_arg(raw_args, "--report-id", ""),
            )
        )
    if "--refresh-fallback-model-path" in args:
        sys.exit(
            print_model_path_backfill_report(
                limit=_parse_int_arg(raw_args, "--limit", 10),
                dry_run="--dry-run" in args,
                as_json="--json" in args,
                include_v2_fallback=True,
                report_id=_parse_str_arg(raw_args, "--report-id", ""),
            )
        )
    if "--scan-report-quality" in args:
        sys.exit(
            print_report_quality_scan(
                report_id=_parse_str_arg(raw_args, "--report-id", ""),
                limit=_parse_int_arg(raw_args, "--limit", 10),
                as_json="--json" in args,
                persist="--record" in args,
            )
        )
    if "--fix-bad-titles" in args:
        sys.exit(
            print_bad_title_fix_report(
                report_id=_parse_str_arg(raw_args, "--report-id", ""),
                limit=_parse_int_arg(raw_args, "--limit", 10),
                dry_run="--dry-run" in args,
                as_json="--json" in args,
            )
        )
    sys.exit(
        main(
            validate_run="--validate-run" in args,
            dry_run="--dry-run" in args,
            report_only="--report-only" in args,
        )
    )
