from __future__ import annotations

import contextlib
import ctypes
import json
import locale
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml
from dotenv import load_dotenv

from src.notifier import EmailNotifier, resolve_imap_server, verify_email_arrival

ROOT = Path(__file__).resolve().parent
RESULT_MARKER = "__SCHEDULER_RESULT__="
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
    267008: "ready",
    267009: "running",
    267010: "disabled",
    267011: "never_ran",
}
TASK_RESULT_MESSAGES: Dict[int, str] = {
    0: "The operation completed successfully.",
    267008: "Task is ready to run at the next scheduled time.",
    267009: "Task is currently running.",
    267010: "Task is disabled.",
    267011: "Task has not run yet.",
}
DOCTOR_PASSING_HINTS = {"success", "ready", "running", "never_ran"}
QUALITY_INFO_WARNING_PREFIXES = ("dedupe_removed_updates",)
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
    "task_setup_script": "setup_scheduled_tasks.ps1",
    "offline_task_setup_script": "setup_offline_tasks.ps1",
    "repair_task_script": "repair_scheduled_tasks.ps1",
    "task_names": ["Web_Agent_Send_1200_v2", "Web_Agent_Send_2100_v2"],
    "legacy_task_names": ["Web_Agent_Send_1200", "Web_Agent_Send_2100"],
    "monitor_auxiliary_tasks": True,
    "require_offline_tasks": False,
    "run_as_user_env": "WEB_AGENT_RUNAS_USER",
    "run_as_password_env": "WEB_AGENT_RUNAS_PASSWORD",
    "send_slot_dir": "logs/send_slots",
    "send_slot_stale_seconds": 10800,
    "send_slots": [
        {"id": "1200", "time": "12:00"},
        {"id": "2100", "time": "21:00"},
    ],
    "send_window_before_minutes": 10,
    "send_window_after_minutes": 180,
    "doctor_task_name": "Web_Agent_Doctor_0900",
    "preflight_task_name": "Web_Agent_Preflight_2030",
    "log_archive_after_days": 14,
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
        "send_slot_dir",
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
    message = TASK_RESULT_MESSAGES.get(code, "").strip()
    if not message:
        raw_message = ctypes.FormatError(win32_code).strip()
        if raw_message and raw_message != "<no description>":
            message = raw_message.rstrip(".")

    return {
        "code": str(code),
        "hex": f"0x{unsigned_code:08X}",
        "hint": TASK_RESULT_HINTS.get(code, "unknown"),
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
    lines.append(f"Removed logs this run: {len(removed_logs)}")
    lines.append(f"Removed validation reports this run: {len(removed_validation_reports)}")

    if lock_info:
        lock_pid = int(lock_info.get("pid", 0) or 0)
        lock_state = "active" if is_pid_running(lock_pid) else "stale"
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


def _quote_command_arg(value: Any) -> str:
    text = str(value)
    if not text:
        return '""'
    if any(char.isspace() for char in text) or '"' in text:
        return '"' + text.replace('"', '\\"') + '"'
    return text


def build_task_repair_commands(scheduler_config: Dict[str, Any]) -> list[Dict[str, str]]:
    legacy_task_names = [str(name).lstrip("\\") for name in (scheduler_config.get("legacy_task_names") or [])]
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
    for task_name in legacy_task_names:
        if task_name:
            commands.append(
                {
                    "name": f"delete_legacy_{task_name}",
                    "purpose": "Remove an old scheduled task that can duplicate sends.",
                    "command": f"schtasks /Delete /TN {task_name} /F",
                }
            )
    commands.append(
        {
            "name": "rebuild_offline_tasks",
            "purpose": "Recreate 12:00 and 21:00 tasks with offline-capable Windows credentials.",
            "command": (
                "powershell -NoProfile -ExecutionPolicy Bypass -File "
                f"{_quote_command_arg(offline_script)}"
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


def build_doctor_payload() -> Dict[str, Any]:
    config, scheduler_config = load_runtime_config()
    load_dotenv(ROOT / ".env")
    status_payload = build_scheduler_status_payload()
    repair_commands = build_task_repair_commands(scheduler_config)
    checks: list[Dict[str, Any]] = []

    configured_slots = list(scheduler_config.get("send_slots") or [])
    configured_slot_times = [str(slot.get("time", "") or "") for slot in configured_slots]
    expected_slot_times = {"12:00", "21:00"}
    if set(configured_slot_times) == expected_slot_times and len(configured_slot_times) == 2:
        checks.append(
            _doctor_item(
                "send_schedule_config",
                "ok",
                "Configured send windows are exactly 12:00 and 21:00.",
                {"send_slots": configured_slots},
            )
        )
    else:
        checks.append(
            _doctor_item(
                "send_schedule_config",
                "warn",
                f"Expected exactly 12:00 and 21:00 send windows, got: {', '.join(configured_slot_times) or 'none'}",
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

    send_calendar = build_send_calendar_payload(
        scheduler_config,
        target_datetime=datetime.now(),
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
            if verification.get("verified", False):
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

    source_health = dict(last_run.get("source_health") or last_success.get("source_health") or {})
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
        <p>建议动作：先运行 <code>python D:\\Web_Agent\\scheduler_runner.py --doctor --self-heal</code>；如果是离线模式告警，请执行 <code>setup_offline_tasks.ps1</code> 并输入 Windows 密码。</p>
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


def run_task_self_heal(
    scheduler_config: Dict[str, Any],
    candidates: list[Dict[str, Any]],
    dry_run: bool = False,
) -> Dict[str, Any]:
    primary_task_names = {str(name) for name in (scheduler_config.get("task_names") or [])}
    require_offline_tasks = bool(scheduler_config.get("require_offline_tasks", False))
    needs_offline_rebuild = False
    for candidate in candidates:
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
    }
    if not candidates:
        result["message"] = "No scheduled task repair candidates were found."
        return result
    if not script_path.exists():
        result["message"] = f"Task setup script is missing: {script_path.as_posix()}"
        return result
    if dry_run:
        result["success"] = True
        result["message"] = "Dry run only. No changes were applied."
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
    result["success"] = completed.returncode == 0
    result["message"] = "Scheduled task repair completed." if result["success"] else "Scheduled task repair failed."
    return result


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
    return verify_email_arrival(
        imap_server=imap_server,
        imap_port=int(os.getenv("EMAIL_IMAP_PORT", str(arrival_config.get("imap_port", 993)))),
        username=os.getenv("EMAIL_IMAP_USERNAME", os.getenv("EMAIL_SENDER", "")),
        password=os.getenv("EMAIL_IMAP_PASSWORD", os.getenv("EMAIL_PASSWORD", "")),
        subject_contains=subject,
        since_minutes=int(arrival_config.get("since_minutes", 30)),
        mailbox=str(arrival_config.get("mailbox", "INBOX")),
        timeout_seconds=int(arrival_config.get("timeout_seconds", config.get("network", {}).get("timeout_seconds", 25))),
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
    for slot in slots:
        slot_id = str(slot.get("id", "") or "").strip()
        slot_time = str(slot.get("time", "") or "").strip()
        if not slot_id or not slot_time:
            continue
        try:
            hour, minute = parse_time_of_day(slot_time)
        except (TypeError, ValueError):
            continue
        scheduled_at = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
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


def acquire_send_slot(slot_dir: Path, slot_id: str, stale_seconds: int) -> Tuple[bool, Dict[str, Any]]:
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
        if age_seconds <= stale_seconds:
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
            }
        )
        write_json(slot_path, payload)
        return
    slot_path.unlink(missing_ok=True)


def _parse_status_datetime(value: Any) -> Optional[datetime]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


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
        health_status = raw_status
        if raw_status == "missing":
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
        "update_count": status.get("update_count", 0),
        "new_articles_count": status.get("new_articles_count", 0),
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


def acquire_run_lock(lock_path: Path, stale_lock_seconds: int) -> Tuple[bool, Dict[str, Any]]:
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
        if age_seconds <= stale_lock_seconds:
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


def _run_main_worker() -> int:
    configure_utf8_stdio()
    started_at = datetime.now()
    try:
        import importlib

        main_module = importlib.import_module("main")
        result = dict(main_module.main() or {})
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


def run_main_once(
    max_run_seconds: int,
    email_mode: str = "send",
    run_profile: str = "",
) -> Dict[str, Any]:
    started_at = datetime.now()
    started_monotonic = time.monotonic()
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["WEB_AGENT_EMAIL_MODE"] = email_mode
    if run_profile:
        env["WEB_AGENT_RUN_PROFILE"] = run_profile
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
    while True:
        try:
            line = output_queue.get(timeout=0.5)
            parsed = parse_worker_result_line(line)
            if parsed is not None:
                result = parsed
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
            return _build_timeout_result(max_run_seconds, started_at)

    stdout_thread.join(timeout=2)

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


def main(validate_run: bool = False) -> int:
    configure_utf8_stdio()
    os.chdir(ROOT)
    config, scheduler_config = load_runtime_config()
    log_dir = Path(scheduler_config["log_dir"])
    status_path = Path(
        scheduler_config["validation_status_file"] if validate_run else scheduler_config["status_file"]
    )
    last_success_path = Path(scheduler_config["last_success_file"])
    lock_path = Path(scheduler_config["lock_file"])
    log_dir.mkdir(parents=True, exist_ok=True)
    run_label = "validation" if validate_run else "scheduler"
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
            if archived_logs:
                print(f"Archived {len(archived_logs)} old log files.")
            if removed_logs:
                print(f"Cleaned up {len(removed_logs)} expired log files.")
            if removed_archived_logs:
                print(f"Cleaned up {len(removed_archived_logs)} expired archived log files.")
            if removed_validation_reports:
                print(f"Cleaned up {len(removed_validation_reports)} expired validation report files.")
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
                    "cleanup_summary": {
                        "archived_logs": archived_logs,
                        "removed_logs": removed_logs,
                        "removed_archived_logs": removed_archived_logs,
                        "removed_validation_reports": removed_validation_reports,
                    },
                }
                write_scheduler_status(status_path, status, scheduler_config)
                print("Another scheduler run is still active. Skipping this trigger.")
                return 0

            try:
                skip_for_window = False
                last_status: Dict[str, Any] = {}
                if not validate_run:
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
                        "cleanup_summary": {
                            "archived_logs": archived_logs,
                            "removed_logs": removed_logs,
                            "removed_archived_logs": removed_archived_logs,
                            "removed_validation_reports": removed_validation_reports,
                        },
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
                if not validate_run:
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
                            "cleanup_summary": {
                                "archived_logs": archived_logs,
                                "removed_logs": removed_logs,
                                "removed_archived_logs": removed_archived_logs,
                                "removed_validation_reports": removed_validation_reports,
                            },
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
                            "cleanup_summary": {
                                "archived_logs": archived_logs,
                                "removed_logs": removed_logs,
                                "removed_archived_logs": removed_archived_logs,
                                "removed_validation_reports": removed_validation_reports,
                            },
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
                    attempt_status = run_main_once(
                        max_run_seconds,
                        email_mode="dry-run" if validate_run else "send",
                        run_profile="validation_fast" if validate_run else "",
                    )
                    attempt_status["attempts_used"] = attempt
                    attempt_status["log_file"] = log_file.as_posix()
                    attempt_status["run_mode"] = run_label
                    final_status = attempt_status

                    retryable = bool(attempt_status.get("retryable", False))
                    delivery_failed = attempt_status.get("delivery_status") == "failed"
                    if attempt_status.get("success", False) and not delivery_failed:
                        print(f"{run_label.capitalize()} attempt {attempt} succeeded.")
                        break
                    if attempt >= max_attempts or not retryable:
                        print(f"{run_label.capitalize()} attempt {attempt} finished without a retry.")
                        break
                    print(f"{run_label.capitalize()} attempt {attempt} failed. Waiting {retry_delay_seconds} seconds before retry.")
                    time.sleep(retry_delay_seconds)

                if not final_status.get("success", False) and not validate_run:
                    failure_email_sent = send_failure_email(config, scheduler_config, final_status, log_file)
                    if failure_email_sent:
                        print("Failure alert email sent successfully.")
                    else:
                        print("Failure alert email was not sent.")

                final_status["failure_email_sent"] = failure_email_sent
                final_status["log_file"] = log_file.as_posix()
                if send_slot_info:
                    final_status["send_slot"] = send_slot_info
                final_status["cleanup_summary"] = {
                    "archived_logs": archived_logs,
                    "removed_logs": removed_logs,
                    "removed_archived_logs": removed_archived_logs,
                    "removed_validation_reports": removed_validation_reports,
                }
                if send_slot_info:
                    finalize_send_slot(send_slot_info, final_status)
                write_scheduler_status(status_path, final_status, scheduler_config)
                if (
                    not validate_run
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
    args = set(sys.argv[1:])
    if "--worker" in args:
        sys.exit(_run_main_worker())
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
    sys.exit(main(validate_run="--validate-run" in args))
