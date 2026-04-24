import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from scheduler_runner import (
    DEFAULT_SCHEDULER_CONFIG,
    RESULT_MARKER,
    acquire_run_lock,
    build_doctor_payload,
    build_doctor_alert_html,
    build_scheduler_status_payload,
    build_status_snapshot,
    build_status_text,
    build_failure_email_html,
    cleanup_old_logs,
    cleanup_validation_reports,
    collect_task_self_heal_candidates,
    describe_task_result,
    parse_worker_result_line,
    parse_schtasks_list_output,
    print_doctor_report,
    release_run_lock,
    run_task_self_heal,
    should_send_doctor_alert,
    should_skip_for_idempotency,
    main as scheduler_main,
    update_doctor_history,
    write_doctor_snapshot,
    write_json,
)


class SchedulerRunnerTests(unittest.TestCase):
    def test_acquire_run_lock_prevents_parallel_runs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            lock_path = Path(temp_dir) / "scheduler.lock"
            acquired, lock_info = acquire_run_lock(lock_path, stale_lock_seconds=3600)
            self.assertTrue(acquired)
            self.assertEqual(lock_info["pid"], os.getpid())

            acquired_again, existing = acquire_run_lock(lock_path, stale_lock_seconds=3600)
            self.assertFalse(acquired_again)
            self.assertEqual(existing["pid"], os.getpid())

            release_run_lock(lock_path)
            self.assertFalse(lock_path.exists())

    def test_acquire_run_lock_replaces_stale_lock(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            lock_path = Path(temp_dir) / "scheduler.lock"
            write_json(
                lock_path,
                {
                    "pid": 999999,
                    "acquired_at": (datetime.now() - timedelta(hours=5)).isoformat(timespec="seconds"),
                    "hostname": "stale-host",
                },
            )

            acquired, lock_info = acquire_run_lock(lock_path, stale_lock_seconds=60)
            self.assertTrue(acquired)
            self.assertEqual(lock_info["pid"], os.getpid())
            release_run_lock(lock_path)

    def test_build_failure_email_html_includes_log_and_status(self):
        status = {
            "status": "timeout",
            "started_at": "2026-04-08T23:30:00",
            "finished_at": "2026-04-08T23:45:00",
            "html_report_path": "archive/report_test.html",
            "markdown_report_path": "archive/report_test.md",
            "error": "main.py exceeded timeout",
        }
        html = build_failure_email_html(status, Path("logs/test.log"))
        self.assertIn("AI 日报调度失败", html)
        self.assertIn("archive/report_test.html", html)
        self.assertIn("logs/test.log", html)
        self.assertIn("timeout", html)

    def test_parse_worker_result_line_extracts_json_payload(self):
        payload = {"success": True, "status": "success", "delivery_status": "sent"}
        parsed = parse_worker_result_line(f"{RESULT_MARKER}{json.dumps(payload)}")
        self.assertEqual(parsed, payload)
        self.assertIsNone(parse_worker_result_line("normal log line"))

    def test_build_status_snapshot_strips_verbose_collector_rows(self):
        snapshot = build_status_snapshot(
            {
                "success": True,
                "collector_summary": {
                    "status_text": "ok",
                    "rows": [
                        {"label": "A", "status": "success", "error": "", "duration_seconds": 1.2},
                        {"label": "B", "status": "error", "error": "boom", "duration_seconds": 3.4},
                    ],
                },
            }
        )
        self.assertNotIn("rows", snapshot["collector_summary"])
        self.assertEqual(len(snapshot["collector_summary"]["failure_rows"]), 1)
        self.assertEqual(snapshot["collector_summary"]["failure_rows"][0]["label"], "B")

    def test_cleanup_old_logs_only_removes_expired_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir)
            old_log = log_dir / "old.log"
            fresh_log = log_dir / "fresh.log"
            old_log.write_text("old", encoding="utf-8")
            fresh_log.write_text("fresh", encoding="utf-8")

            old_ts = (datetime.now() - timedelta(days=45)).timestamp()
            os.utime(old_log, (old_ts, old_ts))

            removed = cleanup_old_logs(log_dir, retention_days=30)
            self.assertEqual(removed, ["old.log"])
            self.assertFalse(old_log.exists())
            self.assertTrue(fresh_log.exists())

    def test_cleanup_validation_reports_removes_old_html_and_markdown(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            validation_dir = Path(temp_dir)
            old_html = validation_dir / "report_20260401_0001.html"
            old_md = validation_dir / "report_20260401_0001.md"
            fresh_html = validation_dir / "report_20260414_0001.html"
            old_html.write_text("<html></html>", encoding="utf-8")
            old_md.write_text("# report", encoding="utf-8")
            fresh_html.write_text("<html></html>", encoding="utf-8")

            old_ts = (datetime.now() - timedelta(days=10)).timestamp()
            os.utime(old_html, (old_ts, old_ts))
            os.utime(old_md, (old_ts, old_ts))

            removed = cleanup_validation_reports(validation_dir, retention_days=7)

            self.assertEqual(sorted(removed), ["report_20260401_0001.html", "report_20260401_0001.md"])
            self.assertFalse(old_html.exists())
            self.assertFalse(old_md.exists())
            self.assertTrue(fresh_html.exists())

    def test_should_skip_for_idempotency_when_recent_success_exists(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            status_path = Path(temp_dir) / "last_run.json"
            write_json(
                status_path,
                {
                    "success": True,
                    "delivery_status": "sent",
                    "finished_at": datetime.now().isoformat(timespec="seconds"),
                    "html_report_path": "archive/report_recent.html",
                },
            )
            should_skip, last_status = should_skip_for_idempotency(status_path, window_minutes=90)
            self.assertTrue(should_skip)
            self.assertEqual(last_status["html_report_path"], "archive/report_recent.html")

    def test_should_skip_for_idempotency_allows_old_success(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            status_path = Path(temp_dir) / "last_run.json"
            write_json(
                status_path,
                {
                    "success": True,
                    "delivery_status": "sent",
                    "finished_at": (datetime.now() - timedelta(hours=5)).isoformat(timespec="seconds"),
                },
            )
            should_skip, _ = should_skip_for_idempotency(status_path, window_minutes=90)
            self.assertFalse(should_skip)

    def test_should_skip_for_idempotency_chains_previous_success_after_skip(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            status_path = Path(temp_dir) / "last_run.json"
            write_json(
                status_path,
                {
                    "success": True,
                    "status": "skipped_recent_success",
                    "delivery_status": "skipped",
                    "finished_at": datetime.now().isoformat(timespec="seconds"),
                    "previous_success": {
                        "finished_at": (datetime.now() - timedelta(minutes=30)).isoformat(timespec="seconds"),
                        "html_report_path": "archive/report_recent.html",
                    },
                },
            )

            should_skip, last_status = should_skip_for_idempotency(status_path, window_minutes=240)

            self.assertTrue(should_skip)
            self.assertEqual(last_status["delivery_status"], "sent")
            self.assertEqual(last_status["html_report_path"], "archive/report_recent.html")

    def test_parse_schtasks_list_output_extracts_key_fields(self):
        output = """
任务名:                             \\Web_Agent_Send_1200
下次运行时间:                       2026/4/12 12:00:00
模式:                               就绪
登录状态:                           交互方式/后台方式
上次运行时间:                       2026/4/12 02:47:00
上次结果:                           0
"""
        parsed = parse_schtasks_list_output(output)
        self.assertEqual(parsed["task_name"], "\\Web_Agent_Send_1200")
        self.assertEqual(parsed["next_run_time"], "2026/4/12 12:00:00")
        self.assertEqual(parsed["status"], "就绪")
        self.assertEqual(parsed["logon_mode"], "交互方式/后台方式")
        self.assertEqual(parsed["last_result"], "0")
        self.assertEqual(parsed["last_result_hex"], "0x00000000")
        self.assertEqual(parsed["last_result_hint"], "success")
        self.assertEqual(parsed["last_result_message"], "The operation completed successfully.")

    def test_build_status_text_includes_recent_success_and_tasks(self):
        last_status = {
            "status": "skipped_recent_success",
            "delivery_status": "skipped",
            "finished_at": "2026-04-12T02:50:00",
            "log_file": "D:/Web_Agent/logs/scheduler_run_test.log",
            "cleanup_summary": {
                "removed_logs": ["old1.log", "old2.log"],
                "removed_validation_reports": ["report_20260401_0001.html"],
            },
            "previous_success": {
                "finished_at": "2026-04-12T02:47:00",
                "html_report_path": "archive/report_20260412_0247.html",
            },
        }
        task_infos = [
            {
                "task_name": "Web_Agent_Send_1200",
                "available": True,
                "status": "就绪",
                "next_run_time": "2026/4/12 12:00:00",
                "logon_mode": "交互方式/后台方式",
                "last_result": "0",
                "last_result_hex": "0x00000000",
                "last_result_hint": "success",
                "last_result_message": "The operation completed successfully.",
            }
        ]
        validation_status = {
            "status": "dry_run",
            "delivery_status": "dry_run",
            "finished_at": "2026-04-12T03:00:00",
            "log_file": "D:/Web_Agent/logs/validation_run_test.log",
        }
        text = build_status_text(
            last_status,
            task_infos,
            {"pid": 1234, "acquired_at": "2026-04-12T02:49:00"},
            validation_status=validation_status,
        )
        self.assertIn("Recent status: skipped_recent_success", text)
        self.assertIn("Previous success HTML: archive/report_20260412_0247.html", text)
        self.assertIn("Recent validation status: dry_run", text)
        self.assertIn("Validation log: D:/Web_Agent/logs/validation_run_test.log", text)
        self.assertIn("Removed logs this run: 2", text)
        self.assertIn("Removed validation reports this run: 1", text)
        self.assertIn("Web_Agent_Send_1200: 就绪", text)
        self.assertIn("Current lock: stale, pid=1234", text)
        self.assertIn("(0x00000000, success)", text)
        self.assertIn("message: The operation completed successfully.", text)

    def test_describe_task_result_includes_hex_and_hint(self):
        result = describe_task_result("-2147020576")
        self.assertEqual(result["code"], "-2147020576")
        self.assertEqual(result["hex"], "0x800710E0")
        self.assertEqual(result["hint"], "unknown")
        self.assertEqual(result["message"], "操作员或系统管理员拒绝了请求。")

    def test_build_scheduler_status_payload_summarizes_last_run_lock_and_tasks(self):
        mocked_tasks = [
            {
                "task_name": "Web_Agent_Send_1200",
                "available": True,
                "status": "就绪",
                "next_run_time": "2026/4/12 12:00:00",
                "logon_mode": "交互方式/后台方式",
                "last_result": "267009",
                "last_result_hex": "0x00041301",
                "last_result_hint": "running",
                "last_result_message": "Task is currently running.",
            }
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            status_path = temp_root / "last_run.json"
            validation_status_path = temp_root / "last_validation_run.json"
            lock_path = temp_root / "scheduler.lock"
            write_json(status_path, {"status": "success", "delivery_status": "sent"})
            write_json(validation_status_path, {"status": "dry_run", "delivery_status": "dry_run"})
            write_json(lock_path, {"pid": 4567, "acquired_at": "2026-04-12T03:00:00"})

            scheduler_config = dict(DEFAULT_SCHEDULER_CONFIG)
            scheduler_config.update(
                {
                    "status_file": str(status_path),
                    "validation_status_file": str(validation_status_path),
                    "lock_file": str(lock_path),
                    "task_names": ["Web_Agent_Send_1200"],
                }
            )

            with patch("scheduler_runner.load_runtime_config", return_value=({}, scheduler_config)):
                with patch("scheduler_runner.query_scheduled_task", side_effect=mocked_tasks):
                    with patch("scheduler_runner.is_pid_running", return_value=True):
                        payload = build_scheduler_status_payload()

        self.assertEqual(payload["last_run"]["status"], "success")
        self.assertEqual(payload["last_validation_run"]["status"], "dry_run")
        self.assertEqual(payload["lock"]["state"], "active")
        self.assertEqual(payload["tasks"][0]["last_result_hint"], "running")

    def test_build_doctor_payload_reports_failures_and_warnings(self):
        mocked_status = {
            "current_time": "2026-04-12T03:10:00",
            "last_run": {
                "success": True,
                "status": "success",
                "delivery_status": "sent",
                "finished_at": "2026-04-12T03:05:00",
                "log_file": "D:/Web_Agent/logs/scheduler_run_ok.log",
            },
            "lock": {"state": "idle", "pid": "", "acquired_at": ""},
            "tasks": [
                {
                    "task_name": "\\Web_Agent_Send_1200",
                    "available": True,
                    "status": "就绪",
                    "next_run_time": "2026/4/12 12:00:00",
                    "last_result": "-2147020576",
                    "last_result_hint": "unknown",
                    "last_result_message": "操作员或系统管理员拒绝了请求。",
                },
                {
                    "task_name": "\\Web_Agent_Send_2100",
                    "available": True,
                    "status": "就绪",
                    "next_run_time": "2026/4/12 21:00:00",
                    "last_result": "0",
                    "last_result_hint": "success",
                    "last_result_message": "The operation completed successfully.",
                },
            ],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            archive_dir = temp_root / "archive"
            log_dir = temp_root / "logs"
            archive_dir.mkdir()
            log_dir.mkdir()
            (temp_root / "reports_manifest.json").write_text("{}", encoding="utf-8")

            scheduler_config = dict(DEFAULT_SCHEDULER_CONFIG)
            scheduler_config.update(
                {
                    "log_dir": str(log_dir),
                    "status_file": str(log_dir / "last_run.json"),
                    "lock_file": str(log_dir / "scheduler.lock"),
                }
            )
            config = {"archive": {"report_dir": str(archive_dir)}}

            with patch("scheduler_runner.ROOT", temp_root):
                with patch("scheduler_runner.load_runtime_config", return_value=(config, scheduler_config)):
                    with patch("scheduler_runner.build_scheduler_status_payload", return_value=mocked_status):
                        with patch.dict(
                            os.environ,
                            {
                                "EMAIL_RECIPIENT": "user@example.com",
                                "EMAIL_SENDER": "bot@example.com",
                                "EMAIL_PASSWORD": "secret",
                            },
                            clear=False,
                        ):
                            payload = build_doctor_payload()

        self.assertEqual(payload["overall"], "warn")
        self.assertEqual(payload["counts"]["fail"], 0)
        self.assertEqual(payload["counts"]["warn"], 1)
        warned = [item for item in payload["checks"] if item["level"] == "warn"]
        self.assertTrue(any("Web_Agent_Send_1200" in item["name"] for item in warned))

    def test_build_doctor_payload_reports_missing_email_as_failure(self):
        mocked_status = {
            "current_time": "2026-04-12T03:10:00",
            "last_run": {},
            "lock": {"state": "idle", "pid": "", "acquired_at": ""},
            "tasks": [],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            scheduler_config = dict(DEFAULT_SCHEDULER_CONFIG)
            scheduler_config.update(
                {
                    "log_dir": str(temp_root / "logs"),
                    "status_file": str(temp_root / "logs" / "last_run.json"),
                    "lock_file": str(temp_root / "logs" / "scheduler.lock"),
                }
            )
            (temp_root / "logs").mkdir()
            config = {"archive": {"report_dir": "archive"}}

            with patch("scheduler_runner.ROOT", temp_root):
                with patch("scheduler_runner.load_runtime_config", return_value=(config, scheduler_config)):
                    with patch("scheduler_runner.build_scheduler_status_payload", return_value=mocked_status):
                        with patch.dict(
                            os.environ,
                            {
                                "EMAIL_RECIPIENT": "",
                                "EMAIL_SENDER": "",
                                "EMAIL_PASSWORD": "",
                            },
                            clear=False,
                        ):
                            payload = build_doctor_payload()

        self.assertEqual(payload["overall"], "fail")
        failed = [item for item in payload["checks"] if item["level"] == "fail"]
        self.assertTrue(any(item["name"] == "email_env" for item in failed))

    def test_write_doctor_snapshot_persists_payload(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            snapshot_path = Path(temp_dir) / "doctor_latest.json"
            scheduler_config = dict(DEFAULT_SCHEDULER_CONFIG)
            scheduler_config["doctor_status_file"] = str(snapshot_path)
            payload = {"overall": "ok", "counts": {"ok": 1, "warn": 0, "fail": 0}}

            written_path = write_doctor_snapshot(payload, scheduler_config)

            self.assertEqual(written_path, snapshot_path)
            persisted = json.loads(snapshot_path.read_text(encoding="utf-8-sig"))
            self.assertEqual(persisted["overall"], "ok")

    def test_update_doctor_history_tracks_warn_streak(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            history_path = Path(temp_dir) / "doctor_history.json"
            scheduler_config = dict(DEFAULT_SCHEDULER_CONFIG)
            scheduler_config["doctor_history_file"] = str(history_path)

            first = update_doctor_history(
                {"current_time": "2026-04-12T03:20:00", "overall": "warn", "counts": {}, "checks": []},
                scheduler_config,
            )
            second = update_doctor_history(
                {"current_time": "2026-04-12T03:30:00", "overall": "warn", "counts": {}, "checks": []},
                scheduler_config,
            )
            third = update_doctor_history(
                {"current_time": "2026-04-12T03:40:00", "overall": "ok", "counts": {}, "checks": []},
                scheduler_config,
            )

            self.assertEqual(first["warn_streak"], 1)
            self.assertEqual(second["warn_streak"], 2)
            self.assertEqual(third["warn_streak"], 0)

    def test_should_send_doctor_alert_respects_warn_threshold_and_dedup(self):
        scheduler_config = dict(DEFAULT_SCHEDULER_CONFIG)
        scheduler_config["doctor_warn_streak_threshold"] = 2

        payload = {
            "overall": "warn",
            "checks": [{"name": "task:noon", "level": "warn", "detail": "problem"}],
        }
        history = {
            "warn_streak": 2,
            "last_alert_signature": "",
            "last_alert_overall": "",
        }
        self.assertTrue(should_send_doctor_alert(payload, history, scheduler_config))

        history["last_alert_signature"] = "task:noon:warn:problem"
        history["last_alert_overall"] = "warn"
        self.assertFalse(should_send_doctor_alert(payload, history, scheduler_config))

    def test_build_doctor_alert_html_lists_non_ok_checks(self):
        payload = {
            "overall": "fail",
            "current_time": "2026-04-12T03:50:00",
            "counts": {"ok": 3, "warn": 1, "fail": 1},
            "checks": [
                {"name": "email_env", "level": "fail", "detail": "Missing required email settings"},
                {"name": "task:noon", "level": "warn", "detail": "Task result unknown"},
            ],
        }
        history = {"warn_streak": 2}
        html = build_doctor_alert_html(payload, history)
        self.assertIn("AI 日报健康检查告警", html)
        self.assertIn("email_env", html)
        self.assertIn("Task result unknown", html)

    def test_collect_task_self_heal_candidates_uses_unavailable_and_unknown_tasks(self):
        payload = {
            "status": {
                "tasks": [
                    {"task_name": "\\Web_Agent_Send_1200", "available": False, "error": "query failed"},
                    {
                        "task_name": "\\Web_Agent_Send_2100",
                        "available": True,
                        "last_result_hint": "unknown",
                        "last_result": "-2147020576",
                        "last_result_hex": "0x800710E0",
                        "last_result_message": "operator denied request",
                    },
                    {
                        "task_name": "\\Web_Agent_Send_0900",
                        "available": True,
                        "last_result_hint": "success",
                        "last_result": "0",
                    },
                ]
            }
        }
        candidates = collect_task_self_heal_candidates(payload)
        self.assertEqual(len(candidates), 2)
        self.assertEqual(candidates[0]["reason"], "unavailable")
        self.assertEqual(candidates[1]["reason"], "bad_last_result")

    def test_run_task_self_heal_supports_dry_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            script_path = Path(temp_dir) / "setup_scheduled_tasks.ps1"
            script_path.write_text("Write-Host 'ok'", encoding="utf-8")
            scheduler_config = dict(DEFAULT_SCHEDULER_CONFIG)
            scheduler_config["task_setup_script"] = str(script_path)
            candidates = [{"task_name": "Web_Agent_Send_1200", "reason": "bad_last_result"}]

            result = run_task_self_heal(scheduler_config, candidates, dry_run=True)

            self.assertTrue(result["attempted"])
            self.assertTrue(result["success"])
            self.assertEqual(result["message"], "Dry run only. No changes were applied.")

    def test_print_doctor_report_writes_snapshot_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            snapshot_path = Path(temp_dir) / "doctor_latest.json"
            history_path = Path(temp_dir) / "doctor_history.json"
            scheduler_config = dict(DEFAULT_SCHEDULER_CONFIG)
            scheduler_config["doctor_status_file"] = str(snapshot_path)
            scheduler_config["doctor_history_file"] = str(history_path)
            scheduler_config["send_doctor_alert_email"] = True
            payload = {
                "overall": "warn",
                "counts": {"ok": 1, "warn": 1, "fail": 0},
                "checks": [{"name": "task", "level": "warn", "detail": "needs review"}],
            }

            with patch("scheduler_runner.build_doctor_payload", return_value=payload):
                with patch("scheduler_runner.load_runtime_config", return_value=({}, scheduler_config)):
                    with patch("scheduler_runner.send_doctor_alert_email", return_value=True):
                        with patch("sys.stdout.write"):
                            exit_code = print_doctor_report(as_json=False, persist_history=True)

            self.assertEqual(exit_code, 0)
            persisted = json.loads(snapshot_path.read_text(encoding="utf-8-sig"))
            self.assertEqual(persisted["overall"], "warn")
            self.assertEqual(persisted["history_summary"]["warn_streak"], 1)
            self.assertTrue(persisted["history_summary"]["recorded"])
            persisted_history = json.loads(history_path.read_text(encoding="utf-8-sig"))
            self.assertEqual(len(persisted_history["entries"]), 1)

    def test_print_doctor_report_self_heal_failure_returns_nonzero(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            snapshot_path = Path(temp_dir) / "doctor_latest.json"
            history_path = Path(temp_dir) / "doctor_history.json"
            scheduler_config = dict(DEFAULT_SCHEDULER_CONFIG)
            scheduler_config["doctor_status_file"] = str(snapshot_path)
            scheduler_config["doctor_history_file"] = str(history_path)
            payload = {
                "overall": "warn",
                "counts": {"ok": 1, "warn": 1, "fail": 0},
                "checks": [{"name": "task", "level": "warn", "detail": "needs review"}],
                "status": {"tasks": [{"task_name": "\\Web_Agent_Send_1200", "available": False, "error": "query failed"}]},
            }

            with patch("scheduler_runner.build_doctor_payload", return_value=payload):
                with patch("scheduler_runner.load_runtime_config", return_value=({}, scheduler_config)):
                    with patch("scheduler_runner.send_doctor_alert_email", return_value=False):
                        with patch("scheduler_runner.run_task_self_heal", return_value={"attempted": True, "success": False, "message": "boom"}):
                            with patch("sys.stdout.write"):
                                exit_code = print_doctor_report(
                                    as_json=False,
                                    self_heal=True,
                                    dry_run=False,
                                    persist_history=False,
                                )

            self.assertEqual(exit_code, 1)

    def test_print_doctor_report_default_mode_does_not_write_history(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            snapshot_path = Path(temp_dir) / "doctor_latest.json"
            history_path = Path(temp_dir) / "doctor_history.json"
            scheduler_config = dict(DEFAULT_SCHEDULER_CONFIG)
            scheduler_config["doctor_status_file"] = str(snapshot_path)
            scheduler_config["doctor_history_file"] = str(history_path)
            payload = {
                "overall": "warn",
                "counts": {"ok": 1, "warn": 1, "fail": 0},
                "checks": [{"name": "task", "level": "warn", "detail": "needs review"}],
            }

            with patch("scheduler_runner.build_doctor_payload", return_value=payload):
                with patch("scheduler_runner.load_runtime_config", return_value=({}, scheduler_config)):
                    with patch("scheduler_runner.send_doctor_alert_email", return_value=True):
                        with patch("sys.stdout.write"):
                            exit_code = print_doctor_report(as_json=False, persist_history=False)

            self.assertEqual(exit_code, 0)
            persisted = json.loads(snapshot_path.read_text(encoding="utf-8-sig"))
            self.assertFalse(persisted["history_summary"]["recorded"])
            self.assertFalse(history_path.exists())

    def test_main_releases_lock_when_idempotency_skips_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            log_dir = temp_root / "logs"
            status_path = log_dir / "last_run.json"
            lock_path = log_dir / "scheduler.lock"
            log_dir.mkdir(parents=True, exist_ok=True)

            write_json(
                status_path,
                {
                    "success": True,
                    "delivery_status": "sent",
                    "finished_at": datetime.now().isoformat(timespec="seconds"),
                    "html_report_path": "archive/report_recent.html",
                    "markdown_report_path": "archive/report_recent.md",
                },
            )

            scheduler_config = dict(DEFAULT_SCHEDULER_CONFIG)
            scheduler_config.update(
                {
                    "log_dir": str(log_dir),
                    "status_file": str(status_path),
                    "lock_file": str(lock_path),
                    "log_retention_days": 30,
                    "idempotency_window_minutes": 90,
                    "task_names": [],
                }
            )

            with patch("scheduler_runner.load_runtime_config", return_value=({}, scheduler_config)):
                exit_code = scheduler_main()

            self.assertEqual(exit_code, 0)
            self.assertFalse(lock_path.exists())
            latest = json.loads(status_path.read_text(encoding="utf-8-sig"))
            self.assertEqual(latest["status"], "skipped_recent_success")

    def test_validate_run_uses_separate_status_file_and_bypasses_idempotency(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            log_dir = temp_root / "logs"
            status_path = log_dir / "last_run.json"
            validation_status_path = log_dir / "last_validation_run.json"
            lock_path = log_dir / "scheduler.lock"
            log_dir.mkdir(parents=True, exist_ok=True)

            write_json(
                status_path,
                {
                    "success": True,
                    "delivery_status": "sent",
                    "finished_at": datetime.now().isoformat(timespec="seconds"),
                    "html_report_path": "archive/report_recent.html",
                },
            )

            scheduler_config = dict(DEFAULT_SCHEDULER_CONFIG)
            scheduler_config.update(
                {
                    "log_dir": str(log_dir),
                    "status_file": str(status_path),
                    "validation_status_file": str(validation_status_path),
                    "lock_file": str(lock_path),
                    "log_retention_days": 30,
                    "idempotency_window_minutes": 90,
                    "task_names": [],
                }
            )

            attempt_result = {
                "success": True,
                "status": "dry_run",
                "retryable": False,
                "delivery_status": "dry_run",
                "started_at": "2026-04-14T10:00:00",
                "finished_at": "2026-04-14T10:05:00",
                "html_report_path": "archive/report_validation.html",
                "markdown_report_path": "archive/report_validation.md",
            }

            with patch("scheduler_runner.load_runtime_config", return_value=({}, scheduler_config)):
                with patch("scheduler_runner.run_main_once", return_value=dict(attempt_result)) as mocked_run:
                    exit_code = scheduler_main(validate_run=True)

            self.assertEqual(exit_code, 0)
            self.assertFalse(lock_path.exists())
            mocked_run.assert_called_once_with(900, email_mode="dry-run", run_profile="validation_fast")
            self.assertTrue(validation_status_path.exists())
            latest_validation = json.loads(validation_status_path.read_text(encoding="utf-8-sig"))
            self.assertEqual(latest_validation["status"], "dry_run")
            self.assertEqual(latest_validation["run_mode"], "validation")
            latest_production = json.loads(status_path.read_text(encoding="utf-8-sig"))
            self.assertEqual(latest_production["delivery_status"], "sent")
            self.assertEqual(latest_production["html_report_path"], "archive/report_recent.html")


if __name__ == "__main__":
    unittest.main()
