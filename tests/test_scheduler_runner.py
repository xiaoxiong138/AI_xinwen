import json
import io
import os
import shutil
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from scheduler_runner import (
    DEFAULT_SCHEDULER_CONFIG,
    RESULT_MARKER,
    EMAIL_COMMITTED_MARKER,
    acquire_run_lock,
    acquire_send_slot,
    archive_old_logs,
    build_doctor_payload,
    build_doctor_alert_html,
    build_email_arrival_check,
    build_legacy_cleanup_payload,
    build_legacy_cleanup_text,
    build_monitored_task_names,
    build_product_diagnostics,
    build_repair_plan,
    build_repair_plan_text,
    build_scheduler_status_payload,
    build_scheduler_launcher_status,
    load_application_main,
    build_send_calendar_payload,
    build_send_calendar_text,
    build_task_repair_commands,
    build_v8_production_acceptance,
    build_paper_freshness_production_acceptance,
    _parse_status_datetime,
    _paper_reappearance_is_supported,
    _paper_snapshots_match,
    get_report_model_path_backfill_candidates,
    normalize_model_path_breakdown,
    fix_report_bad_titles,
    format_arxiv_retry_paths,
    scan_report_quality,
    suggest_title_from_snapshot,
    suggest_title_from_snapshot_with_source,
    build_status_snapshot,
    build_status_text,
    build_last_success_snapshot,
    build_send_slot_id,
    build_task_backups_payload,
    build_task_backups_text,
    build_task_restore_payload,
    build_task_restore_text,
    build_failure_email_html,
    classify_quality_warnings,
    cleanup_old_logs,
    cleanup_task_backups,
    cleanup_validation_reports,
    collect_task_self_heal_candidates,
    describe_task_result,
    export_scheduled_task_xml,
    is_interactive_task,
    parse_worker_result_line,
    parse_email_committed_line,
    _build_committed_email_result,
    parse_schtasks_list_output,
    print_doctor_report,
    print_legacy_cleanup_report,
    print_repair_plan,
    print_send_calendar_report,
    print_task_backups_report,
    print_task_restore_report,
    release_run_lock,
    refresh_report_quality_after_run,
    recover_committed_send_slot,
    resolve_send_slot,
    finalize_send_slot,
    run_task_self_heal,
    should_send_doctor_alert,
    should_skip_for_idempotency,
    main as scheduler_main,
    update_doctor_history,
    write_send_calendar,
    write_doctor_snapshot,
    write_json,
)
from main import build_source_health_summary
from src.database import Database
from src.notifier import resolve_imap_server


class SchedulerRunnerTests(unittest.TestCase):
    def test_last_success_snapshot_preserves_fresh_and_reappeared_paper_counts(self):
        snapshot = build_last_success_snapshot({
            "paper_count": 22,
            "fresh_paper_count": 20,
            "reappeared_paper_with_update_count": 2,
        })

        self.assertEqual(snapshot["paper_count"], 22)
        self.assertEqual(snapshot["fresh_paper_count"], 20)
        self.assertEqual(snapshot["reappeared_paper_with_update_count"], 2)

    def test_arxiv_retry_path_text_names_failed_and_recovered_show_counts(self):
        text = format_arxiv_retry_paths([{
            "source": "ArxivCollector[World Model]",
            "attempted_show_counts": [500, 100, 50],
            "successful_show_count": 50,
            "result": "success",
        }])

        self.assertIn("ArxivCollector[World Model]:500->100->50", text)
        self.assertIn("recovered at 50", text)

    def test_status_datetime_normalizes_aware_values_to_local_naive_time(self):
        local_value = _parse_status_datetime("2026-08-12T21:05:00")
        aware_value = _parse_status_datetime("2026-08-12T13:05:00Z")
        expected_aware_value = datetime.fromisoformat("2026-08-12T13:05:00+00:00").astimezone().replace(tzinfo=None)

        self.assertIsNotNone(local_value)
        self.assertIsNotNone(aware_value)
        self.assertIsNone(local_value.tzinfo)
        self.assertIsNone(aware_value.tzinfo)
        self.assertEqual(aware_value, expected_aware_value)

    def test_scheduled_task_setup_uses_hidden_synchronous_launcher(self):
        root = Path(__file__).resolve().parents[1]
        setup_text = (root / "setup_scheduled_tasks.ps1").read_text(encoding="utf-8-sig")
        doctor_setup_text = (root / "setup_doctor_task.ps1").read_text(encoding="utf-8-sig")
        preflight_setup_text = (root / "setup_preflight_task.ps1").read_text(encoding="utf-8-sig")
        launcher_text = (root / "run_scheduler_hidden.ps1").read_text(encoding="utf-8-sig")

        self.assertIn('WindowStyle Hidden', setup_text)
        self.assertIn('run_scheduler_hidden.ps1', setup_text)
        self.assertNotIn('$taskCommand = \'"\' + $python', setup_text)
        self.assertIn('IdleSettings.StopOnIdleEnd = $false', setup_text)
        self.assertIn('$WorkingDirectory = $PSScriptRoot', launcher_text)
        self.assertIn('Set-Location -LiteralPath $WorkingDirectory', launcher_text)
        self.assertIn('& $PythonExe $RunnerPath @RunnerArguments', launcher_text)
        self.assertIn('$runnerExitCode = [int]$LASTEXITCODE', launcher_text)
        self.assertIn('exit $runnerExitCode', launcher_text)
        self.assertIn('scheduler_launcher_latest.json', launcher_text)
        self.assertIn('Write-LauncherStatus -Status "running"', launcher_text)
        self.assertIn('Move-Item -LiteralPath $temporaryStatusPath', launcher_text)
        self.assertIn('at least 8 MB is required', launcher_text)
        for auxiliary_text, expected_args in (
            (doctor_setup_text, '--doctor --record'),
            (preflight_setup_text, '--doctor --record'),
        ):
            self.assertIn('run_scheduler_hidden.ps1', auxiliary_text)
            self.assertIn(expected_args, auxiliary_text)
            self.assertIn('StartWhenAvailable = $true', auxiliary_text)
            self.assertIn('RestartCount = 3', auxiliary_text)
            self.assertIn('IdleSettings.StopOnIdleEnd = $false', auxiliary_text)
        self.assertNotIn('--self-heal', preflight_setup_text)

    def test_scheduler_launcher_status_marks_dead_running_process_interrupted(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir)
            (log_dir / "scheduler_launcher_latest.json").write_text(
                json.dumps({"status": "running", "pid": 999999, "exit_code": 0}),
                encoding="utf-8",
            )

            status = build_scheduler_launcher_status(log_dir)

        self.assertTrue(status["available"])
        self.assertEqual(status["status"], "interrupted")
        self.assertFalse(status["pid_running"])

    def test_application_main_is_loaded_from_explicit_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            main_path = Path(temp_dir) / "main.py"
            main_path.write_text("def main():\n    return {'success': True, 'source': __file__}\n", encoding="utf-8")

            entrypoint = load_application_main(main_path)
            result = entrypoint()

        self.assertTrue(result["success"])
        self.assertEqual(Path(result["source"]).resolve(), main_path.resolve())

    def test_application_main_requires_callable_entrypoint(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            main_path = Path(temp_dir) / "main.py"
            main_path.write_text("main = 'not callable'\n", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "no callable main"):
                load_application_main(main_path)

    def test_production_application_main_uses_spawn_importable_module_name(self):
        entrypoint = load_application_main()

        self.assertEqual(entrypoint.__module__, "main")
        self.assertEqual(entrypoint.__globals__["_collector_worker"].__module__, "main")

    def test_production_application_collector_worker_survives_windows_spawn(self):
        entrypoint = load_application_main()
        collector = entrypoint.__globals__["RSSCollector"](feeds=[])

        items, error = entrypoint.__globals__["run_collector_with_timeout"](
            collector,
            timeout_seconds=20,
            network_timeout=10,
        )

        self.assertEqual(items, [])
        self.assertIsNone(error)

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

    def test_acquire_run_lock_trusts_fresh_lock_even_when_pid_probe_fails(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            lock_path = Path(temp_dir) / "scheduler.lock"
            write_json(
                lock_path,
                {
                    "pid": 999999,
                    "acquired_at": datetime.now().isoformat(timespec="seconds"),
                    "hostname": "fresh-host",
                },
            )

            with patch("scheduler_runner.is_pid_running", return_value=False):
                acquired, lock_info = acquire_run_lock(lock_path, stale_lock_seconds=3600)

            self.assertFalse(acquired)
            self.assertEqual(lock_info["pid"], 999999)
            self.assertFalse(lock_info["pid_running"])
            self.assertTrue(lock_path.exists())

    def test_acquire_run_lock_reclaims_dead_pid_after_short_grace(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            lock_path = Path(temp_dir) / "scheduler.lock"
            write_json(
                lock_path,
                {
                    "pid": 999999,
                    "acquired_at": (datetime.now() - timedelta(minutes=2)).isoformat(timespec="seconds"),
                    "hostname": "dead-host",
                },
            )

            with patch("scheduler_runner.is_pid_running", return_value=False):
                acquired, lock_info = acquire_run_lock(lock_path, stale_lock_seconds=3600)

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

    def test_email_commit_marker_disables_retry_after_followup_failure(self):
        payload = {
            "run_id": "20260812_210000",
            "report_id": "report-1",
            "email_subject": "AI Daily",
            "html_report_path": "archive/report.html",
        }
        parsed = parse_email_committed_line(f"{EMAIL_COMMITTED_MARKER}{json.dumps(payload)}")
        result = _build_committed_email_result(
            parsed,
            datetime(2026, 8, 12, 21, 0),
            status="sent_followup_timeout",
            error="IMAP follow-up timed out",
        )

        self.assertEqual(parsed, payload)
        self.assertTrue(result["success"])
        self.assertFalse(result["retryable"])
        self.assertEqual(result["delivery_status"], "sent")

    def test_run_main_once_passes_send_slot_id_to_worker(self):
        captured_env = {}

        class CompletedProcess:
            returncode = 0
            stdout = io.StringIO("")

            def poll(self):
                return 0

        def fake_popen(*args, **kwargs):
            captured_env.update(kwargs["env"])
            return CompletedProcess()

        with patch("scheduler_runner.subprocess.Popen", side_effect=fake_popen):
            result = __import__("scheduler_runner").run_main_once(60, send_slot_id="20260812_2100")

        self.assertEqual(captured_env["WEB_AGENT_SEND_SLOT_ID"], "20260812_2100")
        self.assertFalse(result["success"])

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

    def test_archive_old_logs_moves_expired_files_before_deletion(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir)
            archive_dir = log_dir / "archive"
            old_log = log_dir / "old.log"
            fresh_log = log_dir / "fresh.log"
            old_log.write_text("old", encoding="utf-8")
            fresh_log.write_text("fresh", encoding="utf-8")

            old_ts = (datetime.now() - timedelta(days=20)).timestamp()
            os.utime(old_log, (old_ts, old_ts))

            archived = archive_old_logs(log_dir, archive_dir, archive_after_days=14)

            self.assertEqual(archived, ["archive/old.log"])
            self.assertFalse(old_log.exists())
            self.assertTrue((archive_dir / "old.log").exists())
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

    def test_cleanup_task_backups_limits_age_and_count(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            backup_dir = Path(temp_dir)
            paths = [
                backup_dir / "Web_Agent_Send_1200_20260506_120000.xml",
                backup_dir / "Web_Agent_Send_2100_20260505_210000.xml",
                backup_dir / "Web_Agent_Send_1200_20260401_120000.xml",
                backup_dir / "Web_Agent_Send_2100_20260401_210000.xml",
            ]
            for index, path in enumerate(paths):
                path.write_text("<Task></Task>", encoding="utf-8")
                modified_ts = (datetime.now() - timedelta(days=index)).timestamp()
                os.utime(path, (modified_ts, modified_ts))
            old_ts = (datetime.now() - timedelta(days=45)).timestamp()
            os.utime(paths[2], (old_ts, old_ts))
            os.utime(paths[3], (old_ts - 10, old_ts - 10))
            ignored_text = backup_dir / "notes.txt"
            ignored_text.write_text("keep", encoding="utf-8")

            removed = cleanup_task_backups(backup_dir, retention_days=30, keep_count=2)

            self.assertEqual(sorted(removed), sorted([paths[2].name, paths[3].name]))
            self.assertTrue(paths[0].exists())
            self.assertTrue(paths[1].exists())
            self.assertFalse(paths[2].exists())
            self.assertFalse(paths[3].exists())
            self.assertTrue(ignored_text.exists())

    def test_cleanup_task_backups_honors_count_cap_for_fresh_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            backup_dir = Path(temp_dir)
            paths = [
                backup_dir / "Web_Agent_Send_1200_20260506_120000.xml",
                backup_dir / "Web_Agent_Send_2100_20260505_210000.xml",
                backup_dir / "Web_Agent_Send_1200_20260504_120000.xml",
            ]
            for index, path in enumerate(paths):
                path.write_text("<Task></Task>", encoding="utf-8")
                modified_ts = (datetime.now() - timedelta(minutes=index)).timestamp()
                os.utime(path, (modified_ts, modified_ts))

            removed = cleanup_task_backups(backup_dir, retention_days=30, keep_count=2)

            self.assertEqual(removed, [paths[2].name])
            self.assertTrue(paths[0].exists())
            self.assertTrue(paths[1].exists())
            self.assertFalse(paths[2].exists())

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

    def test_should_skip_for_idempotency_prefers_last_success_snapshot(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            status_path = Path(temp_dir) / "last_run.json"
            success_path = Path(temp_dir) / "last_success.json"
            write_json(
                status_path,
                {
                    "success": False,
                    "status": "notification_failed",
                    "delivery_status": "failed",
                    "finished_at": datetime.now().isoformat(timespec="seconds"),
                },
            )
            write_json(
                success_path,
                {
                    "finished_at": (datetime.now() - timedelta(minutes=20)).isoformat(timespec="seconds"),
                    "html_report_path": "archive/report_success.html",
                },
            )

            should_skip, last_status = should_skip_for_idempotency(
                status_path,
                window_minutes=240,
                last_success_path=success_path,
            )

            self.assertTrue(should_skip)
            self.assertEqual(last_status["delivery_status"], "sent")
            self.assertEqual(last_status["html_report_path"], "archive/report_success.html")

    def test_build_send_slot_id_uses_noon_and_evening_windows(self):
        self.assertEqual(build_send_slot_id(datetime(2026, 4, 28, 13, 0, 0)), "20260428_1300")
        self.assertEqual(build_send_slot_id(datetime(2026, 4, 28, 21, 0, 0)), "20260428_2100")
        self.assertEqual(build_send_slot_id(datetime(2026, 4, 28, 18, 0, 0)), "")

    def test_resolve_send_slot_reports_outside_window(self):
        resolved = resolve_send_slot(datetime(2026, 4, 28, 18, 0, 0))

        self.assertFalse(resolved["allowed"])
        self.assertEqual(resolved["reason"], "outside_send_window")

    def test_resolve_send_slot_allows_previous_day_window_after_midnight(self):
        scheduler_config = dict(DEFAULT_SCHEDULER_CONFIG)
        scheduler_config["send_window_after_minutes"] = 240

        resolved = resolve_send_slot(datetime(2026, 4, 29, 0, 30, 0), scheduler_config)

        self.assertTrue(resolved["allowed"])
        self.assertEqual(resolved["slot_id"], "20260428_2100")
        self.assertEqual(resolved["scheduled_at"], "2026-04-28T21:00:00")

    def test_acquire_send_slot_prevents_duplicate_claims_and_persists_success(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            slot_dir = Path(temp_dir) / "send_slots"

            acquired, slot_info = acquire_send_slot(slot_dir, "20260428_2100", stale_seconds=3600)
            duplicate_acquired, duplicate_info = acquire_send_slot(slot_dir, "20260428_2100", stale_seconds=3600)

            self.assertTrue(acquired)
            self.assertFalse(duplicate_acquired)
            self.assertEqual(duplicate_info["status"], "in_progress")

            finalize_send_slot(
                slot_info,
                {
                    "success": True,
                    "delivery_status": "sent",
                    "finished_at": "2026-04-28T21:08:41",
                    "run_id": "20260428_210002",
                    "html_report_path": "archive/report.html",
                },
            )
            persisted = json.loads(Path(slot_info["slot_path"]).read_text(encoding="utf-8-sig"))
            self.assertEqual(persisted["status"], "sent")
            self.assertEqual(persisted["run_id"], "20260428_210002")

    def test_acquire_send_slot_reclaims_dead_pid_after_grace_period(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            slot_dir = Path(temp_dir)
            slot_path = slot_dir / "20260716_1300.json"
            slot_path.write_text(
                json.dumps(
                    {
                        "slot_id": "20260716_1300",
                        "pid": 999999,
                        "status": "in_progress",
                        "acquired_at": (datetime.now() - timedelta(minutes=2)).isoformat(timespec="seconds"),
                    }
                ),
                encoding="utf-8",
            )

            acquired, slot_info = acquire_send_slot(slot_dir, "20260716_1300", stale_seconds=3600)

            self.assertTrue(acquired)
            self.assertEqual(slot_info["status"], "in_progress")
            self.assertNotEqual(slot_info["pid"], 999999)

    def test_recover_committed_send_slot_treats_durable_smtp_commit_as_success(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            slot_path = Path(temp_dir) / "20260812_2100.json"
            write_json(slot_path, {
                "slot_id": "20260812_2100",
                "status": "sent",
                "run_id": "run-1",
                "html_report_path": "archive/report.html",
                "finished_at": "2026-08-12T21:08:00",
            })

            recovered = recover_committed_send_slot(
                {"slot_path": slot_path.as_posix()},
                started_at=datetime(2026, 8, 12, 21, 0),
            )

        self.assertTrue(recovered["success"])
        self.assertEqual(recovered["status"], "sent_commit_recovered")
        self.assertEqual(recovered["delivery_status"], "sent")

    def test_send_calendar_records_noon_and_evening_slots(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            slot_dir = temp_root / "send_slots"
            calendar_dir = temp_root / "logs"
            scheduler_config = dict(DEFAULT_SCHEDULER_CONFIG)
            scheduler_config.update(
                {
                    "send_slot_dir": str(slot_dir),
                    "send_calendar_dir": str(calendar_dir),
                    "last_success_file": str(temp_root / "last_success.json"),
                }
            )
            slot_dir.mkdir(parents=True, exist_ok=True)
            write_json(
                slot_dir / "20260428_1300.json",
                {
                    "slot_id": "20260428_1300",
                    "status": "sent",
                    "run_id": "20260428_130000",
                    "finished_at": "2026-04-28T13:05:00",
                    "html_report_path": "archive/report_noon.html",
                },
            )

            payload = build_send_calendar_payload(
                scheduler_config,
                target_datetime=datetime(2026, 4, 28, 21, 3, 0),
                last_run={
                    "run_mode": "scheduler",
                    "status": "skipped_duplicate_slot",
                    "finished_at": "2026-04-28T21:03:00",
                    "send_slot": {"slot_id": "20260428_2100"},
                },
            )

            self.assertEqual(payload["date"], "2026-04-28")
            self.assertEqual([slot["slot_id"] for slot in payload["slots"]], ["20260428_1300", "20260428_2100"])
            self.assertEqual(payload["slots"][0]["status"], "sent")
            self.assertEqual(payload["slots"][0]["health_status"], "sent")
            self.assertEqual(payload["slots"][1]["last_run_status"], "skipped_duplicate_slot")
            self.assertEqual(payload["slots"][1]["health_status"], "pending")

            calendar_path = write_send_calendar(
                scheduler_config,
                {
                    "run_mode": "scheduler",
                    "status": "success",
                    "delivery_status": "sent",
                    "finished_at": "2026-04-28T12:05:00",
                },
            )
            self.assertEqual(calendar_path, calendar_dir / "send_calendar_20260428.json")
            self.assertTrue(calendar_path.exists())

    def test_build_send_calendar_text_is_readable(self):
        text = build_send_calendar_text(
            {
                "date": "2026-04-28",
                "generated_at": "2026-04-28T21:05:00",
                "slots": [
                    {
                    "slot_id": "20260428_1300",
                    "time": "13:00",
                    "status": "sent",
                    "finished_at": "2026-04-28T13:05:00",
                    "html_report_path": "archive/report_noon.html",
                    },
                    {
                        "slot_id": "20260428_2100",
                        "time": "21:00",
                        "status": "missing",
                        "health_status": "missed",
                    },
                ],
                "last_success": {
                    "finished_at": "2026-04-28T13:05:00",
                    "html_report_path": "archive/report_noon.html",
                },
            }
        )

        self.assertIn("Send calendar: 2026-04-28", text)
        self.assertIn("13:00 20260428_1300: sent", text)
        self.assertIn("21:00 20260428_2100: missed", text)
        self.assertIn("Last success: 2026-04-28T13:05:00", text)

    def test_send_calendar_marks_missing_slots_by_time_window(self):
        scheduler_config = dict(DEFAULT_SCHEDULER_CONFIG)
        with tempfile.TemporaryDirectory() as temp_dir:
            scheduler_config["send_slot_dir"] = str(Path(temp_dir) / "send_slots")
            payload = build_send_calendar_payload(
                scheduler_config,
                target_datetime=datetime(2026, 4, 28, 16, 1, 0),
            )

        self.assertEqual(payload["slots"][0]["health_status"], "missed")
        self.assertEqual(payload["slots"][1]["health_status"], "upcoming")

    def test_send_calendar_marks_abandoned_in_progress_slot_as_stale(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            slot_dir = Path(temp_dir) / "send_slots"
            slot_dir.mkdir(parents=True)
            write_json(
                slot_dir / "20260428_1300.json",
                {
                    "slot_id": "20260428_1300",
                    "status": "in_progress",
                    "acquired_at": "2026-04-28T13:00:00",
                    "pid": 1234,
                },
            )
            scheduler_config = dict(DEFAULT_SCHEDULER_CONFIG)
            scheduler_config.update(
                {
                    "send_slot_dir": str(slot_dir),
                    "send_slot_stale_seconds": 3600,
                }
            )

            payload = build_send_calendar_payload(
                scheduler_config,
                target_datetime=datetime(2026, 4, 28, 17, 0, 0),
            )

        self.assertEqual(payload["slots"][0]["status"], "stale")
        self.assertEqual(payload["slots"][0]["health_status"], "missed")

    def test_v8_production_acceptance_requires_three_clean_sent_reports(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            slot_dir = root / "send_slots"
            slot_dir.mkdir()
            db_path = root / "reports.db"
            db = Database(str(db_path))
            for index in range(3):
                run_id = f"2026042{index + 6}_130000"
                report_id = f"report_{index}"
                diagnostics = {
                    "report_design_version": "v8-editorial-reader",
                    "quality_gate": {
                        "final_html_quality_status": "passed",
                        "final_html_bad_title_count": 0,
                        "untranslated_fact_count": 0,
                        "exact_duplicate_sentence_count": 0,
                        "paper_mechanism_missing_count": 0,
                        "paper_result_context_missing_count": 0,
                        "appendix_body_overlap_count": 0,
                        "truncated_focus_text_count": 0,
                        "max_opening_repeat_count": 1,
                        "focus_source_concentration": 0.2,
                    },
                }
                db.record_report_run(
                    report_id=report_id,
                    run_id=run_id,
                    slot_id=f"2026042{index + 6}_1300",
                    html_report_path=f"archive/{report_id}.html",
                    quality_status="passed",
                    quality_diagnostics=diagnostics,
                )
                write_json(
                    slot_dir / f"2026042{index + 6}_1300.json",
                    {
                        "slot_id": f"2026042{index + 6}_1300",
                        "status": "sent",
                        "run_id": run_id,
                        "finished_at": f"2026-04-2{index + 6}T13:05:00",
                        "html_report_path": f"archive/{report_id}.html",
                    },
                )

            result = build_v8_production_acceptance(
                {"send_slot_dir": str(slot_dir)},
                db_path=str(db_path),
            )

        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["verified_count"], 3)
        self.assertEqual(result["passed_count"], 3)

    def test_paper_freshness_acceptance_requires_three_clean_production_days(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            slot_dir = root / "send_slots"
            slot_dir.mkdir()
            db_path = root / "reports.db"
            db = Database(str(db_path))
            for index in range(3):
                run_id = f"2026080{index + 1}_130000"
                report_id = f"paper-report-{index}"
                db.record_report_run(
                    report_id,
                    run_id,
                    slot_id=f"2026080{index + 1}_1300",
                    quality_status="passed",
                    quality_diagnostics={
                        "report_design_version": "v10-learning-digest",
                        "paper_freshness": {"paper_freshness_status": "passed"},
                    },
                    delivery_status="sent",
                )
                db.record_report_items(report_id, [
                    {
                        "id": index * 20 + paper_index,
                        "content_type": "paper",
                        "url": f"https://arxiv.org/abs/2608.{index}{paper_index:04d}",
                    }
                    for paper_index in range(20)
                ])
                write_json(slot_dir / f"2026080{index + 1}_1300.json", {
                    "slot_id": f"2026080{index + 1}_1300",
                    "status": "sent",
                    "run_id": run_id,
                    "finished_at": f"2026-08-0{index + 1}T13:05:00",
                })

            result = build_paper_freshness_production_acceptance(
                {"send_slot_dir": str(slot_dir)},
                db_path=str(db_path),
            )

        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["verified_days"], 3)
        self.assertEqual(result["passed_report_count"], 3)

    def test_freshness_acceptance_checks_both_reports_but_counts_one_day(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            slot_dir = root / "send_slots"
            slot_dir.mkdir()
            db_path = root / "reports.db"
            db = Database(str(db_path))
            first_urls = [f"https://arxiv.org/abs/2608.1{index:04d}" for index in range(20)]
            second_urls = first_urls[:2] + [f"https://arxiv.org/abs/2608.2{index:04d}" for index in range(18)]
            for slot, urls in (("1300", first_urls), ("2100", second_urls)):
                report_id = f"same-day-{slot}"
                run_id = f"run-{slot}"
                db.record_report_run(
                    report_id,
                    run_id,
                    slot_id=f"20260812_{slot}",
                    quality_status="passed",
                    quality_diagnostics={
                        "report_design_version": "v10-learning-digest",
                        "paper_freshness": {"paper_freshness_status": "passed"},
                    },
                    delivery_status="sent",
                )
                db.record_report_items(report_id, [
                    {"id": index, "content_type": "paper", "url": url}
                    for index, url in enumerate(urls)
                ])
                write_json(slot_dir / f"20260812_{slot}.json", {
                    "slot_id": f"20260812_{slot}",
                    "status": "sent",
                    "run_id": run_id,
                    "finished_at": f"2026-08-12T{slot[:2]}:05:00",
                })

            result = build_paper_freshness_production_acceptance(
                {"send_slot_dir": str(slot_dir)},
                db_path=str(db_path),
            )

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["verified_days"], 1)
        self.assertEqual(result["verified_report_count"], 2)
        failed_row = next(row for row in result["reports"] if row["adjacent_report_paper_overlap_rate"] == 0.1)
        self.assertEqual(failed_row["adjacent_report_paper_overlap_count"], 2)
        self.assertEqual(len(failed_row["adjacent_report_paper_overlap_keys"]), 2)

    def test_freshness_acceptance_compares_cross_midnight_reports_in_send_order(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            slot_dir = root / "send_slots"
            slot_dir.mkdir()
            db_path = root / "reports.db"
            db = Database(str(db_path))
            report_specs = (
                ("z-night-before", "20260812_2100", "2026-08-12T21:05:00", "2608.1"),
                ("a-next-noon", "20260813_1300", "2026-08-13T13:05:00", "2608.2"),
                ("m-next-night", "20260813_2100", "2026-08-13T21:05:00", "2608.3"),
            )
            previous_urls = []
            for report_index, (report_id, slot_id, finished_at, prefix) in enumerate(report_specs):
                urls = [f"https://arxiv.org/abs/{prefix}{index:04d}" for index in range(20)]
                if report_index == 1:
                    urls[:2] = previous_urls[:2]
                previous_urls = urls
                run_id = f"run-{report_id}"
                db.record_report_run(
                    report_id,
                    run_id,
                    slot_id=slot_id,
                    quality_status="passed",
                    quality_diagnostics={
                        "report_design_version": "v10-learning-digest",
                        "paper_freshness": {"paper_freshness_status": "passed"},
                    },
                    delivery_status="sent",
                )
                db.record_report_items(report_id, [
                    {"id": index, "content_type": "paper", "url": url}
                    for index, url in enumerate(urls)
                ])
                write_json(slot_dir / f"{slot_id}.json", {
                    "slot_id": slot_id,
                    "status": "sent",
                    "run_id": run_id,
                    "finished_at": finished_at,
                })

            result = build_paper_freshness_production_acceptance(
                {"send_slot_dir": str(slot_dir)},
                db_path=str(db_path),
            )

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["verified_days"], 2)
        self.assertEqual(result["verified_report_count"], 3)
        self.assertEqual(result["reports"][0]["report_id"], "m-next-night")
        noon_row = next(row for row in result["reports"] if row["report_id"] == "a-next-noon")
        self.assertEqual(noon_row["adjacent_report_paper_overlap_rate"], 0.1)
        self.assertIn("adjacent_overlap=0.100", noon_row["issues"])

    def test_freshness_acceptance_deduplicates_multiple_slots_for_same_report(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            slot_dir = root / "send_slots"
            slot_dir.mkdir()
            db_path = root / "reports.db"
            db = Database(str(db_path))
            db.record_report_run(
                "single-report",
                "single-run",
                quality_status="passed",
                quality_diagnostics={
                    "report_design_version": "v10-learning-digest",
                    "paper_freshness": {"paper_freshness_status": "passed"},
                },
                delivery_status="sent",
            )
            db.record_report_items("single-report", [
                {
                    "id": index,
                    "content_type": "paper",
                    "url": f"https://arxiv.org/abs/2608.{index:05d}",
                }
                for index in range(20)
            ])
            for slot, finished_at in (("1300", "2026-08-12T13:05:00"), ("2100", "2026-08-12T21:05:00")):
                write_json(slot_dir / f"20260812_{slot}.json", {
                    "slot_id": f"20260812_{slot}",
                    "status": "sent",
                    "run_id": "single-run",
                    "finished_at": finished_at,
                })

            result = build_paper_freshness_production_acceptance(
                {"send_slot_dir": str(slot_dir)},
                db_path=str(db_path),
            )

        self.assertEqual(result["status"], "pending")
        self.assertEqual(result["verified_days"], 1)
        self.assertEqual(result["verified_report_count"], 1)

    def test_freshness_acceptance_matches_arxiv_and_doi_aliases(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            slot_dir = root / "send_slots"
            slot_dir.mkdir()
            db_path = root / "reports.db"
            db = Database(str(db_path))
            report_items = (
                {
                    "id": 1,
                    "content_type": "paper",
                    "canonical_url": "https://arxiv.org/abs/2608.01234",
                    "url": "https://doi.org/10.1234/robot.45",
                },
                {
                    "id": 2,
                    "content_type": "paper",
                    "canonical_url": "https://publisher.example/robot-45",
                    "url": "https://doi.org/10.1234/robot.45",
                },
            )
            for day, item in enumerate(report_items, start=1):
                report_id = f"alias-report-{day}"
                run_id = f"alias-run-{day}"
                slot_id = f"2026080{day}_1300"
                db.record_report_run(
                    report_id,
                    run_id,
                    slot_id=slot_id,
                    quality_status="passed",
                    quality_diagnostics={
                        "report_design_version": "v10-learning-digest",
                        "paper_freshness": {
                            "paper_freshness_status": "passed",
                            "effective_min_visible_paper_count": 1,
                        },
                    },
                    delivery_status="sent",
                )
                db.record_report_items(report_id, [item])
                write_json(slot_dir / f"{slot_id}.json", {
                    "slot_id": slot_id,
                    "status": "sent",
                    "run_id": run_id,
                    "finished_at": f"2026-08-0{day}T13:05:00",
                })

            result = build_paper_freshness_production_acceptance(
                {"send_slot_dir": str(slot_dir)},
                db_path=str(db_path),
            )

        self.assertEqual(result["status"], "failed")
        latest = next(row for row in result["reports"] if row["report_id"] == "alias-report-2")
        self.assertEqual(latest["adjacent_report_paper_overlap_rate"], 1.0)
        self.assertEqual(latest["unjustified_7d_repeat_count"], 1)

    def test_freshness_acceptance_does_not_merge_same_title_with_distinct_arxiv_ids(self):
        title = "Predict-then-act world models for dynamic robot manipulation"

        self.assertFalse(
            _paper_snapshots_match(
                {"title": title, "url": "https://arxiv.org/abs/2608.00001"},
                {"title": title, "url": "https://arxiv.org/abs/2608.00002"},
            )
        )

    def test_freshness_acceptance_requires_database_sent_confirmation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            slot_dir = root / "send_slots"
            slot_dir.mkdir()
            db_path = root / "reports.db"
            db = Database(str(db_path))
            db.record_report_run(
                "pending-report",
                "pending-run",
                slot_id="20260812_2100",
                quality_status="passed",
                quality_diagnostics={
                    "report_design_version": "v10-learning-digest",
                    "paper_freshness": {
                        "paper_freshness_status": "passed",
                        "effective_min_visible_paper_count": 1,
                    },
                },
                delivery_status="pending",
            )
            db.record_report_items("pending-report", [{
                "id": 1,
                "content_type": "paper",
                "url": "https://arxiv.org/abs/2608.00001",
            }])
            write_json(slot_dir / "20260812_2100.json", {
                "slot_id": "20260812_2100",
                "status": "sent",
                "run_id": "pending-run",
                "finished_at": "2026-08-12T21:05:00",
            })

            result = build_paper_freshness_production_acceptance(
                {"send_slot_dir": str(slot_dir)},
                db_path=str(db_path),
            )

        self.assertEqual(result["status"], "pending")
        self.assertEqual(result["verified_report_count"], 0)

    def test_freshness_acceptance_rejects_duplicate_inside_one_sent_report(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            slot_dir = root / "send_slots"
            slot_dir.mkdir()
            db_path = root / "reports.db"
            db = Database(str(db_path))
            db.record_report_run(
                "duplicate-report",
                "duplicate-run",
                slot_id="20260812_2100",
                quality_status="passed",
                quality_diagnostics={
                    "report_design_version": "v10-learning-digest",
                    "paper_freshness": {
                        "paper_freshness_status": "passed",
                        "effective_min_visible_paper_count": 1,
                    },
                },
                delivery_status="sent",
            )
            db.record_report_items("duplicate-report", [
                {"id": 1, "content_type": "paper", "url": "https://arxiv.org/abs/2608.00001v1"},
                {"id": 2, "content_type": "paper", "url": "https://arxiv.org/abs/2608.00001v2"},
            ])
            write_json(slot_dir / "20260812_2100.json", {
                "slot_id": "20260812_2100",
                "status": "sent",
                "run_id": "duplicate-run",
                "finished_at": "2026-08-12T21:05:00",
            })

            result = build_paper_freshness_production_acceptance(
                {"send_slot_dir": str(slot_dir)},
                db_path=str(db_path),
            )

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["reports"][0]["paper_within_report_duplicate_count"], 1)
        self.assertIn("within_report_duplicates=1", result["reports"][0]["issues"])

    def test_freshness_acceptance_rejects_zero_paper_report_even_with_zero_overlap(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            slot_dir = root / "send_slots"
            slot_dir.mkdir()
            db_path = root / "reports.db"
            db = Database(str(db_path))
            db.record_report_run(
                "empty-report",
                "empty-run",
                quality_status="passed",
                quality_diagnostics={
                    "report_design_version": "v10-learning-digest",
                    "paper_freshness": {
                        "paper_freshness_status": "passed",
                        "effective_min_visible_paper_count": 10,
                    },
                },
                delivery_status="sent",
            )
            write_json(slot_dir / "20260812_2100.json", {
                "slot_id": "20260812_2100",
                "status": "sent",
                "run_id": "empty-run",
                "finished_at": "2026-08-12T21:05:00",
            })

            result = build_paper_freshness_production_acceptance(
                {"send_slot_dir": str(slot_dir)},
                db_path=str(db_path),
            )

        self.assertEqual(result["status"], "failed")
        self.assertIn("fresh_paper_count_below_min=0/10", result["reports"][0]["issues"])

    def test_freshness_acceptance_does_not_count_reappeared_updates_toward_minimum(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            slot_dir = root / "send_slots"
            slot_dir.mkdir()
            db_path = root / "reports.db"
            db = Database(str(db_path))
            db.record_report_run(
                "updates-only-report",
                "updates-only-run",
                quality_status="passed",
                quality_diagnostics={
                    "report_design_version": "v10-learning-digest",
                    "paper_freshness": {
                        "paper_freshness_status": "passed",
                        "effective_min_visible_paper_count": 10,
                    },
                },
                delivery_status="sent",
            )
            db.record_report_items("updates-only-report", [
                {
                    "id": index,
                    "content_type": "paper",
                    "url": f"https://arxiv.org/abs/2608.{index:05d}",
                    "is_reappeared_update": True,
                }
                for index in range(1, 11)
            ])
            write_json(slot_dir / "20260812_2100.json", {
                "slot_id": "20260812_2100",
                "status": "sent",
                "run_id": "updates-only-run",
                "finished_at": "2026-08-12T21:05:00",
            })

            result = build_paper_freshness_production_acceptance(
                {"send_slot_dir": str(slot_dir)},
                db_path=str(db_path),
            )

        row = result["reports"][0]
        self.assertEqual(result["status"], "failed")
        self.assertEqual(row["paper_count"], 10)
        self.assertEqual(row["fresh_paper_count"], 0)
        self.assertEqual(row["reappeared_paper_with_update_count"], 10)
        self.assertIn("fresh_paper_count_below_min=0/10", row["issues"])

    def test_freshness_acceptance_marks_eligible_short_report_below_ideal_target(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            slot_dir = root / "send_slots"
            slot_dir.mkdir()
            db_path = root / "reports.db"
            db = Database(str(db_path))
            db.record_report_run(
                "short-fresh-report",
                "short-fresh-run",
                quality_status="passed",
                quality_diagnostics={
                    "report_design_version": "v10-learning-digest",
                    "paper_freshness": {
                        "paper_freshness_status": "passed",
                        "effective_min_visible_paper_count": 10,
                        "configured_min_visible_paper_count": 10,
                        "configured_target_min_paper_count": 18,
                        "configured_target_max_paper_count": 25,
                    },
                },
                delivery_status="sent",
            )
            db.record_report_items("short-fresh-report", [
                {
                    "id": index,
                    "content_type": "paper",
                    "url": f"https://arxiv.org/abs/2608.{index:05d}",
                }
                for index in range(12)
            ])
            write_json(slot_dir / "20260812_2100.json", {
                "slot_id": "20260812_2100",
                "status": "sent",
                "run_id": "short-fresh-run",
                "finished_at": "2026-08-12T21:05:00",
            })

            result = build_paper_freshness_production_acceptance(
                {"send_slot_dir": str(slot_dir)},
                db_path=str(db_path),
            )

        row = result["reports"][0]
        self.assertEqual(result["status"], "pending")
        self.assertTrue(row["passed"])
        self.assertEqual(row["minimum_paper_count"], 10)
        self.assertEqual(row["recommended_minimum_paper_count"], 18)
        self.assertTrue(row["paper_target_underfilled"])

    def test_freshness_acceptance_rejects_failed_embedded_freshness_gate(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            slot_dir = root / "send_slots"
            slot_dir.mkdir()
            db_path = root / "reports.db"
            db = Database(str(db_path))
            db.record_report_run(
                "failed-gate-report",
                "failed-gate-run",
                quality_status="passed",
                quality_diagnostics={
                    "report_design_version": "v10-learning-digest",
                    "paper_freshness": {"paper_freshness_status": "failed"},
                },
                delivery_status="sent",
            )
            db.record_report_items("failed-gate-report", [
                {
                    "id": index,
                    "content_type": "paper",
                    "url": f"https://arxiv.org/abs/2608.{index:05d}",
                }
                for index in range(20)
            ])
            write_json(slot_dir / "20260812_2100.json", {
                "slot_id": "20260812_2100",
                "status": "sent",
                "run_id": "failed-gate-run",
                "finished_at": "2026-08-12T21:05:00",
            })

            result = build_paper_freshness_production_acceptance(
                {"send_slot_dir": str(slot_dir)},
                db_path=str(db_path),
            )

        self.assertEqual(result["status"], "failed")
        self.assertIn("paper_freshness_status=failed", result["reports"][0]["issues"])

    def test_freshness_acceptance_rejects_failed_domain_quota_gate(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            slot_dir = root / "send_slots"
            slot_dir.mkdir()
            db_path = root / "reports.db"
            db = Database(str(db_path))
            db.record_report_run(
                "failed-domain-report",
                "failed-domain-run",
                quality_status="passed",
                quality_diagnostics={
                    "report_design_version": "v10-learning-digest",
                    "paper_freshness": {
                        "paper_freshness_status": "passed",
                        "paper_domain_quota_status": "failed",
                        "paper_domain_quota_exceeded": {
                            "world_model": {"count": 7, "max": 6},
                        },
                    },
                },
                delivery_status="sent",
            )
            db.record_report_items("failed-domain-report", [
                {
                    "id": index,
                    "content_type": "paper",
                    "url": f"https://arxiv.org/abs/2608.{index:05d}",
                }
                for index in range(20)
            ])
            write_json(slot_dir / "20260812_2100.json", {
                "slot_id": "20260812_2100",
                "status": "sent",
                "run_id": "failed-domain-run",
                "finished_at": "2026-08-12T21:05:00",
            })

            result = build_paper_freshness_production_acceptance(
                {"send_slot_dir": str(slot_dir)},
                db_path=str(db_path),
            )

        self.assertEqual(result["status"], "failed")
        self.assertIn("paper_domain_quota_status=failed", result["reports"][0]["issues"])

    def test_paper_freshness_acceptance_ignores_pre_upgrade_v10_reports(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            slot_dir = root / "send_slots"
            slot_dir.mkdir()
            db_path = root / "reports.db"
            db = Database(str(db_path))
            db.record_report_run(
                "legacy-v10",
                "legacy-run",
                slot_id="20260801_1300",
                quality_status="passed",
                quality_diagnostics={"report_design_version": "v10-learning-digest"},
                delivery_status="sent",
            )
            db.record_report_items("legacy-v10", [{
                "id": 1,
                "content_type": "paper",
                "url": "https://arxiv.org/abs/2608.00001",
            }])
            write_json(slot_dir / "20260801_1300.json", {
                "slot_id": "20260801_1300",
                "status": "sent",
                "run_id": "legacy-run",
                "finished_at": "2026-08-01T13:05:00",
            })

            result = build_paper_freshness_production_acceptance(
                {"send_slot_dir": str(slot_dir)},
                db_path=str(db_path),
            )

        self.assertEqual(result["status"], "pending")
        self.assertEqual(result["verified_days"], 0)

    def test_first_upgraded_report_still_checks_pre_upgrade_cooldown_history(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            slot_dir = root / "send_slots"
            slot_dir.mkdir()
            db_path = root / "reports.db"
            db = Database(str(db_path))
            for report_id, run_id, day, freshness_enabled in (
                ("legacy", "legacy-run", 1, False),
                ("upgraded", "upgraded-run", 2, True),
            ):
                diagnostics = {"report_design_version": "v10-learning-digest"}
                if freshness_enabled:
                    diagnostics["paper_freshness"] = {"paper_freshness_status": "passed"}
                db.record_report_run(
                    report_id,
                    run_id,
                    slot_id=f"2026080{day}_1300",
                    quality_status="passed",
                    quality_diagnostics=diagnostics,
                    delivery_status="sent",
                )
                db.record_report_items(report_id, [{
                    "id": day,
                    "content_type": "paper",
                    "url": "https://arxiv.org/abs/2608.00001",
                    "paper_status_label": "今日新增",
                    "is_reappeared_update": False,
                }])
                write_json(slot_dir / f"2026080{day}_1300.json", {
                    "slot_id": f"2026080{day}_1300",
                    "status": "sent",
                    "run_id": run_id,
                    "finished_at": f"2026-08-0{day}T13:05:00",
                })

            result = build_paper_freshness_production_acceptance(
                {"send_slot_dir": str(slot_dir)},
                db_path=str(db_path),
            )

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["verified_days"], 1)
        self.assertEqual(result["reports"][0]["unjustified_7d_repeat_count"], 1)

    def test_freshness_acceptance_rejects_update_label_without_changed_evidence(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            slot_dir = root / "send_slots"
            slot_dir.mkdir()
            db_path = root / "reports.db"
            db = Database(str(db_path))
            for day, version, label, reason in (
                (1, 1, "今日新增", ""),
                (2, 1, "版本更新", "claimed update"),
            ):
                report_id = f"report-{day}"
                run_id = f"run-{day}"
                db.record_report_run(
                    report_id,
                    run_id,
                    quality_status="passed",
                    quality_diagnostics={
                        "report_design_version": "v10-learning-digest",
                        "paper_freshness": {"paper_freshness_status": "passed"},
                    },
                    delivery_status="sent",
                )
                repeated_item = {
                    "id": day,
                    "content_type": "paper",
                    "url": f"https://arxiv.org/abs/2608.00001v{version}",
                    "arxiv_version": version,
                    "paper_status_label": label,
                    "paper_change_reason": reason,
                    "is_reappeared_update": day == 2,
                    "facts": {"method": "same method", "metric_result": "same result"},
                }
                fresh_items = [
                    {
                        "id": day * 100 + index,
                        "content_type": "paper",
                        "url": f"https://arxiv.org/abs/2608.{day}{index:04d}",
                        "paper_status_label": "今日新增",
                        "is_reappeared_update": False,
                    }
                    for index in range(19)
                ]
                db.record_report_items(report_id, [repeated_item, *fresh_items])
                write_json(slot_dir / f"2026080{day}_1300.json", {
                    "slot_id": f"2026080{day}_1300",
                    "status": "sent",
                    "run_id": run_id,
                    "finished_at": f"2026-08-0{day}T13:05:00",
                })

            result = build_paper_freshness_production_acceptance(
                {"send_slot_dir": str(slot_dir)},
                db_path=str(db_path),
            )

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["reports"][0]["unjustified_7d_repeat_count"], 1)
        self.assertEqual(result["reports"][0]["unjustified_7d_repeat_keys"], ["arxiv:2608.00001"])

    def test_freshness_acceptance_allows_real_code_release(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            slot_dir = root / "send_slots"
            slot_dir.mkdir()
            db_path = root / "reports.db"
            db = Database(str(db_path))
            for day, code_url in ((1, ""), (2, "https://github.com/example/project")):
                report_id = f"report-{day}"
                run_id = f"run-{day}"
                db.record_report_run(
                    report_id,
                    run_id,
                    quality_status="passed",
                    quality_diagnostics={
                        "report_design_version": "v10-learning-digest",
                        "paper_freshness": {"paper_freshness_status": "passed"},
                    },
                    delivery_status="sent",
                )
                repeated_item = {
                    "id": day,
                    "content_type": "paper",
                    "url": "https://arxiv.org/abs/2608.00001",
                    "paper_status_label": "代码已发布" if day == 2 else "今日新增",
                    "paper_change_reason": "new repository released" if day == 2 else "",
                    "is_reappeared_update": day == 2,
                    "facts": {
                        "code_repository": code_url,
                        "evidence": ["The official code repository is now released."] if day == 2 else [],
                    },
                }
                fresh_items = [
                    {
                        "id": day * 100 + index,
                        "content_type": "paper",
                        "url": f"https://arxiv.org/abs/2608.{day}{index:04d}",
                        "paper_status_label": "今日新增",
                        "is_reappeared_update": False,
                    }
                    for index in range(19)
                ]
                db.record_report_items(report_id, [repeated_item, *fresh_items])
                write_json(slot_dir / f"2026080{day}_1300.json", {
                    "slot_id": f"2026080{day}_1300",
                    "status": "sent",
                    "run_id": run_id,
                    "finished_at": f"2026-08-0{day}T13:05:00",
                })

            result = build_paper_freshness_production_acceptance(
                {"send_slot_dir": str(slot_dir)},
                db_path=str(db_path),
            )

        self.assertEqual(result["status"], "pending")
        self.assertEqual(result["reports"][0]["unjustified_7d_repeat_count"], 0)

    def test_reappearance_evidence_rejects_generic_important_progress_label(self):
        current = {
            "content_type": "paper",
            "paper_status_label": "重要进展",
            "paper_change_reason": "deployment changed",
            "is_reappeared_update": True,
            "facts": {"deployment_context": "tested in another lab setting"},
        }
        previous = [{"facts": {"deployment_context": "tested in a lab setting"}}]

        self.assertFalse(_paper_reappearance_is_supported(current, previous))

    def test_reappearance_evidence_accepts_changed_experiment_and_rejects_unchanged_metric(self):
        previous = [{"facts": {"metric_result": "Simulation success rate was 72%."}}]
        changed = {
            "content_type": "paper",
            "paper_status_label": "新增实验",
            "paper_change_reason": "Real-robot success rate reached 86% across 120 trials.",
            "is_reappeared_update": True,
            "facts": {"metric_result": "Real-robot success rate reached 86% across 120 trials."},
        }
        unchanged = {
            **changed,
            "paper_change_reason": "same experiment",
            "facts": {"metric_result": "Simulation success rate was 72%."},
        }

        self.assertTrue(_paper_reappearance_is_supported(changed, previous))
        self.assertFalse(_paper_reappearance_is_supported(unchanged, previous))

    def test_acceptance_identity_matches_arxiv_mirrors_and_tracking_urls(self):
        from scheduler_runner import _paper_snapshot_key, _paper_snapshot_keys

        self.assertEqual(
            _paper_snapshot_key({"url": "https://ar5iv.labs.arxiv.org/html/2608.01234v2"}),
            "arxiv:2608.01234",
        )
        self.assertIn(
            "https://publisher.example/paper?a=1",
            _paper_snapshot_keys({"url": "http://www.publisher.example/paper?utm_source=email&a=1#results"}),
        )

    def test_reappearance_evidence_rejects_paraphrased_version_change(self):
        current = {
            "content_type": "paper",
            "url": "https://arxiv.org/abs/2608.00001v2",
            "arxiv_version": 2,
            "paper_status_label": "版本更新",
            "paper_change_reason": "method changed",
            "is_reappeared_update": True,
            "facts": {"method": "uses a transformer policy with visual-language inputs for robot control"},
        }
        previous = [{
            "url": "https://arxiv.org/abs/2608.00001v1",
            "arxiv_version": 1,
            "facts": {"method": "uses a transformer policy with visual language inputs for robot control"},
        }]

        self.assertFalse(_paper_reappearance_is_supported(current, previous))

    def test_print_send_calendar_report_outputs_fresh_calendar(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            calendar_dir = temp_root / "logs"
            calendar_dir.mkdir(parents=True)
            scheduler_config = dict(DEFAULT_SCHEDULER_CONFIG)
            scheduler_config.update(
                {
                    "send_calendar_dir": str(calendar_dir),
                    "log_dir": str(calendar_dir),
                    "status_file": str(calendar_dir / "last_run.json"),
                    "last_success_file": str(calendar_dir / "last_success.json"),
                }
            )

            with patch("scheduler_runner.load_runtime_config", return_value=({}, scheduler_config)):
                with patch("sys.stdout.write") as mocked_write:
                    exit_code = print_send_calendar_report(as_json=False)

            self.assertEqual(exit_code, 0)
            output = "".join(str(call.args[0]) for call in mocked_write.call_args_list)
            self.assertIn("Send calendar:", output)
            self.assertIn("Slots:", output)

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

    def test_describe_task_result_explains_interrupted_ntstatus(self):
        parsed = describe_task_result(-1073741510)

        self.assertEqual(parsed["hex"], "0xC000013A")
        self.assertEqual(parsed["hint"], "interrupted")
        self.assertIn("interrupted", parsed["message"])
        self.assertNotIn("disk", parsed["message"].lower())

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

    def test_is_interactive_task_detects_chinese_and_english_logon_modes(self):
        self.assertTrue(is_interactive_task({"logon_mode": "只使用交互方式"}))
        self.assertTrue(is_interactive_task({"logon_mode": "Interactive only"}))
        self.assertFalse(is_interactive_task({"logon_mode": "交互方式/后台方式"}))
        self.assertFalse(is_interactive_task({"logon_mode": "Interactive/Background"}))
        self.assertFalse(is_interactive_task({"logon_mode": "S4U"}))
        self.assertFalse(is_interactive_task({"logon_mode": "Password"}))

    def test_build_monitored_task_names_includes_auxiliary_tasks_once(self):
        scheduler_config = dict(DEFAULT_SCHEDULER_CONFIG)
        scheduler_config.update(
            {
                "task_names": ["Web_Agent_Send_1200_v2", "Web_Agent_Send_2100_v2"],
                "doctor_task_name": "Web_Agent_Doctor_0900",
                "preflight_task_name": "Web_Agent_Preflight_2030",
            }
        )

        names = build_monitored_task_names(scheduler_config)

        self.assertEqual(
            names,
            [
                "Web_Agent_Send_1200_v2",
                "Web_Agent_Send_2100_v2",
                "Web_Agent_Doctor_0900",
                "Web_Agent_Preflight_2030",
            ],
        )

    def test_classify_quality_warnings_treats_dedupe_as_info(self):
        classified = classify_quality_warnings(["dedupe_removed_updates:2", "selected_papers_below_minimum:8/10"])

        self.assertEqual(classified["info"], ["dedupe_removed_updates:2"])
        self.assertEqual(classified["warn"], ["selected_papers_below_minimum:8/10"])

    def test_resolve_imap_server_infers_common_smtp_hosts(self):
        self.assertEqual(resolve_imap_server("smtp.qq.com"), "imap.qq.com")
        self.assertEqual(resolve_imap_server("smtp.example.com"), "imap.example.com")
        self.assertEqual(resolve_imap_server("smtp.gmail.com", "imap.custom.test"), "imap.custom.test")

    def test_build_email_arrival_check_requires_recorded_subject(self):
        result = build_email_arrival_check(
            {"email": {"arrival_check": {"enabled": True}}},
            dict(DEFAULT_SCHEDULER_CONFIG),
            {"finished_at": "2026-05-06T12:05:00"},
        )

        self.assertFalse(result["verified"])
        self.assertEqual(result["status"], "skipped_missing_subject")
        self.assertIn("email_subject", result["error"])

    def test_build_email_arrival_check_expands_window_from_last_success_time(self):
        captured = {}

        def fake_verify_email_arrival(**kwargs):
            captured.update(kwargs)
            return {"enabled": True, "verified": False, "status": "not_found"}

        finished_at = (datetime.now() - timedelta(hours=3)).isoformat(timespec="seconds")
        with patch.dict(
            os.environ,
            {
                "EMAIL_SMTP_SERVER": "smtp.qq.com",
                "EMAIL_IMAP_USERNAME": "sender@example.com",
                "EMAIL_IMAP_PASSWORD": "password",
                "EMAIL_SENDER": "sender@example.com",
            },
            clear=False,
        ), patch("scheduler_runner.verify_email_arrival", side_effect=fake_verify_email_arrival):
            build_email_arrival_check(
                {"email": {"arrival_check": {"enabled": True, "since_minutes": 30}}},
                dict(DEFAULT_SCHEDULER_CONFIG),
                {"finished_at": finished_at, "email_subject": "[2026-05-26 21:02] AI Frontier Intelligence Daily"},
            )

        self.assertGreaterEqual(captured["since_minutes"], 180)
        self.assertEqual(captured["expected_sender"], "sender@example.com")

    def test_report_model_path_backfill_candidates_find_latest_legacy_snapshots(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db = Database(str(Path(temp_dir) / "ai_news.db"))
            db.insert_article({"title": "Legacy item", "url": "https://example.com/legacy"})
            db.insert_article({"title": "V2 item", "url": "https://example.com/v2"})
            articles = db.get_articles_for_run("", processed_only=False)
            article_by_url = {article["url"]: article for article in articles}
            db.record_report_run("report-1", "run-1", quality_diagnostics={"model_path_breakdown": {"legacy_pending_backfill": 1}})
            db.record_report_items(
                "report-1",
                [
                    {
                        "id": article_by_url["https://example.com/legacy"]["id"],
                        "report_rank": 1,
                        "report_section": "watch",
                        "title": "Legacy item",
                    },
                    {
                        "id": article_by_url["https://example.com/v2"]["id"],
                        "report_rank": 2,
                        "report_section": "watch",
                        "title": "V2 item",
                        "model_used": "deepseek-v4-pro",
                    },
                ],
            )

            candidates = get_report_model_path_backfill_candidates(db, "report-1")

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["title"], "Legacy item")

    def test_report_model_path_refresh_candidates_include_ambiguous_v2_fallbacks(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db = Database(str(Path(temp_dir) / "ai_news.db"))
            db.insert_article({"title": "V2 fallback item", "url": "https://example.com/v2-fallback"})
            article = db.get_articles_for_run("", processed_only=False)[0]
            db.record_report_run("report-1", "run-1", quality_diagnostics={"model_path_breakdown": {"fallback_v2": 1}})
            db.record_report_items(
                "report-1",
                [
                    {
                        "id": article["id"],
                        "report_rank": 1,
                        "report_section": "watch",
                        "title": "V2 fallback item",
                        "analysis_version": "v2",
                        "model_used": "",
                    }
                ],
            )

            default_candidates = get_report_model_path_backfill_candidates(db, "report-1")
            refresh_candidates = get_report_model_path_backfill_candidates(
                db,
                "report-1",
                include_v2_fallback=True,
            )

        self.assertEqual(default_candidates, [])
        self.assertEqual(len(refresh_candidates), 1)
        self.assertEqual(refresh_candidates[0]["title"], "V2 fallback item")

    def test_scan_report_quality_lists_bad_title_issues(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = str(Path(temp_dir) / "ai_news.db")
            db = Database(db_path)
            db.insert_article({"title": "AgentWatch demonstrates proactive monitoring", "url": "https://example.com/bad-title"})
            article = db.get_articles_for_run("", processed_only=False)[0]
            db.record_report_run("report-1", "run-1")
            db.record_report_items(
                "report-1",
                [
                    {
                        "id": article["id"],
                        "report_rank": 1,
                        "report_section": "must_read",
                        "title_cn": "AgentWatchdemonstrproactive AWS mo",
                        "summary": "AgentWatch demonstrates proactive monitoring.",
                        "evidence_quality": 0.8,
                        "information_density": 0.8,
                    }
                ],
            )
            with patch("scheduler_runner.ROOT", Path(temp_dir)):
                result = scan_report_quality(report_id="report-1", persist=True)
                latest = db.get_report_run("report-1")

        self.assertEqual(result["counts"]["bad_title_count"], 1)
        self.assertEqual(result["issues"][0]["issue_types"], ["bad_title"])
        self.assertEqual(result["focus_issue_count"], 1)
        self.assertEqual(result["brief_issue_count"], 0)
        self.assertEqual(latest["quality_diagnostics"]["content_quality"]["bad_title_count"], 1)
        self.assertEqual(latest["quality_diagnostics"]["last_report_quality_scan"]["focus_issue_count"], 1)

    def test_scan_report_quality_clears_stale_bad_title_diagnostics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = str(Path(temp_dir) / "ai_news.db")
            db = Database(db_path)
            db.insert_article({"title": "AgentWatch monitoring update", "url": "https://example.com/ok-title"})
            article = db.get_articles_for_run("", processed_only=False)[0]
            db.record_report_run(
                "report-1",
                "run-1",
                quality_status="degraded",
                quality_diagnostics={
                    "warnings": ["bad_titles_present:1", "dedupe_removed_updates:1"],
                    "quality_gate": {
                        "status": "degraded",
                        "bad_title_count": 1,
                        "failed_item_urls": ["https://example.com/ok-title"],
                    },
                    "title_repair": {"bad_title_repaired_count": 1, "bad_title_unresolved_count": 1},
                },
            )
            db.record_report_items(
                "report-1",
                [
                    {
                        "id": article["id"],
                        "report_rank": 1,
                        "report_section": "must_read",
                        "title_cn": "AgentWatch展示主动式AWS监控进展",
                        "summary": "AgentWatch展示主动式AWS监控进展。",
                        "facts": {"who": "AgentWatch", "action": "展示", "target": "主动式AWS监控"},
                        "evidence_quality": 0.8,
                        "information_density": 0.8,
                    }
                ],
            )
            with patch("scheduler_runner.ROOT", Path(temp_dir)):
                result = scan_report_quality(report_id="report-1", persist=True)
                latest = db.get_report_run("report-1")

        self.assertEqual(result["issue_count"], 0)
        self.assertEqual(latest["quality_status"], "passed")
        self.assertEqual(latest["quality_diagnostics"]["quality_gate"]["status"], "passed")
        self.assertEqual(latest["quality_diagnostics"]["quality_gate"]["failed_item_urls"], [])
        self.assertEqual(latest["quality_diagnostics"]["title_repair"]["bad_title_unresolved_count"], 0)
        self.assertEqual(latest["quality_diagnostics"]["warnings"], ["dedupe_removed_updates:1"])

    def test_scan_report_quality_never_overrides_v8_final_html_failure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = str(Path(temp_dir) / "ai_news.db")
            db = Database(db_path)
            db.insert_article({"title": "Clean item", "url": "https://example.com/clean-v8"})
            article = db.get_articles_for_run("", processed_only=False)[0]
            db.record_report_run(
                "report-v8",
                "run-v8",
                quality_status="failed",
                quality_diagnostics={
                    "report_design_version": "v8-editorial-reader",
                    "quality_gate": {
                        "status": "failed",
                        "final_html_quality_status": "failed",
                        "reading_budget_underfilled": True,
                    },
                },
            )
            db.record_report_items(
                "report-v8",
                [
                    {
                        "id": article["id"],
                        "report_rank": 1,
                        "report_section": "must_read",
                        "title_cn": "Clean Item发布可验证更新",
                        "summary": "Clean Item发布可验证更新。",
                        "facts": {"who": "Clean Item", "action": "发布", "target": "可验证更新"},
                        "evidence_quality": 0.8,
                        "information_density": 0.8,
                    }
                ],
            )

            result = scan_report_quality(report_id="report-v8", persist=True, db_path=db_path)
            latest = db.get_report_run("report-v8")

        self.assertEqual(result["focus_issue_count"], 0)
        self.assertEqual(latest["quality_status"], "failed")
        self.assertEqual(latest["quality_diagnostics"]["quality_gate"]["status"], "failed")
        self.assertEqual(
            latest["quality_diagnostics"]["quality_gate"]["final_html_quality_status"],
            "failed",
        )

    def test_scan_report_quality_marks_passed_run_degraded_when_issues_remain(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = str(Path(temp_dir) / "ai_news.db")
            db = Database(db_path)
            db.insert_article({"title": "AgentWatch demonstrates proactive monitoring", "url": "https://example.com/bad-title"})
            article = db.get_articles_for_run("", processed_only=False)[0]
            db.record_report_run("report-1", "run-1", quality_status="passed")
            db.record_report_items(
                "report-1",
                [
                    {
                        "id": article["id"],
                        "report_rank": 1,
                        "report_section": "must_read",
                        "title_cn": "AgentWatchdemonstrproactive AWS mo",
                        "summary": "AgentWatch demonstrates proactive monitoring.",
                        "evidence_quality": 0.8,
                        "information_density": 0.8,
                    }
                ],
            )

            result = scan_report_quality(report_id="report-1", persist=True, db_path=db_path)
            latest = db.get_report_run("report-1")

        self.assertEqual(result["issue_count"], 1)
        self.assertEqual(result["focus_issue_count"], 1)
        self.assertEqual(latest["quality_status"], "degraded")

    def test_scan_report_quality_keeps_brief_only_issues_out_of_degraded_status(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = str(Path(temp_dir) / "ai_news.db")
            db = Database(db_path)
            db.insert_article({"title": "Thin brief item", "url": "https://example.com/thin-brief"})
            article = db.get_articles_for_run("", processed_only=False)[0]
            db.record_report_run("report-1", "run-1", quality_status="passed")
            db.record_report_items(
                "report-1",
                [
                    {
                        "id": article["id"],
                        "report_rank": 1,
                        "report_section": "brief",
                        "title_cn": "低证据快讯仍保留为短讯",
                        "summary": "这是一条证据不足的短讯。",
                        "evidence_quality": 0.2,
                        "information_density": 0.2,
                    }
                ],
            )

            result = scan_report_quality(report_id="report-1", persist=True, db_path=db_path)
            latest = db.get_report_run("report-1")

        self.assertEqual(result["issue_count"], 1)
        self.assertEqual(result["focus_issue_count"], 0)
        self.assertEqual(result["brief_issue_count"], 1)
        self.assertEqual(latest["quality_status"], "passed")
        self.assertEqual(latest["quality_diagnostics"]["last_report_quality_scan"]["brief_issue_count"], 1)

    def test_refresh_report_quality_after_run_updates_status_snapshot(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = str(Path(temp_dir) / "ai_news.db")
            db = Database(db_path)
            db.insert_article({"title": "Amazon Bedrock AgentCore payment preview", "url": "https://example.com/ok-title"})
            article = db.get_articles_for_run("", processed_only=False)[0]
            db.record_report_run("report-1", "run-1", quality_status="degraded")
            db.record_report_items(
                "report-1",
                [
                    {
                        "id": article["id"],
                        "report_rank": 1,
                        "report_section": "must_read",
                        "title_cn": "Amazon Bedrock发布AgentCore支付预览",
                        "summary": "Amazon Bedrock 发布 AgentCore 支付预览，开发者可以在代理流程中测试支付能力。",
                        "facts": {
                            "who": "Amazon Bedrock",
                            "action": "发布",
                            "target": "AgentCore支付预览",
                        },
                        "evidence_quality": 0.8,
                        "information_density": 0.8,
                    }
                ],
            )

            status = refresh_report_quality_after_run({"report_id": "report-1", "quality_status": "degraded"}, db_path=db_path)

        self.assertEqual(status["quality_status"], "passed")
        self.assertEqual(status["post_send_quality_scan"]["issue_count"], 0)
        self.assertEqual(status["post_send_quality_scan"]["focus_issue_count"], 0)
        self.assertEqual(status["quality_diagnostics"]["quality_gate"]["status"], "passed")

    def test_fix_report_bad_titles_only_updates_title_fields(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = str(Path(temp_dir) / "ai_news.db")
            db = Database(db_path)
            db.insert_article(
                {
                    "title": "AgentWatch demonstrates proactive AWS monitoring",
                    "url": "https://example.com/bad-title",
                    "facts": {"who": "AgentWatch", "action": "demonstrates", "target": "proactive AWS monitoring"},
                }
            )
            article = db.get_articles_for_run("", processed_only=False)[0]
            db.record_report_run("report-1", "run-1")
            db.record_report_items(
                "report-1",
                [
                    {
                        "id": article["id"],
                        "report_rank": 1,
                        "report_section": "watch",
                        "title_cn": "AgentWatchdemonstrproactive AWS mo",
                        "summary": "AgentWatch展示了主动式AWS监控的新进展。Keep this summary unchanged.",
                        "facts": {"who": "AgentWatch", "action": "demonstrates", "target": "proactive AWS monitoring"},
                        "evidence_quality": 0.8,
                        "information_density": 0.8,
                    }
                ],
            )
            with patch("scheduler_runner.ROOT", Path(temp_dir)):
                result = fix_report_bad_titles(report_id="report-1")
            conn = db._get_conn()
            try:
                row = conn.execute("SELECT snapshot_json FROM report_items WHERE report_id = 'report-1'").fetchone()
                snapshot = json.loads(row["snapshot_json"])
            finally:
                conn.close()
            latest = db.get_report_run("report-1")

        self.assertEqual(result["updated"], 1)
        self.assertEqual(snapshot["summary"], "AgentWatch展示了主动式AWS监控的新进展。Keep this summary unchanged.")
        self.assertNotEqual(snapshot["title_cn"], "AgentWatchdemonstrproactive AWS mo")
        self.assertEqual(result["title_repair"]["bad_title_unresolved_count"], 0)
        self.assertEqual(latest["quality_diagnostics"]["title_repair"]["bad_title_repaired_count"], 1)
        self.assertEqual(latest["quality_diagnostics"]["title_repair"]["examples"][0]["source"], "summary")

    def test_suggest_title_from_snapshot_uses_facts(self):
        self.assertIn(
            "AgentWatch",
            suggest_title_from_snapshot({"facts": {"who": "AgentWatch", "action": "demonstrates", "target": "proactive AWS monitoring"}}),
        )
        self.assertEqual(
            suggest_title_from_snapshot_with_source(
                {"facts": {"who": "AgentWatch", "action": "demonstrates", "target": "proactive AWS monitoring"}}
            )["source"],
            "facts",
        )

    def test_build_task_repair_commands_include_model_path_backfill(self):
        commands = build_task_repair_commands(dict(DEFAULT_SCHEDULER_CONFIG))
        self.assertTrue(any(command["name"] == "backfill_latest_report_model_path" for command in commands))
        self.assertTrue(any("--backfill-model-path" in command["command"] for command in commands))
        self.assertTrue(any(command["name"] == "refresh_fallback_model_path" for command in commands))
        self.assertTrue(any("--refresh-fallback-model-path" in command["command"] for command in commands))

    def test_normalize_model_path_breakdown_maps_old_v2_label(self):
        self.assertEqual(
            normalize_model_path_breakdown({"v2": 2, "deepseek-v4-pro": 1}),
            {"fallback_v2": 2, "deepseek-v4-pro": 1},
        )

    def test_product_diagnostics_include_title_repair_examples(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db = Database(str(Path(temp_dir) / "ai_news.db"))
            db.record_report_run(
                "report-1",
                "run-1",
                quality_status="passed",
                quality_diagnostics={
                    "title_repair": {
                        "bad_title_repaired_count": 1,
                        "bad_title_unresolved_count": 0,
                        "examples": [
                            {
                                "rank": 1,
                                "section": "must_read",
                                "article_id": 42,
                                "old_title": "AgentWatchdemonstrproactive AWS mo",
                                "new_title": "AgentWatch展示了主动式AWS监控的新进展",
                            }
                        ],
                    }
                },
            )
            with patch("scheduler_runner.ROOT", Path(temp_dir)):
                diagnostics = build_product_diagnostics({"feedback": {"enabled": False}}, {}, {})

        self.assertEqual(diagnostics["title_repair"]["bad_title_repaired_count"], 1)
        self.assertEqual(
            diagnostics["title_repair"]["examples"][0]["new_title"],
            "AgentWatch展示了主动式AWS监控的新进展",
        )

    def test_product_diagnostics_include_post_send_quality_scan(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            Database(str(temp_root / "ai_news.db"))
            last_success = {
                "post_send_quality_scan": {
                    "report_id": "report-1",
                    "scanned": 8,
                    "issue_count": 0,
                    "focus_issue_count": 0,
                    "brief_issue_count": 0,
                    "error": "",
                }
            }

            with patch("scheduler_runner.ROOT", temp_root):
                diagnostics = build_product_diagnostics({"feedback": {"enabled": False}}, {}, last_success)

        self.assertEqual(diagnostics["post_send_quality_scan"]["scanned"], 8)
        self.assertEqual(diagnostics["post_send_quality_scan"]["issue_count"], 0)
        self.assertEqual(diagnostics["post_send_quality_scan"]["focus_issue_count"], 0)

    def test_product_diagnostics_include_design_version_and_source_health(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            db = Database(str(temp_root / "ai_news.db"))
            db.record_report_run(
                "report-1",
                "run-1",
                quality_status="passed",
                quality_diagnostics={
                    "report_design_version": "v2.1-intelligence-email",
                    "source_health": {
                        "source_count": 3,
                        "risky_source_count": 1,
                        "unstable_source_count": 1,
                        "unstable_rows": [{"label": "RSSCollector[TechCrunch AI]"}],
                    },
                    "source_weight_adjustments": {
                        "enabled": True,
                        "weights": {"RSSCollector[TechCrunch AI]": -1.2},
                    },
                },
            )

            with patch("scheduler_runner.ROOT", temp_root):
                diagnostics = build_product_diagnostics({"feedback": {"enabled": False}}, {}, {})

        self.assertEqual(diagnostics["report_design_version"], "v2.1-intelligence-email")
        self.assertEqual(diagnostics["source_health"]["unstable_source_count"], 1)
        self.assertEqual(diagnostics["source_weight_adjustments"]["weights"]["RSSCollector[TechCrunch AI]"], -1.2)

    def test_product_diagnostics_expose_arxiv_health_breakdown(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            db = Database(str(temp_root / "ai_news.db"))
            db.record_report_run(
                "report-arxiv",
                "run-arxiv",
                quality_status="passed",
                quality_diagnostics={
                    "report_design_version": "v10-learning-digest",
                    "collection": {
                        "collector_success_count": 5,
                        "arxiv_zero_result_warning_count": 1,
                        "arxiv_true_zero_result_count": 1,
                        "arxiv_no_match_result_count": 2,
                        "arxiv_http_error_count": 2,
                        "arxiv_parse_error_count": 3,
                        "arxiv_fallback_recovery_count": 1,
                        "arxiv_retry_paths": [{
                            "source": "ArxivCollector[World Model]",
                            "category": "cs.AI",
                            "attempted_show_counts": [500, 100, 50],
                            "successful_show_count": 50,
                            "result": "success",
                        }],
                    },
                },
            )

            with patch("scheduler_runner.ROOT", temp_root):
                diagnostics = build_product_diagnostics(
                    {"feedback": {"enabled": False}, "report": {"design_version": "v10-learning-digest"}},
                    {},
                    {},
                )

        self.assertEqual(diagnostics["arxiv_zero_result_warning_count"], 1)
        self.assertEqual(diagnostics["arxiv_true_zero_result_count"], 1)
        self.assertEqual(diagnostics["arxiv_no_match_result_count"], 2)
        self.assertEqual(diagnostics["arxiv_http_error_count"], 2)
        self.assertEqual(diagnostics["arxiv_parse_error_count"], 3)
        self.assertEqual(diagnostics["arxiv_fallback_recovery_count"], 1)
        self.assertEqual(diagnostics["arxiv_retry_paths"][0]["successful_show_count"], 50)

    def test_product_diagnostics_expose_hard_and_ideal_paper_targets(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            db = Database(str(temp_root / "ai_news.db"))
            db.record_report_run(
                "short-report",
                "short-run",
                quality_status="passed",
                quality_diagnostics={
                    "report_design_version": "v10-learning-digest",
                    "quality_gate": {"status": "passed", "fresh_paper_count": 12},
                    "paper_freshness": {
                        "paper_freshness_status": "passed",
                        "configured_min_visible_paper_count": 10,
                        "effective_min_visible_paper_count": 10,
                        "configured_target_min_paper_count": 18,
                    },
                },
            )

            with patch("scheduler_runner.ROOT", temp_root):
                diagnostics = build_product_diagnostics(
                    {
                        "feedback": {"enabled": False},
                        "report": {
                            "design_version": "v10-learning-digest",
                            "min_visible_paper_count": 10,
                            "paper_target_min_count": 18,
                        },
                    },
                    {},
                    {},
                )

        self.assertEqual(diagnostics["fresh_paper_count"], 12)
        self.assertEqual(diagnostics["configured_minimum_paper_count"], 10)
        self.assertEqual(diagnostics["minimum_paper_count"], 10)
        self.assertEqual(diagnostics["configured_target_min_paper_count"], 18)
        self.assertEqual(diagnostics["recommended_minimum_paper_count"], 18)
        self.assertTrue(diagnostics["paper_target_underfilled"])

    def test_product_diagnostics_falls_back_to_current_target_for_legacy_report(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            db = Database(str(temp_root / "ai_news.db"))
            db.record_report_run(
                "legacy-report",
                "legacy-run",
                quality_status="passed",
                quality_diagnostics={
                    "report_design_version": "v10-learning-digest",
                    "quality_gate": {"status": "passed", "fresh_paper_count": 12},
                    "paper_freshness": {
                        "paper_freshness_status": "passed",
                        "effective_min_visible_paper_count": 10,
                    },
                },
            )

            with patch("scheduler_runner.ROOT", temp_root):
                diagnostics = build_product_diagnostics(
                    {
                        "feedback": {"enabled": False},
                        "report": {
                            "design_version": "v10-learning-digest",
                            "min_visible_paper_count": 10,
                            "paper_target_min_count": 18,
                        },
                    },
                    {},
                    {},
                )

        self.assertEqual(diagnostics["recommended_minimum_paper_count"], 18)
        self.assertTrue(diagnostics["paper_target_underfilled"])

    def test_product_diagnostics_use_last_real_collection_after_rerender(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            db = Database(str(temp_root / "ai_news.db"))
            db.record_report_run(
                "collected-report",
                "collected-run",
                quality_status="passed",
                quality_diagnostics={
                    "report_design_version": "v10-learning-digest",
                    "collection": {
                        "collector_success_count": 6,
                        "arxiv_zero_result_warning_count": 1,
                        "arxiv_http_error_count": 2,
                        "arxiv_parse_error_count": 0,
                        "arxiv_fallback_recovery_count": 2,
                    },
                    "quality_gate": {"status": "passed", "fresh_paper_count": 18},
                },
            )
            db.record_report_run(
                "rerender-report",
                "rerender-run",
                quality_status="passed",
                quality_diagnostics={
                    "report_design_version": "v10-learning-digest",
                    "collection": {
                        "collector_success_count": 0,
                        "arxiv_zero_result_warning_count": 0,
                        "arxiv_http_error_count": 0,
                    },
                    "quality_gate": {"status": "passed", "fresh_paper_count": 23},
                },
            )

            with patch("scheduler_runner.ROOT", temp_root):
                diagnostics = build_product_diagnostics(
                    {"feedback": {"enabled": False}, "report": {"design_version": "v10-learning-digest"}},
                    {},
                    {},
                )

        self.assertEqual(diagnostics["fresh_paper_count"], 23)
        self.assertEqual(diagnostics["arxiv_collection_report_id"], "collected-report")
        self.assertEqual(diagnostics["arxiv_zero_result_warning_count"], 1)
        self.assertEqual(diagnostics["arxiv_http_error_count"], 2)
        self.assertEqual(diagnostics["arxiv_fallback_recovery_count"], 2)

    def test_product_diagnostics_preserve_v8_final_html_failure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            db = Database(str(temp_root / "ai_news.db"))
            db.record_report_run(
                "report-v8",
                "run-v8",
                quality_status="failed",
                quality_diagnostics={
                    "report_design_version": "v8-editorial-reader",
                    "quality_gate": {
                        "status": "failed",
                        "editorial_quality_status": "passed",
                        "final_html_quality_status": "failed",
                        "final_html_bad_title_count": 2,
                        "untranslated_fact_count": 1,
                        "exact_duplicate_sentence_count": 3,
                        "paper_mechanism_missing_count": 1,
                        "paper_result_context_missing_count": 1,
                        "appendix_body_overlap_count": 0,
                        "visible_text_chars": 6400,
                        "focus_source_concentration": 0.2,
                        "memory_item_count": 3,
                        "featured_paper_count": 6,
                    },
                },
            )

            with patch("scheduler_runner.ROOT", temp_root):
                diagnostics = build_product_diagnostics(
                    {
                        "feedback": {"enabled": False},
                        "report": {"design_version": "v8-editorial-reader"},
                    },
                    {},
                    {},
                )

        self.assertEqual(diagnostics["last_report_quality_status"], "failed")
        self.assertEqual(diagnostics["final_html_quality_status"], "failed")
        self.assertEqual(diagnostics["final_html_bad_title_count"], 2)
        self.assertEqual(diagnostics["untranslated_fact_count"], 1)
        self.assertEqual(diagnostics["exact_duplicate_sentence_count"], 3)
        self.assertEqual(diagnostics["visible_text_chars"], 6400)
        self.assertEqual(diagnostics["memory_item_count"], 3)
        self.assertEqual(diagnostics["featured_paper_count"], 6)

    def test_source_health_summary_marks_unstable_sources(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db = Database(str(Path(temp_dir) / "ai_news.db"))
            current_runs = [
                {
                    "label": "RSSCollector[TechCrunch AI]",
                    "status": "error",
                    "inserted_count": 0,
                    "collected_count": 0,
                    "duration_seconds": 1,
                    "error": "ssl",
                }
            ]
            db.record_collector_runs("run-1", current_runs)
            db.record_collector_runs("run-2", current_runs)

            summary = build_source_health_summary(current_runs, db, history_limit=10)

        self.assertEqual(summary["unstable_source_count"], 1)
        self.assertEqual(summary["unstable_rows"][0]["label"], "RSSCollector[TechCrunch AI]")

    def test_collector_run_persists_arxiv_retry_diagnostics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db = Database(str(Path(temp_dir) / "ai_news.db"))
            db.record_collector_runs(
                "run-arxiv",
                [{
                    "label": "ArxivCollector[World Model]",
                    "status": "success",
                    "inserted_count": 3,
                    "collected_count": 4,
                    "duration_seconds": 1.2,
                    "error": "",
                    "diagnostics": {
                        "request_error_count": 1,
                        "successful_page_count": 1,
                        "fallback_show_counts": [500],
                    },
                }],
            )

            rows = db.get_recent_collector_runs("ArxivCollector[", limit=1)

        self.assertEqual(rows[0]["diagnostics"]["request_error_count"], 1)
        self.assertEqual(rows[0]["diagnostics"]["fallback_show_counts"], [500])

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
                    "legacy_task_names": [],
                    "monitor_auxiliary_tasks": False,
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
            "last_success": {
                "success": True,
                "status": "success",
                "delivery_status": "sent",
                "finished_at": "2026-04-12T03:05:00",
                "html_report_path": "archive/report_ok.html",
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
        self.assertEqual(payload["counts"]["warn"], 2)
        warned = [item for item in payload["checks"] if item["level"] == "warn"]
        self.assertTrue(any("Web_Agent_Send_1200" in item["name"] for item in warned))
        self.assertTrue(any(item["name"] == "task_repair_commands" for item in warned))
        self.assertTrue(any(command["name"] == "rebuild_interactive_send_tasks" for command in payload["repair_commands"]))
        self.assertTrue(any("setup_scheduled_tasks.ps1" in command["command"] for command in payload["repair_commands"]))

    def test_build_doctor_payload_reports_post_send_quality_scan(self):
        mocked_status = {
            "current_time": "2026-04-12T13:10:00",
            "last_run": {
                "success": True,
                "status": "success",
                "delivery_status": "sent",
                "finished_at": "2026-04-12T13:05:00",
                "log_file": "D:/Web_Agent/logs/scheduler_run_ok.log",
            },
            "last_success": {
                "success": True,
                "status": "success",
                "delivery_status": "sent",
                "finished_at": "2026-04-12T13:05:00",
                "html_report_path": "archive/report_ok.html",
                "log_file": "D:/Web_Agent/logs/scheduler_run_ok.log",
                "post_send_quality_scan": {
                    "report_id": "report-1",
                    "scanned": 8,
                    "issue_count": 2,
                    "focus_issue_count": 0,
                    "brief_issue_count": 2,
                    "section_issue_counts": {"brief": 2},
                    "error": "",
                },
            },
            "lock": {"state": "idle", "pid": "", "acquired_at": ""},
            "tasks": [],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            archive_dir = temp_root / "archive"
            log_dir = temp_root / "logs"
            archive_dir.mkdir()
            log_dir.mkdir()
            (temp_root / "reports_manifest.json").write_text("{}", encoding="utf-8")
            Database(str(temp_root / "ai_news.db"))

            scheduler_config = dict(DEFAULT_SCHEDULER_CONFIG)
            scheduler_config.update(
                {
                    "log_dir": str(log_dir),
                    "status_file": str(log_dir / "last_run.json"),
                    "lock_file": str(log_dir / "scheduler.lock"),
                    "task_names": [],
                    "legacy_task_names": [],
                    "monitor_auxiliary_tasks": False,
                }
            )
            config = {"archive": {"report_dir": str(archive_dir)}, "feedback": {"enabled": False}}

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

        post_scan_check = next(item for item in payload["checks"] if item["name"] == "post_send_quality_scan")
        self.assertEqual(post_scan_check["level"], "ok")
        self.assertEqual(post_scan_check["data"]["scanned"], 8)
        self.assertEqual(post_scan_check["data"]["brief_issue_count"], 2)
        self.assertEqual(payload["post_send_quality_scan"]["issue_count"], 2)
        self.assertEqual(payload["post_send_quality_scan"]["focus_issue_count"], 0)

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

    def test_build_doctor_payload_warns_when_workspace_disk_is_low(self):
        mocked_status = {
            "current_time": "2026-08-12T15:00:00",
            "last_run": {},
            "lock": {"state": "idle", "pid": "", "acquired_at": ""},
            "tasks": [],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            (temp_root / "logs").mkdir()
            scheduler_config = dict(DEFAULT_SCHEDULER_CONFIG)
            scheduler_config.update(
                {
                    "log_dir": str(temp_root / "logs"),
                    "status_file": str(temp_root / "logs" / "last_run.json"),
                    "lock_file": str(temp_root / "logs" / "scheduler.lock"),
                    "task_names": [],
                    "legacy_task_names": [],
                    "monitor_auxiliary_tasks": False,
                    "minimum_free_disk_mb": 512,
                    "critical_free_disk_mb": 8,
                }
            )
            fake_usage = shutil._ntuple_diskusage(total=1024 * 1024 * 1024, used=984 * 1024 * 1024, free=40 * 1024 * 1024)
            with patch("scheduler_runner.ROOT", temp_root):
                with patch("scheduler_runner.load_runtime_config", return_value=({"archive": {}}, scheduler_config)):
                    with patch("scheduler_runner.build_scheduler_status_payload", return_value=mocked_status):
                        with patch("scheduler_runner.shutil.disk_usage", return_value=fake_usage):
                            with patch.dict(os.environ, {"EMAIL_RECIPIENT": "a@b.com", "EMAIL_SENDER": "c@d.com", "EMAIL_PASSWORD": "x"}):
                                payload = build_doctor_payload()

        disk_check = next(item for item in payload["checks"] if item["name"] == "disk_space")
        self.assertEqual(disk_check["level"], "warn")
        self.assertEqual(disk_check["data"]["free_mb"], 40.0)

    def test_build_doctor_payload_warns_when_offline_tasks_are_required_but_interactive(self):
        mocked_status = {
            "current_time": "2026-04-12T03:10:00",
            "last_run": {"success": True, "status": "success", "delivery_status": "sent"},
            "last_success": {"finished_at": "2026-04-12T03:05:00", "html_report_path": "archive/report.html"},
            "lock": {"state": "idle", "pid": "", "acquired_at": ""},
            "tasks": [
                {
                    "task_name": "\\Web_Agent_Send_1200_v2",
                    "available": True,
                    "status": "就绪",
                    "next_run_time": "2026/4/12 12:00:00",
                    "logon_mode": "只使用交互方式",
                    "last_result": "0",
                    "last_result_hint": "success",
                }
            ],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            (temp_root / "logs").mkdir()
            (temp_root / "archive").mkdir()
            (temp_root / "reports_manifest.json").write_text("{}", encoding="utf-8")
            scheduler_config = dict(DEFAULT_SCHEDULER_CONFIG)
            scheduler_config.update(
                {
                    "log_dir": str(temp_root / "logs"),
                    "status_file": str(temp_root / "logs" / "last_run.json"),
                    "last_success_file": str(temp_root / "logs" / "last_success.json"),
                    "lock_file": str(temp_root / "logs" / "scheduler.lock"),
                    "task_names": ["Web_Agent_Send_1200_v2"],
                    "monitor_auxiliary_tasks": False,
                    "require_offline_tasks": True,
                }
            )
            config = {"archive": {"report_dir": str(temp_root / "archive")}}

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
        warned = [item for item in payload["checks"] if item["level"] == "warn"]
        self.assertTrue(any("Offline mode required" in item["detail"] for item in warned))

    def test_build_doctor_payload_warns_when_legacy_send_tasks_are_enabled(self):
        mocked_status = {
            "current_time": "2026-04-12T03:10:00",
            "last_run": {"success": True, "status": "success", "delivery_status": "sent"},
            "last_success": {"finished_at": "2026-04-12T03:05:00", "html_report_path": "archive/report.html"},
            "lock": {"state": "idle", "pid": "", "acquired_at": ""},
            "tasks": [],
            "legacy_tasks": [
                {
                    "task_name": "\\Web_Agent_Send_2100",
                    "available": True,
                    "scheduled_task_state": "Enabled",
                    "last_result_hint": "success",
                }
            ],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            (temp_root / "logs").mkdir()
            (temp_root / "archive").mkdir()
            (temp_root / "reports_manifest.json").write_text("{}", encoding="utf-8")
            scheduler_config = dict(DEFAULT_SCHEDULER_CONFIG)
            scheduler_config.update(
                {
                    "log_dir": str(temp_root / "logs"),
                    "status_file": str(temp_root / "logs" / "last_run.json"),
                    "last_success_file": str(temp_root / "logs" / "last_success.json"),
                    "lock_file": str(temp_root / "logs" / "scheduler.lock"),
                }
            )
            config = {"archive": {"report_dir": str(temp_root / "archive")}}

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

        warned = [item for item in payload["checks"] if item["level"] == "warn"]
        self.assertTrue(any(item["name"] == "duplicate_legacy_tasks" for item in warned))

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
        scheduler_config["send_doctor_alert_email"] = True

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
            "repair_commands": [
                {"name": "rebuild_interactive_send_tasks", "command": "powershell -File setup_scheduled_tasks.ps1"}
            ],
        }
        history = {"warn_streak": 2}
        html = build_doctor_alert_html(payload, history)
        self.assertIn("AI 日报需要处理", html)
        self.assertIn("email_env", html)
        self.assertIn("Task result unknown", html)
        self.assertIn("setup_scheduled_tasks.ps1", html)
        self.assertNotIn("setup_offline_tasks.ps1", html)

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

    def test_collect_task_self_heal_candidates_includes_interactive_offline_tasks(self):
        payload = {
            "status": {
                "tasks": [
                    {
                        "task_name": "\\Web_Agent_Send_1200_v2",
                        "available": True,
                        "logon_mode": "Interactive only",
                        "last_result_hint": "success",
                        "last_result": "0",
                    }
                ]
            }
        }
        scheduler_config = dict(DEFAULT_SCHEDULER_CONFIG)
        scheduler_config.update(
            {
                "task_names": ["Web_Agent_Send_1200_v2"],
                "require_offline_tasks": True,
            }
        )

        candidates = collect_task_self_heal_candidates(payload, scheduler_config)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["reason"], "offline_interactive_required")

    def test_collect_task_self_heal_candidates_includes_enabled_legacy_tasks(self):
        payload = {
            "status": {
                "tasks": [],
                "legacy_tasks": [
                    {
                        "task_name": "\\Web_Agent_Send_1200",
                        "available": True,
                        "scheduled_task_state": "已启用",
                    },
                    {
                        "task_name": "\\Web_Agent_Send_2100",
                        "available": True,
                        "scheduled_task_state": "已禁用",
                    },
                ],
            }
        }

        candidates = collect_task_self_heal_candidates(payload, dict(DEFAULT_SCHEDULER_CONFIG))

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["reason"], "enabled_legacy_task")
        self.assertEqual(candidates[0]["task_name"], "\\Web_Agent_Send_1200")

    def test_build_task_repair_commands_include_legacy_cleanup_and_offline_rebuild(self):
        scheduler_config = dict(DEFAULT_SCHEDULER_CONFIG)
        scheduler_config.update(
            {
                "legacy_task_names": ["Web_Agent_Send_1200", "Web_Agent_Send_2100"],
                "offline_task_setup_script": "setup_offline_tasks.ps1",
                "require_offline_tasks": True,
            }
        )

        commands = build_task_repair_commands(scheduler_config)
        command_text = "\n".join(command["command"] for command in commands)

        self.assertIn("repair_scheduled_tasks.ps1", command_text)
        self.assertIn("schtasks /Delete /TN Web_Agent_Send_1200 /F", command_text)
        self.assertIn("schtasks /Delete /TN Web_Agent_Send_2100 /F", command_text)
        self.assertIn("setup_offline_tasks.ps1", command_text)
        self.assertIn("repair_scheduled_tasks.ps1 -UseS4U -NoPrompt", command_text)
        self.assertIn("setup_offline_tasks.ps1 -UseS4U", command_text)
        self.assertIn("scheduler_runner.py", command_text)

    def test_build_task_repair_commands_use_interactive_rebuild_when_offline_not_required(self):
        scheduler_config = dict(DEFAULT_SCHEDULER_CONFIG)
        scheduler_config.update(
            {
                "legacy_task_names": ["Web_Agent_Send_1200", "Web_Agent_Send_2100"],
                "task_setup_script": "setup_scheduled_tasks.ps1",
                "offline_task_setup_script": "setup_offline_tasks.ps1",
                "repair_task_script": "repair_scheduled_tasks.ps1",
                "require_offline_tasks": False,
            }
        )

        commands = build_task_repair_commands(scheduler_config)
        command_text = "\n".join(command["command"] for command in commands)

        self.assertTrue(any(command["name"] == "rebuild_interactive_send_tasks" for command in commands))
        self.assertIn("setup_scheduled_tasks.ps1", command_text)
        self.assertIn("schtasks /Delete /TN Web_Agent_Send_1200 /F", command_text)
        self.assertIn("schtasks /Delete /TN Web_Agent_Send_2100 /F", command_text)
        self.assertNotIn("setup_offline_tasks.ps1", command_text)
        self.assertNotIn("repair_scheduled_tasks.ps1 -UseS4U -NoPrompt", command_text)
        self.assertNotIn("setup_offline_tasks.ps1 -UseS4U", command_text)

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

    def test_run_task_self_heal_blocks_offline_repair_without_password_env(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            script_path = Path(temp_dir) / "setup_offline_tasks.ps1"
            script_path.write_text("Write-Host 'ok'", encoding="utf-8")
            scheduler_config = dict(DEFAULT_SCHEDULER_CONFIG)
            scheduler_config["offline_task_setup_script"] = str(script_path)
            scheduler_config["run_as_password_env"] = "WEB_AGENT_TEST_PASSWORD"
            candidates = [{"task_name": "Web_Agent_Send_1200_v2", "reason": "offline_interactive_required"}]

            with patch.dict(os.environ, {"WEB_AGENT_TEST_PASSWORD": ""}, clear=False):
                result = run_task_self_heal(scheduler_config, candidates, dry_run=False)

            self.assertTrue(result["attempted"])
            self.assertFalse(result["success"])
            self.assertIn("requires a Windows password", result["message"])

    def test_run_task_self_heal_cleans_legacy_tasks_before_password_block(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            script_path = Path(temp_dir) / "setup_offline_tasks.ps1"
            script_path.write_text("Write-Host 'ok'", encoding="utf-8")
            scheduler_config = dict(DEFAULT_SCHEDULER_CONFIG)
            scheduler_config.update(
                {
                    "offline_task_setup_script": str(script_path),
                    "run_as_password_env": "WEB_AGENT_TEST_PASSWORD",
                    "require_offline_tasks": True,
                    "task_names": ["Web_Agent_Send_1200_v2"],
                }
            )
            candidates = [
                {"task_name": "\\Web_Agent_Send_1200", "reason": "enabled_legacy_task"},
                {"task_name": "\\Web_Agent_Send_1200_v2", "reason": "offline_interactive_required"},
            ]

            with patch.dict(os.environ, {"WEB_AGENT_TEST_PASSWORD": ""}, clear=False):
                with patch("scheduler_runner.subprocess.run") as mocked_run:
                    mocked_run.return_value.returncode = 0
                    mocked_run.return_value.stdout = "SUCCESS"
                    mocked_run.return_value.stderr = ""
                    result = run_task_self_heal(scheduler_config, candidates, dry_run=False)

            self.assertFalse(result["success"])
            self.assertTrue(result["legacy_cleanup"]["success"])
            self.assertIn("requires a Windows password", result["message"])
            self.assertEqual(mocked_run.call_count, 2)
            self.assertEqual(mocked_run.call_args_list[-1].args[0][:4], ["schtasks", "/Delete", "/TN", "Web_Agent_Send_1200"])

    def test_export_scheduled_task_xml_writes_backup_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            backup_dir = Path(temp_dir) / "task_backups"

            with patch("scheduler_runner.subprocess.run") as mocked_run:
                mocked_run.return_value.returncode = 0
                mocked_run.return_value.stdout = "<Task></Task>"
                mocked_run.return_value.stderr = ""
                result = export_scheduled_task_xml("\\Web_Agent_Send_1200", backup_dir)

            self.assertTrue(result["success"])
            backup_path = Path(result["backup_path"])
            self.assertTrue(backup_path.exists())
            self.assertEqual(backup_path.read_text(encoding="utf-8"), "<Task></Task>")
            self.assertEqual(mocked_run.call_args.args[0][:4], ["schtasks", "/Query", "/TN", "Web_Agent_Send_1200"])

    def test_build_repair_plan_separates_legacy_cleanup_and_offline_rebuild(self):
        payload = {
            "overall": "warn",
            "status": {
                "legacy_tasks": [
                    {
                        "task_name": "\\Web_Agent_Send_1200",
                        "available": True,
                        "scheduled_task_state": "已启用",
                    }
                ],
                "tasks": [
                    {
                        "task_name": "\\Web_Agent_Send_1200_v2",
                        "available": True,
                        "logon_mode": "Interactive only",
                        "last_result_hint": "success",
                    }
                ],
            },
        }
        scheduler_config = dict(DEFAULT_SCHEDULER_CONFIG)
        scheduler_config.update(
            {
                "task_names": ["Web_Agent_Send_1200_v2"],
                "require_offline_tasks": True,
                "run_as_password_env": "WEB_AGENT_TEST_PASSWORD",
                "repair_task_script": "repair_scheduled_tasks.ps1",
            }
        )

        with patch.dict(os.environ, {"WEB_AGENT_TEST_PASSWORD": ""}, clear=False):
            plan = build_repair_plan(payload, scheduler_config)

        self.assertTrue(plan["legacy_cleanup_needed"])
        self.assertTrue(plan["offline_rebuild_needed"])
        self.assertFalse(plan["password_present"])
        self.assertTrue(any(item["name"] == "delete_legacy_tasks" for item in plan["action_items"]))
        cleanup = next(item for item in plan["action_items"] if item["name"] == "delete_legacy_tasks")
        self.assertIn("--cleanup-legacy-tasks", cleanup["command"])
        self.assertIn("--confirm", cleanup["command"])
        self.assertIn("task_backups", cleanup["backup_dir"])
        s4u_rebuild = next(item for item in plan["action_items"] if item["name"] == "rebuild_s4u_background_tasks")
        self.assertFalse(s4u_rebuild["blocked"])
        self.assertFalse(s4u_rebuild["requires_password"])
        self.assertIn("-UseS4U", s4u_rebuild["command"])
        rebuild = next(item for item in plan["action_items"] if item["name"] == "rebuild_offline_tasks")
        self.assertTrue(rebuild["blocked"])
        text = build_repair_plan_text(plan)
        self.assertIn("delete_legacy_tasks", text)
        self.assertIn("rebuild_s4u_background_tasks", text)
        self.assertIn("rebuild_offline_tasks", text)
        self.assertIn("WEB_AGENT_TEST_PASSWORD", text)

    def test_print_repair_plan_outputs_json(self):
        payload = {"overall": "ok", "status": {"tasks": [], "legacy_tasks": []}}
        scheduler_config = dict(DEFAULT_SCHEDULER_CONFIG)

        with patch("scheduler_runner.build_doctor_payload", return_value=payload):
            with patch("scheduler_runner.load_runtime_config", return_value=({}, scheduler_config)):
                with patch("sys.stdout.write") as mocked_write:
                    exit_code = print_repair_plan(as_json=True)

        self.assertEqual(exit_code, 0)
        output = "".join(str(call.args[0]) for call in mocked_write.call_args_list)
        self.assertIn('"candidate_count": 0', output)

    def test_legacy_cleanup_preview_requires_confirm_for_deletion(self):
        payload = {
            "status": {
                "legacy_tasks": [
                    {
                        "task_name": "\\Web_Agent_Send_1200",
                        "available": True,
                        "scheduled_task_state": "已启用",
                    }
                ],
                "tasks": [],
            }
        }
        scheduler_config = dict(DEFAULT_SCHEDULER_CONFIG)

        with patch("scheduler_runner.delete_scheduled_task") as mocked_delete:
            cleanup = build_legacy_cleanup_payload(payload, scheduler_config, confirm=False)

        self.assertEqual(cleanup["status"], "preview")
        self.assertEqual(cleanup["candidate_count"], 1)
        self.assertIn("--confirm", cleanup["message"])
        mocked_delete.assert_not_called()
        text = build_legacy_cleanup_text(cleanup)
        self.assertIn("Legacy cleanup status: preview", text)
        self.assertIn("\\Web_Agent_Send_1200", text)

    def test_legacy_cleanup_confirm_deletes_candidates(self):
        payload = {
            "status": {
                "legacy_tasks": [
                    {
                        "task_name": "\\Web_Agent_Send_1200",
                        "available": True,
                        "scheduled_task_state": "已启用",
                    }
                ],
                "tasks": [],
            }
        }
        scheduler_config = dict(DEFAULT_SCHEDULER_CONFIG)

        with patch(
            "scheduler_runner.delete_scheduled_task",
            return_value={"task_name": "Web_Agent_Send_1200", "success": True, "returncode": 0},
        ) as mocked_delete:
            cleanup = build_legacy_cleanup_payload(payload, scheduler_config, confirm=True)

        self.assertEqual(cleanup["status"], "completed")
        self.assertTrue(cleanup["success"])
        self.assertEqual(Path(cleanup["backup_dir"]), Path(DEFAULT_SCHEDULER_CONFIG["task_backup_dir"]))
        mocked_delete.assert_called_once()
        self.assertEqual(mocked_delete.call_args.args[0], "\\Web_Agent_Send_1200")
        self.assertIsInstance(mocked_delete.call_args.kwargs["backup_dir"], Path)

    def test_print_legacy_cleanup_report_outputs_preview(self):
        payload = {"overall": "ok", "status": {"tasks": [], "legacy_tasks": []}}
        scheduler_config = dict(DEFAULT_SCHEDULER_CONFIG)

        with patch("scheduler_runner.build_doctor_payload", return_value=payload):
            with patch("scheduler_runner.load_runtime_config", return_value=({}, scheduler_config)):
                with patch("sys.stdout.write") as mocked_write:
                    exit_code = print_legacy_cleanup_report(as_json=False, confirm=False)

        self.assertEqual(exit_code, 0)
        output = "".join(str(call.args[0]) for call in mocked_write.call_args_list)
        self.assertIn("Legacy cleanup status: preview", output)

    def test_build_task_backups_payload_lists_restore_commands(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            backup_dir = Path(temp_dir) / "task_backups"
            backup_dir.mkdir(parents=True)
            backup_path = backup_dir / "Web_Agent_Send_1200_20260506_223000.xml"
            backup_path.write_text("<Task></Task>", encoding="utf-8")
            scheduler_config = dict(DEFAULT_SCHEDULER_CONFIG)
            scheduler_config["task_backup_dir"] = str(backup_dir)

            payload = build_task_backups_payload(scheduler_config)
            text = build_task_backups_text(payload)

            self.assertEqual(payload["backup_count"], 1)
            self.assertEqual(payload["backups"][0]["task_name"], "Web_Agent_Send_1200")
            self.assertIn("--restore-task-backup", payload["backups"][0]["preview_command"])
            self.assertIn("--confirm", payload["backups"][0]["restore_command"])
            self.assertIn("/XML", payload["backups"][0]["raw_schtasks_restore_command"])
            self.assertIn("Web_Agent_Send_1200", text)
            self.assertIn("preview:", text)
            self.assertIn("restore:", text)
            self.assertIn("raw_schtasks_restore:", text)

    def test_print_task_backups_report_outputs_text(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            backup_dir = Path(temp_dir) / "task_backups"
            backup_dir.mkdir(parents=True)
            (backup_dir / "Web_Agent_Send_2100_20260506_223500.xml").write_text("<Task></Task>", encoding="utf-8")
            scheduler_config = dict(DEFAULT_SCHEDULER_CONFIG)
            scheduler_config["task_backup_dir"] = str(backup_dir)

            with patch("scheduler_runner.load_runtime_config", return_value=({}, scheduler_config)):
                with patch("sys.stdout.write") as mocked_write:
                    exit_code = print_task_backups_report(as_json=False)

            self.assertEqual(exit_code, 0)
            output = "".join(str(call.args[0]) for call in mocked_write.call_args_list)
            self.assertIn("Task backups: 1", output)
            self.assertIn("Web_Agent_Send_2100", output)

    def test_build_task_restore_payload_previews_by_default(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            backup_path = Path(temp_dir) / "Web_Agent_Send_1200_20260506_223000.xml"
            backup_path.write_text("<Task></Task>", encoding="utf-8")

            with patch("scheduler_runner.subprocess.run") as mocked_run:
                payload = build_task_restore_payload(str(backup_path), confirm=False)

            self.assertTrue(payload["success"])
            self.assertEqual(payload["status"], "preview")
            self.assertEqual(payload["task_name"], "Web_Agent_Send_1200")
            self.assertIn("/XML", payload["command"])
            self.assertIn("--confirm", payload["message"])
            mocked_run.assert_not_called()
            text = build_task_restore_text(payload)
            self.assertIn("Restore status: preview", text)
            self.assertIn(str(backup_path), text)

    def test_build_task_restore_payload_reports_missing_backup(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            missing_path = Path(temp_dir) / "Web_Agent_Send_1200_20260506_223000.xml"

            payload = build_task_restore_payload(str(missing_path), confirm=False)

            self.assertFalse(payload["success"])
            self.assertEqual(payload["status"], "missing_backup")
            self.assertIn("not found", payload["message"])

    def test_build_task_restore_payload_confirm_executes_schtasks_create(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            backup_path = Path(temp_dir) / "Web_Agent_Send_1200_20260506_223000.xml"
            backup_path.write_text("<Task></Task>", encoding="utf-8")

            with patch("scheduler_runner.subprocess.run") as mocked_run:
                mocked_run.return_value.returncode = 0
                mocked_run.return_value.stdout = "SUCCESS"
                mocked_run.return_value.stderr = ""
                payload = build_task_restore_payload(str(backup_path), confirm=True)

            self.assertTrue(payload["success"])
            self.assertEqual(payload["status"], "completed")
            self.assertEqual(mocked_run.call_args.args[0][:4], ["schtasks", "/Create", "/TN", "Web_Agent_Send_1200"])
            self.assertIn(str(backup_path), mocked_run.call_args.args[0])

    def test_print_task_restore_report_outputs_preview(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            backup_path = Path(temp_dir) / "Web_Agent_Send_1200_20260506_223000.xml"
            backup_path.write_text("<Task></Task>", encoding="utf-8")

            with patch("sys.stdout.write") as mocked_write:
                exit_code = print_task_restore_report(str(backup_path), as_json=False, confirm=False)

            self.assertEqual(exit_code, 0)
            output = "".join(str(call.args[0]) for call in mocked_write.call_args_list)
            self.assertIn("Restore status: preview", output)
            self.assertIn("--confirm", output)

    def test_run_task_self_heal_uses_offline_rebuild_for_primary_permission_failures(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            script_path = Path(temp_dir) / "setup_offline_tasks.ps1"
            script_path.write_text("Write-Host 'ok'", encoding="utf-8")
            scheduler_config = dict(DEFAULT_SCHEDULER_CONFIG)
            scheduler_config.update(
                {
                    "offline_task_setup_script": str(script_path),
                    "run_as_password_env": "WEB_AGENT_TEST_PASSWORD",
                    "require_offline_tasks": True,
                    "task_names": ["Web_Agent_Send_1200_v2"],
                }
            )
            candidates = [{"task_name": "\\Web_Agent_Send_1200_v2", "reason": "bad_last_result"}]

            with patch.dict(os.environ, {"WEB_AGENT_TEST_PASSWORD": ""}, clear=False):
                result = run_task_self_heal(scheduler_config, candidates, dry_run=False)

            self.assertTrue(result["attempted"])
            self.assertFalse(result["success"])
            self.assertIn("requires a Windows password", result["message"])

    def test_run_task_self_heal_redacts_offline_password_in_result_command(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            script_path = Path(temp_dir) / "setup_offline_tasks.ps1"
            script_path.write_text("Write-Host 'ok'", encoding="utf-8")
            scheduler_config = dict(DEFAULT_SCHEDULER_CONFIG)
            scheduler_config["offline_task_setup_script"] = str(script_path)
            scheduler_config["run_as_user_env"] = "WEB_AGENT_TEST_USER"
            scheduler_config["run_as_password_env"] = "WEB_AGENT_TEST_PASSWORD"
            candidates = [{"task_name": "Web_Agent_Send_1200_v2", "reason": "offline_interactive_required"}]

            with patch.dict(
                os.environ,
                {
                    "WEB_AGENT_TEST_USER": "DESKTOP\\user",
                    "WEB_AGENT_TEST_PASSWORD": "secret-password",
                },
                clear=False,
            ):
                with patch("scheduler_runner.subprocess.run") as mocked_run:
                    mocked_run.return_value.returncode = 0
                    mocked_run.return_value.stdout = "ok"
                    mocked_run.return_value.stderr = ""
                    result = run_task_self_heal(scheduler_config, candidates, dry_run=False)

            self.assertTrue(result["success"])
            self.assertIn("<redacted>", result["command"])
            self.assertNotIn("secret-password", result["command"])
            actual_command = mocked_run.call_args.args[0]
            self.assertIn("secret-password", actual_command)

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

    def test_print_doctor_report_rebuilds_payload_after_successful_self_heal(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            snapshot_path = Path(temp_dir) / "doctor_latest.json"
            history_path = Path(temp_dir) / "doctor_history.json"
            scheduler_config = dict(DEFAULT_SCHEDULER_CONFIG)
            scheduler_config["doctor_status_file"] = str(snapshot_path)
            scheduler_config["doctor_history_file"] = str(history_path)
            before_payload = {
                "overall": "warn",
                "counts": {"ok": 1, "warn": 1, "fail": 0},
                "checks": [{"name": "task", "level": "warn", "detail": "needs repair"}],
                "status": {"tasks": [{"task_name": "\\Web_Agent_Send_2100", "available": False, "error": "query failed"}]},
            }
            after_payload = {
                "overall": "ok",
                "counts": {"ok": 2, "warn": 0, "fail": 0},
                "checks": [{"name": "task", "level": "ok", "detail": "repaired"}],
                "status": {"tasks": [{"task_name": "\\Web_Agent_Send_2100", "available": True, "last_result_hint": "success"}]},
            }

            with patch("scheduler_runner.build_doctor_payload", side_effect=[before_payload, after_payload]):
                with patch("scheduler_runner.load_runtime_config", return_value=({}, scheduler_config)):
                    with patch("scheduler_runner.send_doctor_alert_email", return_value=False):
                        with patch(
                            "scheduler_runner.run_task_self_heal",
                            return_value={"attempted": True, "success": True, "message": "repaired"},
                        ):
                            with patch("sys.stdout.write"):
                                exit_code = print_doctor_report(
                                    as_json=False,
                                    self_heal=True,
                                    dry_run=False,
                                    persist_history=True,
                                )

            self.assertEqual(exit_code, 0)
            persisted = json.loads(snapshot_path.read_text(encoding="utf-8-sig"))
            self.assertEqual(persisted["overall"], "ok")
            self.assertEqual(persisted["pre_heal_summary"]["overall"], "warn")
            self.assertTrue(persisted["self_heal"]["success"])

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
            last_success_path = log_dir / "last_success.json"
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
                    "last_success_file": str(last_success_path),
                    "lock_file": str(lock_path),
                    "send_slot_dir": str(log_dir / "send_slots"),
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

    def test_main_records_last_success_snapshot_after_sent_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            log_dir = temp_root / "logs"
            status_path = log_dir / "last_run.json"
            last_success_path = log_dir / "last_success.json"
            lock_path = log_dir / "scheduler.lock"
            log_dir.mkdir(parents=True, exist_ok=True)

            scheduler_config = dict(DEFAULT_SCHEDULER_CONFIG)
            scheduler_config.update(
                {
                    "log_dir": str(log_dir),
                    "status_file": str(status_path),
                    "last_success_file": str(last_success_path),
                    "lock_file": str(lock_path),
                    "send_slot_dir": str(log_dir / "send_slots"),
                    "log_retention_days": 30,
                    "idempotency_window_minutes": 90,
                    "task_names": [],
                    "max_attempts": 2,
                    "send_window_after_minutes": 1440,
                }
            )
            attempt_result = {
                "success": True,
                "status": "sent_followup_timeout",
                "retryable": False,
                "delivery_status": "sent",
                "started_at": "2026-04-24T12:00:00",
                "finished_at": "2026-04-24T12:05:00",
                "run_id": "20260424_120000",
                "html_report_path": "archive/report_20260424_1205.html",
                "markdown_report_path": "archive/report_20260424_1205.md",
                "email_subject": "[2026-04-24 12:05] AI Frontier Intelligence Daily",
                "paper_count": 12,
                "update_count": 20,
            }

            with patch("scheduler_runner.load_runtime_config", return_value=({}, scheduler_config)):
                with patch("scheduler_runner.run_main_once", return_value=dict(attempt_result)) as mocked_run:
                    exit_code = scheduler_main()

            self.assertEqual(exit_code, 0)
            persisted = json.loads(last_success_path.read_text(encoding="utf-8-sig"))
            self.assertEqual(persisted["delivery_status"], "sent")
            self.assertEqual(persisted["html_report_path"], "archive/report_20260424_1205.html")
            self.assertEqual(persisted["email_subject"], "[2026-04-24 12:05] AI Frontier Intelligence Daily")
            mocked_run.assert_called_once()

    def test_main_does_not_retry_when_failed_worker_left_durable_sent_slot(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            log_dir = temp_root / "logs"
            slot_dir = log_dir / "send_slots"
            status_path = log_dir / "last_run.json"
            last_success_path = log_dir / "last_success.json"
            log_dir.mkdir(parents=True)
            scheduler_config = dict(DEFAULT_SCHEDULER_CONFIG)
            scheduler_config.update({
                "log_dir": str(log_dir),
                "status_file": str(status_path),
                "last_success_file": str(last_success_path),
                "validation_status_file": str(log_dir / "last_validation_run.json"),
                "lock_file": str(log_dir / "scheduler.lock"),
                "send_slot_dir": str(slot_dir),
                "send_calendar_dir": str(log_dir),
                "log_archive_dir": str(log_dir / "archive"),
                "task_backup_dir": str(log_dir / "task_backups"),
                "validation_report_dir": str(temp_root / "validation"),
                "log_retention_days": 30,
                "task_names": [],
                "max_attempts": 2,
                "retry_delay_seconds": 0,
            })
            failed = {
                "success": False,
                "status": "exception",
                "retryable": True,
                "delivery_status": "failed",
                "error": "worker exited before printing commit marker",
            }

            def fail_after_commit(*_args, **_kwargs):
                slot_path = slot_dir / "20260812_2100.json"
                payload = json.loads(slot_path.read_text(encoding="utf-8-sig"))
                payload.update({
                    "status": "sent",
                    "run_id": "run-committed",
                    "report_id": "report-committed",
                    "html_report_path": "archive/report-committed.html",
                    "finished_at": "2026-08-12T21:08:00",
                })
                write_json(slot_path, payload)
                return dict(failed)

            resolved_slot = {
                "allowed": True,
                "slot_id": "20260812_2100",
                "scheduled_at": "2026-08-12T21:00:00",
                "window_start": "2026-08-12T20:50:00",
                "window_end": "2026-08-13T00:00:00",
            }
            with patch("scheduler_runner.load_runtime_config", return_value=({}, scheduler_config)):
                with patch("scheduler_runner.resolve_send_slot", return_value=resolved_slot):
                    with patch("scheduler_runner.run_main_once", side_effect=fail_after_commit) as mocked_run:
                        with patch("scheduler_runner.refresh_report_quality_after_run", side_effect=lambda value: value):
                            exit_code = scheduler_main()

            latest = json.loads(status_path.read_text(encoding="utf-8-sig"))
            last_success = json.loads(last_success_path.read_text(encoding="utf-8-sig"))

        self.assertEqual(exit_code, 0)
        mocked_run.assert_called_once()
        self.assertEqual(latest["status"], "sent_commit_recovered")
        self.assertEqual(latest["delivery_status"], "sent")
        self.assertEqual(last_success["run_id"], "run-committed")

    def test_validate_run_uses_separate_status_file_and_bypasses_idempotency(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            log_dir = temp_root / "logs"
            status_path = log_dir / "last_run.json"
            last_success_path = log_dir / "last_success.json"
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
                    "last_success_file": str(last_success_path),
                    "validation_status_file": str(validation_status_path),
                    "lock_file": str(lock_path),
                    "send_slot_dir": str(log_dir / "send_slots"),
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

    def test_dry_run_uses_full_profile_without_touching_last_success(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            log_dir = temp_root / "logs"
            status_path = log_dir / "last_run.json"
            last_success_path = log_dir / "last_success.json"
            validation_status_path = log_dir / "last_validation_run.json"
            lock_path = log_dir / "scheduler.lock"
            log_dir.mkdir(parents=True, exist_ok=True)
            write_json(last_success_path, {"delivery_status": "sent", "html_report_path": "archive/production.html"})

            scheduler_config = dict(DEFAULT_SCHEDULER_CONFIG)
            scheduler_config.update(
                {
                    "log_dir": str(log_dir),
                    "status_file": str(status_path),
                    "last_success_file": str(last_success_path),
                    "validation_status_file": str(validation_status_path),
                    "lock_file": str(lock_path),
                    "send_slot_dir": str(log_dir / "send_slots"),
                    "log_retention_days": 30,
                    "task_names": [],
                    "max_attempts": 1,
                }
            )
            attempt_result = {
                "success": True,
                "status": "dry_run",
                "retryable": False,
                "delivery_status": "dry_run",
                "html_report_path": "archive/report_v8.html",
                "markdown_report_path": "archive/report_v8.md",
            }

            with patch("scheduler_runner.load_runtime_config", return_value=({}, scheduler_config)):
                with patch("scheduler_runner.run_main_once", return_value=dict(attempt_result)) as mocked_run:
                    exit_code = scheduler_main(dry_run=True)

            self.assertEqual(exit_code, 0)
            mocked_run.assert_called_once_with(
                int(scheduler_config["max_run_seconds"]),
                email_mode="dry-run",
                run_profile="",
            )
            latest = json.loads(validation_status_path.read_text(encoding="utf-8-sig"))
            self.assertEqual(latest["run_mode"], "dry-run")
            self.assertEqual(latest["delivery_status"], "dry_run")
            production = json.loads(last_success_path.read_text(encoding="utf-8-sig"))
            self.assertEqual(production["html_report_path"], "archive/production.html")

    def test_report_only_run_is_non_sending_and_uses_rerender_mode(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            log_dir = temp_root / "logs"
            validation_status_path = log_dir / "last_validation_run.json"
            last_success_path = log_dir / "last_success.json"
            log_dir.mkdir(parents=True, exist_ok=True)
            write_json(last_success_path, {"delivery_status": "sent", "html_report_path": "archive/production.html"})
            scheduler_config = dict(DEFAULT_SCHEDULER_CONFIG)
            scheduler_config.update(
                {
                    "log_dir": str(log_dir),
                    "status_file": str(log_dir / "last_run.json"),
                    "last_success_file": str(last_success_path),
                    "validation_status_file": str(validation_status_path),
                    "lock_file": str(log_dir / "scheduler.lock"),
                    "send_slot_dir": str(log_dir / "send_slots"),
                    "log_retention_days": 30,
                    "task_names": [],
                    "max_attempts": 1,
                }
            )
            attempt_result = {
                "success": True,
                "status": "dry_run",
                "retryable": False,
                "delivery_status": "dry_run",
                "html_report_path": "archive/report_rerender.html",
            }

            with patch("scheduler_runner.load_runtime_config", return_value=({}, scheduler_config)):
                with patch("scheduler_runner.run_main_once", return_value=dict(attempt_result)) as mocked_run:
                    exit_code = scheduler_main(report_only=True)

            self.assertEqual(exit_code, 0)
            mocked_run.assert_called_once_with(
                int(scheduler_config["max_run_seconds"]),
                email_mode="dry-run",
                run_profile="",
                report_only=True,
            )
            latest = json.loads(validation_status_path.read_text(encoding="utf-8-sig"))
            self.assertEqual(latest["run_mode"], "rerender")
            self.assertEqual(latest["delivery_status"], "dry_run")
            production = json.loads(last_success_path.read_text(encoding="utf-8-sig"))
            self.assertEqual(production["html_report_path"], "archive/production.html")


if __name__ == "__main__":
    unittest.main()
