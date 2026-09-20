import json

from tools.prepare_v11_editorial_review import _review_delivery_context


def test_review_delivery_context_exposes_real_volume_and_arrival_evidence():
    diagnostics = {
        "quality_gate": {
            "email_delivery_volume_count": 3,
            "email_delivery_volume_paths": [
                "archive/report_part1.html",
                "archive/report_part2.html",
                "archive/report_part3.html",
            ],
        },
        "ui_audit": {
            "status": "passed",
            "metrics_path": "artifacts/ui/metrics.json",
        },
    }
    report = {
        "delivery_status": "sent",
        "html_report_path": "archive/report.html",
        "quality_diagnostics": json.dumps(diagnostics),
    }
    subjects = ["新闻卷", "技术卷", "论文卷"]
    slot = {
        "status": "sent",
        "email_subjects": subjects,
        "email_volume_sent_count": 3,
        "delivery_verification": {
            "status": "found",
            "volumes": [
                {"matched_subject": subject, "status": "found"}
                for subject in subjects
            ],
        },
    }

    context = _review_delivery_context(report, slot)

    assert context == {
        "delivery_status": "sent",
        "html_report_path": "archive/report.html",
        "expected_volume_count": 3,
        "sent_volume_count": 3,
        "subjects": subjects,
        "archive_paths": [
            "archive/report_part1.html",
            "archive/report_part2.html",
            "archive/report_part3.html",
        ],
        "arrival_status": "found",
        "matched_subjects": subjects,
        "ui_audit_status": "passed",
        "ui_audit_metrics_path": "artifacts/ui/metrics.json",
    }
