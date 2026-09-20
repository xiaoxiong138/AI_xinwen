import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.editorial_engine import contains_mojibake
from src.ui_audit import run_email_ui_audit
from tools.generate_v11_ui_fixture import load_real_samples


def test_v11_ui_fixture_covers_real_news_technical_and_paper_copy():
    samples = load_real_samples()

    section_counts = {
        section: sum(item.get("primary_section") == section for item in samples)
        for section in ("news", "technical", "paper")
    }

    assert section_counts == {"news": 2, "technical": 1, "paper": 1}
    assert any(item.get("content_type") == "podcast" for item in samples)
    assert all(item.get("analysis_version") == "codex-research-v2" for item in samples)
    assert all(not contains_mojibake(item.get("title_cn")) for item in samples)
    assert all(not contains_mojibake(item.get("summary")) for item in samples)


def test_ui_audit_reports_layout_metric_summary():
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        report_path = root / "report.html"
        report_path.write_text("<html><body>report</body></html>", encoding="utf-8")
        script_path = root / "tools" / "email_ui_audit.js"
        script_path.parent.mkdir(parents=True)
        script_path.write_text("", encoding="utf-8")
        output_dir = root / "audit"

        def fake_run(*args, **kwargs):
            payload = {
                "passed": True,
                "reportCount": 1,
                "renderCount": 2,
                "failedRenderCount": 0,
                "rows": [
                    {
                        "mode": "desktop",
                        "passed": True,
                        "contentFontSizePx": 16,
                        "contentLineHeightRatio": 1.72,
                        "containerWidthPx": 720,
                        "contentPaddingLeftPx": 30,
                        "contentPaddingRightPx": 30,
                    },
                    {
                        "mode": "mobile",
                        "passed": True,
                        "contentFontSizePx": 15.5,
                        "contentLineHeightRatio": 1.68,
                        "containerWidthPx": 370,
                        "contentPaddingLeftPx": 18,
                        "contentPaddingRightPx": 18,
                    },
                ],
            }
            (output_dir / "metrics.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with patch(
            "src.ui_audit.resolve_ui_audit_node", return_value=("node", {})
        ), patch("src.ui_audit.subprocess.run", side_effect=fake_run):
            result = run_email_ui_audit(
                [report_path.as_posix()],
                output_dir=output_dir,
                root=root,
            )

    assert result["status"] == "passed"
    assert result["layout_metrics"] == {
        "minimum_content_font_size_px": 15.5,
        "minimum_line_height_ratio": 1.68,
        "minimum_mobile_content_padding_px": 18.0,
        "maximum_desktop_container_width_px": 720.0,
    }


def test_ui_audit_rejects_directory_instead_of_rendering_its_index():
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)

        result = run_email_ui_audit(
            [root.as_posix()],
            output_dir=root / "audit",
            root=root,
        )

    assert result["status"] == "error"
    assert result["failures"] == ["invalid_email_volume_paths"]
    assert result["invalid_paths"] == [root.as_posix()]
    assert result["error"] == "Email UI audit inputs must be HTML files."
