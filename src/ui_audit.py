from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable


def resolve_ui_audit_node() -> tuple[str, Dict[str, str]]:
    dependencies_root = Path.home() / ".cache" / "codex-runtimes" / "codex-primary-runtime" / "dependencies"
    configured_node = str(os.getenv("CODEX_NODE_EXE", "") or "").strip()
    bundled_node = dependencies_root / "node" / "bin" / "node.exe"
    node_exe = configured_node or (str(bundled_node) if bundled_node.exists() else shutil.which("node") or "")
    environment = dict(os.environ)
    if not environment.get("NODE_PATH"):
        bundled_modules = dependencies_root / "node" / "node_modules"
        if bundled_modules.exists():
            environment["NODE_PATH"] = str(bundled_modules)
    return node_exe, environment


def run_email_ui_audit(
    volume_paths: Iterable[str],
    *,
    output_dir: Path,
    root: Path,
    timeout_seconds: int = 180,
) -> Dict[str, Any]:
    resolved_paths = [
        path if path.is_absolute() else root / path
        for path in (Path(value) for value in volume_paths if str(value or "").strip())
    ]
    missing_paths = [path.as_posix() for path in resolved_paths if not path.exists()]
    invalid_paths = [
        path.as_posix()
        for path in resolved_paths
        if path.exists() and (not path.is_file() or path.suffix.lower() not in {".html", ".htm"})
    ]
    if not resolved_paths or missing_paths or invalid_paths:
        return {
            "status": "error",
            "passed": False,
            "report_count": len(resolved_paths),
            "render_count": 0,
            "failed_render_count": 0,
            "metrics_path": "",
            "output_dir": output_dir.as_posix(),
            "missing_paths": missing_paths,
            "invalid_paths": invalid_paths,
            "failures": [
                "invalid_email_volume_paths"
                if invalid_paths
                else "missing_email_volume_paths"
            ],
            "error": (
                "Email UI audit inputs must be HTML files."
                if invalid_paths
                else "One or more email delivery volumes are missing."
            ),
        }
    node_exe, environment = resolve_ui_audit_node()
    script_path = root / "tools" / "email_ui_audit.js"
    if not node_exe or not script_path.exists():
        return {
            "status": "error",
            "passed": False,
            "report_count": len(resolved_paths),
            "render_count": 0,
            "failed_render_count": 0,
            "metrics_path": "",
            "output_dir": output_dir.as_posix(),
            "missing_paths": [],
            "failures": ["ui_audit_runtime_missing"],
            "error": "Node.js or the email UI audit script is unavailable.",
        }
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        node_exe,
        str(script_path),
        "--output-dir",
        str(output_dir),
        *(str(path) for path in resolved_paths),
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=str(root),
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=max(30, int(timeout_seconds or 180)),
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "status": "error",
            "passed": False,
            "report_count": len(resolved_paths),
            "render_count": 0,
            "failed_render_count": 0,
            "metrics_path": "",
            "output_dir": output_dir.as_posix(),
            "missing_paths": [],
            "failures": ["ui_audit_execution_error"],
            "error": str(exc),
        }
    metrics_path = output_dir / "metrics.json"
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8-sig")) if metrics_path.exists() else {}
    except (OSError, json.JSONDecodeError):
        metrics = {}
    rows = list(metrics.get("rows") or [])
    content_font_sizes = [
        float(row.get("contentFontSizePx", 0) or 0)
        for row in rows
        if float(row.get("contentFontSizePx", 0) or 0) > 0
    ]
    line_height_ratios = [
        float(row.get("contentLineHeightRatio", 0) or 0)
        for row in rows
        if float(row.get("contentLineHeightRatio", 0) or 0) > 0
    ]
    mobile_paddings = [
        min(
            float(row.get("contentPaddingLeftPx", 0) or 0),
            float(row.get("contentPaddingRightPx", 0) or 0),
        )
        for row in rows
        if str(row.get("mode") or "").startswith("mobile")
    ]
    desktop_widths = [
        float(row.get("containerWidthPx", 0) or 0)
        for row in rows
        if str(row.get("mode") or "") == "desktop"
    ]
    layout_metrics = {
        "minimum_content_font_size_px": min(content_font_sizes) if content_font_sizes else 0.0,
        "minimum_line_height_ratio": min(line_height_ratios) if line_height_ratios else 0.0,
        "minimum_mobile_content_padding_px": min(mobile_paddings) if mobile_paddings else 0.0,
        "maximum_desktop_container_width_px": max(desktop_widths) if desktop_widths else 0.0,
    }
    failure_rows = [
        {
            "report_path": str(row.get("reportPath") or ""),
            "mode": str(row.get("mode") or ""),
            "failures": list(row.get("failures") or []),
            "missing_source_note_item_count": int(
                row.get("missingSourceNoteItemCount", 0) or 0
            ),
            "missing_claim_label_item_count": int(
                row.get("missingClaimLabelItemCount", 0) or 0
            ),
            "missing_source_note_item_samples": list(
                row.get("missingSourceNoteItemSamples") or []
            ),
            "missing_claim_label_item_samples": list(
                row.get("missingClaimLabelItemSamples") or []
            ),
            "screenshot_path": str(row.get("screenshotPath") or ""),
        }
        for row in rows
        if not bool(row.get("passed", False))
    ]
    passed = bool(metrics.get("passed", False)) and completed.returncode == 0
    return {
        "status": "passed" if passed else "failed",
        "passed": passed,
        "report_count": int(metrics.get("reportCount", len(resolved_paths)) or len(resolved_paths)),
        "render_count": int(metrics.get("renderCount", len(rows)) or len(rows)),
        "failed_render_count": int(metrics.get("failedRenderCount", len(failure_rows)) or len(failure_rows)),
        "metrics_path": metrics_path.as_posix() if metrics_path.exists() else "",
        "output_dir": output_dir.as_posix(),
        "layout_metrics": layout_metrics,
        "missing_paths": [],
        "failures": failure_rows,
        "error": "" if metrics else (completed.stderr or completed.stdout)[-1000:],
        "exit_code": int(completed.returncode),
    }
