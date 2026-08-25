from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, quote, urlparse

import yaml

from src.database import Database, resolve_database_path
from scheduler_runner import scan_report_quality

HEALTH_MARKER = "WEB_AGENT_FEEDBACK_OK"
ROOT = Path(__file__).resolve().parent


def normalize_model_path_breakdown(breakdown: dict) -> dict:
    normalized = {}
    for key, value in (breakdown or {}).items():
        normalized_key = "fallback_v2" if str(key) == "v2" else str(key)
        normalized[normalized_key] = normalized.get(normalized_key, 0) + int(value or 0)
    return normalized


def build_feedback_stats(db_path: str) -> dict:
    db = Database(db_path)
    latest_report = db.get_latest_report_run()
    quality_diagnostics = latest_report.get("quality_diagnostics") or {}
    content_quality = dict(quality_diagnostics.get("content_quality") or {})
    content_quality.setdefault("bad_title_count", 0)
    report_id = latest_report.get("report_id", "")
    quality_scan = scan_report_quality(report_id=report_id, limit=10, persist=False, db_path=db_path) if report_id else {}
    return {
        "feedback_count_7d": db.get_feedback_count(days=7),
        "feedback_count_30d": db.get_feedback_count(days=30),
        "recent_feedback": db.get_recent_feedback(limit=10),
        "reading_queue_count": len(db.get_reading_queue(statuses=["tracked"], limit=100)),
        "preference_weights": db.get_preference_weights(),
        "last_model_path_refresh": quality_diagnostics.get("last_model_path_refresh", {}),
        "quality_issues_top10": quality_scan.get("issues", []),
        "latest_report": {
            "report_id": latest_report.get("report_id", ""),
            "quality_status": latest_report.get("quality_status", ""),
            "html_report_path": latest_report.get("html_report_path", ""),
            "content_quality": content_quality,
            "title_repair": quality_diagnostics.get("title_repair", {}),
            "model_path_breakdown": normalize_model_path_breakdown(
                quality_diagnostics.get("model_path_breakdown", {})
            ),
        },
    }


def _file_link(path_text: str) -> str:
    path_text = str(path_text or "").strip()
    if not path_text:
        return ""
    path = Path(path_text)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve().as_uri()


def render_stats_html(stats: dict) -> str:
    weights = stats.get("preference_weights") or {}
    rows = []
    for key_type in sorted(weights):
        for key, weight in sorted((weights.get(key_type) or {}).items(), key=lambda item: str(item[0])):
            rows.append(
                "<tr>"
                f"<td>{html.escape(str(key_type))}</td>"
                f"<td>{html.escape(str(key))}</td>"
                f"<td>{float(weight):.3f}</td>"
                "</tr>"
            )
    if not rows:
        rows.append("<tr><td colspan='3'>No preference weights yet.</td></tr>")
    feedback_rows = []
    for item in stats.get("recent_feedback") or []:
        source = item.get("source_detail") or item.get("platform") or item.get("source") or ""
        topic = item.get("topic") or item.get("category") or ""
        report_link = _file_link(str(item.get("html_report_path", "") or ""))
        report_cell = html.escape(str(item.get("report_id", "") or ""))
        if report_link:
            report_cell = f"<a href='{html.escape(report_link)}'>{report_cell}</a>"
        feedback_rows.append(
            "<tr>"
            f"<td>{html.escape(str(item.get('created_at', '') or ''))}</td>"
            f"<td>{html.escape(str(item.get('signal', '') or ''))}</td>"
            f"<td>{report_cell}</td>"
            f"<td>{html.escape(str(item.get('article_id', '') or ''))}</td>"
            f"<td>{html.escape(str(item.get('title', '') or ''))}</td>"
            f"<td>{html.escape(str(source))}</td>"
            f"<td>{html.escape(str(topic))}</td>"
            "</tr>"
        )
    if not feedback_rows:
        feedback_rows.append("<tr><td colspan='7'>No feedback records yet.</td></tr>")
    issue_rows = []
    for item in stats.get("quality_issues_top10") or []:
        issue_rows.append(
            "<tr>"
            f"<td>{html.escape(str(item.get('rank', '') or ''))}</td>"
            f"<td>{html.escape(str(item.get('section', '') or ''))}</td>"
            f"<td>{html.escape(str(item.get('article_id', '') or ''))}</td>"
            f"<td>{html.escape(str(item.get('title', '') or ''))}</td>"
            f"<td>{html.escape(', '.join(item.get('issue_types') or []))}</td>"
            "</tr>"
        )
    if not issue_rows:
        issue_rows.append("<tr><td colspan='5'>No quality issues in the latest report snapshot.</td></tr>")
    title_repair_rows = []
    for item in ((stats.get("latest_report") or {}).get("title_repair") or {}).get("examples") or []:
        title_repair_rows.append(
            "<tr>"
            f"<td>{html.escape(str(item.get('rank', '') or ''))}</td>"
            f"<td>{html.escape(str(item.get('section', '') or ''))}</td>"
            f"<td>{html.escape(str(item.get('article_id', '') or ''))}</td>"
            f"<td>{html.escape(str(item.get('source', '') or ''))}</td>"
            f"<td>{html.escape(str(item.get('old_title', '') or ''))}</td>"
            f"<td>{html.escape(str(item.get('new_title', '') or ''))}</td>"
            "</tr>"
        )
    if not title_repair_rows:
        title_repair_rows.append("<tr><td colspan='6'>No title repairs recorded for the latest report.</td></tr>")
    latest = stats.get("latest_report") or {}
    last_refresh = stats.get("last_model_path_refresh") or {}
    model_path = json.dumps(latest.get("model_path_breakdown") or {}, ensure_ascii=False, sort_keys=True)
    content_quality = json.dumps(latest.get("content_quality") or {}, ensure_ascii=False, sort_keys=True)
    title_repair = latest.get("title_repair") or {}
    refresh_text = json.dumps(last_refresh, ensure_ascii=False, sort_keys=True) if last_refresh else "none"
    return (
        "<html><body style='font-family:Segoe UI,Arial,sans-serif;max-width:860px;margin:32px auto;'>"
        "<h2>AI Daily Feedback Stats</h2>"
        f"<p><strong>7d feedback:</strong> {int(stats.get('feedback_count_7d', 0) or 0)}</p>"
        f"<p><strong>30d feedback:</strong> {int(stats.get('feedback_count_30d', 0) or 0)}</p>"
        "<h3>Latest report</h3>"
        f"<p><strong>Report:</strong> {html.escape(str(latest.get('report_id', '') or ''))}</p>"
        f"<p><strong>Quality:</strong> {html.escape(str(latest.get('quality_status', '') or ''))}</p>"
        f"<p><strong>Content quality:</strong> {html.escape(content_quality)}</p>"
        f"<p><strong>Title repair:</strong> repaired={int(title_repair.get('bad_title_repaired_count', 0) or 0)}, unresolved={int(title_repair.get('bad_title_unresolved_count', 0) or 0)}</p>"
        f"<p><strong>Model path:</strong> {html.escape(model_path)}</p>"
        f"<p><strong>Last model refresh:</strong> {html.escape(refresh_text)}</p>"
        "<h3>Preference weights</h3>"
        "<table border='1' cellspacing='0' cellpadding='6'>"
        "<tr><th>Type</th><th>Key</th><th>Weight</th></tr>"
        + "".join(rows)
        + "</table>"
        "<h3>Quality issues Top 10</h3>"
        "<table border='1' cellspacing='0' cellpadding='6'>"
        "<tr><th>Rank</th><th>Section</th><th>Article</th><th>Title</th><th>Issues</th></tr>"
        + "".join(issue_rows)
        + "</table>"
        "<h3>Title repairs Top 10</h3>"
        "<table border='1' cellspacing='0' cellpadding='6'>"
        "<tr><th>Rank</th><th>Section</th><th>Article</th><th>Source</th><th>Old title</th><th>New title</th></tr>"
        + "".join(title_repair_rows)
        + "</table>"
        "<h3>Recent feedback</h3>"
        "<table border='1' cellspacing='0' cellpadding='6'>"
        "<tr><th>Time</th><th>Signal</th><th>Report</th><th>Article</th><th>Title</th><th>Source</th><th>Topic</th></tr>"
        + "".join(feedback_rows)
        + "</table>"
        "<p style='color:#666'>Local only. Data is calculated from recent email feedback.</p>"
        "</body></html>"
    )


def render_feedback_detail_html(
    db_path: str,
    report_id: str,
    article_id: int,
    recorded_signal: str = "",
) -> str:
    db = Database(db_path)
    item = db.get_report_item(report_id, article_id) or db.get_article_by_id(article_id) or {}
    title = str(item.get("editorial_title") or item.get("title_cn") or item.get("title") or "未命名条目")
    body = str(
        item.get("paper_technical_intro")
        or item.get("analysis_body")
        or item.get("summary")
        or item.get("content")
        or ""
    ).strip()
    source = str(item.get("source_detail") or item.get("platform") or item.get("source") or "")
    original_url = str(item.get("canonical_url") or item.get("url") or "").strip()
    base = f"/feedback?report_id={quote(str(report_id))}&item_id={int(article_id)}"
    signals = [
        ("not_useful", "没用"),
        ("mute_similar", "屏蔽类似"),
        ("too_long", "内容太长"),
        ("not_memorable", "没记住重点"),
    ]
    if str(item.get("content_type", "")) == "paper":
        signals.insert(2, ("paper_too_shallow", "论文太浅"))
    else:
        signals.insert(2, ("too_shallow", "内容太浅"))
    controls = "".join(
        f"<a href='{base}&signal={signal}&origin=detail'>{html.escape(label)}</a>"
        for signal, label in signals
    )
    notice = ""
    if recorded_signal:
        notice = (
            "<div class='notice'>反馈已记录，后续排序会参考最近 30 天的偏好。</div>"
        )
    original = (
        f"<a class='original' href='{html.escape(original_url)}'>打开原文</a>"
        if original_url
        else ""
    )
    return (
        "<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>条目反馈</title><style>"
        "body{margin:0;background:#f4f5f2;color:#1f2926;font-family:Segoe UI,Microsoft YaHei,sans-serif;}"
        ".page{max-width:720px;margin:0 auto;padding:36px 20px 60px;}"
        ".eyebrow{font-size:12px;color:#64716d;margin-bottom:10px;}"
        "h1{font-size:26px;line-height:1.35;margin:0 0 16px;}"
        ".body{font-size:16px;line-height:1.8;color:#36413e;margin-bottom:22px;}"
        ".notice{padding:12px 14px;background:#e5f1e9;border-left:3px solid #28704a;margin-bottom:20px;}"
        ".actions{display:flex;flex-wrap:wrap;gap:10px;margin:14px 0 24px;}"
        ".actions a{padding:9px 12px;border:1px solid #c8ceca;color:#263f36;text-decoration:none;background:#fff;}"
        ".original{color:#28604d;text-decoration:none;font-weight:600;}"
        "</style></head><body><main class='page'>"
        f"{notice}<div class='eyebrow'>{html.escape(source)} · 报告 {html.escape(str(report_id))}</div>"
        f"<h1>{html.escape(title)}</h1>"
        f"<div class='body'>{html.escape(body)}</div>"
        "<div class='eyebrow'>进一步反馈</div>"
        f"<div class='actions'>{controls}</div>{original}"
        "</main></body></html>"
    )


def render_reading_queue_html(db_path: str, notice: str = "") -> str:
    queue = Database(db_path).get_reading_queue(statuses=["tracked", "read"], limit=50)
    rows = []
    for item in queue:
        snapshot = item.get("snapshot") or {}
        title = str(snapshot.get("editorial_title") or snapshot.get("title_cn") or snapshot.get("title") or "未命名条目")
        url = str(snapshot.get("canonical_url") or snapshot.get("url") or "")
        report_id = str(item.get("report_id") or "")
        article_id = int(item.get("article_id") or 0)
        status = str(item.get("status") or "tracked")
        action_status = "read" if status == "tracked" else "tracked"
        action_label = "标为已读" if status == "tracked" else "重新跟进"
        action = (
            f"/queue/action?report_id={quote(report_id)}&item_id={article_id}&status={action_status}"
        )
        original = f"<a href='{html.escape(url)}'>打开原文</a>" if url else ""
        rows.append(
            "<div class='row'>"
            f"<div class='meta'>{html.escape(str(item.get('topic_key') or ''))} · {html.escape(status)}</div>"
            f"<h2>{html.escape(title)}</h2>"
            f"<div class='actions'>{original}<a href='{action}'>{action_label}</a>"
            f"<a href='/queue/action?report_id={quote(report_id)}&item_id={article_id}&status=archived'>归档</a></div>"
            "</div>"
        )
    if not rows:
        rows.append("<div class='row'>阅读队列为空。在邮件中点击“跟进”后，条目会出现在这里。</div>")
    notice_html = f"<div class='notice'>{html.escape(notice)}</div>" if notice else ""
    return (
        "<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>我的跟进队列</title><style>"
        "body{margin:0;background:#f4f5f2;color:#1f2926;font-family:Segoe UI,Microsoft YaHei,sans-serif;}"
        ".page{max-width:760px;margin:0 auto;padding:34px 20px 60px;}"
        ".row{padding:18px 0;border-bottom:1px solid #d8dedb;}"
        ".meta{color:#6a7571;font-size:12px}.row h2{font-size:18px;line-height:1.45;margin:6px 0 10px;}"
        ".actions a{display:inline-block;margin-right:14px;color:#28604d;text-decoration:none;font-weight:650;}"
        ".notice{padding:10px 12px;background:#e5f1e9;border-left:3px solid #28704a;margin:12px 0;}"
        "</style></head><body><main class='page'><h1>我的跟进队列</h1>"
        f"{notice_html}{''.join(rows)}</main></body></html>"
    )


class FeedbackHandler(BaseHTTPRequestHandler):
    db_path = "ai_news.db"

    def _send_html(self, status: int, body: str) -> None:
        encoded = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_json(self, status: int, payload: dict) -> None:
        encoded = json.dumps(payload, ensure_ascii=False, indent=2, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_health(self) -> None:
        self._send_json(200, {"status": "ok", "marker": HEALTH_MARKER})

    def do_HEAD(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/health":
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("X-Health-Marker", HEALTH_MARKER)
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path not in {"/", "/health", "/feedback", "/detail", "/stats", "/stats.json", "/queue", "/queue/action"}:
            self._send_html(404, "<h3>Not found</h3>")
            return
        if parsed.path == "/health":
            self._send_health()
            return
        if parsed.path == "/":
            self._send_html(200, f"<h3>AI Daily feedback service is running</h3><p>{HEALTH_MARKER}</p>")
            return
        if parsed.path == "/stats":
            self._send_html(200, render_stats_html(build_feedback_stats(self.db_path)))
            return
        if parsed.path == "/stats.json":
            self._send_json(200, build_feedback_stats(self.db_path))
            return
        if parsed.path == "/queue":
            self._send_html(200, render_reading_queue_html(self.db_path))
            return

        params = parse_qs(parsed.query)
        report_id = (params.get("report_id") or [""])[0].strip()
        item_id = (params.get("item_id") or [""])[0].strip()
        if parsed.path == "/queue/action":
            status = (params.get("status") or [""])[0].strip()
            if not report_id or not item_id or status not in {"tracked", "read", "archived"}:
                self._send_html(400, "<h3>Reading queue parameters are incomplete</h3>")
                return
            try:
                Database(self.db_path).upsert_reading_queue(report_id, int(item_id), status=status)
            except Exception as exc:
                self._send_html(400, f"<h3>Failed to update reading queue</h3><p>{html.escape(str(exc))}</p>")
                return
            self._send_html(200, render_reading_queue_html(self.db_path, notice="阅读状态已更新。"))
            return
        if parsed.path == "/detail":
            if not report_id or not item_id:
                self._send_html(400, "<h3>Detail parameters are incomplete</h3>")
                return
            self._send_html(
                200,
                render_feedback_detail_html(self.db_path, report_id, int(item_id)),
            )
            return
        signal = (params.get("signal") or [""])[0].strip()
        if not report_id or not item_id or not signal:
            self._send_html(400, "<h3>Feedback parameters are incomplete</h3>")
            return

        try:
            result = Database(self.db_path).record_article_feedback(
                report_id=report_id,
                article_id=int(item_id),
                signal=signal,
                source="email",
            )
        except Exception as exc:
            self._send_html(400, f"<h3>Failed to record feedback</h3><p>{html.escape(str(exc))}</p>")
            return

        self._send_html(
            200,
            render_feedback_detail_html(
                self.db_path,
                report_id,
                int(item_id),
                recorded_signal=signal,
            ),
        )

    def log_message(self, format: str, *args) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="Local feedback server for AI daily report emails.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18765)
    parser.add_argument("--db", default="")
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parent
    config_path = project_root / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    FeedbackHandler.db_path = resolve_database_path(
        {**config, "database": {"path": args.db}} if args.db else config,
        project_root,
    )
    server = ThreadingHTTPServer((args.host, args.port), FeedbackHandler)
    print(f"Feedback server listening on http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
