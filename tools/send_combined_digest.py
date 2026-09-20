from __future__ import annotations

import argparse
import html
import json
import os
import sqlite3
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import yaml
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.notifier import EmailNotifier, resolve_imap_server, verify_email_arrival


SECTION_LABELS = {
    "news": "新闻、博客与访谈",
    "technical": "技术方法与工程实践",
}
CLAIM_LABELS = {
    "verified_fact": "已核验事实",
    "official_claim": "官方披露",
    "interview_opinion": "访谈观点",
    "analysis": "分析判断",
    "research_result": "研究结果",
}


def _database_path() -> Path:
    configured = str(os.getenv("WEB_AGENT_DATABASE_PATH") or "").strip()
    if configured:
        return Path(configured)
    local_app_data = Path(os.getenv("LOCALAPPDATA") or (Path.home() / "AppData" / "Local"))
    return local_app_data / "Web_Agent" / "data" / "ai_news.db"


def _load_report(
    database_path: Path,
    report_id: str,
) -> tuple[dict, list[dict]]:
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    try:
        if report_id:
            report = connection.execute(
                "SELECT * FROM report_runs WHERE report_id = ?",
                (report_id,),
            ).fetchone()
        else:
            report = connection.execute(
                """
                SELECT * FROM report_runs
                WHERE quality_status = 'passed'
                ORDER BY created_at DESC
                LIMIT 1
                """
            ).fetchone()
        if report is None:
            raise RuntimeError("No matching quality-passed report snapshot was found.")
        report_data = dict(report)
        if str(report_data.get("quality_status") or "") != "passed":
            raise RuntimeError(
                f"Report {report_data.get('report_id')} did not pass the quality gate."
            )
        rows = connection.execute(
            """
            SELECT rank, section, snapshot_json
            FROM report_items
            WHERE report_id = ?
            ORDER BY rank, id
            """,
            (report_data["report_id"],),
        ).fetchall()
    finally:
        connection.close()

    items = []
    for row in rows:
        snapshot = json.loads(row["snapshot_json"] or "{}")
        snapshot["_snapshot_rank"] = int(row["rank"] or 0)
        items.append(snapshot)
    return report_data, items


def _primary_section(item: dict) -> str:
    facts = item.get("facts") if isinstance(item.get("facts"), dict) else {}
    return str(facts.get("primary_section") or item.get("primary_section") or "").strip()


def _text(value: object) -> str:
    return " ".join(str(value or "").split())


def _paragraphs(value: object) -> str:
    raw = str(value or "").strip()
    parts = [part.strip() for part in raw.split("\n\n") if part.strip()]
    if not parts and raw:
        parts = [raw]
    return "".join(f"<p>{html.escape(part)}</p>" for part in parts)


def _render_card(item: dict, index: int) -> str:
    title = _text(item.get("editorial_title") or item.get("title_cn") or item.get("title"))
    source = _text(item.get("source_detail") or item.get("platform") or "原始来源")
    publish_date = _text(item.get("publish_date"))[:10]
    body = item.get("analysis_body") or item.get("summary") or item.get("editorial_lead")
    evidence = _text(item.get("evidence_line"))
    url = str(item.get("canonical_url") or item.get("url") or "").strip()
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise RuntimeError(f"Invalid source URL in report snapshot: {url}")
    facts = item.get("facts") if isinstance(item.get("facts"), dict) else {}
    claim_type = str(item.get("claim_type") or facts.get("claim_type") or "").strip()
    claim_label = CLAIM_LABELS.get(claim_type, "来源材料")
    supplemental = "supplemental_older_source" in set(item.get("quality_flags") or [])
    source_meta = " · ".join(part for part in (source, publish_date) if part)
    older_label = '<span class="older">补充阅读</span>' if supplemental else ""
    evidence_html = ""
    if evidence and evidence not in _text(body):
        evidence_html = f'<p class="evidence"><b>证据：</b>{html.escape(evidence)}</p>'
    return f"""
    <article class="item v10-entry">
      <div class="index">{index:02d}</div>
      <div class="copy">
        <div class="meta"><span class="v11-source-note">{html.escape(source_meta)}</span> · <span class="v11-claim-label">{html.escape(claim_label)}</span> {older_label}</div>
        <h3 class="v10-title">{html.escape(title)}</h3>
        <div class="body v10-body">{_paragraphs(body)}</div>
        {evidence_html}
        <a class="source" href="{html.escape(url, quote=True)}">阅读原文</a>
      </div>
    </article>
    """


def build_combined_html(report: dict, items: list[dict], sections: list[str]) -> str:
    selected = [item for item in items if _primary_section(item) in sections]
    selected.sort(key=lambda item: int(item.get("_snapshot_rank") or 0))
    seen_urls: set[str] = set()
    for item in selected:
        url = str(item.get("canonical_url") or item.get("url") or "").strip()
        if not url or url in seen_urls:
            raise RuntimeError(f"Duplicate or missing URL in combined digest: {url}")
        seen_urls.add(url)

    counts = Counter(_primary_section(item) for item in selected)
    if counts["news"] < 20 or counts["technical"] < 20:
        raise RuntimeError(
            f"Combined digest is underfilled: news={counts['news']}, technical={counts['technical']}"
        )

    section_html = []
    item_number = 1
    for section in sections:
        section_items = [item for item in selected if _primary_section(item) == section]
        cards = []
        for item in section_items:
            cards.append(_render_card(item, item_number))
            item_number += 1
        section_html.append(
            f'<section><h2>{SECTION_LABELS[section]} <span>{len(section_items)} 条</span></h2>'
            + "".join(cards)
            + "</section>"
        )

    source_report = html.escape(str(report.get("report_id") or ""))
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AI 前沿晚间合并日报</title>
<style>
body{{margin:0;background:#f2f5f4;color:#17211f;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",Arial,sans-serif;line-height:1.75}}
.wrap,.container{{max-width:720px;margin:0 auto;background:#fff}}header{{padding:34px 34px 28px;background:#173f38;color:#fff}}
.eyebrow{{font-size:12px;letter-spacing:.08em;color:#b9d8cf}}h1{{margin:8px 0 10px;font-size:30px;line-height:1.25}}
.deck{{margin:0;color:#d8e8e3;font-size:15px}}.counts{{display:flex;gap:12px;margin-top:22px}}
.count{{border-left:3px solid #77c5ae;padding-left:10px;font-size:14px}}.count b{{display:block;font-size:23px;line-height:1.1}}
main{{padding:10px 34px 36px}}section{{padding-top:24px}}h2{{margin:0 0 4px;padding-bottom:10px;border-bottom:2px solid #c9d9d4;font-size:22px}}
h2 span{{font-size:13px;color:#60736e;font-weight:500}}.item{{display:flex;gap:14px;padding:22px 0;border-bottom:1px solid #dce5e2}}
.index{{flex:0 0 34px;color:#377d6d;font-size:13px;font-weight:700}}.copy{{min-width:0;flex:1}}.meta{{font-size:12px;color:#63736f}}
.older{{margin-left:6px;padding:2px 5px;background:#eef3f1;color:#536762}}h3{{margin:5px 0 10px;font-size:18px;line-height:1.45}}
.v10-body{{font-size:16px;line-height:1.75}}.body p{{margin:0 0 9px;font-size:16px}}.evidence{{margin:9px 0!important;padding:8px 10px;background:#f3f7f5;color:#354b45;font-size:13px!important}}
.source{{display:inline-block;margin-top:2px;color:#176d59;font-size:13px;font-weight:700;text-decoration:none}}footer{{padding:22px 34px;background:#edf3f1;color:#61716d;font-size:11px}}
@media(max-width:560px){{header{{padding:26px 20px 22px}}main{{padding:8px 20px 28px}}h1{{font-size:25px}}.counts{{display:block}}.count{{margin-top:9px}}.item{{gap:8px}}.index{{flex-basis:27px}}h3{{font-size:17px}}}}
</style></head><body><div class="wrap container">
<header><div class="eyebrow">AI FRONTIER INTELLIGENCE · EVENING EDITION</div><h1>今晚的新闻与技术，一封读完</h1>
<p class="deck">从通过质量闸门的晚间快照中合并整理；论文已移除，正文、证据与原文入口完整保留。</p>
<div class="counts"><div class="count"><b>{counts['news']}</b>新闻 / 博客 / 访谈</div><div class="count"><b>{counts['technical']}</b>技术方法 / 工程实践</div></div></header>
<main class="content">{''.join(section_html)}</main>
<footer>生成时间 {generated_at} · 来源报告 {source_report} · 共 {len(selected)} 条，未包含论文栏目。</footer>
</div></body></html>"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Send selected V11 sections as one compact email.")
    parser.add_argument("--report-id", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8")) or {}
    report, items = _load_report(_database_path(), args.report_id)
    sections = ["news", "technical"]
    html_content = build_combined_html(report, items, sections)
    maximum_bytes = int(config.get("report", {}).get("email_html_max_bytes", 104448) or 104448)
    size_bytes = len(html_content.encode("utf-8"))
    if size_bytes > maximum_bytes:
        raise RuntimeError(f"Combined email is too large: {size_bytes}/{maximum_bytes} bytes")

    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_path = ROOT / "archive" / f"combined_news_technical_{stamp}.html"
    output_path.write_text(html_content, encoding="utf-8")
    subject = f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}] AI 前沿晚间日报｜新闻 + 技术合并版"
    result = {
        "report_id": report["report_id"],
        "output_path": output_path.as_posix(),
        "size_bytes": size_bytes,
        "subject": subject,
        "sent": False,
        "arrival": {"status": "not_checked"},
    }
    if args.dry_run:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    recipient = str(os.getenv("EMAIL_RECIPIENT") or "").strip()
    sender = str(os.getenv("EMAIL_SENDER") or "").strip()
    password = str(os.getenv("EMAIL_PASSWORD") or "").strip()
    if not recipient or not sender or not password:
        raise RuntimeError("Email credentials are incomplete.")
    notifier = EmailNotifier(
        smtp_server=os.getenv("EMAIL_SMTP_SERVER", "smtp.gmail.com"),
        smtp_port=int(os.getenv("EMAIL_SMTP_PORT", "587")),
        sender_email=sender,
        sender_password=password,
        timeout_seconds=int(config.get("email", {}).get("timeout_seconds", 30) or 30),
        max_attempts=int(config.get("email", {}).get("max_attempts", 3) or 3),
        retry_delay_seconds=int(config.get("email", {}).get("retry_delay_seconds", 5) or 5),
    )
    result["sent"] = bool(notifier.send_email(recipient, subject, html_content))
    if not result["sent"]:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 1

    arrival_config = dict(config.get("email", {}).get("arrival_check", {}) or {})
    if arrival_config.get("enabled", False):
        result["arrival"] = verify_email_arrival(
            imap_server=resolve_imap_server(
                os.getenv("EMAIL_SMTP_SERVER", "smtp.gmail.com"),
                str(arrival_config.get("imap_server") or ""),
            ),
            imap_port=int(arrival_config.get("imap_port", 993) or 993),
            username=sender,
            password=password,
            subject_contains=subject,
            since_minutes=int(arrival_config.get("since_minutes", 30) or 30),
            mailbox=arrival_config.get("mailboxes") or arrival_config.get("mailbox") or "INBOX",
            timeout_seconds=int(arrival_config.get("timeout_seconds", 25) or 25),
            expected_sender=sender,
            retry_attempts=int(arrival_config.get("retry_attempts", 3) or 3),
            retry_delay_seconds=int(arrival_config.get("retry_delay_seconds", 8) or 8),
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["arrival"].get("status") in {"found", "not_checked"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
