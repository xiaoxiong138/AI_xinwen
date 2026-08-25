from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).with_name("v8_real_samples.json")
FIELDS = (
    "id",
    "title",
    "title_cn",
    "url",
    "canonical_url",
    "summary",
    "summary_preview",
    "content_type",
    "source_detail",
    "source_tier",
    "evidence_quality",
    "information_density",
    "facts",
    "category",
    "topic",
    "publish_date",
    "score",
    "model_used",
)


def parse_json(value: Any, fallback: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value or ""))
    except (TypeError, json.JSONDecodeError):
        return fallback


def select_samples(rows: List[sqlite3.Row], content_type: str, limit: int = 20) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        item = {field: row[field] for field in FIELDS}
        item["facts"] = parse_json(item.get("facts"), {})
        url = str(item.get("canonical_url") or item.get("url") or "")
        if item.get("content_type") != content_type or not url or url in seen or not item["facts"]:
            continue
        item["summary"] = str(item.get("summary") or "")[:900]
        item["summary_preview"] = str(item.get("summary_preview") or "")[:300]
        selected.append(item)
        seen.add(url)
        if len(selected) >= limit:
            break
    if len(selected) < limit:
        raise RuntimeError(f"Need {limit} {content_type} samples, found {len(selected)}")
    return selected


def main() -> None:
    conn = sqlite3.connect(ROOT / "ai_news.db")
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            f"SELECT {', '.join(FIELDS)} FROM articles WHERE processed = 1 ORDER BY created_at DESC, id DESC LIMIT 1000"
        ).fetchall()
    finally:
        conn.close()
    payload = {
        "description": "Fixed production samples for V8 editorial regression.",
        "papers": select_samples(rows, "paper"),
        "news": select_samples(rows, "news"),
    }
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(payload['papers'])} papers and {len(payload['news'])} news items to {OUTPUT}")


if __name__ == "__main__":
    main()
