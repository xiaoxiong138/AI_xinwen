from __future__ import annotations

import datetime
import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional


def resolve_database_path(
    config: Optional[Dict[str, Any]] = None,
    root: Optional[Path] = None,
    default: str = "ai_news.db",
) -> str:
    configured = str(os.getenv("WEB_AGENT_DB_PATH", "") or "").strip()
    if not configured and config:
        configured = str((config.get("database") or {}).get("path", "") or "").strip()
    path = Path(configured or default).expanduser()
    if not path.is_absolute():
        path = (root or Path.cwd()) / path
    return str(path.resolve())


class Database:
    ANALYSIS_COLUMNS = {
        "title_cn": "TEXT",
        "summary_preview": "TEXT",
        "why_it_matters": "TEXT",
        "why_now": "TEXT",
        "expected_effect": "TEXT",
        "future_impact": "TEXT",
        "facts": "TEXT",
        "evidence_quality": "REAL DEFAULT 0.0",
        "information_density": "REAL DEFAULT 0.0",
        "model_used": "TEXT",
        "canonical_url": "TEXT",
        "source_tier": "TEXT",
        "analysis_version": "TEXT",
        "quality_flags": "TEXT",
        "rewrite_attempts": "INTEGER DEFAULT 0",
        "arxiv_version": "INTEGER DEFAULT 1",
    }
    VALID_FEEDBACK_SIGNALS = {
        "useful",
        "not_useful",
        "track",
        "mute_similar",
        "too_shallow",
        "paper_too_shallow",
        "paper_unclear",
        "too_long",
        "too_generic",
        "source_suspicious",
        "not_memorable",
    }

    @staticmethod
    def _normalize_delivery_at_utc(value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        try:
            parsed = datetime.datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return ""
        if parsed.tzinfo is None:
            parsed = parsed.astimezone()
        parsed = parsed.astimezone(datetime.timezone.utc).replace(tzinfo=None)
        return parsed.strftime("%Y-%m-%d %H:%M:%S")

    def __init__(self, db_path: str = "ai_news.db"):
        self.db_path = db_path
        self._init_db()

    def _get_conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                url TEXT UNIQUE NOT NULL,
                source TEXT NOT NULL,
                source_detail TEXT,
                content TEXT,
                summary TEXT,
                score REAL DEFAULT 0.0,
                publish_date TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                processed INTEGER DEFAULT 0,
                keywords TEXT,
                category TEXT,
                author TEXT,
                content_type TEXT DEFAULT 'news',
                platform TEXT,
                topic TEXT,
                run_id TEXT
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS collector_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                label TEXT NOT NULL,
                status TEXT NOT NULL,
                inserted_count INTEGER DEFAULT 0,
                collected_count INTEGER DEFAULT 0,
                duration_seconds REAL DEFAULT 0,
                error TEXT,
                diagnostics TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS paper_enrichment_cache (
                url TEXT PRIMARY KEY,
                abstract TEXT,
                author TEXT,
                publish_date TEXT,
                refreshed_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS report_runs (
                report_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                slot_id TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                html_report_path TEXT,
                markdown_report_path TEXT,
                quality_status TEXT,
                quality_diagnostics TEXT,
                delivery_status TEXT,
                delivery_at TEXT
            )
            """
        )
        cursor.execute("PRAGMA table_info(report_runs)")
        report_run_columns = {row[1] for row in cursor.fetchall()}
        if "delivery_status" not in report_run_columns:
            cursor.execute("ALTER TABLE report_runs ADD COLUMN delivery_status TEXT")
        if "delivery_at" not in report_run_columns:
            cursor.execute("ALTER TABLE report_runs ADD COLUMN delivery_at TEXT")
        cursor.execute("PRAGMA table_info(collector_runs)")
        collector_run_columns = {row[1] for row in cursor.fetchall()}
        if "diagnostics" not in collector_run_columns:
            cursor.execute("ALTER TABLE collector_runs ADD COLUMN diagnostics TEXT")
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS report_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                report_id TEXT NOT NULL,
                article_id INTEGER,
                rank INTEGER DEFAULT 0,
                section TEXT,
                snapshot_json TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS article_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                report_id TEXT NOT NULL,
                article_id INTEGER,
                signal TEXT NOT NULL,
                source TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS preference_weights (
                key_type TEXT NOT NULL,
                key TEXT NOT NULL,
                weight REAL DEFAULT 0.0,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (key_type, key)
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS topic_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                report_id TEXT NOT NULL,
                article_id INTEGER,
                topic_key TEXT NOT NULL,
                entity_key TEXT,
                event_fingerprint TEXT NOT NULL,
                facts_json TEXT,
                editorial_conclusion TEXT,
                delta_summary TEXT,
                confidence_label TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (report_id, article_id)
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS topic_dossiers (
                topic_key TEXT PRIMARY KEY,
                label TEXT NOT NULL,
                domain_key TEXT,
                current_routes_json TEXT,
                representative_items_json TEXT,
                recent_changes_json TEXT,
                open_questions_json TEXT,
                recommended_reading_json TEXT,
                last_report_id TEXT,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS reading_queue (
                article_id INTEGER PRIMARY KEY,
                report_id TEXT NOT NULL,
                topic_key TEXT,
                status TEXT NOT NULL DEFAULT 'tracked',
                snapshot_json TEXT,
                tracked_at TEXT DEFAULT CURRENT_TIMESTAMP,
                last_seen_at TEXT DEFAULT CURRENT_TIMESTAMP,
                last_change_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_topic_events_topic_created ON topic_events(topic_key, created_at DESC)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_reading_queue_status ON reading_queue(status, tracked_at DESC)")
        existing_columns = {row["name"] for row in cursor.execute("PRAGMA table_info(articles)").fetchall()}
        for column_name, column_type in self.ANALYSIS_COLUMNS.items():
            if column_name not in existing_columns:
                cursor.execute(f"ALTER TABLE articles ADD COLUMN {column_name} {column_type}")
        conn.commit()
        conn.close()

    def article_exists(self, url: str) -> bool:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM articles WHERE url = ?", (url,))
        exists = cursor.fetchone() is not None
        conn.close()
        return exists

    def insert_article(self, article: Dict[str, Any]) -> bool:
        if self.article_exists(article["url"]):
            return False
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO articles (
                    title, url, source, source_detail, content, publish_date,
                    author, content_type, platform, topic, run_id, score, facts, evidence_quality, information_density,
                    model_used, canonical_url, source_tier, analysis_version, quality_flags, rewrite_attempts,
                    arxiv_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    article.get("title", ""),
                    article["url"],
                    article.get("source", ""),
                    article.get("source_detail", ""),
                    article.get("content", ""),
                    article.get("publish_date", datetime.datetime.now().isoformat()),
                    article.get("author", ""),
                    article.get("content_type", "news"),
                    article.get("platform", ""),
                    article.get("topic", ""),
                    article.get("run_id", ""),
                    float(article.get("initial_score", article.get("score", 0.0)) or 0.0),
                    self._serialize_facts(article.get("facts", {})),
                    float(article.get("evidence_quality", 0.0) or 0.0),
                    float(article.get("information_density", 0.0) or 0.0),
                    article.get("model_used", ""),
                    article.get("canonical_url", article.get("url", "")),
                    article.get("source_tier", ""),
                    article.get("analysis_version", ""),
                    self._serialize_facts(article.get("quality_flags", [])),
                    int(article.get("rewrite_attempts", 0) or 0),
                    int(article.get("arxiv_version", 1) or 1),
                ),
            )
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False
        except Exception as exc:
            print(f"Error inserting article {article.get('title')}: {exc}")
            return False
        finally:
            conn.close()

    def update_article_source_snapshot(self, url: str, article: Dict[str, Any]) -> bool:
        """Refresh source metadata for a newly verified copy of an existing URL."""
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE articles
                SET title = ?, source = ?, source_detail = ?, content = ?, publish_date = ?,
                    author = ?, content_type = ?, platform = ?, topic = ?, run_id = ?,
                    canonical_url = ?, source_tier = ?
                WHERE url = ?
                """,
                (
                    article.get("title", ""),
                    article.get("source", ""),
                    article.get("source_detail", ""),
                    article.get("content", ""),
                    article.get("publish_date", ""),
                    article.get("author", ""),
                    article.get("content_type", "news"),
                    article.get("platform", ""),
                    article.get("topic", ""),
                    article.get("run_id", ""),
                    article.get("canonical_url", article.get("url", "")),
                    article.get("source_tier", ""),
                    url,
                ),
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    def get_unprocessed_articles(self, run_id: Optional[str] = None) -> List[Dict[str, Any]]:
        conn = self._get_conn()
        cursor = conn.cursor()
        if run_id:
            cursor.execute("SELECT * FROM articles WHERE processed = 0 AND run_id = ? ORDER BY created_at DESC", (run_id,))
        else:
            cursor.execute("SELECT * FROM articles WHERE processed = 0 ORDER BY created_at DESC")
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def update_article_processing(
        self,
        url: str,
        summary: str,
        score: float,
        keywords: List[str],
        category: str,
        title_cn: str = "",
        summary_preview: str = "",
        why_it_matters: str = "",
        why_now: str = "",
        expected_effect: str = "",
        future_impact: str = "",
        facts: Optional[Dict[str, Any]] = None,
        evidence_quality: float = 0.0,
        information_density: float = 0.0,
        model_used: str = "",
        analysis_version: str = "",
        quality_flags: Optional[List[str]] = None,
        rewrite_attempts: Optional[int] = None,
    ):
        conn = self._get_conn()
        cursor = conn.cursor()
        keywords_str = ",".join(keywords) if isinstance(keywords, list) else str(keywords)
        cursor.execute(
            """
            UPDATE articles
            SET summary = ?, score = ?, keywords = ?, category = ?, processed = 1,
                title_cn = ?, summary_preview = ?, why_it_matters = ?, why_now = ?, expected_effect = ?, future_impact = ?,
                facts = ?, evidence_quality = ?, information_density = ?, model_used = ?,
                analysis_version = COALESCE(NULLIF(?, ''), analysis_version),
                quality_flags = ?,
                rewrite_attempts = COALESCE(?, rewrite_attempts)
            WHERE url = ?
            """,
            (
                summary,
                score,
                keywords_str,
                category,
                title_cn,
                summary_preview,
                why_it_matters,
                why_now,
                expected_effect,
                future_impact,
                self._serialize_facts(facts or {}),
                float(evidence_quality or 0.0),
                float(information_density or 0.0),
                model_used,
                analysis_version,
                self._serialize_facts(quality_flags or []),
                rewrite_attempts,
                url,
            ),
        )
        conn.commit()
        conn.close()

    def _serialize_facts(self, facts: Any) -> str:
        if not facts:
            return ""
        try:
            return json.dumps(facts, ensure_ascii=False, sort_keys=True)
        except TypeError:
            return json.dumps({"raw": str(facts)}, ensure_ascii=False, sort_keys=True)

    def _deserialize_article_row(self, row: sqlite3.Row) -> Dict[str, Any]:
        item = dict(row)
        raw_facts = item.get("facts")
        if isinstance(raw_facts, str) and raw_facts.strip():
            try:
                item["facts"] = json.loads(raw_facts)
            except json.JSONDecodeError:
                item["facts"] = {}
        else:
            item["facts"] = {}
        item["evidence_quality"] = float(item.get("evidence_quality", 0.0) or 0.0)
        item["information_density"] = float(item.get("information_density", 0.0) or 0.0)
        item["arxiv_version"] = int(item.get("arxiv_version", 1) or 1)
        raw_quality_flags = item.get("quality_flags")
        if isinstance(raw_quality_flags, str) and raw_quality_flags.strip():
            try:
                item["quality_flags"] = json.loads(raw_quality_flags)
            except json.JSONDecodeError:
                item["quality_flags"] = []
        else:
            item["quality_flags"] = []
        return item

    def get_articles_for_run(self, run_id: str, processed_only: bool = True) -> List[Dict[str, Any]]:
        conn = self._get_conn()
        cursor = conn.cursor()
        if processed_only:
            cursor.execute("SELECT * FROM articles WHERE run_id = ? AND processed = 1 ORDER BY score DESC, publish_date DESC", (run_id,))
        else:
            cursor.execute("SELECT * FROM articles WHERE run_id = ? ORDER BY created_at DESC", (run_id,))
        rows = cursor.fetchall()
        conn.close()
        return [self._deserialize_article_row(row) for row in rows]

    def get_articles_by_urls(self, urls: List[str]) -> List[Dict[str, Any]]:
        normalized_urls = list(dict.fromkeys(str(url or "").strip() for url in urls if str(url or "").strip()))
        if not normalized_urls:
            return []
        conn = self._get_conn()
        cursor = conn.cursor()
        placeholders = ",".join("?" for _ in normalized_urls)
        cursor.execute(
            f"SELECT * FROM articles WHERE url IN ({placeholders}) AND processed = 1 ORDER BY score DESC, publish_date DESC",
            normalized_urls,
        )
        rows = cursor.fetchall()
        conn.close()
        return [self._deserialize_article_row(row) for row in rows]

    def get_top_articles_today(self, limit: int = 100) -> List[Dict[str, Any]]:
        conn = self._get_conn()
        cursor = conn.cursor()
        yesterday = (datetime.datetime.now() - datetime.timedelta(hours=24)).strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute(
            "SELECT * FROM articles WHERE created_at >= ? AND processed = 1 ORDER BY score DESC LIMIT ?",
            (yesterday, limit),
        )
        rows = cursor.fetchall()
        conn.close()
        return [self._deserialize_article_row(row) for row in rows]

    def get_recent_processed_articles(
        self,
        hours: int = 24,
        content_type: Optional[str] = None,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        conn = self._get_conn()
        cursor = conn.cursor()
        utc_now = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
        since = (utc_now - datetime.timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")
        if content_type == "paper":
            cursor.execute(
                """
                SELECT * FROM articles
                WHERE datetime(publish_date) >= datetime(?)
                  AND processed = 1 AND content_type = 'paper'
                ORDER BY score DESC, publish_date DESC
                LIMIT ?
                """,
                (since, limit),
            )
        elif content_type:
            cursor.execute(
                """
                SELECT * FROM articles
                WHERE created_at >= ? AND processed = 1 AND content_type = ?
                ORDER BY score DESC, publish_date DESC
                LIMIT ?
                """,
                (since, content_type, limit),
            )
        else:
            cursor.execute(
                """
                SELECT * FROM articles
                WHERE created_at >= ? AND processed = 1
                ORDER BY score DESC, publish_date DESC
                LIMIT ?
                """,
                (since, limit),
            )
        rows = cursor.fetchall()
        conn.close()
        return [self._deserialize_article_row(row) for row in rows]

    def get_recent_articles_missing_analysis(self, hours: int = 24, limit: int = 200) -> List[Dict[str, Any]]:
        conn = self._get_conn()
        cursor = conn.cursor()
        since = (datetime.datetime.now() - datetime.timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute(
            """
            SELECT * FROM articles
            WHERE created_at >= ? AND processed = 1
              AND (
                    title_cn IS NULL OR title_cn = ''
                 OR summary_preview IS NULL OR summary_preview = ''
                 OR why_it_matters IS NULL OR why_it_matters = ''
                 OR why_now IS NULL OR why_now = ''
                 OR expected_effect IS NULL OR expected_effect = ''
                 OR future_impact IS NULL OR future_impact = ''
                 OR facts IS NULL OR facts = ''
                 OR evidence_quality IS NULL OR evidence_quality < 0.35
                 OR information_density IS NULL OR information_density < 0.35
                 OR model_used IS NULL OR model_used = ''
              )
            ORDER BY score DESC, publish_date DESC
            LIMIT ?
            """,
            (since, limit),
        )
        rows = cursor.fetchall()
        conn.close()
        return [self._deserialize_article_row(row) for row in rows]

    def record_collector_runs(self, run_id: str, collector_runs: List[Dict[str, Any]]) -> None:
        if not collector_runs:
            return
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.executemany(
            """
            INSERT INTO collector_runs (
                run_id, label, status, inserted_count, collected_count, duration_seconds, error, diagnostics
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    run_id,
                    str(item.get("label", "")),
                    str(item.get("status", "")),
                    int(item.get("inserted_count", 0) or 0),
                    int(item.get("collected_count", 0) or 0),
                    float(item.get("duration_seconds", 0) or 0),
                    str(item.get("error", "") or ""),
                    self._serialize_facts(item.get("diagnostics", {})),
                )
                for item in collector_runs
            ],
        )
        conn.commit()
        conn.close()

    def get_recent_collector_runs(self, label_prefix: str = "", limit: int = 20) -> List[Dict[str, Any]]:
        conn = self._get_conn()
        cursor = conn.cursor()
        if label_prefix:
            cursor.execute(
                """
                SELECT * FROM collector_runs
                WHERE label LIKE ?
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                (f"{label_prefix}%", limit),
            )
        else:
            cursor.execute(
                """
                SELECT * FROM collector_runs
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                (limit,),
            )
        rows = cursor.fetchall()
        conn.close()
        result = []
        for row in rows:
            item = dict(row)
            try:
                item["diagnostics"] = json.loads(item.get("diagnostics") or "{}")
            except (TypeError, json.JSONDecodeError):
                item["diagnostics"] = {}
            result.append(item)
        return result

    def get_recent_processed_articles_since(self, hours: int = 72, limit: int = 500) -> List[Dict[str, Any]]:
        conn = self._get_conn()
        cursor = conn.cursor()
        since = (datetime.datetime.now() - datetime.timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute(
            """
            SELECT * FROM articles
            WHERE created_at >= ? AND processed = 1
            ORDER BY created_at DESC, score DESC
            LIMIT ?
            """,
            (since, limit),
        )
        rows = cursor.fetchall()
        conn.close()
        return [self._deserialize_article_row(row) for row in rows]

    def get_cached_paper_enrichment(self, urls: List[str], max_age_hours: int = 168) -> Dict[str, Dict[str, Any]]:
        if not urls:
            return {}
        conn = self._get_conn()
        cursor = conn.cursor()
        placeholders = ",".join("?" for _ in urls)
        since = (datetime.datetime.now() - datetime.timedelta(hours=max_age_hours)).strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute(
            f"""
            SELECT * FROM paper_enrichment_cache
            WHERE url IN ({placeholders}) AND refreshed_at >= ?
            """,
            [*urls, since],
        )
        rows = cursor.fetchall()
        conn.close()
        return {row["url"]: dict(row) for row in rows}

    def upsert_paper_enrichment_cache(self, items: List[Dict[str, Any]]) -> None:
        if not items:
            return
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.executemany(
            """
            INSERT INTO paper_enrichment_cache (url, abstract, author, publish_date, refreshed_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(url) DO UPDATE SET
                abstract = excluded.abstract,
                author = excluded.author,
                publish_date = excluded.publish_date,
                refreshed_at = CURRENT_TIMESTAMP
            """,
            [
                (
                    str(item.get("url", "")),
                    str(item.get("content", "") or ""),
                    str(item.get("author", "") or ""),
                    str(item.get("publish_date", "") or ""),
                )
                for item in items
                if item.get("url")
            ],
        )
        conn.commit()
        conn.close()

    def get_article_by_id(self, article_id: int) -> Optional[Dict[str, Any]]:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM articles WHERE id = ?", (int(article_id),))
        row = cursor.fetchone()
        conn.close()
        return self._deserialize_article_row(row) if row else None

    def record_report_run(
        self,
        report_id: str,
        run_id: str,
        slot_id: str = "",
        html_report_path: str = "",
        markdown_report_path: str = "",
        quality_status: str = "",
        quality_diagnostics: Optional[Dict[str, Any]] = None,
        delivery_status: str = "",
    ) -> None:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO report_runs (
                report_id, run_id, slot_id, html_report_path, markdown_report_path, quality_status, quality_diagnostics,
                delivery_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(report_id) DO UPDATE SET
                run_id = excluded.run_id,
                slot_id = excluded.slot_id,
                html_report_path = excluded.html_report_path,
                markdown_report_path = excluded.markdown_report_path,
                quality_status = excluded.quality_status,
                quality_diagnostics = excluded.quality_diagnostics,
                delivery_status = excluded.delivery_status
            """,
            (
                report_id,
                run_id,
                slot_id,
                html_report_path,
                markdown_report_path,
                quality_status,
                self._serialize_facts(quality_diagnostics or {}),
                str(delivery_status or ""),
            ),
        )
        conn.commit()
        conn.close()

    def update_report_delivery_status(
        self,
        report_id: str,
        delivery_status: str,
        delivery_at: str = "",
    ) -> None:
        conn = self._get_conn()
        cursor = conn.cursor()
        normalized_status = str(delivery_status or "")
        normalized_delivery_at = self._normalize_delivery_at_utc(delivery_at)
        if normalized_status == "sent":
            cursor.execute(
                """UPDATE report_runs
                   SET delivery_status = ?, delivery_at = COALESCE(NULLIF(?, ''), delivery_at, CURRENT_TIMESTAMP)
                   WHERE report_id = ?""",
                (normalized_status, normalized_delivery_at, str(report_id or "")),
            )
        else:
            cursor.execute(
                "UPDATE report_runs SET delivery_status = ? WHERE report_id = ?",
                (normalized_status, str(report_id or "")),
            )
        conn.commit()
        conn.close()

    def mark_report_runs_sent(self, sent_records: List[Dict[str, Any]]) -> int:
        normalized = []
        for record in sent_records:
            run_id = str(record.get("run_id", "") or "").strip()
            html_path = str(record.get("html_report_path", "") or "").replace("\\", "/").strip()
            delivery_at = self._normalize_delivery_at_utc(record.get("delivery_at", ""))
            if run_id or html_path:
                normalized.append((run_id, html_path, delivery_at))
        if not normalized:
            return 0
        conn = self._get_conn()
        cursor = conn.cursor()
        updated = 0
        for run_id, html_path, delivery_at in normalized:
            row_count = 0
            if run_id:
                cursor.execute(
                    """UPDATE report_runs
                       SET delivery_status = 'sent',
                           delivery_at = COALESCE(delivery_at, NULLIF(?, ''), CURRENT_TIMESTAMP)
                       WHERE run_id = ?
                         AND (COALESCE(delivery_status, '') = '' OR (delivery_status = 'sent' AND delivery_at IS NULL))""",
                    (delivery_at, run_id),
                )
                row_count = max(0, int(cursor.rowcount or 0))
            if not row_count and html_path:
                cursor.execute(
                    """UPDATE report_runs
                       SET delivery_status = 'sent',
                           delivery_at = COALESCE(delivery_at, NULLIF(?, ''), CURRENT_TIMESTAMP)
                       WHERE REPLACE(COALESCE(html_report_path, ''), '\\', '/') = ?
                         AND (COALESCE(delivery_status, '') = '' OR (delivery_status = 'sent' AND delivery_at IS NULL))""",
                    (delivery_at, html_path),
                )
                row_count = max(0, int(cursor.rowcount or 0))
            updated += row_count
        conn.commit()
        conn.close()
        return updated

    def record_report_items(self, report_id: str, items: List[Dict[str, Any]]) -> None:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM report_items WHERE report_id = ?", (report_id,))
        cursor.executemany(
            """
            INSERT INTO report_items (report_id, article_id, rank, section, snapshot_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (
                    report_id,
                    item.get("id"),
                    int(item.get("report_rank", index) or index),
                    str(item.get("report_section", "") or ""),
                    json.dumps(item, ensure_ascii=False, sort_keys=True, default=str),
                )
                for index, item in enumerate(items, start=1)
            ],
        )
        conn.commit()
        conn.close()

    def get_latest_report_run(self, exclude_validation: bool = False) -> Dict[str, Any]:
        conn = self._get_conn()
        cursor = conn.cursor()
        if exclude_validation:
            cursor.execute(
                """SELECT * FROM report_runs
                   WHERE REPLACE(COALESCE(html_report_path, ''), '\\', '/') NOT LIKE '%/validation/%'
                   ORDER BY created_at DESC, rowid DESC LIMIT 1"""
            )
        else:
            cursor.execute("SELECT * FROM report_runs ORDER BY created_at DESC, rowid DESC LIMIT 1")
        row = cursor.fetchone()
        conn.close()
        return self._deserialize_report_run(row)

    def get_latest_report_run_with_collection(self, limit: int = 100) -> Dict[str, Any]:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            """SELECT * FROM report_runs
               WHERE REPLACE(COALESCE(html_report_path, ''), '\\', '/') NOT LIKE '%/validation/%'
               ORDER BY created_at DESC, rowid DESC LIMIT ?""",
            (max(1, int(limit or 100)),),
        )
        rows = cursor.fetchall()
        conn.close()
        for row in rows:
            report = self._deserialize_report_run(row)
            collection = dict((report.get("quality_diagnostics") or {}).get("collection") or {})
            collector_count = sum(
                int(collection.get(key, 0) or 0)
                for key in (
                    "collector_success_count",
                    "collector_empty_count",
                    "collector_failed_count",
                    "collector_timeout_count",
                    "collector_skipped_count",
                )
            )
            if collector_count > 0:
                return report
        return {}

    def get_report_run(self, report_id: str) -> Dict[str, Any]:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM report_runs WHERE report_id = ?", (str(report_id or ""),))
        row = cursor.fetchone()
        conn.close()
        return self._deserialize_report_run(row)

    def get_report_item(self, report_id: str, article_id: int) -> Dict[str, Any]:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT rank, section, snapshot_json
            FROM report_items
            WHERE report_id = ? AND article_id = ?
            ORDER BY rank
            LIMIT 1
            """,
            (str(report_id or ""), int(article_id)),
        )
        row = cursor.fetchone()
        conn.close()
        if not row:
            return {}
        try:
            item = json.loads(row["snapshot_json"] or "{}")
        except (TypeError, json.JSONDecodeError):
            item = {}
        item.setdefault("report_rank", row["rank"])
        item.setdefault("report_section", row["section"])
        return item

    def get_recent_report_items(
        self,
        days: int = 14,
        limit: int = 600,
        exclude_report_id: str = "",
        sent_only: bool = False,
        content_type: str = "",
    ) -> List[Dict[str, Any]]:
        conn = self._get_conn()
        cursor = conn.cursor()
        # SQLite CURRENT_TIMESTAMP is UTC, so the cooldown boundary must use UTC too.
        utc_now = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
        since = (utc_now - datetime.timedelta(days=max(1, int(days)))).strftime("%Y-%m-%d %H:%M:%S")
        params: List[Any] = [since]
        exclusion = ""
        if exclude_report_id:
            exclusion = "AND i.report_id != ?"
            params.append(str(exclude_report_id))
        content_filter = ""
        normalized_content_type = str(content_type or "").strip()
        if normalized_content_type:
            content_filter = "AND json_extract(i.snapshot_json, '$.content_type') = ?"
            params.append(normalized_content_type)
        sent_filter = (
            """AND COALESCE(r.slot_id, '') NOT IN ('', '__report_only__', '__dry_run__')
               AND r.delivery_status = 'sent'
               AND REPLACE(COALESCE(r.html_report_path, ''), '\\', '/') NOT LIKE '%/validation/%'"""
            if sent_only
            else ""
        )
        params.append(max(1, int(limit)))
        cursor.execute(
            f"""
            SELECT i.report_id, i.article_id, i.rank, i.section, i.snapshot_json, i.created_at,
                   r.delivery_at
            FROM report_items i
            LEFT JOIN report_runs r ON r.report_id = i.report_id
            WHERE COALESCE(r.delivery_at, i.created_at) >= ? {exclusion} {sent_filter} {content_filter}
            ORDER BY COALESCE(r.delivery_at, i.created_at) DESC, i.rank ASC
            LIMIT ?
            """,
            params,
        )
        rows = cursor.fetchall()
        conn.close()
        result: List[Dict[str, Any]] = []
        for row in rows:
            try:
                item = json.loads(row["snapshot_json"] or "{}")
            except (TypeError, json.JSONDecodeError):
                item = {}
            item.setdefault("id", row["article_id"])
            item["_history_report_id"] = row["report_id"]
            item["_history_created_at"] = row["created_at"]
            item["_history_delivery_at"] = row["delivery_at"] or ""
            item.setdefault("report_rank", row["rank"])
            item.setdefault("report_section", row["section"])
            result.append(item)
        return result

    def get_latest_sent_report_items(
        self,
        content_type: str = "",
        exclude_report_id: str = "",
    ) -> List[Dict[str, Any]]:
        conn = self._get_conn()
        cursor = conn.cursor()
        params: List[Any] = []
        exclusion = ""
        if exclude_report_id:
            exclusion = "AND report_id != ?"
            params.append(str(exclude_report_id))
        cursor.execute(
            f"""
            SELECT report_id
            FROM report_runs
            WHERE delivery_status = 'sent'
              AND COALESCE(slot_id, '') NOT IN ('', '__report_only__', '__dry_run__')
              AND REPLACE(COALESCE(html_report_path, ''), '\\', '/') NOT LIKE '%/validation/%'
              {exclusion}
            ORDER BY COALESCE(delivery_at, created_at) DESC, rowid DESC
            LIMIT 1
            """,
            params,
        )
        latest = cursor.fetchone()
        if not latest:
            conn.close()
            return []
        report_id = str(latest["report_id"] or "")
        item_params: List[Any] = [report_id]
        content_filter = ""
        normalized_content_type = str(content_type or "").strip()
        if normalized_content_type:
            content_filter = "AND json_extract(snapshot_json, '$.content_type') = ?"
            item_params.append(normalized_content_type)
        cursor.execute(
            f"""
            SELECT i.article_id, i.rank, i.section, i.snapshot_json, i.created_at, r.delivery_at
            FROM report_items i
            JOIN report_runs r ON r.report_id = i.report_id
            WHERE i.report_id = ? {content_filter.replace('snapshot_json', 'i.snapshot_json')}
            ORDER BY i.rank ASC
            """,
            item_params,
        )
        rows = cursor.fetchall()
        conn.close()
        result: List[Dict[str, Any]] = []
        for row in rows:
            try:
                item = json.loads(row["snapshot_json"] or "{}")
            except (TypeError, json.JSONDecodeError):
                item = {}
            item.setdefault("id", row["article_id"])
            item["_history_report_id"] = report_id
            item["_history_created_at"] = row["created_at"]
            item["_history_delivery_at"] = row["delivery_at"] or ""
            item.setdefault("report_rank", row["rank"])
            item.setdefault("report_section", row["section"])
            result.append(item)
        return result

    def record_topic_events(self, report_id: str, items: List[Dict[str, Any]]) -> None:
        rows = []
        for item in items:
            article_id = item.get("id")
            topic_key = str(item.get("continuity_topic_key") or item.get("domain_key") or "").strip()
            fingerprint = str(item.get("event_fingerprint") or "").strip()
            if not article_id or not topic_key or not fingerprint:
                continue
            facts = item.get("facts_cn") if isinstance(item.get("facts_cn"), dict) else item.get("facts")
            rows.append(
                (
                    str(report_id),
                    int(article_id),
                    topic_key,
                    str(item.get("continuity_entity_key") or ""),
                    fingerprint,
                    self._serialize_facts(facts or {}),
                    str(item.get("editorial_lead") or ""),
                    str(item.get("delta_summary") or ""),
                    str(item.get("confidence_label") or ""),
                )
            )
        if not rows:
            return
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.executemany(
            """
            INSERT INTO topic_events (
                report_id, article_id, topic_key, entity_key, event_fingerprint,
                facts_json, editorial_conclusion, delta_summary, confidence_label
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(report_id, article_id) DO UPDATE SET
                topic_key = excluded.topic_key,
                entity_key = excluded.entity_key,
                event_fingerprint = excluded.event_fingerprint,
                facts_json = excluded.facts_json,
                editorial_conclusion = excluded.editorial_conclusion,
                delta_summary = excluded.delta_summary,
                confidence_label = excluded.confidence_label
            """,
            rows,
        )
        conn.commit()
        conn.close()

    def upsert_topic_dossiers(self, report_id: str, dossiers: List[Dict[str, Any]]) -> None:
        rows = []
        for dossier in dossiers:
            topic_key = str(dossier.get("topic_key") or "").strip()
            if not topic_key:
                continue
            rows.append(
                (
                    topic_key,
                    str(dossier.get("label") or topic_key),
                    str(dossier.get("domain_key") or ""),
                    self._serialize_facts(dossier.get("current_routes") or []),
                    self._serialize_facts(dossier.get("representative_items") or []),
                    self._serialize_facts(dossier.get("recent_changes") or []),
                    self._serialize_facts(dossier.get("open_questions") or []),
                    self._serialize_facts(dossier.get("recommended_reading") or []),
                    str(report_id),
                )
            )
        if not rows:
            return
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.executemany(
            """
            INSERT INTO topic_dossiers (
                topic_key, label, domain_key, current_routes_json, representative_items_json,
                recent_changes_json, open_questions_json, recommended_reading_json,
                last_report_id, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(topic_key) DO UPDATE SET
                label = excluded.label,
                domain_key = excluded.domain_key,
                current_routes_json = excluded.current_routes_json,
                representative_items_json = excluded.representative_items_json,
                recent_changes_json = excluded.recent_changes_json,
                open_questions_json = excluded.open_questions_json,
                recommended_reading_json = excluded.recommended_reading_json,
                last_report_id = excluded.last_report_id,
                updated_at = CURRENT_TIMESTAMP
            """,
            rows,
        )
        conn.commit()
        conn.close()

    def get_topic_dossiers(self, limit: int = 10) -> List[Dict[str, Any]]:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM topic_dossiers ORDER BY updated_at DESC LIMIT ?", (max(1, int(limit)),))
        rows = cursor.fetchall()
        conn.close()
        result: List[Dict[str, Any]] = []
        json_columns = {
            "current_routes_json": "current_routes",
            "representative_items_json": "representative_items",
            "recent_changes_json": "recent_changes",
            "open_questions_json": "open_questions",
            "recommended_reading_json": "recommended_reading",
        }
        for row in rows:
            item = dict(row)
            for raw_key, target_key in json_columns.items():
                try:
                    item[target_key] = json.loads(item.get(raw_key) or "[]")
                except (TypeError, json.JSONDecodeError):
                    item[target_key] = []
                item.pop(raw_key, None)
            result.append(item)
        return result

    def upsert_reading_queue(self, report_id: str, article_id: int, status: str = "tracked") -> Dict[str, Any]:
        if status not in {"tracked", "read", "archived"}:
            raise ValueError(f"Unsupported reading queue status: {status}")
        item = self.get_report_item(report_id, article_id) or self.get_article_by_id(article_id) or {}
        topic_key = str(item.get("continuity_topic_key") or item.get("domain_key") or item.get("topic") or "").strip()
        snapshot = json.dumps(item, ensure_ascii=False, sort_keys=True, default=str)
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO reading_queue (
                article_id, report_id, topic_key, status, snapshot_json,
                tracked_at, last_seen_at, last_change_at
            ) VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            ON CONFLICT(article_id) DO UPDATE SET
                report_id = excluded.report_id,
                topic_key = excluded.topic_key,
                status = excluded.status,
                snapshot_json = excluded.snapshot_json,
                last_seen_at = CURRENT_TIMESTAMP,
                last_change_at = CASE
                    WHEN reading_queue.status != excluded.status THEN CURRENT_TIMESTAMP
                    ELSE reading_queue.last_change_at
                END
            """,
            (int(article_id), str(report_id), topic_key, status, snapshot),
        )
        conn.commit()
        conn.close()
        return {"report_id": str(report_id), "article_id": int(article_id), "topic_key": topic_key, "status": status}

    def get_reading_queue(self, statuses: Optional[List[str]] = None, limit: int = 20) -> List[Dict[str, Any]]:
        selected_statuses = statuses or ["tracked"]
        placeholders = ",".join("?" for _ in selected_statuses)
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            f"""
            SELECT * FROM reading_queue
            WHERE status IN ({placeholders})
            ORDER BY tracked_at DESC
            LIMIT ?
            """,
            [*selected_statuses, max(1, int(limit))],
        )
        rows = cursor.fetchall()
        conn.close()
        result: List[Dict[str, Any]] = []
        for row in rows:
            queue_item = dict(row)
            try:
                snapshot = json.loads(queue_item.get("snapshot_json") or "{}")
            except (TypeError, json.JSONDecodeError):
                snapshot = {}
            queue_item["snapshot"] = snapshot
            queue_item.pop("snapshot_json", None)
            result.append(queue_item)
        return result

    def _deserialize_report_run(self, row: Optional[sqlite3.Row]) -> Dict[str, Any]:
        if not row:
            return {}
        item = dict(row)
        raw_diagnostics = item.get("quality_diagnostics")
        if isinstance(raw_diagnostics, str) and raw_diagnostics.strip():
            try:
                item["quality_diagnostics"] = json.loads(raw_diagnostics)
            except json.JSONDecodeError:
                item["quality_diagnostics"] = {}
        else:
            item["quality_diagnostics"] = {}
        return item

    def record_article_feedback(self, report_id: str, article_id: int, signal: str, source: str = "email") -> Dict[str, Any]:
        signal = str(signal or "").strip()
        if signal not in self.VALID_FEEDBACK_SIGNALS:
            raise ValueError(f"Unsupported feedback signal: {signal}")
        article = self.get_article_by_id(int(article_id))
        if not article:
            raise ValueError(f"Article not found: {article_id}")
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO article_feedback (report_id, article_id, signal, source)
            VALUES (?, ?, ?, ?)
            """,
            (report_id, int(article_id), signal, source),
        )
        conn.commit()
        conn.close()
        if signal == "track":
            self.upsert_reading_queue(report_id, int(article_id), status="tracked")
        self.recompute_preference_weights()
        return {"report_id": report_id, "article_id": int(article_id), "signal": signal, "title": article.get("title_cn") or article.get("title", "")}

    def get_feedback_count(self, days: int = 7) -> int:
        conn = self._get_conn()
        cursor = conn.cursor()
        since = (datetime.datetime.now() - datetime.timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("SELECT COUNT(*) AS count FROM article_feedback WHERE created_at >= ?", (since,))
        row = cursor.fetchone()
        conn.close()
        return int(row["count"] if row else 0)

    def get_recent_feedback(self, limit: int = 10) -> List[Dict[str, Any]]:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT
                f.report_id,
                f.article_id,
                f.signal,
                f.source,
                f.created_at,
                r.html_report_path,
                r.markdown_report_path,
                a.title,
                a.title_cn,
                a.source_detail,
                a.platform,
                a.topic,
                a.category
            FROM article_feedback f
            LEFT JOIN articles a ON a.id = f.article_id
            LEFT JOIN report_runs r ON r.report_id = f.report_id
            ORDER BY f.created_at DESC
            LIMIT ?
            """,
            (int(limit),),
        )
        rows = cursor.fetchall()
        conn.close()
        result: List[Dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["title"] = item.get("title_cn") or item.get("title") or ""
            result.append(item)
        return result

    def recompute_preference_weights(self, lookback_days: int = 30) -> Dict[str, float]:
        conn = self._get_conn()
        cursor = conn.cursor()
        since = (datetime.datetime.now() - datetime.timedelta(days=lookback_days)).strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute(
            """
            SELECT f.signal, a.source_detail, a.platform, a.topic, a.category, a.content_type, a.facts
            FROM article_feedback f
            LEFT JOIN articles a ON a.id = f.article_id
            WHERE f.created_at >= ?
            """,
            (since,),
        )
        rows = cursor.fetchall()
        weights: Dict[tuple[str, str], float] = {}
        signal_weights = {
            "useful": 0.25,
            "track": 0.35,
            "not_useful": -0.25,
            "mute_similar": -0.5,
            "too_shallow": 0.12,
            "paper_too_shallow": 0.12,
            "paper_unclear": 0.12,
            "too_long": -0.15,
            "too_generic": -0.2,
            "source_suspicious": -0.25,
            "not_memorable": 0.08,
        }
        for row in rows:
            signal = str(row["signal"] or "")
            delta = signal_weights.get(signal, 0.0)
            if not delta:
                continue
            keys = [
                ("source", str(row["source_detail"] or row["platform"] or "").strip()),
                ("topic", str(row["topic"] or row["category"] or "").strip()),
            ]
            raw_facts = row["facts"]
            if isinstance(raw_facts, str) and raw_facts.strip():
                try:
                    facts = json.loads(raw_facts)
                except json.JSONDecodeError:
                    facts = {}
                entity = str(facts.get("who") or "").strip()
                if entity:
                    keys.append(("entity", entity))
            for key_type, key in keys:
                if not key:
                    continue
                weight_key = (key_type, key)
                weights[weight_key] = max(-1.0, min(1.0, weights.get(weight_key, 0.0) + delta))

            style_keys: List[tuple[str, float]] = []
            if signal in {"too_shallow", "paper_too_shallow", "paper_unclear", "not_memorable"}:
                style_keys.append(("mechanism_depth", 0.18))
            if signal == "too_long":
                style_keys.append(("concision", 0.2))
            if signal in {"useful", "track"} and str(row["content_type"] or "") == "paper":
                style_keys.append(("paper_depth", 0.12 if signal == "useful" else 0.2))
            for key, style_delta in style_keys:
                weight_key = ("editorial_style", key)
                weights[weight_key] = max(-1.0, min(1.0, weights.get(weight_key, 0.0) + style_delta))

        cursor.execute("DELETE FROM preference_weights")
        cursor.executemany(
            """
            INSERT INTO preference_weights (key_type, key, weight, updated_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            """,
            [(key_type, key, weight) for (key_type, key), weight in weights.items()],
        )
        conn.commit()
        conn.close()
        return {f"{key_type}:{key}": weight for (key_type, key), weight in weights.items()}

    def get_preference_weights(self) -> Dict[str, Dict[str, float]]:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT key_type, key, weight FROM preference_weights")
        rows = cursor.fetchall()
        conn.close()
        result: Dict[str, Dict[str, float]] = {}
        for row in rows:
            result.setdefault(str(row["key_type"]), {})[str(row["key"])] = float(row["weight"] or 0.0)
        return result
