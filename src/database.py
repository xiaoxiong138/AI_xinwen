from __future__ import annotations

import datetime
import sqlite3
from typing import Any, Dict, List, Optional


class Database:
    ANALYSIS_COLUMNS = {
        "title_cn": "TEXT",
        "summary_preview": "TEXT",
        "why_it_matters": "TEXT",
        "why_now": "TEXT",
        "expected_effect": "TEXT",
        "future_impact": "TEXT",
    }

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
                    author, content_type, platform, topic, run_id, score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
    ):
        conn = self._get_conn()
        cursor = conn.cursor()
        keywords_str = ",".join(keywords) if isinstance(keywords, list) else str(keywords)
        cursor.execute(
            """
            UPDATE articles
            SET summary = ?, score = ?, keywords = ?, category = ?, processed = 1,
                title_cn = ?, summary_preview = ?, why_it_matters = ?, why_now = ?, expected_effect = ?, future_impact = ?
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
                url,
            ),
        )
        conn.commit()
        conn.close()

    def get_articles_for_run(self, run_id: str, processed_only: bool = True) -> List[Dict[str, Any]]:
        conn = self._get_conn()
        cursor = conn.cursor()
        if processed_only:
            cursor.execute("SELECT * FROM articles WHERE run_id = ? AND processed = 1 ORDER BY score DESC, publish_date DESC", (run_id,))
        else:
            cursor.execute("SELECT * FROM articles WHERE run_id = ? ORDER BY created_at DESC", (run_id,))
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

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
        return [dict(row) for row in rows]

    def get_recent_processed_articles(
        self,
        hours: int = 24,
        content_type: Optional[str] = None,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        conn = self._get_conn()
        cursor = conn.cursor()
        since = (datetime.datetime.now() - datetime.timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")
        if content_type == "paper":
            cursor.execute(
                """
                SELECT * FROM articles
                WHERE created_at >= ? AND processed = 1 AND content_type = 'paper'
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
        return [dict(row) for row in rows]

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
              )
            ORDER BY score DESC, publish_date DESC
            LIMIT ?
            """,
            (since, limit),
        )
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def record_collector_runs(self, run_id: str, collector_runs: List[Dict[str, Any]]) -> None:
        if not collector_runs:
            return
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.executemany(
            """
            INSERT INTO collector_runs (
                run_id, label, status, inserted_count, collected_count, duration_seconds, error
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
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
        return [dict(row) for row in rows]

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
        return [dict(row) for row in rows]

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
