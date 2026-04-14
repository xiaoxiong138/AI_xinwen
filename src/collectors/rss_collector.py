from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import feedparser

from .base import BaseCollector
from ..relevance import classify_paper_topic, clean_snippet, infer_platform, is_ai_web_content, is_relevant_paper


class RSSCollector(BaseCollector):
    def __init__(self, feeds: List[Dict[str, Any]], days_back: int = 3):
        self.feeds = feeds
        self.days_back = days_back

    def collect(self) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        cutoff_time = datetime.now(timezone.utc) - timedelta(days=self.days_back)

        for feed_config in self.feeds:
            name = feed_config["name"]
            url = feed_config["url"]
            max_entries = int(feed_config.get("max_entries", 10))
            content_type = feed_config.get("content_type", "news")
            explicit_platform = feed_config.get("platform", "")
            print(f"Collecting RSS from: {name}")

            try:
                feed = feedparser.parse(url)
                if feed.bozo:
                    print(f"Warning parsing {name}: {feed.bozo_exception}")
                for entry in feed.entries[:max_entries]:
                    published_dt = self._parse_entry_date(entry)
                    if published_dt < cutoff_time:
                        continue
                    content = self._extract_content(entry)
                    title = getattr(entry, "title", "").strip()
                    if content_type == "paper":
                        if not is_relevant_paper(title, content):
                            continue
                        topic, _ = classify_paper_topic(title, content)
                    else:
                        if not is_ai_web_content(title, content):
                            continue
                        topic = feed_config.get("topic", "AI")
                    link = getattr(entry, "link", "")
                    results.append(
                        {
                            "source": "RSS",
                            "source_detail": name,
                            "title": title,
                            "url": link,
                            "content": clean_snippet(content, limit=1200),
                            "publish_date": published_dt.isoformat(),
                            "author": getattr(entry, "author", name),
                            "content_type": content_type,
                            "platform": explicit_platform or infer_platform(link, name),
                            "topic": topic,
                        }
                    )
            except Exception as exc:
                print(f"Error collecting RSS {name}: {exc}")

        print(f"Collected {len(results)} articles from RSS.")
        return results

    def _parse_entry_date(self, entry: Any) -> datetime:
        if hasattr(entry, "published_parsed") and entry.published_parsed:
            return datetime.fromtimestamp(time.mktime(entry.published_parsed), tz=timezone.utc)
        if hasattr(entry, "updated_parsed") and entry.updated_parsed:
            return datetime.fromtimestamp(time.mktime(entry.updated_parsed), tz=timezone.utc)
        return datetime.now(timezone.utc)

    def _extract_content(self, entry: Any) -> str:
        if hasattr(entry, "summary"):
            return entry.summary
        if hasattr(entry, "description"):
            return entry.description
        if hasattr(entry, "content") and entry.content:
            first = entry.content[0]
            if isinstance(first, dict):
                return first.get("value", "")
        return getattr(entry, "title", "")
