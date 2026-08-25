from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, unquote, urlparse
from urllib.request import Request, urlopen

import feedparser

from .base import BaseCollector
from ..relevance import clean_snippet, infer_platform, is_ai_web_content


class WebSearchCollector(BaseCollector):
    SEARCH_URL = "https://news.google.com/rss/search"

    def __init__(
        self,
        searches: List[Dict[str, Any]],
        locale: str = "US:en",
        days_back: int = 1,
        fallback_days: Optional[List[int]] = None,
    ):
        self.searches = searches
        self.locale = locale
        self.days_back = days_back
        self.fallback_days = fallback_days or [days_back, 2, 3]
        country, language = locale.split(":")
        self.country = country
        self.language = language

    def collect(self) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        seen_urls = set()
        for search in self.searches:
            name = search.get("name", "Web Search")
            query = search.get("query", "").strip()
            if not query:
                continue
            max_results = int(search.get("max_results", 10))
            forced_platform = search.get("platform", "")
            content_type = search.get("content_type", "news")
            search_fallback_days = search.get("fallback_days") or self.fallback_days
            print(f"Collecting web search results for: {name}")
            added = 0
            for days_window in search_fallback_days:
                if added >= max_results:
                    break
                newly_added = self._collect_for_window(
                    results=results,
                    seen_urls=seen_urls,
                    query=query,
                    name=name,
                    max_results=max_results - added,
                    forced_platform=forced_platform,
                    content_type=content_type,
                    days_back=int(days_window),
                    topic=search.get("topic", "AI"),
                )
                added += newly_added
                if newly_added:
                    print(f"  -> Added {newly_added} results from {name} within {days_window} day(s)")
            print(f"  -> Total added {added} results from {name}")
        print(f"Collected {len(results)} web results.")
        return results

    def _collect_for_window(
        self,
        results: List[Dict[str, Any]],
        seen_urls: set,
        query: str,
        name: str,
        max_results: int,
        forced_platform: str,
        content_type: str,
        days_back: int,
        topic: str,
    ) -> int:
        cutoff_time = datetime.now(timezone.utc) - timedelta(days=days_back)
        try:
            feed = feedparser.parse(
                f"{self.SEARCH_URL}?q={query.replace(' ', '+')}&hl={self.language}&gl={self.country}&ceid={self.locale}"
            )
            if getattr(feed, "bozo", False):
                print(f"Warning parsing search feed for {name}: {feed.bozo_exception}")
        except Exception as exc:
            print(f"Error collecting search results for {name}: {exc}")
            return 0

        added = 0
        for entry in feed.entries:
            published_dt = self._parse_entry_date(entry)
            if published_dt < cutoff_time:
                continue
            title = entry.get("title", "").strip()
            url = entry.get("link", "")
            snippet = entry.get("summary", "") or title
            if not url or url in seen_urls:
                continue
            canonical_url = self._canonicalize_google_news_url(entry, url)
            if canonical_url in seen_urls:
                continue
            if not is_ai_web_content(title, snippet):
                continue
            seen_urls.add(canonical_url)
            platform = forced_platform or infer_platform(canonical_url, name)
            results.append(
                {
                    "source": "Web Search",
                    "source_detail": name,
                    "title": title,
                    "url": canonical_url,
                    "content": clean_snippet(snippet, limit=600),
                    "publish_date": published_dt.isoformat(),
                    "author": platform,
                    "content_type": content_type,
                    "platform": platform,
                    "topic": topic,
                    "original_url": url if canonical_url != url else "",
                }
            )
            added += 1
            if added >= max_results:
                break
        return added

    def _canonicalize_google_news_url(self, entry: Any, url: str) -> str:
        if urlparse(url or "").netloc.lower() != "news.google.com":
            return url
        for candidate in self._google_news_candidate_urls(entry, url):
            if candidate and urlparse(candidate).netloc.lower() != "news.google.com":
                return candidate
        try:
            request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(request, timeout=8) as response:
                resolved_url = response.geturl()
            if resolved_url and urlparse(resolved_url).netloc.lower() != "news.google.com":
                return resolved_url
        except Exception:
            return url
        return url

    def _google_news_candidate_urls(self, entry: Any, url: str) -> List[str]:
        candidates: List[str] = []
        parsed = urlparse(url or "")
        query = parse_qs(parsed.query)
        for key in ("url", "u"):
            for value in query.get(key, []):
                decoded = unquote(value)
                if decoded.startswith("http"):
                    candidates.append(decoded)
        source = entry.get("source", {}) if hasattr(entry, "get") else {}
        if isinstance(source, dict):
            href = source.get("href")
            if isinstance(href, str) and href.startswith("http"):
                candidates.append(href)
        for link in entry.get("links", []) if hasattr(entry, "get") else []:
            if isinstance(link, dict):
                href = link.get("href", "")
                if href.startswith("http"):
                    candidates.append(href)
        return candidates

    def _parse_entry_date(self, entry: Any) -> datetime:
        if hasattr(entry, "published_parsed") and entry.published_parsed:
            return datetime.fromtimestamp(time.mktime(entry.published_parsed), tz=timezone.utc)
        if hasattr(entry, "updated_parsed") and entry.updated_parsed:
            return datetime.fromtimestamp(time.mktime(entry.updated_parsed), tz=timezone.utc)
        return datetime.now(timezone.utc)
