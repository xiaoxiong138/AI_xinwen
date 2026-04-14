from __future__ import annotations

import re
import time
from datetime import datetime, timedelta, timezone
from html import unescape
from typing import Any, Dict, List, Mapping, Optional
from urllib.parse import urljoin
from urllib.request import urlopen

from .base import BaseCollector
from ..relevance import is_relevant_paper, score_paper_relevance


class ArxivCollector(BaseCollector):
    DEFAULT_TOPIC_LIMITS = {
        "Physical AI": 10,
        "World Model": 10,
        "Robotics": 10,
    }
    DEFAULT_TOPIC_QUERIES = {
        "Physical AI": 'all:"physical ai" OR all:"embodied ai" OR all:"vision-language-action" OR all:"robot foundation model" OR all:"embodied agent" OR all:"diffusion policy"',
        "World Model": 'all:"world model" OR all:"world models" OR all:"video prediction" OR all:"predictive model" OR all:"latent dynamics" OR all:"dynamics model"',
        "Robotics": 'cat:cs.RO OR all:robotics OR all:robot OR all:humanoid OR all:manipulation OR all:locomotion',
    }
    TOPIC_CATEGORY_MAP = {
        "Physical AI": ["cs.RO", "cs.AI", "cs.CV"],
        "World Model": ["cs.AI", "cs.CV", "cs.LG"],
        "Robotics": ["cs.RO"],
    }

    def __init__(
        self,
        categories: List[str],
        max_results: int = 30,
        candidate_pool: int = 160,
        days_back: int = 1,
        topic_limits: Optional[Mapping[str, int]] = None,
        fallback_days: Optional[List[int]] = None,
        topic_queries: Optional[Mapping[str, str]] = None,
    ):
        self.categories = categories
        self.candidate_pool = candidate_pool
        self.days_back = days_back
        self.fallback_days = fallback_days or [days_back, 3, 7, 14]
        raw_limits = dict(topic_limits or self.DEFAULT_TOPIC_LIMITS)
        self.topic_limits = {topic: max(0, int(limit)) for topic, limit in raw_limits.items()}
        self.topic_queries = dict(self.DEFAULT_TOPIC_QUERIES)
        if topic_queries:
            self.topic_queries.update({str(topic): query for topic, query in topic_queries.items()})
        topic_limit_sum = sum(self.topic_limits.values())
        self.max_results = min(max_results, topic_limit_sum) if topic_limit_sum else max_results

    def _score_item(self, title: str, abstract: str, topic: str) -> float:
        inferred_topic, relevance = score_paper_relevance(title, abstract)
        score = float(relevance)
        if inferred_topic == topic:
            score += 3.0
        return score

    def _select_by_topic(self, topic_items: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        selected: List[Dict[str, Any]] = []
        seen_urls = set()
        summary_parts = []
        for topic, limit in self.topic_limits.items():
            ranked = sorted(
                topic_items.get(topic, []),
                key=lambda item: (item.get("initial_score", 0), item.get("publish_date", "")),
                reverse=True,
            )
            topic_selected = []
            for item in ranked:
                if item["url"] in seen_urls:
                    continue
                topic_selected.append(item)
                seen_urls.add(item["url"])
                if len(topic_selected) >= limit:
                    break
            selected.extend(topic_selected)
            summary_parts.append(f"{topic}={len(topic_selected)}")
        selected = sorted(
            selected,
            key=lambda item: (item.get("initial_score", 0), item.get("publish_date", "")),
            reverse=True,
        )
        print("Selected papers by topic: " + ", ".join(summary_parts))
        return selected[: self.max_results]

    def _fetch_text(self, url: str) -> str:
        with urlopen(url, timeout=20) as response:
            return response.read().decode("utf-8", errors="ignore")

    def _clean_html_text(self, value: str) -> str:
        text = re.sub(r"<[^>]+>", " ", value or "")
        text = unescape(text)
        return re.sub(r"\s+", " ", text).strip()

    def _topic_categories(self, topic: str) -> List[str]:
        configured = [category for category in self.TOPIC_CATEGORY_MAP.get(topic, []) if category in self.categories]
        return configured or self.categories or self.TOPIC_CATEGORY_MAP.get(topic, [])

    def _fetch_recent_candidates(self, category: str) -> List[Dict[str, str]]:
        html = self._fetch_text(f"https://arxiv.org/list/{category}/recent?show=50")
        matches = re.findall(
            r'<dt>.*?<a href\s*=\s*"/abs/(?P<id>[^"]+)"[^>]*>.*?</dt>\s*<dd>.*?<div class=[\'"]list-title mathjax[\'"]><span class=[\'"]descriptor[\'"]>Title:</span>\s*(?P<title>.*?)\s*</div>(?P<meta>.*?)</dd>',
            html,
            re.S,
        )
        candidates: List[Dict[str, str]] = []
        seen_ids = set()
        for paper_id, raw_title, raw_meta in matches:
            if paper_id in seen_ids:
                continue
            seen_ids.add(paper_id)
            authors_match = re.search(r"<div class='list-authors'>(.*?)</div>", raw_meta, re.S)
            subjects_match = re.search(r"<div class='list-subjects'>(.*?)</div>", raw_meta, re.S)
            comments_match = re.search(r"<div class='list-comments mathjax'>(.*?)</div>", raw_meta, re.S)

            authors = self._clean_html_text(authors_match.group(1)) if authors_match else ""
            subjects = self._clean_html_text(subjects_match.group(1)) if subjects_match else ""
            comments = self._clean_html_text(comments_match.group(1)) if comments_match else ""
            content = " ".join(part for part in [self._clean_html_text(raw_title), subjects, comments] if part).strip()
            candidates.append(
                {
                    "id": paper_id,
                    "url": urljoin("https://arxiv.org", f"/abs/{paper_id}"),
                    "title": self._clean_html_text(raw_title),
                    "content": content,
                    "author": authors,
                }
            )
        return candidates

    def _fetch_abs_metadata(self, paper_url: str) -> Optional[Dict[str, Any]]:
        html = self._fetch_text(paper_url)
        abstract_match = re.search(r'<meta name="citation_abstract" content="(.*?)"\s*/?>', html, re.S)
        date_match = re.search(r'<meta name="citation_date" content="(.*?)"\s*/?>', html, re.S)
        author_matches = re.findall(r'<meta name="citation_author" content="(.*?)"\s*/?>', html, re.S)

        if not abstract_match:
            return None

        abstract = self._clean_html_text(abstract_match.group(1))
        authors = [self._clean_html_text(author) for author in author_matches[:5] if self._clean_html_text(author)]
        published = None
        if date_match:
            try:
                published = datetime.strptime(self._clean_html_text(date_match.group(1)), "%Y/%m/%d").replace(
                    tzinfo=timezone.utc
                )
            except ValueError:
                published = None

        return {
            "abstract": abstract,
            "author": ", ".join(authors),
            "published": published,
        }

    def _enrich_selected_items(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        enriched: List[Dict[str, Any]] = []
        for item in items:
            updated = dict(item)
            try:
                metadata = self._fetch_abs_metadata(item["url"])
            except Exception:
                metadata = None

            if metadata:
                abstract = metadata.get("abstract") or updated.get("content", "")
                if len(abstract) > len(updated.get("content", "")):
                    updated["content"] = abstract
                    updated["initial_score"] = self._score_item(updated["title"], abstract, updated["topic"])
                if metadata.get("author"):
                    updated["author"] = metadata["author"]
                if metadata.get("published"):
                    updated["publish_date"] = metadata["published"].isoformat()
            enriched.append(updated)
            time.sleep(0.2)
        return enriched

    def enrich_articles(self, items: List[Dict[str, Any]], max_items: Optional[int] = None) -> List[Dict[str, Any]]:
        selected = items[:max_items] if max_items else items
        return self._enrich_selected_items(selected)

    def _collect_topic_results_from_recent_pages(self, topic: str, cutoff_time: datetime) -> List[Dict[str, Any]]:
        print(f"Using arXiv recent pages for {topic}.")
        items: List[Dict[str, Any]] = []
        seen_urls = set()
        candidate_limit = max(self.topic_limits.get(topic, 0) * 3, 12)

        for category in self._topic_categories(topic):
            try:
                candidates = self._fetch_recent_candidates(category)
            except Exception as exc:
                print(f"Fallback list fetch failed for {topic} / {category}: {exc}")
                continue

            for candidate in candidates:
                if candidate["url"] in seen_urls:
                    continue
                seen_urls.add(candidate["url"])

                if len(items) >= candidate_limit:
                    break

                title = candidate["title"]
                if not title:
                    continue

                abstract = candidate.get("content", "") or title
                if not is_relevant_paper(title, abstract):
                    continue

                items.append(
                    {
                        "source": "ArXiv",
                        "source_detail": topic,
                        "title": title,
                        "url": candidate["url"],
                        "content": abstract,
                        "publish_date": datetime.now(timezone.utc).isoformat(),
                        "author": candidate.get("author", ""),
                        "content_type": "paper",
                        "platform": "ArXiv",
                        "topic": topic,
                        "initial_score": self._score_item(title, abstract, topic),
                    }
                )
            if len(items) >= candidate_limit:
                break

        return items

    def _collect_topic_results(self, topic: str, query: str, cutoff_time: datetime) -> List[Dict[str, Any]]:
        return self._collect_topic_results_from_recent_pages(topic, cutoff_time)

    def collect(self) -> List[Dict[str, Any]]:
        topic_items: Dict[str, List[Dict[str, Any]]] = {topic: [] for topic in self.topic_limits}
        for days_window in self.fallback_days:
            cutoff_time = datetime.now(timezone.utc) - timedelta(days=int(days_window))
            topic_items = {topic: [] for topic in self.topic_limits}

            for topic, limit in self.topic_limits.items():
                if limit <= 0:
                    continue
                query = self.topic_queries.get(topic)
                if not query:
                    continue
                print(f"Collecting ArXiv papers for {topic} within {days_window} day(s)")
                topic_items[topic].extend(self._collect_topic_results(topic, query, cutoff_time))
                time.sleep(0.5)

            selected = self._select_by_topic(topic_items)
            if len(selected) >= self.max_results:
                print(f"Reached paper target with {days_window}-day window.")
                return self._enrich_selected_items(selected)

        final_results = self._select_by_topic(topic_items)
        print(f"Collected {len(final_results)} relevant ArXiv papers.")
        return self._enrich_selected_items(final_results)
