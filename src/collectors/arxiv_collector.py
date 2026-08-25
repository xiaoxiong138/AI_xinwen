from __future__ import annotations

import re
import time
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
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
        "Agent / Models": ["cs.AI", "cs.LG"],
        "Multimodal / Video": ["cs.CV", "cs.AI"],
        "Infra / Efficient AI": ["cs.LG", "cs.AI"],
    }
    TOPIC_TERMS = {
        "Physical AI": (
            "physical ai", "embodied ai", "embodied agent", "vision-language-action",
            "vla", "robot foundation model", "diffusion policy",
        ),
        "World Model": (
            "world model", "latent dynamics", "dynamics model", "video prediction",
            "future prediction", "predictive model", "jepa",
        ),
        "Robotics": (
            "robot", "robotics", "humanoid", "manipulation", "locomotion",
            "grasping", "navigation",
        ),
        "Agent / Models": (
            "agent", "agentic", "tool use", "tool-use", "reasoning model",
            "language model", "multi-agent", "workflow",
        ),
        "Multimodal / Video": (
            "multimodal", "vision-language", "video generation", "video understanding",
            "text-to-video", "image-to-video", "visual language",
        ),
        "Infra / Efficient AI": (
            "efficient inference", "inference serving", "quantization", "model compression",
            "distributed training", "mixture of experts", "sparse model", "throughput",
        ),
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
        self.fetch_diagnostics: Dict[str, Any] = {
            "request_attempt_count": 0,
            "request_error_count": 0,
            "page_parse_error_count": 0,
            "successful_page_count": 0,
            "parsed_candidate_count": 0,
            "explicit_zero_page_count": 0,
            "fallback_show_counts": [],
            "retry_paths": [],
            "true_zero_result": False,
        }
        self._abs_metadata_cache: Dict[str, Optional[Dict[str, Any]]] = {}

    def _score_item(self, title: str, abstract: str, topic: str) -> float:
        inferred_topic, relevance = score_paper_relevance(title, abstract)
        score = float(relevance)
        if inferred_topic == topic:
            score += 3.0
        return score

    def _matches_topic(self, topic: str, title: str, abstract: str) -> bool:
        terms = self.TOPIC_TERMS.get(topic, ())
        if not terms:
            return True
        text = f"{title} {abstract}".lower()
        return any(term in text for term in terms)

    def _select_by_topic(
        self,
        topic_items: Dict[str, List[Dict[str, Any]]],
        total_limit: Optional[int] = None,
        topic_limit_multiplier: int = 1,
    ) -> List[Dict[str, Any]]:
        selected: List[Dict[str, Any]] = []
        seen_urls = set()
        summary_parts = []
        for topic, limit in self.topic_limits.items():
            limit = max(0, int(limit) * max(1, int(topic_limit_multiplier or 1)))
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
        return selected[: int(total_limit or self.max_results)]

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
        show_count = max(50, min(int(self.candidate_pool or 50), 2000))
        last_error: Optional[Exception] = None
        attempted_show_counts: List[int] = []
        local_request_error_count = 0
        local_parse_error_count = 0
        for candidate_count in dict.fromkeys((show_count, 500, 100, 50)):
            attempted_show_counts.append(candidate_count)
            self.fetch_diagnostics["request_attempt_count"] += 1
            try:
                html = self._fetch_text(
                    f"https://arxiv.org/list/{category}/recent?show={candidate_count}"
                )
                self.fetch_diagnostics["fallback_show_counts"].append(candidate_count)
            except Exception as exc:
                self.fetch_diagnostics["request_error_count"] += 1
                local_request_error_count += 1
                last_error = exc
                continue
            matches = re.findall(
                r'<dt>.*?<a href\s*=\s*"/abs/(?P<id>[^"]+)"[^>]*>.*?</dt>\s*<dd>.*?<div class=[\'"]list-title mathjax[\'"]><span class=[\'"]descriptor[\'"]>Title:</span>\s*(?P<title>.*?)\s*</div>(?P<meta>.*?)</dd>',
                html,
                re.S,
            )
            if matches:
                self.fetch_diagnostics["successful_page_count"] += 1
                self.fetch_diagnostics["parsed_candidate_count"] += len(matches)
                self.fetch_diagnostics["retry_paths"].append({
                    "category": category,
                    "attempted_show_counts": attempted_show_counts.copy(),
                    "successful_show_count": candidate_count,
                    "result": "success",
                })
                break
            page_text = self._clean_html_text(html).lower()
            if any(
                marker in page_text
                for marker in (
                    "no articles found",
                    "no submissions found",
                    "no entries found",
                    "there are no articles",
                    "0 total entries",
                    "showing 0-0 of 0",
                )
            ):
                self.fetch_diagnostics["successful_page_count"] += 1
                self.fetch_diagnostics["explicit_zero_page_count"] += 1
                self.fetch_diagnostics["true_zero_result"] = True
                self.fetch_diagnostics["retry_paths"].append({
                    "category": category,
                    "attempted_show_counts": attempted_show_counts.copy(),
                    "successful_show_count": candidate_count,
                    "result": "true_zero",
                })
                return []
            self.fetch_diagnostics["page_parse_error_count"] += 1
            local_parse_error_count += 1
            last_error = ValueError(f"arxiv_page_parse_empty:{category}:show={candidate_count}")
        else:
            self.fetch_diagnostics["retry_paths"].append({
                "category": category,
                "attempted_show_counts": attempted_show_counts.copy(),
                "successful_show_count": None,
                "result": (
                    "http_error"
                    if local_request_error_count == len(attempted_show_counts)
                    else "parse_error"
                    if local_parse_error_count == len(attempted_show_counts)
                    else "mixed_error"
                ),
            })
            if last_error is not None:
                raise last_error
            raise ValueError(f"arxiv_page_parse_empty:{category}:show={show_count}")
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
        if paper_url in self._abs_metadata_cache:
            return self._abs_metadata_cache[paper_url]
        html = self._fetch_text(paper_url)
        abstract_match = re.search(r'<meta name="citation_abstract" content="(.*?)"\s*/?>', html, re.S)
        date_match = re.search(r'<meta name="citation_date" content="(.*?)"\s*/?>', html, re.S)
        author_matches = re.findall(r'<meta name="citation_author" content="(.*?)"\s*/?>', html, re.S)
        version_matches = [int(value) for value in re.findall(r"\[v(\d+)\]", html, re.I)]
        submission_dates: List[tuple[int, datetime]] = []
        history_match = re.search(
            r'<div class="submission-history"[^>]*>(.*?)</div>',
            html,
            re.I | re.S,
        )
        history_text = self._clean_html_text(history_match.group(1)) if history_match else ""
        for version_value, raw_date in re.findall(r"\[v(\d+)\]\s*([^\[]+?UTC)", history_text, re.I):
            try:
                parsed_date = parsedate_to_datetime(self._clean_html_text(raw_date))
                if parsed_date.tzinfo is None:
                    parsed_date = parsed_date.replace(tzinfo=timezone.utc)
                submission_dates.append((int(version_value), parsed_date.astimezone(timezone.utc)))
            except (TypeError, ValueError, OverflowError):
                continue

        if not abstract_match:
            self._abs_metadata_cache[paper_url] = None
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

        result = {
            "abstract": abstract,
            "author": ", ".join(authors),
            "published": published,
            "arxiv_version": max(version_matches, default=1),
            "updated": max(submission_dates, key=lambda item: item[0])[1] if submission_dates else published,
        }
        self._abs_metadata_cache[paper_url] = result
        return result

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
                effective_date = metadata.get("updated") or metadata.get("published")
                if effective_date:
                    updated["publish_date"] = effective_date.isoformat()
                version = max(1, int(metadata.get("arxiv_version", 1) or 1))
                canonical_url = re.sub(r"v\d+$", "", str(updated.get("url", "")), flags=re.I)
                updated["canonical_url"] = canonical_url
                updated["arxiv_version"] = version
                if version > 1:
                    updated["url"] = f"{canonical_url}v{version}"
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
        candidate_limit = max(self.topic_limits.get(topic, 0) * 4, 24)

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
                if not self._matches_topic(topic, title, abstract):
                    continue

                items.append(
                    {
                        "source": "ArXiv",
                        "source_detail": topic,
                        "title": title,
                        "url": candidate["url"],
                        "content": abstract,
                        "publish_date": "",
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
        self.fetch_diagnostics = {
            "request_attempt_count": 0,
            "request_error_count": 0,
            "page_parse_error_count": 0,
            "successful_page_count": 0,
            "parsed_candidate_count": 0,
            "explicit_zero_page_count": 0,
            "fallback_show_counts": [],
            "retry_paths": [],
            "true_zero_result": False,
        }
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

            selected = self._select_by_topic(
                topic_items,
                max(self.max_results * 2, self.max_results),
                topic_limit_multiplier=2,
            )
            enriched = self._enrich_selected_items(selected)
            recent_selected = [
                item
                for item in enriched
                if item.get("publish_date")
                and datetime.fromisoformat(str(item["publish_date"]).replace("Z", "+00:00")) >= cutoff_time
            ]
            recent_by_topic: Dict[str, List[Dict[str, Any]]] = {topic: [] for topic in self.topic_limits}
            for item in recent_selected:
                recent_by_topic.setdefault(str(item.get("topic", "")), []).append(item)
            final_results = self._select_by_topic(recent_by_topic)
            if len(final_results) >= self.max_results:
                print(f"Reached paper target with {days_window}-day window.")
                self.fetch_diagnostics["true_zero_result"] = False
                return final_results

        final_results = final_results if 'final_results' in locals() else []
        self.fetch_diagnostics["true_zero_result"] = bool(
            not final_results
            and self.fetch_diagnostics.get("explicit_zero_page_count", 0)
            and not self.fetch_diagnostics.get("page_parse_error_count", 0)
        )
        self.fetch_diagnostics["no_match_result"] = bool(
            not final_results
            and self.fetch_diagnostics.get("successful_page_count", 0)
            and not self.fetch_diagnostics.get("explicit_zero_page_count", 0)
            and not self.fetch_diagnostics.get("page_parse_error_count", 0)
        )
        print(f"Collected {len(final_results)} relevant ArXiv papers.")
        return final_results
