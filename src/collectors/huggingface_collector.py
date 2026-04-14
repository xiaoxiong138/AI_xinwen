from __future__ import annotations

import datetime
from typing import Any, Dict, List

import requests

from .base import BaseCollector
from ..relevance import classify_paper_topic, is_relevant_paper


class HuggingFaceCollector(BaseCollector):
    def __init__(self):
        self.api_url = "https://huggingface.co/api/daily_papers"

    def collect(self) -> List[Dict[str, Any]]:
        results_list: List[Dict[str, Any]] = []
        try:
            today = datetime.datetime.now().strftime("%Y-%m-%d")
            response = requests.get(f"{self.api_url}?date={today}", timeout=20)
            if response.status_code == 200:
                papers = response.json()
                for paper in papers:
                    title = paper.get("title", "")
                    paper_id = paper.get("paper", {}).get("id", "")
                    summary = paper.get("summary", "")
                    upvotes = paper.get("upvotes", 0)
                    if not is_relevant_paper(title, summary or title):
                        continue
                    topic, _ = classify_paper_topic(title, summary or title)
                    results_list.append(
                        {
                            "source": "HuggingFace",
                            "source_detail": "Daily Papers",
                            "title": title,
                            "url": f"https://huggingface.co/papers/{paper_id}" if paper_id else "",
                            "content": summary or title,
                            "publish_date": today,
                            "score": upvotes,
                            "author": "Hugging Face Community",
                            "content_type": "paper",
                            "platform": "Hugging Face",
                            "topic": topic,
                        }
                    )
            else:
                print(f"Failed to fetch HF papers for {today}: {response.status_code}")
        except Exception as exc:
            print(f"Error collecting HF papers: {exc}")
        print(f"Collected {len(results_list)} papers from Hugging Face.")
        return results_list
