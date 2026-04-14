from __future__ import annotations

import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from jinja2 import Environment, FileSystemLoader


class ReportGenerator:
    INTERLEAVE_PATTERN = ("update", "paper", "update")
    MAX_CONSECUTIVE_SAME_KIND = 2

    def __init__(self, template_dir: str = "templates"):
        os.makedirs(template_dir, exist_ok=True)
        self.env = Environment(loader=FileSystemLoader(template_dir))

    def _score_value(self, item: Dict[str, Any]) -> float:
        return float(item.get("selection_score", item.get("score", 0)) or 0)

    def _sort_items(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return sorted(
            items,
            key=lambda item: (
                self._score_value(item),
                float(item.get("score", 0) or 0),
                item.get("publish_date", ""),
            ),
            reverse=True,
        )

    def build_mixed_items(self, papers: List[Dict[str, Any]], updates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        paper_queue = self._sort_items(papers)
        update_queue = self._sort_items(updates)
        mixed: List[Dict[str, Any]] = []

        last_kind = ""
        same_kind_streak = 0
        pattern_index = 0

        while paper_queue or update_queue:
            desired_kind = self.INTERLEAVE_PATTERN[pattern_index % len(self.INTERLEAVE_PATTERN)]
            forced_switch = last_kind and same_kind_streak >= self.MAX_CONSECUTIVE_SAME_KIND

            next_item = None
            next_kind = ""

            if not forced_switch:
                if desired_kind == "paper" and paper_queue:
                    next_item = paper_queue.pop(0)
                    next_kind = "paper"
                elif desired_kind == "update" and update_queue:
                    next_item = update_queue.pop(0)
                    next_kind = "update"

            if next_item is None:
                if forced_switch and last_kind == "paper" and update_queue:
                    next_item = update_queue.pop(0)
                    next_kind = "update"
                elif forced_switch and last_kind == "update" and paper_queue:
                    next_item = paper_queue.pop(0)
                    next_kind = "paper"
                elif paper_queue and update_queue:
                    top_paper = self._score_value(paper_queue[0])
                    top_update = self._score_value(update_queue[0])
                    if last_kind == "paper":
                        if top_update >= top_paper - 0.35:
                            next_item = update_queue.pop(0)
                            next_kind = "update"
                        else:
                            next_item = paper_queue.pop(0)
                            next_kind = "paper"
                    elif last_kind == "update":
                        if top_paper >= top_update - 0.35:
                            next_item = paper_queue.pop(0)
                            next_kind = "paper"
                        else:
                            next_item = update_queue.pop(0)
                            next_kind = "update"
                    else:
                        if top_update >= top_paper:
                            next_item = update_queue.pop(0)
                            next_kind = "update"
                        else:
                            next_item = paper_queue.pop(0)
                            next_kind = "paper"
                elif update_queue:
                    next_item = update_queue.pop(0)
                    next_kind = "update"
                elif paper_queue:
                    next_item = paper_queue.pop(0)
                    next_kind = "paper"

            if next_item is None:
                break

            mixed.append(next_item)
            if next_kind == last_kind:
                same_kind_streak += 1
            else:
                last_kind = next_kind
                same_kind_streak = 1
            pattern_index += 1

        return mixed

    def _preview_text(self, item: Dict[str, Any]) -> str:
        preview = str(item.get("summary_preview", "") or "").strip()
        if preview:
            return preview

        summary = re.split(r"(?<=[。！？])\s*", str(item.get("summary", "") or "").strip())
        for sentence in summary:
            sentence = sentence.strip()
            if sentence:
                return sentence[:54] + ("..." if len(sentence) > 54 else "")
        return ""

    def _trim_reason(self, text: str, limit: int = 52) -> str:
        cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
        cleaned = re.sub(r"^[：:，,\-\s]+", "", cleaned)
        cleaned = re.sub(r"[。！？!?.]+$", "", cleaned)
        if not cleaned:
            return ""
        return cleaned if len(cleaned) <= limit else cleaned[: limit - 1].rstrip("，、；： ") + "…"

    def _highlight_reason(self, item: Dict[str, Any]) -> str:
        topic = str(item.get("display_topic", "") or "")
        content_type = str(item.get("content_type", "") or "")
        why_it_matters = self._trim_reason(item.get("why_it_matters", ""))
        why_now = self._trim_reason(item.get("why_now", ""))
        expected_effect = self._trim_reason(item.get("expected_effect", ""))
        future_impact = self._trim_reason(item.get("future_impact", ""))

        if content_type == "paper":
            if "世界模型" in topic:
                return expected_effect or why_it_matters or "先看它能否把预测能力变成更低试错成本"
            if "具身智能" in topic:
                return expected_effect or why_now or "先看它能否把实验结果带进真实环境闭环"
            if "机器人" in topic:
                return expected_effect or future_impact or "先看它能否把论文效果转成实机成功率"
            return why_it_matters or expected_effect or "先看它是否真的更接近可部署阶段"

        if "产品发布" in topic:
            return expected_effect or why_it_matters or "先看它会先替代哪些高频人工步骤"
        if "企业合作" in topic:
            return why_it_matters or future_impact or "先看合作能否补齐客户、渠道或交付短板"
        if "开源生态" in topic:
            return future_impact or why_it_matters or "先看它会不会变成社区默认做法"
        if "基础设施" in topic:
            return expected_effect or future_impact or "先看它会先改写哪些成本和供给关系"
        if "应用落地" in topic:
            return expected_effect or why_it_matters or "先看这套方案能不能跨场景复制"
        if "模型/研究" in topic:
            return why_it_matters or expected_effect or "先看能力提升会不会兑现成更稳交付"
        return why_it_matters or expected_effect or future_impact or "先看这条变化会先影响哪一层"

    def _card_reason(self, item: Dict[str, Any]) -> str:
        topic = str(item.get("display_topic", "") or "")
        content_type = str(item.get("content_type", "") or "")
        why_it_matters = self._trim_reason(item.get("why_it_matters", ""), 34)
        expected_effect = self._trim_reason(item.get("expected_effect", ""), 34)
        future_impact = self._trim_reason(item.get("future_impact", ""), 34)

        if content_type == "paper":
            if "世界模型" in topic:
                return expected_effect or "先看能否变成更低试错成本"
            if "具身智能" in topic:
                return expected_effect or "先看能否带到真实环境"
            if "机器人" in topic:
                return expected_effect or future_impact or "先看能否转成实机成功率"
            return why_it_matters or "先看是否更接近可部署"

        if "产品发布" in topic:
            return expected_effect or "先看会先替代哪些人工步骤"
        if "企业合作" in topic:
            return why_it_matters or "先看能否补齐客户或交付短板"
        if "开源生态" in topic:
            return future_impact or "先看会不会变成社区默认做法"
        if "基础设施" in topic:
            return expected_effect or "先看会先改写哪些成本关系"
        if "应用落地" in topic:
            return expected_effect or "先看能不能跨场景复制"
        if "模型/研究" in topic:
            return why_it_matters or "先看会不会兑现成更稳交付"
        return why_it_matters or expected_effect or future_impact or "先看这条变化先影响哪一层"

    def _decorate_highlights(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        decorated: List[Dict[str, Any]] = []
        for item in items:
            decorated_item = dict(item)
            decorated_item["highlight_reason"] = self._highlight_reason(item)
            decorated.append(decorated_item)
        return decorated

    def _decorate_items(self, items: List[Dict[str, Any]], offset: int = 0, prefix: str = "card") -> List[Dict[str, Any]]:
        decorated: List[Dict[str, Any]] = []
        for index, item in enumerate(items, start=1 + offset):
            decorated_item = dict(item)
            decorated_item["order"] = index
            decorated_item["anchor_id"] = f"{prefix}-{index:02d}"
            decorated_item["summary_preview"] = self._preview_text(item)
            decorated_item["impact_tag"] = item.get("impact_tag", "值得关注")
            decorated_item["content_kind"] = item.get(
                "content_kind",
                "论文" if item.get("content_type") == "paper" else "全网动态",
            )
            decorated_item["display_topic"] = item.get(
                "display_topic",
                item.get("topic_cn") or item.get("category") or "其他",
            )
            decorated_item["quick_reason"] = self._card_reason(decorated_item)
            decorated_item["quick_reason"] = self._card_reason(decorated_item)
            decorated_item["is_expanded_default"] = False
            decorated.append(decorated_item)
        return decorated

    def _hero_description(
        self,
        papers: List[Dict[str, Any]],
        updates: List[Dict[str, Any]],
        report_summary: Dict[str, Any],
    ) -> str:
        lead_summary = str(report_summary.get("lead_summary", "") or "").strip()
        if lead_summary:
            first_sentence = re.split(r"(?<=[。！？])\s*", lead_summary)[0].strip()
            if first_sentence:
                return first_sentence
        return (
            f"本期聚焦具身智能、世界模型、机器人与全网 AI 动态，"
            f"共精选 {len(papers)} 篇论文和 {len(updates)} 条高价值情报。"
        )

    def _hero_scope(
        self,
        papers: List[Dict[str, Any]],
        updates: List[Dict[str, Any]],
        report_summary: Dict[str, Any],
    ) -> List[str]:
        scope = [
            f"{len(papers)} 篇精选论文",
            f"{len(updates)} 条全网动态",
            "论文与情报混排展示",
        ]
        hot_topics = [str(item).strip() for item in (report_summary.get("hot_topics") or []) if str(item).strip()]
        scope.extend(hot_topics[:3])
        unique_scope: List[str] = []
        seen = set()
        for item in scope:
            if item in seen:
                continue
            seen.add(item)
            unique_scope.append(item)
        return unique_scope[:6]

    def _pick_top_highlights(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        sorted_all = self._sort_items(items)
        updates = [item for item in sorted_all if item.get("content_type") != "paper"]
        papers = [item for item in sorted_all if item.get("content_type") == "paper"]

        selected: List[Dict[str, Any]] = []
        selected_urls = set()
        update_target = 3 if len(updates) >= 3 else max(1, len(updates))
        paper_target = 2 if len(papers) >= 2 else max(1, len(papers))

        for candidate in updates[:update_target] + papers[:paper_target]:
            url = candidate.get("url")
            if not url or url in selected_urls:
                continue
            selected.append(candidate)
            selected_urls.add(url)

        for candidate in sorted_all:
            if len(selected) >= 5:
                break
            url = candidate.get("url")
            if not url or url in selected_urls:
                continue
            selected.append(candidate)
            selected_urls.add(url)

        return self._sort_items(selected)[:5]

    def _exclude_highlight_duplicates(
        self,
        card_items: List[Dict[str, Any]],
        top_highlights: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        highlight_urls = {str(item.get("url", "")) for item in top_highlights if item.get("url")}
        body_items = [item for item in card_items if str(item.get("url", "")) not in highlight_urls]
        return self._decorate_items(body_items, 0, "mix-body")

    def _render_card_block(self, lines: List[str], items: List[Dict[str, Any]], heading: str, note: str = "") -> None:
        if not items:
            return
        lines.append(f"## {heading}")
        if note:
            lines.append(note)
            lines.append("")
        for article in items:
            lines.append(f"### {article['order']}. {article.get('title_cn')}")
            lines.append(
                f"- 类型: {article.get('content_kind')} | 主题: {article.get('display_topic')} | "
                f"影响标签: {article.get('impact_tag')} | Score: {article.get('score')} | 时间: {str(article.get('publish_date', ''))[:16]}"
            )
            preview = article.get("summary_preview")
            if preview:
                lines.append(f"- 导语: {preview}")
            lines.append(f"- 摘要: {article.get('summary')}")
            lines.append(f"- 为什么值得看: {article.get('why_it_matters')}")
            lines.append(f"- 为什么这么做: {article.get('why_now')}")
            lines.append(f"- 会带来什么效果: {article.get('expected_effect')}")
            lines.append(f"- 未来影响: {article.get('future_impact')}")
            lines.append(f"- 原文: {article.get('url')}")
            lines.append("")

    def generate_html(
        self,
        papers: List[Dict[str, Any]],
        updates: List[Dict[str, Any]],
        mixed_items: List[Dict[str, Any]],
        report_summary: Dict[str, Any],
        title: str = "AI Daily Report",
        collector_summary: Optional[Dict[str, Any]] = None,
        trend_summary: Optional[Dict[str, Any]] = None,
        alert_summary: Optional[Dict[str, Any]] = None,
        layered_updates: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        archive_summary: Optional[Dict[str, Any]] = None,
    ) -> str:
        card_items = self._decorate_items(mixed_items, 0, "mix")
        top_highlights = self._decorate_highlights(self._pick_top_highlights(card_items))
        body_items = self._exclude_highlight_duplicates(card_items, top_highlights)

        context = {
            "title": title,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "paper_count": len(papers),
            "update_count": len(updates),
            "hero_description": self._hero_description(papers, updates, report_summary),
            "hero_scope": self._hero_scope(papers, updates, report_summary),
            "top_highlights": top_highlights,
            "card_items": body_items,
            "report_summary": report_summary,
            "collector_summary": collector_summary or {},
            "trend_summary": trend_summary or {},
            "alert_summary": alert_summary or {},
            "archive_summary": archive_summary or {},
        }
        template = self.env.get_template("daily_report.html")
        return template.render(**context)

    def generate_markdown(
        self,
        papers: List[Dict[str, Any]],
        updates: List[Dict[str, Any]],
        mixed_items: List[Dict[str, Any]],
        report_summary: Dict[str, Any],
        title: str = "AI Daily Report",
        collector_summary: Optional[Dict[str, Any]] = None,
        trend_summary: Optional[Dict[str, Any]] = None,
        alert_summary: Optional[Dict[str, Any]] = None,
        layered_updates: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        archive_summary: Optional[Dict[str, Any]] = None,
    ) -> str:
        trend_summary = trend_summary or {}
        alert_summary = alert_summary or {}
        archive_summary = archive_summary or {}

        card_items = self._decorate_items(mixed_items, 0, "mix")
        top_highlights = self._decorate_highlights(self._pick_top_highlights(card_items))
        body_items = self._exclude_highlight_duplicates(card_items, top_highlights)

        lines = [f"# {title} - {datetime.now().strftime('%Y-%m-%d %H:%M')}", ""]
        lines.append(self._hero_description(papers, updates, report_summary))
        lines.append("")

        hero_scope = self._hero_scope(papers, updates, report_summary)
        if hero_scope:
            lines.append("范围: " + " | ".join(hero_scope))
            lines.append("")

        if alert_summary.get("needs_alert"):
            lines.append("## 异常提醒")
            for issue in alert_summary.get("issues", []):
                lines.append(f"- {issue}")
            lines.append("")

        if top_highlights:
            lines.append("## 今日最重要 5 条")
            for item in top_highlights:
                lines.append(
                    f"- {item.get('title_cn')} | {item.get('content_kind')} | {item.get('display_topic')} | {item.get('impact_tag')} | Score {item.get('score')}"
                )
                preview = item.get("summary_preview")
                if preview:
                    lines.append(f"  导语: {preview}")
                highlight_reason = item.get("highlight_reason")
                if highlight_reason:
                    lines.append(f"  先看理由: {highlight_reason}")
            lines.append("")

        lines.append("## 今日总览")
        lines.append(report_summary.get("lead_summary", ""))
        lines.append("")

        lines.append("## 论文趋势")
        lines.append(report_summary.get("paper_summary", ""))
        lines.append("")

        lines.append("## 全网动态趋势")
        lines.append(report_summary.get("update_summary", ""))
        lines.append("")

        hot_topics = report_summary.get("hot_topics") or []
        if hot_topics:
            lines.append("## 热点标签")
            lines.append("、".join(hot_topics))
            lines.append("")

        key_takeaways = report_summary.get("key_takeaways") or []
        if key_takeaways:
            lines.append("## 关键结论")
            for item in key_takeaways:
                lines.append(f"- {item}")
            lines.append("")

        watchlist = report_summary.get("watchlist") or []
        if watchlist:
            lines.append("## 后续观察")
            for item in watchlist:
                lines.append(f"- {item}")
            lines.append("")

        trend_items = trend_summary.get("items") or []
        if trend_items:
            lines.append("## 趋势追踪")
            for item in trend_items:
                label = item.get("label") or item.get("name") or "趋势"
                summary = item.get("summary") or ""
                lines.append(f"- {label}: {summary}")
            lines.append("")

        self._render_card_block(
            lines,
            body_items,
            f"混排情报流（正文 {len(body_items)} 条）",
            "顶部重点已单独摘要，正文去掉了重复条目，论文与情报按重要性混排展示。",
        )

        if archive_summary.get("entries"):
            lines.append("## 历史归档")
            for entry in archive_summary["entries"][:10]:
                lines.append(f"- {entry['label']} | HTML: {entry['html_path']} | Markdown: {entry['markdown_path']}")
            lines.append("")

        return "\n".join(lines)
