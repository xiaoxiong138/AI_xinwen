from __future__ import annotations

import json
import os
import re
from collections import Counter
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from ..relevance import (
    category_to_cn,
    classify_paper_topic,
    clean_snippet,
    content_kind_to_cn,
    count_keyword_hits,
    infer_platform,
    is_ai_web_content,
    normalize_text,
    topic_to_cn,
)


class LLMProcessor:
    EVENT_JUDGMENT_CHOICES = ["same_event", "not_same_event", "unsure"]
    TITLE_REPLACEMENTS = [
        ("physical ai", "具身智能"),
        ("embodied ai", "具身智能"),
        ("world model", "世界模型"),
        ("world models", "世界模型"),
        ("robotics", "机器人"),
        ("robot", "机器人"),
        ("humanoid", "人形机器人"),
        ("foundation model", "基础模型"),
        ("vision-language-action", "视觉语言动作"),
        ("open source", "开源"),
        ("agent", "智能体"),
        ("agents", "智能体"),
        ("infrastructure", "基础设施"),
        ("video", "视频"),
        ("benchmark", "基准"),
        ("nvidia", "英伟达"),
        ("google", "谷歌"),
        ("microsoft", "微软"),
        ("amazon", "亚马逊"),
        ("apple", "苹果"),
        ("tesla", "特斯拉"),
    ]
    KEYWORD_MAP = [
        ("robotics", "机器人"),
        ("physical ai", "具身智能"),
        ("embodied ai", "具身智能"),
        ("world model", "世界模型"),
        ("agent", "智能体"),
        ("multimodal", "多模态"),
        ("open source", "开源"),
        ("inference", "推理"),
        ("training", "训练"),
        ("benchmark", "基准"),
    ]
    GENERIC_WHY_PATTERNS = [r"有助于快速把握", r"适合用于判断", r"反映了.+?(研究价值|应用潜力)", r"值得关注其.+?影响", r"帮助快速了解"]
    GENERIC_ANALYSIS_PATTERNS = [r"竞争加速期", r"抢占技术、市场或生态位置", r"推进效率", r"相关方向.+?随之变化", r"如果方法成立", r"可能改变.+?重点"]
    THIN_TITLE_PATTERNS = [r"^AI.+新进展$", r"^AI.+新动作$", r"^AI.+新变化$", r"^AI产品有新进展$", r"^AI基础设施有新动作$", r"^AI研究有新进展$"]
    PAPER_VALUE_MAP = {
        "具身智能": "如果这套方法能在真实环境中保持稳定，它会直接影响感知、规划与动作闭环是否真正走向可部署。",
        "世界模型": "这类研究决定模型能否先预测再行动，直接关系到试错成本、规划质量和系统稳定性。",
        "机器人": "真正的价值在于它能否把实验室里的性能提升，转化成现实任务中的成功率、效率和可靠性。",
        "其他": "它的价值不只在提出新想法，更在于是否给后续系统设计和实验验证提供了可复用的新路径。",
    }
    CATEGORY_IMPACT_MAP = {
        "模型/研究": "如果相关方法被更多团队复用，后续很可能快速外溢到模型训练、推理效率或评测方式。",
        "产品发布": "真正要看的不是发布动作本身，而是它能否带来用户采用、生态适配和竞品跟进。",
        "基础设施": "基础设施层的动作通常最先改变算力供给、训练成本和企业采用节奏，影响会向上游和下游同时传导。",
        "开源生态": "开源节奏会直接影响社区复现速度、工具链成熟度以及后续二次创新的上限。",
        "行业动态": "这类变化往往会先改变资本、合作和市场预期，再进一步影响产品节奏与技术路线。",
        "企业合作": "合作能否转化成真正的资源整合和商业落地，通常比发布消息本身更值得持续跟踪。",
        "社交讨论": "讨论热度背后往往对应着市场预期变化，值得结合后续产品和融资动作一起看。",
        "视频解读": "这类内容的价值在于帮助快速识别市场正在集中讨论哪些方向，以及哪些观点开始形成共识。",
        "应用落地": "真正的落地案例最能检验部署成本、客户接受度和规模化复制能力。",
        "其他": "如果这条信息持续发酵，通常会影响相关方向的关注度、资源配置和后续动作节奏。",
    }

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        api_key_env = config.get("api_key_env", "OPENAI_API_KEY")
        api_key = os.getenv(api_key_env)
        self.enabled = bool(api_key)
        self.model = config.get("model", "gpt-4o-mini")
        self.temperature = float(config.get("temperature", 0.3))
        self.client = OpenAI(api_key=api_key, base_url=config.get("base_url", "https://api.openai.com/v1")) if self.enabled else None
        if not self.enabled:
            print(f"Warning: API Key {api_key_env} not found. Falling back to heuristic processing.")

    def _clean_json_string(self, json_str: str) -> str:
        json_str = (json_str or "").strip()
        if json_str.startswith("```json"):
            json_str = json_str[7:]
        elif json_str.startswith("```"):
            json_str = json_str[3:]
        if json_str.endswith("```"):
            json_str = json_str[:-3]
        return json_str.strip()

    def _clean_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", text or "").strip()

    def _trim_clause(self, text: str, limit: int = 90) -> str:
        text = re.sub(r"[。！？；：]+$", "", self._clean_text(text))
        return text if len(text) <= limit else text[:limit].rstrip("，：; ") + "..."

    def _ensure_sentence(self, text: str) -> str:
        text = self._trim_clause(text, 120)
        return "" if not text else (text if text.endswith("。") else text + "。")

    def _split_sentences(self, text: str) -> List[str]:
        seen = set()
        result = []
        for chunk in re.split(r"(?<=[。！？])\s*", self._clean_text(text)):
            cleaned = self._trim_clause(chunk, 140)
            if len(cleaned) < 8:
                continue
            key = normalize_text(cleaned)
            if key in seen:
                continue
            seen.add(key)
            result.append(cleaned)
        return result

    def _limit_sentences(self, sentences: List[str], min_count: int = 3, max_count: int = 4, max_chars: int = 190) -> str:
        selected: List[str] = []
        for sentence in sentences[:max_count]:
            candidate = "".join(self._ensure_sentence(item) for item in selected + [sentence])
            if selected and len(candidate) > max_chars and len(selected) >= min_count:
                break
            selected.append(sentence)
        while len(selected) > min_count and len("".join(self._ensure_sentence(item) for item in selected)) > max_chars:
            selected.pop()
        return "".join(self._ensure_sentence(item) for item in selected[:max_count])

    def _looks_chinese_enough(self, text: str) -> bool:
        chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text or ""))
        latin_chars = len(re.findall(r"[A-Za-z]", text or ""))
        return chinese_chars >= max(6, latin_chars)

    def _contains_any(self, text: str, tokens: List[str]) -> bool:
        lowered = (text or "").lower()
        return any(token.lower() in lowered for token in tokens)

    def _looks_thin_title(self, text: str) -> bool:
        text = self._clean_text(text)
        return len(text) < 15 or any(re.search(pattern, text) for pattern in self.THIN_TITLE_PATTERNS)

    def _looks_generic_why(self, text: str) -> bool:
        text = self._clean_text(text)
        return len(text) < 24 or any(re.search(pattern, text) for pattern in self.GENERIC_WHY_PATTERNS)

    def _looks_generic_analysis(self, text: str) -> bool:
        text = self._clean_text(text)
        return len(text) < 20 or any(re.search(pattern, text) for pattern in self.GENERIC_ANALYSIS_PATTERNS)

    def _strip_title_suffix(self, title: str) -> str:
        title = self._clean_text(title)
        for separator in [" - ", " | ", " — ", " – "]:
            parts = [part.strip() for part in title.split(separator) if part.strip()]
            if len(parts) >= 2 and len(parts[-1]) <= 28:
                return separator.join(parts[:-1]).strip()
        return title

    def _extract_subject(self, article: Dict[str, Any]) -> str:
        text = self._clean_text(article.get("title_cn") or "") or self._strip_title_suffix(article.get("title", ""))
        for source, target in self.TITLE_REPLACEMENTS:
            text = re.sub(source, target, text, flags=re.IGNORECASE)
        text = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9（）()·\-\s]", " ", text)
        text = self._clean_text(text)
        for marker in ["宣布", "推出", "发布", "完成", "投资", "收购", "合作", "通过", "披露", "将"]:
            if marker in text:
                candidate = self._clean_text(text.split(marker, 1)[0])
                if 2 <= len(candidate) <= 20 and self._looks_chinese_enough(candidate):
                    return candidate
        for chunk in re.split(r"[：:，,。！？!（）()\-]", text):
            chunk = self._clean_text(chunk)
            if 4 <= len(chunk) <= 24 and self._looks_chinese_enough(chunk):
                return chunk
        return text[:20] if text and self._looks_chinese_enough(text) else "相关机构"

    def _source_label(self, article: Dict[str, Any]) -> str:
        source = self._clean_text(article.get("source_detail") or article.get("source") or article.get("platform") or "")
        mapping = {"Google News": "行业媒体", "Arxiv": "ArXiv", "ArXiv": "ArXiv", "Web": "网络来源", "Blog": "博客来源", "YouTube": "视频内容", "X": "社交平台"}
        if source in mapping:
            return mapping[source]
        if self._looks_chinese_enough(source):
            return source
        return "论文来源" if article.get("content_type") == "paper" else "行业来源"

    def _summary_source_sentences(self, article: Dict[str, Any], text: str) -> List[str]:
        body = self._clean_text(text)
        title = self._clean_text(article.get("title", ""))
        if title:
            body = re.sub(re.escape(title), "", body, count=1, flags=re.IGNORECASE)
        sentences = [s for s in self._split_sentences(body) if self._looks_chinese_enough(s)]
        if not sentences and title:
            sentences = [s for s in self._split_sentences(title) if self._looks_chinese_enough(s)]
        return sentences

    def _paper_value_line(self, topic_cn: str) -> str:
        return self.PAPER_VALUE_MAP.get(topic_cn, self.PAPER_VALUE_MAP["其他"])

    def _category_impact_line(self, category_cn: str) -> str:
        return self.CATEGORY_IMPACT_MAP.get(category_cn, self.CATEGORY_IMPACT_MAP["其他"])

    def _refine_title_text(self, text: str, min_len: int = 16, max_len: int = 34) -> str:
        text = re.sub(r"^(这篇论文|本文|这条全网AI动态|这条内容|文章|该研究|该动态|公开信息显示|从当前信息看)[：:，, ]*", "", self._clean_text(text))
        text = re.sub(r"[。！？]+$", "", text).strip("，：; ")
        if not text or not self._looks_chinese_enough(text) or len(text) < min_len:
            return ""
        if len(text) <= max_len:
            return text
        clauses = [self._clean_text(chunk) for chunk in re.split(r"[，,]", text) if self._clean_text(chunk)]
        compact = ""
        for clause in clauses:
            candidate = clause if not compact else f"{compact}，{clause}"
            if len(candidate) > max_len:
                break
            compact = candidate
        return compact if compact and len(compact) >= min_len else text[: max_len - 1].rstrip("，：; ") + "…"

    def _title_from_summary(self, summary: str) -> str:
        sentences = self._split_sentences(summary)
        if not sentences:
            return ""
        title = self._refine_title_text(sentences[0], 16, 34)
        if title:
            return title
        if len(sentences) >= 2:
            return self._refine_title_text(f"{self._trim_clause(sentences[0], 18)}，{self._trim_clause(sentences[1], 14)}", 16, 34)
        return ""

    def _summary_preview(self, summary: str) -> str:
        sentences = self._split_sentences(summary)
        if not sentences:
            return ""
        preview = self._trim_clause(sentences[0], 34)
        if len(preview) < 20 and len(sentences) >= 2:
            preview = self._trim_clause(f"{preview}，{self._trim_clause(sentences[1], 14)}", 36)
        return self._ensure_sentence(preview)

    def _make_generic_cn_title(self, article: Dict[str, Any], category_cn: str, topic_cn: str) -> str:
        if article.get("content_type") == "paper":
            return f"{topic_cn}研究提出新方法"
        templates = {"产品发布": "AI产品出现新动作", "基础设施": "AI基础设施出现新动作", "开源生态": "开源AI生态出现新变化", "行业动态": "AI行业出现新变化", "企业合作": "AI合作出现新进展", "社交讨论": "AI讨论焦点更新", "视频解读": "AI视频解读更新", "应用落地": "AI落地案例更新", "模型/研究": "AI研究出现新进展"}
        return templates.get(category_cn, "AI动态出现新进展")

    def _heuristic_title_cn(self, article: Dict[str, Any], category: str, topic_cn: str, summary: str = "") -> str:
        category_cn = category_to_cn(category)
        translated = self._strip_title_suffix(article.get("title", ""))
        for source, target in self.TITLE_REPLACEMENTS:
            translated = re.sub(source, target, translated, flags=re.IGNORECASE)
        translated = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9（）()·\-\s]", " ", translated)
        translated_title = self._refine_title_text(translated, 16, 34)
        summary_title = self._title_from_summary(summary)
        if summary_title and not self._looks_thin_title(summary_title):
            return summary_title
        if translated_title and not self._looks_thin_title(translated_title):
            return translated_title
        return summary_title or translated_title or self._make_generic_cn_title(article, category_cn, topic_cn)

    def _editorialize_title(self, article: Dict[str, Any], title_cn: str, summary: str, category: str, topic_cn: str) -> str:
        candidate = self._clean_text(title_cn)
        if article.get("content_type") == "paper":
            if candidate and len(candidate) >= 16 and any(token in candidate for token in ["提出", "实现", "解决", "提升", "验证", "构建"]):
                return self._refine_title_text(candidate, 16, 34) or candidate
            summary_title = self._title_from_summary(summary)
            if summary_title:
                return summary_title
            subject = self._extract_subject(article)
            if subject and subject != "相关机构":
                return f"{subject}试图解决{topic_cn}关键瓶颈"
            return f"{topic_cn}研究提出更可用的新方法"

        if candidate and len(candidate) >= 16 and any(token in candidate for token in ["发布", "推出", "融资", "合作", "部署", "投资", "收购"]):
            return self._refine_title_text(candidate, 16, 34) or candidate
        summary_title = self._title_from_summary(summary)
        if summary_title:
            return summary_title
        subject = self._extract_subject(article)
        category_cn = category_to_cn(category)
        if subject and subject != "相关机构":
            return f"{subject}在{category_cn}方向有了新动作"
        return f"{category_cn}方向出现关键新动作"

    def _editorialize_preview(self, article: Dict[str, Any], preview: str, summary: str, category: str, topic_cn: str) -> str:
        candidate = self._clean_text(preview)
        if candidate and len(candidate) >= 18 and len(candidate) <= 52:
            return self._ensure_sentence(candidate)

        sentences = self._split_sentences(summary)
        if article.get("content_type") == "paper":
            if len(sentences) >= 2:
                return self._ensure_sentence(self._trim_clause(sentences[1], 42))
            return self._ensure_sentence(f"重点在于它如何改进{topic_cn}里的核心环节")

        if len(sentences) >= 2:
            return self._ensure_sentence(self._trim_clause(sentences[1], 42))
        category_cn = category_to_cn(category)
        return self._ensure_sentence(f"关键要看这次动作会先改变哪些产品、客户或成本结构")

    def _editorialize_title_v2(self, article: Dict[str, Any], title_cn: str, summary: str, category: str, topic_cn: str) -> str:
        candidate = self._clean_text(title_cn)
        subject = self._extract_subject(article)
        summary_title = self._title_from_summary(summary)
        generic_title_markers = ["关键瓶颈", "新的动作", "新动作", "相关机构", "论文来源", "方向出现", "值得跟进"]
        raw_text = normalize_text(article.get("title", ""), summary, candidate, str(category))
        paper_topic_label = topic_cn if self._looks_chinese_enough(topic_cn) and "?" not in topic_cn else {
            "Physical AI": "具身智能",
            "World Model": "世界模型",
            "Robotics": "机器人",
        }.get(str(article.get("topic") or category or ""), "相关论文")
        category_cn = category if self._looks_chinese_enough(category) and "?" not in category else category_to_cn(category)
        if not self._looks_chinese_enough(category_cn) or "?" in category_cn:
            category_cn = {
                "Product Release": "产品发布",
                "Open Source": "开源生态",
                "Infrastructure": "基础设施",
                "Industry": "行业动态",
                "Partnership": "企业合作",
                "Model/Research": "模型/研究",
                "Application": "应用落地",
                "Social": "社交讨论",
                "Video": "视频解读",
                "Other": "行业动态",
            }.get(str(category or ""), "行业动态")
        if any(token in raw_text for token in ["launch", "release", "workflow", "agent", "copilot", "推出", "发布", "上线"]):
            category_cn = "产品发布"
        elif any(token in raw_text for token in ["open source", "开源", "github"]):
            category_cn = "开源生态"
        elif any(token in raw_text for token in ["gpu", "chip", "cluster", "infra", "server", "算力", "芯片"]):
            category_cn = "基础设施"
        elif any(token in raw_text for token in ["partner", "collaboration", "joint", "合作"]):
            category_cn = "企业合作"

        if article.get("content_type") == "paper":
            strong_paper_tokens = ["提出", "实现", "解决", "提升", "验证", "构建", "统一", "降低", "加速"]
            if candidate and len(candidate) >= 16 and any(token in candidate for token in strong_paper_tokens) and not any(marker in candidate for marker in generic_title_markers):
                return self._refine_title_text(candidate, 16, 36) or candidate
            if summary_title and any(token in summary_title for token in strong_paper_tokens) and not any(marker in summary_title for marker in generic_title_markers):
                return self._refine_title_text(summary_title, 16, 36) or summary_title

            paper_templates = {
                "具身智能": f"{subject}尝试把具身方法推向真实任务" if subject and subject != "相关机构" else "具身智能研究尝试走向真实任务",
                "世界模型": f"{subject}尝试让世界模型更能先预测再行动" if subject and subject != "相关机构" else "世界模型研究尝试降低试错成本",
                "机器人": f"{subject}尝试提升机器人真实任务成功率" if subject and subject != "相关机构" else "机器人研究尝试走向更稳定实机",
            }
            styled = paper_templates.get(paper_topic_label, f"{paper_topic_label}研究提出更可部署的新方法")
            return self._refine_title_text(styled, 16, 36) or styled

        strong_update_tokens = ["发布", "推出", "融资", "合作", "部署", "投资", "收购", "开源", "上线", "切入", "扩张", "争夺"]
        if candidate and len(candidate) >= 16 and any(token in candidate for token in strong_update_tokens) and not any(marker in candidate for marker in generic_title_markers):
            return self._refine_title_text(candidate, 16, 36) or candidate
        if summary_title and any(token in summary_title for token in strong_update_tokens) and not any(marker in summary_title for marker in generic_title_markers):
            return self._refine_title_text(summary_title, 16, 36) or summary_title

        style_templates = {
            "产品发布": f"{subject}把AI能力推进到真实工作流" if subject and subject != "相关机构" else "AI产品开始切入更深的工作流环节",
            "企业合作": f"{subject}借合作补齐AI落地关键短板" if subject and subject != "相关机构" else "AI合作开始转向更深的资源整合",
            "开源生态": f"{subject}试图把方案做成社区默认选项" if subject and subject != "相关机构" else "开源AI生态继续争夺事实标准",
            "基础设施": f"{subject}继续争夺AI算力与部署入口" if subject and subject != "相关机构" else "AI基础设施竞争继续前移到入口层",
            "行业动态": f"{subject}正在重排资源与市场预期" if subject and subject != "相关机构" else "AI行业资源配置开始出现新变化",
            "应用落地": f"{subject}开始验证AI方案的业务复制能力" if subject and subject != "相关机构" else "AI应用开始进入真实落地验证阶段",
            "模型/研究": f"{subject}继续推进模型能力与效率边界" if subject and subject != "相关机构" else "模型能力和效率路线继续向前推进",
            "社交讨论": f"{subject}相关讨论正在放大市场预期差" if subject and subject != "相关机构" else "AI讨论热点开始影响市场预期",
            "视频解读": f"{subject}折射出行业关注点正在变化" if subject and subject != "相关机构" else "视频热点背后反映出行业关注点变化",
        }
        styled = style_templates.get(category_cn, f"{category_cn}方向出现更值得跟进的新动作")
        return self._refine_title_text(styled, 16, 36) or styled

    def _editorialize_preview_v2(self, article: Dict[str, Any], preview: str, summary: str, category: str, topic_cn: str) -> str:
        candidate = self._clean_text(preview)
        subject = self._extract_subject(article)
        generic_preview_markers = ["相关机构", "行业来源", "新的动作", "新动作", "关键瓶颈", "方向", "论文来源"]
        raw_text = normalize_text(article.get("title", ""), summary, candidate, str(category))
        paper_topic_label = topic_cn if self._looks_chinese_enough(topic_cn) and "?" not in topic_cn else {
            "Physical AI": "具身智能",
            "World Model": "世界模型",
            "Robotics": "机器人",
        }.get(str(article.get("topic") or category or ""), "相关论文")
        category_cn = category if self._looks_chinese_enough(category) and "?" not in category else category_to_cn(category)
        if not self._looks_chinese_enough(category_cn) or "?" in category_cn:
            category_cn = {
                "Product Release": "产品发布",
                "Open Source": "开源生态",
                "Infrastructure": "基础设施",
                "Industry": "行业动态",
                "Partnership": "企业合作",
                "Model/Research": "模型/研究",
                "Application": "应用落地",
                "Social": "社交讨论",
                "Video": "视频解读",
                "Other": "行业动态",
            }.get(str(category or ""), "行业动态")
        if any(token in raw_text for token in ["launch", "release", "workflow", "agent", "copilot", "推出", "发布", "上线"]):
            category_cn = "产品发布"
        elif any(token in raw_text for token in ["open source", "开源", "github"]):
            category_cn = "开源生态"
        elif any(token in raw_text for token in ["gpu", "chip", "cluster", "infra", "server", "算力", "芯片"]):
            category_cn = "基础设施"
        elif any(token in raw_text for token in ["partner", "collaboration", "joint", "合作"]):
            category_cn = "企业合作"
        if candidate and len(candidate) >= 18 and len(candidate) <= 52:
            if SequenceMatcher(None, normalize_text(candidate), normalize_text(clean_snippet(summary, 70))).ratio() < 0.88 and not any(marker in candidate for marker in generic_preview_markers):
                return self._ensure_sentence(candidate)

        sentences = self._split_sentences(summary)
        if article.get("content_type") == "paper":
            paper_preview_templates = {
                "具身智能": "关键看它能否把感知、规划和动作闭环做得更稳。",
                "世界模型": "重点不只是生成效果，而是能否把试错成本压到更低。",
                "机器人": "更重要的是它是否能把实验结果转成真实任务成功率。",
            }
            return paper_preview_templates.get(paper_topic_label, "关键看这套方法能否从论文结果走向真实部署。")

        preview_templates = {
            "产品发布": "更值得看的是它会先替代哪些高频人工步骤。",
            "企业合作": "真正的看点是合作能否转成客户、渠道或交付增量。",
            "开源生态": "后续重点是社区会不会围绕它形成新的默认做法。",
            "基础设施": "这类动作通常最先改写成本、供给和平台绑定关系。",
            "行业动态": "更关键的是这条消息会不会继续传导到资本和产品节奏。",
            "应用落地": "核心不是案例本身，而是它能否被复制到更多真实业务里。",
            "模型/研究": "更重要的是能力提升会不会兑现成更低成本和更稳交付。",
            "社交讨论": "需要继续看讨论热度会不会变成真实产品和融资动作。",
            "视频解读": "更值得看的是视频背后的观点是否很快被行业验证。",
        }
        if category_cn not in preview_templates and len(sentences) >= 2:
            second = self._trim_clause(sentences[1], 46)
            if second and SequenceMatcher(None, normalize_text(second), normalize_text(article.get("title_cn", ""))).ratio() < 0.82 and not any(marker in second for marker in generic_preview_markers):
                return self._ensure_sentence(second)
        default_preview = f"{subject}这次动作更值得看它会先改变哪些客户、产品或成本结构。" if subject and subject != "相关机构" else "更值得看这次动作会先改变哪些客户、产品或成本结构。"
        return preview_templates.get(category_cn, default_preview)

    def _heuristic_category(self, article: Dict[str, Any], text: str) -> str:
        if article.get("content_type", "news") == "paper":
            topic, _ = classify_paper_topic(article.get("title", ""), text)
            return topic
        platform = article.get("platform") or infer_platform(article.get("url", ""), article.get("source_detail", ""))
        if platform == "YouTube":
            return "视频解读"
        if platform == "X":
            return "社交讨论"
        if any(token in text for token in ["release", "launch", "推出", "发布"]):
            return "产品发布"
        if any(token in text for token in ["open source", "开源"]):
            return "开源生态"
        if any(token in text for token in ["funding", "acquire", "partnership", "融资", "收购"]):
            return "行业动态"
        if any(token in text for token in ["cooperate", "collaboration", "合作", "joint"]):
            return "企业合作"
        if any(token in text for token in ["gpu", "chip", "infra", "server", "训练集群", "算力"]):
            return "基础设施"
        if any(token in text for token in ["deploy", "application", "workflow", "落地", "上线", "use case"]):
            return "应用落地"
        return "模型/研究"

    def _heuristic_score(self, article: Dict[str, Any], text: str, category: str) -> float:
        score = max(float(article.get("score", 5.0) or 5.0), float(article.get("initial_score", 0) or 0))
        score += count_keyword_hits(text, ["gpt", "claude", "gemini", "robot", "world model", "foundation model"]) * 0.4
        if article.get("content_type") == "paper":
            score += 2.0 + (1.5 if category in {"Physical AI", "Robotics", "World Model"} else 0.0)
        if article.get("platform") == "YouTube":
            score += 0.2
        if article.get("platform") == "X":
            score += 0.1
        return round(min(score, 10.0), 1)

    def _heuristic_keywords(self, article: Dict[str, Any], text: str, category: str) -> List[str]:
        base = []
        category_cn = topic_to_cn(category) if article.get("content_type") == "paper" else category_to_cn(category)
        if category_cn and category_cn != "其他":
            base.append(category_cn)
        if article.get("platform"):
            base.append(article["platform"])
        for token, label in self.KEYWORD_MAP:
            if token in text and label not in base:
                base.append(label)
            if len(base) >= 5:
                break
        return base[:5] or ["AI"]

    def _paper_analysis_bundle(self, article: Dict[str, Any], topic_cn: str, summary: str) -> Dict[str, str]:
        text = self._clean_text(f"{article.get('title_cn', '')} {article.get('title', '')} {summary}")
        if self._contains_any(text, ["数据集", "benchmark", "基准", "评测", "标注"]):
            return {"why_it_matters": "它补的是数据和评测底座，后续同方向模型是否真的有效，很可能都要拿这类基准来比较。", "why_now": "因为该方向过去缺少统一且高质量的标注或评测方法，很多结果难以横向比较，也难判断是否具备真实泛化能力。", "expected_effect": "它会先提升研究复现和横向对比效率，让模型改进不再只停留在单篇论文里的局部指标。", "future_impact": "如果被广泛采用，相关领域的论文会逐步围绕这套数据或基准展开，评测标准也会更快收敛。"}
        if self._contains_any(text, ["量化", "压缩", "推理", "kv", "cache", "memory", "速度", "latency", "吞吐"]):
            return {"why_it_matters": "它直击的是推理成本和吞吐瓶颈，这类改进通常比单纯提分更快进入真实部署链路。", "why_now": "因为模型上下文变长、推理请求变密之后，内存和带宽已经成为部署长文本与多轮任务的核心约束。", "expected_effect": "它会先体现在更低的显存占用、更短的响应延迟，或者同等硬件下可承载更多并发请求。", "future_impact": "这类效率路线一旦稳定，会推动长上下文、边缘部署和低成本推理方案更快落地。"}
        if self._contains_any(text, ["仿真", "physics", "物理", "world model", "世界模型", "sim2real"]):
            return {"why_it_matters": "它解决的是仿真可信度和物理一致性问题，这是世界模型能否进入机器人和具身任务的硬门槛。", "why_now": "因为当前不少世界模型在视觉上像真，但一进入长时预测、接触过程或操控任务就会暴露物理失真。", "expected_effect": "它会先提升模型对运动、接触和状态演化的建模可靠性，减少只会生成画面但不支撑决策的问题。", "future_impact": "如果这一方向持续有效，世界模型会从生成演示走向任务规划、仿真训练和实机迁移的中间层能力。"}
        if self._contains_any(text, ["抓取", "操控", "policy", "策略", "控制", "规划", "manipulation", "action"]):
            return {"why_it_matters": "它指向机器人策略的核心短板，即精细操作、跨场景泛化和样本效率，这比单一场景提分更重要。", "why_now": "因为机器人系统开始从固定演示任务走向开放环境，旧方法在泛化、稳定性和训练代价上都开始吃紧。", "expected_effect": "它会先体现在更高的任务成功率、更快的训练收敛，或者更低的实机试错成本。", "future_impact": "这类方法一旦稳定，机器人训练会更少依赖大量人工示教，并推动策略学习向可部署阶段逼近。"}
        return {"why_it_matters": self._paper_value_line(topic_cn), "why_now": f"因为现有{topic_cn}方法在真实任务、物理合理性或计算效率上仍有明显瓶颈，研究者需要提出更可用的新方案。", "expected_effect": f"它会先影响{topic_cn}任务中的训练效率、结果稳定性或部署可行性，而不是只停留在线指标改进。", "future_impact": f"如果后续复现结果稳定，{topic_cn}方向的数据构造、模型设计和评测方式都可能被重新定义。"}

    def _update_analysis_bundle(self, article: Dict[str, Any], category_cn: str, summary: str) -> Dict[str, str]:
        subject = self._extract_subject(article)
        text = self._clean_text(f"{article.get('title_cn', '')} {article.get('title', '')} {summary}")
        if category_cn == "基础设施" or self._contains_any(text, ["chip", "gpu", "npu", "server", "算力", "芯片", "集群", "数据中心"]):
            return {"why_it_matters": "这类动作争夺的是算力供给和部署入口，通常会比上层应用更早改写成本结构和生态主导权。", "why_now": f"因为AI训练与推理需求持续抬升，{subject if subject != '相关机构' else '相关厂商'}需要提前锁定芯片、集群或平台入口，避免被卡在供给瓶颈上。", "expected_effect": "它会先体现在算力供给更稳、部署栈更统一，或者单位推理成本和系统延迟进一步下降。", "future_impact": "后续很可能传导到云厂商采购、开发框架绑定和企业选型，最终改变基础设施层的议价权分布。"}
        if category_cn == "产品发布" and self._contains_any(text, ["agent", "助手", "电脑", "browser", "workflow", "自动化", "copilot"]):
            return {"why_it_matters": "这不是普通功能更新，而是在把模型能力从问答界面推进到真实工作流执行层，直接关系到用户采用深度。", "why_now": f"因为{subject if subject != '相关机构' else '厂商'}需要证明模型不仅会回答，还能接入软件、浏览器或桌面任务并真正替代部分人工流程。", "expected_effect": "它会先提升自动化覆盖范围，把原本需要人工点选和切换的流程交给模型完成，从而拉高产品粘性。", "future_impact": "一旦用户形成依赖，竞品会被迫跟进类似能力，人机协作界面也会从聊天窗口继续转向任务代理。"}
        if category_cn == "产品发布":
            return {"why_it_matters": "产品动作是否能把技术能力转成可用体验，往往比单次模型指标更能说明一家公司的真实竞争力。", "why_now": f"因为{subject if subject != '相关机构' else '团队'}需要尽快把能力封装成可交付产品，抢在市场教育完成前占住用户入口。", "expected_effect": "它会先推动开发者试用、客户验证和内部集成，决定这项能力能否从发布会走向持续使用。", "future_impact": "如果采用顺利，后续价格策略、产品形态和生态合作都会围绕这条产品线继续展开。"}
        if category_cn == "企业合作":
            return {"why_it_matters": "合作值不值得看，关键不在官宣本身，而在它是否补齐了客户、渠道、数据或交付能力中的关键短板。", "why_now": f"因为{subject if subject != '相关机构' else '合作双方'}单靠自身很难同时补足技术、销售和交付链条，需要借合作缩短落地路径。", "expected_effect": "它会先提高资源对接和联合交付效率，让方案更快进入真实客户或业务场景。", "future_impact": "如果合作深入，后续生态分工、渠道归属和行业标准话语权都可能被重新排列。"}
        if category_cn == "开源生态":
            return {"why_it_matters": "开源动作决定的是社区采用速度和事实标准的形成节奏，往往比单家厂商公告更能放大长期影响。", "why_now": "因为社区正在寻找可复用、可二次开发的底层方案，谁先形成工具链和样板，谁就更容易成为默认路线。", "expected_effect": "它会先降低试用门槛和二次开发成本，让更多团队围绕同一套接口、模型或框架快速迭代。", "future_impact": "一旦生态形成网络效应，后续插件、服务和商业支持都会向这条路线集中。"}
        return {"why_it_matters": self._category_impact_line(category_cn), "why_now": f"因为{subject if subject != '相关机构' else '相关团队'}需要在当前窗口期尽快证明路线有效，并争取更多资源和市场注意力。", "expected_effect": "它会先影响产品推进速度、资源整合效率或客户采用节奏，而不是停留在消息层面。", "future_impact": f"如果后续动作持续，{category_cn}方向的资源配置、竞争策略和落地路径都会被进一步拉开差距。"}

    def _analysis_bundle_v2(self, article: Dict[str, Any], category: str, topic_cn: str, summary: str) -> Dict[str, str]:
        raw_text = normalize_text(article.get("title", ""), article.get("title_cn", ""), summary, str(category), str(topic_cn))
        subject = self._extract_subject(article)

        if article.get("content_type") == "paper":
            paper_topic_label = topic_cn if self._looks_chinese_enough(topic_cn) and "?" not in topic_cn else {
                "Physical AI": "具身智能",
                "World Model": "世界模型",
                "Robotics": "机器人",
            }.get(str(article.get("topic") or category or ""), "相关论文")

            if paper_topic_label == "世界模型":
                return {
                    "why_it_matters": "这类研究决定世界模型能不能从会生成画面，继续走到会支撑规划、控制和低试错决策。",
                    "why_now": "因为世界模型已经不缺演示效果，真正短缺的是在长时预测、闭环控制和真实环境里保持一致性的能力。",
                    "expected_effect": "它会先影响模型是否能先预测再行动，并把试错成本从真实系统前移到更便宜的模拟与潜空间里。",
                    "future_impact": "如果这条路线持续成立，世界模型会更快进入机器人、自动驾驶和复杂任务规划的中间层栈。",
                }
            if paper_topic_label == "具身智能":
                return {
                    "why_it_matters": "具身方向真正稀缺的不是单次提分，而是把感知、规划和动作闭环稳定带进真实环境。",
                    "why_now": "因为具身系统正从演示型任务走向开放场景，旧方法在稳定性、样本效率和部署成本上越来越吃紧。",
                    "expected_effect": "它会先改变真实任务中的成功率、训练收敛速度，或者减少实机试错和人工示教负担。",
                    "future_impact": "一旦可复现，这类方法会推动具身系统从实验室 demo 更快过渡到可持续部署阶段。",
                }
            if paper_topic_label == "机器人":
                return {
                    "why_it_matters": "机器人论文的价值最终取决于它能不能把实验指标转成实机成功率、鲁棒性和更低交付成本。",
                    "why_now": "因为机器人系统开始面对更开放的任务和环境，原有依赖固定脚本或高人工示教的路线扩展性越来越差。",
                    "expected_effect": "它会先反映在更高的任务完成率、更少的人工介入，或更低的训练与标注代价。",
                    "future_impact": "如果后续验证稳定，机器人训练和部署链路都会从重工程调参转向更标准化的能力栈。",
                }
            return self._paper_analysis_bundle(article, paper_topic_label, summary)

        category_cn = category if self._looks_chinese_enough(category) and "?" not in category else category_to_cn(category)
        if not self._looks_chinese_enough(category_cn) or "?" in category_cn:
            category_cn = {
                "Product Release": "产品发布",
                "Open Source": "开源生态",
                "Infrastructure": "基础设施",
                "Industry": "行业动态",
                "Partnership": "企业合作",
                "Model/Research": "模型/研究",
                "Application": "应用落地",
                "Social": "社交讨论",
                "Video": "视频解读",
                "Other": "行业动态",
            }.get(str(category or ""), "行业动态")

        if any(token in raw_text for token in ["launch", "release", "workflow", "agent", "copilot", "browser", "推出", "发布", "上线"]):
            category_cn = "产品发布"
        elif any(token in raw_text for token in ["open source", "开源", "github"]):
            category_cn = "开源生态"
        elif any(token in raw_text for token in ["gpu", "chip", "cluster", "infra", "server", "算力", "芯片"]):
            category_cn = "基础设施"
        elif any(token in raw_text for token in ["partner", "collaboration", "joint", "合作"]):
            category_cn = "企业合作"
        elif any(token in raw_text for token in ["funding", "investment", "acquire", "融资", "投资", "收购"]):
            category_cn = "行业动态"

        update_templates = {
            "产品发布": {
                "why_it_matters": "产品发布最值得看的不是功能名，而是它有没有把模型能力推进到更深的真实工作流和付费入口。",
                "why_now": f"因为{subject if subject != '相关机构' else '厂商'}需要尽快证明模型不只会回答问题，还能直接接管更多高频执行步骤。",
                "expected_effect": "它会先减少人工切换、重复操作和流程摩擦，把更多原本手动完成的环节纳入自动化链路。",
                "future_impact": "如果用户形成依赖，后续竞争会从模型能力本身扩展到工作流入口、集成深度和默认使用习惯。",
            },
            "企业合作": {
                "why_it_matters": "合作真正的价值在于它是否补齐了客户、渠道、数据、交付或行业入口中的关键短板。",
                "why_now": f"因为{subject if subject != '相关机构' else '合作双方'}单靠自己很难同时补足技术、销售和落地链条，需要借合作缩短进入真实客户的路径。",
                "expected_effect": "它会先体现在资源对接更快、联合交付更顺，或者方案更容易进入已有客户和行业场景。",
                "future_impact": "如果合作不断加深，后续生态分工、渠道归属和行业标准的话语权都可能被重新排序。",
            },
            "开源生态": {
                "why_it_matters": "开源动作会更快决定社区默认工具链和事实标准，长期影响往往大于一次单点官宣。",
                "why_now": "因为社区正在寻找更可复用、可二次开发且成本可控的底层方案，谁先做成样板，谁就更容易变成默认路线。",
                "expected_effect": "它会先降低试用门槛和二次开发成本，让更多团队围绕同一接口、框架或模型快速迭代。",
                "future_impact": "一旦形成网络效应，后续插件、服务、托管和商业支持都会向这条开源路线集中。",
            },
            "基础设施": {
                "why_it_matters": "基础设施动作争夺的是算力供给、部署入口和平台绑定关系，通常最先改写整个行业的成本底座。",
                "why_now": f"因为{subject if subject != '相关机构' else '相关厂商'}必须在需求持续上行之前先锁定芯片、集群、云资源或平台入口。",
                "expected_effect": "它会先反映在供给更稳、部署栈更统一，或者单位训练与推理成本进一步下降。",
                "future_impact": "后续会继续传导到云采购、框架绑定、企业选型和生态议价权，影响会从底层向上扩散。",
            },
            "行业动态": {
                "why_it_matters": "行业级动作往往先改写资本预期、资源配置和叙事方向，再进一步影响产品节奏和合作边界。",
                "why_now": f"因为{subject if subject != '相关机构' else '相关团队'}需要在窗口期快速证明路线成立，并争取更多预算、注意力和外部资源。",
                "expected_effect": "它会先改变市场关注点、资源流向，或者让同类公司加快跟进与防御动作。",
                "future_impact": "如果后续消息持续累积，同方向的融资、合作、产品节奏和竞争格局都会被继续拉开。",
            },
            "应用落地": {
                "why_it_matters": "落地类信息最重要的是它是否证明方案能跨场景复制，而不只是停留在单个样板案例。",
                "why_now": f"因为{subject if subject != '相关机构' else '相关团队'}需要尽快把技术能力变成可衡量的业务结果，证明这条路线不是实验性质。",
                "expected_effect": "它会先带来更清晰的 ROI、客户验证或流程缩短，让方案更容易进入下一轮扩张。",
                "future_impact": "如果复制顺利，后续行业会更快出现模板化解决方案和更明确的采购标准。",
            },
            "模型/研究": {
                "why_it_matters": "研究类动态真正重要的是它会不会把能力提升兑现成更低成本、更稳部署或更明确的产品方向。",
                "why_now": f"因为{subject if subject != '相关机构' else '相关团队'}需要把研究优势尽快转成可感知的效率、能力或场景差异。",
                "expected_effect": "它会先影响模型路线、评测重点或产品封装方式，而不只是增加一条研究新闻。",
                "future_impact": "如果后续被反复验证，这类研究路线会逐步进入主流产品和基础模型迭代节奏。",
            },
            "社交讨论": {
                "why_it_matters": "社交讨论的价值不在热度本身，而在它是否提前暴露出行业预期差、争议点和下一步动作方向。",
                "why_now": "因为讨论层经常先于正式发布和融资动作，提前反映市场正在集中关注什么、担心什么。",
                "expected_effect": "它会先影响从业者注意力、舆论焦点和产品叙事，进而推动更多正式动作出现。",
                "future_impact": "如果讨论持续升温，后续往往会被验证成新的热点赛道、争议标准或资本叙事。",
            },
            "视频解读": {
                "why_it_matters": "视频类内容的关键不在流量，而在它是否把分散信号整理成行业正在验证的共同判断。",
                "why_now": "因为市场信息过载时，视频往往会先放大哪些方向开始形成共识、哪些观点正在被反复提及。",
                "expected_effect": "它会先提升某些方向的可见度和讨论密度，让更多团队围绕同类问题给出公开回应。",
                "future_impact": "如果后续有更多正式产品、论文或融资动作跟上，这类视频信号就会变成更明确的趋势线索。",
            },
        }
        return update_templates.get(category_cn, self._update_analysis_bundle(article, category_cn, summary))

    def _heuristic_summary(self, article: Dict[str, Any], text: str, category: str, topic_cn: str) -> str:
        source = self._source_label(article)
        subject = self._extract_subject(article)
        snippet_sentences = self._summary_source_sentences(article, clean_snippet(text, 260))
        if article.get("content_type") == "paper":
            detail_1 = snippet_sentences[0] if snippet_sentences else f"公开信息显示，这项工作围绕{topic_cn}中的关键问题提出了新的处理思路。"
            detail_2 = snippet_sentences[1] if len(snippet_sentences) > 1 else "作者重点描述了方法设计、任务场景以及实验验证方式，希望证明方案在具体任务中的有效性。"
            return self._limit_sentences([f"这项来自{source}的{topic_cn}研究，重点解决真实系统中的关键瓶颈。", detail_1, detail_2, self._paper_value_line(topic_cn)])
        category_cn = category_to_cn(category)
        detail_1 = snippet_sentences[0] if snippet_sentences else f"{source}披露了{category_cn}方向的一项新动作。"
        detail_2 = snippet_sentences[1] if len(snippet_sentences) > 1 else "从当前信息看，这次变化不只是单点更新，更可能影响相关产品路线、合作节奏或市场竞争。"
        return self._limit_sentences([f"{subject}在{category_cn}方向出现了新的动作。", detail_1, detail_2, self._category_impact_line(category_cn)])

    def _heuristic_why_it_matters(self, article: Dict[str, Any], category: str, topic_cn: Optional[str] = None, summary: str = "") -> str:
        category_cn = topic_cn or (topic_to_cn(category) if article.get("content_type") == "paper" else category_to_cn(category))
        bundle = self._analysis_bundle_v2(article, category, category_cn, summary)
        return self._ensure_sentence(bundle["why_it_matters"])

    def _heuristic_analysis(self, article: Dict[str, Any], category: str, topic_cn: str, summary: str) -> Dict[str, str]:
        category_cn = topic_cn if article.get("content_type") == "paper" else category_to_cn(category)
        bundle = self._analysis_bundle_v2(article, category, category_cn, summary)
        return {key: self._ensure_sentence(value) for key, value in bundle.items() if key in {"why_now", "expected_effect", "future_impact"}}

    def _normalize_summary_output(self, summary: str, article: Dict[str, Any], category: str, topic_cn: str) -> str:
        summary = re.sub(r"^(这篇论文|本文|这条动态|这条全网AI动态|文章|该研究|该动态)[：:，, ]*", "", self._clean_text(summary))
        sentences = self._split_sentences(summary)
        title = self._strip_title_suffix(article.get("title", ""))
        if title and sentences and SequenceMatcher(None, normalize_text(title), normalize_text(sentences[0])).ratio() >= 0.84 and len(sentences) > 1:
            sentences = sentences[1:]
        compact = self._limit_sentences(sentences)
        if len(compact) >= 70 and len(self._split_sentences(compact)) >= 2:
            return compact
        fallback_text = self._clean_text(f"{article.get('title', '')} {article.get('content', '')}")
        return self._heuristic_summary(article, fallback_text, category, topic_cn)

    def _normalize_why_output(self, why_it_matters: str, article: Dict[str, Any], category: str, topic_cn: str, summary: str) -> str:
        why_it_matters = re.sub(r"^(值得关注|为什么值得看)[：: ]*", "", self._clean_text(why_it_matters))
        why_it_matters = self._trim_clause(why_it_matters, 78)
        if not why_it_matters or self._looks_generic_why(why_it_matters) or SequenceMatcher(None, normalize_text(why_it_matters), normalize_text(clean_snippet(summary, 90))).ratio() >= 0.72:
            return self._heuristic_why_it_matters(article, category, topic_cn, summary)
        return self._ensure_sentence(why_it_matters)

    def _normalize_analysis_output(self, value: str, fallback: str, summary: str = "", max_len: int = 72) -> str:
        value = re.sub(r"^(为什么这么做|会实现什么效果|会带来什么效果|未来有什么影响|未来影响)[：: ]*", "", self._clean_text(value))
        value = self._trim_clause(value, max_len)
        if not value or self._looks_generic_analysis(value) or (summary and SequenceMatcher(None, normalize_text(value), normalize_text(clean_snippet(summary, 90))).ratio() >= 0.72):
            return self._ensure_sentence(fallback)
        return self._ensure_sentence(value)

    def _normalize_preview_output(self, preview: str, summary: str) -> str:
        preview = re.sub(r"^(副标题|提要|摘要导语)[：: ]*", "", self._clean_text(preview))
        preview = self._trim_clause(preview, 52)
        if not preview or len(preview) < 18 or not self._looks_chinese_enough(preview) or SequenceMatcher(None, normalize_text(preview), normalize_text(clean_snippet(summary, 70))).ratio() >= 0.92:
            return self._summary_preview(summary)
        return self._ensure_sentence(preview)

    def _normalize_single_result(self, article: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(result)
        raw_category = merged.get("category") or article.get("category") or article.get("topic") or "Other"
        content_type = article.get("content_type", "news")
        if content_type == "paper" and raw_category not in {"Physical AI", "World Model", "Robotics"}:
            raw_category = article.get("topic") or raw_category
        topic_cn = merged.get("topic_cn") or (topic_to_cn(raw_category) if content_type == "paper" else category_to_cn(raw_category))
        summary = self._normalize_summary_output(merged.get("summary") or "", article, raw_category, topic_cn)
        title_cn = self._clean_text(merged.get("title_cn") or "")
        if not title_cn or self._looks_thin_title(title_cn):
            title_cn = self._heuristic_title_cn(article, raw_category, topic_cn, summary)
        title_cn = self._editorialize_title_v2(article, title_cn, summary, raw_category, topic_cn)
        article_with_title = dict(article)
        article_with_title["title_cn"] = title_cn
        why_it_matters = self._normalize_why_output(merged.get("why_it_matters", ""), article_with_title, raw_category, topic_cn, summary)
        heuristic_analysis = self._heuristic_analysis(article_with_title, raw_category, topic_cn, summary)
        keywords = merged.get("keywords") or self._heuristic_keywords(article, normalize_text(article.get("title", ""), article.get("content", "")), raw_category)
        if not isinstance(keywords, list):
            keywords = [token.strip() for token in str(keywords).split(",") if token.strip()]
        summary_preview = self._normalize_preview_output(merged.get("summary_preview", ""), summary)
        summary_preview = self._editorialize_preview_v2(article_with_title, summary_preview, summary, raw_category, topic_cn)
        return {
            "title_cn": title_cn,
            "summary": summary,
            "score": float(merged.get("score", article.get("score", 0)) or 0),
            "keywords": keywords[:5],
            "category": raw_category,
            "topic_cn": topic_cn,
            "content_kind": content_kind_to_cn(content_type),
            "why_it_matters": why_it_matters,
            "why_now": self._normalize_analysis_output(merged.get("why_now", ""), heuristic_analysis["why_now"], summary),
            "expected_effect": self._normalize_analysis_output(merged.get("expected_effect", ""), heuristic_analysis["expected_effect"], summary),
            "future_impact": self._normalize_analysis_output(merged.get("future_impact", ""), heuristic_analysis["future_impact"], summary),
            "summary_preview": summary_preview,
        }

    def _fallback_process(self, article: Dict[str, Any]) -> Dict[str, Any]:
        title = self._clean_text(article.get("title", ""))
        content = self._clean_text(article.get("content", ""))
        text = self._clean_text(f"{title} {content}")
        category = "Other" if article.get("content_type") != "paper" and not is_ai_web_content(title, content) else self._heuristic_category(article, text.lower())
        topic_cn = topic_to_cn(category) if article.get("content_type") == "paper" else category_to_cn(category)
        summary = self._heuristic_summary(article, text, category, topic_cn)
        return self._normalize_single_result(article, {"title_cn": self._heuristic_title_cn(article, category, topic_cn, summary), "summary": summary, "score": self._heuristic_score(article, text.lower(), category), "keywords": self._heuristic_keywords(article, text.lower(), category), "category": category, "topic_cn": topic_cn})

    def prepare_report_item(self, article: Dict[str, Any], runtime_result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        base = dict(article)
        normalized = self._normalize_single_result(base, dict(runtime_result or {}) or base)
        base.update(normalized)
        base["display_topic"] = normalized["topic_cn"] if base.get("content_type") == "paper" else category_to_cn(normalized["category"])
        return base

    def _pairwise_similarity_features(self, left: Dict[str, Any], right: Dict[str, Any]) -> Tuple[float, int]:
        left_title = normalize_text(left.get("title", ""), left.get("title_cn", ""))
        right_title = normalize_text(right.get("title", ""), right.get("title_cn", ""))
        return SequenceMatcher(None, left_title, right_title).ratio(), len(set(re.split(r"\W+", left_title)) & set(re.split(r"\W+", right_title)))

    def _heuristic_event_similarity(self, left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, str]:
        title_ratio, overlap = self._pairwise_similarity_features(left, right)
        summary_ratio = SequenceMatcher(None, normalize_text(left.get("summary", ""), left.get("content", ""))[:300], normalize_text(right.get("summary", ""), right.get("content", ""))[:300]).ratio()
        left_topic = left.get("display_topic") or left.get("topic_cn") or ""
        right_topic = right.get("display_topic") or right.get("topic_cn") or ""
        if title_ratio >= 0.94 or (title_ratio >= 0.86 and overlap >= 4 and summary_ratio >= 0.7):
            return {"decision": "same_event", "reason": "标题和摘要高度相似，基本可视为同一事件。"}
        if title_ratio <= 0.55 and summary_ratio <= 0.45 and left_topic != right_topic:
            return {"decision": "not_same_event", "reason": "标题、摘要和主题差异较大，不属于同一事件。"}
        return {"decision": "unsure", "reason": "存在相似信号，但仍需要进一步判断是否为同一事件。"}

    def judge_event_similarity(self, left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, str]:
        heuristic = self._heuristic_event_similarity(left, right)
        if heuristic["decision"] != "unsure" or not self.enabled or self.client is None:
            return heuristic
        system_prompt = "你是 AI 情报去重助手。请判断两条动态是否在报道同一事件。输出 JSON，不要输出 Markdown。"
        user_prompt = f"""请判断以下两条动态是否为同一事件：\n\n动态A：\n- 标题：{left.get('title_cn') or left.get('title')}\n- 摘要：{clean_snippet(left.get('summary') or left.get('content', ''), 240)}\n- 来源：{left.get('source_detail') or left.get('source')}\n- 时间：{left.get('publish_date', '')}\n\n动态B：\n- 标题：{right.get('title_cn') or right.get('title')}\n- 摘要：{clean_snippet(right.get('summary') or right.get('content', ''), 240)}\n- 来源：{right.get('source_detail') or right.get('source')}\n- 时间：{right.get('publish_date', '')}\n\n返回 JSON：{{"decision": "same_event 或 not_same_event 或 unsure", "reason": "一句中文理由"}}"""
        try:
            response = self.client.chat.completions.create(model=self.model, messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], temperature=min(self.temperature, 0.1), max_tokens=250)
            result = json.loads(self._clean_json_string(response.choices[0].message.content or ""))
            decision = result.get("decision", "unsure")
            if decision not in self.EVENT_JUDGMENT_CHOICES:
                decision = "unsure"
            return {"decision": decision, "reason": self._clean_text(result.get("reason", "")) or heuristic["reason"]}
        except Exception as exc:
            print(f"Error judging event similarity: {exc}")
            return heuristic

    def _heuristic_report_summary(self, papers: List[Dict[str, Any]], updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        paper_topics = Counter(item.get("topic_cn") or item.get("display_topic") or "其他" for item in papers)
        update_categories = Counter(item.get("display_topic") or item.get("topic_cn") or "其他" for item in updates)
        hot_topics = [topic for topic, _ in (paper_topics + update_categories).most_common(5)] or ["AI"]
        paper_summary = "、".join(f"{topic}{count}篇" for topic, count in paper_topics.most_common(3)) or "本期论文样本较少。"
        update_summary = "、".join(f"{topic}{count}条" for topic, count in update_categories.most_common(4)) or "本期动态样本较少。"
        takeaways = [f"论文侧重点围绕{topic}展开。" for topic, _ in paper_topics.most_common(2)] or ["本期论文样本较少，建议继续观察后续更新。"]
        takeaways.extend([f"动态侧最活跃的方向是{topic}。" for topic, _ in update_categories.most_common(2)])
        watchlist = hot_topics[:3] or ["世界模型", "机器人", "AI基础设施"]
        return {"lead_summary": f"本期共整理 {len(papers)} 篇论文和 {len(updates)} 条全网AI动态，重点集中在{'、'.join(hot_topics[:3])}等方向，整体呈现出研究深化与产业落地并进的趋势。", "paper_summary": f"论文侧主要分布为：{paper_summary}。", "update_summary": f"动态侧重点包括：{update_summary}。", "hot_topics": hot_topics[:5], "key_takeaways": takeaways[:4], "watchlist": [f"继续跟踪{topic}方向的后续进展。" for topic in watchlist[:3]]}

    @retry(stop=stop_after_attempt(1), wait=wait_exponential(multiplier=1, min=2, max=5))
    def process_article(self, article: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        title = article.get("title", "")
        content = (article.get("content", "") or "")[:3000]
        if not self.enabled or self.client is None:
            return self._fallback_process(article)
        system_prompt = "你是 AI 情报编辑。请将给定内容整理为高质量中文情报卡片。输出必须是 JSON，不要输出 Markdown，不要解释。"
        user_prompt = f"""请分析以下内容并输出 JSON：\n\n原始标题: {title}\n来源: {article.get('source', '')}\n平台: {article.get('platform', '')}\n内容类型: {article.get('content_type', 'news')}\n已有主题: {article.get('topic', '')}\n正文: {content}\n\n返回字段必须为：{{"title_cn":"18-34个中文字符的信息型标题","summary_preview":"22-44个中文字符的一行副标题","summary":"120-180字中文摘要，3-4句","score":"0-10之间的一位小数","keywords":["3-5个中文短词"],"category":"Physical AI / Robotics / World Model / 模型/研究 / 产品发布 / 基础设施 / 开源生态 / 行业动态 / 企业合作 / 社交讨论 / 视频解读 / 应用落地 / Other","topic_cn":"中文主题名","why_it_matters":"40-70字中文","why_now":"30-60字中文，解释为什么这么做","expected_effect":"30-60字中文，解释会先改变什么","future_impact":"30-60字中文，解释会影响哪一层竞争、预算、标准或生态"}}\n\n要求：1. 不要复述标题，不要写空话。2. summary 和 why_it_matters 不能同义改写。3. why_now、expected_effect、future_impact 必须分工明确。4. 中文表达要像行业编辑。5. 只输出 JSON。"""
        try:
            response = self.client.chat.completions.create(model=self.model, messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], temperature=self.temperature, max_tokens=1000)
            result = json.loads(self._clean_json_string(response.choices[0].message.content or ""))
            if "summary" not in result or "score" not in result:
                return self._fallback_process(article)
            return self._normalize_single_result(article, result)
        except Exception as exc:
            print(f"Error calling LLM for {title}: {exc}")
            return self._fallback_process(article)

    def summarize_report(self, papers: List[Dict[str, Any]], updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        fallback = self._heuristic_report_summary(papers, updates)
        if not self.enabled or self.client is None:
            return fallback
        paper_lines = [f"- {item.get('title_cn')} | {item.get('display_topic')} | score={item.get('score')} | {clean_snippet(item.get('summary', ''), 80)}" for item in papers[:6]]
        update_lines = [f"- {item.get('title_cn')} | {item.get('display_topic')} | score={item.get('score')} | {clean_snippet(item.get('summary', ''), 80)}" for item in updates[:8]]
        system_prompt = "你是 AI 情报主编，请用中文写日报总览。输出 JSON，不要输出 Markdown。"
        user_prompt = f"""请根据以下日报内容生成总览 JSON：\n\n论文样本：\n{chr(10).join(paper_lines)}\n\n动态样本：\n{chr(10).join(update_lines)}\n\n返回字段：{{"lead_summary":"2-4句中文","paper_summary":"2-3句中文","update_summary":"2-3句中文","hot_topics":["3-5个中文热点标签"],"key_takeaways":["2-4条中文关键结论"],"watchlist":["2-3条中文后续观察点"]}}"""
        try:
            response = self.client.chat.completions.create(model=self.model, messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], temperature=min(self.temperature, 0.2), max_tokens=900)
            result = json.loads(self._clean_json_string(response.choices[0].message.content or ""))
            hot_topics = result.get("hot_topics") if isinstance(result.get("hot_topics"), list) else []
            key_takeaways = result.get("key_takeaways") if isinstance(result.get("key_takeaways"), list) else []
            watchlist = result.get("watchlist") if isinstance(result.get("watchlist"), list) else []
            return {"lead_summary": self._clean_text(result.get("lead_summary", "")) or fallback["lead_summary"], "paper_summary": self._clean_text(result.get("paper_summary", "")) or fallback["paper_summary"], "update_summary": self._clean_text(result.get("update_summary", "")) or fallback["update_summary"], "hot_topics": hot_topics[:5] or fallback["hot_topics"], "key_takeaways": [self._clean_text(item) for item in key_takeaways[:4] if self._clean_text(item)] or fallback["key_takeaways"], "watchlist": [self._clean_text(item) for item in watchlist[:3] if self._clean_text(item)] or fallback["watchlist"]}
        except Exception as exc:
            print(f"Error summarizing report: {exc}")
            return fallback
