# Codex AI 日报研究提示词

请使用网页搜索完成一份“AI 前沿编辑日报”的资料研究，并严格返回一个 JSON 对象。不要输出 Markdown 代码围栏、引用列表或额外说明。

本轮研究包 schema 固定为 `codex-research-v3`。旧 schema 不包含关键数字反向校验，不能继续生成。

## 数量与时间窗口

- 输出 75-85 条互不重复、已经写完正文并核验过证据的候选成稿，每个 URL 只能归入一个 `primary_section`。这是质量淘汰前的提交量，不是要求邮件展示全部候选。
- `news` 至少提交 30 条，质量校验后必须保留不少于 20 条：新闻、官方博客、完整访谈、播客、产品发布或产业动态。
- `news` 中至少提交 4 条可取得文字稿、字幕或精确时间戳的访谈/播客/视频，以及至少 8 条官方或研究团队博客；质量校验后仍须分别保留至少 2 条和 5 条，不能用节目简介冒充完整访谈。
- `technical` 至少提交 27 条，质量校验后必须保留不少于 20 条：非论文的最新技术方法、训练路线、系统架构、功能机制、开源实现、推理与部署实践。
- `paper` 至少提交 20 篇，质量校验后必须保留不少于 15 篇：最近论文或论文的重要版本更新。
- 新闻优先最近 48 小时，技术内容和论文优先最近 7 天。新鲜度不是按候选生成瞬间计算，而是按 `generated_at + 6 小时` 的发送安全时刻计算；午间包必须在当天 13:30 后仍达标，晚间包必须在当天 21:30 后仍达标。接近 48 小时或 7 天边界、会在正式发送前过期的条目不得占用新鲜来源名额。超过时间窗口但确有学习价值时，在 `quality_flags` 加入 `supplemental_older_source`；补充新闻不得早于 7 天，补充技术和论文不得早于 30 天，超过上限必须替换来源。最终邮件中补充新闻最多 5 条、补充技术最多 5 条、补充论文最多 3 篇，候选池应提供足够新鲜内容让选稿器满足这一上限。
- `publish_date` 必须来自来源页面，并且不得晚于当前时间；考虑时区显示差异，校验器最多容忍未来 6 小时，超过即拒收，不能用未来日期绕过新鲜度检查。
- 候选成稿中至少保证 24 条新闻在发送安全时刻仍处于最近 48 小时内、22 条技术内容和 17 篇论文在发送安全时刻仍处于最近 7 天内；质量校验后必须分别保留至少 15、15、12 条，补充旧来源不能占用这些新鲜来源名额。不得集中使用将在未来 6 小时内跨过时效边界的材料来凑配额。
- 写入前必须与生产数据库最近 7 天所有 `delivery_status=sent` 的报告做 canonical URL 和事件级去重；最终 `python codex_research.py --json` 必须显示 `production_ready_status=passed`、`sent_history_overlap_count=0`，不得把同版修订作为重复例外。
- 重点覆盖：World Model、Physical AI / Robotics、Agent / Models、Infra / Open Source、Products / Business。
- 最终保留的论文必须控制领域分布：World Model 3-5 篇，Physical AI / Robotics 4-6 篇，Agent / Models 3-5 篇，Infra / Open Source 2-4 篇，其他有明确学习价值的方向 1-3 篇；同一领域不得超过 6 篇。提交的 20+ 篇候选也要按这一比例保留冗余，不能把余量全部堆在一个领域。找不到合格论文时继续检索，不得用相邻领域凑数或虚改分类。

技术内容以下列分布作为采集目标，当天可按实际高质量来源调整：World Model / Physical AI / Robotics 6 条；Agent、记忆、工具调用、工作流 4 条；训练、后训练、数据工程 4 条；推理、部署、硬件、性能优化 3 条；多模态、模型架构、开源实现 3 条。最终 20+ 条的硬覆盖底线分别为 3、2、2、1、1 条；达到底线但未达目标时可通过，但必须在诊断中列出 `technical_category_target_underfilled`，不得用低质量材料强行补齐目标。

## 来源要求

- 优先官方博客、研究机构、论文页、arXiv/OpenReview、GitHub、项目文档、完整访谈、播客节目页或文字稿。
- 最终保留的 20 条技术内容中，至少 16 条必须直接来自官方技术博客、研究团队、GitHub 仓库、项目文档、模型卡、系统卡、版本说明或会议原始演讲；媒体和个人转述只能用于发现，找到原始来源后必须替换为原始链接。若没有原始材料，该条不得占用技术栏目 20 条硬配额。
- 每次先建立不少于 160 个去重 canonical URL 的发现候选池，再筛成 75-85 条可交付候选成稿；发现池至少包含 `news` 50 条、`technical` 50 条、`paper` 40 篇，所有最终成稿 URL 必须存在于发现清单中。不要围绕上一封的来源做局部替换。新闻池应覆盖 OpenAI、Anthropic、Google/DeepMind、Meta、Microsoft、AWS、NVIDIA、Hugging Face、xAI、机器人厂商、AI 初创公司、监管机构和可靠产业媒体；技术池应额外检索官方工程博客、开发者文档、GitHub Releases、模型卡、系统卡、推理框架和训练基础设施团队。
- 访谈池优先带完整 transcript、字幕或章节时间码的节目和会议演讲；无法定位具体原话的音视频只能作为发现线索，不能进入最终 20 条。
- 访谈、播客和视频的 `source_excerpt` 或 `facts.evidence` 必须记录嘉宾的具体主张、理由、技术细节、实验、限制或争议；节目时长、上线日期、嘉宾名单和页面简介不能作为观点证据。
- 论文池分别独立检索 World Model、Physical AI/Robotics、Agent/Models、Infra/Open Source 和其他方向，先满足各自配额，再合并去重。
- 聚合站只用于发现，最终 `url` 必须是实际打开并核验过的 canonical URL。
- 不收录搜索结果页、首页、Google News 跳转、SEO 汇总、无出处传闻或只有标题没有正文的信息。
- 每条都要从原文正文提取 `source_excerpt` 和 `evidence_locator`，不要凭搜索摘要补写。

## 中文编辑标准

- 标题准确、自然、具体，不机械翻译，不中英拼接。
- 新闻正文 180-300 个中文字符，写清谁做了什么、具体证据、背景和影响；访谈或播客 300-500 字，分成 2-3 个自然段，区分嘉宾观点与已发生事实。不要把 300-500 字全部挤在一个段落里。
- 技术正文 220-380 个中文字符，必须解释机制、输入输出、架构或工程约束，并至少包含一个可核验的数字、实验、代码、配置或部署事实。
- 每条 `technical` 额外输出 `technical_plain_summary`：80-160 个中文字符、2-3 个完整句子，放在技术正文之前帮助读者建立直觉。必须用一个与真实机制逐项对应的生动类比，例如把路由器比作“分诊台”时，要继续说明谁是输入、如何分流、结果送往哪里；不能只写“像大脑”“像魔法”或口号，也不能凭类比新增原文没有的能力、数字和结论。20+ 条技术内容不要统一用“可以把它理解为”开头，应交替使用直接场景、反差、流程类比等自然写法。
- 每条 `technical` 的 `facts` 必须填写 `method`、`baseline` 和 `limitation`，并至少填写 `dataset_or_benchmark`、`metric_result`、`code_or_project`、`deployment_context` 中的一项。`baseline` 要具体说明替代或对比的旧流程/旧方案，`limitation` 要说明当前验证边界，不能填“暂无”。
- 新闻、访谈和技术正文的第一段必须直接出现具体主体或产品/方法，并说明实际动作、主张或技术机制；不要先写行业背景、阅读建议、“本文介绍”或空泛趋势判断，再把实质内容拖到后文。
- 论文额外输出两段已编辑中文：`paper_plain_summary` 用 90-180 字通俗说明论文究竟做了什么；`paper_technical_intro` 用 160-300 字解释关键方法、实验结果、baseline 和局限。两段不能重复，也不能只翻译标题。
- 每条 `paper` 的 `facts` 必须填写 `method`、`baseline` 和 `limitation`，并至少填写 `dataset_or_benchmark`、`metric_result`、`code_or_project`、`deployment_context` 中的一项；没有对照或局限信息时继续读取论文正文，仍找不到就替换候选。
- 每条正文必须可以独立阅读，不依赖“见原文”“需要进一步确认”来填补核心信息。
- 新闻和技术正文至少包含 3 个信息完整的自然句，访谈或播客至少包含 4 个；不得复制同一句或用单个超长句凑字数。最终正文必须以中文转述为主，英文只保留必要的模型名、术语和原文标识。
- `summary`、`paper_plain_summary` 和 `paper_technical_intro` 必须以完整句号、问号或感叹号结束，不得包含 `...`、`…`，不得以逗号、冒号、连接词或尚未说完的从句结尾。不要依赖后续模板替你补句号。
- 技术与论文正文出现百分比、延迟、吞吐、成功率或倍数时，必须同时写明对应的数据集、硬件/部署环境或 baseline；不能只保留孤立数字。
- 正文中的百分比、百分点、延迟、吞吐、倍数、金额、任务数、样本数、轮次和训练/推理时长等所有可测量数字，必须逐项同时出现在 `source_excerpt`、`facts.evidence` 或 `facts.metric_result` 中，且两处数值和单位一致；不能用摘要自身或标题作为数字证据。若数字只出现在方法、架构或背景字段，也必须再复制到 evidence 或 metric_result，否则校验器会以 `unsupported_summary_numeric` 拒收。
- 还要做反向检查：原文出现基金规模、客户/公司/任务/样本数量、模态数、成功率、提升幅度、延迟、吞吐、金额、训练成本或与 baseline 的量化差异时，从中选出最多 3 个最有解释力的数字写入 `facts.key_numbers`。统一使用阿拉伯数字和明确单位，例如“40 家公司”“11 种模态”“5000 万美元”。这些数字必须同时保留在原文证据字段和最终公开正文中；不能只把数字留在原文摘录里，也不能把版本号、日期、论文编号当作关键数字。原文确实没有有意义的量化证据时才允许填写空数组。
- 质量过滤后的研究包中，`facts.key_numbers` 非空的条目至少达到：新闻 8 条、非论文技术 10 条、论文 10 篇。该要求用于防止所有条目机械填写空数组；如果来源缺乏量化证据，应更换为证据更完整的来源，不能拿日期、版本号或论文编号凑数。
- 禁止使用“值得关注、未来可能带来影响、出现了新的动作、相关机构、学习重点是、技术上它主要围绕、需要观察、是否真正”等空泛模板句。
- 事实归因必须保留，但不要让每条正文都以“团队表示／公司称／发布方披露／官方公告显示”开头。同一归因开头在整包中最多使用 8 次，连续两条不得采用同一开头。可以把归因自然放进具体动作或数字所在句，但不能因此省略观点、厂商自报结果与已验证事实的区别。
- 不得用同一段落骨架批量替换机构名、产品名、方法名或数字来生成多条正文。校验器会在遮蔽实体与量化值后比较同栏目摘要，任意两条相似度达到 0.92 都会阻止候选包提升；必须依据各自原文重新组织论述顺序、机制和证据。

## JSON 格式

```json
{
  "schema_version": "codex-research-v3",
  "generated_at": "带时区的 ISO-8601 时间",
  "discovery_candidates": [
    {
      "primary_section": "news | technical | paper",
      "url": "实际打开或准备核验的 canonical URL",
      "source_detail": "来源名称",
      "publish_date": "可确认时填写 ISO-8601 日期或时间，否则留空"
    }
  ],
  "items": [
    {
      "primary_section": "news | technical | paper",
      "title": "原文标题",
      "title_cn": "准确具体的中文标题",
      "url": "原文 canonical URL",
      "source_detail": "发布方、实验室、期刊、节目或媒体",
      "platform": "Blog | News | Paper | Podcast | Interview | Video | GitHub | Website",
      "content_type": "news | paper | interview | podcast | video | project",
      "claim_type": "verified_fact | official_claim | interview_opinion | analysis | research_result",
      "technical_category": "embodied_world_model | agent_systems | training_data | inference_deployment | multimodal_architecture；仅 technical 必填，其他留空",
      "publish_date": "ISO-8601 日期或时间",
      "author": "作者、嘉宾、公司或研究团队",
      "topic": "World Model | Physical AI / Robotics | Agent / Models | Infra / Open Source | Products / Business",
      "category": "World Model | Physical AI / Robotics | 模型/研究 | 产品发布 | 基础设施 | 开源生态 | 行业动态 | 企业合作 | 访谈观点 | 播客解读 | 应用落地",
      "summary_preview": "一句具体副标题",
      "summary": "符合对应类型长度和文体要求、可直接用于邮件的完整中文正文",
      "technical_plain_summary": "仅 technical 填写：80-160 字、2-3 句、以机制映射类比讲清技术直觉，其他类型留空",
      "paper_plain_summary": "仅 paper 填写：90-180 字通俗首段，其他类型留空",
      "paper_technical_intro": "仅 paper 填写：160-300 字方法、实验、baseline 与局限，其他类型留空",
      "source_excerpt": "原文中支撑核心判断的短摘录或忠实中文转述",
      "evidence_locator": "章节、小标题、时间戳、表格、图号或段落位置",
      "facts": {
        "who": "具体主体",
        "action": "具体动作",
        "target": "产品、模型、论文方法、客户或场景",
        "evidence": ["1-4 条具体证据、数字、实验结果或原文主张"],
        "audience": "影响对象",
        "method": "方法机制，没有则为空",
        "architecture": "架构或模块关系，没有则为空",
        "training_objective": "训练目标或损失，没有则为空",
        "input_output": "输入输出，没有则为空",
        "dataset_or_benchmark": "数据集、基准或应用环境，没有则为空",
        "metric_result": "指标或结果，没有则为空",
        "key_numbers": ["原文最有解释力、且最终正文必须保留的 0-3 个量化事实"],
        "baseline": "对照方法或竞品，没有则为空",
        "limitation": "局限或尚未验证部分，没有则为空",
        "code_or_project": "代码或项目地址，没有则为空",
        "deployment_context": "部署场景、硬件或工程约束，没有则为空"
      },
      "keywords": ["3-6 个关键词"],
      "score": 0.0,
      "evidence_quality": 0.0,
      "information_density": 0.0,
      "why_it_matters": "基于事实说明它改变了什么认知或流程",
      "why_now": "为什么此时发生",
      "expected_effect": "短期会改变什么",
      "future_impact": "只在证据充分时填写，否则留空",
      "quality_flags": []
    }
  ]
}
```

`score` 使用 0-10，`evidence_quality` 和 `information_density` 使用 0-1。先完成搜索、打开原文、核验来源、去重和配额检查，再一次性输出合法 JSON。`discovery_candidates` 必须保留完整发现清单，不能只复制最终 `items`；校验器会检查至少 160 个去重 URL、三类发现池下限以及每条最终成稿是否来自发现清单。

不得直接写入或覆盖 `data/codex_research/latest.json`。`generated_at` 必须在全部正文、证据和发现清单完成后、准备执行最终校验时写入，不能使用任务启动时间。先把完整 JSON 写入独立候选文件，例如午间使用 `data/codex_research/candidate_1300.json`、晚间使用 `data/codex_research/candidate_2100.json`，然后运行：

```powershell
python codex_research.py --candidate data/codex_research/candidate_1300.json --promote-on-pass --json
```

只有输出同时满足 `status=passed`、`promotion.status=promoted`、`ready_for_dry_run=true`、`production_ready_status=passed` 和 `sent_history_overlap_count=0`，本轮研究才算完成；该命令会在全部检查通过后执行原子替换。校验失败时保留正式 `latest.json` 不动，根据 `discovery_candidate_count`、`discovery_section_counts`、`submitted_not_in_discovery_count`、`submitted_section_counts`、`rejected_section_counts`、`rejection_reason_counts_by_section`、`accepted_section_rates`、`unsupported_summary_numeric_examples`、`key_number_contract_missing_examples`、`key_number_evidence_missing_examples` 和 `key_number_public_copy_missing_examples` 修复候选文件后重新执行提升命令。某栏目被拒后低于最终配额时，按该栏具体淘汰原因补充新的合格来源，不得降低校验标准、直接改正式文件或复用历史条目。

`claim_type` 必须与文体一致：论文用 `research_result`；访谈和播客观点用 `interview_opinion`；公司尚未被第三方验证的性能或商业结果用 `official_claim`；已有原始记录或多源验证的事件用 `verified_fact`；媒体或作者推演用 `analysis`。正文措辞必须体现这种证据差异：每条 `official_claim` 的正文必须明确出现“公司称／团队介绍／发布方披露／官方公告显示”等归因表达，尤其要放在性能、收益、客户数量和商业结果之前；只写“某公司发布了某产品”或直接陈述指标不算完成归因。`interview_opinion` 必须明确写谁“认为、解释、主张或指出”；`analysis` 必须明确写成分析、判断或推测，不能伪装成已经发生的事实。校验出现 `claim_language_mismatch_examples` 时，逐条改写列出的正文开头或替换来源，不能只修改 `claim_type` 绕过检查。
