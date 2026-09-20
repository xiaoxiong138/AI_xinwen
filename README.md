# Web Agent

用于自动采集 AI 论文与行业动态、生成日报，并通过邮件定时发送的本地自动化项目。

## 当前能力

- 采集来源：
  - ArXiv
  - RSS 博客与新闻源
  - Web Search 聚合结果
  - Codex 定时研究任务生成的已核验研究收件箱
- 内容处理：
  - 新闻、技术、论文三类独立候选池和质量合同
  - 来源正文证据、发布日期、事实类型和跨期去重校验
  - 中文编辑稿不可变摘要、最终可见文本和邮件 UI 发送前闸门
- 报告输出：
  - 新闻／技术／论文优先分卷、超限栏目自动子分卷的 HTML 日报
  - Markdown 日报
  - 报告归档索引
- 自动化能力：
  - Windows 计划任务定时发送
  - 运行锁防叠跑
  - 超时与重试
  - 失败告警
  - 健康检查
  - 自愈入口
  - 安全快验收 `--validate-run`

## 目录说明

- [main.py](/D:/Web_Agent/main.py)
  主流程入口，负责采集、处理、出报表、发邮件。
- [scheduler_runner.py](/D:/Web_Agent/scheduler_runner.py)
  调度入口，负责定时运行、锁、状态文件、健康检查和自愈。
- [config.yaml](/D:/Web_Agent/config.yaml)
  采集源、报告、调度配置。
- [src](/D:/Web_Agent/src)
  采集器、数据库、处理器、通知器等实现。
- [templates](/D:/Web_Agent/templates)
  HTML 模板。
- [tests](/D:/Web_Agent/tests)
  回归测试。
- [archive](/D:/Web_Agent/archive)
  正式日报归档目录。
- [archive/validation](/D:/Web_Agent/archive/validation)
  快验收报告目录，不进入正式归档索引。
- [logs](/D:/Web_Agent/logs)
  调度日志、状态文件、健康检查快照。

## 环境准备

1. 安装 Python 3.11+
2. 安装依赖：

```powershell
pip install -r requirements.txt
```

3. 复制环境变量模板并填写：

```powershell
Copy-Item .env.example .env
```

至少需要配置：

- `EMAIL_SENDER`
- `EMAIL_PASSWORD`
- `EMAIL_SMTP_SERVER`
- `EMAIL_SMTP_PORT`
- `EMAIL_RECIPIENT`

当前 V11 主链路不调用 DeepSeek 或 OpenAI API；Codex 研究任务将结果写入
`data/codex_research/latest.json`，发送程序只读取并校验该本地收件箱。
午间与晚间研究任务分别在 07:45 和 15:30 启动，为 13:00 和 21:00
正式发送预留约五小时的检索、发现清单整理、证据补充与质量复检时间。
研究包本身的有效窗口为 6 小时，用于覆盖早间任务到午间发送的时间差；
新闻、技术和论文仍按各自 48 小时／7 天来源时效规则独立检查。

## 常用命令

手动执行正式日报：

```powershell
python D:\Web_Agent\main.py
```

通过调度器执行一次正式发送：

```powershell
python D:\Web_Agent\scheduler_runner.py
```

执行一次安全快验收：

说明：
会真实跑采集、处理、出报告，但不会发真实邮件。

```powershell
python D:\Web_Agent\scheduler_runner.py --validate-run
```

查看当前状态：

```powershell
python D:\Web_Agent\scheduler_runner.py --status
python D:\Web_Agent\scheduler_runner.py --status --json
```

执行健康检查：

```powershell
python D:\Web_Agent\scheduler_runner.py --doctor
python D:\Web_Agent\scheduler_runner.py --doctor --record
python D:\Web_Agent\scheduler_runner.py --doctor --json
```

健康检查自愈演练：

```powershell
python D:\Web_Agent\scheduler_runner.py --doctor --self-heal --dry-run
```

真正执行任务自愈：

```powershell
python D:\Web_Agent\scheduler_runner.py --doctor --self-heal
```

运行测试：

```powershell
python -m pytest -q
```

对生成后的邮件分卷执行桌面、移动端、暗色模式和禁图渲染审计：

```powershell
& D:\Web_Agent\tools\run_email_ui_audit.ps1 `
  -ReportPath @(
    'D:\Web_Agent\archive\report_YYYYMMDD_HHMM_part1.html',
    'D:\Web_Agent\archive\report_YYYYMMDD_HHMM_part2.html',
    'D:\Web_Agent\archive\report_YYYYMMDD_HHMM_part3.html'
  ) `
  -OutputDir 'D:\Web_Agent\artifacts\email_ui_audit\YYYYMMDD_HHMM'
```

审计会生成完整页面截图和 `metrics.json`，并在横向溢出、损坏图片、导读来源缺失、单条来源／事实标签缺失、正文过小、行距过紧、移动端留白不足或桌面内容过宽时返回非零退出码。正式任务会在 SMTP 发送前执行同一套审计。

生成一组包含 24 条新闻、24 条技术内容和 15 篇论文的 V11 三卷 UI 样稿，并执行同样的真实浏览器审计。样稿固定混入已核验的真实新闻、播客、技术和论文回归样本，其余合成条目用于补足 55+ 条长邮件的极端布局压力；样稿通过不能替代正式邮件的真实内容审计：

```powershell
python D:\Web_Agent\tools\generate_v11_ui_fixture.py
& D:\Web_Agent\tools\run_email_ui_audit.ps1 `
  -ReportPath @(
    'D:\Web_Agent\artifacts\v11_ui_fixture\v11_fixture_part1.html',
    'D:\Web_Agent\artifacts\v11_ui_fixture\v11_fixture_part2.html',
    'D:\Web_Agent\artifacts\v11_ui_fixture\v11_fixture_part3.html'
  ) `
  -OutputDir 'artifacts\v11_ui_fixture\audit'
```

为某个真实发送槽位生成 3 条新闻、3 条技术、3 篇论文的编辑抽查包：

```powershell
python D:\Web_Agent\tools\prepare_v11_editorial_review.py --slot-id YYYYMMDD_1300
```

审核文件的 `email_delivery` 会列出实际发送主题、分卷归档路径、邮箱到达匹配和自动 UI 审计结果。逐条打开原文，并在 `data/v11_editorial_reviews/<slot_id>.json` 中填写准确性、具体性、可读性和具体审核说明；再按 `email_delivery.subjects` 在 QQ 桌面端与移动端打开对应真实邮件，将 `client_rendering` 中的显示、截断、间距检查及说明填写完整后执行：

```powershell
python D:\Web_Agent\tools\prepare_v11_editorial_review.py --slot-id YYYYMMDD_1300 --check
```

编辑抽查未完成或任一维度失败的报告不会计入 V11 连续生产验收。

## V11 内容合同

- 产品模式与设计版本分别为 `intelligence_v11_editorial_library` 和 `v11-editorial-library`；V10 历史报告继续使用兼容阅读路径，但不得冒充 V11 生产样本。
- 正式研究收件箱配置 `required_schema_version: codex-research-v3`；缺失版本或旧版本会在解析和入库前以 `schema_version_mismatch` 阻断。未配置该字段的历史工具仍可读取旧包，但不能用于当前生产发送。
- V11 技术栏目单独要求 `technical_primary_source_ratio >= 0.80`；至少 16/20 条技术内容必须来自官方、研究团队、代码仓库、项目文档或其他一手材料。该指标未达标时真实发送会被质量闸门阻止，并在 doctor 中显示具体比例。
- 技术方向的 6/4/4/3/3 是采集目标，不再作为机械硬配额；最终硬覆盖底线为 3/2/2/1/1。达到 20+ 条和硬覆盖底线即可进入后续质量校验，未达到目标的方向记录在 `technical_category_target_underfilled`，不会用低质量材料强行补齐。
- 研究阶段必须保存至少 160 个去重 URL 的 `discovery_candidates` 清单，其中新闻、非论文技术、论文发现池至少为 50／50／40；最终提交 75-85 条完整候选，且每条成稿都必须能回溯到发现清单。提交量要求新闻不少于 30 条、非论文技术不少于 27 条、论文不少于 20 篇。
- 收件箱校验后，正式邮件必须保留至少 20 条新闻／博客／访谈、20 条非论文技术内容和 15 篇论文；同一事件只计入一个主栏目。
- V11 生产验收合同版本 12 要求研究包使用 `codex-research-v3`、发现清单、三类候选池、成稿追溯、归因开头多样性和跨条目模板重复检查全部通过，并要求三个主栏目与最近 7 天所有真实发送记录的 canonical URL 和事件标识重合数均为 0；`sent_history_overlap_count` 或 `cross_item_template_repeat_count` 非零时不得发送，也不得计入连续 3 天／6 封验收。
- `codex-research-v3` 为每条材料增加 `facts.key_numbers` 反向校验：原文存在有解释力的金额、样本量、任务数、性能、延迟、吞吐或 baseline 差异时，最多选择 3 个关键数字，并要求其同时存在于原文证据和最终公开正文。只在原文确无有效量化信息时允许空数组。
- V3 研究包还要求关键数字覆盖至少 8 条新闻、10 条非论文技术内容和 10 篇论文；`key_number_quota_status` 未通过时不会生成正式邮件。
- 每次收件箱校验都会记录 `inbox_sha256` 与 `discovery_manifest_sha256`；生产验收缺少任一指纹时失败，确保报告可追溯到当时实际使用的研究包和发现清单。
- `python scheduler_runner.py --doctor --json` 的 `codex_research_inbox.data.readiness_summary` 汇总提交量、通过量、通过率、主要拒绝原因、栏目缺口、历史重合和 `ready_for_dry_run`，用于在生成邮件前快速判断研究包是否可用。
- doctor 的 `codex_research_candidates` 会检查午间和晚间候选文件；仅对比正式包更新的候选执行深度校验，并区分 `blocked`、`ready_not_promoted` 与已经被正式包覆盖的 `superseded` 状态。
- 研究任务不得直接覆盖 `data/codex_research/latest.json`。午间和晚间分别写入 `candidate_1300.json`、`candidate_2100.json`，再运行 `python codex_research.py --candidate <候选文件> --promote-on-pass --json`；只有完整质量、历史去重和 V3 schema 校验全部通过时才会原子替换正式研究包，失败不会破坏上一份可用数据。提升前还会复核候选文件 SHA-256，防止校验结束后文件被并发改写。
- 新闻优先最近 48 小时，技术与论文优先最近 7 天。较旧但仍有学习价值的内容必须明确标记为补充阅读；补充新闻不得超过 7 天且每封最多 5 条，补充技术不得超过 30 天且每封最多 5 条，补充论文不得超过 30 天且每封最多 3 篇。
- 新闻正文为 180-300 字，访谈／播客为 300-500 字，技术正文为 220-380 字；证据不足的候选不扩写，也不能用于凑数量。
- 发布方归因必须保留，但同一“团队表示／公司称／发布方披露”等开头整包最多出现 8 次，最终排序中不得连续两条使用同一归因开头；超限会在研究包校验或最终 HTML 闸门阻止发送。
- 邮件优先按新闻、非论文技术、论文三类分卷承载完整中文整理稿；若任一栏目单卷仍超过客户端截断阈值，系统会按原始顺序继续拆成带分段编号的子卷，而不是回退为一封超长邮件或删减正文。因此实际发送可能为 3-5 卷，数量仍按整期统计。每个最终子卷都必须通过正文保真、关键数字保真以及桌面、手机、暗色和禁图四种渲染审计，且 HTML 体积不得超过 `email_html_max_bytes`。
- 真实回归集同时包含必须拒绝的坏样本和必须逐字保留的合格样本，覆盖新闻、访谈／播客、非论文技术和论文；模板或编辑器改动不得改变已审核正文。
- 生产验收要求连续 3 天、共 6 封真实邮件全部通过数量、质量、UI、投递和跨期新鲜度检查。
- 每封真实邮件还必须完成 9 条编辑抽查，逐条核验准确性、具体性和可读性，不能只依赖自动布尔指标。

## Windows 定时任务

安装正式发送任务：

```powershell
powershell -ExecutionPolicy Bypass -File D:\Web_Agent\setup_scheduled_tasks.ps1
```

安装健康检查任务：

```powershell
powershell -ExecutionPolicy Bypass -File D:\Web_Agent\setup_doctor_task.ps1
```

当前设计目标：

- `Web_Agent_Send_1300_v2`
- `Web_Agent_Send_2100_v2`
- `Web_Agent_Doctor_0900`

## 状态文件

- [last_run.json](/D:/Web_Agent/logs/last_run.json)
  最近一次正式发送状态
- [last_validation_run.json](/D:/Web_Agent/logs/last_validation_run.json)
  最近一次安全快验收状态
- [doctor_latest.json](/D:/Web_Agent/logs/doctor_latest.json)
  最近一次健康检查结果
- [doctor_history.json](/D:/Web_Agent/logs/doctor_history.json)
  健康检查历史与告警去重状态

## 当前约定

- 正式日报输出到 [archive](/D:/Web_Agent/archive)
- 快验收报告输出到 [archive/validation](/D:/Web_Agent/archive/validation)
- 正式归档索引会自动过滤验证报告
- 验证模式会抑制质量告警，避免样本缩小造成假告警
- 13:00 与 21:00 正式发送前必须存在同版次、足量且通过校验的 Codex 研究收件箱
- 任一内容数量、证据、最终 HTML 或 UI 审计硬门槛失败时不发送邮件

## 适合继续优化的方向

- 连续完成 3 天、6 封 V11 真实生产邮件验收
- 访谈、播客和视频摘编必须保留至少两个自然段；段落在采集、数据库或最终 HTML 中被压平时，内容保真闸门会阻止发送
- 新闻、访谈和非论文技术的第一段必须直接说明主体、对象与实际动作或机制；只有背景铺垫的开头不能进入正式栏目
- 访谈证据必须包含嘉宾的具体主张、理由、方法、结果或限制；节目时长、发布日期和简介不能代替观点证据
- `official_claim`、`interview_opinion` 与 `analysis` 不只显示不同标签，正文也必须使用发布方自报、受访者观点或分析推测的对应归因语言
- 非论文技术条目必须同时保存方法、对照方案、验证依据和局限；缺少任一部分只能补证据或替换候选，不能扩写模板话进入技术栏目
- 论文同样必须具备方法、baseline、验证依据和局限；只提供标题翻译、方法名或单个结果数字的论文不能计入 15 篇正式配额
- 正文原稿和论文两段文字必须是完整句子，不得含省略号或半句结尾；截断内容在采集入口直接淘汰，不能等邮件模板修补
- 根据各栏拒收原因继续调节 30／27／20 候选余量，而不是降低质量门槛
- 建立 QQ 邮箱和常用移动客户端的人工渲染抽查记录
- 在生产样本稳定后再微调信息密度、段落长度和栏目配额
