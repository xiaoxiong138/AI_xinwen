# Web Agent

用于自动采集 AI 论文与行业动态、生成日报，并通过邮件定时发送的本地自动化项目。

## 当前能力

- 采集来源：
  - ArXiv
  - RSS 博客与新闻源
  - Web Search 聚合结果
- 内容处理：
  - LLM 摘要、打分、分类
  - 回退规则处理
- 报告输出：
  - HTML 日报
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

- `DEEPSEEK_API_KEY`
- `EMAIL_SENDER`
- `EMAIL_PASSWORD`
- `EMAIL_SMTP_SERVER`
- `EMAIL_SMTP_PORT`
- `EMAIL_RECIPIENT`

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

- `Web_Agent_Send_1200`
- `Web_Agent_Send_2100`
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

## 适合继续优化的方向

- 给中午发送任务增加一次独立的自动验收或失败后自愈触发
- 进一步清理历史中文乱码文案
- 增加 README 中的架构图或流程图
- 接入 GitHub 远端后补首个发布版本标签
