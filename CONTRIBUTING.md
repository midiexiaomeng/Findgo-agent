# Contributing to Findgo

感谢你对这个项目的兴趣。

这是一个围绕「中文图像检索」的实验性项目，代码量不大，但每一层都想做扎实。如果你要改点什么，下面这些信息能省掉不少来回。

## 开发环境

### 前置条件

- Python 3.9 及以上（开发与测试用 3.11）
- 10 GB 以上磁盘（模型 1.6GB + 向量库 + 测试数据）
- 可选：CUDA 11.8 以上，没有 GPU 会自动回 CPU，只是编码慢一些

### 本地运行

```powershell
# 安装核心依赖
pip install -r requirements-core.txt

# 开发依赖（pytest、pre-commit）
pip install -r requirements-dev.txt

# 下载 CLIP + BLIP（约 1.6GB）
python scripts\download_models.py

# 生成本地配置
python scripts\init_agent_config.py

# 环境自检，确认依赖、CUDA、模型、向量库都正常
python cli.py doctor
```

`configs/agent.yaml` 由脚本生成，不进 git。里面是你的本机路径与 LLM 地址，改坏了直接删掉重新生成。

### 常用命令

| 命令 | 说明 |
| --- | --- |
| `pytest tests -q` | 跑全部测试，离线可跑 |
| `pytest tests/test_sandbox.py -v` | 跑单个模块 |
| `python cli.py doctor` | 环境自检 |
| `python cli.py index "D:\Photos"` | 建索引 |
| `python cli.py search "海边日落"` | 文本检索 |
| `pre-commit run --all-files` | 代码检查 |
| `start-app.bat` | 启动桌面应用 |

### 关于测试不跑模型

测试全部用 fake pipeline 代替真实模型，`pytest tests` 不会触发模型加载，也不会联网。如果你的改动需要真实模型验证，写到 `scripts/` 下做成独立脚本，不要塞进 `tests/`。

## 项目结构

```
src/findgo/
├── agent/            Agent 层：编排、规划、执行、记忆、动态工具
├── retriever/        内核层：CLIP、索引、检索、向量库、重排
├── tools/            16 个工具实现
├── ui/               界面组件，只渲染不写业务
├── utils/            沙箱、schema、日志
├── trash.py          暂存区
├── downloads.py      下载缓存
├── long_term.py      Qdrant 长期记忆
├── diagnostics.py    环境自检
└── settings_dialog.py 设置面板

desktop_app.py        启动入口，只做窗口组装
configs/              工具注册、Agent 配置模板
tests/                pytest
docs/                 架构说明与开发笔记
```

分层的原则是单向依赖：`ui` 可以调 `agent`，`agent` 可以调 `tools`，`tools` 可以调 `retriever`，反过来不行。如果发现要反向引用，通常说明职责放错了位置。

## 新增一个工具

这是最常见的扩展方式，五步：

**1. 写类**

在 `src/findgo/tools/` 下新建文件，继承 `BaseTool`：

```python
from pydantic import BaseModel, Field

from .base_tool import BaseTool


class ImageSizeStatsArgs(BaseModel):
    folder: str = Field(description="要统计的目录")


class ImageSizeStatsTool(BaseTool):
    name = "image_size_stats"
    description = "统计目录内图片的宽高分布"
    args_schema = ImageSizeStatsArgs
    permission = "read"
    timeout = 60

    def run(self, folder: str = "") -> dict:
        ...
        return {"total": 0, "buckets": {}}
```

`permission` 决定 Agent 能不能自主调用：

| 值 | 含义 |
| --- | --- |
| `read` | 只读，Agent 自主调用 |
| `index_write` | 改向量库，Agent 自主调用 |
| `file_write` | 在 `data/` 下写文件，Agent 自主调用 |
| `danger_delete` | 删原图，执行器直接拦截，必须界面确认 |

**2. 导出**

在 `tools/__init__.py` 里加进 `__all__`。

**3. 注册**

`agent/orchestrator.py` 的 `self.tools` 字典里加一行实例化。

**4. 登记**

`configs/tools.yaml` 补一条，界面上的工具列表从这里读。

**5. 补测试**

最少两个：正常路径、参数校验失败。放在 `tests/test_tools_*.py` 里，用 `tmp_path` 造临时目录，不要依赖真实数据集。

## 改内核层

`retriever/` 下的改动影响面大，改之前先确认这三件事：

- `Searcher.search_text` 的签名变了没有。测试里的 `FakeSearcher` 要同步改，否则会静默失败。
- 向量库抽象层的方法在 Qdrant 和 Chroma 两个实现里都加了吗。`VectorStore` 是抽象基类，只改一边会留下运行时的 `NotImplementedError`。
- `Pipeline._lock` 是全局锁。新加的耗时操作如果不该串行，别放在锁里面。

## 安全相关的改动

涉及 `utils/sandbox.py`、`agent/dynamic.py`、`trash.py` 的改动请特别小心，这几处是唯一的边界。

三条底线：

1. 任何路径操作先过 `sandbox.check(path, mode)`，不要直接 `open()`
2. 动态工具的命名空间里不要注入原生 `Path`，用 `_guarded_path_class()` 构造的受限版本
3. `purge` 类操作必须保留人工确认，不要为了顺手把它降级成 `file_write`

改完请跑 `pytest tests/test_sandbox.py tests/test_dynamic_escape.py tests/test_trash.py`。

## 代码风格

- 类型注解尽量补全，公开函数必须有返回值类型
- 注释写「为什么」，不写「做什么」。`# 取第一行作为标题` 这种没信息量
- 中文注释，标识符英文
- 不要引入新的重型依赖。加之前先想清楚能不能用标准库或已有依赖解决
- 单个文件超过 600 行就该考虑拆了

提交前跑一遍：

```powershell
pre-commit run --all-files
pytest tests -q
```

## Pull Request

欢迎提 PR。合并前会看这几点：

- 测试过没过，新功能有没有对应测试
- 有没有改到安全边界却没说明
- commit message 能不能看出改了什么

一个 PR 只做一件事。顺手格式化了十个文件的那种 PR 会被要求拆开。

## Issue 规则

### Bug

描述现象即可，不需要给修复方案。请提供：

- 环境：系统、Python 版本、有无 GPU
- 复现步骤：能稳定触发的操作顺序
- 期望行为与实际行为
- 日志：`logs/app.log` 里的相关行，或终端报错原文

**不要**贴大段推测性分析。先按步骤复现一次，确认现象是真的。

### Feature

写清楚三件事：想要什么、为什么需要、做到什么程度算完成。不需要给实现方案，那是维护者的事。

### 安全问题

沙箱绕过、路径逃逸、密钥泄露这类问题请不要开公开 Issue，直接发邮件或用 GitHub 的私密漏洞报告。

## 提交贡献

提交贡献即表示你同意你的代码以 MIT 协议授权。
