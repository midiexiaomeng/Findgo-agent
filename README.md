# imgsearch · 本地多模态 RAG 图像检索

用一句中文描述找回本地照片，或者丢一张图去找相似图片。模型全部跑在本机，照片不出硬盘。

三级检索架构：**Chinese-CLIP 跨模态嵌入 → ChromaDB/Qdrant 向量召回 → Qwen2.5 结合 BLIP 描述语义精排**。

---

## 能力

| 能力 | 说明 |
|---|---|
| 以文搜图 | 输入「夕阳下的海边」「一只橘猫趴在窗台上」这类描述 |
| 以图搜图 | 上传一张参考图，先由 BLIP 生成描述，再按描述做语义精排 |
| 双向量库 | ChromaDB 与 Qdrant 共用一套 `VectorStore` 接口，改一行配置切换 |
| 完全本地 | 嵌入、描述、重排全部离线；Qdrant 支持嵌入式模式，无需起服务 |
| 结果直显 | 界面以图片墙展示，同时给出完整路径、三路分数和图像描述 |
| 增量索引 | 按绝对路径生成稳定 ID，重复索引自动跳过 |

---

## 三级架构

```
                        ┌──────────────────────────────┐
   文本查询 ──────────▶ │  Chinese-CLIP Text Encoder   │──┐
                        └──────────────────────────────┘  │   512 维
                                                         │   同一语义空间
                        ┌──────────────────────────────┐  │
   图片查询 ──────────▶ │  Chinese-CLIP Image Encoder  │──┘
                        └──────────────────────────────┘
                                     │
                                     ▼
                        ┌──────────────────────────────┐
                        │  ANN 召回  Top-K = 50       │  高召回，宁多勿漏
                        │  ChromaDB / Qdrant          │
                        └──────────────────────────────┘
                                     │
                                     ▼  截断到 16 个候选
                        ┌──────────────────────────────┐
                        │  BLIP 描述 + Qwen2.5-7B     │  语义精排
                        │  分批打分 → 加权融合         │
                        └──────────────────────────────┘
                                     │
                                     ▼
                        图片墙 + 路径 + 三路分数
```

**第一级解决表示问题。** 图像和文本被映射到同一个 512 维空间，所以「以图搜图」和「以文搜图」可以共用同一条检索链路，只是查询向量来自不同编码器。

**第二级解决速度问题。** 50 个候选里通常包含正确答案，但排序不一定对。这一级追求召回率，不追求精度。

**第三级解决语义问题。** CLIP 判断的是整体视觉相似度，对「室内有个人在工作」这种带关系、带否定的查询容易跑偏。BLIP 把候选图转成文字，Qwen 再读文字做判断，补上这一层语义。

---

## 两种部署方式

发布包解压后，双击对应脚本即可。两套环境完全独立，各自创建虚拟环境，互不污染，
也不依赖系统里已经装了什么包。

### 方式一：本地模型（deploy-local.bat）

双击 `deploy-local.bat`，自动完成：

1. 创建 `.venv-local`，只装运行必需的依赖，不含测试工具
2. 按显卡安装 CUDA 或 CPU 版 torch
3. 从 hf-mirror 下载 CLIP 和 BLIP
4. 检测 Ollama：优先用项目内便携版（`ollama\ollama.exe`），没有则尝试 winget 安装，然后拉取 `qwen2.5:7b-instruct`
5. 生成 `config.local.yaml`
6. 可选：直接建索引

完成后双击 `start-local.bat` 打开窗口。

命令行等价写法：

```powershell
.\scripts\deploy-local.ps1
.\scripts\deploy-local.ps1 -Cpu
.\scripts\deploy-local.ps1 -Model qwen2.5:3b-instruct
.\scripts\deploy-local.ps1 -Index D:\Photos
```

### 便携版 Ollama（不写 C 盘）

不想让 Ollama 装到 `C:\Users\<name>\AppData` 的话，双击 `install-ollama.bat`。
程序和模型都放在项目目录：

```
ollama/
├── ollama.exe
├── ollama_llama_server.exe
├── models/            Qwen 权重在这里
└── start-ollama.bat
```

首次会下载约 1.1GB 的官方 zip，解压后拉取 `qwen2.5:7b-instruct`。GitHub 慢的话
手动下载 `ollama-windows-amd64.zip`，然后：

```powershell
.\scripts\install-ollama-portable.ps1 -Zip D:\downloads\ollama-windows-amd64.zip
```

`start-local.bat` 会自动检测 11434 端口，没监听就拉起 `ollama\ollama.exe serve`，
不需要单独开窗口。

### 方式二：云端 API（deploy-api.bat）

双击 `deploy-api.bat`，同样创建独立的 `.venv-api`，下载 CLIP 和 BLIP，但不装
Ollama。LLM 重排改为调用云端 API，运行时会提示输入 Key。CLIP 和 BLIP 仍然
在本机运行，图片不会上传。

支持预设服务商：

| 参数 | 服务商 | 默认模型 |
|---|---|---|
| `-Provider dashscope` | 阿里百炼 | qwen-plus |
| `-Provider deepseek` | DeepSeek | deepseek-chat |
| `-Provider moonshot` | 月之暗面 | moonshot-v1-8k |
| `-Provider openai` | OpenAI | gpt-4o-mini |
| `-Provider custom` | 任意兼容端点 | 手动输入 |

```powershell
.\scripts\deploy-api.ps1 -Provider deepseek -ApiKey sk-xxx
.\scripts\deploy-api.ps1 -Provider custom -BaseUrl http://localhost:8000/v1 -ApiKey x -Model qwen
```

完成后双击 `start-api.bat`。

`config.api.yaml` 里保存了 API Key，不要提交到仓库或发给别人。

### 两套的差别

| | 本地模型 | API |
|---|---|---|
| 虚拟环境 | `.venv-local` | `.venv-api` |
| 配置文件 | `config.local.yaml` | `config.api.yaml` |
| 启动器 | `start-local.bat` | `start-api.bat` |
| LLM 运行位置 | 本机 Ollama | 云端 |
| 离线可用 | 是 | 精排需要网络 |
| 显存需求 | 7B 约 5GB | 仅 CLIP/BLIP 约 1.5GB |
| 首次部署体积 | 约 6.5GB | 约 1.8GB |

两套共用同一份 `data/` 和 `models/`，切换只影响 LLM 那一层，索引不需要重建。

### 手动部署

不想用脚本的话：

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-core.txt
python scripts\download_models.py
python scripts\make_config.py --mode local --out config.local.yaml
python desktop.py
```

## 建立索引

```bash
python cli.py index D:\Photos
```

首次会加载 CLIP 和 BLIP，之后都是增量索引。界面上也能建，见下一步。

## 没有图片可测？

内置脚本可以从 COCO val2017 里筛出一套真实照片库：

```bash
python scripts/prepare_test_dataset.py --download
python cli.py index datasets/coco_2000/images --no-caption
```

默认下载 5000 张验证集（778MB），筛掉最短边小于 480px 的，随机抽 2000 张，
输出到 `datasets/coco_2000/`。同时生成 `manifest.json` 和一份中文测试查询
`sample_queries.txt`。加 `--no-captions` 可以省掉 241MB 的标注文件。

官方地址下载慢时，可以从 OpenDataLab、Kaggle、阿里天池搜 `COCO 2017`，下载
`val2017.zip` 后用 `--zip` 指向本地文件。

如果只是想要好看的演示图，[Unsplash Lite](https://unsplash.com/data/lite/latest)
有 25000 张高分辨率摄影图（约 20GB），取其中 2000 张放进 `datasets/` 即可。

## 启动

浏览器模式：

```bash
python app.py
```

打开 `http://127.0.0.1:7860`。

### 桌面窗口

本地模型版双击 `start-local.bat`，API 版双击 `start-api.bat`。两者都会以无控制台
方式打开独立窗口，并各自加载对应的配置文件。

开发调试时也可以直接用开发环境启动：

```bash
python desktop.py --mode edge
```

Edge 的 `--app` 模式会打开一个没有地址栏、没有标签页的窗口，可以固定到任务栏。
不需要安装任何额外依赖。

想要更原生的窗口（自定义标题、任务栏不显示 Edge），装一个 pywebview：

```bash
pip install pywebview
python desktop.py --mode webview
```

检查当前机器支持哪种模式：

```bash
python desktop.py --check
```

三种模式共用同一套界面和同一个后端，数据完全一致。如果 7860 端口上已经有
服务在跑，启动器会直接复用，不会再起一份。

---

## 目录结构

```
multimodal-rag-search/
├── app.py                      启动入口
├── cli.py                      命令行：index / search / image / stats / doctor / serve
├── config.yaml                 全部可调参数
├── requirements.txt
├── imgsearch/
│   ├── config.py               配置（YAML ↔ dataclass，环境变量覆盖）
│   ├── runtime.py              设备、精度、路径、日志
│   ├── clip_encoder.py         第一级：Chinese-CLIP
│   ├── captioner.py            BLIP 描述 + SQLite 缓存
│   ├── vectorstore/
│   │   ├── base.py             VectorStore 抽象基类 + VectorHit
│   │   ├── chroma_store.py     ChromaDB 实现
│   │   ├── qdrant_store.py     Qdrant 实现（local / server）
│   │   └── factory.py          配置 → 实例
│   ├── llm/
│   │   ├── client.py           Ollama / OpenAI 兼容客户端
│   │   └── reranker.py         第三级：分批打分 + 融合
│   ├── indexer.py              扫描、校验、编码、写库
│   ├── searcher.py             两条检索链路
│   └── pipeline.py             组装与后端切换
├── webui/app.py                Gradio 界面
├── scripts/
│   ├── setup.ps1 / setup.sh    环境安装
│   ├── download_models.py      模型下载
│   └── build_package.ps1       打发布包
├── tests/                      26 个测试，双后端契约测试
├── models/                     模型权重（不入包）
└── data/                       向量库与缓存（不入包）
```

---

## 关键配置

改 `config.yaml` 后重启生效。

| 配置 | 默认 | 说明 |
|---|---|---|
| `device` | `auto` | `auto` 自动选 CUDA / MPS / CPU |
| `embedding.model` | `chinese-clip-vit-base-patch16` | 短名优先查 `models/`，找不到回退 HuggingFace |
| `embedding.embedding_dim` | `512` | 只做校验，实际以模型 config 为准 |
| `vector_store.backend` | `chromadb` | `chromadb` 或 `qdrant` |
| `caption.enabled` | `true` | 关闭后 LLM 重排会退化，因为没有描述可读 |
| `rerank.backend` | `ollama` | `ollama` / `openai` / `none` |
| `rerank.max_candidates` | `16` | 进入 LLM 的候选数，不是召回数 |
| `search.recall_k` | `50` | ANN 召回数 |
| `search.top_k` | `12` | 最终展示数 |
| `search.fusion.vector_weight` | `0.35` | 融合时向量分权重 |
| `search.fusion.llm_weight` | `0.65` | 融合时 LLM 分权重 |

---

## 切换向量库

改配置：

```yaml
vector_store:
  backend: qdrant
```

或者命令行临时指定：

```bash
python cli.py index D:\Photos --backend qdrant
python cli.py search "海边日落" --backend qdrant
```

界面「设置与诊断」页也能切。

两个后端的数据是**各自独立**的，切换后要重新建索引。代码层面两者走同一套接口：

```python
from imgsearch.config import Config
from imgsearch.vectorstore import create_vector_store

cfg = Config.from_yaml()
store = create_vector_store(cfg, backend="qdrant")   # 或 "chromadb"

store.add(ids, vectors, metadatas)
hits = store.query(query_vector, top_k=50)
store.delete(ids)
store.reset()
```

| | ChromaDB | Qdrant |
|---|---|---|
| 部署 | 进程内，零服务 | `local` 嵌入式，或 `server` 连服务 |
| 多进程 | 可以 | 本地模式有文件锁，同一目录只能一个进程 |
| 适合 | 单机个人使用，默认选它 | 想接 Qdrant 服务、或多端共享 |
| 删除 collection | 干净 | 本地模式有残留数据坑，本项目的 `reset()` 已绕开 |

Qdrant 连服务模式：

```yaml
vector_store:
  backend: qdrant
  qdrant:
    mode: server
    url: http://localhost:6333
    api_key: ""
```

---

## 接别的 LLM

只要兼容 OpenAI 协议，vLLM、LM Studio、llama.cpp server 都能用：

```yaml
rerank:
  backend: openai
  model: Qwen2.5-7B-Instruct
  openai:
    base_url: http://localhost:8000/v1
    api_key: not-needed
```

vLLM 启动示例：

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000 --max-model-len 4096
```

llama.cpp 用 GGUF 也可以，起 server 后同样填 `base_url`。

LLM 不可用时不会报错，检索自动退化为纯向量排序，界面会标出「纯向量排序」。这是有意设计的降级路径。

---

## 打包发布

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_package.ps1
```

生成 `dist/imgsearch-2.0.0.zip`。打包时会排除 `models/`、`data/`、`datasets/`、
`.venv-local/`、`.venv-api/`、`legacy/`、生成的 `config.local.yaml` 和
`config.api.yaml`，以及各类缓存。

发布包在目标机器上的使用流程：

```
1. 解压
2. 双击 deploy-local.bat 或 deploy-api.bat
3. 双击 start-local.bat 或 start-api.bat
```

模型不在包里是刻意设计：CLIP + BLIP 约 1.4GB，7B 量化模型约 4.7GB，塞进压缩包既慢又容易损坏。下载脚本幂等、可续传，比带一个大包更可靠。

---

## 性能量级（RTX 4060 Laptop 8G 参考）

| 阶段 | 量级 |
|---|---|
| CLIP 单张编码 | 10~30ms |
| BLIP 单张描述 | 80~150ms |
| 索引 1000 张（含描述） | 3~6 分钟 |
| 向量召回 50 条 | 5~15ms |
| Qwen2.5-7B 重排 16 条（2 批） | 6~12 秒 |

显存占用：CLIP fp16 约 0.3GB，BLIP 约 1GB，Ollama 7B Q4 约 5GB。索引结束后 BLIP 会自动释放，避免和 7B 抢显存。

机器吃不消时按这个顺序降级：`qwen2.5:3b-instruct` → 关闭 LLM 重排 → 关闭 BLIP 描述。

---

## 设计取舍

**描述在索引时生成，不在搜索时生成。** 每次搜索都对 50 张候选现算 BLIP，光描述就要几十秒。描述是图片的静态属性，算一次存进向量库 metadata，搜索时直接读。搜索时唯一需要现算的只有以图搜图的那张查询图。

**进 LLM 的候选截断到 16 个。** 50 个候选的 prompt 会到 2000+ token，7B 模型在长列表上的排序能力明显下降，延迟也不可接受。50 个召回里前 16 个已经覆盖绝大多数正确答案。

**让 LLM 打分而不是直接排序。** 「把下面 16 张图按相关性排序」对 7B 模型是不稳定任务，容易出现漏项、重复、格式崩坏。分批打绝对分数，再在本地排序，鲁棒性高得多，也方便融合。

**向量分先 min-max 再融合。** CLIP 的图文余弦相似度实际分布在 0.15~0.45，直接和 0~1 的 LLM 分加权，LLM 分会完全压倒向量信号，第一级召回就白做了。

**ID 用 uuid5 而不是 `hash()`。** Python 的字符串 hash 每个进程都加盐，重启后同一张图的 ID 会变，增量索引全部失效，删除也删不掉。uuid5(路径) 是确定性的。

**以图搜图必须先生成描述。** Qwen2.5-7B 是纯文本模型，读不了图。把查询图翻译成一段文字，它才有东西和候选描述比对。

**BLIP 输出英文是正常的。** base 模型只支持英文描述，Qwen2.5 中英都能读，prompt 里已说明「描述可能是英文」。把 BLIP 换成中文多模态模型（如 Qwen2.5-VL）会更统一，但索引成本会高一个量级。

---

## 常见问题

**首次搜索卡很久。** 在加载 CLIP，RTX 4060 上约 5~10 秒，之后常驻。界面会提示。

**LLM 显示未连接。** 检查 `ollama serve` 是否在跑，`ollama list` 里有没有 `qwen2.5:7b-instruct`。没有也能搜，只是没有精排。

**提示 collection 维度不匹配。** 换过嵌入模型但没重建索引。加 `--reset` 重新索引。

**Qdrant 报「另一个进程正在占用」。** 本地模式有文件锁，关掉另一个界面/CLI 进程，或改用 `mode: server`。

**图片显示不出来。** 结果里的路径是绝对路径，文件被移动或删除后界面会标「文件已丢失」，重新索引即可。

**中文路径有问题吗。** 代码里全链路用 `Path` 和 UTF-8，Windows 中文路径已考虑；启动脚本用了 `chcp 65001`。

**模型下载慢或失败。** 默认镜像 `hf-mirror.com`，也可以手动下好放进 `models/<短名>/`，程序会自动识别。下载脚本可以反复执行续传。

---

## Python API

```python
from imgsearch import build_pipeline

pipe = build_pipeline()

# 以文搜图
resp = pipe.search_text("夕阳下的海边", top_k=10, recall_k=50)
for r in resp.results:
    print(r.rank, r.score, r.filename, r.path)

# 以图搜图
resp = pipe.search_image("query.jpg", top_k=10)

# 建索引
stats = pipe.index_folder("D:/Photos", recursive=True)
print(stats.summary())

pipe.close()
```

完整结果结构：

```python
resp.results[0]
# SearchResult(rank=1, id='...', path='D:/Photos/a.jpg', filename='a.jpg',
#              caption='a cat sitting on a windowsill',
#              vec_score=0.341, llm_score=0.92, score=0.712, exists=True)
```

---

## 测试

```bash
python -m pytest tests -q
```

26 个用例覆盖：两个向量库的同一套契约、LLM 打分解析、失败降级、融合排序、端到端索引与检索、配置解析、切库后的一致性。

---

MIT License
