# 部署与启动

## 环境要求

- Python 3.9 及以上
- 磁盘：模型约 1.6GB（CLIP + BLIP），若用本地 LLM 另需 3~6GB
- 显卡：可选。有 NVIDIA 显卡会明显加速嵌入与描述；纯 CPU 也能跑，只是慢

## 1. 安装依赖

```powershell
pip install -r requirements-core.txt
```

如果还没有 torch，按显卡选择：

```powershell
# CUDA 12.4
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
# 纯 CPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## 2. 下载模型

```powershell
python scripts\download_models.py
```

下载到 `models/`，默认走 hf-mirror，支持断点续传。

## 3. 配置 LLM

复制模板：

```powershell
python scripts\init_agent_config.py
```

编辑 `configs/agent.yaml`，或直接用环境变量（推荐）：

```powershell
setx DEEPSEEK_API_KEY "sk-你的key"
```

然后**重开终端**。环境变量优先于配置文件。

检索侧的重排配置在 `config.yaml` 的 `rerank` 段，默认已指向 DeepSeek。

## 4. 建立索引

```powershell
python cli.py index "D:\你的图片目录"
```

首次会加载 CLIP 和 BLIP，之后是增量索引。

## 5. 启动

**桌面应用（推荐）**

双击 `start-app.bat`，无控制台、不占端口。

或：

```powershell
python desktop_app.py
```

**网页版（可选）**

```powershell
python run.py
```

打开 `http://127.0.0.1:7861`。

## 常见问题

**首次对话很久没反应。** 在加载 CLIP，约 10 到 20 秒，之后每次几秒。

**提示 API 401。** Key 没读到。确认是在新终端里启动，`echo $env:DEEPSEEK_API_KEY` 有值。

**检索结果全是纯向量排序。** 看日志里的 `LLM 不可用` 原因，通常是 `base_url` 或 Key 不对。

**ChromaDB 报 HNSW 索引错误。** 本项目在 Windows + ChromaDB 1.5 上实测过索引写入异常，建议使用 Qdrant（`config.yaml` 里 `vector_store.backend: qdrant`）。

**8GB 显存跑本地大模型很慢。** 改用 API 后端，或把 `embedding.device` 设为 `cpu`，让 LLM 独占显存。
