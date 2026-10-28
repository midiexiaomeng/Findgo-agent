"""配置层测试：默认值、YAML 合并、环境变量覆盖、模型路径解析。"""

from __future__ import annotations

from pathlib import Path

from findgo.retriever.config import Config
from findgo.retriever.runtime import resolve_model_source


def test_defaults_are_consistent():
    cfg = Config()
    assert cfg.vector_store.backend in ("chromadb", "qdrant")
    assert cfg.embedding.embedding_dim == 512
    assert cfg.search.recall_k >= cfg.search.top_k
    assert cfg.rerank.max_candidates >= cfg.search.top_k


def test_yaml_partial_merge(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "vector_store:\n  backend: qdrant\n  qdrant:\n    mode: server\n",
        encoding="utf-8",
    )
    cfg = Config.from_yaml(config_file)

    assert cfg.vector_store.backend == "qdrant"
    assert cfg.vector_store.qdrant.mode == "server"
    # 未写的字段保留默认值
    assert cfg.vector_store.collection == "images"
    assert cfg.vector_store.qdrant.url == "http://localhost:6333"


def test_unknown_key_is_ignored(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("not_a_section:\n  foo: 1\n", encoding="utf-8")
    cfg = Config.from_yaml(config_file)  # 不应抛异常
    assert isinstance(cfg, Config)


def test_env_override(monkeypatch):
    monkeypatch.setenv("FINDGO_VECTOR_STORE__BACKEND", "qdrant")
    cfg = Config()
    cfg._apply_env_overrides()
    assert cfg.vector_store.backend == "qdrant"


def test_resolve_model_source_prefers_local(tmp_path):
    local = tmp_path / "my-clip"
    local.mkdir()
    (local / "config.json").write_text("{}", encoding="utf-8")

    resolved = resolve_model_source("my-clip", tmp_path, "OFA-Sys/chinese-clip")
    assert Path(resolved) == local


def test_resolve_model_source_falls_back(tmp_path):
    resolved = resolve_model_source("nope", tmp_path, "OFA-Sys/chinese-clip")
    assert resolved == "OFA-Sys/chinese-clip"


def test_paths_are_absolute():
    cfg = Config()
    assert cfg.models_dir.is_absolute()
    assert cfg.data_dir.is_absolute()
