"""离线模式与远端模型后端测试。"""

from __future__ import annotations

import pytest
from findgo.retriever.captioner import Captioner
from findgo.retriever.clip_encoder import ClipEncoder
from findgo.retriever.config import Config
from findgo.retriever.llm.client import LLMClient


def test_offline_forces_backend_off():
    cfg = Config()
    cfg.offline = True
    cfg.rerank.backend = "openai"

    client = LLMClient(cfg)
    assert client.backend == "none"

    ok, reason = client.is_available()
    assert ok is False
    assert "离线" in reason


def test_offline_keeps_local_backend():
    cfg = Config()
    cfg.offline = True
    cfg.rerank.backend = "transformers"

    client = LLMClient(cfg)
    assert client.backend == "transformers"


def test_online_keeps_configured_backend():
    cfg = Config()
    cfg.offline = False
    cfg.rerank.backend = "openai"

    client = LLMClient(cfg)
    assert client.backend == "openai"


def test_clip_api_mode_skips_local_load():
    cfg = Config()
    cfg.embedding.backend = "api"
    cfg.embedding.api_base_url = "https://api.example.com/v1"

    encoder = ClipEncoder(cfg)
    assert encoder.loaded is True
    assert encoder.source == "https://api.example.com/v1"


def test_clip_local_mode_not_loaded_before_load():
    cfg = Config()
    cfg.embedding.backend = "local"

    encoder = ClipEncoder(cfg)
    assert encoder.loaded is False


def test_caption_api_mode_ready_without_model():
    cfg = Config()
    cfg.caption.backend = "api"
    cfg.caption.api_base_url = "https://api.example.com/v1"

    captioner = Captioner(cfg)
    assert captioner._api is True


def test_caption_api_without_url_returns_empty(tmp_path):
    cfg = Config()
    cfg.caption.backend = "api"
    cfg.caption.api_base_url = ""

    captioner = Captioner(cfg)
    captions = captioner._generate_api([{"path": str(tmp_path / "a.jpg")}])
    assert captions == [""]


def test_offline_flag_default_false():
    assert Config().offline is False


def test_warmup_default_true():
    assert Config().warmup_on_start is True


def test_export_dir_default():
    assert Config().paths.export_dir == "./data/exports"


@pytest.mark.parametrize("backend", ["local", "api"])
def test_embedding_backend_roundtrip(backend):
    cfg = Config()
    cfg.embedding.backend = backend
    assert cfg.embedding.backend == backend
