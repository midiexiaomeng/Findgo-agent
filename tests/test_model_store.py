"""ModelStore 测试：增删、持久化、激活切换、配置转换。"""

from __future__ import annotations

from findgo.model_store import ModelStore


def test_add_and_active(tmp_path):
    store = ModelStore(tmp_path / "models.json")
    entry = store.add(
        "deepseek", "deepseek-flash", "https://api.deepseek.com/v1", "sk-1"
    )
    assert store.active == entry["id"]
    assert len(store.models) == 1
    assert store.get_active()["model"] == "deepseek-flash"


def test_persist_and_reload(tmp_path):
    path = tmp_path / "models.json"
    store = ModelStore(path)
    store.add("deepseek", "deepseek-flash", "https://x/v1", "sk-1")
    store.save()

    store2 = ModelStore(path)
    assert len(store2.models) == 1
    assert store2.models[0]["api_key"] == "sk-1"


def test_duplicate_replaces(tmp_path):
    store = ModelStore(tmp_path / "models.json")
    store.add("deepseek", "deepseek-flash", "https://x/v1", "sk-1")
    store.add("deepseek", "deepseek-flash", "https://x/v1", "sk-2")
    assert len(store.models) == 1
    assert store.models[0]["api_key"] == "sk-2"


def test_remove_updates_active(tmp_path):
    store = ModelStore(tmp_path / "models.json")
    first = store.add("a", "m1", "u1", "k")
    second = store.add("b", "m2", "u2", "k")
    store.set_active(first["id"])
    store.remove(first["id"])
    assert store.active == second["id"]


def test_to_llm_config():
    cfg = ModelStore.to_llm_config(
        {"model": "m", "base_url": "https://x/v1", "api_key": "k"}
    )
    assert cfg["backend"] == "openai"
    assert cfg["model"] == "m"
    assert cfg["base_url"] == "https://x/v1"
    assert cfg["api_key"] == "k"
