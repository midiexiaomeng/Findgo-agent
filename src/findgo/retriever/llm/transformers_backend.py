"""直接用 transformers 加载本地模型，不依赖任何外部推理服务。

和 Ollama 后端的区别：
- 模型就是 HuggingFace 格式的 safetensors，放在 models/ 下
- 进程内加载，界面常驻期间只加载一次
- 不需要额外安装运行时，也不需要 11434 端口

适用：8GB 显存建议 1.5B 或 3B。7B 的 fp16 权重要 15GB，跑不动。
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Sequence

from ..config import Config
from ..runtime import pick_device, pick_dtype

logger = logging.getLogger(__name__)


class TransformersChat:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.rc = cfg.rerank
        self._model = None
        self._tokenizer = None
        self._lock = threading.Lock()
        self.device = "cpu"
        self._path = ""

    # ── 路径与状态 ──────────────────────────────────────────────────────────

    @property
    def model_path(self) -> str:
        if self._path:
            return self._path

        raw = Path(self.rc.transformers.model_path).expanduser()
        if raw.is_absolute():
            candidates = [raw]
        else:
            # 先找 models/<名字>，再按项目根目录解析相对路径
            candidates = [self.cfg.models_dir / raw.name, self.cfg.resolve(raw)]

        for cand in candidates:
            if cand.exists():
                self._path = str(cand)
                return self._path

        self._path = str(candidates[0])
        return self._path

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def is_available(self) -> tuple[bool, str]:
        path = Path(self.model_path)
        if not path.exists():
            return False, (f"本地模型不存在: {path}。" "运行 python scripts/download_llm_hf.py 下载")
        has_weights = any(path.glob("*.safetensors")) or any(path.glob("pytorch_model*.bin"))
        if not has_weights:
            return False, f"目录里没有模型权重: {path}"
        return True, f"本地 Transformers 模型: {path.name}"

    # ── 加载与推理 ──────────────────────────────────────────────────────────

    def load(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return

            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            path = self.model_path
            self.device = pick_device(self.rc.transformers.device or self.cfg.device)
            dtype = pick_dtype(self.device, True)

            logger.info("加载本地 LLM: %s (device=%s, dtype=%s)", path, self.device, dtype)
            self._tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
            if self._tokenizer.pad_token_id is None:
                self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

            self._model = AutoModelForCausalLM.from_pretrained(
                path,
                torch_dtype=dtype,
                trust_remote_code=True,
            )
            self._model.to(self.device)
            self._model.eval()
            logger.info("本地 LLM 就绪")

    def chat(
        self,
        messages: Sequence[dict[str, str]],
        max_tokens: int | None = None,
    ) -> str:
        self.load()
        import torch

        text = self._tokenizer.apply_chat_template(
            list(messages),
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=int(self.rc.transformers.max_input_tokens),
        ).to(self.device)

        with torch.inference_mode():
            output = self._model.generate(
                **inputs,
                max_new_tokens=int(max_tokens or self.rc.max_tokens),
                do_sample=False,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )

        generated = output[0][inputs["input_ids"].shape[1] :]
        return self._tokenizer.decode(generated, skip_special_tokens=True).strip()

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
