"""现成 guardrail 基线：直接拿别人训好的安全分类器来比。

为什么必须放这一列：审稿人第一个会问的就是"你和现成的 Llama Guard 比过吗"。
没有这一列，整个 A1 表就只是"我们自己的三个玩具比谁更好"。

本模块支持两类现成模型：

1. **编码器型**（默认，实测可用）：
   `protectai/deberta-v3-base-prompt-injection-v2`，二分类 SAFE / INJECTION，
   约 700MB，CPU 上也能跑。`predict_proba` 直接返回 P(INJECTION)。
2. **生成式型**（`llama-guard`）：Meta 的 Llama Guard 系列，靠 chat template
   输出 "safe"/"unsafe"。需要先在 HF 上同意条款并登录，仓库里不带权重。

**一个必须写进论文的差别**：guardrail 是"开箱即用"的——它没有用我们的训练集，
所以对它来说不存在"训练"，它的阈值也不该用我们的训练集校准，否则等于给它
额外喂了标签。`fit()` 是空操作，阈值固定 0.5，这是所有现成检测器的共同待遇。
"""
from __future__ import annotations

import os

from typing import Any, Sequence

from ..hfenv import ensure_hf_env
from .base import BaseDetector

ensure_hf_env()

# 默认用 **base** 版（DeBERTa-v3-base，184M 参数 ≈ 700MB 权重）。
# 曾试过更轻的 `deberta-v3-small-prompt-injection-v2`（44M），但它是 **gated**
# 模型，没有授权拿不到（报 "restricted and you are not in the authorized list"）。
# 8GB 内存的机器上这个模型单独跑没问题（实测峰值约 1.1GB），
# 但不能和其他模型同进程加载——所以 A1 里每个检测器都单独起子进程。
DEFAULT_ENC_MODEL = os.environ.get(
    "INJLAB_GUARDRAIL_MODEL",
    "protectai/deberta-v3-base-prompt-injection-v2",
)


class GuardrailDetector(BaseDetector):
    """现成 guardrail：不做训练，只做推理。"""

    def __init__(self, model_name: str = DEFAULT_ENC_MODEL, max_len: int = 512,
                 device: str | None = None, batch_size: int = 16,
                 verbose: bool = True) -> None:
        self.model_name = model_name
        self.max_len = max_len
        self.batch_size = batch_size
        self.verbose = verbose
        self.device_name = device
        self.name = f"guardrail[{model_name.split('/')[-1]}]"
        self._pipe = None
        self._torch = None

    # ---------- 内部 ----------
    def _lazy(self) -> None:
        if self._pipe is not None:
            return
        import torch
        from transformers import (AutoModelForSequenceClassification,
                                  AutoTokenizer, pipeline)

        self._torch = torch
        if self.device_name is None:
            if torch.backends.mps.is_available():
                self.device_name = "mps"
            elif torch.cuda.is_available():
                self.device_name = "cuda"
            else:
                self.device_name = "cpu"

        tok = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        try:
            self._pipe = pipeline("text-classification", model=model, tokenizer=tok,
                                  device=self.device_name, top_k=None,
                                  truncation=True, max_length=self.max_len)
            # 先跑一条热身：MPS 上某些算子在第一次调用时才报错
            self._pipe("warmup")
        except Exception as e:  # MPS 偶发不支持某些 op → 退回 CPU
            if self.verbose:
                print(f"  [guardrail] {self.device_name} 不可用（{type(e).__name__}），退回 CPU")
            self.device_name = "cpu"
            self._pipe = pipeline("text-classification", model=model, tokenizer=tok,
                                  device="cpu", top_k=None,
                                  truncation=True, max_length=self.max_len)
        if self.verbose:
            print(f"  [guardrail] {self.name} on {self.device_name}")

    def _injection_index(self) -> int:
        """找出"注入"那一类的下标——不同模型的 label 名不一样。"""
        cfg = self._pipe.model.config
        id2label = {int(k): str(v) for k, v in getattr(cfg, "id2label", {}).items()}
        for idx, lab in id2label.items():
            low = lab.lower()
            if any(w in low for w in ("inject", "unsafe", "harm", "malicious", "label_1")):
                return idx
        return 1  # 兜底：官方卡里 SAFE=0 / INJECTION=1

    # ---------- 接口 ----------
    def fit(self, texts: Sequence[str], labels: Sequence[str], **_: Any) -> "GuardrailDetector":
        """开箱即用模型不训练。保留这个方法只是为了让接口和其他检测器一致。"""
        self._lazy()
        return self

    def predict_proba(self, texts: Sequence[str]) -> list[float]:
        self._lazy()
        idx = self._injection_index()
        out: list[float] = []
        for i in range(0, len(texts), self.batch_size):
            part = [self._prep(t) for t in texts[i:i + self.batch_size]]
            res = self._pipe(part, batch_size=self.batch_size)
            for item in res:
                # top_k=None 时返回 [[{label,score},...], ...]
                scores = {int(r["label"].split("_")[-1]) if r["label"].startswith("LABEL_")
                          else None: r["score"] for r in item}
                if None in scores:  # label 是文本名，用 label→id 映射
                    id2label = {str(v).lower(): int(k) for k, v in
                                self._pipe.model.config.id2label.items()}
                    scores = {id2label.get(str(r["label"]).lower(), -1): r["score"]
                              for r in item}
                out.append(float(scores.get(idx, 0.0)))
        return out

    @staticmethod
    def _prep(text: str) -> str:
        # DeBERTa 的 tokenizer 对超长文本会截断；这里保证空串不会崩
        return text if text.strip() else " "
