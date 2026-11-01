"""DistilBERT 基线检测器（A1 实验里"传统 ML vs 预训练模型"的那一列）。

刻意**不**用 `Trainer`：这个课题要画的是"攻击前后分数怎么变"，
需要能直接拿到 logits、能自己控制 padding 和分批，用 Trainer 反而绕。

三个 macOS/MPS 的坑（都在代码里处理了）：
1. MPS 不支持 float64 —— 别在训练循环里出现 `.double()` 或 float64 张量。
2. `DataLoader(num_workers>0)` 在 macOS 上会 spawn 子进程，而本项目在
   notebook / Streamlit 里跑，spawn 会炸 —— 所以这里 num_workers=0。
3. `pin_memory` 在 MPS 上无效，留着只会浪费时间。
"""
from __future__ import annotations

import random
from typing import Any, Sequence

from .base import BaseDetector


def _pick_device() -> str:
    try:
        import torch
    except ModuleNotFoundError:  # pragma: no cover
        return "cpu"
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class DistilBertDetector(BaseDetector):
    """二分类（benign / injection）微调。

    用法：
        det = DistilBertDetector(epochs=3)
        det.fit(train_rows)          # rows: [{"text":..., "label":...}, ...]
        scores = det.predict_proba(texts)

    参数说明：
        max_len    截断长度。**注意**：这里故意用朴素的截断，就是为了让
                   "截断攻击"实验能复现——真实系统才该用 chunking.max_chunk_score。
        chunked    True 时改用分块滑窗取最大分，对应论文里的缓解措施。
    """

    def __init__(self, model_name: str = "distilbert-base-uncased",
                 epochs: int = 3, batch_size: int = 16, lr: float = 2e-5,
                 max_len: int = 256, chunked: bool = False,
                 chunk_size: int = 256, chunk_stride: int = 128,
                 device: str | None = None, seed: int = 20260101,
                 verbose: bool = True) -> None:
        self.model_name = model_name
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.max_len = max_len
        self.chunked = chunked
        self.chunk_size = chunk_size
        self.chunk_stride = chunk_stride
        self.device_name = device or _pick_device()
        self.seed = seed
        self.verbose = verbose
        self.name = f"distilbert[{model_name}]{'+chunk' if chunked else ''}"
        self._model = None
        self._tok = None
        self._torch = None

    # ---------- 内部 ----------
    def _lazy(self) -> None:
        if self._model is not None:
            return
        # 不包 try/except：让 ModuleNotFoundError 原样冒泡，
        # e.name 才是 "torch" 而不是 None，cli._fit 才能给出准确提示。
        import torch
        from transformers import (AutoModelForSequenceClassification,
                                  AutoTokenizer)

        self._torch = torch
        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=2).to(self.device_name)

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"  [distilbert] {msg}")

    @staticmethod
    def _to_binary(label: str) -> int:
        return 0 if label == "benign" else 1

    # ---------- 训练 ----------
    def fit(self, texts: Sequence[str], labels: Sequence[str],
            fpr_target: float = 0.01, **_: Any) -> "DistilBertDetector":
        self._lazy()
        torch = self._torch
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        # 与其他检测器保持同一接口：fit(texts, labels, fpr_target=...)
        # fpr_target 对神经网络没用（它不按分位数定阈值），阈值一律交给
        # runner 在训练集上校准 —— 这一点必须全校统一，否则实验不可比。
        texts = list(texts)
        y = [self._to_binary(lab) for lab in labels]

        # 动态 padding：每个 batch 单独 pad 到该 batch 最长，比全局 pad 快不少
        enc = self._tok(texts, truncation=True, max_length=self.max_len,
                        padding=False, return_tensors=None)
        ds = [{"input_ids": enc["input_ids"][i],
               "attention_mask": enc["attention_mask"][i],
               "labels": y[i]} for i in range(len(texts))]

        from torch.utils.data import DataLoader
        from transformers import DataCollatorWithPadding

        collate = DataCollatorWithPadding(tokenizer=self._tok, return_tensors="pt")
        # num_workers=0：macOS + MPS 下多进程会 spawn 崩
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True,
                            collate_fn=collate, num_workers=0, pin_memory=False)

        opt = torch.optim.AdamW(self._model.parameters(), lr=self.lr)
        self._model.train()
        for ep in range(self.epochs):
            total, seen = 0.0, 0
            for batch in loader:
                batch = {k: v.to(self.device_name) for k, v in batch.items()}
                out = self._model(**batch)
                loss = out.loss
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += float(loss.item()) * len(batch["labels"])
                seen += len(batch["labels"])
            self._log(f"epoch {ep + 1}/{self.epochs}  loss={total / max(1, seen):.4f}")
        self._model.eval()
        return self

    # ---------- 推理 ----------
    def predict_proba(self, texts: Sequence[str]) -> list[float]:
        self._lazy()
        if self.chunked:
            from ..chunking import max_chunk_score

            return [max_chunk_score(lambda t: self._forward([t])[0], t,
                                    size=self.chunk_size, stride=self.chunk_stride)[0]
                    for t in texts]
        return self._forward(list(texts))

    def _forward(self, texts: list[str], batch: int = 32) -> list[float]:
        torch = self._torch
        out: list[float] = []
        self._model.eval()
        with torch.no_grad():
            for i in range(0, len(texts), batch):
                part = texts[i:i + batch]
                enc = self._tok(part, truncation=True, max_length=self.max_len,
                                padding=True, return_tensors="pt").to(self.device_name)
                logits = self._model(**enc).logits
                probs = torch.softmax(logits.float(), dim=-1)[:, 1]  # float32，别用 double
                out.extend(float(p) for p in probs.cpu())
        return out
