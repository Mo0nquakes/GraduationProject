"""关键词检测器 —— 零依赖桩模型。

两个用途：
1. 让整条管线今天就能跑（不装任何包）。
2. 充当论文里的**规则基线**：它怎么崩，恰好就是 A6 实验（难负样本 / 捷径学习）
   要量化的东西。别把它当"凑数"，它是你的对照组。
"""
from __future__ import annotations

import math
from typing import Sequence

from .base import BaseDetector

CUES: dict[str, float] = {
    "ignore previous": 3.0,
    "ignore all previous": 3.5,
    "disregard": 2.5,
    "forget everything": 2.5,
    "system prompt": 2.0,
    "you are now": 2.0,
    "new instructions": 2.0,
    "override": 1.5,
    "reveal": 1.5,
    "verbatim": 1.2,
    "unfiltered": 1.5,
    "admin password": 3.0,
    "api key": 2.0,
    "attacker@": 2.5,
    "<!--": 1.5,
    "dear ai": 1.5,
    "ai-only instruction": 3.0,
    "system override": 3.0,
}


class KeywordDetector(BaseDetector):
    name = "keyword_stub"

    def __init__(self) -> None:
        self.bias: float = 0.0
        self.fitted = False

    @staticmethod
    def _raw(text: str) -> float:
        low = text.lower()
        return sum(w for cue, w in CUES.items() if cue in low)

    def fit(self, texts: Sequence[str], labels: Sequence[str], fpr_target: float = 0.05,
            **kw) -> "KeywordDetector":  # noqa: ARG002
        benign = [self._raw(t) for t, y in zip(texts, labels) if y == "benign"]
        if benign:
            benign.sort()
            # 取良性分数的 (1-fpr_target) 分位作为阈值：让训练集上的误报率接近目标值
            idx = min(len(benign) - 1, int(len(benign) * (1 - fpr_target)))
            self.bias = benign[idx]
        self.fitted = True
        return self

    def predict_proba(self, texts: Sequence[str]) -> list[float]:
        return [1.0 / (1.0 + math.exp(-2.0 * (self._raw(t) - self.bias))) for t in texts]
