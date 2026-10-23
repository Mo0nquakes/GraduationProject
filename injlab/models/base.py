"""检测器统一接口。

全系统的关键契约：任何检测器只暴露两个方法 + 一个 name。
新增 DistilBERT / 现成 guardrail / LLM-as-judge，只要满足这个协议，
实验矩阵、API、界面都不用改一行 —— 这是后面做 A1/A3/A5 对比的前提。
"""
from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable


@runtime_checkable
class Detector(Protocol):
    name: str

    def fit(self, texts: Sequence[str], labels: Sequence[str], **kw) -> "Detector": ...

    def predict_proba(self, texts: Sequence[str]) -> list[float]:
        """返回"恶意"概率，长度与 texts 一致。"""
        ...


class BaseDetector:
    name = "base"

    def fit(self, texts: Sequence[str], labels: Sequence[str], **kw) -> "BaseDetector":  # noqa: ARG002
        raise NotImplementedError

    def predict_proba(self, texts: Sequence[str]) -> list[float]:  # noqa: ARG002
        raise NotImplementedError

    def predict(self, texts: Sequence[str], threshold: float = 0.5) -> list[int]:
        return [1 if p >= threshold else 0 for p in self.predict_proba(texts)]
