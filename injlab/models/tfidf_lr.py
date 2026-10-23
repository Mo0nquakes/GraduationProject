"""TF-IDF + 逻辑回归 —— 真正的机器学习基线（需要 scikit-learn）。

两个变体值得各跑一次，结论会直接进论文 A1 表：
- analyzer="word"    : 词级，抓"语义模板"，但被零宽字符/同形字一打就掉
- analyzer="char_wb" : 字符级，对编码混淆天然更稳，是你的隐蔽对照组
"""
from __future__ import annotations

from typing import Sequence

from .base import BaseDetector


class TfidfLogReg(BaseDetector):
    def __init__(self, analyzer: str = "word", ngram_range: tuple[int, int] = (1, 2),
                 name: str | None = None) -> None:
        self.analyzer = analyzer
        self.ngram_range = ngram_range
        self.name = name or f"tfidf_lr[{analyzer}]"
        self.vec = None
        self.clf = None

    def fit(self, texts: Sequence[str], labels: Sequence[str], **kw) -> "TfidfLogReg":  # noqa: ARG002
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        y = [1 if lab != "benign" else 0 for lab in labels]
        self.vec = TfidfVectorizer(analyzer=self.analyzer, ngram_range=self.ngram_range,
                                   sublinear_tf=True, min_df=1, lowercase=True)
        X = self.vec.fit_transform(texts)
        self.clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
        self.clf.fit(X, y)
        return self

    def predict_proba(self, texts: Sequence[str]) -> list[float]:
        if self.vec is None or self.clf is None:
            raise RuntimeError("先调用 fit()")
        X = self.vec.transform(list(texts))
        return [float(p) for p in self.clf.predict_proba(X)[:, 1]]
