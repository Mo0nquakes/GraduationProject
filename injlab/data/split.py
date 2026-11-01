"""去重 + 分层划分。

去重是这类论文的方法论底线：跨语料近重复会让测试集泄漏进训练集，
指标虚高，评审一查就崩。这里先做精确去重（sha256 of normalized text），
近似去重（MinHash/SimHash）留作 TODO —— 见 docs/开题报告.docx 7.1 节。
"""
from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from typing import Any


def normalize(text: str) -> str:
    return " ".join(text.lower().split())


def text_hash(text: str) -> str:
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()


def dedup(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """保留每个 hash 的首次出现，返回 (去重后数据, 被丢弃条数)。"""
    seen: set[str] = set()
    kept: list[dict[str, Any]] = []
    for r in rows:
        h = text_hash(r["text"])
        if h in seen:
            continue
        seen.add(h)
        kept.append({**r, "sha256": h})
    return kept, len(rows) - len(kept)


def stratified_split(
    rows: list[dict[str, Any]],
    test_ratio: float = 0.30,
    seed: int = 20260101,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """按 label 分层切分；同一 seed 结果完全可复现。"""
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_label[r["label"]].append(r)

    rng = random.Random(seed)
    train: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    for label in sorted(by_label):
        group = list(by_label[label])
        rng.shuffle(group)
        n_test = max(1, round(len(group) * test_ratio)) if len(group) > 1 else 0
        test.extend(group[:n_test])
        train.extend(group[n_test:])
    rng.shuffle(train)
    rng.shuffle(test)
    for r in train:
        r["split"] = "train"
    for r in test:
        r["split"] = "test"
    return train, test
