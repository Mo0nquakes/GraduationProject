"""评价指标 —— 刻意不用 sklearn.metrics。

原因：安全系统要报的不是 Accuracy，而是
    TPR @ FPR = 1%   （业界 guardrail 的标准报法，PINT / Llama Guard 都在用）
    ASR              （攻击成功率 = 恶意样本被判成良性的比例）
这两个 sklearn 都没有现成函数。手写一遍，也顺便让你在答辩时能白板推出来。
"""
from __future__ import annotations

import math
from typing import Sequence


def tpr_at_fpr(y: Sequence[int], scores: Sequence[float], fpr_target: float = 0.01) -> dict:
    """在误报率不超过 fpr_target 的前提下，能达到的最大召回率。

    做法：按分数降序扫一遍，等价于遍历所有可能阈值，O(n log n)。
    """
    pairs = sorted(zip(scores, y), key=lambda t: -t[0])
    n_pos = sum(1 for _, v in pairs if v == 1)
    n_neg = len(pairs) - n_pos
    if n_pos == 0 or n_neg == 0:
        return {"tpr": float("nan"), "fpr": float("nan"), "thr": None}

    tp = fp = 0
    best = {"tpr": 0.0, "fpr": 0.0, "thr": float("inf")}
    i = 0
    while i < len(pairs):
        s = pairs[i][0]
        # 判定规则全系统统一为 score > thr，所以先按"严格大于 s"结算一次
        fpr, tpr = fp / n_neg, tp / n_pos
        if fpr <= fpr_target and tpr > best["tpr"]:
            best = {"tpr": tpr, "fpr": fpr, "thr": s}
        while i < len(pairs) and pairs[i][0] == s:  # 并入并列分数
            if pairs[i][1] == 1:
                tp += 1
            else:
                fp += 1
            i += 1
    if tp / n_pos > best["tpr"] and fp / n_neg <= fpr_target:
        best = {"tpr": tp / n_pos, "fpr": fp / n_neg, "thr": float("-inf")}
    return best


def threshold_at_fpr(y: Sequence[int], scores: Sequence[float],
                     fpr_target: float = 0.01) -> float:
    """在训练集上按误报率约束选阈值（负样本分数的 (1-fpr) 分位）。

    比"扫描所有阈值取最优"更稳：样本量小的时候扫描法很容易落到 inf（无解）。
    这也是工业界 guardrail 的常规做法。
    """
    neg = sorted(s for s, v in zip(scores, y) if v == 0)
    if not neg:
        return 0.5
    idx = min(len(neg) - 1, int(round((1.0 - fpr_target) * (len(neg) - 1))))
    return float(neg[idx])


def auc(y: Sequence[int], scores: Sequence[float]) -> float:
    """基于秩的 AUC，含并列处理（等价于 Mann-Whitney U）。"""
    pairs = sorted(zip(scores, y))
    n_pos = sum(1 for _, v in pairs if v == 1)
    n_neg = len(pairs) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    rank_sum = 0.0
    i = 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + j + 1) / 2.0  # 1-based 平均秩
        rank_sum += sum(avg_rank for k in range(i, j) if pairs[k][1] == 1)
        i = j
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def prf(y: Sequence[int], pred: Sequence[int]) -> dict:
    tp = sum(1 for a, b in zip(y, pred) if a == 1 and b == 1)
    fp = sum(1 for a, b in zip(y, pred) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(y, pred) if a == 1 and b == 0)
    tn = sum(1 for a, b in zip(y, pred) if a == 0 and b == 0)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": prec, "recall": rec, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def asr(scores: Sequence[float], threshold: float) -> float:
    """攻击成功率：恶意样本被判定为良性（score <= threshold）的比例。"""
    if not scores:
        return float("nan")
    return sum(1 for s in scores if s <= threshold) / len(scores)


def fmt(x: float, nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return " n/a "
    return f"{x:.{nd}f}"
