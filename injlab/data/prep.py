"""语料处理：清洗 → 去重 → 配平 → 分组切分。

每一步都对应论文里要交代的一句话，所以函数名取得很直白：

    清洗   为什么需要：攻击者用零宽字符和全角字母绕过，训练数据里如果
           全是干净文本，模型学不到这些变体。但**清洗只用于去重比对和
           规范化视图**，训练时保留原文（否则等于提前把防御做进了数据）。
    去重   为什么需要：模板生成的数据集里同一句话出现 N 次，会让测试集
           和训练集共享同一条样本，指标虚高。
    配平   为什么需要：类不平衡会让模型偏向多数类，FPR 失真。
    分组切分 为什么需要：同一段话的多个变体分到两边就是泄漏。
"""
from __future__ import annotations

import hashlib
import random
import re
import unicodedata
from collections import Counter, defaultdict

from ..labels import BENIGN, DIRECT, INDIRECT, LABELS

_WS = re.compile(r"\s+")
_CTRL_CATS = {"Cf", "Cc", "Cs", "Co", "Cn"}


def clean_text(s: str) -> str:
    """NFKC 归一 + 剥离零宽/格式控制字符 + 折叠空白。

    这是"规范化视图"，用于去重和展示；训练用的仍是原文。
    """
    s = unicodedata.normalize("NFKC", s or "")
    s = "".join(ch for ch in s if unicodedata.category(ch) not in _CTRL_CATS)
    s = _WS.sub(" ", s)
    return s.strip()


def dedup_key(s: str) -> str:
    """精确去重键：清洗后小写折叠。"""
    return hashlib.sha256(clean_text(s).lower().encode("utf-8")).hexdigest()


def shingles(s: str, n: int = 5) -> set[str]:
    t = re.sub(r"[^\w\s]", " ", clean_text(s).lower())
    words = t.split()
    if len(words) < n:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def minhash_sig(s: str, perms: int = 32, n: int = 5) -> tuple[int, ...]:
    """MinHash 签名（纯标准库，不引 datasketch）。

    每个 shingle 用 blake2b(seed||shingle) 得到一个 64 位整数，
    取 perms 个不同 seed 的最小值作为签名。两个文档签名在任一位置相等
    的概率≈它们的 Jaccard 相似度。
    """
    sh = shingles(s, n)
    if not sh:
        return tuple([0] * perms)
    sig = []
    for p in range(perms):
        prefix = f"{p}:".encode()
        sig.append(min(
            int.from_bytes(hashlib.blake2b(prefix + x.encode(), digest_size=8).digest(), "big")
            for x in sh
        ))
    return tuple(sig)


def dedup(records: list, near: bool = True, bands: int = 8, progress_every: int = 500) -> tuple[list, dict]:
    """先精确去重，再近似去重（MinHash banding）。

    返回 (保留的记录, 统计信息)。统计信息要写进 manifest，论文里报"去掉了多少"。
    """
    seen: set[str] = set()
    kept: list = []
    exact_dup = 0
    for r in records:
        k = dedup_key(r.text)
        if k in seen:
            exact_dup += 1
            continue
        seen.add(k)
        kept.append(r)

    near_dup = 0
    if near:
        perms = bands * 4
        buckets: dict[tuple, int] = {}
        survivors: list = []
        for i, r in enumerate(kept):
            sig = minhash_sig(r.text, perms=perms)
            hit = False
            for b in range(bands):
                key = (b, sig[b * 4:(b + 1) * 4])
                if key in buckets:
                    hit = True
                    break
            if hit:
                near_dup += 1
                continue
            for b in range(bands):
                buckets[(b, sig[b * 4:(b + 1) * 4])] = i
            survivors.append(r)
            if progress_every and i and i % progress_every == 0:
                print(f"    去重进度 {i}/{len(kept)}")
        kept = survivors

    return kept, {"input": len(records), "exact_dup": exact_dup,
                  "near_dup": near_dup, "kept": len(kept)}


def balance(records: list, max_per_class: int | None = None, seed: int = 0) -> list:
    """按标签截断到每类最多 max_per_class 条（随机抽样，固定种子）。"""
    if not max_per_class:
        return records
    rng = random.Random(seed)
    by: dict[str, list] = defaultdict(list)
    for r in records:
        by[r.label].append(r)
    out = []
    for lab in LABELS:
        items = by.get(lab, [])
        if len(items) > max_per_class:
            items = rng.sample(items, max_per_class)
        out.extend(items)
    return out


def group_split(records: list, test_ratio: float = 0.3, seed: int = 0) -> tuple[list, list]:
    """按 group 切分，保证同一 group 的样本只落在一侧。

    group 为空时退化为按文本哈希取模——仍然是确定性的，
    换台机器跑结果一致。
    """
    rng = random.Random(seed)
    groups: dict[str, list] = defaultdict(list)
    for r in records:
        g = getattr(r, "group", "") or dedup_key(r.text)
        groups[g].append(r)

    keys = sorted(groups)
    rng.shuffle(keys)

    n_test_target = int(round(len(records) * test_ratio))
    test, train = [], []
    for k in keys:
        if len(test) < n_test_target:
            test.extend(groups[k])
        else:
            train.extend(groups[k])

    # 兜底：某一侧为空说明数据太少，随机对半
    if not train or not test:
        shuffled = list(records)
        rng.shuffle(shuffled)
        cut = max(1, int(len(shuffled) * test_ratio))
        return shuffled[cut:], shuffled[:cut]
    return train, test


def summary(records: list) -> dict:
    """统计每个标签、来源、硬负样本占比。"""
    labels = Counter(r.label for r in records)
    sources = Counter(r.source for r in records)
    hard = sum(1 for r in records if getattr(r, "hard", False))
    return {
        "n": len(records),
        "by_label": {k: labels.get(k, 0) for k in LABELS},
        "by_source": dict(sources.most_common()),
        "hard_negatives": hard,
        "empty_text": sum(1 for r in records if not r.text.strip()),
        "avg_chars": round(sum(len(r.text) for r in records) / max(1, len(records)), 1),
    }

def leakage_report(train: list, test: list, threshold: float = 0.8,
                   perms: int = 32, n: int = 5) -> dict:
    """检查训练/测试之间是否还有近似重复（数据泄漏自检）。

    为什么必须单独做这一步：分组切分只保证"同 group 不跨侧"，如果切分时
    group 字段缺失或不干净，同一段注入换个标点就会分到两侧。这里用 MinHash
    签名做两两比对，把最像的几对打印出来，让作者自己判断该不该合并。

    返回 {"max_sim", "n_pairs_over", "worst": [(sim, train_i, test_i), ...]}
    """
    if not train or not test:
        return {"max_sim": 0.0, "n_pairs_over": 0, "worst": []}
    sig_tr = [minhash_sig(r.text, perms=perms, n=n) for r in train]
    sig_te = [minhash_sig(r.text, perms=perms, n=n) for r in test]
    pairs: list[tuple[float, int, int]] = []
    over = 0
    for j, s2 in enumerate(sig_te):
        for i, s1 in enumerate(sig_tr):
            sim = sum(1 for a, b in zip(s1, s2) if a == b) / perms
            if sim >= threshold:
                over += 1
            if sim > 0.5:
                pairs.append((sim, i, j))
    pairs.sort(reverse=True)
    return {
        "max_sim": round(pairs[0][0], 3) if pairs else 0.0,
        "n_pairs_over": over,
        "threshold": threshold,
        "worst": [(round(s_, 3), tr_i, te_i) for s_, tr_i, te_i in pairs[:5]],
    }
