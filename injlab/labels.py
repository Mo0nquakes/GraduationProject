"""标签空间：三分类为主，对外同时报二分类口径。

三分类的理由（论文 3.1 节）：
- direct_injection  用户在输入框里直接投毒
- indirect_injection 恶意指令藏在被检索的文档 / 网页 / 工具返回里
两者事故性质完全不同，混成一类会掩盖最关键的失效模式。
"""
from __future__ import annotations

BENIGN = "benign"
DIRECT = "direct_injection"
INDIRECT = "indirect_injection"

LABELS: tuple[str, ...] = (BENIGN, DIRECT, INDIRECT)
LABEL2ID = {name: i for i, name in enumerate(LABELS)}
ID2LABEL = {i: name for name, i in LABEL2ID.items()}

# 二分类口径：恶意 = 任意一种注入
MALICIOUS_IDS = frozenset({LABEL2ID[DIRECT], LABEL2ID[INDIRECT]})


def to_binary(label: str) -> int:
    """标签字符串 -> 0/1（1 = 恶意）。与 PINT / Llama Guard 口径对齐用。"""
    return 0 if label == BENIGN else 1


def is_malicious(label: str) -> bool:
    return label != BENIGN
