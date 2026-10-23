"""攻击接口：text -> text。

关键约束：攻击只作用于**恶意样本**（标签不变），目的是让检测器把它判成良性。
这就是"逃逸攻击（evasion）"的定义，也是 ASR 的定义边界。
"""
from __future__ import annotations


class BaseAttack:
    name = "base"
    family = "obfuscation"

    def apply(self, text: str) -> str:
        raise NotImplementedError

    def __call__(self, text: str) -> str:
        return self.apply(text)
