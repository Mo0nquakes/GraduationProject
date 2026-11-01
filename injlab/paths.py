"""数据目录的定位逻辑，集中一处，免得三个调用方各自写一遍并各自写错。

踩过的坑：`sorted(glob("*"))[-1]` 取"最新"的那个，会在跑过一次
`build-data --sources fixture`（离线自检用的 65 条玩具集）之后，把 fixture
当成主数据集，于是 A1 报出 TPR 0.643 / FPR 0.333 这种既难看又没意义的数字，
而真实数据其实好得多。所以这里的规则是：**默认跳过 fixture**，
只有显式点名时才用。
"""
from __future__ import annotations

from pathlib import Path

TOY_MARKERS = ("fixture", "toy", "smoke")


def is_toy(name: str) -> bool:
    low = name.lower()
    return any(m in low for m in TOY_MARKERS)


def list_datasets(processed_dir: str | Path = "data/processed") -> list[Path]:
    return sorted(p.parent for p in Path(processed_dir).glob("*/train.jsonl"))


def latest_dataset(processed_dir: str | Path = "data/processed",
                   include_toy: bool = False) -> Path | None:
    cands = [p for p in list_datasets(processed_dir) if include_toy or not is_toy(p.name)]
    if not cands:
        cands = list_datasets(processed_dir)  # 只剩玩具集时也得能用
    return cands[-1] if cands else None
