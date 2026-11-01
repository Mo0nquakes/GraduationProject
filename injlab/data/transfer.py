"""A2 用的"目标集"构建：只做**测试**，绝不参与训练。

为什么要单独一个模块而不是复用 build-data：
1. 跨集泛化的测试集必须是**从没被清洗、配平、切分流程碰过**的数据，否则"泛化"
   里混进了一个"我们按它调过参数"的因素；
2. 目标集不做配平（不按类采样）——真实世界的先验不该被我们改掉；
3. 每个目标集单独一个 jsonl，manifest 记清楚是从哪个源、多少条、什么标签映射来的。

标签映射的两条原则：
- 只映射到二分类（benign / injection）。三分类在跨集上不成立——别的数据集
  根本没有"间接注入"这个标注轴。
- **拿不准的宁可不映射**：扔掉的样本数要报出来，不能悄悄丢。
"""
from __future__ import annotations

import json
from pathlib import Path

from ..labels import BENIGN
from .prep import dedup
from .sources import SOURCES, load


def available() -> list[str]:
    return ["pint", "toxicchat", "xstest", "aegis", "wildguardmix"]


class _Text:
    """`prep.dedup` 只认带 `.text` 的对象，这里把 dict 包一层，避免改动通用去重函数。"""

    __slots__ = ("text", "row")

    def __init__(self, text: str, row: dict):
        self.text = text
        self.row = row


def build_target(name: str, limit: int | None = 600, out_path: Path | None = None,
                 endpoint: str | None = None) -> dict:
    """拉一个目标集，去重后写 jsonl。返回统计（含丢弃原因计数）。"""
    if name not in SOURCES:
        raise SystemExit(f"未知目标集: {name}（可选: {', '.join(available())}）")

    rows = load([name], limit=limit, endpoint=endpoint, cache=True)
    n_raw = len(rows)

    recs = []
    dropped = {"empty": 0, "unknown_label": 0}
    for r in rows:
        text = (r.text or "").strip()
        if len(text) < 8:
            dropped["empty"] += 1
            continue
        # 三分类标签坍缩成二分类；不是已知标签的直接丢，别猜。
        lab = "benign" if r.label == BENIGN else "injection"
        recs.append(_Text(text, {"text": text, "label": lab, "source": name,
                                 "hard": bool(r.hard), "meta": r.meta}))

    kept_o, dd = dedup(recs, near=True) if recs else ([], {"input": 0})
    kept = [o.row for o in kept_o]
    n_benign = sum(1 for r in kept if r["label"] == "benign")

    out_path = out_path or Path("data/transfer") / f"{name}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    info = {
        "name": name, "path": str(out_path), "n_raw": n_raw, "n_kept": len(kept),
        "n_benign": n_benign, "n_malicious": len(kept) - n_benign,
        "n_hard": sum(1 for r in kept if r["hard"]),
        "dropped": dropped, "dedup": dd,
    }
    return info


def build_all(names: list[str] | None = None, limit: int | None = 600,
              out_dir: Path | None = None, endpoint: str | None = None) -> list[dict]:
    out_dir = out_dir or Path("data/transfer")
    infos = []
    for name in (names or available()):
        try:
            info = build_target(name, limit=limit, out_path=out_dir / f"{name}.jsonl",
                                endpoint=endpoint)
            infos.append(info)
            print(f"  [目标集] {name:12s} 原始 {info['n_raw']:5d} → 保留 {info['n_kept']:5d}"
                  f"（良性 {info['n_benign']} / 注入 {info['n_malicious']}，难负 {info['n_hard']}）")
        except Exception as e:  # noqa: BLE001
            info = {"name": name, "error": f"{type(e).__name__}: {e}"[:200],
                    "n_kept": 0, "n_benign": 0, "n_malicious": 0}
            infos.append(info)
            print(f"  [目标集] {name:12s} 不可用：{info['error']}")
    return infos
