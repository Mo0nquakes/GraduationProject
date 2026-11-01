"""清理 runs/ 里的玩具运行（demo_*），只留正式实验。

为什么需要它：`python -m injlab demo` 每跑一次就建一个 `runs/<ts>_demo_<detector>/`，
自检脚本里还有好几处调用，几天下来 runs/ 里 40 个目录有 30 多个是 demo。
这些数字是 91 条玩具数据上的，**一个都不能进论文**，混在实验列表里既误导人，
也让"这次实验用了哪份数据"变得不可追溯。

默认 dry-run，加 --apply 才真删。

关于数据库：脚本**不动** `experiment_run` 与 `detection_log` 表。原因不是懒——
`detection_log` 通过 `experiment_run_id` 外键挂在这张表上，删父行要么报错、
要么得级联删掉审计记录，而那条哈希链正是本系统要证明的东西。正确做法是
"文件清掉、审计留档"，而不是为了目录干净去动链。
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

KEEP = ("_a1", "_at", "_rag", "_trunc")


def classify(name: str) -> str:
    return "toy" if "_demo_" in name else "real"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--apply", action="store_true", help="真的删，不加只打印")
    args = ap.parse_args()

    root = Path(args.runs)
    if not root.is_dir():
        raise SystemExit(f"没有这个目录: {root}")

    toy = sorted(p for p in root.iterdir() if p.is_dir() and classify(p.name) == "toy")
    real = sorted(p for p in root.iterdir() if p.is_dir() and classify(p.name) == "real")
    size = sum(sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) for p in toy)

    print(f"[runs] 正式实验 {len(real)} 个，玩具运行 {len(toy)} 个（合计 {size/1024:.0f} KB）")
    if real:
        print("       保留: " + ", ".join(p.name for p in real[-6:]))
    if not toy:
        print("[runs] 没有需要清理的目录")
        return 0

    for p in toy:
        if args.apply:
            shutil.rmtree(p)
        print(("  已删除 " if args.apply else "  将删除 ") + p.name)
    if not args.apply:
        print("\n[dry-run] 加 --apply 才会真的删")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
