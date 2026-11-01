"""把同一检测器多个种子的 A1 结果汇总成「均值 ± 标准差」。

为什么非要这个：单次运行的 0.865 和换一版权重的 0.892，评审一句"这是种子差异还是
方法差异"就答不上来。神经网络的初始化、dropout、数据顺序都会动小数点后第二位，
不报方差等于把噪声当结论。经验法则：**不报方差的对比，等于没做对比**。

输出:
    docs/seeds.md        人能读的表
    runs/seeds.json      机器读的原始值（画误差棒/写论文都能用）
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

METRICS = [
    ("tpr_at_fpr", "TPR@FPR=1%", 3),
    ("tpr_at_fpr_10", "TPR@FPR=10%", 3),
    ("auc", "AUC", 3),
    ("fpr", "FPR", 3),
    ("f1", "F1", 3),
    ("latency_ms", "延迟 (ms)", 2),
]


def load_runs(patterns: list[str], only_seeded: bool = True) -> list[dict]:
    out = []
    for pat in patterns:
        for f in sorted(ROOT.glob(pat)):
            rep = json.loads(Path(f).read_text(encoding="utf-8"))
            rows = [r for r in rep.get("results", []) if r.get("status") == "ok"]
            if only_seeded:
                # 只汇总"明确标了种子"的运行。早期那些没记种子的跑动（有的还是在
                # fixture 玩具集上训的）混进来会把标准差算成一个假的大数。
                rows = [r for r in rows if r.get("seed") is not None]
                if not rows:
                    continue
            out.append({"run": Path(f).parent.name, "rows": rows,
                        "seeds": sorted({r.get("seed") for r in rows})})
    return out


def summarize(runs: list[dict]) -> dict[str, dict]:
    by_det: dict[str, dict[str, list[float]]] = {}
    for r in runs:
        for row in r["rows"]:
            d = row["detector"]
            bucket = by_det.setdefault(d, {k: [] for k, _, _ in METRICS})
            for key, _, _ in METRICS:
                v = row.get(key)
                if isinstance(v, (int, float)):
                    bucket[key].append(float(v))
    out = {}
    for d, bucket in by_det.items():
        out[d] = {}
        for key, _, nd in METRICS:
            vals = bucket[key]
            if not vals:
                continue
            out[d][key] = {
                "n": len(vals),
                "mean": st.fmean(vals),
                "std": st.stdev(vals) if len(vals) > 1 else 0.0,
                "min": min(vals),
                "max": max(vals),
                "raw": [round(v, 4) for v in vals],
                "nd": nd,
            }
    return out


def render(agg: dict[str, dict], runs: list[dict]) -> str:
    seeds = sorted({s for r in runs for s in r["seeds"]}) or ["—"]
    lines = [
        "# 多随机种子汇总",
        "",
        f"- 种子：{', '.join(str(s) for s in seeds)}",
        f"- 参与汇总的运行：{len(runs)} 次（`{', '.join(r['run'] for r in runs)}`）",
        "- 口径：同一份数据集、同一套阈值规则（训练集良性分位数，FPR 目标 1%）、同一批攻击。",
        "- 标准差不写的时候是**只有一次运行**，不是方差为零。",
        "",
        "| 检测器 | TPR@FPR=1% | TPR@FPR=10% | AUC | FPR | F1 | 延迟 (ms) | n |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for d, m in agg.items():
        cells = []
        for key, _, nd in METRICS:
            v = m.get(key)
            if not v:
                cells.append("—")
                continue
            if v["n"] > 1:
                cells.append(f"{v['mean']:.{nd}f} ± {v['std']:.{nd}f}")
            else:
                cells.append(f"{v['mean']:.{nd}f}")
        lines.append(f"| {d} | " + " | ".join(cells) + f" | {m.get('tpr_at_fpr', {}).get('n', 0)} |")

    multi = {d: m for d, m in agg.items() if m.get("tpr_at_fpr", {}).get("n", 0) > 1}
    if multi:
        lines += ["", "## 单次运行的原始值", ""]
        for d, m in multi.items():
            for key, label, nd in METRICS:
                if key in ("latency_ms",):
                    continue
                raw = "、".join(f"{v:.{nd}f}" for v in m[key]["raw"])
                lines.append(f"- **{d}** {label}：{raw}")
    lines += [
        "",
        "> 写论文时的用法：正文报「均值 ± 标准差」，表格脚注写明「n=3 个随机种子」；",
        "> 如果两次运行的标准差大于要论证的效应量，那这个效应量就不成立。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=["runs/*_a1/a1.json"])
    ap.add_argument("--out", default="docs/seeds.md")
    ap.add_argument("--json", default="runs/seeds.json")
    ap.add_argument("--all", action="store_true",
                    help="连没记种子的历史运行一起汇总（一般不要：会把玩具集的结果混进来）")
    args = ap.parse_args()

    runs = load_runs(args.runs, only_seeded=not args.all)
    if not runs:
        raise SystemExit("没找到任何 a1.json，先跑 python -m injlab a1 --seed N")
    agg = summarize(runs)
    (ROOT / args.out).write_text(render(agg, runs), encoding="utf-8")
    (ROOT / args.json).write_text(json.dumps(agg, ensure_ascii=False, indent=2), encoding="utf-8")
    print(render(agg, runs))
    print(f"\n→ {args.out}\n→ {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
