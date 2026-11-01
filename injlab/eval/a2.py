"""A2：跨集泛化（源域训练 → 目标域零样本评测）。

设计上的三条硬规矩，和 A1 一样是为了"可比性"，但内容不同：

1. **源域只用一份**：所有检测器都在同一份 `data/processed/<ts>_deepset+synth` 上训练。
   换源域就不是 A2 而是 A0 了。
2. **阈值只标定一次**：在源域训练集的良性分数上定，之后原封不动搬到每个目标集。
   目标集只用来打分，绝不参与任何 fit/threshold 计算——否则跨集泛化是自欺。
3. **误报与漏检分开报**：跨集最典型的失败是域外良性文本被大量误判（FDR 爆炸），
   只报 TPR 会把这件事整个藏起来。

`oracle` 那一列是"如果允许在目标域重标阈值"的上限，明确标注不可部署，只用来
度量阈值迁移损失。
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from ..config import Config

DEFAULT_DETECTORS = ("keyword", "tfidf-word", "tfidf-char", "guardrail", "distilbert")


def _fmt(v, nd: int = 3) -> str:
    if v is None:
        return "—"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if f != f:  # NaN
        return "n/a"
    return f"{f:.{nd}f}"


def run_a2(cfg: Config, data_dir: Path, targets: dict[str, Path],
           detectors: list[str] | None = None, out_dir: Path | None = None,
           seed: int | None = None) -> Path:
    detectors = list(detectors or DEFAULT_DETECTORS)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = out_dir or (Path(cfg.runs_dir) / f"{ts}_a2")
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = {k: Path(v) for k, v in targets.items() if Path(v).exists()}
    if not targets:
        raise SystemExit("没有任何目标集可用：先跑 `python -m injlab build-targets`")

    report: dict = {
        "kind": "a2", "source_domain": str(data_dir),
        "targets": {k: str(v) for k, v in targets.items()},
        "fpr_target": cfg.fpr_target, "seed": seed, "results": [],
    }

    for det in detectors:
        print(f"  [A2] {det} ...", flush=True)
        one = out_dir / f"_{det}.json"
        cmd = [sys.executable, "-m", "injlab._a2_one",
               "--detector", det, "--data", str(data_dir),
               "--out", str(one), "--fpr", str(cfg.fpr_target)]
        for name, path in targets.items():
            cmd += ["--target", f"{name}={path}"]
        if seed is not None:
            cmd += ["--seed", str(seed)]
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0 or not one.exists():
            err = (proc.stderr or proc.stdout or "").strip().splitlines()
            report["results"].append({
                "detector": det, "status": "failed",
                "error": err[-1][:300] if err else f"exit {proc.returncode}",
            })
            print(f"    ✗ 失败：{err[-1][:160] if err else 'unknown'}")
        else:
            row = json.loads(one.read_text(encoding="utf-8"))
            row["wall_seconds"] = round(time.perf_counter() - t0, 1)
            report["results"].append(row)
            t = row.get("targets", {})
            tprs = " ".join(f"{k}={_fmt(v.get('tpr_fixed'))}" for k, v in t.items())
            print(f"    ✓ {tprs}")

    (out_dir / "a2.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "report.md").write_text(render_markdown(report), encoding="utf-8")
    print(f"\n  结果: {out_dir}/report.md")
    return out_dir


def render_markdown(report: dict) -> str:
    ok = [r for r in report["results"] if r.get("status") == "ok"]
    names = list(report["targets"].keys())
    L = ["### A2 跨集泛化（源域训练 → 目标域零样本）", ""]
    L.append(f"- 源域：`{report['source_domain']}`")
    L.append(f"- 阈值：源域训练集良性分位，FPR 目标 {report['fpr_target']:.1%}（**目标集不参与任何标定**）")
    L.append(f"- 目标集：{', '.join(f'`{n}`' for n in names)}")
    L.append("")

    if not ok:
        L.append("> 没有任何检测器跑成功。")
        return "\n".join(L)

    L.append("#### TPR（固定阈值，部署口径）")
    L.append("")
    head = "| 检测器 | " + " | ".join(names) + " |"
    L.append(head)
    L.append("|" + "---|" * (len(names) + 1))
    for r in ok:
        cells = [_fmt(r["targets"].get(n, {}).get("tpr_fixed")) for n in names]
        L.append(f"| {r['detector']} | " + " | ".join(cells) + " |")
    L.append("")

    L.append("#### 误报率 FDR（固定阈值）—— 跨集的主要失败模式")
    L.append("")
    L.append(head)
    L.append("|" + "---|" * (len(names) + 1))
    for r in ok:
        cells = [_fmt(r["targets"].get(n, {}).get("fdr_fixed")) for n in names]
        L.append(f"| {r['detector']} | " + " | ".join(cells) + " |")
    L.append("")

    L.append("#### oracle 上限（在目标域重标阈值，**不可部署**，仅度量阈值迁移损失）")
    L.append("")
    L.append(head)
    L.append("|" + "---|" * (len(names) + 1))
    for r in ok:
        cells = [_fmt(r["targets"].get(n, {}).get("tpr_oracle")) for n in names]
        L.append(f"| {r['detector']} | " + " | ".join(cells) + " |")
    L.append("")

    L.append("#### 目标集构成")
    L.append("")
    L.append("| 目标集 | 总数 | 良性 | 注入 | 难负样本 |")
    L.append("|---|---|---|---|---|")
    for n in names:
        v = next((r["targets"].get(n, {}) for r in ok if n in r["targets"]), {})
        L.append(f"| {n} | {v.get('n', '—')} | {v.get('n_benign', '—')} | "
                 f"{v.get('n_malicious', '—')} | {v.get('n_hard', '—')} |")
    L.append("")

    failed = [r for r in report["results"] if r.get("status") != "ok"]
    if failed:
        L.append("> 未跑成功：" + "；".join(f"{r['detector']}（{r.get('error', '')[:80]}）"
                                        for r in failed))
    return "\n".join(L)
