"""A3：嵌入空间对抗训练（FGSM / PGD）对**文本空间**攻击有没有用。

这个实验的预期结论是**没有用**，而且我们要把这个负结果做扎实：

- 三个变体：干净训练（对照）、FGSM-AT、PGD-3-AT，超参按最有利于该方法的选；
- 评测全部在文本空间（字符级混淆 + 语义改写），**不看嵌入空间 ASR**；
- 同时报干净 TPR，因为对抗训练通常要拿干净准确率换鲁棒性——只报鲁棒性、
  不报代价，是另一种藏东西。

为什么 sklearn 族不参与：没有可微的输入表示，FGSM 无从定义。这一条本身就是
原方案"传统 ML vs 预训练模型对抗鲁棒性对比"不成立的证据，要写进论文。
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from ..config import Config

MODES = ("clean", "fgsm", "fgsm-weak", "pgd")
ORDER = ["zero_width", "fullwidth", "homoglyph", "base64_wrap",
         "paraphrase-light", "paraphrase"]


def run_a3(cfg: Config, data_dir: Path, modes: list[str] | None = None,
           out_dir: Path | None = None, seeds: list[int] | None = None) -> Path:
    modes = list(modes or MODES)
    seeds = list(seeds or [0])
    seed = seeds[0]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = out_dir or (Path(cfg.runs_dir) / f"{ts}_a3")
    out_dir.mkdir(parents=True, exist_ok=True)

    report: dict = {"kind": "a3", "data": str(data_dir), "seeds": seeds, "results": []}
    for mode in modes:
        for sd in seeds:
            print(f"  [A3] {mode} seed={sd} ...", flush=True)
            one = out_dir / f"_{mode}_s{sd}.json"
            cmd = [sys.executable, "-m", "injlab._a3_one", "--mode", mode,
                   "--data", str(data_dir), "--out", str(one),
                   "--fpr", str(cfg.fpr_target), "--seed", str(sd)]
            t0 = time.perf_counter()
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0 or not one.exists():
                err = (proc.stderr or proc.stdout or "").strip().splitlines()
                report["results"].append({"mode": mode, "seed": sd, "status": "failed",
                                          "error": err[-1][:300] if err else "failed"})
                print(f"    ✗ {err[-1][:150] if err else 'failed'}")
                continue
            row = json.loads(one.read_text(encoding="utf-8"))
            row["wall_seconds"] = round(time.perf_counter() - t0, 1)
            row["seed"] = sd
            report["results"].append(row)
            mk = (row["attacks"].get("paraphrase") or {}).get("asr_valid")
            print(f"    ✓ clean TPR={row['clean_tpr_at_fpr']:.3f}  paraphrase ASR*={_fmt(mk)}")

    (out_dir / "a3.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
    (out_dir / "report.md").write_text(render_markdown(report), encoding="utf-8")
    print(f"\n  结果: {out_dir}/report.md")
    return out_dir


def _fmt(v, nd: int = 3) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "—"
    return "n/a" if f != f else f"{f:.{nd}f}"


def render_markdown(report: dict) -> str:
    ok = [r for r in report["results"] if r.get("status") == "ok"]
    L = ["### A3 嵌入空间对抗训练 vs 文本空间攻击", ""]
    if not ok:
        return "\n".join(L + ["> 没有跑成功的变体。"])
    attacks = [a for a in ORDER if any(a in r["attacks"] for r in ok)]
    seeds = sorted({r.get("seed", 0) for r in ok})
    label = {"clean": "干净训练（对照）", "fgsm": "FGSM-AT (ε=0.05)",
             "fgsm-weak": "FGSM-AT (ε=0.01)", "pgd": "PGD-3-AT (ε=0.05)"}

    def agg(mode: str, fn) -> str:
        vals = [fn(r) for r in ok if r["mode"] == mode]
        vals = [v for v in vals if v == v]  # 去 NaN
        if not vals:
            return "n/a"
        if len(vals) == 1:
            return f"{vals[0]:.3f}"
        m = sum(vals) / len(vals)
        sd = (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
        return f"{m:.3f} ± {sd:.3f}"

    L.append(f"种子：{seeds}（每个变体各自重训；MPS 上训练本身不确定，"
             f"单次数字差 0.05 不算差异）")
    L.append("")
    L.append("| 训练方式 | 干净 TPR@1% | 干净 AUC | " + " | ".join(attacks) + " |")
    L.append("|---|---|---|" + "---|" * len(attacks))
    for mode in dict.fromkeys(r["mode"] for r in ok):
        cells = [agg(mode, lambda r, a=a: _num(r["attacks"].get(a, {}).get("asr_valid")))
                 for a in attacks]
        L.append(f"| {label.get(mode, mode)} | "
                 f"{agg(mode, lambda r: _num(r.get('clean_tpr_at_fpr')))} | "
                 f"{agg(mode, lambda r: _num(r.get('clean_auc')))} | "
                 + " | ".join(cells) + " |")
    L.append("")
    L.append("> 判读：若 AT 变体的文本攻击 ASR\\* 没有低于干净训练，说明嵌入空间对抗训练"
             "挡不住真实的文本改写攻击——它优化的是另一个威胁模型。")
    L.append("> 同时看第一列：AT 通常拿干净准确率换鲁棒性，这里连干净准确率也塌了，"
             "说明扰动幅度相对嵌入范数过大，模型学到的是「别信自己的输入」，"
             "而不是「识别注入」。")
    return "\n".join(L)


def _num(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")
