"""A4：文本空间攻击谱系 × 检测器鲁棒性矩阵。

矩阵的每一格是 ASR*（只算意图仍可读的攻击）。同时报三个数：
- `asr`：字面 ASR（不论文本是否被改坏）；
- `asr_valid`：ASR*，只看意图可读的攻击；
- `n_unknown`：判据给不出结论的样本数——这个数大，说明结论弱，必须写进论文。

这三个数放一起，读者能立刻判断"这个高 ASR 是真绕过还是把文本改烂了"。
只报最后一个数的论文，基本都在藏东西。
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

# 呈现顺序：先字符级，再语义，最后多轮。
ORDER = ["zero_width", "fullwidth", "homoglyph", "base64_wrap",
         "paraphrase-light", "paraphrase", "multi-turn"]


def run_a4(cfg: Config, data_dir: Path, detectors: list[str] | None = None,
           out_dir: Path | None = None) -> Path:
    detectors = list(detectors or DEFAULT_DETECTORS)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = out_dir or (Path(cfg.runs_dir) / f"{ts}_a4")
    out_dir.mkdir(parents=True, exist_ok=True)

    report: dict = {"kind": "a4", "data": str(data_dir),
                    "fpr_target": cfg.fpr_target, "results": []}
    for det in detectors:
        print(f"  [A4] {det} ...", flush=True)
        one = out_dir / f"_{det}.json"
        cmd = [sys.executable, "-m", "injlab._a4_one", "--detector", det,
               "--data", str(data_dir), "--out", str(one), "--fpr", str(cfg.fpr_target)]
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0 or not one.exists():
            err = (proc.stderr or proc.stdout or "").strip().splitlines()
            report["results"].append({"detector": det, "status": "failed",
                                      "error": err[-1][:300] if err else f"exit {proc.returncode}"})
            print(f"    ✗ {err[-1][:150] if err else 'failed'}")
            continue
        row = json.loads(one.read_text(encoding="utf-8"))
        row["wall_seconds"] = round(time.perf_counter() - t0, 1)
        report["results"].append(row)
        print("    ✓ " + " ".join(
            f"{k}={_fmt(v.get('asr_valid'))}" for k, v in row["attacks"].items()))

    (out_dir / "a4.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
    (out_dir / "report.md").write_text(render_markdown(report), encoding="utf-8")
    print(f"\n  结果: {out_dir}/report.md")
    return out_dir


def _fmt(v, nd: int = 3) -> str:
    if v is None:
        return "—"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if f != f:
        return "n/a"
    return f"{f:.{nd}f}"


def render_markdown(report: dict) -> str:
    ok = [r for r in report["results"] if r.get("status") == "ok"]
    L = ["### A4 攻击谱系 × 检测器（ASR\\*，只算意图仍可读的攻击）", ""]
    if not ok:
        return "\n".join(L + ["> 没有检测器跑成功。"])
    attacks = [a for a in ORDER if any(a in r["attacks"] for r in ok)]
    L.append("| 攻击 | " + " | ".join(r["detector"] for r in ok) + " | 族 |")
    L.append("|---|" + "---|" * len(ok) + "---|")
    for a in attacks:
        cells = []
        for r in ok:
            v = r["attacks"].get(a, {})
            cells.append(_fmt(v.get("asr_valid")))
        fam = next((r["attacks"][a].get("family", "") for r in ok if a in r["attacks"]), "")
        L.append(f"| {a} | " + " | ".join(cells) + f" | {fam} |")
    L.append("")

    L.append("#### 字面 ASR（不论文本是否被改坏）")
    L.append("")
    L.append("| 攻击 | " + " | ".join(r["detector"] for r in ok) + " |")
    L.append("|---|" + "---|" * len(ok))
    for a in attacks:
        L.append(f"| {a} | " + " | ".join(_fmt(r["attacks"].get(a, {}).get("asr")) for r in ok) + " |")
    L.append("")

    L.append("#### 判据给不出结论的样本数（n_unknown，越大结论越弱）")
    L.append("")
    L.append("| 攻击 | " + " | ".join(r["detector"] for r in ok) + " |")
    L.append("|---|" + "---|" * len(ok))
    for a in attacks:
        L.append(f"| {a} | " + " | ".join(str(r["attacks"].get(a, {}).get("n_unknown", "—")) for r in ok) + " |")
    L.append("")

    mt = [r for r in ok if "multi-turn" in r["attacks"]]
    if mt:
        L.append("#### 多轮：分轮扫描 vs 拼接打分")
        L.append("")
        L.append("| 检测器 | 分轮 ASR\\* | 拼接 ASR\\* | 分轮平均分 |")
        L.append("|---|---|---|---|")
        for r in mt:
            v = r["attacks"]["multi-turn"]
            L.append(f"| {r['detector']} | {_fmt(v.get('asr_valid'))} | "
                     f"{_fmt(v.get('asr_concat_valid'))} | {_fmt(v.get('mean_turn_score'))} |")
        L.append("")
        L.append("> 分轮 ASR\\* 高于拼接 ASR\\* ⇒ 只逐轮扫描会漏掉跨轮组合出来的攻击。")
    return "\n".join(L)
