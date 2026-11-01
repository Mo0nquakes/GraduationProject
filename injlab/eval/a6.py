"""A6：难负样本消融 —— 教检测器"别看到线索词就报警"，代价是什么？

研究问题
--------
提示注入检测器有一个极廉价的捷径：**看到 `ignore previous instructions` 就报警**。
在当前语料上这个捷径几乎不受惩罚，因为正常文本里没人写这句话。

但安全产品每天要处理"引用/讨论/测试"这类文本：员工转发钓鱼邮件、文档解释提示
注入原理、测试用例里写着攻击串。**这些全是良性的，而且全躲不掉。**

A6 问两件事：
1. 训练集里加入难负样本，检测器还会不会抓真注入？（代价）
2. 误报率降了多少？降的是"干净良性"还是"难良性"？（收益归谁）

这个实验的价值在于它是**唯一一个能同时看到"检测率"和"误报率"代价**的实验。
A1 只看检测率，A2 只看跨域，都不足以回答"这个模型能不能上线"。

实验设计（对照组只有一个变量）
------------------------------
| 档位 | 训练集 | 测试集 | 阈值来源 |
|---|---|---|---|
| without | 源域训练集 | 源域测试集 + 难负测试 | 源域训练集**干净**良性 |
| with | 源域训练集 + 难负训练 | 同上（**同一张卷子**） | 同上（**同一个阈值来源**） |

阈值必须钉死，否则 `with` 档的阈值会被难负样本推高，误报率"自动"变好，
而这个改善跟模型学没学会区分毫无关系。

难负样本是 constructed 的（见 `data/hardneg.py` 里为什么不用采集的），
局限必须写进论文：外部效度弱于真实采集，只覆盖五类语用。
"""
from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from ..config import Config
from ..data import hardneg
from ..data.prep import group_split

DEFAULT_DETECTORS = ("keyword", "tfidf-word", "tfidf-char", "distilbert")
MODES = ("without", "with", "stripped")
LABEL = {"without": "无难负样本", "with": "含难负样本", "stripped": "对照组：去线索词"}


def run_a6(cfg: Config, data_dir: Path, detectors: Sequence[str] = DEFAULT_DETECTORS,
           out_dir: Path | None = None, seeds: Sequence[int] | None = None,
           per_family: int = 24, hard_test_ratio: float = 0.5) -> Path:
    detectors = list(detectors or DEFAULT_DETECTORS)
    seeds = list(seeds) if seeds else [None]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = out_dir or (Path(cfg.runs_dir) / f"{ts}_a6")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 难负样本只造一次，落盘留证，训练/测试按模板分组切开（同一模板不跨侧）
    hard = hardneg.build(n_per_family=per_family, seed=cfg.seed)
    hard_tr, hard_te = group_split(hard, test_ratio=hard_test_ratio, seed=cfg.seed)
    # 控制组：同样的句子、同样的长度，只把线索词换成中性说法。
    stripped_tr = hardneg.strip_cues(hard_tr)
    for name, rows in (("hardneg_train", hard_tr), ("hardneg_test", hard_te),
                       ("stripped_train", stripped_tr)):
        (out_dir / f"{name}.jsonl").write_text(
            "\n".join(json.dumps(dataclasses.asdict(r), ensure_ascii=False) for r in rows) + "\n",
            encoding="utf-8")

    print(f"  [A6] 难负样本 {len(hard)} 条（训练 {len(hard_tr)} / 测试 {len(hard_te)}）"
          f" 语用分布 {hardneg.stats(hard)['by_family']}")

    report: dict[str, Any] = {
        "kind": "a6",
        "data": str(data_dir),
        "fpr_target": cfg.fpr_target,
        "hardneg": hardneg.stats(hard),
        "n_hard_train": len(hard_tr),
        "n_hard_test": len(hard_te),
        "results": [],
    }

    for det in detectors:
        for mode in MODES:
            print(f"  [A6] {det} / {LABEL[mode]} ...", flush=True)
            one = out_dir / f"_{det}__{mode}.json"
            cmd = [sys.executable, "-m", "injlab._a6_one",
                   "--detector", det, "--data", str(data_dir), "--mode", mode,
                   "--hard-train", str(out_dir / "hardneg_train.jsonl"),
                   "--stripped-train", str(out_dir / "stripped_train.jsonl"),
                   "--hard-test", str(out_dir / "hardneg_test.jsonl"),
                   "--out", str(one), "--fpr", str(cfg.fpr_target)]
            if seeds[0] is not None:
                cmd += ["--seed", str(seeds[0])]
            t0 = time.perf_counter()
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0 or not one.exists():
                err = (proc.stderr or proc.stdout or "").strip().splitlines()
                report["results"].append({
                    "detector": det, "mode": mode, "status": "failed",
                    "error": (err[-1] if err else f"exit {proc.returncode}")[:300],
                })
                print(f"    ✗ {err[-1][:150] if err else 'failed'}")
                continue
            row = json.loads(one.read_text(encoding="utf-8"))
            row["wall_seconds"] = round(time.perf_counter() - t0, 1)
            report["results"].append(row)

    (out_dir / "a6.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
    (out_dir / "report.md").write_text(render_markdown(report), encoding="utf-8")
    print(f"\n  结果: {out_dir}/report.md")
    return out_dir


def _rows_of(report: dict, det: str) -> dict[str, dict]:
    return {r["mode"]: r for r in report["results"]
            if r["detector"] == det and r.get("status") == "ok"}


def render_markdown(report: dict) -> str:
    ok = [r for r in report["results"] if r.get("status") == "ok"]
    dets = []
    for r in ok:
        if r["detector"] not in dets:
            dets.append(r["detector"])

    L: list[str] = ["### A6 难负样本消融\n"]
    L.append(f"- 数据：`{report['data']}`；阈值统一取自**源域训练集干净良性**，两档完全相同")
    L.append(f"- 难负样本：{report['hardneg']['n']} 条，训练 {report['n_hard_train']} / 测试 "
             f"{report['n_hard_test']}；语用分布 {report['hardneg']['by_family']}")
    L.append(f"- 其中含字面线索词的：{report['hardneg']['with_literal_cue']} 条\n")

    if ok:
        r0 = ok[0]
        L.append(f"- 测试集构成：正样本 {r0['n_pos_test']}，干净良性 {r0['n_clean_neg_test']}，"
                 f"难良性 {r0['n_hard_neg_test']}\n")

    L.append("| 检测器 | 训练集 | TPR@1% | FPR(干净良性) | FPR(难良性) | AUC | 阈值 |")
    L.append("|---|---|---|---|---|---|---|")
    for det in dets:
        got = _rows_of(report, det)
        for mode in MODES:
            row = got.get(mode)
            if row is None:
                L.append(f"| {det} | {LABEL[mode]} | — | — | — | — | 失败 |")
                continue
            L.append(f"| {det} | {LABEL[mode]} | {row['tpr_at_fpr']:.3f} | "
                     f"{row['fpr_clean']:.3f} | {row['fpr_hard']:.3f} | "
                     f"{row['auc']:.3f} | {row['threshold']:.3f} |")

    L.append("\n**三臂对照**\n")
    L.append("三臂里只有「加不加难负样本」该变；对照组用的是一模一样的句子，"
             "只把线索词换成中性说法——**它是用来回答「会不会只是数据变多了」的**。\n")
    L.append("| 检测器 | 对比 | ΔTPR@1% | ΔFPR(难良性) | ΔAUC |")
    L.append("|---|---|---|---|---|")
    for det in dets:
        got = _rows_of(report, det)
        base = got.get("without")
        if not base:
            continue

        def delta(row: dict, a: str) -> str:
            v = row[a] - base[a]
            if v != v:  # nan
                return "n/a"
            return f"{v:+.3f}"

        for arm, name in (("with", "含难负 − 无"), ("stripped", "对照组 − 无")):
            row = got.get(arm)
            if not row:
                continue
            L.append(f"| {det} | {name} | {delta(row, 'tpr_at_fpr')} | "
                     f"{delta(row, 'fpr_hard')} | {delta(row, 'auc')} |")
        wi, st = got.get("with"), got.get("stripped")
        if wi and st:
            L.append(f"| {det} | **含难负 − 对照组（真正的难效应）** | "
                     f"{wi['tpr_at_fpr'] - st['tpr_at_fpr']:+.3f} | "
                     f"{wi['fpr_hard'] - st['fpr_hard']:+.3f} | "
                     f"{wi['auc'] - st['auc']:+.3f} |")

    L.append("\n> ΔFPR(难良性) 为负 = 难负样本起了作用；ΔTPR@1% 为负 = 代价。"
             "两项一起看才有意义：只降误报不谈检测率，等于把报警器关小声。")
    L.append("> 难负样本为**构造**（五类语用 × 句式），外部效度弱于真实采集，论文须注明。\n")
    return "\n".join(L)
