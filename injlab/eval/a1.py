"""A1：全基线对比（同一份数据、同一套攻击、同一套阈值口径）。

这个模块存在的意义是把"可比性"固化成代码。手工跑五次 demo 再拼表，一定会
在某次忘了改阈值口径或忘了同一份攻击，最后表里的数字互相不可比而自己不知道。

三条硬规矩：
1. **同一份 train/test**：从 `data/processed/<ts>_<src>/` 读，不重切。
2. **同一套攻击**：遍历同一个 `ATTACKS`，不按检测器挑攻击。
3. **同一个阈值口径**：所有检测器都用 `metrics.threshold_at_fpr(train_benign, 1%)`，
   包括现成 guardrail。理由是它只用良性样本分数定阈值，不碰正样本标签，
   对"开箱即用"模型不构成额外照顾。

任何检测器装配失败（缺依赖、模型下不下来）都不会中断整张表，只在结果里记
`skipped`——一张残缺的表也远好过跑不出来。
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from .. import metrics
from ..config import Config
from ..eval import intent
from ..labels import LABELS

DEFAULT_DETECTORS = ("keyword", "tfidf-word", "tfidf-char", "guardrail", "distilbert")


def load_split(dataset_dir: Path) -> tuple[list[dict], list[dict]]:
    train = [json.loads(l) for l in (dataset_dir / "train.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    test = [json.loads(l) for l in (dataset_dir / "test.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    return train, test


def _binary_y(rows: Sequence[dict]) -> list[int]:
    return [0 if r["label"] == "benign" else 1 for r in rows]


def _build(kind: str):
    """按名字装配检测器；失败时抛出，由调用方记成 skipped。"""
    from ..models.stub import KeywordDetector

    if kind == "keyword":
        return KeywordDetector()
    if kind in ("tfidf-word", "tfidf-char"):
        from ..models.tfidf_lr import TfidfLogReg

        return TfidfLogReg(analyzer="word" if kind == "tfidf-word" else "char_wb")
    if kind in ("guardrail", "guardrail-base"):
        from ..models.guardrail import GuardrailDetector

        # small 版是 gated 拿不到，两个名字都指向 base 版
        return GuardrailDetector(verbose=False)
    if kind in ("distilbert", "distilbert-chunk"):
        from ..models.distilbert import DistilBertDetector

        return DistilBertDetector(chunked=(kind == "distilbert-chunk"), verbose=False)
    raise ValueError(f"未知检测器: {kind}")


def run_a1(cfg: Config, dataset_dir: Path, detectors: Sequence[str] = DEFAULT_DETECTORS,
           out_dir: Path | None = None, attacks=None) -> dict[str, Any]:
    from ..attacks.obfuscation import ATTACKS

    attacks = attacks or list(ATTACKS.values())
    train, test = load_split(dataset_dir)
    tr_texts = [r["text"] for r in train]
    te_texts = [r["text"] for r in test]
    tr_y, te_y = _binary_y(train), _binary_y(test)

    out_dir = out_dir or Path(cfg.runs_dir) / f"{datetime.now():%Y%m%d_%H%M%S}_a1"
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for kind in detectors:
        # 一个检测器一个子进程：退出即把内存还给系统。
        # 8GB 机器上这是唯一能保证"峰值 = 最大那个模型"的做法。
        tmp = out_dir / f".{kind}.json"
        cmd = [sys.executable, "-m", "injlab._a1_one",
               "--detector", kind, "--data", str(dataset_dir),
               "--out", str(tmp), "--fpr", str(cfg.fpr_target)]
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0 or not tmp.exists():
            err = (proc.stderr or proc.stdout or "").strip().splitlines()
            row = {"detector": kind, "status": "skipped",
                   "error": (err[-1] if err else "子进程失败")[:200]}
            results.append(row)
            print(f"  [跳过] {kind}: {row['error']}")
            continue
        row = json.loads(tmp.read_text(encoding="utf-8"))
        tmp.unlink(missing_ok=True)
        row["wall_seconds"] = round(time.perf_counter() - t0, 1)
        results.append(row)
        print(f"  [完成] {kind:<22} TPR@1%={row['tpr_at_fpr']:.3f} "
              f"AUC={row['auc']:.3f} FPR={row['fpr']:.3f} "
              f"延迟={row['latency_ms']:.2f}ms  ({row['wall_seconds']}s)")

    report = {
        "dataset": str(dataset_dir), "n_train": len(train), "n_test": len(test),
        "n_benign_train": sum(1 for y in tr_y if y == 0),
        "fpr_target": cfg.fpr_target,
        "label_counts": {lab: sum(1 for r in test if r["label"] == lab) for lab in LABELS},
        "results": results,
    }
    (out_dir / "a1.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "report.md").write_text(render_markdown(report), encoding="utf-8")
    print(f"\n[out] {out_dir}/report.md")
    return report


def render_markdown(report: dict[str, Any]) -> str:
    """生成论文可用的两张表（markdown，粘到 Word 前先看一眼）。"""
    ok = [r for r in report["results"] if r["status"] == "ok"]
    L: list[str] = []
    L.append(f"### A1 全基线对比\n")
    L.append(f"- 数据集：`{report['dataset']}`  train={report['n_train']} / test={report['n_test']}"
             f"（良性 {report['n_benign_train']} 训练 / {ok[0]['n_benign_test'] if ok else 0} 测试）")
    L.append(f"- 阈值口径：训练集良性分数分位数，FPR 目标 {report['fpr_target']:.1%}")
    L.append(f"- 测试集标签分布：{report['label_counts']}\n")
    L.append("| 检测器 | TPR@FPR=1% | TPR@FPR=10% | AUC | FPR | P | R | F1 | 延迟(ms) |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for r in ok:
        L.append(f"| {r['detector']} | {r['tpr_at_fpr']:.3f} | {r['tpr_at_fpr_10']:.3f} | "
                 f"{r['auc']:.3f} | {r['fpr']:.3f} | {r['precision']:.3f} | "
                 f"{r['recall']:.3f} | {r['f1']:.3f} | {r['latency_ms']:.2f} |")
    skipped = [r for r in report["results"] if r["status"] != "ok"]
    for r in skipped:
        L.append(f"| {r['detector']} | — | — | — | — | — | — | — | 装配失败：{r.get('error', '')} |")

    if ok:
        atk_names = list(ok[0]["attacks"].keys())
        L.append("\n### A4 攻击谱系 × 检测器（ASR*，只算意图仍可读的攻击）\n")
        L.append("| 攻击 | " + " | ".join(r["detector"] for r in ok) + " |")
        L.append("|---" * (len(ok) + 1) + "|")
        for a in atk_names:
            cells = []
            for r in ok:
                v = r["attacks"][a]["asr_valid"]
                cells.append("n/a" if v != v else f"{v:.3f}")
            L.append(f"| {a} | " + " | ".join(cells) + " |")
        L.append("\n> ASR* 为空（n/a）表示该攻击下没有任何一条样本能通过意图保留判据。")
        L.append("> 全量 ASR 见 `a1.json`，两者差距大说明该攻击主要靠把文本改烂。")
    return "\n".join(L) + "\n"
