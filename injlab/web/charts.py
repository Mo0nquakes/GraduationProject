"""把 runs 里的实验产物画成图，存成 PNG 给前端 <img> 用。

为什么用 matplotlib 预生成而不是前端画：评委席上的浏览器不一定开得动几个
图表库，而 PNG 是死文件，断网、换机器、塞进论文都一样。代价是交互性——所以
图只负责"看一眼趋势"，精确数字仍然从 JSON 渲染成表格。
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 必须在 pyplot 之前，否则会尝试连 Cocoa 窗口
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

# 中文字体：macOS 上这几个基本都有；没有就退回默认值不报错。
for _name in ("PingFang SC", "Heiti SC", "Songti SC", "Arial Unicode MS", "Noto Sans CJK SC"):
    if any(f.name == _name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [_name, "DejaVu Sans"]
        break
plt.rcParams["axes.unicode_minus"] = False

INK = "#1f2933"
GRID = "#d8dee6"
BLUE = "#2f5d8a"
RED = "#b91c1c"
GREEN = "#15803d"


def _save(fig, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return out


def latest_a1(runs_dir: str | Path = "runs") -> dict | None:
    """挑**覆盖检测器最多**的那次 A1，而不是最近的那次。

    因为 `injlab a1 --detectors distilbert` 这种补跑也会生成 *_a1 目录，
    按时间取最新会把单检测器的图放到"全基线对比"上——图上只剩一根柱子，
    看起来像别的检测器全挂了。同数量时取最新。
    """
    best, best_key = None, (-1, "")
    for f in Path(runs_dir).glob("*_a1/a1.json"):
        try:
            rep = json.loads(f.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        n = sum(1 for r in rep.get("results", []) if r.get("status") == "ok")
        key = (n, f.parent.name)
        if key > best_key:
            best, best_key = rep, key
    return best


def chart_a1(report: dict, out: Path) -> Path:
    """各检测器在 FPR=1% 与 10% 两个操作点上的召回。"""
    rows = [r for r in report["results"] if r["status"] == "ok"]
    names = [r["detector"] for r in rows]
    t1 = [r["tpr_at_fpr"] for r in rows]
    t10 = [r["tpr_at_fpr_10"] for r in rows]

    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    x = range(len(names))
    w = 0.36
    ax.bar([i - w / 2 for i in x], t1, w, label="TPR @ FPR=1%", color=BLUE)
    ax.bar([i + w / 2 for i in x], t10, w, label="TPR @ FPR=10%", color="#9db8d1")
    for i, (a, b) in enumerate(zip(t1, t10)):
        ax.text(i - w / 2, a + 0.02, f"{a:.2f}", ha="center", fontsize=8, color=INK)
        ax.text(i + w / 2, b + 0.02, f"{b:.2f}", ha="center", fontsize=8, color="#5b6b7c")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("恶意样本召回率", fontsize=9, color=INK)
    ax.set_title(f"A1 全基线对比（负样本 N={rows[0]['n_benign_test'] if rows else 0}）",
                 fontsize=10, color=INK)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=8, frameon=False)
    return _save(fig, out)


def chart_attack(report: dict, out: Path) -> Path:
    """攻击 × 检测器的 ASR*，用热力网格表现"谁被谁打穿"。"""
    rows = [r for r in report["results"] if r["status"] == "ok"]
    if not rows:
        return out
    atks = list(rows[0]["attacks"].keys())
    names = [r["detector"] for r in rows]
    grid = [[r["attacks"][a]["asr_valid"] for r in rows] for a in atks]

    fig, ax = plt.subplots(figsize=(1.5 + 0.95 * len(names), 0.55 * len(atks) + 1.5))
    for j, a in enumerate(atks):
        for i, _n in enumerate(names):
            v = grid[j][i]
            na = v != v
            ax.add_patch(plt.Rectangle((i, len(atks) - 1 - j), 1, 1,
                                       facecolor="#eef2f7" if na else
                                       plt.cm.Reds(0.08 + 0.72 * (v if not na else 0)),
                                       edgecolor="white", lw=1.4))
            ax.text(i + 0.5, len(atks) - 0.5 - j, "n/a" if na else f"{v:.2f}",
                    ha="center", va="center", fontsize=8,
                    color="#8894a3" if na else (INK if (na or v < 0.55) else "white"))
    ax.set_xlim(0, len(names))
    ax.set_ylim(0, len(atks))
    ax.set_xticks([i + 0.5 for i in range(len(names))])
    ax.set_xticklabels(names, fontsize=9)
    ax.set_yticks([len(atks) - 0.5 - j for j in range(len(atks))])
    ax.set_yticklabels(atks, fontsize=9)
    ax.set_title("攻击谱系 ASR*（数字越大＝越容易被绕过）", fontsize=10, color=INK)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    return _save(fig, out)


def chart_truncation(scores: dict, out: Path) -> Path:
    """截断漏检 vs 分块滑动窗口检出。scores = {"naive": 0.5, "chunked": 1.0}"""
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    labels = ["朴素截断 512", "分块滑窗 256/128"]
    vals = [scores["naive"], scores["chunked"]]
    cols = [RED if v <= 0.5 else GREEN for v in vals]
    ax.barh(labels, vals, color=cols, height=0.5)
    for i, v in enumerate(vals):
        ax.text(v + 0.02, i, f"{v:.2f}", va="center", fontsize=9, color=INK)
    ax.axvline(0.5, color="#8894a3", ls="--", lw=1)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("文档被判为恶意的分数（阈值 0.5）", fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return _save(fig, out)


def build_all(runs_dir: str | Path = "runs", out_dir: str | Path = "data/charts") -> dict[str, str]:
    out = Path(out_dir)
    made: dict[str, str] = {}
    rep = latest_a1(runs_dir)
    if rep:
        made["a1"] = str(chart_a1(rep, out / "a1_tpr.png"))
        made["attack"] = str(chart_attack(rep, out / "attack_asr.png"))
    return made
