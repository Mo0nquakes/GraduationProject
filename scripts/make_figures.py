"""生成开题报告/论文需要的三张图：架构图、用例图、ER 图。

为什么用代码画而不是拖 draw.io：
1. 三张图都在文档里有正文对应，改一个功能要同步改图——手画的图一定忘；
2. ER 图**从 schema.sql 的真实外键生成**，表一改图自动变，答辩被追问也不虚；
3. 输出 PNG（300dpi，贴 Word）+ PDF（矢量，贴 LaTeX），一次生成两版。

用法:
    python scripts/make_figures.py            # 输出到 docs/figs/
    python scripts/make_figures.py --only er  # 只画一张
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

INK = "#1a1a1a"
INK2 = "#4a4a4a"
LINE = "#c8ccd4"
BLUE = "#2f5d8a"
BLUE_L = "#e8eef5"
GREEN = "#3f6f50"
GREEN_L = "#eaf1ec"
AMBER = "#8a6a2f"
AMBER_L = "#f6f0e2"
GREY_L = "#f4f5f7"

CJK = ["PingFang SC", "Hiragino Sans GB", "Heiti SC", "Songti SC",
       "Arial Unicode MS", "Noto Sans CJK SC", "Microsoft YaHei"]


def setup_font() -> None:
    from matplotlib import font_manager

    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in CJK:
        if name in have:
            plt.rcParams["font.sans-serif"] = [name] + CJK
            plt.rcParams["axes.unicode_minus"] = False
            return
    print("[warn] 没找到中文字体，图里中文会变方框")


def box(ax, x, y, w, h, text, fc="#ffffff", ec=LINE, fs=9.5, bold=False, r=0.02):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                linewidth=0.9, edgecolor=ec, facecolor=fc, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=INK, zorder=3, fontweight="bold" if bold else "normal",
            linespacing=1.5)


def arrow(ax, p, q, color=INK2, style="-|>", lw=0.9, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=8,
                                 linewidth=lw, color=color, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=2, shrinkB=2, zorder=1))


# ---------------------------------------------------------------- 架构图
def fig_architecture(out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10.4, 5.4))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 52)
    ax.axis("off")

    ax.text(0.5, 50.2, "① 接入层", fontsize=10, color=BLUE, fontweight="bold")
    box(ax, 0.5, 43.4, 24, 5.9, "浏览器控制台\n（单页 HTML/CSS/JS）", BLUE_L, BLUE, 9.0)
    box(ax, 26.0, 43.4, 24, 5.9, "命令行 CLI\npython -m injlab …", BLUE_L, BLUE, 9.0)
    box(ax, 51.5, 43.4, 22, 5.9, "交互式 API\nOpenAPI /docs", BLUE_L, BLUE, 9.0)

    ax.text(0.5, 40.6, "② 服务层", fontsize=10, color=BLUE, fontweight="bold")
    box(ax, 0.5, 33.6, 73, 5.7,
        "FastAPI：/detect  /script  /attacks  /experiments  /audit/verify", "#ffffff", BLUE, 9.0)

    ax.text(0.5, 30.8, "③ 核心能力层", fontsize=10, color=GREEN, fontweight="bold")
    cores = [
        "检测器\n规则 / 词级 / 字符级\n神经模型",
        "攻击谱系\n零宽·全角·同形字\nBase64 / 截断",
        "红队回路\n生成 → 评测 → 并入\n→ 重训",
        "评测指标\nTPR@FPR / ASR\n意图保留判据",
        "数据管线\n清洗 / 去重 / 切分\n泄漏检查",
    ]
    for i, t in enumerate(cores):
        box(ax, 0.5 + i * 14.7, 23.2, 13.9, 6.4, t, GREEN_L, GREEN, 8.0)

    ax.text(0.5, 20.6, "④ 数据与存储层", fontsize=10, color=AMBER, fontweight="bold")
    box(ax, 0.5, 13.6, 23.5, 5.9, "SQLite\n8 表 + 哈希链审计", AMBER_L, AMBER, 9.0)
    box(ax, 25.2, 13.6, 23.5, 5.9, "runs/\n每次实验的产物", AMBER_L, AMBER, 9.0)
    box(ax, 49.9, 13.6, 23.5, 5.9, "data/\n语料 / 数据集 / 权重", AMBER_L, AMBER, 9.0)

    for x in (12.5, 38.0, 62.5):
        arrow(ax, (x, 43.3), (x, 39.5), BLUE)
    arrow(ax, (37.0, 33.5), (37.0, 29.9), GREEN)
    arrow(ax, (37.0, 23.1), (37.0, 19.8), AMBER)

    # 被保护系统（RAG 靶子）
    box(ax, 76.5, 33.6, 22.5, 15.7,
        "被保护系统\n（最小 RAG 靶子）\n\n检索 → 拼提示词 → 生成\n\n恶意文档从检索结果\n进入提示词 = 间接注入",
        "#ffffff", "#9aa3b0", 8.0)
    arrow(ax, (73.7, 36.4), (76.3, 36.4), "#9aa3b0", style="<|-|>")

    ax.text(0.5, 7.4,
            "设计约束：全部离线可跑；神经模型一个进程只加载一个（峰值内存 = 最大单模型）；所有实验产物落盘可复现。",
            fontsize=8.0, color=INK2)
    ax.text(0.5, 4.0,
            "信任边界：用户输入 / 检索到的外部文档 / 工具返回 / 模型输出 —— 后三者都可能携带指令。",
            fontsize=8.0, color=INK2)
    p = out / "fig_architecture.png"
    fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out / "fig_architecture.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p


# ---------------------------------------------------------------- 用例图
USE_CASES = [
    ("提交文本检测", "analyst"),
    ("批量检测与导出", "analyst"),
    ("查看逐块高亮", "analyst"),
    ("发起红队攻击", "redteam"),
    ("运行对抗训练", "engineer"),
    ("对比实验（A1–A7）", "engineer"),
    ("管理数据集版本", "engineer"),
    ("校验审计链", "auditor"),
]
ACTORS = [
    ("安全分析员", "analyst", "直接使用者：日常检测与研判"),
    ("红队工程师", "redteam", "构造攻击、评估绕过"),
    ("模型工程师", "engineer", "训练检测器、跑对比实验"),
    ("审计员", "auditor", "只读：验证日志未被篡改"),
]
ACTOR_X = {"analyst": 6.0, "redteam": 6.0, "engineer": 6.0, "auditor": 6.0}


def fig_usecase(out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    ax.set_xlim(-8, 100)
    ax.set_ylim(0, 64)
    ax.axis("off")

    # 系统边界
    nx, ny, nw, nh = 30.0, 1.5, 66.0, 58.0
    ax.add_patch(FancyBboxPatch((nx, ny), nw, nh, boxstyle="round,pad=0.4,rounding_size=0.8",
                                linewidth=1.0, edgecolor="#9aa3b0", facecolor="none", zorder=1))
    ax.text(nx + 1.6, ny + nh - 1.8, "检测系统 injlab", fontsize=10, color="#5a6270",
            fontweight="bold", zorder=2)

    ys = [54.0, 47.2, 40.4, 33.6, 26.8, 20.0, 13.2, 6.4]
    pos = {}
    for (label, actor), y in zip(USE_CASES, ys):
        cx, cy, w, h = nx + nw / 2 + 3, y, 40.0, 5.6
        ax.add_patch(Ellipse((cx, cy), w, h, linewidth=0.9, edgecolor=BLUE,
                             facecolor=BLUE_L, zorder=2))
        ax.text(cx, cy, label, ha="center", va="center", fontsize=9.2, color=INK, zorder=3)
        pos[label] = (cx - w / 2, cy)

    # 演员
    ax_y = [47.0, 34.0, 21.0, 8.0]
    for (name, key, _desc), y in zip(ACTORS, ax_y):
        ax.plot([3.0], [y + 1.2], marker="o", markersize=7, color=INK2)
        ax.plot([3.0, 3.0], [y + 1.0, y - 1.2], color=INK2, lw=1.1)
        ax.plot([1.6, 4.4], [y + 0.2, y + 0.2], color=INK2, lw=1.1)
        ax.plot([3.0, 1.8], [y - 1.2, y - 3.0], color=INK2, lw=1.1)
        ax.plot([3.0, 4.2], [y - 1.2, y - 3.0], color=INK2, lw=1.1)
        ax.text(3.0, y - 4.4, name, ha="center", va="top", fontsize=9.2, color=INK)
        ax.text(3.0, y - 7.6, _desc, ha="center", va="top", fontsize=7.2, color=INK2)

    for label, actor in USE_CASES:
        px, py_ = pos[label]
        src_y = dict(zip(("analyst", "redteam", "engineer", "auditor"), ax_y))[actor] + 1.0
        arrow(ax, (4.6, src_y), (px, py_), "#7d8796", style="-", lw=0.8)

    ax.text(nx, ny + nh + 1.6, "用例图：四类参与者与八项功能", fontsize=8.4, color=INK2)
    p = out / "fig_usecase.png"
    fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out / "fig_usecase.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p


# ---------------------------------------------------------------- ER 图
def _read_schema():
    from injlab.db.store import SCHEMA_PATH

    con = sqlite3.connect(":memory:")
    con.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    tables = [r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
    cols = {t: [(r[1], r[2]) for r in con.execute(f"PRAGMA table_info({t})")] for t in tables}
    fks = {}
    for t in tables:
        fl = [(r[3], r[2], r[4]) for r in con.execute(f"PRAGMA foreign_key_list({t})")]
        if fl:
            fks[t] = fl  # (from_col, to_table, to_col)
    return tables, cols, fks


def fig_er(out: Path) -> Path:
    tables, cols, fks = _read_schema()

    # 手工排位置（层次：数据集 → 样本/攻击 → 实验 → 日志），但字段与外键全自动
    layout = {
        "dataset_version": (2.0, 40.0),
        "sample": (24.0, 40.0),
        "attack": (2.0, 24.0),
        "attack_sample": (24.0, 24.0),
        "defense_policy": (2.0, 8.0),
        "model_version": (24.0, 8.0),
        "experiment_run": (46.0, 8.0),
        "detection_log": (68.0, 8.0),
    }
    W, LINE_H, PAD = 20.0, 1.55, 1.1
    fig, ax = plt.subplots(figsize=(11.4, 6.2))
    ax.set_xlim(0, 96)
    ax.set_ylim(0, 64)
    ax.axis("off")

    geo = {}
    for t, (x, y) in layout.items():
        cs = cols[t]
        h = PAD * 2 + LINE_H * (len(cs) + 1)
        ax.add_patch(FancyBboxPatch((x, y), W, h, boxstyle="round,pad=0.12,rounding_size=0.35",
                                    linewidth=0.9, edgecolor=LINE, facecolor="#ffffff", zorder=2))
        ax.add_patch(FancyBboxPatch((x, y + h - LINE_H * 1.5), W, LINE_H * 1.5,
                                    boxstyle="round,pad=0.12,rounding_size=0.35",
                                    linewidth=0.9, edgecolor=LINE, facecolor=BLUE_L, zorder=3))
        ax.text(x + W / 2, y + h - LINE_H * 0.75, t, ha="center", va="center",
                fontsize=8.6, color=BLUE, fontweight="bold", zorder=4)
        for i, (name, decl) in enumerate(cs):
            yy = y + h - LINE_H * (2.5 + i)
            pk = " (PK)" if decl and "PRIMARY KEY" in decl.upper() else ""
            ax.text(x + 0.9, yy, f"{name}{pk}", ha="left", va="center", fontsize=7.2,
                    color=INK if pk else INK2, zorder=4)
        geo[t] = (x, y, W, h)

    for src, fl in fks.items():
        sx, sy, sw, sh = geo[src]
        for from_col, dst, _to_col in fl:
            if dst not in geo:
                continue
            dx, dy, dw, dh = geo[dst]
            if src == dst:  # 自引用：右侧绕回
                p1 = (sx + sw, sy + sh * 0.32)
                p2 = (sx + sw + 5.0, sy + sh * 0.32)
                p3 = (sx + sw + 5.0, sy + sh * 0.72)
                p4 = (sx + sw, sy + sh * 0.72)
                ax.plot([p1[0], p2[0], p3[0], p4[0]], [p1[1], p2[1], p3[1], p4[1]],
                        color="#7d8796", lw=0.8, zorder=1)
                arrow(ax, (p2[0], p2[1]), (p3[0], p3[1]), "#7d8796", style="-|>", lw=0.8)
                ax.text(p2[0] + 2.6, (p2[1] + p3[1]) / 2, "上一条", fontsize=7.0,
                        color="#7d8796", ha="center", va="center")
                continue
            if abs(sx - dx) < 1:  # 同列，垂直
                a = (sx + sw * 0.5, sy) if sy > dy else (sx + sw * 0.5, sy + sh)
                b = (dx + dw * 0.5, dy + dh) if sy > dy else (dx + dw * 0.5, dy)
            else:
                a = (sx + sw, sy + sh * 0.5) if sx < dx else (sx, sy + sh * 0.5)
                b = (dx, dy + dh * 0.5) if sx < dx else (dx + dw, dy + dh * 0.5)
            arrow(ax, a, b, "#7d8796", style="-|>", lw=0.8)

    ax.text(2.0, 61.4, "实体关系图（由 schema.sql 的真实外键自动生成，改表即改图）",
            fontsize=8.4, color=INK2)
    p = out / "fig_er.png"
    fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out / "fig_er.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p


# ---------------------------------------------------------------- 技术路线
STAGES = [
    ("阶段一\n数据", "公开语料筛选\n清洗去重·难负样本\n三分类标注", "train/test.jsonl\nmanifest.json"),
    ("阶段二\n基线", "规则 + TF-IDF\n训练集校准阈值\n测试集报 TPR@FPR", "A1 基线表"),
    ("阶段三\n神经检测", "DistilBERT 微调\n分块滑窗推理\n字符区间回映", "A1 神经行\n高亮接口"),
    ("阶段四\n攻防", "文本空间攻击谱系\n鲁棒性矩阵\n红队回路对抗训练", "A3–A6 ASR 曲线"),
    ("阶段五\n系统", "三级流水线服务化\nRAG 靶子端到端\n审计链与控制台", "A7 + 演示系统"),
]


def fig_roadmap(out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(11.0, 3.1))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 30)
    ax.axis("off")

    w, gap = 17.4, 2.9
    for i, (title, what, product) in enumerate(STAGES):
        x = 0.8 + i * (w + gap)
        box(ax, x, 12.2, w, 12.2, title + "\n\n" + what, "#ffffff", BLUE, 7.6)
        # 产出
        box(ax, x, 4.4, w, 6.2, product, BLUE_L, LINE, 7.2)
        arrow(ax, (x + w / 2, 12.1), (x + w / 2, 10.7), "#9aa3b0", lw=0.8)
        if i < len(STAGES) - 1:
            arrow(ax, (x + w + 0.3, 18.3), (x + w + gap - 0.3, 18.3), BLUE, lw=1.1)

    ax.text(0.8, 27.2, "技术路线：五个阶段与其产出（阶段之间是硬依赖，不是并列的活儿）",
            fontsize=8.6, color=INK2)
    ax.text(0.8, 1.2, "贯穿约束：每一阶段的产物都带版本号入库，任何一条结论都能回答“用哪版数据、哪个模型、什么阈值”。",
            fontsize=7.8, color=INK2)
    p = out / "fig_roadmap.png"
    fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out / "fig_roadmap.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/figs")
    ap.add_argument("--only", choices=["roadmap", "arch", "usecase", "er"], default=None)
    args = ap.parse_args()

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    setup_font()

    todo = {"roadmap": fig_roadmap, "arch": fig_architecture,
            "usecase": fig_usecase, "er": fig_er}
    if args.only:
        todo = {args.only: todo[args.only]}
    for key, fn in todo.items():
        p = fn(out)
        print(f"  {key:8s} → {p.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
