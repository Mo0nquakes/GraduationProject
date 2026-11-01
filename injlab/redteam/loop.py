"""A5：红队回路对抗训练（主体实验）。

一句话：**让攻击者不断出招，把成功的招数并进训练集，看防御能不能跟上，
以及能不能挡住没见过的招数。**

回路三步：
1. 用当前检测器给攻击变体打分，挑出"绕过成功且意图仍可读"的样本；
2. 把这些样本（连同原标签）并进训练集，重训；
3. 在**留出的攻击族**上评估——这一族从头到尾不参与训练。

第 3 步是整个实验成立的前提。不留出攻击族，第 2 步的"提升"只是把答案
背下来了；加上留出族，才能回答评审的第 3 条意见"训练后能否应对未见攻击"。

## 与 A3 的对照

A3 是在**嵌入空间**做扰动（现实里不存在的攻击者），A5 是在**文本空间**做改写
（真实攻击者）。同一台机器的同一份数据，两条路线放在一张图上比——这就是论文
的核心图。
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

from .. import metrics
from ..attacks.semantic import LightParaphraseAttack, ParaphraseAttack
from ..config import Config
from ..eval import intent
from ..labels import DIRECT, INDIRECT

# 参与训练的"招数"与留出评估的"招数"。留出的那一族只出现在最后一列。
TRAIN_ATTACKS = ("paraphrase-light", "fullwidth", "zero_width")
HELDOUT_ATTACKS = ("paraphrase", "homoglyph")

MAX_AUG_PER_ROUND = 200


def _load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def _attacks():
    from ..attacks.obfuscation import ATTACKS

    table = {"paraphrase-light": LightParaphraseAttack(),
             "paraphrase": ParaphraseAttack(seed=7)}
    for k in ("fullwidth", "zero_width", "homoglyph"):
        table[k] = ATTACKS[k]
    return table


def _evaluate(det, rows: list[dict], thr: float, attacks: dict, names: tuple[str, ...]) -> dict:
    """在给定攻击族上算 ASR*（只算意图可读的攻击）。"""
    mal = [r for r in rows if r["label"] != "benign"]
    out = {}
    for name in names:
        atk = attacks[name]
        mut = [atk.apply(r["text"]) for r in mal]
        sc = det.predict_proba(mut)
        kept = []
        for r, m, s in zip(mal, mut, sc):
            ok, _why = intent.check(getattr(atk, "family", "obfuscation"), r["text"], m)
            if ok is True:
                kept.append((m, s))
        asr = (sum(1 for _m, s in kept if s <= thr) / len(kept)) if kept else float("nan")
        out[name] = {"asr_valid": asr, "n_valid": len(kept), "n_mal": len(mal)}
    return out


def run_loop(cfg: Config, data_dir: Path, rounds: int = 3,
             out_dir: Path | None = None, max_aug: int = MAX_AUG_PER_ROUND,
             detector: str = "distilbert") -> Path:
    from ..eval.a1 import _binary_y, _build, load_split

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = out_dir or (Path(cfg.runs_dir) / f"{ts}_a5")
    out_dir.mkdir(parents=True, exist_ok=True)

    train, test = load_split(data_dir)
    attacks = _attacks()

    report: dict = {"kind": "a5", "data": str(data_dir), "rounds": rounds,
                    "detector": detector,
                    "train_attacks": list(TRAIN_ATTACKS),
                    "heldout_attacks": list(HELDOUT_ATTACKS), "history": []}

    work_train = list(train)
    # 第 0 轮 = 干净基线，作为对照
    for rnd in range(rounds + 1):
        tag = "clean" if rnd == 0 else f"rt{rnd}"
        print(f"\n  [A5] 第 {rnd} 轮（{tag}）：训练集 {len(work_train)} 条", flush=True)
        t0 = time.perf_counter()

        det = _build(detector)
        if detector.startswith("distilbert"):
            # 回路会不断往训练集里加样本，尺寸比 A1 大；8GB 内存上必须调小
            det.batch_size = 4
            det.max_len = 192
        det.name = f"{detector}-{tag}"
        det.fit([r["text"] for r in work_train], [r["label"] for r in work_train],
                fpr_target=cfg.fpr_target)

        tr_scores = det.predict_proba([r["text"] for r in work_train])
        thr = float(metrics.threshold_at_fpr(_binary_y(work_train), tr_scores, cfg.fpr_target))

        # 回路训练的代价必须一起报：只报 ASR 降了、不报干净精度掉了，是另一种藏东西
        te_y = _binary_y(test)
        te_scores = det.predict_proba([r["text"] for r in test])
        clean_tpr = metrics.tpr_at_fpr(te_y, te_scores, cfg.fpr_target)["tpr"]
        neg = [s for y, s in zip(te_y, te_scores) if y == 0]
        fpr = sum(1 for s in neg if s > thr) / max(1, len(neg))

        seen = _evaluate(det, test, thr, attacks, TRAIN_ATTACKS)
        held = _evaluate(det, test, thr, attacks, HELDOUT_ATTACKS)
        row = {"round": rnd, "tag": tag, "n_train": len(work_train), "threshold": thr,
               "clean_tpr_at_fpr": clean_tpr, "fpr": fpr,
               "train_family": seen, "heldout_family": held,
               "seconds": round(time.perf_counter() - t0, 1)}
        report["history"].append(row)
        print("    " + "  ".join(f"{k}={v['asr_valid']:.3f}" for k, v in seen.items())
              + " | heldout "
              + "  ".join(f"{k}={v['asr_valid']:.3f}" for k, v in held.items()), flush=True)

        if rnd == rounds:
            break

        # ---- 回路：挑本轮绕过成功的变体，并进训练集
        aug: list[dict] = []
        mal = [r for r in train if r["label"] != "benign"]
        for name in TRAIN_ATTACKS:
            atk = attacks[name]
            mut = [atk.apply(r["text"]) for r in mal]
            sc = det.predict_proba(mut)
            for r, m, s in zip(mal, mut, sc):
                if s > thr:
                    continue  # 没绕过，不算"成功招数"
                ok, _why = intent.check(getattr(atk, "family", "obfuscation"), r["text"], m)
                if ok is not True:
                    continue  # 意图不可读的样本会把模型教坏
                aug.append({"text": m, "label": r["label"], "source": f"rt:{name}",
                            "origin": r["text"]})
        # 去重（按文本），再限制每轮增量
        seen_txt: set[str] = set()
        uniq = []
        for a in aug:
            if a["text"] in seen_txt:
                continue
            seen_txt.add(a["text"])
            uniq.append(a)
        uniq = uniq[:max_aug]
        report.setdefault("augmentation", []).append(
            {"round": rnd + 1, "n_candidates": len(aug), "n_added": len(uniq)})
        print(f"    回路：绕过成功 {len(aug)} 条 → 去重后并入 {len(uniq)} 条", flush=True)
        work_train = work_train + uniq

    (out_dir / "a5.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
    (out_dir / "report.md").write_text(render_markdown(report), encoding="utf-8")
    print(f"\n  结果: {out_dir}/report.md")
    return out_dir


def render_markdown(report: dict) -> str:
    L = ["### A5 红队回路对抗训练（ASR\\* 越低越好）", ""]
    L.append(f"- 训练用攻击族：{', '.join(report['train_attacks'])}")
    L.append(f"- **留出攻击族（全程不参与训练）**：{', '.join(report['heldout_attacks'])}")
    L.append("")
    L.append("| 轮次 | 训练集 | 干净 TPR@1% | FPR | " + " | ".join(report["train_attacks"])
             + " | " + " | ".join(report["heldout_attacks"]) + " |")
    L.append("|---|---|---|---|" + "---|" * (len(report["train_attacks"]) + len(report["heldout_attacks"])))
    for h in report["history"]:
        cells = [f"{h['train_family'][a]['asr_valid']:.3f}" for a in report["train_attacks"]]
        cells += [f"{h['heldout_family'][a]['asr_valid']:.3f}" for a in report["heldout_attacks"]]
        ct, fp = h.get("clean_tpr_at_fpr"), h.get("fpr")
        ct = "n/a" if ct is None or ct != ct else f"{ct:.3f}"
        fp = "n/a" if fp is None or fp != fp else f"{fp:.3f}"
        L.append(f"| {h['tag']} | {h['n_train']} | {ct} | {fp} | "
                 + " | ".join(cells) + " |")
    L.append("")
    L.append("> 左半边（训练见过的族）下降是应该的；**右半边（没见过的族）也下降，"
             "才说明学到的是防御而不是背答案**。")
    aug = report.get("augmentation") or []
    if aug:
        L.append("")
        L.append("每轮并入的对抗样本：" + "，".join(
            f"第 {a['round']} 轮 {a['n_added']}/{a['n_candidates']}" for a in aug))
    return "\n".join(L)
