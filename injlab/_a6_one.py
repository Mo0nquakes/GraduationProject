"""A6 单个检测器 / 单个消融档位的评测（独立进程，理由同 `_a1_one`）。

两个档位共用**同一个测试集**和**同一个阈值来源**：

- 测试集 = 源域测试集 + 难负样本测试集（有/无难负样本训练，考的是同一张卷子）；
- 阈值 = 只在**源域训练集里的干净良性**上标定，两档一模一样。

第二条是实验成立的关键。如果 `with` 档用"训练集全部良性（含难负）"标阈值，
它的阈值会被难负样本推高，于是误报率自动变好看，而这个改善跟模型学没学会
区分毫无关系——那只是在比较两个不同的工作点。**消融实验里唯一允许变化的
是训练集，别的一律钉死。**
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def _read(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--mode", choices=["without", "with", "stripped"], required=True)
    ap.add_argument("--hard-train", default=None)
    ap.add_argument("--stripped-train", default=None,
                    help="控制组：难负样本去掉线索词后的版本")
    ap.add_argument("--hard-test", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fpr", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    try:
        import torch

        torch.set_num_threads(2)
    except Exception:  # noqa: BLE001
        pass

    from . import metrics
    from .config import load as load_config
    from .eval.a1 import _binary_y, _build, load_split

    cfg = load_config()
    cfg.fpr_target = args.fpr

    train, test = load_split(Path(args.data))
    hard_train = _read(Path(args.hard_train)) if args.hard_train else []
    hard_test = _read(Path(args.hard_test)) if args.hard_test else []

    if args.mode == "with":
        train = train + hard_train
    elif args.mode == "stripped":
        train = train + (_read(Path(args.stripped_train)) if args.stripped_train else [])
    test = test + hard_test

    tr_texts = [r["text"] for r in train]
    # 注意：fit() 收的是**标签字符串**（"benign"/"direct_injection"/...），不是 0/1。
    # 这里一度传了 `_binary_y` 的整数，结果所有样本都被当成非良性、训练集只有一类，
    # sklearn 直接报 "needs samples of at least 2 classes"。二值化只用于算指标。
    tr_labels = [r["label"] for r in train]
    # 阈值来源：**只取源域训练集里的干净良性**（不含难负），两档完全一致。
    clean_train_benign = [r["text"] for r in train if r["label"] == "benign" and not r.get("hard")]

    det = _build(args.detector, seed=args.seed)
    t0 = time.perf_counter()
    det.fit(tr_texts, tr_labels)
    fit_seconds = time.perf_counter() - t0

    cal_scores = [float(s) for s in det.predict_proba(clean_train_benign)]
    thr = metrics.threshold_at_fpr([0] * len(cal_scores), cal_scores, cfg.fpr_target)

    te_texts = [r["text"] for r in test]
    t1 = time.perf_counter()
    scores = [float(s) for s in det.predict_proba(te_texts)]
    latency_ms = (time.perf_counter() - t1) * 1000 / max(1, len(te_texts))

    y = _binary_y(test)
    is_hard = [bool(r.get("hard")) for r in test]
    clean_neg = [i for i in range(len(test)) if y[i] == 0 and not is_hard[i]]
    hard_neg = [i for i in range(len(test)) if y[i] == 0 and is_hard[i]]
    pos = [i for i in range(len(test)) if y[i] == 1]

    def _rate(idx: list[int]) -> float:
        if not idx:
            return float("nan")
        return sum(1 for i in idx if scores[i] > thr) / len(idx)

    legacy = {
        "tpr": metrics.tpr_at_fpr(y, scores, cfg.fpr_target)["tpr"],
        "tpr_at_fpr_10": metrics.tpr_at_fpr(y, scores, 0.10)["tpr"],
        "auc": metrics.auc(y, scores),
    }
    y_clean = [0] * len(clean_neg) + [1] * len(pos)
    s_clean = [scores[i] for i in clean_neg + pos]
    clean_legacy = {
        "tpr": metrics.tpr_at_fpr(y_clean, s_clean, cfg.fpr_target)["tpr"],
        "auc": metrics.auc(y_clean, s_clean),
    }

    row = {
        "detector": args.detector,
        "mode": args.mode,
        "status": "ok",
        "seed": args.seed,
        "threshold": thr,
        "n_train": len(train),
        "n_hard_train": len(hard_train),
        "n_test": len(test),
        "n_pos_test": len(pos),
        "n_clean_neg_test": len(clean_neg),
        "n_hard_neg_test": len(hard_neg),
        "tpr_at_fpr": legacy["tpr"],
        "tpr_at_fpr_10": legacy["tpr_at_fpr_10"],
        "auc": legacy["auc"],
        "auc_clean": clean_legacy["auc"],
        "tpr_clean": clean_legacy["tpr"],
        "fpr_clean": _rate(clean_neg),
        "fpr_hard": _rate(hard_neg),
        "fpr_all": _rate(clean_neg + hard_neg),
        "fit_seconds": round(fit_seconds, 1),
        "latency_ms": round(latency_ms, 3),
    }
    Path(args.out).write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"    {args.mode:<7} TPR@1%={row['tpr_at_fpr']:.3f} FPR_clean={row['fpr_clean']:.3f} "
          f"FPR_hard={row['fpr_hard']:.3f} AUC={row['auc']:.3f} ({row['fit_seconds']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
