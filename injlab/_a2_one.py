"""A2 单个检测器的跨集评测（独立进程，理由同 `_a1_one`）。

一次加载模型，把**所有**目标集打分完再退出——跨集评测里 4 个目标集 × 5 个检测器
如果按对拆进程，DistilBERT 要重复加载 4 次，白白多花两分钟。

评测口径（写论文时必须一起报，缺一个都会被问住）：
1. `tpr_fixed`：阈值**只在源域训练集良性分数**上标定，然后原封不动搬到目标集。
   这是真实部署的样子——你不可能先拿到目标领域的良性流量再定阈值。
2. `tpr_oracle`：阈值在**目标集自己的良性分数**上重标定。这是"如果允许针对这个
   领域调阈值，上限在哪"。两者差距 = 阈值迁移损失，是 A2 最有信息量的一个数。
3. `fdr`：目标集良性被判恶意的比例。跨集最大的失败模式不是漏检，是**误报爆炸**
   （域外良性文本长得不像训练时的良性），所以这一列必须显式报。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def _read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", required=True)
    ap.add_argument("--data", required=True, help="源域数据目录（含 train.jsonl）")
    ap.add_argument("--target", action="append", default=[],
                    help="name=path.jsonl，可重复")
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
    from .eval import intent
    from .eval.a1 import _binary_y, _build, load_split

    cfg = load_config()
    cfg.fpr_target = args.fpr

    train, _test = load_split(Path(args.data))
    tr_texts = [r["text"] for r in train]
    tr_y = _binary_y(train)

    row: dict = {"detector": args.detector, "status": "ok", "seed": args.seed,
                 "targets": {}}
    t0 = time.perf_counter()
    det = _build(args.detector, seed=args.seed)
    # 神经检测器直接用已落盘的权重：跨集评测不需要重训（重训既慢，又会把
    # A1 那份权重覆盖掉，导致两次实验用的其实是不同模型）。
    reused = False
    if args.detector.startswith("distilbert"):
        try:
            reused = bool(det.load_or_none())
        except Exception:  # noqa: BLE001
            reused = False
    if not reused:
        det.fit(tr_texts, [r["label"] for r in train])
    tr_scores = det.predict_proba(tr_texts)
    thr_fixed = float(metrics.threshold_at_fpr(tr_y, tr_scores, cfg.fpr_target))
    thr_fixed = max(0.0, min(1.0, thr_fixed))

    for spec in args.target:
        if "=" not in spec:
            raise SystemExit(f"--target 需要 name=path 形式，收到: {spec}")
        name, path = spec.split("=", 1)
        rows = _read_jsonl(Path(path))
        if not rows:
            row["targets"][name] = {"status": "empty"}
            continue
        texts = [r["text"] for r in rows]
        y = [0 if r["label"] == "benign" else 1 for r in rows]

        det.predict_proba(texts[:2])  # 预热，别把冷启动算进 latency
        ta = time.perf_counter()
        sc = det.predict_proba(texts)
        lat = (time.perf_counter() - ta) * 1000.0 / max(1, len(texts))

        thr_oracle = float(metrics.threshold_at_fpr(y, sc, cfg.fpr_target))
        thr_oracle = max(0.0, min(1.0, thr_oracle))
        n_benign = sum(1 for v in y if v == 0)

        def _fdr(thr: float) -> float:
            return (sum(1 for s, v in zip(sc, y) if v == 0 and s > thr)
                    / max(1, n_benign))

        pred = [1 if s > thr_fixed else 0 for s in sc]
        prf = metrics.prf(y, pred)

        # 攻击视角：目标集里被漏掉的恶意样本，按意图可读性再切一刀
        miss_valid = 0
        miss_n = 0
        for r, s in zip(rows, sc):
            if r["label"] == "benign":
                continue
            miss_n += 1
            ok, _why = intent.intent_preserved(r["text"], r["text"])
            if ok is not False and s <= thr_fixed:
                miss_valid += 1

        row["targets"][name] = {
            "status": "ok",
            "n": len(rows), "n_benign": n_benign, "n_malicious": len(rows) - n_benign,
            "n_hard": sum(1 for r in rows if r.get("hard")),
            "tpr_fixed": metrics.tpr_at_fpr(y, sc, cfg.fpr_target)["tpr"],
            "fdr_fixed": _fdr(thr_fixed),
            "thr_fixed": thr_fixed,
            "tpr_oracle": metrics.tpr_at_fpr(y, sc, cfg.fpr_target)["tpr"]
            if abs(thr_oracle - thr_fixed) < 1e-9 else
            (sum(1 for s, v in zip(sc, y) if v == 1 and s > thr_oracle) / max(1, len(rows) - n_benign)),
            "thr_oracle": thr_oracle,
            "fdr_oracle": _fdr(thr_oracle),
            "auc": metrics.auc(y, sc),
            "f1_fixed": prf["f1"], "precision_fixed": prf["precision"],
            "recall_fixed": prf["recall"],
            "missed_malicious": miss_valid,
            "latency_ms": lat,
        }

    row["fit_seconds"] = round(time.perf_counter() - t0, 1)
    Path(args.out).write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
