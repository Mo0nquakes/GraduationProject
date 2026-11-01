"""单个检测器的 A1 评测，**独立进程**跑。

为什么要拆进程：目标机器是 8GB 的 MacBook Air。guardrail（180–700MB 权重）
和 DistilBERT 如果装在同一个 Python 进程里，即使 `del` + `empty_cache()`，
MPS 的缓存与 PyTorch 的分配器也未必把内存还给系统，实测会 OOM 甚至让整机卡死。
一个模型一个进程，退出即回收，峰值永远只有"最大的那个模型"。

由 `injlab.eval.a1.run_a1` 以子进程方式调用，不直接面向用户。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fpr", type=float, default=0.01)
    args = ap.parse_args()

    # 限线程：机器上还有别的活，别把 CPU 抢光
    try:
        import torch

        torch.set_num_threads(2)
    except Exception:  # noqa: BLE001
        pass

    from . import metrics
    from .config import load as load_config
    from .eval import intent
    from .eval.a1 import _build, _binary_y, load_split

    cfg = load_config()
    cfg.fpr_target = args.fpr
    train, test = load_split(Path(args.data))
    tr_texts = [r["text"] for r in train]
    te_texts = [r["text"] for r in test]
    tr_y, te_y = _binary_y(train), _binary_y(test)

    t0 = time.perf_counter()
    row: dict = {"detector": args.detector, "status": "ok"}
    det = _build(args.detector)
    det.fit(tr_texts, [r["label"] for r in train])
    tr_scores = det.predict_proba(tr_texts)
    thr = metrics.threshold_at_fpr(tr_y, tr_scores, cfg.fpr_target)
    thr = max(0.0, min(1.0, float(thr)))

    ta = time.perf_counter()
    te_scores = det.predict_proba(te_texts)
    lat = (time.perf_counter() - ta) * 1000.0 / max(1, len(te_texts))
    pred = [1 if s > thr else 0 for s in te_scores]
    prf = metrics.prf(te_y, pred)
    fpr = (sum(1 for s, y in zip(te_scores, te_y) if y == 0 and s > thr)
           / max(1, sum(1 for y in te_y if y == 0)))

    from .attacks.obfuscation import ATTACKS

    per_attack: dict[str, dict] = {}
    for name, atk in ATTACKS.items():
        mixed, orig = [], {}
        for r in test:
            if r["label"] == "benign":
                mixed.append(r["text"])
            else:
                orig[len(mixed)] = r["text"]
                mixed.append(atk.apply(r["text"]))
        sc = det.predict_proba(mixed)
        mal_idx = [i for i, r in enumerate(test) if r["label"] != "benign"]
        mal = [sc[i] for i in mal_idx]
        valid = [intent.intent_preserved(orig[i], mixed[i]) for i in mal_idx]
        kept = [(ok, s) for (ok, _w), s in zip(valid, mal) if ok is True]
        per_attack[name] = {
            "asr": metrics.asr(mal, thr),
            "asr_valid": (sum(1 for _o, s in kept if s <= thr) / len(kept)) if kept else float("nan"),
            "n_valid": len(kept),
            "n_unknown": sum(1 for ok, _w in valid if ok is None),
            "n_mal": len(mal),
        }

    row.update(
        threshold=thr,
        tpr_at_fpr=metrics.tpr_at_fpr(te_y, te_scores, cfg.fpr_target)["tpr"],
        tpr_at_fpr_10=metrics.tpr_at_fpr(te_y, te_scores, 0.10)["tpr"],
        auc=metrics.auc(te_y, te_scores), fpr=fpr,
        precision=prf["precision"], recall=prf["recall"], f1=prf["f1"],
        latency_ms=lat, n_test=len(test),
        n_benign_test=sum(1 for y in te_y if y == 0),
        attacks=per_attack, fit_seconds=round(time.perf_counter() - t0, 1),
    )
    Path(args.out).write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
