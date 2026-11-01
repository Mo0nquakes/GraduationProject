"""A3 单个变体的训练与评测（独立进程，避免多个 256MB 模型同时驻留）。

一个变体 = 一种训练方式（clean / fgsm-at / pgd-at）。评测一律在**文本空间**：
嵌入空间的"攻击成功率"没有对应的真实攻击者，拿它当主指标等于自证。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=["clean", "fgsm", "fgsm-weak", "pgd"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fpr", type=float, default=0.01)
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    try:
        import torch

        torch.set_num_threads(2)
    except Exception:  # noqa: BLE001
        pass

    from . import metrics
    from .attacks.obfuscation import ATTACKS
    from .attacks.semantic import LightParaphraseAttack, ParaphraseAttack
    from .eval import intent
    from .eval.a1 import _binary_y, load_split

    train, test = load_split(Path(args.data))
    tr_texts = [r["text"] for r in train]
    tr_y = _binary_y(train)
    te_y = _binary_y(test)

    t0 = time.perf_counter()
    if args.mode == "clean":
        from .models.distilbert import DistilBertDetector

        det = DistilBertDetector(seed=args.seed)
        det.name = "distilbert-clean"
    else:
        from .models.adversarial import AdversarialDistilBert

        # "fgsm-weak" 是同一方法的弱扰动版：审稿人一定会问"你是不是只是 eps 太大
        # 把模型打坏了"，所以必须有一个温和设置来回应。
        adv_mode = "fgsm" if args.mode.startswith("fgsm") else args.mode
        eps = 0.01 if args.mode == "fgsm-weak" else args.eps
        det = AdversarialDistilBert(seed=args.seed, adv_mode=adv_mode,
                                    eps=eps, steps=args.steps, mix=0.5)
    # A3 只是对照实验，产物不落盘，免得覆盖 A1 的干净模型
    det.save = lambda: None
    det.fit(tr_texts, [r["label"] for r in train], fpr_target=args.fpr)
    train_s = round(time.perf_counter() - t0, 1)

    tr_scores = det.predict_proba(tr_texts)
    thr = float(metrics.threshold_at_fpr(tr_y, tr_scores, args.fpr))
    te_scores = det.predict_proba([r["text"] for r in test])
    clean = metrics.tpr_at_fpr(te_y, te_scores, args.fpr)["tpr"]

    attacks = {**ATTACKS,
               "paraphrase-light": LightParaphraseAttack(),
               "paraphrase": ParaphraseAttack(seed=7)}
    attacks.pop("identity", None)
    mal = [r for r in test if r["label"] != "benign"]

    per: dict[str, dict] = {}
    for name, atk in attacks.items():
        mut = [atk.apply(r["text"]) for r in mal]
        sc = det.predict_proba(mut)
        kept = []
        for r, m, s in zip(mal, mut, sc):
            ok, _why = intent.check(getattr(atk, "family", "obfuscation"), r["text"], m)
            if ok is True:
                kept.append(s)
        per[name] = {
            "asr_valid": (sum(1 for s in kept if s <= thr) / len(kept)) if kept else float("nan"),
            "asr": metrics.asr(sc, thr),
            "n_valid": len(kept),
        }

    row = {"mode": args.mode, "status": "ok", "seed": args.seed,
           "eps": eps if args.mode != "clean" else 0.0,
           "steps": args.steps,
           "threshold": thr, "clean_tpr_at_fpr": clean,
           "clean_auc": metrics.auc(te_y, te_scores),
           "train_seconds": train_s, "attacks": per}
    Path(args.out).write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
