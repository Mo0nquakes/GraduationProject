"""A4 单个检测器的攻击谱系评测（独立进程）。

一条样本要同时记录三件事，少一件结论就不成立：
1. `raw`：攻击后检测器分数是否低于阈值（**字面意义上的绕过**）；
2. `valid`：这段文本攻击意图是否还读得出来（`eval.intent.check`，三态）；
3. `asr_valid`：只看 valid==True 的那部分算出的 ASR。

评测里最容易被做假的就是第 2 步：把"文本改烂了"算成"攻击成功了"。
本模块把所有 n_unknown 都报出来，宁可样本变少也不假装有效。

多轮单独处理：分轮打分（检测器只看到单轮）与拼接打分（能看全上下文）分开报，
两者之差就是"分轮扫描"这个设计的漏洞大小。
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
    ap.add_argument("--out", required=True)
    ap.add_argument("--fpr", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    try:
        import torch

        torch.set_num_threads(2)
    except Exception:  # noqa: BLE001
        pass

    from . import metrics
    from .attacks.obfuscation import ATTACKS
    from .attacks.semantic import SEMANTIC_ATTACKS, MultiTurnAttack
    from .config import load as load_config
    from .eval import intent
    from .eval.a1 import _binary_y, _build, load_split

    cfg = load_config()
    cfg.fpr_target = args.fpr
    train, test = load_split(Path(args.data))
    tr_texts = [r["text"] for r in train]
    tr_y = _binary_y(train)

    det = _build(args.detector, seed=None)
    det.fit(tr_texts, [r["label"] for r in train])
    thr = float(metrics.threshold_at_fpr(tr_y, det.predict_proba(tr_texts), cfg.fpr_target))
    thr = max(0.0, min(1.0, thr))

    mal = [r for r in test if r["label"] != "benign"]
    row: dict = {"detector": args.detector, "status": "ok", "threshold": thr,
                 "n_malicious": len(mal), "attacks": {}}

    def _summarise(pairs: list[tuple[bool | None, float]], kind: str) -> dict:
        raw = sum(1 for _s in [p[1] for p in pairs] if _s <= thr) / max(1, len(pairs))
        valid = [s for ok, s in pairs if ok is True]
        unknown = sum(1 for ok, _s in pairs if ok is None)
        asr_valid = (sum(1 for s in valid if s <= thr) / len(valid)) if valid else float("nan")
        return {"asr": raw, "asr_valid": asr_valid, "n": len(pairs),
                "n_valid": len(valid), "n_unknown": unknown}

    # ---- 单文本攻击：4 种字符级 + 2 种语义改写
    all_single = {**ATTACKS, **SEMANTIC_ATTACKS}
    for name, atk in all_single.items():
        if name == "identity":
            continue
        mut = [atk.apply(r["text"]) for r in mal]
        sc = det.predict_proba(mut)
        fam = getattr(atk, "family", "obfuscation")
        verdicts = [intent.check(fam, r["text"], m) for r, m in zip(mal, mut)]
        pairs = [(ok, float(s)) for (ok, _why), s in zip(verdicts, sc)]
        row["attacks"][name] = {**_summarise(pairs, fam), "family": fam}

    # ---- 多轮：分轮 vs 拼接
    mt = MultiTurnAttack()
    per_turn: list[tuple[bool | None, float]] = []
    concat: list[tuple[bool | None, float]] = []
    turn_scores_all: list[list[float]] = []
    for r in mal:
        turns = mt.turns(r["text"])
        flat = "\n".join(turns)
        ts = det.predict_proba(turns)
        turn_scores_all.append([float(x) for x in ts])
        # 分轮扫描：每轮单独判定，只要有一轮超标就被拦下 → 攻击"成功"要求全都不超
        per_turn.append((intent.check("multiturn", r["text"], flat)[0], max(ts)))
        concat.append((intent.check("multiturn", r["text"], flat)[0],
                       float(det.predict_proba([flat])[0])))

    a = _summarise(per_turn, "multiturn")
    b = _summarise(concat, "multiturn")
    row["attacks"]["multi-turn"] = {
        "family": "multiturn",
        "asr": a["asr"], "asr_valid": a["asr_valid"], "n": a["n"],
        "n_valid": a["n_valid"], "n_unknown": a["n_unknown"],
        "asr_concat": b["asr"], "asr_concat_valid": b["asr_valid"],
        "mean_turn_score": sum(sum(t) / len(t) for t in turn_scores_all) / max(1, len(turn_scores_all)),
    }

    row["elapsed"] = round(time.perf_counter() - 0.0, 1)
    Path(args.out).write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
