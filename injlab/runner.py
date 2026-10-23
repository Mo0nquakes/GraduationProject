"""实验运行器：一次 run 产出 config / metrics / predictions 三件套并落库。

方法论上刻意做对的一件事：**阈值只在训练集上校准**，然后原封不动用到测试集。
如果用测试集自己挑阈值（很多人这么干），TPR@FPR 会系统性虚高，
评审一问"你的阈值怎么定的"就答不上来。
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from . import metrics
from .config import Config
from .db.store import Store
from .labels import LABELS
from .models.base import BaseDetector


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True, timeout=5, cwd=Path(__file__).resolve().parent)
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def ingest(store: Store, rows: Sequence[dict[str, Any]], name: str,
           split_rule: str = "stratified,test_ratio=0.30,seed=20260101") -> tuple[int, dict[str, int]]:
    """把一批样本写进库，返回 (dataset_version_id, sha256 -> sample_id)。"""
    mix: dict[str, int] = {}
    for r in rows:
        src = r.get("source") or "unknown"
        mix[src] = mix.get(src, 0) + 1
    dv_id = store.add_dataset_version(
        name=name,
        source_mix=mix,
        split_rule=split_rule,
        sha256=hashlib.sha256("".join(sorted(r["sha256"] for r in rows)).encode()).hexdigest(),
    )
    store.add_samples(rows, dv_id)
    id_map = {
        row["sha256"]: int(row["id"])
        for row in store.conn.execute(
            "SELECT id, sha256 FROM sample WHERE dataset_version_id = ?", (dv_id,)
        )
    }
    return dv_id, id_map


def _score(detector: BaseDetector, texts: Sequence[str]) -> tuple[list[float], float]:
    """返回 (分数, 单条平均延迟 ms)。"""
    t0 = time.perf_counter()
    scores = detector.predict_proba(list(texts))
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return scores, elapsed_ms / max(1, len(texts))


def _binary_y(rows: Sequence[dict[str, Any]]) -> list[int]:
    return [0 if r["label"] == "benign" else 1 for r in rows]


def _recall_by_label(rows: Sequence[dict[str, Any]], scores: Sequence[float],
                     thr: float) -> dict[str, float]:
    buckets: dict[str, list[int]] = {}
    for r, s in zip(rows, scores):
        buckets.setdefault(r["label"], []).append(1 if s > thr else 0)
    return {lab: (sum(v) / len(v) if v else float("nan")) for lab, v in sorted(buckets.items())}


def run_experiment(
    cfg: Config,
    detector: BaseDetector,
    train: Sequence[dict[str, Any]],
    test: Sequence[dict[str, Any]],
    attacks: Sequence[Any],
    store: Store | None = None,
    id_map: dict[str, int] | None = None,
    model_version_id: int | None = None,
    run_name: str = "run",
    policy_id: int | None = None,
) -> dict[str, Any]:
    run_dir = Path(cfg.runs_dir) / f"{datetime.now():%Y%m%d_%H%M%S}_{run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    id_map = id_map or {}

    # 1) 训练集上校准阈值（阈值的唯一合法来源）
    tr_scores, _ = _score(detector, [r["text"] for r in train])
    tr_y = _binary_y(train)
    calib = metrics.tpr_at_fpr(tr_y, tr_scores, cfg.fpr_target)
    thr = metrics.threshold_at_fpr(tr_y, tr_scores, cfg.fpr_target)
    thr = max(0.0, min(1.0, float(thr)))

    # 2) 干净测试集
    te_scores, lat = _score(detector, [r["text"] for r in test])
    te_y = _binary_y(test)
    pred = [1 if s > thr else 0 for s in te_scores]
    clean = {
        "tpr_at_fpr": metrics.tpr_at_fpr(te_y, te_scores, cfg.fpr_target),
        "tpr_at_fpr_10": metrics.tpr_at_fpr(te_y, te_scores, 0.10),
        "n_benign_test": sum(1 for v in te_y if v == 0),
        "auc": metrics.auc(te_y, te_scores),
        **metrics.prf(te_y, pred),
        "threshold": thr,
        "latency_ms_per_item": lat,
        "recall_by_label": _recall_by_label(test, te_scores, thr),
    }

    predictions: list[dict[str, Any]] = [
        {"attack": "clean", "label": r["label"], "score": s,
         "pred": int(s > thr), "sha256": r["sha256"]}
        for r, s in zip(test, te_scores)
    ]

    # 3) 攻击谱系：只改恶意样本，良性样本原样保留（保证 FPR 口径不变）
    attack_results: list[dict[str, Any]] = []
    for atk in attacks:
        mixed: list[dict[str, Any]] = []
        origin_sha: dict[int, str] = {}
        for r in test:
            if r["label"] == "benign":
                mixed.append(r)
            else:
                origin_sha[len(mixed)] = r["sha256"]
                mixed.append({**r, "text": atk.apply(r["text"])})

        atk_scores, atk_lat = _score(detector, [r["text"] for r in mixed])
        mal_idx = [i for i, r in enumerate(mixed) if r["label"] != "benign"]
        mal_scores = [atk_scores[i] for i in mal_idx]
        ay = _binary_y(mixed)
        apred = [1 if s > thr else 0 for s in atk_scores]
        prf = metrics.prf(ay, apred)
        attack_results.append({
            "attack": atk.name,
            "family": getattr(atk, "family", "obfuscation"),
            "asr": metrics.asr(mal_scores, thr),
            "tpr_at_fpr": metrics.tpr_at_fpr(ay, atk_scores, cfg.fpr_target)["tpr"],
            "tpr": prf["recall"],
            "f1": prf["f1"],
            "latency_ms_per_item": atk_lat,
        })
        predictions.extend(
            {"attack": atk.name, "label": r["label"], "score": s,
             "pred": int(s > thr), "sha256": r["sha256"]}
            for r, s in zip(mixed, atk_scores)
        )

        # 落库：攻击生成样本 + 该样本是否逃逸成功
        if store is not None:
            atk_id = store.upsert_attack(atk.name, getattr(atk, "family", "obfuscation"))
            rows_to_insert: list[tuple[int, int, str, int]] = []
            for i in mal_idx:
                sid = id_map.get(origin_sha[i])
                if sid is None:
                    continue
                rows_to_insert.append((atk_id, sid, mixed[i]["text"], int(atk_scores[i] <= thr)))
            if rows_to_insert:
                store.add_attack_samples(rows_to_insert)

    result: dict[str, Any] = {
        "run": run_name,
        "detector": detector.name,
        "n_train": len(train),
        "n_test": len(test),
        "threshold": thr,
        "calibration": calib,
        "clean": clean,
        "attacks": attack_results,
        "labels": list(LABELS),
        "config": cfg.to_dict(),
        "git_commit": git_commit(),
    }

    # 4) 落库：run 记录 + 检测日志（审计链）
    if store is not None:
        run_id = store.add_experiment_run(run_name, cfg.to_dict(), policy_id, result["git_commit"])
        result["experiment_run_id"] = run_id
        if model_version_id is not None:
            for r, s in zip(test, te_scores):
                store.log_detection(
                    model_version_id=model_version_id,
                    experiment_run_id=run_id,
                    input_kind="sample",
                    input_ref=id_map.get(r["sha256"]),
                    text_sha256=r["sha256"],
                    score=float(s),
                    pred="malicious" if s > thr else "benign",
                    latency_ms=lat,
                )
        store.finish_experiment_run(run_id, result)

    # 5) 落盘
    result["run_dir"] = str(run_dir)
    (run_dir / "config.json").write_text(
        json.dumps(cfg.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    with (run_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    return result
