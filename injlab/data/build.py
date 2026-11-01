"""build-data 的主流程：拉取 → 清洗 → 去重 → 配平 → 分组切分 → 落盘 → 入库。

产物（每次运行都建新的带时间戳目录，绝不覆盖上一次）：

    data/processed/<name>/
        train.jsonl      训练集
        test.jsonl       测试集
        manifest.json    来源、去重统计、每类条数、文件 sha256、参数

manifest 里的 sha256 会同时写进数据库的 dataset_version 表，
所以"这次实验用的是哪版数据"永远可追溯 —— 这是论文可复现性的地基。
"""
from __future__ import annotations

import hashlib
import json
import os
import time

from ..config import load as load_config
from ..labels import LABELS
from . import prep, sources

DEFAULT_OUT = "data/processed"


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_jsonl(path: str, records: list, split: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            row = {
                "text": r.text,
                "label": r.label,
                "source": r.source,
                "split": split,
                "is_hard_negative": int(getattr(r, "hard", False)),
                "group": getattr(r, "group", ""),
                "meta": getattr(r, "meta", {}) or {},
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _friendly_name(names) -> str:
    return "+".join(names)


def build(sources_names=None, out_dir: str = DEFAULT_OUT, limit: int | None = None,
          max_per_class: int | None = None, test_ratio: float | None = None,
          seed: int | None = None, endpoint: str | None = None,
          cache: bool = True, near_dedup: bool = True,
          use_db: bool = True, dry_run: bool = False) -> dict:
    cfg = load_config()
    names = list(sources_names or sources.DEFAULT_SOURCES)
    test_ratio = cfg.test_ratio if test_ratio is None else test_ratio
    seed = cfg.seed if seed is None else seed

    print("=" * 72)
    print(f" build-data  sources={names}  limit={limit}  out={out_dir}")
    print("=" * 72)

    # ---- 1. 拉取 ----
    raw = sources.load(names, limit=limit, endpoint=endpoint, cache=cache)
    if not raw:
        raise SystemExit("一条样本都没拉到。检查网络，或先用 --sources fixture 冒烟。")
    stats_raw = prep.summary(raw)

    # ---- 2. 清洗：只丢空文本，其余保留原文 ----
    recs = [r for r in raw if prep.clean_text(r.text).strip()]
    n_empty = len(raw) - len(recs)
    if n_empty:
        print(f"  清洗: 丢弃空文本 {n_empty} 条")

    # ---- 3. 去重 ----
    print("  去重中（精确 + MinHash 近似）...")
    recs, dedup_stats = prep.dedup(recs, near=near_dedup)
    print(f"  去重: 输入 {dedup_stats['input']}  精确重复 {dedup_stats['exact_dup']}  "
          f"近似重复 {dedup_stats['near_dup']}  保留 {dedup_stats['kept']}")

    # ---- 4. 配平 ----
    recs = prep.balance(recs, max_per_class=max_per_class, seed=seed)
    stats_bal = prep.summary(recs)

    # ---- 5. 分组切分 ----
    train, test = prep.group_split(recs, test_ratio=test_ratio, seed=seed)

    # ---- 5b. 泄漏自检：切分之后还要再确认一次 ----
    leak = prep.leakage_report(train, test)
    if leak["n_pairs_over"] > 0:
        print(f"  [警告] 发现 {leak['n_pairs_over']} 对训练/测试近似重复"
              f"（最高相似度 {leak['max_sim']:.3f}）—— 先修去重再跑实验，"
              f"否则测出来的鲁棒性是假的")

    # ---- 6. 落盘 ----
    stamp = time.strftime("%Y%m%d-%H%M%S")
    # 目录名用"实际有数据的源"，避免某个源失败时名字骗人
    actual = [k for k, v in stats_bal["by_source"].items() if v]
    name = f"{stamp}_{_friendly_name(actual) or 'empty'}"
    run_dir = os.path.join(out_dir, name)
    if not dry_run:
        os.makedirs(run_dir, exist_ok=True)
        train_path = os.path.join(run_dir, "train.jsonl")
        test_path = os.path.join(run_dir, "test.jsonl")
        _write_jsonl(train_path, train, "train")
        _write_jsonl(test_path, test, "test")
        train_sha, test_sha = _sha256_file(train_path), _sha256_file(test_path)
    else:
        train_path = test_path = "(dry-run)"
        train_sha = test_sha = "dry-run"

    manifest = {
        "name": name,
        "created_at": stamp,
        "sources": names,
        "endpoint": endpoint or os.environ.get("HF_ENDPOINT") or sources.DEFAULT_ENDPOINT,
        "params": {"limit": limit, "max_per_class": max_per_class,
                   "test_ratio": test_ratio, "seed": seed, "near_dedup": near_dedup},
        "counts": {"raw": stats_raw["n"], "after_clean": len(raw) - n_empty,
                   "after_dedup": dedup_stats["kept"],
                   "after_balance": len(recs),
                   "train": len(train), "test": len(test)},
        "dedup": dedup_stats,
        "label_balance": stats_bal["by_label"],
        "source_mix": stats_bal["by_source"],
        "hard_negatives": stats_bal["hard_negatives"],
        "avg_chars": stats_bal["avg_chars"],
        "leakage": leak,
        "files": {"train.jsonl": train_sha, "test.jsonl": test_sha},
    }

    if not dry_run:
        with open(os.path.join(run_dir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

    # ---- 7. 入库 ----
    if use_db and not dry_run:
        from ..db.store import Store

        store = Store(cfg.db_path)
        store.init()
        dv = store.add_dataset_version(
            name=name,
            source_mix=stats_bal["by_source"],
            split_rule=f"group_split(test_ratio={test_ratio},seed={seed})",
            sha256=train_sha,
        )
        rows = []
        for split, items in (("train", train), ("test", test)):
            for r in items:
                rows.append({
                    "text": r.text, "label": r.label, "source": r.source,
                    "split": split, "sha256": prep.dedup_key(r.text),
                    "is_hard_negative": int(getattr(r, "hard", False)),
                })
        store.add_samples(rows, dv)
        store.close()
        manifest["dataset_version_id"] = dv
        print(f"  入库: dataset_version_id={dv}  samples={len(rows)}")

    # ---- 8. 打印摘要 ----
    print("-" * 72)
    print(f"  原始 {manifest['counts']['raw']} → 去重后 {manifest['counts']['after_dedup']}"
          f" → 配平后 {manifest['counts']['after_balance']}"
          f" → train {len(train)} / test {len(test)}")
    print("  标签分布: " + "  ".join(f"{k}={v}" for k, v in manifest["label_balance"].items()))
    print(f"  难负样本 {manifest['hard_negatives']} 条  平均长度 {manifest['avg_chars']} 字符")
    print("  来源分布: " + "  ".join(f"{k}={v}" for k, v in manifest["source_mix"].items()))
    missing = [k for k, v in manifest["label_balance"].items() if v == 0]
    if missing:
        print(f"  [警告] 这些类没有样本: {', '.join(missing)} —— 多源一起用，或加合成数据兜底")
    print("-" * 72)
    if not dry_run:
        print(f"  输出目录: {run_dir}")
    return manifest
