"""SQLite 封装 + 防篡改审计链。

审计链做法：每条 logs 的 cur_hash = sha256(prev_hash + 规范化 payload)。
改任何一条历史记录，后续所有 hash 都对不上，`verify_chain()` 立刻报错。
这是本科毕设里性价比最高的"真安全"特性（30 行代码，答辩必问）。
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

GENESIS = "GENESIS"
SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def _canonical(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


class Store:
    def __init__(self, db_path: str | Path = "data/injlab.db") -> None:
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False：FastAPI 把同步端点丢到线程池里执行，
        # 连接在 startup 线程创建、在请求线程使用，默认会直接报错。
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")

    # ---------- 建库 ----------
    def init(self) -> None:
        self.conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # ---------- 写入 ----------
    def add_dataset_version(self, name: str, source_mix: dict, split_rule: str, sha256: str) -> int:
        cur = self.conn.execute(
            "INSERT INTO dataset_version (name, source_mix, split_rule, sha256) VALUES (?,?,?,?)",
            (name, _canonical(source_mix), split_rule, sha256),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def add_samples(self, rows: Iterable[dict[str, Any]], dataset_version_id: int) -> list[int]:
        ids: list[int] = []
        for r in rows:
            cur = self.conn.execute(
                "INSERT OR IGNORE INTO sample "
                "(dataset_version_id, text, label, source, split, sha256, is_hard_negative) "
                "VALUES (?,?,?,?,?,?,?)",
                (
                    dataset_version_id,
                    r["text"],
                    r["label"],
                    r.get("source"),
                    r["split"],
                    r["sha256"],
                    int(r.get("is_hard_negative", 0)),
                ),
            )
            ids.append(int(cur.lastrowid))
        self.conn.commit()
        return ids

    def upsert_attack(self, name: str, family: str, params: dict | None = None) -> int:
        row = self.conn.execute("SELECT id FROM attack WHERE name = ?", (name,)).fetchone()
        if row:
            return int(row["id"])
        cur = self.conn.execute(
            "INSERT INTO attack (name, family, params_json) VALUES (?,?,?)",
            (name, family, _canonical(params or {})),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def add_attack_samples(self, rows: Iterable[tuple[int, int, str, int | None]]) -> None:
        self.conn.executemany(
            "INSERT INTO attack_sample (attack_id, source_sample_id, text, achieved_evasion) "
            "VALUES (?,?,?,?)",
            list(rows),
        )
        self.conn.commit()

    def add_defense_policy(self, name: str, threshold: float, normalize_on: bool,
                           chunk_size: int, chunk_stride: int, params: dict | None = None) -> int:
        row = self.conn.execute("SELECT id FROM defense_policy WHERE name = ?", (name,)).fetchone()
        if row:
            return int(row["id"])
        cur = self.conn.execute(
            "INSERT INTO defense_policy (name, threshold, normalize_on, chunk_size, chunk_stride, params_json) "
            "VALUES (?,?,?,?,?,?)",
            (name, threshold, int(normalize_on), chunk_size, chunk_stride, _canonical(params or {})),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def add_model_version(self, name: str, arch: str, ckpt_path: str | None,
                          hyper: dict | None, seed: int) -> int:
        cur = self.conn.execute(
            "INSERT INTO model_version (name, arch, ckpt_path, hyper_json, seed) VALUES (?,?,?,?,?)",
            (name, arch, ckpt_path, _canonical(hyper or {}), seed),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def add_experiment_run(self, name: str, config: dict, policy_id: int | None,
                           git_commit: str | None = None) -> int:
        cur = self.conn.execute(
            "INSERT INTO experiment_run (name, config_json, policy_id, git_commit) VALUES (?,?,?,?)",
            (name, _canonical(config), policy_id, git_commit),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def finish_experiment_run(self, run_id: int, metrics: dict) -> None:
        self.conn.execute(
            "UPDATE experiment_run SET metrics_json = ?, finished_at = datetime('now') WHERE id = ?",
            (_canonical(metrics), run_id),
        )
        self.conn.commit()

    # ---------- 审计链 ----------
    def log_detection(self, *, model_version_id: int, text_sha256: str, score: float,
                      pred: str, input_kind: str = "live", input_ref: int | None = None,
                      experiment_run_id: int | None = None, latency_ms: float | None = None) -> int:
        last = self.conn.execute(
            "SELECT id, cur_hash FROM detection_log ORDER BY id DESC LIMIT 1"
        ).fetchone()
        prev_id = int(last["id"]) if last else None
        prev_hash = last["cur_hash"] if last else GENESIS
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        payload = {
            "model_version_id": model_version_id,
            "experiment_run_id": experiment_run_id,
            "input_kind": input_kind,
            "input_ref": input_ref,
            "text_sha256": text_sha256,
            "score": float(score),
            "pred": pred,
            "latency_ms": None if latency_ms is None else float(latency_ms),
            "ts": ts,
        }
        cur_hash = hashlib.sha256((prev_hash + _canonical(payload)).encode("utf-8")).hexdigest()
        cur = self.conn.execute(
            "INSERT INTO detection_log "
            "(model_version_id, experiment_run_id, input_kind, input_ref, text_sha256, "
            " score, pred, latency_ms, prev_id, prev_hash, cur_hash, ts) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (model_version_id, experiment_run_id, input_kind, input_ref, text_sha256,
             payload["score"], pred, payload["latency_ms"], prev_id, prev_hash, cur_hash, ts),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def verify_chain(self) -> tuple[bool, str]:
        """逐条重算哈希链。返回 (是否完整, 说明)。"""
        prev_hash = GENESIS
        n = 0
        for row in self.conn.execute("SELECT * FROM detection_log ORDER BY id ASC"):
            n += 1
            payload = {
                "model_version_id": row["model_version_id"],
                "experiment_run_id": row["experiment_run_id"],
                "input_kind": row["input_kind"],
                "input_ref": row["input_ref"],
                "text_sha256": row["text_sha256"],
                "score": float(row["score"]),
                "pred": row["pred"],
                "latency_ms": None if row["latency_ms"] is None else float(row["latency_ms"]),
                "ts": row["ts"],
            }
            expect = hashlib.sha256((prev_hash + _canonical(payload)).encode("utf-8")).hexdigest()
            if row["prev_hash"] != prev_hash:
                return False, f"第 {row['id']} 条 prev_hash 不匹配（疑似被删除或重排）"
            if row["cur_hash"] != expect:
                return False, f"第 {row['id']} 条 cur_hash 不匹配（疑似内容被篡改）"
            prev_hash = row["cur_hash"]
        return True, f"链完整，共 {n} 条记录"

    # ---------- 统计 ----------
    def counts(self) -> dict[str, int]:
        tables = [r["name"] for r in self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )]
        return {t: self.conn.execute(f'SELECT COUNT(*) AS c FROM "{t}"').fetchone()["c"] for t in tables}
