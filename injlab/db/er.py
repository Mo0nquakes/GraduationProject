"""从 schema.sql 生成 Mermaid ER 图。

为什么不手画：手画的 ER 图一定和代码脱节，中期检查/答辩时一被追问就露馅。
这里用 PRAGMA 读真实外键，schema 改了图就跟着改，且能直接粘进 draw.io
（draw.io 支持 Import from Mermaid）或 Notion/Typora。
"""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path

from .store import SCHEMA_PATH

_REL_LABEL = {
    "dataset_version_id": "包含",
    "source_sample_id": "派生自",
    "attack_id": "使用",
    "model_version_id": "产出",
    "experiment_run_id": "归属",
    "policy_id": "采用",
    "prev_id": "链接上一条",
}

_SAFE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _mermaid_type(decl: str | None) -> str:
    t = (decl or "TEXT").strip().upper()
    t = re.split(r"[ (\[]", t)[0]
    return t if _SAFE.match(t) else "TEXT"


def mermaid(sql: str = "") -> str:
    sql = sql or SCHEMA_PATH.read_text(encoding="utf-8")
    con = sqlite3.connect(":memory:")
    con.executescript(sql)
    tables = [r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    )]
    lines: list[str] = ["erDiagram"]

    # 关系（先画，Mermaid 会自己排布）
    for t in tables:
        if not _SAFE.match(t):
            continue
        for fk in con.execute(f'PRAGMA foreign_key_list("{t}")'):
            parent, child_col = fk[2], fk[3]
            label = _REL_LABEL.get(child_col, "关联")
            lines.append(f'    {parent} ||--o{{ {t} : "{label}"')

    # 实体与字段
    for t in tables:
        if not _SAFE.match(t):
            continue
        lines.append(f"    {t} {{")
        for col in con.execute(f'PRAGMA table_info("{t}")'):
            name, decl, pk = col[1], col[2], col[5]
            key = " PK" if pk else ""
            lines.append(f"        {_mermaid_type(decl)} {name}{key}")
        lines.append("    }")
    con.close()
    return "\n".join(lines)


def write_markdown(path: str | Path = "docs/er_diagram.md") -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    body = mermaid()
    out.write_text(
        "# ER 图（由 injlab/db/schema.sql 自动生成，勿手改）\n\n"
        "```mermaid\n" + body + "\n```\n",
        encoding="utf-8",
    )
    return out
