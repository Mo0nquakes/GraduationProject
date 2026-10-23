"""全局配置：dataclass 承载默认值，YAML 可选覆盖。

刻意不依赖 PyYAML：没装也能跑，装了才读 configs/*.yaml。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass
class Config:
    seed: int = 20260101
    db_path: str = "data/injlab.db"
    runs_dir: str = "runs"
    toy_per_class: int = 60
    test_ratio: float = 0.30
    fpr_target: float = 0.01
    threshold: float = 0.50
    max_len: int = 512
    chunk_size: int = 256
    chunk_stride: int = 128

    def to_dict(self) -> dict:
        return asdict(self)


def load(path: str | Path = "configs/default.yaml") -> Config:
    """读 YAML 覆盖默认值；无 PyYAML 或文件不存在则用默认值。"""
    cfg = Config()
    p = Path(path)
    if not p.exists():
        return cfg
    try:
        import yaml  # noqa: PLC0415  —— 延迟导入，避免硬依赖
    except ModuleNotFoundError:
        print(f"[config] 未安装 PyYAML，忽略 {p}，使用内置默认值")
        return cfg
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    known = {f.name for f in fields(cfg)}
    for k, v in raw.items():
        if k in known:
            setattr(cfg, k, v)
        else:
            print(f"[config] 未知字段已忽略: {k}")
    return cfg
