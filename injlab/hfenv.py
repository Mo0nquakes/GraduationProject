"""统一 HF 环境：镜像 + 超时。

为什么要单独一个模块：`huggingface.co` 在本机直连会挂死（不是报错，是长时间
无响应），任何 `from_pretrained` 只要没设 HF_ENDPOINT 就会卡在 DNS/连接上，
看上去像"代码有 bug"，实际是网络。所有碰 transformers / datasets 的模块
必须先 import 这个，把它放在 import 语句最前面。
"""
from __future__ import annotations

import os


def ensure_hf_env() -> None:
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
