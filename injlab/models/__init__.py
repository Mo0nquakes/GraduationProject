"""检测器：统一接口 + 具体实现。新增模型只需实现 fit / predict_proba。"""
from .base import BaseDetector, Detector  # noqa: F401
