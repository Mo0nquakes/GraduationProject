"""单个神经检测器的离线打分器（子进程）。

前端点"深度检测"时由 API 以子进程方式调用：加载模型 → 打分 → 退出。
这样 API 主进程永远不持有模型，8GB 机器上不会被挤爆；代价是每次约 10 秒冷启动，
所以前端必须显示等待态，不能假装它是实时的。
"""
from __future__ import annotations

import argparse
import json


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", required=True)
    ap.add_argument("--text", required=True)
    args = ap.parse_args()

    try:
        import torch

        torch.set_num_threads(2)
    except Exception:  # noqa: BLE001
        pass

    from .models.guardrail import GuardrailDetector
    from .models.distilbert import DistilBertDetector

    if args.detector == "guardrail":
        det = GuardrailDetector(verbose=False)
    elif args.detector == "distilbert":
        det = DistilBertDetector(verbose=False)
        if not det.load_or_none():
            # 关键：没有权重就报错退出，绝不能退回去用随机初始化的分类头——
            # 那样会给出看起来正常、实际毫无意义的分数，比报错危险得多。
            raise SystemExit("没有找到已训练的 DistilBERT 权重，先跑一次: python -m injlab a1 --detectors distilbert")
    else:
        raise SystemExit(f"未知深度检测器: {args.detector}")

    score = float(det.predict_proba([args.text])[0])
    print(json.dumps({"detector": args.detector, "score": score}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
