"""命令行入口：python -m injlab <cmd>。

设计取舍：默认命令 `demo` 只需标准库就能跑出完整实验表格。
它是这个仓库的"可执行说明书"——先让你看到结论，再看代码。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import metrics
from .attacks.obfuscation import ATTACKS
from .chunking import make_long_document, max_chunk_score, truncated
from .config import Config, load
from .data import toy
from .data.split import dedup, stratified_split
from .db.er import write_markdown
from .db.store import Store
from .models.base import BaseDetector
from .models.stub import KeywordDetector
from .runner import ingest, run_experiment


def _make_detector(kind: str) -> BaseDetector:
    if kind == "keyword":
        return KeywordDetector()
    if kind in ("tfidf-word", "tfidf-char"):
        analyzer = "word" if kind == "tfidf-word" else "char_wb"
        try:
            from .models.tfidf_lr import TfidfLogReg
        except ModuleNotFoundError:
            raise SystemExit("缺少 scikit-learn，请先 pip install -r requirements.txt")
        return TfidfLogReg(analyzer=analyzer)
    raise SystemExit(f"未知检测器: {kind}（可选 keyword / tfidf-word / tfidf-char）")


def _print_banner(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def cmd_demo(args: argparse.Namespace) -> int:
    cfg: Config = load(args.config)
    if args.seed is not None:
        cfg.seed = args.seed

    rows = toy.build(cfg.toy_per_class)
    rows, dropped = dedup(rows)
    train, test = stratified_split(rows, cfg.test_ratio, cfg.seed)
    print(f"[data] 生成 {len(rows) + dropped} 条 -> 去重丢弃 {dropped} 条 -> "
          f"train {len(train)} / test {len(test)}")

    detector = _make_detector(args.detector)

    store = None
    id_map: dict[str, int] = {}
    model_version_id = None
    dv_id = None
    if not args.no_db:
        store = Store(cfg.db_path)
        store.init()
        dv_id, id_map = ingest(store, rows, name=f"toy-{cfg.toy_per_class}",
                               split_rule=f"stratified,test_ratio={cfg.test_ratio},seed={cfg.seed}")
        print(f"[db] dataset_version_id={dv_id}  {cfg.db_path}")

    detector.fit([r["text"] for r in train], [r["label"] for r in train],
                 fpr_target=cfg.fpr_target)
    if store is not None:
        model_version_id = store.add_model_version(
            name=detector.name, arch=args.detector, ckpt_path=None,
            hyper={"fpr_target": cfg.fpr_target, "analyzer": args.detector}, seed=cfg.seed,
        )

    res = run_experiment(
        cfg, detector, train, test,
        attacks=list(ATTACKS.values()),
        store=store, id_map=id_map, model_version_id=model_version_id,
        run_name=f"demo_{args.detector}", policy_id=None,
    )

    c = res["clean"]
    _print_banner(f"检测器: {res['detector']}   （阈值在训练集上校准）")
    print(f"  阈值 threshold      = {c['threshold']:.4f}")
    print(f"  TPR @ FPR=1%        = {metrics.fmt(c['tpr_at_fpr']['tpr'])}"
          f"   (实际 FPR {metrics.fmt(c['tpr_at_fpr']['fpr'])})")
    print(f"  TPR @ FPR=10%       = {metrics.fmt(c['tpr_at_fpr_10']['tpr'])}")
    print(f"  AUC                 = {metrics.fmt(c['auc'])}")
    print(f"  Precision / Recall / F1 = {metrics.fmt(c['precision'])} / "
          f"{metrics.fmt(c['recall'])} / {metrics.fmt(c['f1'])}")
    print(f"  单条延迟             = {c['latency_ms_per_item']:.4f} ms")
    if c["n_benign_test"] < 100:
        print(f"  [注意] 测试集只有 {c['n_benign_test']} 条良性样本，"
              f"FPR=1% 的估计不可靠（分辨率约 {100 / max(1, c['n_benign_test']):.1f}%）。")
        print("         这就是为什么论文必须报'负样本量 N'，而不是只报一个 TPR@FPR。")
    print("  分标签召回:")
    for lab, v in c["recall_by_label"].items():
        print(f"    {lab:<20} {metrics.fmt(v)}")

    _print_banner("攻击谱系（ASR = 恶意样本被判成良性的比例，越高越脆弱）")
    print(f"  {'attack':<16}{'family':<14}{'ASR':>7}{'TPR':>8}{'F1':>8}")
    print(f"  {'-' * 53}")
    for a in res["attacks"]:
        print(f"  {a['attack']:<16}{a['family']:<14}"
              f"{metrics.fmt(a['asr']):>7}{metrics.fmt(a['tpr']):>8}{metrics.fmt(a['f1']):>8}")

    if store is not None:
        ok, msg = store.verify_chain()
        print(f"\n[audit] 哈希链校验: {'通过' if ok else '失败'} —— {msg}")
        print(f"[db] 表计数: {store.counts()}")
        store.close()

    print(f"\n[out] 产物目录: {res['run_dir']}")
    print("      config.json / metrics.json / predictions.jsonl")
    return 0


def cmd_er(args: argparse.Namespace) -> int:
    path = write_markdown(args.out)
    print(f"[er] 已生成 Mermaid ER 图: {path}")
    print("     可直接粘贴到 https://mermaid.live 或 draw.io (Import from Mermaid)")
    return 0


def cmd_truncation(args: argparse.Namespace) -> int:
    cfg = load(args.config)
    rows, _ = dedup(toy.build(cfg.toy_per_class))
    train, _ = stratified_split(rows, cfg.test_ratio, cfg.seed)
    det = _make_detector(args.detector).fit(
        [r["text"] for r in train], [r["label"] for r in train], fpr_target=cfg.fpr_target)

    doc = make_long_document(body_sentences=args.sentences)
    n_words = len(doc.split())
    score_trunc = float(det.predict_proba([truncated(doc, cfg.max_len)])[0])
    score_chunk, where = max_chunk_score(
        lambda t: float(det.predict_proba([t])[0]), doc,
        size=cfg.chunk_size, stride=cfg.chunk_stride)

    _print_banner("截断攻击：恶意指令在长文档末尾")
    print(f"  文档长度               = {n_words} 词（近似 token）")
    print(f"  朴素截断 {cfg.max_len:>4} 词        得分 = {score_trunc:.4f}"
          f"   -> {'漏检' if score_trunc <= args.threshold else '检出'}")
    print(f"  分块滑窗 {cfg.chunk_size}/{cfg.chunk_stride}      得分 = {score_chunk:.4f}"
          f"   -> {'漏检' if score_chunk <= args.threshold else '检出'}"
          f"   (命中块起始词 {where})")
    print("\n  结论：同一个检测器、同一篇文档，只因为实现方式不同，结论就反了。")
    print("       这就是'实现级漏洞'，也是论文里最容易被认可的一类发现。")
    return 0


def cmd_rag(args: argparse.Namespace) -> int:
    cfg = load(args.config)
    rows, _ = dedup(toy.build(cfg.toy_per_class))
    train, _ = stratified_split(rows, cfg.test_ratio, cfg.seed)
    det = _make_detector(args.detector).fit(
        [r["text"] for r in train], [r["label"] for r in train], fpr_target=cfg.fpr_target)
    score_fn = lambda t: float(det.predict_proba([t])[0])  # noqa: E731

    from .rag.mini_rag import default_store, run_episode

    q = "How do I file an expense report?"
    _print_banner("端到端 ASR：恶意文档污染检索库")
    print(f"  提问: {q}")
    for label, guarded in (("无防御", False), ("有检测器", True)):
        store = default_store(poisoned=True)
        out = run_episode(q, store, k=2,
                          detect_fn=score_fn if guarded else None,
                          threshold=args.threshold)
        verdict = "泄漏" if out["leaked"] else ("已拦截" if out["blocked"] else "未泄漏")
        print(f"  [{label:<6}] 检索到 {out['retrieved']}")
        print(f"           文档分数 {[round(s, 3) for s in out['doc_scores']]} "
              f"-> {verdict}")
        if out["answer"]:
            print(f"           回答: {out['answer'][:80]}")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    cfg = load(args.config)
    store = Store(cfg.db_path)
    ok, msg = store.verify_chain()
    print(f"[audit] {'通过' if ok else '失败'} —— {msg}")
    print(f"[db] {store.counts()}")
    store.close()
    return 0 if ok else 1


def cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ModuleNotFoundError:
        raise SystemExit("缺少 uvicorn，请先 pip install fastapi uvicorn")
    print(f"[serve] http://{args.host}:{args.port}/docs")
    uvicorn.run("injlab.api.main:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="injlab", description="提示注入检测器对抗鲁棒性实验台")
    p.add_argument("--config", default="configs/default.yaml", help="YAML 配置路径")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="跑通完整链路并打印实验表格（默认命令）")
    d.add_argument("--detector", default="keyword",
                   choices=["keyword", "tfidf-word", "tfidf-char"])
    d.add_argument("--seed", type=int, default=None)
    d.add_argument("--no-db", action="store_true", help="不写数据库，纯离线跑")
    d.set_defaults(func=cmd_demo)

    e = sub.add_parser("er", help="由 schema 生成 Mermaid ER 图")
    e.add_argument("--out", default="docs/er_diagram.md")
    e.set_defaults(func=cmd_er)

    t = sub.add_parser("truncation", help="截断攻击 vs 分块滑窗 对照实验")
    t.add_argument("--detector", default="keyword", choices=["keyword", "tfidf-word", "tfidf-char"])
    t.add_argument("--sentences", type=int, default=200)
    t.add_argument("--threshold", type=float, default=0.5)
    t.set_defaults(func=cmd_truncation)

    r = sub.add_parser("rag", help="RAG 靶子端到端 ASR 演示")
    r.add_argument("--detector", default="keyword", choices=["keyword", "tfidf-word", "tfidf-char"])
    r.add_argument("--threshold", type=float, default=0.5)
    r.set_defaults(func=cmd_rag)

    v = sub.add_parser("verify", help="校验检测日志的哈希链")
    v.set_defaults(func=cmd_verify)

    s = sub.add_parser("serve", help="启动 FastAPI 服务")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--reload", action="store_true")
    s.set_defaults(func=cmd_serve)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
