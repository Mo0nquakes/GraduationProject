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
    if kind == "guardrail":
        from .models.guardrail import GuardrailDetector

        return GuardrailDetector()
    if kind in ("distilbert", "distilbert-chunk"):
        from .models.distilbert import DistilBertDetector

        return DistilBertDetector(chunked=(kind == "distilbert-chunk"))
    raise SystemExit(
        f"未知检测器: {kind}（可选 keyword / tfidf-word / tfidf-char / "
        f"distilbert / distilbert-chunk）")


def _fit(det, train, cfg):
    """统一装配：缺依赖时给一句人话，而不是抛 traceback 吓人。"""
    try:
        return det.fit([r["text"] for r in train], [r["label"] for r in train],
                       fpr_target=cfg.fpr_target)
    except ModuleNotFoundError as e:
        raise SystemExit(
            f"检测器 {getattr(det, 'name', '?')} 需要额外的包: {e.name}\n"
            f"请执行: bash scripts/setup_venv.sh   # 或 pip install -r requirements.txt"
        ) from e


def _print_banner(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def cmd_demo(args: argparse.Namespace) -> int:
    cfg: Config = load(args.config)
    if args.seed is not None:
        cfg.seed = args.seed

    rows = toy.build(cfg.toy_per_class, include_hard=args.hard_negatives)
    if args.hard_negatives:
        print("[data] 已混入 3 条难负样本（良性但含 'ignore previous instructions' 等词组）")
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

    _fit(detector, train, cfg)
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

    od = c.get("overdefense") or {}
    if od:
        print(f"\n  过度防御（截错了多少）：")
        print(f"    良性样本整体 FPR      = {metrics.fmt(od['fpr_benign'])}  "
              f"({od['n_benign_flagged']}/{od['n_benign']})")
        print(f"    良性难负样本 FPR      = {metrics.fmt(od['fpr_benign_hard'])}  "
              f"({od['n_benign_hard_flagged']}/{od['n_benign_hard']})")
        print(f"    含触发词的良性 FPR    = {metrics.fmt(od['fpr_benign_cue'])}  "
              f"({od['n_benign_cue_flagged']}/{od['n_benign_cue']})")

    _print_banner("攻击谱系（ASR = 恶意样本被判成良性的比例，越高越脆弱）")
    print(f"  {'attack':<16}{'family':<13}{'ASR':>7}{'ASR*':>7}{'有效':>9}{'TPR':>7}{'F1':>7}")
    print(f"  {'-' * 68}")
    for a in res["attacks"]:
        kept = f"{a['n_intent_kept']}/{a['n_mal']}"
        if a.get("n_intent_unknown"):
            kept += "+?"
        print(f"  {a['attack']:<16}{a['family']:<13}"
              f"{metrics.fmt(a['asr']):>7}{metrics.fmt(a.get('asr_intent_kept')):>7}"
              f"{kept:>9}{metrics.fmt(a['tpr']):>7}{metrics.fmt(a['f1']):>7}")
    print("\n  ASR* = 只统计'改完人还读得懂'的攻击（意图保留）。")
    print("  两者差距大的攻击，说明它是靠把文本改烂来骗过检测器的，不能算真漏洞。")

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
    det = _fit(_make_detector(args.detector), train, cfg)

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
    det = _fit(_make_detector(args.detector), train, cfg)
    score_fn = lambda t: float(det.predict_proba([t])[0])  # noqa: E731

    from .rag.mini_rag import POISONED_DOC, default_store, run_episode

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

    # ---- 攻击谱系 × 端到端：检测器放过了，靶子模型真的会被骗吗？ ----
    _print_banner("被保护模型的实际行为（检测器放行 ≠ 攻击成功）")
    print(f"  {'attack':<16}{'文档分数':>10}{'拦截':>8}{'泄漏':>8}   说明")
    print(f"  {'-' * 66}")
    n_eff = 0
    for name, atk in ATTACKS.items():
        poisoned = atk.apply(POISONED_DOC["text"])
        store = default_store(poisoned=True, poison_text=poisoned)
        out = run_episode(q, store, k=2, detect_fn=score_fn, threshold=args.threshold)
        # 攻击意图是否还读得出来：读不出来就不算"攻击成功"
        from .eval.intent import intent_preserved

        ok, why = intent_preserved(POISONED_DOC["text"], poisoned)
        retrieved = any("2026 revision" in t for t in out["retrieved"])
        if not retrieved:
            note = "没进检索结果（检索层先挡住了，不是检测器的功劳）"
        elif ok is True and out["leaked"]:
            n_eff += 1
            note = "有效攻击：绕过检测并造成泄漏"
        elif out["leaked"]:
            note = "泄漏了，但文本已不可读 → 不算真实漏洞"
        elif out["blocked"]:
            note = "被拦截"
        else:
            note = "未泄漏（靶子没被骗）"
        print(f"  {name:<16}{out['max_score']:>10.3f}"
              f"{'是' if out['blocked'] else '否':>8}{'是' if out['leaked'] else '否':>8}   {note}")
    print("\n  这一列才是'声称保护大模型'时必须给的证据：")
    print("  检测器分数掉下去、而模型仍然照做，才叫攻击成功；只掉分数不改行为，只是指标游戏。")
    return 0


def cmd_a1(args: argparse.Namespace) -> int:
    """A1：同一份数据、同一套攻击、同一套阈值口径下比较全部检测器。"""
    from pathlib import Path as _P

    from .eval.a1 import DEFAULT_DETECTORS, run_a1

    cfg = load(args.config)
    d = _P(args.data) if args.data else sorted(_P("data/processed").glob("*"))[-1]
    if not (d / "train.jsonl").exists():
        raise SystemExit(f"{d} 里没有 train.jsonl，先跑: python -m injlab build-data")
    print(f"[a1] 数据: {d}")
    dets = args.detectors.split(",") if args.detectors else list(DEFAULT_DETECTORS)
    print(f"[a1] 检测器: {', '.join(dets)}")
    run_a1(cfg, d, detectors=dets)
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


def cmd_build_data(args) -> int:
    """拉取真实语料 → 清洗 → 去重 → 配平 → 切分 → 落盘 → 入库。"""
    from .data import build as build_mod

    names = [x.strip() for x in args.sources.split(",") if x.strip()]
    build_mod.build(
        sources_names=names,
        out_dir=args.out,
        limit=args.limit,
        max_per_class=args.max_per_class,
        test_ratio=args.test_ratio,
        seed=args.seed,
        endpoint=args.endpoint,
        cache=not args.no_cache,
        near_dedup=not args.no_near_dedup,
        use_db=not args.no_db,
        dry_run=args.dry_run,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="injlab", description="提示注入检测器对抗鲁棒性实验台")
    p.add_argument("--config", default="configs/default.yaml", help="YAML 配置路径")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="跑通完整链路并打印实验表格（默认命令）")
    d.add_argument("--detector", default="keyword",
                   choices=["keyword", "tfidf-word", "tfidf-char", "distilbert", "distilbert-chunk", "guardrail"])
    d.add_argument("--seed", type=int, default=None)
    d.add_argument("--hard-negatives", action="store_true",
                   help="混入难负样本（A6 实验：观察规则基线崩溃）")
    d.add_argument("--no-db", action="store_true", help="不写数据库，纯离线跑")
    d.set_defaults(func=cmd_demo)

    e = sub.add_parser("er", help="由 schema 生成 Mermaid ER 图")
    e.add_argument("--out", default="docs/er_diagram.md")
    e.set_defaults(func=cmd_er)

    t = sub.add_parser("truncation", help="截断攻击 vs 分块滑窗 对照实验")
    t.add_argument("--detector", default="keyword", choices=["keyword", "tfidf-word", "tfidf-char", "distilbert", "distilbert-chunk", "guardrail"])
    t.add_argument("--sentences", type=int, default=200)
    t.add_argument("--threshold", type=float, default=0.5)
    t.set_defaults(func=cmd_truncation)

    r = sub.add_parser("rag", help="RAG 靶子端到端 ASR 演示")
    r.add_argument("--detector", default="keyword", choices=["keyword", "tfidf-word", "tfidf-char", "distilbert", "distilbert-chunk", "guardrail"])
    r.add_argument("--threshold", type=float, default=0.5)
    r.set_defaults(func=cmd_rag)

    a1 = sub.add_parser("a1", help="A1 全基线对比（真数据，一张表）")
    a1.add_argument("--data", default=None, help="data/processed/<目录>，默认取最新")
    a1.add_argument("--detectors", default=None, help="逗号分隔，默认全部")
    a1.set_defaults(func=cmd_a1)

    v = sub.add_parser("verify", help="校验检测日志的哈希链")
    v.set_defaults(func=cmd_verify)

    s = sub.add_parser("serve", help="启动 FastAPI 服务")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--reload", action="store_true")
    s.set_defaults(func=cmd_serve)

    b = sub.add_parser("build-data", help="拉取真实语料并构建带难负样本的数据集")
    b.add_argument("--sources", default=",".join(
        __import__("injlab.data.sources", fromlist=["DEFAULT_SOURCES"]).DEFAULT_SOURCES),
        help="逗号分隔，可选: fixture,deepset,pint,toxicchat,wildguardmix,xstest")
    b.add_argument("--limit", type=int, default=None, help="每个源最多取多少行（调试用）")
    b.add_argument("--max-per-class", type=int, default=None, help="每类最多保留多少条")
    b.add_argument("--test-ratio", type=float, default=None)
    b.add_argument("--seed", type=int, default=None)
    b.add_argument("--out", default="data/processed")
    b.add_argument("--endpoint", default=None,
                   help="HuggingFace 镜像，默认 https://hf-mirror.com")
    b.add_argument("--no-cache", action="store_true", help="不用 data/raw 缓存，强制重新下载")
    b.add_argument("--no-near-dedup", action="store_true", help="只做精确去重（快，但会残留变体）")
    b.add_argument("--no-db", action="store_true", help="只落盘，不写数据库")
    b.add_argument("--dry-run", action="store_true", help="只统计不落盘")
    b.set_defaults(func=cmd_build_data)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
