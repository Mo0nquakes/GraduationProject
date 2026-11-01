#!/usr/bin/env bash
# A1–A7 一条命令全链路自检。
#
# 两种模式：
#   bash scripts/run_all.sh          # quick：只跑便宜检测器 / 少轮次，约 3–5 分钟，验证链路通
#   bash scripts/run_all.sh full     # full：与论文数字同配置，约 30–40 分钟
#
# 注意：quick 模式**不会**重写 docs/ 里引用的那些 run 目录，
# 它只是证明"每一条实验命令现在都能跑完"。论文数字请以 docs/RESULTS.md 里
# 标注的 run 目录为准。
set -u
cd "$(dirname "$0")/.." || exit 1

PY=".venv/bin/python"
[ -x "$PY" ] || PY="python3"
MODE="${1:-quick}"

if [ "$MODE" = "full" ]; then
    A1_DET="keyword,tfidf-word,tfidf-char,guardrail,distilbert"
    A2_DET="keyword,tfidf-word,tfidf-char,guardrail,distilbert"
    A4_DET="keyword,tfidf-word,tfidf-char,guardrail,distilbert"
    A3_MODES="clean,fgsm,fgsm-weak,pgd"
    A3_SEEDS="0,1,2"
    A5_ROUNDS="3"
else
    A1_DET="keyword,tfidf-char"
    A2_DET="keyword,tfidf-char"
    A4_DET="keyword,tfidf-char"
    A3_MODES="clean,fgsm-weak"
    A3_SEEDS="0"
    A5_ROUNDS="1"
fi

OK=0; FAIL=0
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="runs/_run_all_${STAMP}.log"
mkdir -p runs
: > "$LOG"

run() {
    local name="$1"; shift
    echo ""
    echo "================================================================"
    echo "  $name"
    echo "================================================================"
    if "$@" >>"$LOG" 2>&1; then
        local dir
        dir=$(grep -o 'runs/[0-9_a-z-]*' "$LOG" | tail -1)
        echo "  OK   $name  → ${dir:-见日志}"
        OK=$((OK + 1))
    else
        echo "  FAIL $name  （末尾 15 行）"
        tail -15 "$LOG" | sed 's/^/       /'
        FAIL=$((FAIL + 1))
    fi
}

echo "模式：$MODE   日志：$LOG"
echo "数据：$(ls -1d data/processed/*/ 2>/dev/null | grep -v fixture | tail -1)"

run "A1 全基线对比"  "$PY" -m injlab a1 --detectors "$A1_DET"
run "A2 跨集泛化"    "$PY" -m injlab a2 --detectors "$A2_DET"
run "A4 攻击谱系"    "$PY" -m injlab a4 --detectors "$A4_DET"
run "A3 嵌入空间 AT" "$PY" -m injlab a3 --modes "$A3_MODES" --seeds "$A3_SEEDS"
run "A5 红队回路"    "$PY" -m injlab a5 --rounds "$A5_ROUNDS" --detector tfidf-char
run "A6 难负样本"    "$PY" -m injlab a6 --detectors "${A6_DET:-keyword,tfidf-char}"
if [ "$MODE" = "full" ]; then
    run "A7 端到端 ASR" "$PY" -m injlab a7 --neural-truncation
else
    run "A7 端到端 ASR" "$PY" -m injlab a7 --attacks identity,fullwidth \
        --defenses 无防御,仅检测,规范化+检测
fi

echo ""
echo "================================================================"
printf " A1–A7：通过 %d 项，失败 %d 项\n" "$OK" "$FAIL"
echo " 日志：$LOG"
echo "================================================================"
[ "$FAIL" -eq 0 ]
