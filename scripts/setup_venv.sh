#!/usr/bin/env bash
# 用 Homebrew 的 python3.12 建虚拟环境（系统自带 3.9.6 太老，别用）
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-/opt/homebrew/bin/python3.12}"
if [ ! -x "$PY" ]; then
  echo "找不到 $PY，请先 brew install python@3.12，或用 PYTHON=... 指定解释器"
  exit 1
fi

"$PY" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo
echo "已就绪。后续每次使用："
echo "  source .venv/bin/activate"
echo "  python -m injlab demo"
