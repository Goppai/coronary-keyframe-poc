#!/bin/zsh
cd -- "$(dirname -- "$0")" || exit 1
if [[ ! -x .venv/bin/python ]]; then
  print '未找到项目虚拟环境，请先按 README.md 安装依赖。'
  read -k 1 '?按任意键退出…'
  exit 1
fi
exec .venv/bin/python scripts/web.py --open
