#!/usr/bin/env bash
# 看最近一次冒烟的样本:show_smoke.sh <session 前缀> [起始行] [行数]
d=$(ls -dt $HOME/pat/smoke/$1*/ | head -1)
echo "$d"
grep -n "^========\|^########" "$d/readable.txt" | head -${4:-40}
sed -n "${2:-1},+${3:-80}p" "$d/readable.txt"
