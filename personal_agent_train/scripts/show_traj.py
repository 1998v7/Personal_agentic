"""打印某条轨迹 response 的头尾。用法:python show_traj.py <session 目录> <轨迹序号> [头字符数] [尾字符数]"""
import json, sys
from pathlib import Path
from transformers import AutoTokenizer

d = Path(sys.argv[1]); idx = int(sys.argv[2]); head = int(sys.argv[3]) if len(sys.argv) > 3 else 1500; tail = int(sys.argv[4]) if len(sys.argv) > 4 else 600
tok = AutoTokenizer.from_pretrained(str(Path.home() / "pat/models/Qwen3.5-2B"))
rec = [json.loads(l) for l in (d / "trajectories.jsonl").open(encoding="utf-8")][idx]["trajectory"]
text = tok.decode(rec["response_ids"], skip_special_tokens=False)
prompt = tok.decode(rec["prompt_ids"], skip_special_tokens=False)
print("PROMPT TAIL:\n", prompt[-800:], "\n----- RESPONSE HEAD:\n", text[:head], "\n----- RESPONSE TAIL:\n", text[-tail:])
