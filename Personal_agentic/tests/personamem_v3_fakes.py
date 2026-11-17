"""PersonaMem-v3 对拍用的确定性假模型 / 假裁判 / 假记忆。上游与 IMAct 两边共用;输出只取决于 prompt 文本。

假模型的回答故意覆盖多种形态(非 JSON、空答案、各题型的 JSON 字段、合法/不合法的排序),假裁判按 prompt 里 JSON 示例
的字段名和取值类型造数据,目的是让评分代码走到尽可能多的分支,而不是给出"合理"的答案。
"""
import hashlib
import json
import re

BOOL_HINTS = ("violat", "fail", "leak", "hard", "met", "passed", "correct", "should_act", "warned", "is_", "has_", "used", "stale", "detected", "invoked", "match")
TEXT_HINTS = ("reason", "justif", "evidence", "rationale", "note", "extracted", "explanation", "quote", "summary", "comment", "why", "description", "label", "text", "topic")
_KEY_RE = re.compile(r'"([A-Za-z_][A-Za-z0-9_]*)"\s*:\s*(\S)')
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'#-]{2,}")


def h(*parts) -> int:
    return int(hashlib.sha256(json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest(), 16)


def words_from(prompt: str, n: int, k: int) -> str:
    ws = _WORD_RE.findall(prompt)
    if not ws:
        return "ok"
    return " ".join(ws[(n >> (i * 3)) % len(ws)] for i in range(k))


def fake_answer(prompt: str) -> str:
    n = h("answer", prompt)
    variant = n % 7
    text = words_from(prompt, n, 12 + n % 30)
    if variant == 0:
        return text  # 非 JSON
    if variant == 1:
        return ""  # 空答案
    k = (10, 12, 16)[n % 3]
    perm = list(range(k))
    for i in range(k - 1, 0, -1):  # 确定性洗牌
        j = (n >> (i % 60)) % (i + 1)
        perm[i], perm[j] = perm[j], perm[i]
    if variant == 2:
        perm = perm[: 1 + n % 4]  # 不完整的排序
    obj = {
        "response": text, "reasoning": words_from(prompt, n >> 7, 6), "ranked_indices": perm, "ranked_indexes": perm,
        "should_act": bool(n % 2), "action_class": ("dm", "post", "none", "reminder")[n % 4], "content": text,
        "evidence_cited": words_from(prompt, n >> 11, 3) if n % 5 else "", "warned": bool((n >> 3) % 2), "recommendation": text,
        "hashtags": ["#" + w.lower().strip("#") for w in words_from(prompt, n >> 13, 4).split()], "final_answer": text,
        "title": words_from(prompt, n >> 17, 5), "caption": text, "summary": text, "tool_intended": "create_post",
        "tool_called": "send_dm", "message": text,
    }
    if variant == 3:  # 只保留部分字段
        obj = {k2: v for i, (k2, v) in enumerate(obj.items()) if (n >> i) % 2}
    body = json.dumps(obj, ensure_ascii=False)
    return f"```json\n{body}\n```" if variant != 4 else body


def fake_judge(prompt: str) -> str:
    n = h("judge", prompt)
    if n % 23 == 0:
        return "I cannot judge this."  # 偶发的坏输出
    out = {}
    for m in _KEY_RE.finditer(prompt):
        key, first = m.group(1), m.group(2)
        if key in out or len(out) >= 80:
            continue
        v = h(n, key)
        low = key.lower()
        if first == "[":
            out[key] = [] if v % 3 else [f"item {words_from(prompt, v, 2)}"]
        elif first == "{":
            out[key] = {}
        elif first in "tf":
            out[key] = bool(v % 3 == 0)
        elif first == '"' or any(x in low for x in TEXT_HINTS):
            out[key] = f"because {words_from(prompt, v, 5)}"
        elif any(b in low for b in BOOL_HINTS) and v % 2:
            out[key] = bool(v % 3 == 0)
        else:
            out[key] = v % 11  # 0-10;0-3 的维度会被评分代码夹到合法范围
    return "```json\n" + json.dumps(out, ensure_ascii=False) + "\n```"


class FakeClient:
    """``query_llm(prompt) -> str`` 形态(上游 QueryLLM / IMAct 适配器都满足)。记录收到的每个 prompt。"""

    def __init__(self, role: str):
        self.role = role
        self.prompts: list[str] = []

    def query_llm(self, prompt, *args, **kwargs):
        self.prompts.append(prompt)
        return fake_judge(prompt) if self.role == "judge" else fake_answer(prompt)


def fake_memory(user_id: str) -> str:
    """一个用户一份固定的记忆文本(四个官方标题),足够让 prompt 里出现记忆块。"""
    return (f"## Who they are\n- persona {user_id}; enjoys {words_from(str(user_id) * 9, h('mem', user_id), 4)}\n"
            "## Interests & preferences\n- basketball highlights [x3] · last 04/06\n- spicy noodles\n"
            "## People & places\n- close friend Sam; lives in Atlanta\n## Currently active\n- planning a weekend trip")
