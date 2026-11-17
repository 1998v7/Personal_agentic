"""τ² 的 agent / 用户模拟器 prompt。模板文本从上游代码原样导出为 ``prompt_templates/``。

英文 cell 与旧版口径一致;其他语言在此基础上增加:用户侧的"全程使用该语言"指令和收尾提醒,
agent 侧的语言说明(以及本地化数据库下对工具参数的说明),并把 policy 里的转人工提示换成本语言的一句。
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from string import Template

from .multilingual import Cell, Persona
from .data_model.tasks import Task

_DIR = Path(__file__).parent / "prompt_templates"

_CROSS_LINGUAL_TOOLS = "Tool names, argument keys and enumerated values follow the tool schema. "
_REGIONAL = ("Use the native-script names, addresses and other localized values supplied by the customer and tools. "
             "Keep IDs, codes, email addresses, phone numbers and postal codes in their specified format.")
_CANONICAL = ("Use the canonical database values for tool arguments, including English names and addresses. "
              "Keep IDs, codes, email addresses, phone numbers and postal codes in their specified format.")
_CURRENCY_NOTE = ("\nScenario monetary amounts are in {currency}. Express that currency in the call language; "
                  "do not convert amounts or substitute another currency.")


@lru_cache(maxsize=None)
def _text(name: str) -> str:
    return (_DIR / name).read_text(encoding="utf-8")


@lru_cache(maxsize=None)
def _json(name: str):
    return json.loads(_text(name))


# ---------------------------------------------------------------------- agent
def agent_language_guidance(cell: Cell, persona: Persona) -> str:
    """语言说明:人设的地域背景 + 语言子句;跨语言时再补一句关于工具参数取值的约定。"""
    clause = "\n".join(s for s in (persona.agent_background_clause, cell.pack["agent_language_clause"]) if s)
    if cell.cross_lingual:
        clause += "\n" + _CROSS_LINGUAL_TOOLS + (_REGIONAL if cell.regional is not None else _CANONICAL)
    return clause


def localized_transfer_policy(policy: str, language: str) -> str:
    """policy 里"转人工"那句固定台词,换成通话语言的一句。"""
    lines = _json("transfer_lines.json")
    english = lines["english"]
    if language == "en":
        return policy
    if language in lines["localized"]:
        return policy.replace(english, lines["localized"][language])
    return policy.replace(
        f"send the message '{english}' to the user",
        "tell the user in the call's language that they are being transferred to a human agent and should hold")


def agent_domain_policy(policy: str, cell: Cell, persona: Persona) -> str:
    guidance = agent_language_guidance(cell, persona)
    return localized_transfer_policy(policy, cell.language) + ("\n\n" + guidance if guidance else "")


def agent_system_prompt(domain_policy: str) -> str:
    return _text("agent_system_prompt.txt").format(agent_instruction=_text("agent_instruction.txt"), domain_policy=domain_policy)


# ---------------------------------------------------------------------- 用户模拟器
def _directive(persona: Persona) -> str:
    if persona.origin is None:
        return ""
    return Template(_text("language_directive.txt")).substitute(
        **_json("medium_text.json"), language=persona.display_language, origin=persona.origin, variety=persona.variety)


def _reminder(persona: Persona) -> str:
    if persona.origin is None:
        return ""
    return Template(_text("language_reminder.txt")).substitute(**_json("medium_text.json"), language=persona.display_language)


def persona_guidelines(cell: Cell, persona: Persona) -> str:
    """人设描述(文本模式):性别、各条语言风格、通用与 domain 的本地化指引、本通话模式的本地化指引,以空行连接。"""
    binding = cell.binding
    user_guidelines = "\n\n".join(s for s in (cell.pack.get("user_guidelines"), binding.get("user_guidelines")) if s)
    if cell.regional is not None:
        user_guidelines += _CURRENCY_NOTE.format(currency=cell.regional["settings"]["currency_code"])
    sections = [persona.gender_guidance, *persona.pragmatics_clauses, user_guidelines, binding.get("text_guidelines") or ""]
    return "\n\n".join(s for s in sections if s)


def user_system_prompt(cell: Cell, task: Task, *, use_tools: bool) -> str:
    """用户模拟器的 system prompt:通用守则(含语言指令)+ 该 task 的人设 + 任务场景 + 语言提醒。"""
    persona = cell.persona(task)
    directive = _directive(persona)
    template = Template(_text("user_guidelines_text_tools.txt" if use_tools else "user_guidelines_text.txt"))
    guidelines = template.substitute(language_directive=directive + "\n\n" if directive else "")
    text = persona_guidelines(cell, persona)
    guidelines = guidelines.replace("<PERSONA_GUIDELINES>", f"\n\n{text}\n" if text else "")
    prompt = _text("user_system_prompt.txt").format(global_user_sim_guidelines_with_persona=guidelines, instructions=str(task.user_scenario))
    reminder = _reminder(persona)
    return prompt + "\n\n" + reminder if reminder else prompt
