"""env / 工具共用的小函数:哈希、时间、模糊匹配与重排。"""

import hashlib
import json
import os
import random
from datetime import datetime
from typing import Dict, Union

from thefuzz import fuzz, process

from ..config import DEFAULT_LANGUAGE


def get_hash(obj: Union[dict, str]) -> str:
    """dict / str 的 sha256 十六进制摘要。"""
    if isinstance(obj, dict):
        hash_string = json.dumps(obj, sort_keys=True, default=str, ensure_ascii=False)
    else:
        hash_string = obj
    return hashlib.sha256(hash_string.encode()).hexdigest()


def get_now(format: str = "%Y%m%d_%H%M%S") -> str:
    return format_time(datetime.now(), format=format)


def format_time(time: datetime, format: str = "%Y%m%d_%H%M%S") -> str:
    return time.strftime(format)


def str_to_datetime(time: str) -> datetime:
    return datetime.strptime(time, "%Y-%m-%d %H:%M:%S")


def get_weekday(date: str, language: str = None) -> str:
    language = language or DEFAULT_LANGUAGE
    weekday = str_to_datetime(date).weekday()
    if language == "english":
        return ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][weekday]
    return f"星期{'一二三四五六日'[weekday]}"


def check_time_format(time: str, format="%Y-%m-%d %H:%M:%S") -> bool:
    try:
        datetime.strptime(time, format)
        return True
    except ValueError:
        return False


def check_date_format(date: str) -> bool:
    try:
        datetime.strptime(date, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def rerank(keywords: str, docs: Dict[str, str], with_score: bool = False):
    candidates = [doc for doc in docs.values()]
    doc_dict_reverse = {val: key for key, val in docs.items()}
    docs_sorted = process.extract(keywords, candidates, limit=None, scorer=fuzz.partial_ratio)
    if with_score:
        id_doc_sorted = [(doc_dict_reverse[doc], doc, score) for doc, score in docs_sorted]
    else:
        id_doc_sorted = [(doc_dict_reverse[doc], doc) for doc, _ in docs_sorted]

    # 消融开关:打乱搜索结果,去掉位置偏置
    if os.environ.get("VITA_SHUFFLE_SEARCH", "").lower() in ("1", "true", "yes"):
        random.shuffle(id_doc_sorted)

    return id_doc_sorted


def fuzzy_match(x: str, y: str) -> bool:
    return fuzz.partial_ratio(x, y) >= 40


def fuzzy_ratio_match(x: str, y: str) -> bool:
    return fuzz.ratio(x, y) >= 20
