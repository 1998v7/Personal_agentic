"""VitaBench f60169e8 的公开回执解析。只读文本，不执行 repr 中的代码。"""

import ast
import json
import re
from importlib.resources import files

CATALOG = json.loads(files("imact").joinpath("fixtures/vitabench_tools_catalog.json").read_text())
TOOLS = CATALOG["tools"]
MEMORY_TOOLS = {"read_preference_memory", "query_preference_memory"}

# 上游有别名：AttractionProduct.attraction_id 实际是 ticket_id。
RECORDS = {
    "Store": ("store", "store_id"),
    "StoreProduct": ("product", "product_id"),
    "Shop": ("shop", "shop_id"),
    "ShopProduct": ("product", "product_id"),
    "Hotel": ("hotel", "hotel_id"),
    "HotelProduct": ("product", "room_id"),
    "Attraction": ("attraction", "attraction_id"),
    "AttractionProduct": ("product", "attraction_id"),
    "Flight": ("flight", "flight_id"),
    "FlightProduct": ("product", "seat_id"),
    "Train": ("train", "train_id"),
    "TrainProduct": ("product", "seat_id"),
    "Taxi": ("taxi", "taxi_id"),
    "TaxiProduct": ("product", "product_id"),
    "Order": ("order", "order_id"),
    "BookInfo": ("book", "book_id"),
    "ReservationInfo": ("reservation", "reservation_id"),
}
TRANSACTIONS = {"order", "book", "reservation"}
PRODUCT_PARENTS = {
    "StoreProduct": "store",
    "ShopProduct": "shop",
    "HotelProduct": "hotel",
    "AttractionProduct": "attraction",
    "FlightProduct": "flight",
    "TrainProduct": "train",
    "TaxiProduct": "taxi",
}


def product_parent(record):
    """仅使用已解析商品的类型化外键；不从文本、ID 前缀或工具实参猜测。"""
    kind = PRODUCT_PARENTS.get(record.get("record_type"))
    if not kind or record.get("parent_kind") in TRANSACTIONS:
        return None
    # 上游 AttractionProduct.attraction_id 实为票号；只有嵌套详情的父实体可定景点。
    if kind == "attraction" and record.get("parent_kind") != kind:
        return None
    key = kind + "_id"
    value = record.get(key)
    if not isinstance(value, str) or not value.strip() or value == record.get("item_id"):
        return None
    return kind, key, value


OBJECT = re.compile(r"\b(" + "|".join(RECORDS) + r")\(")
FIELD = re.compile(r"\s*([A-Za-z_]\w*)\s*[:=]")
NUMBERS = {
    "price",
    "total_price",
    "quantity",
    "score",
    "star_rating",
    "book_price",
    "customer_count",
    "shipping_time",
    "estimated_price",
    "ticket_price",
    "latitude",
    "longitude",
}


def _quote_start(text, i):
    # 非引号包围的店名可以包含撇号；列表/字典中的引号则需要跳过。
    before = text[:i].rstrip()
    return not before or before[-1] in "[{:=(,"


def fields(body: str) -> dict:
    """在最外层逗号处分字段，保留名字括号、列表及嵌套 Location。"""
    starts, depth, quote = [], [], None
    i = 0
    while i < len(body):
        ch = body[i]
        if quote:
            if ch == "\\":
                i += 2
                continue
            if ch == quote:
                quote = None
        elif ch in "\"'" and _quote_start(body, i):
            quote = ch
        elif ch in "([{":
            depth.append(ch)
        elif ch in ")]}":
            if depth:
                depth.pop()
        elif not depth and (i == 0 or ch == ","):
            offset = i if i == 0 else i + 1
            match = FIELD.match(body, offset)
            if match:
                starts.append((match.group(1), match.end(), i))
        i += 1
    result = {}
    for j, (key, begin, _) in enumerate(starts):
        end = starts[j + 1][2] if j + 1 < len(starts) else len(body)
        value = body[begin:end].strip().rstrip(",").strip()
        if key == "products":
            # 读入历史订单时上游还会输出字典列表，而非 Product 实例 repr。
            try:
                products = ast.literal_eval(value)
                if isinstance(products, list) and all(isinstance(p, dict) for p in products):
                    json.dumps(products, allow_nan=False)
                    result[key] = products
            except (ValueError, SyntaxError, TypeError):
                pass
            continue
        if value.startswith("Location(") and value.endswith(")"):
            result[key] = fields(value[9:-1])
            continue
        if (
            key in NUMBERS
            or value in {"True", "False", "None"}
            or value.startswith(("[", "{", '"', "'"))
        ):
            try:
                parsed = ast.literal_eval(value)
                # 只保存 JSON 可序列化的标量/容器。
                json.dumps(parsed, allow_nan=False)
                value = parsed
            except (ValueError, SyntaxError, TypeError):
                pass
        result[key] = value
    for key in (
        "location",
        "departure_location",
        "arrival_location",
        "departure_airport_location",
        "arrival_airport_location",
        "departure_station_location",
        "arrival_station_location",
    ):
        value = result.get(key)
        if not isinstance(value, str):
            continue
        match = re.fullmatch(r"address=(.+) longitude=([-\d.]+) latitude=([-\d.]+)", value)
        if match:
            try:
                result[key] = {
                    "address": ast.literal_eval(match[1]),
                    "longitude": float(match[2]),
                    "latitude": float(match[3]),
                }
            except (ValueError, SyntaxError):
                pass
        else:
            match = re.fullmatch(r"(.+) longitude:([-\d.]+)", value)
            if match and "latitude" in result:
                try:
                    result[key] = {
                        "address": match[1],
                        "longitude": float(match[2]),
                        "latitude": float(result["latitude"]),
                    }
                    del result["latitude"]
                except (ValueError, TypeError):
                    pass
    return result


def parse_records(text: str) -> list[dict]:
    """支持详情嵌套、列表换行及上游未闭合的 Order/BookInfo/ReservationInfo。"""
    matches = list(OBJECT.finditer(text))
    records = []
    for index, match in enumerate(matches):
        start, begin = match.start(), match.end()
        depth, quote, end = 1, None, len(text)
        for i in range(begin, len(text)):
            ch = text[i]
            if quote:
                if ch == quote and text[i - 1] != "\\":
                    quote = None
                continue
            if ch in "\"'" and _quote_start(text, i):
                quote = ch
            elif ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
            elif ch == "\n" and depth == 1:
                following = OBJECT.match(text, i + 1)
                if following and RECORDS[following.group(1)][0] != "product":
                    end = i
                    break
        body = text[begin:end]
        if body.endswith(")"):
            body = body[:-1]
        values = fields(body)
        kind, id_field = RECORDS[match.group(1)]
        item_id = values.get(id_field)
        if not isinstance(item_id, str) or not item_id:
            continue
        parent = next((r for r in reversed(records) if r["_start"] < start < r["_end"]), None)
        record = {
            **values,
            "item_id": item_id,
            "kind": kind,
            "record_type": match.group(1),
            "raw": text[start:end],
            "_start": start,
            "_end": end,
        }
        if kind == "product":
            record["product_id"] = item_id
            if parent:
                record["parent_id"] = parent["item_id"]
                record["parent_kind"] = parent["kind"]
                if parent["kind"] not in TRANSACTIONS:
                    record[parent["kind"] + "_id"] = parent["item_id"]
        if kind in TRANSACTIONS:
            record["entity_id"] = item_id
        records.append(record)
    for record in records:
        if record["kind"] in TRANSACTIONS:
            children = [r for r in records if r.get("parent_id") == record["item_id"]]
            if children:
                record["products"] = [
                    {k: v for k, v in r.items() if not k.startswith("_") and k != "raw"}
                    for r in children
                ]
                record["product_ids"] = [r["product_id"] for r in children]
        # 候选自身 raw 不重复包含所有子商品；订单原文仍完整保留。
        elif record["kind"] != "product":
            record["raw"] = record["raw"].split("products=", 1)[0].rstrip(", ")
    return [{k: v for k, v in r.items() if not k.startswith("_")} for r in records]


def response_failure(content) -> bool:
    if isinstance(content, dict) and content.get("error"):
        return True
    if not isinstance(content, str):
        return False
    text = content.strip()
    return bool(
        text.startswith(
            (
                "Error:",
                "Modification failed",
                "Cancellation failed",
                "No available rooms",
                "Insufficient ",
                "The attraction does not ",
                "The flight does not ",
                "The train does not ",
                "The taxi service does not ",
            )
        )
        or text in {"Order already exists", "Order not found", "Order status must be unpaid"}
        or re.search(r"is not in `unpaid` status\.?$", text)
        or re.search(r"(?:BookInfo|ReservationInfo) .+ (?:not found|already exists)\.?$", text)
    )
