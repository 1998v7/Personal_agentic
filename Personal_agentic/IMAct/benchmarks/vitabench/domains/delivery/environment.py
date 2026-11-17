from typing import Optional

from ...domains.delivery.data_model import DeliveryDB
from ...domains.delivery.tools import DeliveryTools
from ...environment.environment import Environment


def get_environment(
        db: Optional[DeliveryDB] = None,
        language: str = None,
) -> Environment:
    # 设置全局语言配置
    if language is not None:
        from ...utils.schema_utils import set_global_language
        set_global_language(language)
    
    if db is None:
        db = {}

    db_dict = {k: v for k, v in db.items() if
               k in ["stores", "orders", "time", "user_id", "weather", "location", "user_historical_behaviors"]}
    if "stores" not in db_dict:
        db_dict["stores"] = {}
    if "orders" not in db_dict:
        db_dict["orders"] = {}

    db = DeliveryDB.model_validate(db_dict)
    tools = DeliveryTools(db)
    return Environment(
        domain_name="delivery",
        tools=tools,
    )

