from typing import Optional

from ...domains.instore.data_model import InStoreDB
from ...domains.instore.tools import InStoreTools
from ...environment.environment import Environment


def get_environment(
        db: Optional[InStoreDB] = None,
        language: str = None,
) -> Environment:
    # 设置全局语言配置
    if language is not None:
        from ...utils.schema_utils import set_global_language
        set_global_language(language)
    
    if db is None:
        db = {}

    db_dict = {k: v for k, v in db.items() if
               k in ["shops", "orders", "reservations", "books", "time", "user_id", "weather", "location",
                     "user_historical_behaviors"]}
    if "shops" not in db_dict:
        db_dict["shops"] = {}
    if "books" not in db_dict:
        db_dict["books"] = {}
    if "reservations" not in db_dict:
        db_dict["reservations"] = {}
    if "orders" not in db_dict:
        db_dict["orders"] = {}

    db = InStoreDB.model_validate(db_dict)
    tools = InStoreTools(db)
    return Environment(
        domain_name="instore",
        tools=tools,
    )

