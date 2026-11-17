from typing import Optional

from ...domains.ota.data_model import OTADB
from ...domains.ota.tools import OTATools
from ...environment.environment import Environment


def get_environment(
        db: Optional[OTADB] = None,
        language: str = None,
) -> Environment:
    # 设置全局语言配置
    if language is not None:
        from ...utils.schema_utils import set_global_language
        set_global_language(language)
    
    if db is None:
        db = {}

    db_dict = {k: v for k, v in db.items() if
               k in ["orders", "hotels", "attractions", "flights", "trains", "taxis", "time", "user_id", "weather", "location",
                     "user_historical_behaviors"]}
    if "hotels" not in db_dict:
        db_dict["hotels"] = {}
    if "attractions" not in db_dict:
        db_dict["attractions"] = {}
    if "flights" not in db_dict:
        db_dict["flights"] = {}
    if "trains" not in db_dict:
        db_dict["trains"] = {}
    if "taxis" not in db_dict:
        db_dict["taxis"] = {}
    if "orders" not in db_dict:
        db_dict["orders"] = {}

    db = OTADB.model_validate(db_dict)
    tools = OTATools(db)
    return Environment(
        domain_name="ota",
        tools=tools,
    )

