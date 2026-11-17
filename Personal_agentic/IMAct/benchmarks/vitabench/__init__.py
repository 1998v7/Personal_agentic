"""VitaBench 2.0 personalization 的 bench 专属部分:任务数据、env / DB / 工具、prompts 与入口。

交互逻辑(用户模拟器、agent、对话循环)由 ``IMAct.harness`` 提供,评分由 ``IMAct.evaluation.vitabench`` 提供;
本目录只放 VitaBench 自己的数据与环境,后期整体删除即可,不影响通用 harness。
"""
