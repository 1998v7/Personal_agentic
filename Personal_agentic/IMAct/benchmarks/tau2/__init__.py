"""τ²-bench 的 bench 专属部分:airline / retail / telecom 三个 domain 的环境、工具、任务与评分标准。

交互逻辑(用户模拟器、对话循环、agent)由 ``IMAct.harness`` 提供,评分由 ``IMAct.evaluation.tau2`` 提供;
本目录只放 τ² 自己的数据与环境,后期整体删除即可,不影响通用 harness。
"""
