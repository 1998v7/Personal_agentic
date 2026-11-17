"""harness agent 与 VitaBench 接入的测试。不需要数据集、不需要真实 LLM:用执行层的 DemoModel 当脚本化模型,
按对话循环的调用方式逐条驱动 agent。"""

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest import mock

import httpx

from tests.exec_helpers import RECEIPTS, DemoProvider, make_runtime, no_sleep  # 先导入:会设置 OPENAI_API_KEY 占位值

from IMAct.benchmarks.vitabench.data_model.personalization_task import SubTask
from IMAct.benchmarks.vitabench.domains import build_environment
from IMAct.benchmarks.vitabench.personalization import agent_system_prompt, episode_timestamp
from IMAct.exec_layer.config import MemoryRouting
from IMAct.exec_layer.memory_mps import MpsMemoryClient, MpsMemoryConfig
from IMAct.harness import EpisodeSpec, HarnessAgent, TerminationReason, run_episode
from IMAct.harness.conversation import ConversationResult
from IMAct.harness.messages import MultiToolMessage, ToolMessage, UserMessage
from IMAct.models.base import ChatResponse

POLICY = "你是外卖助手。当前时间 2026-03-02 12:00:00。\n\n## 当前用户基础信息\n以下是该用户的账户注册信息,可直接使用:\n- user_id: A891207\n- address: 演示办公地址"


def subtask(**kw) -> SubTask:
    return SubTask(
        subtask_id=kw.pop("subtask_id", "sub_A891207_1"),
        domain=kw.pop("domain", "delivery"),
        instruction=kw.pop("instruction", "忙得脚不沾地,中午饭就帮我点个外卖来单位算了"),
        environment=kw.pop("environment", {"time": "2026-03-02 12:00:00"}),
        **kw,
    )


def spec_for(st: SubTask | None = None, *, memory_cfg: MemoryRouting | None = None, **kw) -> EpisodeSpec:
    """VitaBench 的 subtask → harness 的 EpisodeSpec,时间口径与 PersonalizationRunner.prepare 一致。"""
    st = st or subtask()
    fields = dict(
        bench="vitabench", user_id="A891207", episode_id=st.subtask_id, domain=st.domain, instruction=st.instruction,
        timestamp=episode_timestamp(st, memory_cfg or MemoryRouting()), system_prompt=POLICY,
        tools=build_environment(st.domain, st.environment, "chinese").tool_schemas(),
    )
    fields.update(kw)
    return EpisodeSpec(**fields)


def gateway(context="近期偏好:辛香,优先30分钟内配送。", seen=None):
    def handler(req: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(json.loads(req.content))
        return httpx.Response(200, json={"context": context})

    return MpsMemoryClient(MpsMemoryConfig(url="https://gw.example/x"), client=httpx.AsyncClient(transport=httpx.MockTransport(handler)), sleep=no_sleep)


def make_agent(provider=None, *, memory=True, seen=None, **cfg):
    client = gateway(seen=seen) if memory else None
    rt = make_runtime(provider, memory_client=client, **cfg)
    return HarnessAgent(runtime=rt, run_id="run1"), rt


def env_reply(call, content=None) -> ToolMessage:
    """模拟环境对一次工具调用的回执(id 与 ToolCall.id 对应)。"""
    return ToolMessage(id=call.id, name=call.name, role="tool", content=content if content is not None else RECEIPTS[call.name])


def drive(agent, first_user_text="帮我点个午餐外卖", max_steps=20):
    """按对话循环的方式驱动 agent:agent -> env -> agent …,直到助手给出 final 回复。"""
    state = agent.get_init_state()
    msg = UserMessage(role="user", content=first_user_text)
    out = []
    for _ in range(max_steps):
        reply, state = agent.generate_next_message(msg, state)
        out.append(reply)
        if reply.is_tool_call():
            tms = [env_reply(c) for c in reply.tool_calls]
            msg = tms[0] if len(tms) == 1 else MultiToolMessage(role="tool", tool_messages=tms)
        elif (reply.raw_data or {}).get("imact", {}).get("mode") == "final":
            return out, state
        else:
            msg = UserMessage(role="user", content="随便,你看着办就行。")
    raise AssertionError("conversation did not finish")


class AgentBridgeTests(unittest.TestCase):
    def test_required_prefetch_failure_stops_before_action(self):
        provider = DemoProvider()
        agent, rt = make_agent(provider)
        rt.cfg.memory.require_prefetch = True
        try:
            with mock.patch.object(agent, "_prefetch", return_value={"status": "error", "context": "", "error": "timeout"}):
                with self.assertRaisesRegex(RuntimeError, "require_prefetch"):
                    agent.begin_episode(spec_for())
            self.assertIsNone(agent._ctx)
            self.assertEqual(provider.requests, [])
            # 同一入口在 Memory 恢复后可以建立有效会话。
            ctx = agent.begin_episode(spec_for())
            self.assertEqual(ctx.memory["status"], "ok")
        finally:
            rt.close()

    def test_full_episode_conversation(self):
        seen = []
        agent, rt = make_agent(seen=seen)
        try:
            agent.begin_episode(spec_for())
            out, state = drive(agent)
            task_id = state.harness_task_id
            agent.end_episode("user_stop")
            snap = rt.outer.harness.store.inspect(task_id)
        finally:
            rt.close()
        self.assertTrue(task_id.startswith("run1-A891207-sub_A891207_1-"))
        names = [c.name for m in out if m.is_tool_call() for c in m.tool_calls]
        self.assertEqual(names, ["delivery_store_search_recommand", "delivery_product_search_recommand", "create_delivery_order", "pay_delivery_order"])
        self.assertEqual(out[-1].content, "离线演示：已收到合成订单的支付成功回执。")
        self.assertEqual(out[-1].raw_data["imact"]["outcome"], "completed")
        # 网关收到两次请求:start 时的预取 + 模型主动 query_memory。两次都用同一个 query_id / user_id / 日期
        self.assertEqual(len(seen), 2)
        tasks = [json.loads(b["tensorFeatures"]["task"]["stringValues"][0]) for b in seen]
        self.assertEqual(
            tasks[0], {"query_id": "sub_A891207_1", "user_query": "忙得脚不沾地,中午饭就帮我点个外卖来单位算了", "timestamp": "2026-03-02"}
        )  # 预取:user_query=instruction(Memory 文档约定);timestamp 取日期部分
        self.assertEqual(tasks[1]["query_id"], "sub_A891207_1")
        self.assertEqual(tasks[1]["user_query"], "本次午餐相关的口味和配送偏好是什么？")  # 主动查询:默认传模型提的问题
        self.assertEqual(tasks[1]["timestamp"], "2026-03-02")
        self.assertTrue(all(b["tensorFeatures"]["user_id"]["stringValues"] == ["A891207"] for b in seen))
        # terminate 已送达执行层
        self.assertEqual(snap["workspace"]["lifecycle"] if "workspace" in snap else snap.get("lifecycle"), "stopped")

    def test_start_event_carries_memory_domain_tools_and_clean_system_prompt(self):
        provider = DemoProvider()
        agent, rt = make_agent(provider)
        try:
            agent.begin_episode(spec_for())
            drive(agent)
            agent.end_episode()
        finally:
            rt.close()
        ctx = json.loads(provider.requests[0].messages[-1]["content"])
        self.assertEqual(ctx["latest_event"]["type"], "start")
        self.assertEqual(ctx["latest_event"]["domain"], "delivery")
        self.assertEqual(ctx["user_memory"][0]["status"], "ok")
        self.assertIn("辛香", ctx["user_memory"][0]["context"])
        self.assertTrue(any(t["function"]["name"] == "create_delivery_order" for t in ctx["available_tools"]))
        policy = provider.requests[0].messages[1]["content"]
        self.assertIn("外卖助手", policy)
        self.assertIn("演示办公地址", policy)  # 账户信息仍在
        self.assertEqual(provider.requests[0].messages[1]["role"], "system")

    def test_replay_with_deepcopied_state_is_idempotent(self):
        """对话循环在无效消息时会用原状态的深拷贝重试 generate_next_message;必须得到同样的结果且不再调模型。"""
        provider = DemoProvider()
        agent, rt = make_agent(provider)
        try:
            agent.begin_episode(spec_for())
            state0 = agent.get_init_state()
            msg = UserMessage(role="user", content="帮我点个午餐外卖")
            r1, s1 = agent.generate_next_message(msg, deepcopy(state0))
            n = len(provider.requests)
            r2, s2 = agent.generate_next_message(msg, deepcopy(state0))  # 同一个原状态再来一次
            tool_msg = env_reply(r1.tool_calls[0])
            a1, _ = agent.generate_next_message(tool_msg, deepcopy(s1))
            m = len(provider.requests)
            a2, _ = agent.generate_next_message(tool_msg, deepcopy(s1))
        finally:
            rt.close()
        self.assertEqual([c.model_dump() for c in r1.tool_calls], [c.model_dump() for c in r2.tool_calls])
        self.assertEqual(len(provider.requests), m)  # 重放两次都没有新的模型请求
        self.assertEqual(n, 2)  # start 触发 query_memory 往返共 2 次
        self.assertEqual(a1.content, a2.content)
        self.assertEqual(s1.harness_rev, s2.harness_rev)

    def test_parallel_tool_calls_return_multi_tool_message(self):
        decisions = iter([
            {"task_changes": {"goal": "查两家店"}, "working_notes": [], "next_action": {"type": "call_tools", "calls": [
                {"name": "delivery_store_search_recommand", "arguments": {"keywords": ["烧烤"]}},
                {"name": "delivery_store_search_recommand", "arguments": {"keywords": ["快餐"]}}]}},
            {"task_changes": {}, "working_notes": [], "next_action": {"type": "reply_user", "mode": "final", "content": "找到了", "outcome": "partial"}},
        ])

        class P(DemoProvider):
            async def chat(self, request):
                self.requests.append(request)
                return ChatResponse(content=json.dumps(next(decisions), ensure_ascii=False), finish_reason="stop", usage={"prompt_tokens": 3, "completion_tokens": 2})

        agent, rt = make_agent(P(), memory=False)
        try:
            agent.begin_episode(spec_for())
            out, state = drive(agent)
        finally:
            rt.close()
        self.assertEqual(len(out[0].tool_calls), 2)
        self.assertEqual(len({c.id for c in out[0].tool_calls}), 2)
        self.assertEqual(state.harness_pending, {})  # 回执都已收齐
        self.assertEqual(out[-1].content, "找到了")

    def test_messages_always_carry_cost_and_usage(self):
        agent, rt = make_agent(DemoProvider(usage={"prompt_tokens": 100, "completion_tokens": 20}))
        try:
            agent.begin_episode(spec_for())
            out, _ = drive(agent)
        finally:
            rt.close()
        for m in out:
            self.assertIsInstance(m.cost, float)  # 成本统计遇到 None 会放弃整个统计
            self.assertGreater(m.usage["prompt_tokens"], 0)
        self.assertEqual(out[0].usage, {"prompt_tokens": 200, "completion_tokens": 40})  # start:query_memory + 决策 两次调用汇总

    def test_memory_failure_does_not_block_the_task(self):
        def down(req):
            return httpx.Response(500, text="boom")

        client = MpsMemoryClient(MpsMemoryConfig(url="https://gw.example/x"), client=httpx.AsyncClient(transport=httpx.MockTransport(down)), sleep=no_sleep)
        rt = make_runtime(DemoProvider(), memory_client=client)
        agent = HarnessAgent(runtime=rt)
        try:
            agent.begin_episode(spec_for())
            self.assertEqual(agent._ctx.memory["status"], "error")
            out, _ = drive(agent)
        finally:
            rt.close()
        self.assertEqual(out[-1].raw_data["imact"]["outcome"], "completed")

    def test_no_memory_configured_sends_not_requested(self):
        provider = DemoProvider()
        agent, rt = make_agent(provider, memory=False)
        try:
            agent.begin_episode(spec_for())
            self.assertEqual(agent._ctx.memory, {"status": "not_requested"})
        finally:
            rt.close()

    def test_state_from_another_episode_is_rejected(self):
        agent, rt = make_agent(memory=False)
        try:
            agent.begin_episode(spec_for())
            stale = agent.get_init_state()
            agent.begin_episode(spec_for(subtask(subtask_id="sub_A891207_2")))
            with self.assertRaises(RuntimeError):
                agent.generate_next_message(UserMessage(role="user", content="x"), stale)
        finally:
            rt.close()

    def test_get_init_state_requires_begin_episode(self):
        agent, rt = make_agent(memory=False)
        try:
            with self.assertRaises(RuntimeError):
                agent.get_init_state()
        finally:
            rt.close()

    def test_each_episode_gets_its_own_task_id(self):
        agent, rt = make_agent(memory=False)
        try:
            a = agent.begin_episode(spec_for()).task_id
            b = agent.begin_episode(spec_for()).task_id  # 同一 subtask 的第二次尝试
        finally:
            rt.close()
        self.assertNotEqual(a, b)

    def test_end_episode_without_start_is_a_noop(self):
        agent, rt = make_agent(memory=False)
        try:
            agent.begin_episode(spec_for())
            agent.end_episode()  # 没有提交过任何事件
            agent.end_episode()  # 重复调用也安全
        finally:
            rt.close()


class SystemPromptTests(unittest.TestCase):
    def test_agent_prompt_has_profile_and_no_native_memory_section(self):
        """记忆只能通过 start.memory 与 query_memory 进入,system prompt 里不能再带 VitaBench 自己的记忆段。"""
        prompt = agent_system_prompt(subtask(), {"user_id": "A891207", "address": ["演示办公地址", "家"]}, "2026-03-02 12:00:00 星期一", "chinese")
        self.assertIn("2026-03-02 12:00:00 星期一", prompt)
        self.assertIn("- address: 演示办公地址、家", prompt)
        self.assertNotIn("User Preference Memory", prompt)

    def test_proactive_subtask_uses_its_own_template(self):
        from IMAct.benchmarks.vitabench.prompts import get_prompts

        prompts = get_prompts("chinese")
        plain = agent_system_prompt(subtask(), {}, "T", "chinese")
        proactive = agent_system_prompt(subtask(skill_tested=["proactive"]), {}, "T", "chinese")
        self.assertTrue(plain.startswith(prompts.personalization_agent_system_prompt.format(time="T")[:40]))
        self.assertNotEqual(plain, proactive)


class CurrentTimeTests(unittest.TestCase):
    def setUp(self):
        # 只保留验证日期来源所需的字段,不依赖本机下载的完整数据集。
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        tasks = Path(directory.name) / "tasks.json"
        tasks.write_text(json.dumps([{"subtasks": [
            {"subtask_id": "sub_V746582_1", "current_time": "2025-06-23"},
            {"subtask_id": "sub_D975700_2", "current_time": "2026-08-19"},
        ]}]), encoding="utf-8")
        patcher = mock.patch.dict("os.environ", {"IMACT_VITABENCH_TASKS": str(tasks)})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_current_time_comes_from_task_file_not_environment_time(self):
        # 真实数据里 sub_V746582_1 的 current_time=2025-06-23,而 environment.time=2025-06-24 11:30:00
        st = subtask(subtask_id="sub_V746582_1", environment={"time": "2025-06-24 11:30:00"})
        self.assertEqual(episode_timestamp(st, MemoryRouting()), "2025-06-23")

    def test_empty_environment_time_still_gets_current_time(self):
        st = subtask(subtask_id="sub_D975700_2", environment={"time": ""})
        self.assertEqual(episode_timestamp(st, MemoryRouting()), "2026-08-19")

    def test_falls_back_to_environment_time_for_unknown_subtask(self):
        st = subtask(subtask_id="not_in_file", environment={"time": "2026-01-02 08:00:00"})
        self.assertEqual(episode_timestamp(st, MemoryRouting()), "2026-01-02")

    def test_local_memory_keeps_raw_value_and_refuses_to_guess(self):
        cfg = MemoryRouting(backend="local", timestamp_mode="date")
        self.assertEqual(episode_timestamp(subtask(subtask_id="sub_V746582_1"), cfg), "2025-06-23")  # 本地模式不截断
        with self.assertRaisesRegex(ValueError, "current_time"):
            episode_timestamp(subtask(subtask_id="not_in_file"), cfg)


class RunEpisodeHookTests(unittest.TestCase):
    """run_episode 在对话前后通知 agent,并把对话结束原因映射成执行层的 terminate.reason。"""

    @staticmethod
    def _result(reason):
        return ConversationResult(messages=[], termination_reason=reason, duration=0.0)

    def test_hooks_surround_the_episode_and_map_the_reason(self):
        cases = [(TerminationReason.USER_STOP, "user_stop"), (TerminationReason.MAX_STEPS, "max_turns"),
                 (TerminationReason.TOO_MANY_ERRORS, "infrastructure_error"), (TerminationReason.AGENT_STOP, "user_stop"),
                 (TerminationReason.INVALID_AGENT_MESSAGE, "infrastructure_error")]
        for term, expected in cases:
            agent = mock.Mock()
            agent.rt.trace = None
            spec = spec_for()
            with mock.patch("IMAct.harness.episode.Conversation") as conv:
                conv.return_value.run.return_value = self._result(term)
                result = run_episode(agent, spec, mock.Mock(), mock.Mock())
            self.assertIs(result.termination_reason, term)
            agent.begin_episode.assert_called_once_with(spec)
            agent.end_episode.assert_called_once_with(expected)

    def test_exception_still_closes_with_infrastructure_error(self):
        agent = mock.Mock()
        with mock.patch("IMAct.harness.episode.Conversation") as conv:
            conv.return_value.run.side_effect = RuntimeError("boom")
            with self.assertRaises(RuntimeError):
                run_episode(agent, spec_for(), mock.Mock(), mock.Mock())
        agent.end_episode.assert_called_once_with("infrastructure_error")

    def test_failed_prefetch_never_reaches_the_conversation(self):
        agent = mock.Mock()
        agent.begin_episode.side_effect = RuntimeError("首轮 Memory 预取未成功")
        with mock.patch("IMAct.harness.episode.Conversation") as conv:
            with self.assertRaises(RuntimeError):
                run_episode(agent, spec_for(), mock.Mock(), mock.Mock())
        conv.assert_not_called()
        agent.end_episode.assert_not_called()  # 会话还没建立,没有什么可结束的


class IndependenceTests(unittest.TestCase):
    def test_harness_and_benchmark_do_not_import_the_vita_package(self):
        """目标:交互逻辑与 VitaBench 解耦。IMAct 下(除待删除的 vendor)任何文件都不应再 import ``vita``。"""
        import re

        root = Path(__file__).resolve().parents[1] / "IMAct"
        pattern = re.compile(r"^\s*(?:from vita(?:\.|\s)|import vita\b)", re.M)
        offenders = [str(p.relative_to(root)) for p in root.rglob("*.py") if "vendor" not in p.parts and pattern.search(p.read_text(encoding="utf-8"))]
        self.assertEqual(offenders, [])

    def test_harness_does_not_depend_on_the_benchmark_package(self):
        import re

        root = Path(__file__).resolve().parents[1] / "IMAct"
        pattern = re.compile(r"^\s*from\s+(?:\.\.|IMAct\.)benchmarks", re.M)
        offenders = [str(p.relative_to(root)) for p in list((root / "harness").rglob("*.py")) + list((root / "evaluation").rglob("*.py")) if pattern.search(p.read_text(encoding="utf-8"))]
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
