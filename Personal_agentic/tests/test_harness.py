"""通用 harness 的测试:对话循环、用户模拟器、LLMCaller,以及用一个非 VitaBench 的环境端到端跑通。"""

import json
import unittest
from unittest import mock

import httpx

from tests.exec_helpers import RECEIPTS, TOOLS, DemoProvider, make_runtime

from IMAct.evaluation.base import RewardInfo
from IMAct.evaluation.vitabench import RubricTask, VitaBenchEvaluator, format_window
from IMAct.exec_layer.bridge import AsyncBridge
from IMAct.harness import (
    Conversation, EpisodeSpec, HarnessAgent, LLMCaller, TerminationReason, UserSimulator, run_episode,
)
from IMAct.harness.conversation import ConversationResult
from IMAct.harness.messages import (
    AssistantMessage, MultiToolMessage, SystemMessage, ToolCall, ToolMessage, UserMessage, to_chat_messages,
)
from IMAct.models.base import ChatResponse, ModelProvider
from IMAct.models.config import ModelConfig, create_provider
from IMAct.retry import CallError, RetryPolicy

PATCH = "IMAct.harness.llm.create_provider"


class ScriptedProvider(ModelProvider):
    """按顺序吐出预设回复的 provider。"""

    name = "scripted"

    def __init__(self, replies, finish_reason="stop"):
        self.replies = list(replies)
        self.requests = []
        self.finish_reason = finish_reason
        self.closed = False

    async def chat(self, request):
        self.requests.append(request)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        finish = self.finish_reason
        if isinstance(reply, tuple):  # (正文, finish_reason):单条回复指定结束原因
            reply, finish = reply
        return ChatResponse(content=reply, finish_reason=finish, usage={"prompt_tokens": 7, "completion_tokens": 3},
                            model="scripted", raw={"id": f"r{len(self.requests)}"})

    async def aclose(self):
        self.closed = True


def make_llm(provider, *, cfg=None, **kw):
    bridge = AsyncBridge("test-llm")
    with mock.patch(PATCH, return_value=provider):
        llm = LLMCaller(cfg or ModelConfig(model="m", provider="openai_compatible", base_url="https://x"), bridge, **kw)
    return llm, bridge


class FakeEnv:
    """一个跟 VitaBench 完全无关的环境:只有一个 echo 工具。"""

    def __init__(self):
        self.calls = []

    def tool_schemas(self):
        return [{"type": "function", "function": {"name": "echo", "parameters": {"type": "object", "properties": {}}}}]

    def get_response(self, call):
        self.calls.append(call)
        return ToolMessage(id=call.id, name=call.name, role="tool", content=f"echo:{json.dumps(call.arguments)}", requestor=call.requestor)

    def get_states(self):
        return {"old_states": [], "new_states": [{"echo_calls": len(self.calls)}]}


class StubAgent:
    """脚本化的 agent:按顺序返回预设消息,并记录收到的输入。"""

    def __init__(self, replies):
        self.replies = list(replies)
        self.seen = []
        self.rt = mock.Mock(trace=None)

    def get_init_state(self, message_history=None):
        return {"history": list(message_history or [])}

    def generate_next_message(self, message, state):
        self.seen.append(message)
        reply = self.replies.pop(0)
        return reply, state


class StubUser:
    def __init__(self, replies):
        self.replies = list(replies)
        self.seen = []

    def get_init_state(self, message_history=None):
        return {}

    def generate_next_message(self, message, state):
        self.seen.append(message)
        return self.replies.pop(0), state

    @staticmethod
    def is_stop(message):
        return UserSimulator.is_stop(message)


def a_text(text):
    return AssistantMessage(role="assistant", content=text, cost=0.0)


def a_call(*names, **args):
    return AssistantMessage(role="assistant", cost=0.0, tool_calls=[ToolCall(id=f"c{i}", name=n, arguments=args) for i, n in enumerate(names)])


def u_text(text):
    return UserMessage(role="user", content=text, cost=0.0)


class ConversationTests(unittest.TestCase):
    def run_conv(self, agent, user, env=None, **kw):
        return Conversation(agent, user, env or FakeEnv(), **kw).run()

    def test_user_agent_tool_agent_user_flow(self):
        env = FakeEnv()
        agent = StubAgent([a_call("echo", x=1), a_text("好了")])
        user = StubUser([u_text("帮我 echo 一下"), u_text("谢谢 ###STOP###")])
        result = self.run_conv(agent, user, env)
        self.assertEqual(result.termination_reason, TerminationReason.USER_STOP)
        roles = [m.role for m in result.messages]
        self.assertEqual(roles, ["assistant", "user", "assistant", "tool", "assistant", "user"])  # 开场问候 + 一轮工具往返
        self.assertEqual([m.turn_idx for m in result.messages], list(range(6)))
        self.assertEqual(env.calls[0].arguments, {"x": 1})
        self.assertEqual(result.states, {"old_states": [], "new_states": [{"echo_calls": 1}]})
        self.assertEqual(result.agent_cost, 0.0)

    def test_parallel_tool_calls_arrive_as_one_multi_tool_message(self):
        agent = StubAgent([a_call("echo", "echo"), a_text("done")])
        user = StubUser([u_text("go"), u_text("###STOP###")])
        self.run_conv(agent, user)
        self.assertIsInstance(agent.seen[1], MultiToolMessage)
        self.assertEqual(len(agent.seen[1].tool_messages), 2)

    def test_invalid_agent_message_is_retried_then_accepted(self):
        empty = AssistantMessage(role="assistant", content=None, cost=0.0)
        agent = StubAgent([empty, empty, a_text("终于有内容了")])
        user = StubUser([u_text("hi"), u_text("###STOP###")])
        result = self.run_conv(agent, user)
        self.assertEqual(result.termination_reason, TerminationReason.USER_STOP)
        self.assertEqual(len(agent.seen), 3)  # 同一条用户消息被送了 3 次

    def test_three_invalid_agent_messages_terminate(self):
        empty = AssistantMessage(role="assistant", content="   ", cost=0.0)
        agent = StubAgent([empty, empty, empty])
        user = StubUser([u_text("hi")])
        result = self.run_conv(agent, user)
        self.assertEqual(result.termination_reason, TerminationReason.INVALID_AGENT_MESSAGE)

    def test_training_mode_does_not_regenerate_invalid_agent_messages(self):
        """RL 训练时每次重新生成都会成为一条额外样本,所以 max_agent_retries=1:无效就直接结束。"""
        empty = AssistantMessage(role="assistant", content=None, cost=0.0)
        agent = StubAgent([empty, a_text("不该被请求到")])
        user = StubUser([u_text("hi")])
        result = self.run_conv(agent, user, max_agent_retries=1)
        self.assertEqual(result.termination_reason, TerminationReason.INVALID_AGENT_MESSAGE)
        self.assertEqual(len(agent.seen), 1)
        with self.assertRaisesRegex(ValueError, "max_agent_retries"):
            Conversation(agent, user, FakeEnv(), max_agent_retries=0)

    def test_agent_stop_token_ends_the_conversation(self):
        agent = StubAgent([a_text("好的 ###STOP###")])
        user = StubUser([u_text("hi")])
        self.assertEqual(self.run_conv(agent, user).termination_reason, TerminationReason.AGENT_STOP)

    def test_max_steps(self):
        agent = StubAgent([a_text(f"第{i}句") for i in range(10)])
        user = StubUser([u_text(f"问{i}") for i in range(10)])
        result = self.run_conv(agent, user, max_steps=4)
        self.assertEqual(result.termination_reason, TerminationReason.MAX_STEPS)

    def test_user_tool_calls_route_to_the_environment_and_back_to_the_user(self):
        env = FakeEnv()
        agent = StubAgent([a_text("好")])
        user_call = UserMessage(role="user", cost=0.0, tool_calls=[ToolCall(id="u1", name="echo", arguments={}, requestor="user")])
        user = StubUser([user_call, u_text("###STOP###")])
        result = self.run_conv(agent, user, env)
        self.assertEqual(env.calls[0].requestor, "user")
        self.assertIsInstance(user.seen[1], ToolMessage)  # 回执回到用户,而不是助手

    def test_message_history_is_validated(self):
        orphan = ToolMessage(id="x", name="echo", role="tool", content="?")
        with self.assertRaisesRegex(ValueError, "No tool messages expected"):
            Conversation.validate_message_history([orphan])
        missing = [a_call("echo"), a_call("echo")]  # 第一次调用的回执还没到,又来了新的调用
        with self.assertRaisesRegex(ValueError, "tool messages are missing"):
            Conversation.validate_message_history(missing)
        wrong = [a_call("echo"), ToolMessage(id="c0", name="echo", role="tool", content="?", requestor="user")]
        with self.assertRaisesRegex(ValueError, "expected assistant"):
            Conversation.validate_message_history(wrong)

    def test_history_ending_with_a_user_message_goes_straight_to_the_agent(self):
        history = [a_text("你好"), u_text("我想订票")]
        agent = StubAgent([a_text("好的 ###STOP###")])
        result = self.run_conv(agent, StubUser([]), message_history=history)
        self.assertEqual(agent.seen[0].content, "我想订票")
        self.assertEqual(result.termination_reason, TerminationReason.AGENT_STOP)
        self.assertEqual([m.content for m in result.messages][:2], ["你好", "我想订票"])

    def test_history_ending_with_a_stop_is_already_done(self):
        result = self.run_conv(StubAgent([]), StubUser([]), message_history=[a_text("你好"), u_text("算了 ###STOP###")])
        self.assertEqual(result.termination_reason, TerminationReason.USER_STOP)


class UserSimulatorTests(unittest.TestCase):
    def sim(self, replies, **kw):
        provider = ScriptedProvider(replies)
        llm, bridge = make_llm(provider)
        self.addCleanup(bridge.close)
        return UserSimulator(llm, "你是一个用户", "我要点外卖", **kw), provider

    def test_opening_is_sent_verbatim_without_calling_the_model(self):
        sim, provider = self.sim([])
        state = sim.get_init_state()
        reply, state = sim.generate_next_message(a_text("您好"), state)
        self.assertEqual(reply.content, "我要点外卖")
        self.assertEqual(provider.requests, [])

    def test_later_turns_call_the_model_with_roles_flipped(self):
        sim, provider = self.sim(["随便"])
        state = sim.get_init_state()
        _, state = sim.generate_next_message(a_text("您好"), state)
        reply, state = sim.generate_next_message(a_text("想吃什么口味?"), state)
        self.assertEqual(reply.content, "随便")
        self.assertEqual(reply.usage, {"prompt_tokens": 7, "completion_tokens": 3})
        sent = provider.requests[0].messages
        self.assertEqual(sent[0], {"role": "system", "content": "你是一个用户"})
        # 助手的话对模拟器是 user,用户自己说过的话对模拟器是 assistant
        self.assertEqual([(m["role"], m["content"]) for m in sent[1:]], [("user", "您好"), ("assistant", "我要点外卖"), ("user", "想吃什么口味?")])

    def test_stop_markers(self):
        for marker in ("###STOP###", "###TRANSFER###", "###OUT-OF-SCOPE###"):
            self.assertTrue(UserSimulator.is_stop(u_text(f"好了 {marker}")))
        self.assertFalse(UserSimulator.is_stop(u_text("继续")))
        self.assertFalse(UserSimulator.is_stop(UserMessage(role="user", tool_calls=[ToolCall(name="x", arguments={})])))

    def test_trace_records_every_user_reply_including_the_stop_message(self):
        trace = mock.Mock()
        sim, _ = self.sim(["###STOP###"], trace=trace)
        state = sim.get_init_state()
        sim.generate_next_message(a_text("您好"), state)
        sim.generate_next_message(a_text("还有别的吗"), state)
        self.assertEqual([c.args[0] for c in trace.emit.call_args_list], ["user_reply", "user_reply"])
        self.assertIn("###STOP###", trace.emit.call_args_list[1].kwargs["message"]["content"])

    def test_assistant_tool_calls_cannot_be_flipped(self):
        sim, _ = self.sim(["x"])
        state = sim.get_init_state()
        state.opened = True
        with self.assertRaisesRegex(ValueError, "Tool calls are not supported"):
            sim.generate_next_message(a_call("echo"), state)


class LLMCallerTests(unittest.TestCase):
    def test_seed_only_sent_to_providers_that_accept_it(self):
        for provider_name, expected in (("openai_compatible", True), ("openai", True), ("anthropic_messages", False), ("openai_responses", False)):
            with self.subTest(provider=provider_name):
                provider = ScriptedProvider(["ok"])
                llm, bridge = make_llm(provider, cfg=ModelConfig(model="m", provider=provider_name, base_url="https://x"))
                try:
                    llm.chat([SystemMessage(role="system", content="s"), u_text("hi")], seed=7)
                finally:
                    bridge.close()
                self.assertEqual("seed" in provider.requests[0].params, expected)

    def test_config_params_are_defaults_and_call_params_win(self):
        provider = ScriptedProvider(["ok"])
        cfg = ModelConfig(model="m", base_url="https://x", params={"temperature": 0.2, "max_tokens": 100})
        llm, bridge = make_llm(provider, cfg=cfg)
        try:
            llm.chat([u_text("hi")], temperature=0.9)
        finally:
            bridge.close()
        self.assertEqual(provider.requests[0].params, {"temperature": 0.9, "max_tokens": 100})

    def test_transient_errors_are_retried_and_empty_or_truncated_replies_are_errors(self):
        provider = ScriptedProvider([CallError("rate_limit", "429"), "ok"])
        llm, bridge = make_llm(provider, policy=RetryPolicy(max_retries=2, backoff_s=0.0))
        try:
            self.assertEqual(llm.chat([u_text("hi")]).content, "ok")
            self.assertEqual(len(provider.requests), 2)
        finally:
            bridge.close()
        for provider, pattern in ((ScriptedProvider(["  "]), "空正文"), (ScriptedProvider(["半截"], finish_reason="length"), "未完整结束")):
            llm, bridge = make_llm(provider, policy=RetryPolicy(max_retries=0))
            try:
                with self.assertRaisesRegex(CallError, pattern):
                    llm.chat([u_text("hi")])
            finally:
                bridge.close()

    def test_reasoning_model_that_spends_its_budget_thinking_is_retried(self):
        """真实故障:推理模型把 max_tokens 全花在推理上,finish_reason=length、正文为空。重试后应恢复,而不是让整次评分作废。"""
        provider = ScriptedProvider([("", "length"), ("", "stop"), ("评分结果", "stop")])
        llm, bridge = make_llm(provider, policy=RetryPolicy(max_retries=3, backoff_s=0.0))
        try:
            self.assertEqual(llm.chat([u_text("hi")]).content, "评分结果")
        finally:
            bridge.close()
        self.assertEqual(len(provider.requests), 3)

    def test_persistent_truncation_still_fails_after_retries_are_used_up(self):
        provider = ScriptedProvider([("", "length")] * 3)
        llm, bridge = make_llm(provider, policy=RetryPolicy(max_retries=2, backoff_s=0.0))
        try:
            with self.assertRaisesRegex(CallError, "空正文"):
                llm.chat([u_text("hi")])
        finally:
            bridge.close()
        self.assertEqual(len(provider.requests), 3)  # 首次 + 2 次重试,不会无限循环

    def test_bad_response_retry_is_added_to_a_custom_policy_without_touching_the_original(self):
        policy = RetryPolicy(max_retries=1, backoff_s=0.0, retry_on=frozenset({"timeout"}))
        llm, bridge = make_llm(ScriptedProvider([("", "length"), "ok"]), policy=policy)
        try:
            self.assertEqual(llm.chat([u_text("hi")]).content, "ok")
        finally:
            bridge.close()
        self.assertEqual(policy.retry_on, frozenset({"timeout"}))

    def test_trace_stage_records_request_response_and_error(self):
        trace = mock.Mock()
        llm, bridge = make_llm(ScriptedProvider(["ok", CallError("auth", "bad key")]), trace=trace, trace_stage="judge",
                               policy=RetryPolicy(max_retries=0))
        try:
            llm.chat([u_text("hi")])
            with self.assertRaises(CallError):
                llm.chat([u_text("hi")])
        finally:
            bridge.close()
        self.assertEqual([c.args[0] for c in trace.emit.call_args_list], ["judge_request", "judge_response", "judge_request", "judge_error"])

    def test_close_releases_the_provider(self):
        provider = ScriptedProvider(["ok"])
        llm, bridge = make_llm(provider)
        llm.close()
        bridge.close()
        self.assertTrue(provider.closed)

    def test_routes_by_explicit_provider_not_by_model_name(self):
        """协议由 models.yaml 的 provider 字段决定:同一个 LLMCaller 接口,Responses 与 Messages 各走各的端点。"""
        seen = []

        def handler(req):
            body = json.loads(req.content)
            seen.append((req.url.path, body))
            if req.url.path.endswith("/responses"):
                return httpx.Response(200, json={"id": "r", "model": body["model"], "status": "completed", "output_text": "user output"})
            return httpx.Response(200, json={"id": "c", "model": body["model"], "stop_reason": "end_turn", "content": [{"type": "text", "text": "judge output"}]})

        configs = {
            "responses": ModelConfig(provider="openai_responses", model="remote-gpt", base_url="https://gateway/v1", api_key="g",
                                     params={"stream": False, "max_output_tokens": 99}),
            "messages": ModelConfig(provider="anthropic_messages", model="remote-claude", base_url="https://gateway/v1", api_key="c",
                                    params={"stream": False, "max_tokens": 88}),
        }
        outputs = []
        clients = []
        for cfg in configs.values():
            bridge = AsyncBridge("test-route")
            real = create_provider(cfg)
            real._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            clients.append(real._client)
            with mock.patch(PATCH, return_value=real):
                llm = LLMCaller(cfg, bridge)
            try:
                outputs.append(llm.chat([u_text("hello")]).content)
            finally:
                llm.close()
                bridge.close()
        self.assertEqual(outputs, ["user output", "judge output"])
        self.assertEqual([url for url, _ in seen], ["/v1/responses", "/v1/messages"])
        self.assertEqual(seen[0][1]["max_output_tokens"], 99)
        self.assertEqual(seen[1][1]["max_tokens"], 88)
        self.assertTrue(all(c.is_closed for c in clients))


class MessageFormatTests(unittest.TestCase):
    def test_to_chat_messages_covers_every_role(self):
        msgs = [SystemMessage(role="system", content="s"), u_text("u"), a_call("echo", x=1),
                ToolMessage(id="c0", name="echo", role="tool", content="r")]
        out = to_chat_messages(msgs)
        self.assertEqual([m["role"] for m in out], ["system", "user", "assistant", "tool"])
        self.assertEqual(json.loads(out[2]["tool_calls"][0]["function"]["arguments"]), {"x": 1})
        self.assertEqual(out[3]["tool_call_id"], "c0")

    def test_old_trajectory_json_still_loads(self):
        """已保存的 simulation.json 里的消息形状必须仍能读回来。"""
        from pydantic import TypeAdapter

        from IMAct.harness.messages import Message

        raw = [{"role": "assistant", "content": None, "tool_calls": [{"id": "1", "name": "x", "arguments": {}, "requestor": "assistant"}],
                "turn_idx": 0, "timestamp": "20260302_120000", "cost": 0.0, "usage": {"prompt_tokens": 1}, "raw_data": {"imact": {}}},
               {"id": "1", "name": "x", "role": "tool", "content": "ok", "requestor": "assistant", "error": False, "turn_idx": 1, "timestamp": "t"}]
        parsed = TypeAdapter(list[Message]).validate_python(raw)
        self.assertIsInstance(parsed[0], AssistantMessage)
        self.assertIsInstance(parsed[1], ToolMessage)


class EvaluatorTests(unittest.TestCase):
    def evaluator(self, replies):
        provider = ScriptedProvider(replies)
        llm, bridge = make_llm(provider)
        self.addCleanup(bridge.close)
        return VitaBenchEvaluator(llm), provider

    @staticmethod
    def conv(n=4, reason=TerminationReason.USER_STOP):
        return ConversationResult(messages=[u_text(f"消息{i}") for i in range(n)], termination_reason=reason, duration=0.0)

    def test_premature_termination_scores_zero_without_calling_the_judge(self):
        for reason in (TerminationReason.MAX_STEPS, TerminationReason.TOO_MANY_ERRORS, TerminationReason.INVALID_AGENT_MESSAGE):
            ev, provider = self.evaluator([])
            reward = ev.evaluate(self.conv(reason=reason), RubricTask("点餐", ["含肉"]))
            self.assertEqual(reward.reward, 0.0)
            self.assertIn("terminated prematurely", reward.info["note"])
            self.assertEqual(provider.requests, [])

    def test_no_rubrics_means_full_marks(self):
        ev, provider = self.evaluator([])
        self.assertEqual(ev.evaluate(self.conv(), RubricTask("点餐", [])).reward, 1.0)
        self.assertEqual(provider.requests, [])

    def test_all_rubrics_met_scores_one_and_duplicates_are_merged(self):
        rows = [{"rubric_idx": "rubric_0", "justification": "对", "meetExpectation": True},
                {"rubric_idx": "rubric_1", "justification": "对", "meetExpectation": True}]
        ev, provider = self.evaluator([json.dumps(rows)])
        reward = ev.evaluate(self.conv(), RubricTask("点餐", ["含肉", "含肉", "及时"], env_time="2026-03-02 11:35:00"))
        self.assertEqual(reward.reward, 1.0)
        self.assertEqual(len(reward.nl_rubrics), 2)  # 重复的 rubric 只算一次
        self.assertIn("星期一", provider.requests[0].messages[0]["content"])  # 告诉裁判"现在是星期几"

    def test_rubric_state_carries_across_windows(self):
        first = [{"rubric_idx": "rubric_0", "justification": "窗口1满足", "meetExpectation": True}, {"rubric_idx": "rubric_1", "justification": "未满足", "meetExpectation": False}]
        second = [{"rubric_idx": "rubric_0", "justification": "窗口1满足", "meetExpectation": True}, {"rubric_idx": "rubric_1", "justification": "窗口2满足", "meetExpectation": True}]
        ev, provider = self.evaluator([json.dumps(first), json.dumps(second)])
        reward = ev.evaluate(self.conv(n=15), RubricTask("点餐", ["含肉", "及时"]))
        self.assertEqual(reward.reward, 1.0)
        self.assertEqual(reward.info["num_windows"], 2)
        self.assertIn("窗口1满足", provider.requests[1].messages[-1]["content"])  # 第二个窗口看得到第一个窗口的结论

    def test_window_formatting_numbers_messages_globally_and_inlines_tool_calls(self):
        text = format_window([a_call("echo", x=1), u_text("你好")], start=8)
        self.assertEqual(text, "[9] assistant: echo(x=1)\n[10] user: 你好")


class GenericEndToEndTests(unittest.TestCase):
    """用一个与 VitaBench 无关的环境和 Demo Action 模型跑完整个 episode:harness 不依赖任何 benchmark。"""

    def test_episode_with_a_non_vitabench_environment(self):
        rt = make_runtime(DemoProvider(), memory_client=None)

        class Env(FakeEnv):
            def tool_schemas(self):
                return TOOLS

            def get_response(self, call):
                self.calls.append(call)
                return ToolMessage(id=call.id, name=call.name, role="tool", content=RECEIPTS[call.name], requestor="assistant")

        env = Env()
        spec = EpisodeSpec(bench="toy", user_id="u1", episode_id="toy-1", domain="delivery", instruction="点午餐",
                           timestamp="2026-01-01", system_prompt="离线测试。允许模拟下单与支付。", tools=TOOLS)
        class User(ScriptedProvider):
            """助手问什么都回"随便",看到支付成功的回执后才结束对话。"""

            async def chat(self, request):
                self.requests.append(request)
                done = any("支付成功" in (m.get("content") or "") for m in request.messages)
                return ChatResponse(content="###STOP###" if done else "随便,你看着办。", finish_reason="stop", usage={}, raw={"id": "u"})

        llm, bridge = make_llm(User([]))
        try:
            user = UserSimulator(llm, "你是用户", "帮我点个午餐外卖")
            result = run_episode(HarnessAgent(runtime=rt, run_id="toy"), spec, user, env, max_steps=40)
        finally:
            llm.close()
            bridge.close()
            rt.close()
        self.assertEqual(result.termination_reason, TerminationReason.USER_STOP)
        self.assertEqual([c.name for c in env.calls], ["delivery_store_search_recommand", "delivery_product_search_recommand", "create_delivery_order", "pay_delivery_order"])
        self.assertEqual(result.states["new_states"], [{"echo_calls": 4}])
        final_assistant = [m for m in result.messages if m.role == "assistant" and m.content][-1]
        self.assertEqual(final_assistant.raw_data["imact"]["outcome"], "completed")


if __name__ == "__main__":
    unittest.main()
