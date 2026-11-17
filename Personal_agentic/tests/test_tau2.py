"""τ² 接入的测试:原生 agent、对话循环的 τ² 选项、评分器、人设、任务读取、端到端。全部离线(脚本化的假模型)。

数据依赖:需要 ``data/tau2``(上游数据集)。缺数据时,依赖它的用例会被跳过,不会误报通过。
"""

import json
import unittest
from types import SimpleNamespace
from unittest import mock

from tests.test_harness import ScriptedProvider, make_llm, a_call, a_text, u_text, FakeEnv, StubAgent, StubUser

from IMAct.evaluation.base import JudgeError, RewardType
from IMAct.evaluation.tau2 import Tau2EvalTask, Tau2Evaluator, extract_json
from IMAct.harness import Conversation, TerminationReason
from IMAct.harness.conversation import ConversationResult
from IMAct.harness.messages import AssistantMessage, ToolCall, ToolMessage, UserMessage
from IMAct.harness.native_agent import NativeAgent
from IMAct.models.base import ChatResponse

try:
    from IMAct.benchmarks.tau2.multilingual import load_base_cell
    from IMAct.benchmarks.tau2.tasks import get_tasks

    HAS_DATA = bool(get_tasks("airline", num_tasks=1))
except Exception:  # noqa: BLE001
    HAS_DATA = False

need_data = unittest.skipUnless(HAS_DATA, "需要 data/tau2 数据集")


class NativeAgentTests(unittest.TestCase):
    def agent(self, replies, **kw):
        provider = ScriptedProvider(replies)
        llm, bridge = make_llm(provider)
        self.addCleanup(bridge.close)
        return NativeAgent(llm, "你是客服", [{"type": "function", "function": {"name": "f"}}], **kw), provider

    def test_sends_system_prompt_history_and_tools_each_turn(self):
        agent, provider = self.agent(["好的", "还有吗"])
        state = agent.get_init_state(message_history=[a_text("Hi! How can I help you today?")])
        r1, state = agent.generate_next_message(u_text("我要退票"), state)
        r2, state = agent.generate_next_message(u_text("谢谢"), state)
        self.assertEqual([r1.content, r2.content], ["好的", "还有吗"])
        sent = provider.requests[1].messages
        self.assertEqual([m["role"] for m in sent], ["system", "assistant", "user", "assistant", "user"])
        self.assertEqual(sent[0]["content"], "你是客服")
        self.assertEqual(sent[1]["content"], "Hi! How can I help you today?")  # 自己说过的开场白在历史里
        self.assertIsNone(provider.requests[0].tools and None)  # 工具随每次请求发送
        self.assertEqual(provider.requests[0].tools, [{"type": "function", "function": {"name": "f"}}])

    def test_text_and_tool_calls_are_both_kept_like_upstream(self):
        class P(ScriptedProvider):
            async def chat(self, request):
                self.requests.append(request)
                return ChatResponse(content="我查一下", tool_calls=[{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}],
                                    finish_reason="tool_calls", usage={}, raw={"id": "x"})

        provider = P([])
        llm, bridge = make_llm(provider)
        self.addCleanup(bridge.close)
        agent = NativeAgent(llm, "s")
        reply, _ = agent.generate_next_message(u_text("hi"), agent.get_init_state())
        self.assertEqual(reply.content, "我查一下")
        self.assertEqual(reply.tool_calls[0].name, "f")

    def test_history_must_be_valid_for_an_agent(self):
        agent, _ = self.agent([])
        with self.assertRaises(ValueError):
            agent.get_init_state(message_history=[UserMessage(role="user", tool_calls=[ToolCall(name="x", arguments={}, requestor="user")])])

    def test_episode_hooks_are_noops(self):
        agent, _ = self.agent([])
        self.assertIsNone(agent.begin_episode(object()))
        self.assertIsNone(agent.end_episode("user_stop"))


class ConversationOptionTests(unittest.TestCase):
    def run_conv(self, agent, user, env=None, **kw):
        return Conversation(agent, user, env or FakeEnv(), **kw).run()

    def test_agent_stop_token_is_ignored_when_the_agent_cannot_stop(self):
        agent = StubAgent([a_text("好的 ###STOP###"), a_text("再见")])
        user = StubUser([u_text("hi"), u_text("还有一件事"), u_text("bye ###STOP###")])
        result = self.run_conv(agent, user, agent_can_stop=False)
        self.assertEqual(result.termination_reason, TerminationReason.USER_STOP)
        self.assertEqual(len(agent.seen), 2)  # 助手说了 ###STOP### 后对话照常继续,直到用户结束
        # 对照:默认(助手可以结束)时,第一句 ###STOP### 就结束了
        agent2 = StubAgent([a_text("好的 ###STOP###")])
        result2 = self.run_conv(agent2, StubUser([u_text("hi")]), agent_can_stop=True)
        self.assertEqual(result2.termination_reason, TerminationReason.AGENT_STOP)

    def test_custom_greeting_is_in_the_trajectory_and_optionally_in_the_agent_history(self):
        for sees in (False, True):
            agent = StubAgent([a_text("ok")])
            user = StubUser([u_text("hi"), u_text("###STOP###")])
            result = self.run_conv(agent, user, greeting="Hi! How can I help you today?", agent_sees_greeting=sees)
            self.assertEqual(result.messages[0].content, "Hi! How can I help you today?")

    def test_greeting_reaches_the_agent_history_only_when_asked(self):
        class Spy(StubAgent):
            def get_init_state(self, message_history=None):
                self.init_history = list(message_history or [])
                return {}

        for sees, expected in ((False, 0), (True, 1)):
            agent = Spy([a_text("ok")])
            self.run_conv(agent, StubUser([u_text("hi"), u_text("###STOP###")]), agent_sees_greeting=sees)
            self.assertEqual(len(agent.init_history), expected)

    def test_tool_errors_accumulate_and_end_the_conversation(self):
        class Failing(FakeEnv):
            def get_response(self, call):
                self.calls.append(call)
                return ToolMessage(id=call.id, role="tool", content="Error: nope", requestor=call.requestor, error=True)

        agent = StubAgent([a_call("echo"), a_call("echo"), a_call("echo"), a_text("never")])
        user = StubUser([u_text("go")])
        result = self.run_conv(agent, user, Failing(), max_errors=2)
        self.assertEqual(result.termination_reason, TerminationReason.TOO_MANY_ERRORS)

    def test_error_count_is_off_unless_configured(self):
        class Failing(FakeEnv):
            def get_response(self, call):
                return ToolMessage(id=call.id, role="tool", content="Error", requestor=call.requestor, error=True)

        agent = StubAgent([a_call("echo"), a_call("echo"), a_text("done")])
        user = StubUser([u_text("go"), u_text("###STOP###")])
        self.assertEqual(self.run_conv(agent, user, Failing()).termination_reason, TerminationReason.USER_STOP)

    def test_a_tool_round_finishes_before_the_step_limit_when_asked(self):
        """步骤:1 问候→用户,2 用户→助手,3 助手发出工具调用。步数上限是 2 时:
        打开开关,先让环境回执再终止(不留下没有回执的工具调用);关闭则在发出调用那一步就终止。"""
        for finish_first, roles in ((True, ["assistant", "user", "assistant", "tool"]), (False, ["assistant", "user", "assistant"])):
            agent = StubAgent([a_call("echo"), a_text("x")])
            user = StubUser([u_text("go"), u_text("###STOP###")])
            result = self.run_conv(agent, user, max_steps=2, finish_tool_round_before_limits=finish_first)
            self.assertEqual(result.termination_reason, TerminationReason.MAX_STEPS)
            self.assertEqual([m.role for m in result.messages], roles)

    def test_environment_sync_runs_after_every_tool_round(self):
        class Syncing(FakeEnv):
            syncs = 0

            def sync_tools(self):
                Syncing.syncs += 1

        agent = StubAgent([a_call("echo"), a_call("echo"), a_text("ok")])
        self.run_conv(agent, StubUser([u_text("go"), u_text("###STOP###")]), Syncing())
        self.assertEqual(Syncing.syncs, 2)

    def test_rejected_status_needs_an_error_flag(self):
        with self.assertRaises(ValueError):
            ToolMessage(id="1", role="tool", execution_status="rejected")

    def test_user_without_opening_generates_every_turn_including_the_first(self):
        from IMAct.harness import UserSimulator

        provider = ScriptedProvider(["Hi, I want to cancel my flight.", "Yes please. ###STOP###"])
        llm, bridge = make_llm(provider)
        self.addCleanup(bridge.close)
        sim = UserSimulator(llm, "你是用户", opening=None)
        state = sim.get_init_state()
        first, state = sim.generate_next_message(a_text("Hi! How can I help you today?"), state)
        second, state = sim.generate_next_message(a_text("Shall I cancel it?"), state)
        self.assertEqual((first.content, second.content), ("Hi, I want to cancel my flight.", "Yes please. ###STOP###"))
        self.assertEqual(len(provider.requests), 2)
        self.assertEqual(provider.requests[0].messages[1], {"role": "user", "content": "Hi! How can I help you today?"})

    def test_user_tools_are_offered_to_the_model_and_calls_are_marked_as_the_users(self):
        class P(ScriptedProvider):
            async def chat(self, request):
                self.requests.append(request)
                return ChatResponse(content="", tool_calls=[{"id": "u1", "type": "function", "function": {"name": "toggle", "arguments": "{}"}}],
                                    finish_reason="tool_calls", usage={}, raw={"id": "x"})

        from IMAct.harness import UserSimulator

        provider = P([])
        llm, bridge = make_llm(provider)
        self.addCleanup(bridge.close)
        tools = [{"type": "function", "function": {"name": "toggle"}}]
        sim = UserSimulator(llm, "你是用户", opening=None, tools=tools)
        reply, _ = sim.generate_next_message(a_text("Please toggle airplane mode."), sim.get_init_state())
        self.assertEqual(provider.requests[0].tools, tools)
        self.assertEqual(reply.tool_calls[0].requestor, "user")


class EvaluatorUnitTests(unittest.TestCase):
    class Gold:
        def __init__(self, name, args, requestor="assistant", compare_args=None):
            self.name, self.arguments, self.requestor, self.compare_args = name, args, requestor, compare_args

        def compare_with_tool_call(self, call):
            if self.name != call.name:
                return False
            keys = self.compare_args if self.compare_args is not None else call.arguments.keys()
            return {k: call.arguments.get(k) for k in keys} == {k: self.arguments.get(k) for k in keys}

        def model_dump(self, mode="json"):
            return {"name": self.name, "arguments": self.arguments}

    def task(self, **criteria):
        defaults = dict(actions=None, env_assertions=None, communicate_info=None, nl_assertions=None, reward_basis=[])
        defaults.update(criteria)
        return Tau2EvalTask(criteria=SimpleNamespace(**defaults), environment_factory=lambda: None)

    @staticmethod
    def result(messages, reason=TerminationReason.USER_STOP):
        return ConversationResult(messages=messages, termination_reason=reason, duration=0.0)

    def judge(self, replies):
        provider = ScriptedProvider(replies)
        llm, bridge = make_llm(provider)
        self.addCleanup(bridge.close)
        return llm, provider

    def test_unfinished_conversations_score_zero_without_judging(self):
        for reason in (TerminationReason.MAX_STEPS, TerminationReason.TOO_MANY_ERRORS, TerminationReason.INVALID_AGENT_MESSAGE):
            reward = Tau2Evaluator().evaluate(self.result([], reason), self.task())
            self.assertEqual(reward.reward, 0.0)
            self.assertIn("terminated prematurely", reward.info["note"])

    def test_no_criteria_is_full_marks(self):
        self.assertEqual(Tau2Evaluator().evaluate(self.result([]), Tau2EvalTask(criteria=None, environment_factory=lambda: None)).reward, 1.0)

    def test_action_check_uses_compare_args_and_counts_user_calls(self):
        task = self.task(actions=[self.Gold("cancel", {"id": "1", "x": "ignored"}, compare_args=["id"]), self.Gold("toggle", {}, requestor="user")],
                         reward_basis=[RewardType.ACTION])
        msgs = [a_call("cancel", id="1", x="different"), UserMessage(role="user", cost=0.0, tool_calls=[ToolCall(id="u", name="toggle", arguments={}, requestor="user")])]
        info = Tau2Evaluator().evaluate_actions(task, msgs)
        self.assertEqual((info.reward, [c.action_match for c in info.action_checks]), (1.0, [True, True]))
        self.assertEqual(Tau2Evaluator().evaluate_actions(task, msgs[:1]).reward, 0.0)

    def test_substring_communicate_ignores_case_and_commas_and_only_reads_the_agent(self):
        # 上游只去掉**消息里**的逗号(不处理期望信息里的逗号):期望写 "1234.56" 才能匹配消息里的 "1,234.56"
        task = self.task(communicate_info=["1234.56", "Gold member"])
        ev = Tau2Evaluator(communicate_scorer="substring")
        msgs = [u_text("the gold member thing"), a_text("Your refund is 1,234.56 dollars"), a_text("you are a GOLD MEMBER")]
        self.assertEqual(ev.evaluate_communicate(task, msgs).reward, 1.0)
        self.assertEqual(ev.evaluate_communicate(task, msgs[:2]).reward, 0.0)  # 用户说的不算
        # 期望信息自带逗号永远对不上(与上游一致):它是个已知的口径限制,不是这里的 bug
        self.assertEqual(ev.evaluate_communicate(self.task(communicate_info=["1,234.56"]), msgs).reward, 0.0)

    def test_semantic_communicate_asks_the_judge_per_fact_and_validates_replies(self):
        task = self.task(communicate_info=["a", "bb"])
        judge, provider = self.judge([json.dumps({"reasoning": "r1", "met": True}), json.dumps({"reasoning": "r2", "met": False})])
        info = Tau2Evaluator(judge).evaluate_communicate(task, [a_text("hello")])
        self.assertEqual([(c.met, c.justification) for c in info.communicate_checks], [(True, "[llm_judge] r1"), (False, "[llm_judge] r2")])
        self.assertEqual(info.reward, 0.0)
        self.assertEqual(len(provider.requests), 2)
        for bad in ('{"met": "true", "reasoning": "x"}', '{"met": true}', "not json", "[]"):
            judge, _ = self.judge([bad])
            with self.assertRaises(JudgeError, msg=bad):
                Tau2Evaluator(judge).evaluate_communicate(self.task(communicate_info=["a"]), [a_text("x")])

    def test_nl_assertions_need_one_result_per_assertion(self):
        task = self.task(nl_assertions=["A", "B"], reward_basis=[RewardType.NL_ASSERTION])
        good = {"results": [{"expectedOutcome": "A", "reasoning": "ok", "metExpectation": True}, {"expectedOutcome": "B", "reasoning": "no", "metExpectation": False}]}
        judge, _ = self.judge(["```json\n" + json.dumps(good) + "\n```"])
        info = Tau2Evaluator(judge).evaluate_nl_assertions(task, [a_text("x")])
        self.assertEqual((info.reward, [c.met for c in info.nl_assertions]), (0.0, [True, False]))
        for bad in (json.dumps({"results": good["results"][:1]}), json.dumps({"results": []}), json.dumps({"results": [{"reasoning": "r", "metExpectation": "yes"}] * 2}), "{}"):
            judge, _ = self.judge([bad])
            with self.assertRaises(JudgeError, msg=bad):
                Tau2Evaluator(judge).evaluate_nl_assertions(task, [a_text("x")])

    def test_semantic_scoring_without_a_judge_fails_loudly(self):
        with self.assertRaisesRegex(JudgeError, "裁判"):
            Tau2Evaluator().evaluate_nl_assertions(self.task(nl_assertions=["A"]), [])

    def test_extract_json(self):
        self.assertEqual(extract_json('x ```json\n{"a": 1}\n``` y'), '{"a": 1}')
        self.assertEqual(extract_json('before {"a": {"b": 2}} after'), '{"a": {"b": 2}}')
        self.assertEqual(extract_json("plain"), "plain")


@need_data
class TaskAndEnvironmentTests(unittest.TestCase):
    def test_official_split_sizes(self):
        self.assertEqual([len(get_tasks(d)) for d in ("airline", "retail", "telecom")], [50, 114, 114])

    def test_selection_and_errors(self):
        self.assertEqual([t.id for t in get_tasks("airline", task_ids=["3", "1"])], sorted(["1", "3"], key=lambda x: [t.id for t in get_tasks("airline")].index(x)))
        self.assertEqual(len(get_tasks("airline", num_tasks=2)), 2)
        with self.assertRaisesRegex(ValueError, "没有这些 task"):
            get_tasks("airline", task_ids=["nope"])
        with self.assertRaisesRegex(ValueError, "不支持的 domain"):
            get_tasks("hotel")

    def test_telecom_environment_applies_initial_actions_and_exposes_user_tools(self):
        from IMAct.benchmarks.tau2.harness_env import Tau2HarnessEnvironment
        from IMAct.benchmarks.tau2.tasks import get_environment

        task = get_tasks("telecom", num_tasks=1)[0]
        env = Tau2HarnessEnvironment(get_environment("telecom"), task)
        self.assertGreater(len(env.user_tool_schemas()), 20)
        self.assertGreater(len(env.tool_schemas()), 5)

    def test_airline_has_no_user_tools(self):
        from IMAct.benchmarks.tau2.harness_env import Tau2HarnessEnvironment
        from IMAct.benchmarks.tau2.tasks import get_environment

        env = Tau2HarnessEnvironment(get_environment("airline"), get_tasks("airline", num_tasks=1)[0])
        self.assertIsNone(env.user_tool_schemas())

    def test_tool_errors_are_flagged_and_do_not_raise(self):
        from IMAct.benchmarks.tau2.harness_env import Tau2HarnessEnvironment
        from IMAct.benchmarks.tau2.tasks import get_environment

        env = Tau2HarnessEnvironment(get_environment("airline"), get_tasks("airline", num_tasks=1)[0])
        reply = env.get_response(ToolCall(id="1", name="no_such_tool", arguments={}))
        self.assertTrue(reply.error)
        self.assertTrue(reply.content.startswith("Error:"))


@need_data
class EndToEndTests(unittest.TestCase):
    """用 golden 动作驱动的脚本化 agent / 用户,走完 prepare → 对话 → 评分的整条链路,不调用任何真实模型。"""

    def test_golden_replay_scores_full_marks_on_db_basis_tasks(self):
        from IMAct.benchmarks.tau2.runner import Tau2Runner

        class Replayer:
            """按 golden 动作依次出招的 agent;动作用完说一句话。"""

            def __init__(self, queue):
                self.queue, self.n, self.rt = queue, 0, None

            def get_init_state(self, message_history=None):
                return {}

            def begin_episode(self, spec):
                pass

            def end_episode(self, reason="user_stop"):
                pass

            def generate_next_message(self, message, state):
                self.n += 1
                head = self.queue[0] if self.queue and self.queue[0].requestor == "assistant" else None
                if head is not None:
                    self.queue.pop(0)
                    return AssistantMessage(role="assistant", cost=0.0, tool_calls=[ToolCall(id=f"a{self.n}", name=head.name, arguments=head.arguments)]), state
                return AssistantMessage(role="assistant", content="done", cost=0.0), state

        class ScriptedUser:
            def __init__(self, queue):
                self.queue, self.n, self.opening, self.tools = queue, 0, None, None

            def get_init_state(self, message_history=None):
                return {}

            def generate_next_message(self, message, state):
                self.n += 1
                if self.n > 1 and not any(a.requestor == "assistant" for a in self.queue):
                    return UserMessage(role="user", content="thanks ###STOP###", cost=0.0), state
                return UserMessage(role="user", content="hello", cost=0.0), state

            @staticmethod
            def is_stop(m):
                return m.content is not None and "###STOP###" in m.content

        task = next(t for t in get_tasks("retail") if [getattr(b, "value", b) for b in t.evaluation_criteria.reward_basis] == ["DB"] and t.evaluation_criteria.actions)
        queue = list(task.evaluation_criteria.actions)
        runner = Tau2Runner(load_base_cell("retail"), make_agent=lambda prepared, seed: Replayer(queue), user_llm=mock.Mock(), judge=None)
        runner.prepare = lambda t, seed=None: _prepared_with(runner, t, ScriptedUser(queue))
        sim = runner.run_task(task, run_id="t")
        self.assertEqual(sim.termination_reason, TerminationReason.USER_STOP)
        self.assertEqual(sim.messages[0].content, "Hi! How can I help you today?")
        # 纯环境终态评分不需要裁判:没配 judge 也照常打分,并且 golden 重放得满分
        self.assertIsNotNone(sim.reward_info)
        self.assertEqual(sim.reward_info.reward, 1.0)
        self.assertTrue(sim.reward_info.db_check.db_match)

    def test_a_do_nothing_agent_fails_a_task_that_requires_changes(self):
        from IMAct.benchmarks.tau2.runner import Tau2Runner

        task = next(t for t in get_tasks("retail") if t.evaluation_criteria.actions and "DB" in [getattr(b, "value", b) for b in t.evaluation_criteria.reward_basis])
        runner = Tau2Runner(load_base_cell("retail"), make_agent=lambda p, s: None, user_llm=mock.Mock(), judge=None)
        msgs = [AssistantMessage(role="assistant", content="Hi! How can I help you today?", cost=0.0), UserMessage(role="user", content="hi ###STOP###", cost=0.0)]
        reward = Tau2Evaluator().evaluate(ConversationResult(messages=msgs, termination_reason=TerminationReason.USER_STOP, duration=0.0), runner.eval_task(task))
        self.assertEqual(reward.reward, 0.0)
        self.assertFalse(reward.db_check.db_match)

    def test_prompts_have_the_expected_structure(self):
        from IMAct.benchmarks.tau2.runner import Tau2Runner

        task = get_tasks("telecom", num_tasks=1)[0]
        runner = Tau2Runner(load_base_cell("telecom"), agent_llm=mock.Mock(), user_llm=mock.Mock())
        prepared = runner.prepare(task)
        self.assertTrue(prepared.system_prompt.startswith("<instructions>"))
        self.assertIn("<policy>", prepared.system_prompt)
        self.assertIn("Respond in English using Latin script", prepared.system_prompt)
        self.assertIn("<scenario>", prepared.user.system_prompt)
        self.assertIn("Persona Guidelines", prepared.user.system_prompt)
        self.assertIsNotNone(prepared.user.tools)  # telecom 的用户有工具,用的是带工具的守则
        self.assertIn("Make a tool call to perform an action", prepared.user.system_prompt)

    def test_rule_based_scoring_does_not_need_a_judge(self):
        """真实故障:telecom 只用环境断言评分,不需要裁判,但没配 judge 时整个 task 被当成"不评分"(reward 为 None)。"""
        from IMAct.benchmarks.tau2.runner import Tau2Runner

        task = next(t for t in get_tasks("telecom") if [getattr(b, "value", b) for b in t.evaluation_criteria.reward_basis] == ["ENV_ASSERTION"])
        msgs = [AssistantMessage(role="assistant", content="Hi! How can I help you today?", cost=0.0), UserMessage(role="user", content="bye ###STOP###", cost=0.0)]
        done = ConversationResult(messages=msgs, termination_reason=TerminationReason.USER_STOP, duration=0.0)
        runner = Tau2Runner(load_base_cell("telecom"), agent_llm=mock.Mock(), user_llm=mock.Mock(), judge=None)  # 没有裁判,但要评分
        self.assertIsNotNone(runner.evaluator)
        reward = runner.evaluate(task, done)
        self.assertEqual(reward.reward, 0.0)  # 什么都没做,环境断言不满足
        self.assertFalse(reward.info.get("evaluation_error"))
        self.assertTrue(all(not c.met for c in reward.env_assertions))
        # 对照:score=False 才表示不评分
        self.assertIsNone(Tau2Runner(load_base_cell("telecom"), agent_llm=mock.Mock(), user_llm=mock.Mock(), score=False).evaluator)

    def test_a_task_that_needs_the_judge_is_flagged_not_silently_scored_when_none_is_configured(self):
        from IMAct.benchmarks.tau2.runner import Tau2Runner

        task = next(t for t in get_tasks("retail") if "NL_ASSERTION" in [getattr(b, "value", b) for b in t.evaluation_criteria.reward_basis] and t.evaluation_criteria.nl_assertions)
        msgs = [AssistantMessage(role="assistant", content="Hi!", cost=0.0), UserMessage(role="user", content="bye ###STOP###", cost=0.0)]
        done = ConversationResult(messages=msgs, termination_reason=TerminationReason.USER_STOP, duration=0.0)
        reward = Tau2Runner(load_base_cell("retail"), agent_llm=mock.Mock(), user_llm=mock.Mock(), judge=None).evaluate(task, done)
        self.assertTrue(reward.info["evaluation_error"])
        self.assertIn("裁判", reward.info["note"])

    def test_constructor_needs_exactly_one_agent_source(self):
        from IMAct.benchmarks.tau2.runner import Tau2Runner

        with self.assertRaises(ValueError):
            Tau2Runner(load_base_cell("airline"), user_llm=mock.Mock())
        with self.assertRaises(ValueError):
            Tau2Runner(load_base_cell("airline"), agent_llm=mock.Mock(), make_agent=lambda p, s: None, user_llm=mock.Mock())


def _prepared_with(runner, task, user):
    from IMAct.benchmarks.tau2.runner import Prepared
    from IMAct.benchmarks.tau2.harness_env import Tau2HarnessEnvironment
    from IMAct.benchmarks.tau2.tasks import get_environment
    from IMAct.harness import EpisodeSpec

    env = Tau2HarnessEnvironment(get_environment(runner.domain), task)
    spec = EpisodeSpec(bench="tau2", user_id=task.id, episode_id=task.id, domain=runner.domain, instruction="", timestamp=None, system_prompt="", tools=[])
    return Prepared(env, spec, user, runner.cell.greeting(task), "")


if __name__ == "__main__":
    unittest.main()
