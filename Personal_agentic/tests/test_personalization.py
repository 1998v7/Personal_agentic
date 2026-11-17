"""PersonalizationRunner(VitaBench 的 subtask 编排)的测试。不调用任何模型:对话与裁判都用替身。"""

import unittest
from unittest import mock

from IMAct.benchmarks.vitabench.data_model.personalization_task import PersonalizationTask, SubTask
from IMAct.benchmarks.vitabench.personalization import (
    PersonalizationRunner, SubtaskResult, overall_termination, rubrics_of,
)
from IMAct.benchmarks.vitabench.data_model.tasks import EvaluationCriteria
from IMAct.evaluation.base import RewardInfo
from IMAct.exec_layer.config import MemoryRouting
from IMAct.harness.conversation import ConversationResult, TerminationReason
from IMAct.harness.messages import AssistantMessage, UserMessage


def subtask(sid="sub_1", **kw):
    kw.setdefault("environment", {"time": "2026-03-02 12:00:00"})  # 周一
    return SubTask(subtask_id=sid, domain=kw.pop("domain", "delivery"), instruction=kw.pop("instruction", "点个外卖"), **kw)


def task(*subtasks):
    return PersonalizationTask(id="t1", user_id="U1", user_profile={"user_id": "U1", "address": "演示办公地址"}, subtasks=list(subtasks))


def runner(t, judge=None, **kw):
    rt = mock.Mock(trace=None)
    rt.cfg.memory = MemoryRouting()
    return PersonalizationRunner(t, rt=rt, user_llm=mock.Mock(), judge=judge, run_id="r", **kw)


def conv(reason=TerminationReason.USER_STOP, n=2, states=None):
    msgs = [UserMessage(role="user", content="u", cost=0.5)] + [AssistantMessage(role="assistant", content="a", cost=1.0)] * (n - 1)
    return ConversationResult(messages=msgs, termination_reason=reason, duration=1.0, states=states or {"old_states": [], "new_states": []})


class PrepareTests(unittest.TestCase):
    def test_spec_environment_and_user_are_built_from_the_subtask(self):
        r = runner(task(subtask(instruction="帮我点个午餐")), language="chinese")
        spec, env, user = r.prepare(r.task.subtasks[0])
        self.assertEqual((spec.bench, spec.user_id, spec.episode_id, spec.domain, spec.instruction),
                         ("vitabench", "U1", "sub_1", "delivery", "帮我点个午餐"))
        self.assertIn("2026-03-02 12:00:00 星期一", spec.system_prompt)
        self.assertIn("- address: 演示办公地址", spec.system_prompt)
        self.assertEqual(spec.tools, env.tool_schemas())  # agent 看到的工具就是环境实际能执行的工具
        self.assertIn("create_delivery_order", {t["function"]["name"] for t in spec.tools})
        self.assertEqual(user.opening, "帮我点个午餐")  # 开场白就是 subtask 指令,不经过模型
        self.assertIn("帮我点个午餐", user.system_prompt)
        self.assertIn("演示办公地址", user.system_prompt)  # persona = 用户资料

    def test_clock_without_time_carries_over_from_the_previous_subtask(self):
        r = runner(task(subtask("sub_1"), subtask("sub_2", environment={})))
        first, _, _ = r.prepare(r.task.subtasks[0])
        second, _, _ = r.prepare(r.task.subtasks[1])
        self.assertIn("2026-03-02 12:00:00 星期一", second.system_prompt)
        self.assertEqual(first.system_prompt, second.system_prompt)

    def test_each_subtask_gets_its_own_time_and_a_fresh_environment(self):
        r = runner(task(subtask("sub_1"), subtask("sub_2", environment={"time": "2026-03-04 09:00:00"})))
        _, env1, _ = r.prepare(r.task.subtasks[0])
        spec2, env2, _ = r.prepare(r.task.subtasks[1])
        self.assertIn("2026-03-04 09:00:00 星期三", spec2.system_prompt)
        self.assertIsNot(env1, env2)

    def test_cross_domain_subtask_merges_tools(self):
        r = runner(task(subtask(domain="delivery,ota")))
        spec, _, _ = r.prepare(r.task.subtasks[0])
        names = {t["function"]["name"] for t in spec.tools}
        self.assertTrue({"create_delivery_order"} <= names and len(names) > 30)

    def test_completion_guard_only_reaches_the_user_prompt(self):
        r = runner(task(subtask()), completion_guard=True)
        spec, _, user = r.prepare(r.task.subtasks[0])
        self.assertIn("补充结束条件说明", user.system_prompt)
        self.assertNotIn("补充结束条件说明", spec.system_prompt)

    def test_seed_is_passed_to_the_user_simulator(self):
        r = runner(task(subtask()), seed=300)
        _, _, user = r.prepare(r.task.subtasks[0])
        self.assertEqual(user.seed, 300)


class RunTests(unittest.TestCase):
    def test_run_aggregates_rewards_states_messages_and_costs(self):
        t = task(subtask("sub_1"), subtask("sub_2"), subtask("sub_3", evaluation_criteria=None))
        r = runner(t, judge=mock.Mock())
        results = [conv(states={"old_states": ["o1"], "new_states": []}), conv(states={"old_states": [], "new_states": ["n2"]}), conv()]
        rewards = [RewardInfo(reward=1.0), RewardInfo(reward=0.0), None]
        with mock.patch.object(PersonalizationRunner, "run_subtask", side_effect=[SubtaskResult(s.subtask_id, i, c) for i, (s, c) in enumerate(zip(t.subtasks, results))]), \
             mock.patch.object(PersonalizationRunner, "evaluate_subtask", side_effect=rewards):
            sim = r.run()
        self.assertEqual(sim.reward_info.reward, 0.5)  # 没有评分标准的 subtask 不计入
        self.assertEqual(sim.reward_info.info["num_subtasks"], 3)
        self.assertEqual(sim.reward_info.info["num_evaluated"], 2)
        self.assertEqual(sim.reward_info.info["subtask_rewards"], {"subtask_0_reward": 1.0, "subtask_1_reward": 0.0})
        self.assertEqual(len(sim.messages), 6)
        self.assertEqual(sim.states, {"old_states": ["o1"], "new_states": ["n2"]})
        self.assertEqual((sim.agent_cost, sim.user_cost), (3.0, 1.5))
        self.assertEqual(sim.termination_reason, TerminationReason.USER_STOP)
        self.assertEqual(sim.task_id, "t1")

    def test_subtasks_run_in_order_with_their_index(self):
        t = task(subtask("sub_1"), subtask("sub_2"))
        r = runner(t)
        calls = []

        def fake(st, index=0, **kw):
            calls.append((st.subtask_id, index))
            return SubtaskResult(st.subtask_id, index, conv())

        with mock.patch.object(PersonalizationRunner, "run_subtask", side_effect=fake):
            r.run()
        self.assertEqual(calls, [("sub_1", 0), ("sub_2", 1)])

    def test_aggregate_with_nothing_evaluated_is_zero_with_a_note(self):
        r = runner(task(subtask()))
        reward = r.aggregate([None])
        self.assertEqual(reward.reward, 0.0)
        self.assertEqual(reward.info, {"note": "No subtasks were evaluated"})


class EvaluateSubtaskTests(unittest.TestCase):
    def result(self):
        return SubtaskResult("sub_1", 0, conv())

    def test_no_criteria_or_no_judge_means_not_scored(self):
        st = subtask(evaluation_criteria={"overall_rubrics": ["含肉"]})
        self.assertIsNone(runner(task(st), judge=None).evaluate_subtask(st, self.result()))  # 诊断模式:不打分
        bare = subtask()
        self.assertIsNone(runner(task(bare), judge=mock.Mock()).evaluate_subtask(bare, self.result()))

    def test_judge_failure_becomes_a_flagged_zero_instead_of_crashing_the_task(self):
        st = subtask(evaluation_criteria={"overall_rubrics": ["含肉"]})
        r = runner(task(st), judge=mock.Mock())
        r.evaluator = mock.Mock()
        r.evaluator.evaluate.side_effect = RuntimeError("judge down")
        reward = r.evaluate_subtask(st, self.result())
        self.assertEqual(reward.reward, 0.0)
        self.assertTrue(reward.info["evaluation_error"])
        self.assertIn("judge down", reward.info["note"])

    def test_evaluator_receives_instruction_rubrics_and_environment_time(self):
        st = subtask(instruction="点午餐", evaluation_criteria={"overall_rubrics": ["含肉", "及时"]})
        r = runner(task(st), judge=mock.Mock())
        r.evaluator = mock.Mock()
        r.evaluator.evaluate.return_value = RewardInfo(reward=1.0)
        r.evaluate_subtask(st, self.result())
        rubric_task = r.evaluator.evaluate.call_args.args[1]
        self.assertEqual((rubric_task.instruction, rubric_task.rubrics, rubric_task.env_time), ("点午餐", ["含肉", "及时"], "2026-03-02 12:00:00"))


class HelperTests(unittest.TestCase):
    def test_rubrics_come_from_expected_states_first_then_overall_without_duplicates(self):
        criteria = EvaluationCriteria(expected_states=[{"state_rubrics": ["订单含烧烤", "及时送达"]}, {"state_rubrics": None}],
                                      overall_rubrics=["及时送达", "已支付"])
        self.assertEqual(rubrics_of(criteria), ["订单含烧烤", "及时送达", "已支付"])
        self.assertEqual(rubrics_of(None), [])

    def test_overall_termination(self):
        T = TerminationReason
        self.assertEqual(overall_termination([]), T.MAX_STEPS)
        self.assertEqual(overall_termination([T.USER_STOP, T.AGENT_STOP]), T.AGENT_STOP)  # 取最后一个
        self.assertEqual(overall_termination([T.TOO_MANY_ERRORS, T.USER_STOP]), T.TOO_MANY_ERRORS)  # 任何一个出错就整体出错


if __name__ == "__main__":
    unittest.main()
