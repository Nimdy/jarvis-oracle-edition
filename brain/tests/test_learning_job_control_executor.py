import os
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from skills.executors.control import ControlCollectExecutor, ControlVerifyExecutor


class ControlExecutorContractTests(unittest.TestCase):
    def test_collect_uses_metric_contract_not_hardcoded_episodes(self) -> None:
        executor = ControlCollectExecutor()
        job = SimpleNamespace(
            skill_id="custom_control_v1",
            phase="collect",
            data={"counters": {"sandbox_runs": 4}},
            plan={"phases": [
                {"name": "collect", "exit_conditions": ["metric:sandbox_runs>=10"]},
            ]},
        )

        result = executor.run(
            job,
            {
                "sim_available": True,
                "episodes_collected_this_tick": 2,
            },
        )

        self.assertEqual(result.metric_updates["sandbox_runs"], 6.0)

    def test_verify_uses_required_evidence_names_when_present(self) -> None:
        executor = ControlVerifyExecutor()
        job = SimpleNamespace(
            skill_id="custom_control_v1",
            evidence={"required": [
                "sim:test_camera_pan_10runs",
                "real:test_camera_pan_3runs_user_present",
            ]},
        )

        result = executor.run(
            job,
            {
                "user_present": True,
                "sim_test_pass": True,
                "real_test_pass": True,
                "now_iso": "2026-03-20T10:00:00Z",
            },
        )

        self.assertEqual(result.evidence["result"], "pass")
        test_names = [t["name"] for t in result.evidence["tests"]]
        self.assertIn("sim:test_camera_pan_10runs", test_names)
        self.assertIn("real:test_camera_pan_3runs_user_present", test_names)


if __name__ == "__main__":
    unittest.main()
