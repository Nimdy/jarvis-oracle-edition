import unittest

from jarvis_eval import dashboard_adapter as da


class MaturityTrackerHighwaterTests(unittest.TestCase):
    def _categories(self, *, status: str, current, threshold):
        return [
            {
                "id": "test_category",
                "label": "Test Category",
                "gates": [
                    {
                        "id": "test_gate",
                        "label": "Test Gate",
                        "status": status,
                        "current": current,
                        "threshold": threshold,
                        "display": f"{current} / {threshold}",
                    }
                ],
            }
        ]

    def test_merge_marks_ever_met_on_first_activation(self) -> None:
        categories = self._categories(status="active", current=10, threshold=10)
        state = {"version": 1, "updated_at": 0.0, "gates": {}}

        ever_count, changed = da._merge_maturity_highwater(categories, state, now_ts=100.0)

        self.assertTrue(changed)
        self.assertEqual(ever_count, 1)
        gate = categories[0]["gates"][0]
        self.assertTrue(gate["ever_met"])
        self.assertEqual(gate["ever_met_ts"], 100.0)
        self.assertEqual(float(state["gates"]["test_gate"]["best_current"]), 10.0)

    def test_merge_preserves_ever_met_after_regression(self) -> None:
        categories = self._categories(status="progress", current=4, threshold=10)
        state = {
            "version": 1,
            "updated_at": 0.0,
            "gates": {
                "test_gate": {
                    "ever_met": True,
                    "first_met_ts": 42.0,
                    "best_current": 12.0,
                    "threshold": 10.0,
                }
            },
        }

        ever_count, changed = da._merge_maturity_highwater(categories, state, now_ts=200.0)

        self.assertFalse(changed)
        self.assertEqual(ever_count, 1)
        gate = categories[0]["gates"][0]
        self.assertTrue(gate["ever_met"])
        self.assertEqual(gate["ever_met_ts"], 42.0)
        self.assertEqual(float(gate["best_current"]), 12.0)
        self.assertEqual(float(state["gates"]["test_gate"]["best_current"]), 12.0)

    def test_merge_keeps_locked_gate_not_met(self) -> None:
        categories = self._categories(status="locked", current=None, threshold=1)
        state = {"version": 1, "updated_at": 0.0, "gates": {}}

        ever_count, changed = da._merge_maturity_highwater(categories, state, now_ts=300.0)

        self.assertTrue(changed)  # threshold metadata gets initialized on first sighting
        self.assertEqual(ever_count, 0)
        gate = categories[0]["gates"][0]
        self.assertFalse(gate["ever_met"])
        self.assertEqual(gate["ever_met_ts"], 0.0)


class PvlContractHighwaterTests(unittest.TestCase):
    def test_merge_marks_contract_ever_passed(self) -> None:
        verdicts = [
            {
                "contract_id": "hemisphere_ready",
                "status": "pass",
                "evidence": "snapshot hemisphere.total_networks >= 1",
            }
        ]
        state = {"version": 1, "updated_at": 0.0, "contracts": {}}

        ever_count, changed = da._merge_pvl_highwater(verdicts, state, now_ts=500.0)

        self.assertTrue(changed)
        self.assertEqual(ever_count, 1)
        verdict = verdicts[0]
        self.assertTrue(verdict["ever_passed"])
        self.assertEqual(verdict["ever_pass_ts"], 500.0)
        self.assertEqual(
            state["contracts"]["hemisphere_ready"]["last_pass_evidence"],
            "snapshot hemisphere.total_networks >= 1",
        )

    def test_merge_preserves_contract_ever_passed_after_regression(self) -> None:
        verdicts = [
            {
                "contract_id": "hemisphere_ready",
                "status": "fail",
                "evidence": "snapshot hemisphere.total_networks = 0",
            }
        ]
        state = {
            "version": 1,
            "updated_at": 0.0,
            "contracts": {
                "hemisphere_ready": {
                    "ever_passed": True,
                    "first_pass_ts": 250.0,
                    "last_pass_ts": 300.0,
                    "last_pass_evidence": "snapshot hemisphere.total_networks >= 1",
                }
            },
        }

        ever_count, changed = da._merge_pvl_highwater(verdicts, state, now_ts=600.0)

        self.assertFalse(changed)
        self.assertEqual(ever_count, 1)
        verdict = verdicts[0]
        self.assertTrue(verdict["ever_passed"])
        self.assertEqual(verdict["ever_pass_ts"], 250.0)
        self.assertEqual(
            verdict["last_pass_evidence"],
            "snapshot hemisphere.total_networks >= 1",
        )


if __name__ == "__main__":
    unittest.main()
