"""Decision tests for dependency and resource-selection behavior."""
import json
from pathlib import Path
import unittest

from src.plan import choose, plan


class SchedulingTests(unittest.TestCase):
    def setUp(self):
        self.task = json.loads((Path(__file__).resolve().parents[1] / "examples/two_objects.json").read_text())

    def test_separated_moves_recruit_second_arm(self):
        self.assertEqual(choose(self.task)["chosen_arms"], 2)
        self.assertTrue(plan(self.task, 2)["parallel"])

    def test_dependency_changes_execution_order(self):
        self.task["dependencies"] = [["blue", "red"]]
        p = plan(self.task, 2)
        self.assertFalse(p["parallel"])
        self.assertEqual([s["object"] for s in p["steps"]], ["blue", "red"])
        self.assertLessEqual(p["steps"][0]["end_s"], p["steps"][1]["start_s"])
        self.assertEqual(choose(self.task)["chosen_arms"], 1)

    def test_cycle_fails(self):
        self.task["dependencies"] = [["blue", "red"], ["red", "blue"]]
        with self.assertRaisesRegex(ValueError, "cycle"):
            plan(self.task, 1)

    def test_crossing_paths_serialize(self):
        a, b = self.task["objects"]
        a["goal_m"], b["goal_m"] = b["goal_m"], a["goal_m"]
        self.assertFalse(plan(self.task, 2)["parallel"])

    def test_high_recruitment_cost_keeps_one_arm(self):
        self.task["handoff_overhead_s"] = 30
        self.assertEqual(choose(self.task)["chosen_arms"], 1)

    def test_invalid_coordinates_fail(self):
        self.task["objects"][0]["start_m"] = [float("nan"), .1]
        with self.assertRaises(ValueError):
            plan(self.task, 1)


if __name__ == "__main__":
    unittest.main()
