"""Additional internal oracle tests; these do NOT test CohField learning."""
import unittest
from cohfield_school.grade_worlds import WORLDS, GradeSession, _case, _gate


class GateInvariantTests(unittest.TestCase):
    def test_boolean_training_determines_unique_candidate_gate(self):
        for grade in ("G2", "G5"):
            for world in WORLDS[grade]:
                for i in range(12):
                    seed = f"oracle-{i}"
                    train = _case(grade, world, seed, "train")
                    exam = _case(grade, world, seed, "exam")
                    compatible = []
                    for rule in ("AND", "OR", "XOR", "NAND"):
                        if all(_gate(rule, bool(int(key.split(":")[1])), bool(int(key.split(":")[2]))) == item["output"]
                               for key, item in train.probes.items()):
                            compatible.append(rule)
                    self.assertEqual(len(compatible), 1)
                    self.assertEqual(exam.expected, _gate(compatible[0], True, False))
                    self.assertEqual(len(exam.probes), 0)
                    if grade == "G5":
                        self.assertNotEqual(train.public["controls"], exam.public["controls"])

    def test_planning_answers_depend_on_initial_state(self):
        for world in WORLDS["G7"]:
            for i in range(20):
                exam = _case("G7", world, f"plan-{i}", "exam")
                state = exam.public["initial"]
                plan = (["take_key"] if state["door_locked"] and not state["has_key"] else [])
                plan += (["unlock"] if state["door_locked"] else [])
                plan += ["open"]
                self.assertEqual(exam.expected, plan)

    def test_autonomy_probe_respects_costs(self):
        for world in WORLDS["G9"]:
            for i in range(20):
                exam = _case("G9", world, f"probe-{i}", "exam")
                costs = exam.public["probe_costs"]
                winner = min(("measure_sensor", "measure_actuator"), key=costs.get)
                self.assertEqual(exam.expected, winner)
                self.assertLessEqual(costs[winner], exam.public["probe_budget"])

    def test_no_self_certification_for_correct_responses(self):
        s = GradeSession("G4", WORLDS["G4"][0], "retention-probe")
        revealed = s.intervene("inspect")["observation"]["code"]
        s.begin_exam()
        result = s.submit(revealed)
        self.assertEqual(result["provisional_score"]["correct"], 1)
        self.assertFalse(result["retention_verified"])
        self.assertFalse(result["certified"])


if __name__ == "__main__":
    unittest.main()
