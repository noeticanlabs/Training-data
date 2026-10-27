"""Run locally: python -m unittest discover -s cohfield_school/tests -v"""
import unittest

from cohfield_school.world import ExperienceWorld, ZERO_HASH


class WorldTests(unittest.TestCase):
    def test_deterministic_experience_and_receipts(self):
        a = ExperienceWorld("fixed-seed")
        b = ExperienceWorld("fixed-seed")
        self.assertEqual(a.intervene(False, False), b.intervene(False, False))
        self.assertEqual(a.receipts(), b.receipts())
        self.assertEqual(a.receipts()[0]["previous_hash"], ZERO_HASH)

    def test_budget_and_no_duplicate_observations(self):
        s = ExperienceWorld("budget", observation_budget=1)
        s.intervene(False, False)
        with self.assertRaises(ValueError):
            s.intervene(False, False)
        with self.assertRaises(ValueError):
            s.intervene(True, True)

    def test_exam_never_exposes_expected_outcomes(self):
        s = ExperienceWorld("exam")
        s.intervene(True, True)
        s.intervene(False, False)
        cases = s.begin_exam()
        self.assertEqual(len(cases), 2)
        self.assertTrue(all("outcome" not in row for row in cases))
        for row in cases:
            receipt = s.predict(row["a"], row["b"], False)
            self.assertNotIn("outcome", receipt)
        result = s.finalize()
        self.assertFalse(result["certified"])
        self.assertFalse(result["retention_verified"])
        self.assertFalse(result["transfer_verified"])
        self.assertEqual(result["provisional_score"]["total"], 2)

    def test_exam_cannot_retrain(self):
        s = ExperienceWorld("state")
        s.intervene(False, True)
        s.begin_exam()
        with self.assertRaises(ValueError):
            s.intervene(True, False)
        with self.assertRaises(ValueError):
            s.finalize()

    def test_invalid_type_or_replay_rejected(self):
        s = ExperienceWorld("types")
        with self.assertRaises(ValueError):
            s.intervene(1, False)
        s.intervene(False, False)
        cases = s.begin_exam()
        c = cases[0]
        with self.assertRaises(ValueError):
            s.predict(c["a"], c["b"], 1)
        s.predict(c["a"], c["b"], True)
        with self.assertRaises(ValueError):
            s.predict(c["a"], c["b"], False)

    def test_teacher_does_not_advertise_hidden_rule(self):
        s = ExperienceWorld("hidden")
        state = s.public_state()
        self.assertNotIn("rule", state)
        self.assertNotIn("secret", state)
        self.assertEqual(state["authority"], "none")


if __name__ == "__main__":
    unittest.main()
