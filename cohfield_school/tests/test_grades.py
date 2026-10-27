"""Local protocol tests: python -m unittest discover -s cohfield_school/tests -v"""
import json
import unittest

from cohfield_school.grade_worlds import GradeSession, WORLDS


class GradeWorldTests(unittest.TestCase):
    def test_ten_grades_three_worlds_each(self):
        self.assertEqual(list(WORLDS), [f"G{i}" for i in range(10)])
        self.assertTrue(all(len(w) == 3 for w in WORLDS.values()))

    def test_all_grades_run_without_claiming_certification(self):
        for grade, worlds in WORLDS.items():
            for world in worlds:
                with self.subTest(grade=grade, world=world):
                    s = GradeSession(grade, world, "seed-10", budget=5)
                    train = s.training_prompt()
                    self.assertEqual(train["grade"], grade)
                    for probe in train["allowed_probes"]:
                        obs = s.intervene(probe)
                        self.assertEqual(obs["evidence_class"], "simulated_schooling_unverified")
                    exam = s.begin_exam()
                    self.assertNotIn("expected", json.dumps(exam))
                    self.assertNotIn("answer", exam)
                    with self.assertRaises(ValueError):
                        s.intervene(train["allowed_probes"][0] if train["allowed_probes"] else "any")
                    result = s.submit("intentionally-wrong-answer")
                    self.assertFalse(result["certified"])
                    self.assertFalse(result["retention_verified"])
                    self.assertFalse(result["transfer_verified"])
                    self.assertFalse(result["independent_exam_verified"])

    def test_deterministic_replay(self):
        for grade, worlds in WORLDS.items():
            a = GradeSession(grade, worlds[0], "same")
            b = GradeSession(grade, worlds[0], "same")
            self.assertEqual(a.training_prompt(), b.training_prompt())
            self.assertEqual(a.begin_exam(), b.begin_exam())
            self.assertEqual(a.receipt_export(), b.receipt_export())

    def test_wrong_grade_and_repeated_submission(self):
        with self.assertRaises(ValueError):
            GradeSession("G10", "invented", "seed")
        s = GradeSession("G4", WORLDS["G4"][0], "retention")
        s.intervene("inspect")
        e = s.begin_exam()
        self.assertNotIn("code", e["challenge"])
        s.submit(None)
        with self.assertRaises(ValueError):
            s.submit(None)

    def test_budget_fail_closed(self):
        s = GradeSession("G2", "switchboard", "limited", budget=1)
        probes = s.training_prompt()["allowed_probes"]
        self.assertGreaterEqual(len(probes), 2)
        s.intervene(probes[0])
        with self.assertRaises(ValueError):
            s.intervene(probes[1])
        with self.assertRaises(ValueError):
            s.intervene(probes[0])

    def test_exam_has_no_training_probe_access(self):
        s = GradeSession("G3", "moving-cart", "blind")
        s.intervene("tick")
        exam = s.begin_exam()
        self.assertFalse(exam["training_probe_access"])
        self.assertNotIn("position_after", exam["challenge"])


if __name__ == "__main__":
    unittest.main()
