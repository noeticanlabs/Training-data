import json
import unittest
from pathlib import Path
from cohfield_school.grade_worlds import WORLDS


class CurriculumConsistency(unittest.TestCase):
    def test_manifest_consistent_with_runtime(self):
        path = Path(__file__).resolve().parents[1] / "curriculum" / "grades_v0_2.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["grade_count"], 10)
        self.assertEqual(manifest["world_count"], 30)
        for g in manifest["grades"]:
            self.assertEqual([w["slug"] for w in g["worlds"]], list(WORLDS[g["id"]]))
            self.assertEqual(g["graduation_gate"]["promotion_policy"], "blocked_until_independent_qualification")
            self.assertTrue(g["graduation_gate"]["require_external_signing"])
        self.assertFalse(manifest["evidence_policy"]["physical_evidence_authority"])
        self.assertFalse(manifest["evidence_policy"]["paid_observation_authority"])


if __name__ == "__main__":
    unittest.main()
