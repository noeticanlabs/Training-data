#!/usr/bin/env python3
import hashlib, importlib.util, json, tempfile, unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("ingest",HERE/"ingest_multi_perspective_invariant_v001.py")
ingest=importlib.util.module_from_spec(spec); spec.loader.exec_module(ingest)
class Tests(unittest.TestCase):
    def test_quality_flags_control_bytes(self):
        self.assertIn("contains_control_bytes",ingest.quality_flags("abc\x01def"))
        self.assertEqual(ingest.quality_flags("ordinary formula"),[])
    def test_quality_flags_long(self):
        self.assertIn("label_gt_2000_chars",ingest.quality_flags("x"*2001))
    def test_hash_is_stable(self):
        self.assertEqual(ingest.sha256_bytes(b"abc"),hashlib.sha256(b"abc").hexdigest())
    def test_scientific_boundary_literals(self):
        text=(HERE/"config/multi_perspective_invariant_v001.json").read_text(encoding="utf-8")
        cfg=json.loads(text)
        self.assertFalse(cfg["governance"]["automatic_satisfaction_claim"])
        self.assertFalse(cfg["governance"]["automatic_training_admission"])
        self.assertTrue(cfg["governance"]["support_is_separate_from_relevance"])
if __name__=="__main__": unittest.main()
