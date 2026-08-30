#!/usr/bin/env python3
"""Regression tests for domain evidence-strength diagnostics."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import analyze_domain_evidence as analysis


POLICY = {
    "primary_min_score": 2.0,
    "secondary_min_score": 1.0,
    "max_secondary_domains": 5,
    "primary_tie_policy": "mark_ambiguous",
    "allow_fallback_primary": True,
}


class DomainEvidenceAnalysisTests(unittest.TestCase):
    def write_assignments(self, records: list[dict]) -> str:
        temp = tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".jsonl")
        with temp:
            for record in records:
                temp.write(json.dumps(record, sort_keys=True) + "\n")
        self.addCleanup(lambda: Path(temp.name).unlink(missing_ok=True))
        return temp.name

    def test_fallback_and_legacy_only_primary_are_measured(self) -> None:
        path = self.write_assignments([
            {
                "primary_domain_id": "D05",
                "primary_domain_candidates": ["D05"],
                "domain_ids": ["D05"],
                "secondary_domain_ids": [],
                "cross_domain": False,
                "scores": {"D05": 1.0},
                "evidence": [
                    {"kind": "legacy_tag", "value": "tags:lean-math", "domain_id": "D05", "weight": 1.0}
                ],
            },
            {
                "primary_domain_id": "D08",
                "primary_domain_candidates": ["D08"],
                "domain_ids": ["D08"],
                "secondary_domain_ids": [],
                "cross_domain": False,
                "scores": {"D08": 2.0},
                "evidence": [
                    {"kind": "keyword", "value": "title:power grid", "domain_id": "D08", "weight": 2.0}
                ],
            },
        ])

        report = analysis.summarize(path, POLICY)

        self.assertEqual(report["total_assignments"], 2)
        self.assertEqual(report["fallback_primary_count"], 1)
        self.assertEqual(report["legacy_only_primary_count"], 1)
        self.assertEqual(report["single_evidence_primary_count"], 2)
        self.assertEqual(report["fallback_primary_by_domain"], {"D05": 1})

    def test_d00_and_ambiguity_are_not_counted_as_resolved_primary(self) -> None:
        path = self.write_assignments([
            {
                "primary_domain_id": "D00",
                "primary_domain_candidates": ["D00"],
                "domain_ids": ["D00"],
                "secondary_domain_ids": [],
                "cross_domain": False,
                "scores": {"D00": 0.0},
                "evidence": [],
            },
            {
                "primary_domain_id": None,
                "primary_domain_candidates": ["D01", "D05"],
                "domain_ids": ["D01", "D05"],
                "secondary_domain_ids": [],
                "cross_domain": True,
                "scores": {"D01": 2.0, "D05": 2.0},
                "evidence": [
                    {"kind": "keyword", "value": "title:theorem", "domain_id": "D01", "weight": 2.0},
                    {"kind": "keyword", "value": "title:theorem prover", "domain_id": "D05", "weight": 2.0},
                ],
            },
        ])

        report = analysis.summarize(path, POLICY)

        self.assertEqual(report["d00_count"], 1)
        self.assertEqual(report["ambiguous_primary_count"], 1)
        self.assertEqual(report["resolved_substantive_primary_count"], 0)
        self.assertEqual(report["fallback_primary_count"], 0)


if __name__ == "__main__":
    unittest.main()
