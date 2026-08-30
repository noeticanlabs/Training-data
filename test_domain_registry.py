#!/usr/bin/env python3
"""Deterministic regression tests for the governed domain sidecar pipeline."""
from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import build_domain_indexes as builder
import build_domain_validation_receipt as receipt_builder
import validate_domain_calibration as calibration
import validate_domain_indexes as structural


POLICY = {
    "primary_min_score": 2.0,
    "secondary_min_score": 1.0,
    "max_secondary_domains": 5,
    "primary_tie_policy": "mark_ambiguous",
}


class DomainRegistryRegressionTests(unittest.TestCase):
    def test_rule_identity_and_assignment_id_change_with_rules(self) -> None:
        rules_v1 = {"rules_version": "0.02", "keywords": {"D01": ["theorem"]}}
        rules_v2 = {"rules_version": "0.02", "keywords": {"D01": ["lemma"]}}
        identity_v1 = builder.classifier_identity(rules_v1)
        identity_v2 = builder.classifier_identity(rules_v2)

        self.assertTrue(identity_v1.startswith("domain_rules_v0.02_sha256_"))
        self.assertNotEqual(identity_v1, identity_v2)
        self.assertNotEqual(
            builder.stable_assignment_id("research_record", "r1", "0.01", identity_v1),
            builder.stable_assignment_id("research_record", "r1", "0.01", identity_v2),
        )

    def test_equal_top_scores_are_explicitly_ambiguous(self) -> None:
        primary, candidates, domains, secondary = builder.select_domains(
            {"D01": 2.0, "D05": 2.0, "D12": 1.0}, POLICY
        )

        self.assertIsNone(primary)
        self.assertEqual(candidates, ["D01", "D05"])
        self.assertEqual(domains, ["D01", "D05", "D12"])
        self.assertEqual(secondary, ["D12"])

    def test_all_d00_distribution_fails_calibration(self) -> None:
        stats = {
            "total_assignments": 10,
            "primary_counts": {"D00": 10},
            "primary_candidate_counts": {"D00": 10},
            "membership_counts": {"D00": 10},
            "cross_domain_count": 0,
            "ambiguous_primary_count": 0,
        }
        limits = {
            "max_cross_domain_rate": 0.60,
            "max_avg_domains_per_record": 2.50,
            "max_primary_share_any_substantive_domain": 0.70,
            "max_unclassified_rate": 0.85,
            "max_ambiguous_primary_rate": 0.25,
            "min_substantive_primary_domains": 1,
            "min_substantive_membership_domains": 1,
        }
        errors: list[str] = []

        calibration.check_layer("research", stats, limits, errors)

        self.assertTrue(any("unclassified rate" in error for error in errors))

    def test_universal_multilabel_distribution_fails_calibration(self) -> None:
        stats = {
            "total_assignments": 10,
            "primary_counts": {"D01": 5, "D02": 5},
            "primary_candidate_counts": {"D01": 5, "D02": 5},
            "membership_counts": {"D01": 10, "D02": 10},
            "cross_domain_count": 10,
            "ambiguous_primary_count": 0,
        }
        limits = {
            "max_cross_domain_rate": 0.60,
            "max_avg_domains_per_record": 2.50,
            "max_primary_share_any_substantive_domain": 0.70,
            "max_unclassified_rate": 0.85,
            "max_ambiguous_primary_rate": 0.25,
            "min_substantive_primary_domains": 1,
            "min_substantive_membership_domains": 1,
        }
        errors: list[str] = []

        calibration.check_layer("research", stats, limits, errors)

        self.assertTrue(any("cross-domain rate" in error for error in errors))

    def test_d00_training_regression_fails_inheritance_validation(self) -> None:
        research_assignment = {
            "subject_id": "r1",
            "primary_domain_id": "D05",
            "primary_domain_candidates": ["D05"],
            "domain_ids": ["D05"],
            "secondary_domain_ids": [],
            "scores": {"D05": 2.0},
        }
        collapsed_training_assignment = {
            "subject_id": "t1",
            "primary_domain_id": "D00",
            "primary_domain_candidates": ["D00"],
            "domain_ids": ["D00"],
            "secondary_domain_ids": [],
            "cross_domain": False,
            "scores": {"D00": 0.0},
            "evidence": [],
        }
        registry = {"assignment_policy": POLICY}
        errors: list[str] = []

        structural.validate_training_inheritance(
            {"t1": collapsed_training_assignment},
            {"t1": ["r1"]},
            {"r1": research_assignment},
            registry,
            errors,
        )

        self.assertTrue(any("inherited primary domain" in error for error in errors))
        self.assertTrue(any("inherited domains" in error for error in errors))

    def test_malformed_assignment_is_reported_without_validator_crash(self) -> None:
        schema = json.loads(Path("schemas/domain_assignment.schema.json").read_text(
            encoding="utf-8"
        ))
        errors: list[str] = []
        with tempfile.TemporaryDirectory() as temp_dir:
            assignment_path = Path(temp_dir) / "assignments.jsonl"
            assignment_path.write_text("[]\n", encoding="utf-8")

            assignments, _, _ = structural.validate_assignment_file(
                str(assignment_path),
                "research_record",
                {"r1": "source-hash"},
                {"D00", "D01"},
                structural.Draft202012Validator(schema),
                "0.01",
                "domain_rules_v0.02_sha256_0000000000000000",
                POLICY,
                errors,
            )

        self.assertEqual(assignments, {})
        self.assertTrue(any("assignment must be an object" in error for error in errors))

    def test_end_to_end_fixture_build_and_validation_pass(self) -> None:
        research_records = [
            {
                "record_id": "r1",
                "title": "Theorem prover",
                "content": {"text": "A focused formal artifact."},
                "project_ids": [],
                "tags": [],
                "provenance": {"source_hash": "research-hash-1"},
            },
            {
                "record_id": "r2",
                "title": "Power grid stability analysis",
                "content": {"text": "A controlled electrical experiment."},
                "project_ids": [],
                "tags": [],
                "provenance": {"source_hash": "research-hash-2"},
            },
            {
                "record_id": "r3",
                "title": "Data pipeline reproducibility",
                "content": {"text": "A governed corpus experiment."},
                "project_ids": [],
                "tags": [],
                "provenance": {"source_hash": "research-hash-3"},
            },
            {
                "record_id": "r4",
                "title": "Unmapped observation",
                "content": {"text": "No controlled vocabulary match."},
                "project_ids": [],
                "tags": [],
                "provenance": {"source_hash": "research-hash-4"},
            },
        ]
        training_records = [
            {
                "training_example_id": "t1",
                "source_record_ids": ["r1"],
                "transform": {"content_hash": "training-hash-1"},
            },
            {
                "training_example_id": "t2",
                "source_record_ids": ["r4"],
                "transform": {"content_hash": "training-hash-2"},
            },
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            research_path = root / "research.jsonl"
            training_path = root / "training.jsonl"
            research_domain_dir = root / "research" / "domains"
            training_domain_dir = root / "training" / "domains"
            manifest_path = root / "domain_manifest.json"
            structural_report_path = root / "domain_structural_report.json"
            calibration_report_path = root / "domain_calibration_report.json"
            receipt_path = root / "domain_validation_receipt.json"
            research_path.write_text(
                "".join(json.dumps(record, sort_keys=True) + "\n" for record in research_records),
                encoding="utf-8",
            )
            training_path.write_text(
                "".join(json.dumps(record, sort_keys=True) + "\n" for record in training_records),
                encoding="utf-8",
            )
            build_args = [
                "--research", str(research_path),
                "--training", str(training_path),
                "--registry", "config/domain_registry.json",
                "--rules", "config/domain_rules.json",
                "--research-domain-dir", str(research_domain_dir),
                "--training-domain-dir", str(training_domain_dir),
                "--manifest", str(manifest_path),
            ]

            with contextlib.redirect_stdout(io.StringIO()):
                build_status = builder.main(build_args)
                first_research_assignments = (
                    research_domain_dir / "assignments.jsonl"
                ).read_bytes()
                first_training_assignments = (
                    training_domain_dir / "assignments.jsonl"
                ).read_bytes()
                rebuild_status = builder.main(build_args)
                validation_status = structural.main([
                    "--research", str(research_path),
                    "--training", str(training_path),
                    "--research-assignments", str(research_domain_dir / "assignments.jsonl"),
                    "--training-assignments", str(training_domain_dir / "assignments.jsonl"),
                    "--research-index-dir", str(root / "research" / "indexes" / "domains"),
                    "--training-index-dir", str(root / "training" / "indexes" / "domains"),
                    "--registry", "config/domain_registry.json",
                    "--rules", "config/domain_rules.json",
                    "--schema", "schemas/domain_assignment.schema.json",
                    "--manifest", str(manifest_path),
                    "--report", str(structural_report_path),
                ])

                fixture_rules_version = json.loads(
                    Path("config/domain_rules.json").read_text(encoding="utf-8")
                )["rules_version"]
                calibration_report_path.write_text(json.dumps({
                    "status": "PASS",
                    "rules_version": fixture_rules_version,
                    "interpretation": "Fixture-only calibration receipt test.",
                    "observed": {},
                    "guardrails": {},
                    "errors": [],
                    "error_count": 0,
                }), encoding="utf-8")
                receipt_status = receipt_builder.main([
                    "--manifest", str(manifest_path),
                    "--structural-report", str(structural_report_path),
                    "--calibration-report", str(calibration_report_path),
                    "--repository-revision", "0" * 40,
                    "--output", str(receipt_path),
                ])

            self.assertEqual(build_status, 0)
            self.assertEqual(rebuild_status, 0)
            self.assertEqual(
                first_research_assignments,
                (research_domain_dir / "assignments.jsonl").read_bytes(),
            )
            self.assertEqual(
                first_training_assignments,
                (training_domain_dir / "assignments.jsonl").read_bytes(),
            )
            self.assertEqual(validation_status, 0)
            self.assertEqual(receipt_status, 0)
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            self.assertEqual(receipt["status"], "PASS")
            receipt_schema = json.loads(Path(
                "schemas/domain_validation_receipt.schema.json"
            ).read_text(encoding="utf-8"))
            self.assertEqual(
                list(structural.Draft202012Validator(receipt_schema).iter_errors(receipt)),
                [],
            )
            assignments = [
                json.loads(line)
                for line in (research_domain_dir / "assignments.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            r1 = next(item for item in assignments if item["subject_id"] == "r1")
            self.assertIsNone(r1["primary_domain_id"])
            self.assertEqual(r1["primary_domain_candidates"], ["D01", "D05"])


if __name__ == "__main__":
    unittest.main()
