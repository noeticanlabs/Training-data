#!/usr/bin/env python3
"""Regression tests for semantic graph identity and lifecycle invariants."""
from __future__ import annotations

import contextlib
import copy
import io
import json
import os
import tempfile
import unittest

from jsonschema import Draft202012Validator

import validate_research_graph as graph


SRC_A = "research:message:synthetic:a"
SRC_B = "research:message:synthetic:b"


def claim(statement, source=SRC_A):
    record = {
        "schema_version": "0.04",
        "node_type": "claim",
        "node_id": None,
        "source_record_ids": [source],
        "lifecycle": "candidate",
        "review": {"status": "unreviewed"},
        "statement": statement,
        "statement_basis": "verbatim",
        "epistemic_status": "unknown",
    }
    record["node_id"] = graph.expected_node_id(record)
    return record


def relation(relation_type, source, target):
    record = {
        "schema_version": "0.04",
        "relation_id": None,
        "relation_type": relation_type,
        "source_node_id": source,
        "target_node_id": target,
        "source_record_ids": [SRC_A],
        "basis": "explicit",
        "confidence": None,
        "review": {"status": "pending"},
    }
    record["relation_id"] = graph.expected_relation_id(record)
    return record


class ResearchGraphIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.graph_dir = self.temp.name

    def tearDown(self):
        self.temp.cleanup()

    def write_graph(self, nodes, relations):
        for filename, records in (
            ("candidate_nodes.jsonl", nodes),
            ("candidate_relations.jsonl", relations),
        ):
            with open(os.path.join(self.graph_dir, filename), "w",
                      encoding="utf-8") as f:
                for record in records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def validate(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            rc = graph.main([
                "--graph-dir", self.graph_dir,
                "--research-db", "_nonexistent_",
            ])
        return rc, output.getvalue()

    def test_semantic_identity_is_independent_of_provenance(self):
        first = claim("The invariant is bounded.", SRC_A)
        second = claim("The invariant is bounded.", SRC_B)
        self.assertEqual(first["node_id"], second["node_id"])
        self.assertNotEqual(first["source_record_ids"],
                            second["source_record_ids"])

    def test_schema_rejects_unregistered_basis_and_nonstring_claim_id(self):
        node_schema = graph.load_json(os.path.join(
            graph.WD, "schemas", "research_graph_node.schema.json"))
        schema_validator = Draft202012Validator(node_schema)

        invalid_basis = claim("A statement")
        invalid_basis["statement_basis"] = "speculative"
        self.assertTrue(list(schema_validator.iter_errors(invalid_basis)))

        experiment = {
            "schema_version": "0.04",
            "node_type": "experiment",
            "node_id": "graph:experiment:" + "0" * 24,
            "source_record_ids": [SRC_A],
            "lifecycle": "candidate",
            "review": {"status": "unreviewed"},
            "hypothesis": {"claim_node_id": 7},
            "method": "controlled simulation",
            "result": "bounded",
            "interpretation": "consistent",
            "supports": [],
        }
        self.assertTrue(list(schema_validator.iter_errors(experiment)))

    def test_supersession_requires_bidirectional_state(self):
        old = claim("The bound holds without qualification.")
        replacement = claim("The bound holds for positive sources.", SRC_B)
        old["lifecycle"] = "superseded"
        old["superseded_by"] = replacement["node_id"]
        old["review"] = {
            "status": "approved",
            "reviewer": "synthetic-reviewer",
            "evidence_record_ids": [SRC_A],
        }
        replacement["lifecycle"] = "reviewed"
        replacement["review"] = {
            "status": "approved",
            "reviewer": "synthetic-reviewer",
            "evidence_record_ids": [SRC_B],
        }
        edge = relation("supersedes", replacement["node_id"], old["node_id"])
        edge["review"] = {
            "status": "approved",
            "reviewer": "synthetic-reviewer",
            "evidence_record_ids": [SRC_A, SRC_B],
        }
        self.write_graph([old, replacement], [edge])
        rc, output = self.validate()
        self.assertEqual(rc, 0, output)

        broken = copy.deepcopy(old)
        broken["superseded_by"] = None
        self.write_graph([broken, replacement], [edge])
        rc, output = self.validate()
        self.assertEqual(rc, 1)
        self.assertIn("does not point back", output)

    def test_status_and_normalization_require_review_evidence(self):
        record = claim("Normalized claim")
        record["statement_basis"] = "normalized"
        record["epistemic_status"] = "machine_checked"
        record["node_id"] = graph.expected_node_id(record)
        self.write_graph([record], [])
        rc, output = self.validate()
        self.assertEqual(rc, 1)
        self.assertIn("lacks reviewed evidence", output)
        self.assertIn("lacks normalization notes", output)


if __name__ == "__main__":
    unittest.main()
