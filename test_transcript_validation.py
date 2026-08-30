#!/usr/bin/env python3
"""Regression tests for transcript and index reconciliation."""
from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest

import validate_outputs as validator


class TranscriptValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = self.temp.name
        self.paths = {
            "TRANSCRIPTS": os.path.join(root, "transcripts.jsonl"),
            "WORK_ONLY": os.path.join(root, "work.jsonl"),
            "STATS": os.path.join(root, "stats.json"),
            "PII": os.path.join(root, "pii.json"),
            "INDEX": os.path.join(root, "index.json"),
            "PROJECTS": os.path.join(root, "projects.json"),
        }
        self.originals = {name: getattr(validator, name)
                          for name in self.paths}
        for name, path in self.paths.items():
            setattr(validator, name, path)
        self.record = {
            "conversation_id": "conv-1",
            "title": "Synthetic",
            "tags": ["physics"],
            "scope": "work",
            "create_time": 1,
            "source": "chatgpt",
            "messages": [
                {"message_id": "u1", "role": "user", "text": "Question"},
                {"message_id": "a1", "role": "assistant", "text": "Answer"},
            ],
        }
        self.write_fixture(self.record)

    def tearDown(self):
        for name, value in self.originals.items():
            setattr(validator, name, value)
        self.temp.cleanup()

    @staticmethod
    def write_json(path, value):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(value, f)

    @staticmethod
    def write_jsonl(path, records):
        with open(path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")

    def write_fixture(self, record):
        self.write_jsonl(self.paths["TRANSCRIPTS"], [record])
        self.write_jsonl(self.paths["WORK_ONLY"], [record])
        self.write_json(self.paths["STATS"], {
            "kept": {
                "conversations": 1,
                "messages": {"total": len(record["messages"])},
            },
            "drops": {"conversations_by_reason": {}},
            "models": {},
        })
        self.write_json(self.paths["PII"], {
            "totals_by_category": {},
            "conversations_with_hits": [],
        })
        self.write_json(self.paths["INDEX"], {
            "count": 1,
            "generated_at": "synthetic",
            "conversations": [{"conversation_id": "conv-1"}],
        })
        self.write_json(self.paths["PROJECTS"], {
            "total_conversations": 1,
            "scope": "all",
            "tags": {"physics": {"conversations": ["conv-1"]}},
        })

    def validate(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            rc = validator.main()
        return rc, output.getvalue()

    def test_valid_outputs_reconcile(self):
        rc, output = self.validate()
        self.assertEqual(rc, 0, output)
        self.assertIn("ALL CHECKS PASSED", output)

    def test_consecutive_roles_are_fatal(self):
        broken = dict(self.record)
        broken["messages"] = [
            {"message_id": "a1", "role": "assistant", "text": "First"},
            {"message_id": "a2", "role": "assistant", "text": "Second"},
        ]
        self.write_fixture(broken)
        rc, output = self.validate()
        self.assertEqual(rc, 1)
        self.assertIn("role violations: 1", output)
        self.assertIn("VALIDATION FAILED", output)

    def test_duplicate_conversation_ids_are_fatal(self):
        self.write_jsonl(self.paths["TRANSCRIPTS"], [self.record, self.record])
        stats = {
            "kept": {"conversations": 2, "messages": {"total": 4}},
            "drops": {"conversations_by_reason": {}},
            "models": {},
        }
        self.write_json(self.paths["STATS"], stats)
        rc, output = self.validate()
        self.assertEqual(rc, 1)
        self.assertIn("duplicate conversations: 1", output)


if __name__ == "__main__":
    unittest.main()
