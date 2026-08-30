#!/usr/bin/env python3
"""Regression tests for explicit legacy-export governance."""
from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest

import build_training_formats as builder
import validate_training_formats as validator


class TrainingFormatGovernanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = self.temp.name
        self.transcript_dir = os.path.join(self.root, "transcripts")
        self.output_dir = os.path.join(self.root, "training")
        os.makedirs(self.transcript_dir)
        record = {
            "conversation_id": "conv-1",
            "title": "Synthetic export",
            "tags": ["physics"],
            "scope": "work",
            "messages": [
                {"message_id": "u1", "role": "user", "text": "Question"},
                {
                    "message_id": "a1",
                    "role": "assistant",
                    "text": "Answer",
                    "reasoning": "Hidden derivation",
                },
            ],
        }
        with open(os.path.join(self.transcript_dir, "transcripts.jsonl"),
                  "w", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def tearDown(self):
        self.temp.cleanup()

    def build(self, *extra):
        stdout = io.StringIO()
        stderr = io.StringIO()
        args = [
            "--transcripts", self.transcript_dir,
            "--out-dir", self.output_dir,
            *extra,
        ]
        with contextlib.redirect_stdout(stdout), \
                contextlib.redirect_stderr(stderr):
            rc = builder.main(args)
        return rc, stdout.getvalue(), stderr.getvalue()

    @staticmethod
    def jsonl_records(path):
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def test_direct_export_is_fail_closed_by_default(self):
        rc, _, stderr = self.build()
        self.assertEqual(rc, 2)
        self.assertIn("bypasses governed admission", stderr)
        self.assertFalse(os.path.exists(self.output_dir))

    def test_acknowledged_export_declares_bypass_and_preserves_reasoning_flag(self):
        rc, stdout, stderr = self.build(
            "--unsafe-legacy-transcript-export", "--include-reasoning")
        self.assertEqual(rc, 0, stdout + stderr)

        with open(os.path.join(self.output_dir, "split_manifest.json"),
                  encoding="utf-8") as f:
            manifest = json.load(f)
        self.assertEqual(manifest["governance"], {
            "source_mode": "legacy_transcript",
            "unsafe_acknowledged": True,
            "admission_filter": "none",
        })

        for fmt, accessor in (
            ("openai", lambda rec: rec["messages"][-1]["content"]),
            ("sharegpt", lambda rec: rec["conversations"][-1]["value"]),
            ("alpaca", lambda rec: rec["output"]),
        ):
            records = self.jsonl_records(
                os.path.join(self.output_dir, fmt, "val.jsonl"))
            self.assertEqual(len(records), 1)
            self.assertIn("Hidden derivation", accessor(records[0]))

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            validation_rc = validator.main([
                "--training-dir", self.output_dir])
        self.assertEqual(validation_rc, 0, output.getvalue())
        self.assertIn("UNSAFE LEGACY", output.getvalue())

    def test_reasoning_is_absent_without_include_flag(self):
        rc, stdout, stderr = self.build(
            "--unsafe-legacy-transcript-export")
        self.assertEqual(rc, 0, stdout + stderr)
        records = self.jsonl_records(
            os.path.join(self.output_dir, "openai", "val.jsonl"))
        self.assertNotIn("Hidden derivation", records[0]["messages"][-1]["content"])


if __name__ == "__main__":
    unittest.main()
