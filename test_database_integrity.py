#!/usr/bin/env python3
"""Regression tests for governed database construction and validation."""
from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest
from unittest import mock

import build_dual_databases as builder
import validate_databases as validator


def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class DatabaseIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = self.temp.name
        self.transcripts = os.path.join(self.root, "transcripts.jsonl")
        self.research = os.path.join(self.root, "research")
        self.training = os.path.join(self.root, "training")
        write_jsonl(self.transcripts, self.synthetic_conversations())
        self.build()

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def synthetic_conversations():
        def conversation(cid, scope="work", do_not_remember=False):
            return {
                "conversation_id": cid,
                "title": "Synthetic %s" % cid,
                "tags": ["physics"],
                "scope": scope,
                "source": "chatgpt",
                "create_time": 1,
                "update_time": 2,
                "is_do_not_remember": do_not_remember,
                "messages": [
                    {
                        "message_id": "%s-u" % cid,
                        "role": "user",
                        "text": "Question %s" % cid,
                        "create_time": 1,
                    },
                    {
                        "message_id": "%s-a" % cid,
                        "role": "assistant",
                        "text": "Answer %s" % cid,
                        "reasoning": "private chain %s" % cid,
                        "create_time": 2,
                        "model_slug": "synthetic-model",
                    },
                ],
            }

        return [
            conversation("normal"),
            conversation("private", scope="personal"),
            conversation("dnr", do_not_remember=True),
        ]

    def build(self):
        with contextlib.redirect_stdout(io.StringIO()):
            rc = builder.main([
                "--transcripts", self.transcripts,
                "--research-dir", self.research,
                "--training-dir", self.training,
            ])
        self.assertEqual(rc, 0)

    def validate(self, hash_sample=100):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            rc = validator.main([
                "--research-dir", self.research,
                "--training-dir", self.training,
                "--hash-sample", str(hash_sample),
                "--seed", "773",
            ])
        return rc, output.getvalue()

    @staticmethod
    def rewrite_manifest_digest(directory, records_name, digest_key):
        records_path = os.path.join(directory, records_name)
        manifest_path = os.path.join(directory, "manifest.json")
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        manifest[digest_key] = builder.sha256_file(records_path)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

    def mutate_candidate(self, predicate, mutation):
        path = os.path.join(self.training, "candidates.jsonl")
        records = read_jsonl(path)
        for record in records:
            if predicate(record):
                mutation(record)
                break
        else:
            self.fail("candidate selected for mutation was not found")
        write_jsonl(path, records)
        self.rewrite_manifest_digest(
            self.training, "candidates.jsonl", "candidates_sha256")

    def mutate_research(self, predicate, mutation):
        path = os.path.join(self.research, "records.jsonl")
        records = read_jsonl(path)
        for record in records:
            if predicate(record):
                mutation(record)
                break
        else:
            self.fail("research record selected for mutation was not found")
        write_jsonl(path, records)
        self.rewrite_manifest_digest(
            self.research, "records.jsonl", "records_sha256")

    def test_valid_build_is_deterministic_and_fully_sampled(self):
        rc, output = self.validate(hash_sample=100)
        self.assertEqual(rc, 0, output)
        self.assertEqual(validator.STATS["research_hash_verified"], 9)
        self.assertEqual(validator.STATS["candidate_hash_verified"], 3)

        second_research = os.path.join(self.root, "research-second")
        second_training = os.path.join(self.root, "training-second")
        with contextlib.redirect_stdout(io.StringIO()):
            builder.main([
                "--transcripts", self.transcripts,
                "--research-dir", second_research,
                "--training-dir", second_training,
            ])
        for dirname, filename, second_dir in (
            (self.research, "records.jsonl", second_research),
            (self.training, "candidates.jsonl", second_training),
        ):
            with open(os.path.join(dirname, filename), "rb") as left:
                with open(os.path.join(second_dir, filename), "rb") as right:
                    self.assertEqual(left.read(), right.read())

        research_records = read_jsonl(
            os.path.join(self.research, "records.jsonl"))
        self.assertTrue(all(
            rec["training_policy"]["license_status"] == "unknown"
            for rec in research_records))
        candidates = read_jsonl(
            os.path.join(self.training, "candidates.jsonl"))
        self.assertTrue(all(
            "created_at" not in rec["transform"] for rec in candidates))

    def test_source_change_does_not_replace_existing_outputs(self):
        guarded_research = os.path.join(self.root, "guarded-research")
        guarded_training = os.path.join(self.root, "guarded-training")
        os.makedirs(guarded_research)
        os.makedirs(guarded_training)
        research_path = os.path.join(guarded_research, "records.jsonl")
        candidates_path = os.path.join(guarded_training, "candidates.jsonl")
        with open(research_path, "w", encoding="utf-8") as f:
            f.write("existing research\n")
        with open(candidates_path, "w", encoding="utf-8") as f:
            f.write("existing candidates\n")

        source_hash_calls = 0
        real_sha256_file = builder.sha256_file

        def drifting_hash(path):
            nonlocal source_hash_calls
            if path == self.transcripts:
                source_hash_calls += 1
                return ("a" if source_hash_calls == 1 else "b") * 64
            return real_sha256_file(path)

        with mock.patch.object(builder, "sha256_file", side_effect=drifting_hash):
            with self.assertRaisesRegex(RuntimeError, "changed during"):
                builder.main([
                    "--transcripts", self.transcripts,
                    "--research-dir", guarded_research,
                    "--training-dir", guarded_training,
                ])

        with open(research_path, encoding="utf-8") as f:
            self.assertEqual(f.read(), "existing research\n")
        with open(candidates_path, encoding="utf-8") as f:
            self.assertEqual(f.read(), "existing candidates\n")
        self.assertFalse(os.path.exists(research_path + ".tmp"))
        self.assertFalse(os.path.exists(candidates_path + ".tmp"))

    def test_sampler_selection_is_order_independent(self):
        def selected(order):
            sampler = validator.StableHashSampler(4, 91, "regression")
            for item_id in order:
                sampler.add(item_id, builder.sha256_text(item_id), item_id)
            return {entry[1] for entry in sampler._heap}

        ids = ["item-%02d" % i for i in range(20)]
        self.assertEqual(selected(ids), selected(reversed(ids)))
        self.assertEqual(len(selected(ids)), 4)

    def test_do_not_remember_candidate_cannot_be_admitted(self):
        self.mutate_candidate(
            lambda rec: rec["split_group"] == "conversation:dnr",
            lambda rec: rec["admission"].update(status="admitted"),
        )
        rc, output = self.validate()
        self.assertEqual(rc, 1)
        self.assertIn("relaxes source", output)

    def test_pending_unreviewed_candidate_cannot_be_admitted(self):
        self.mutate_candidate(
            lambda rec: rec["split_group"] == "conversation:normal",
            lambda rec: rec["admission"].update(status="admitted"),
        )
        rc, output = self.validate()
        self.assertEqual(rc, 1)
        self.assertIn("non-admitted source", output)
        self.assertIn("unlicensed source", output)

    def test_first_message_must_descend_from_conversation_header(self):
        self.mutate_research(
            lambda rec: rec["record_id"] == "research:message:normal:normal-u",
            lambda rec: rec["provenance"].update(
                parent_record_ids=["research:conversation:wrong"]),
        )
        rc, output = self.validate()
        self.assertEqual(rc, 1)
        self.assertIn("broken message chain", output)

    def test_message_policy_must_match_conversation_header(self):
        self.mutate_research(
            lambda rec: rec["record_id"] == "research:message:normal:normal-u",
            lambda rec: rec["training_policy"].update(
                license_status="owned"),
        )
        rc, output = self.validate()
        self.assertEqual(rc, 1)
        self.assertIn("policy diverges from conversation header", output)

    def test_candidate_sources_cannot_cross_conversation_boundaries(self):
        def cross_boundary(record):
            record["source_record_ids"][1] = \
                "research:message:private:private-a"
            fingerprint = json.dumps({
                "cid": "normal",
                "assistant_mid": "private-a",
                "task": record["task"],
                "prompt": record["input"]["prompt"],
                "target": record["target"]["text"],
            }, ensure_ascii=False, sort_keys=True)
            record["transform"]["content_hash"] = \
                builder.sha256_text(fingerprint)

        self.mutate_candidate(
            lambda rec: rec["split_group"] == "conversation:normal",
            cross_boundary,
        )
        rc, output = self.validate()
        self.assertEqual(rc, 1)
        self.assertIn("crosses conversation boundary", output)

    def test_hash_tamper_is_detected_and_state_resets(self):
        self.mutate_research(
            lambda rec: rec["record_id"] == "research:conversation:normal",
            lambda rec: rec["provenance"].update(source_hash="0" * 64),
        )
        rc, output = self.validate(hash_sample=100)
        self.assertEqual(rc, 1)
        self.assertIn("research source hash mismatch", output)

        self.build()
        rc, output = self.validate(hash_sample=100)
        self.assertEqual(rc, 0, output)
        self.assertEqual(validator.FAILURES, [])


if __name__ == "__main__":
    unittest.main()
