#!/usr/bin/env python3
"""Regression tests for governed heterogeneous project-archive import."""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import pathlib
import tempfile
import unittest

import import_project_archive as importer
import validate_databases as validator


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class ProjectArchiveImportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temp.name)
        self.archive = self.root / "archive"
        self.research = self.root / "research"
        self.training = self.root / "training"
        self.archive.mkdir()
        self._create_archive()

    def tearDown(self):
        self.temp.cleanup()

    def _create_archive(self):
        text = "A governed scientific source.\n"
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        source_id = "SRC-%s" % digest[:16]
        source_a = self.archive / "sources/a.txt"
        source_b = self.archive / "sources/copy.txt"
        source_a.parent.mkdir(parents=True)
        source_a.write_text(text, encoding="utf-8")
        source_b.write_text(text, encoding="utf-8")

        records = [
            {
                "record_type": "source",
                "source_id": source_id,
                "path": "sources/a.txt",
                "filename": "a.txt",
                "extension": ".txt",
                "bytes": len(text.encode("utf-8")),
                "sha256": digest,
                "text_extract_available": True,
                "provenance_class": "normalized_extraction",
            },
            {
                "record_type": "source",
                "source_id": source_id,
                "path": "sources/copy.txt",
                "filename": "copy.txt",
                "extension": ".txt",
                "bytes": len(text.encode("utf-8")),
                "sha256": digest,
                "text_extract_available": True,
                "provenance_class": "loose_file",
            },
            {
                "record_type": "theorem",
                "theorem_id": "THM-1",
                "heading": "Historical source theorem",
                "body": "Preserved as a claim, not promoted to proof.",
                "source_id": source_id,
                "source_path": "sources/a.txt",
                "status": "source_claim_not_independently_validated",
            },
            {
                "record_type": "thread",
                "thread_id": "THR-1",
                "title": "Reconstructed thread",
                "summary": "A retained summary.",
                "record_status": "reconstructed_summary",
                "verbatim_transcript_available_in_this_archive": False,
            },
            {
                "record_type": "crosslink",
                "source": "A",
                "relation": "is related to",
                "target": "B",
            },
        ]
        project_manifest = {
            "generated_at": "2026-08-30T13:35:00-05:00",
            "project": "Synthetic project",
            "counts": {
                "source_file_entries": 2,
                "unique_source_hashes": 1,
                "unique_text_records": 1,
            },
        }
        text_corpus = [{
            "source_id": source_id,
            "path": "sources/a.txt",
            "sha256": digest,
            "extension": ".txt",
            "content": text,
            "content_status": "verbatim_file_text",
        }]
        write_json(
            self.archive / "knowledge_json/00_project_manifest.json",
            project_manifest)
        write_jsonl(
            self.archive / "knowledge_json/99_all_records.jsonl", records)
        write_json(
            self.archive / "knowledge_json/18_text_corpus_unique.json",
            text_corpus)
        self._write_checksums()

    def _write_checksums(self):
        lines = []
        for path in sorted(self.archive.rglob("*")):
            if not path.is_file() or path.name == "SHA256SUMS.txt":
                continue
            digest = importer.sha256_file(path)
            lines.append("%s  %s\n" % (
                digest, path.relative_to(self.archive).as_posix()))
        (self.archive / "SHA256SUMS.txt").write_text(
            "".join(lines), encoding="utf-8")

    def run_import(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            rc = importer.main([
                "--archive-root", str(self.archive),
                "--research-dir", str(self.research),
                "--training-dir", str(self.training),
            ])
        return rc, output.getvalue()

    def run_validation(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            rc = validator.main([
                "--research-dir", str(self.research),
                "--training-dir", str(self.training),
                "--hash-sample", "100",
                "--seed", "811",
            ])
        return rc, output.getvalue()

    def rewrite_research_manifest_digest(self):
        manifest_path = self.research / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["records_sha256"] = importer.sha256_file(
            self.research / "records.jsonl")
        write_json(manifest_path, manifest)

    def test_valid_import_deduplicates_sources_and_admits_nothing(self):
        rc, output = self.run_import()
        self.assertEqual(rc, 0, output)
        records = read_jsonl(self.research / "records.jsonl")
        self.assertEqual(len(records), 5)
        artifacts = [r for r in records if r["record_type"] == "artifact"]
        self.assertEqual(len(artifacts), 1)
        self.assertEqual(len(artifacts[0]["source"]["alias_paths"]), 2)
        self.assertTrue(all(
            r["epistemic_status"] == "unknown" for r in records))
        self.assertTrue(all(
            r["training_policy"]["default_admission"] == "pending_review"
            for r in records))
        self.assertEqual(
            (self.training / "candidates.jsonl").read_text(encoding="utf-8"),
            "")

        rc, output = self.run_validation()
        self.assertEqual(rc, 0, output)
        self.assertEqual(validator.STATS["research_generic"], 5)
        self.assertEqual(validator.STATS["candidates"], 0)

    def test_archive_tamper_is_rejected_before_import(self):
        (self.archive / "sources/a.txt").write_text(
            "tampered\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "checksum verification failed"):
            self.run_import()
        self.assertFalse((self.research / "records.jsonl").exists())

    def test_generic_parent_closure_is_enforced(self):
        self.run_import()
        records_path = self.research / "records.jsonl"
        records = read_jsonl(records_path)
        records[-1]["provenance"]["parent_record_ids"] = [
            "research:archive:missing"]
        write_jsonl(records_path, records)
        self.rewrite_research_manifest_digest()

        rc, output = self.run_validation()
        self.assertEqual(rc, 1)
        self.assertIn("unresolved generic parent_record_ids", output)


if __name__ == "__main__":
    unittest.main()
