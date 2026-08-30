#!/usr/bin/env python3
"""Mutation tests for opaque artifact identity and reuse governance."""
from __future__ import annotations

import contextlib
import copy
import io
import json
import os
import tempfile
import unittest

import validate_artifact_manifest as validator


class ArtifactManifestIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = self.temp.name
        self.png_path = os.path.join(self.root, "file-image.dat")
        self.code_path = os.path.join(self.root, "file-code.dat")
        with open(self.png_path, "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\nsynthetic")
        with open(self.code_path, "wb") as f:
            f.write(b"print('synthetic')\n")
        self.manifest_path = os.path.join(self.root, "manifest.json")
        self.manifest = {
            "schema_version": "0.01",
            "source_commit": "0" * 40,
            "artifacts": [
                self.artifact("file-image.dat", "image/png", "image",
                              training_allowed=False),
                self.artifact("file-code.dat", "text/x-python",
                              "executable_source", training_allowed=True),
            ],
        }
        self.write_manifest(self.manifest)

    def tearDown(self):
        self.temp.cleanup()

    def artifact(self, relpath, media_type, content_class,
                 training_allowed):
        path = os.path.join(self.root, relpath)
        blob = validator.git_blob_sha1(path)
        approved = bool(training_allowed)
        return {
            "artifact_id": "artifact:git-blob:%s" % blob,
            "path": relpath,
            "original_filename": None,
            "byte_size": os.path.getsize(path),
            "git_blob_sha1": blob,
            "content_sha256": validator.sha256_file(path),
            "media_type": media_type,
            "content_class": content_class,
            "sensitivity": "internal" if approved else "unknown",
            "license_status": "permitted" if approved else "unknown",
            "training_allowed": training_allowed,
            "review_status": "approved" if approved else "unreviewed",
            "parser": None,
            "notes": "synthetic fixture",
        }

    def write_manifest(self, manifest):
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f)

    def validate(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            rc = validator.main([
                "--repo-root", self.root,
                "--manifest", self.manifest_path,
            ])
        return rc, output.getvalue()

    def test_valid_registry_closes_over_files_and_hashes(self):
        rc, output = self.validate()
        self.assertEqual(rc, 0, output)
        self.assertIn("ARTIFACT REGISTRY VALID", output)

    def test_training_permission_requires_license_review_and_hash(self):
        broken = copy.deepcopy(self.manifest)
        code = broken["artifacts"][1]
        code["license_status"] = "unknown"
        code["review_status"] = "unreviewed"
        code["content_sha256"] = None
        self.write_manifest(broken)
        rc, output = self.validate()
        self.assertEqual(rc, 1)
        self.assertIn("lacks license clearance", output)
        self.assertIn("lacks approval", output)
        self.assertIn("lacks SHA-256", output)

    def test_unregistered_file_breaks_closure(self):
        with open(os.path.join(self.root, "file-unregistered.dat"), "wb") as f:
            f.write(b"opaque")
        rc, output = self.validate()
        self.assertEqual(rc, 1)
        self.assertIn("registry closure mismatch", output)


if __name__ == "__main__":
    unittest.main()
