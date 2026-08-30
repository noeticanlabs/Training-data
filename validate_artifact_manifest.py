#!/usr/bin/env python3
"""Validate opaque source artifacts, identity metadata, and fail-closed policy."""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os

from jsonschema import Draft202012Validator

WD = os.path.dirname(os.path.abspath(__file__))
FAILURES = []


def check(condition, message):
    if not condition:
        FAILURES.append(message)
        if len(FAILURES) <= 50:
            print("  FAIL:", message)
    return condition


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob_sha1(path):
    size = os.path.getsize(path)
    digest = hashlib.sha1()
    digest.update(("blob %d\0" % size).encode("ascii"))
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def media_signature_matches(path, media_type):
    with open(path, "rb") as f:
        prefix = f.read(8)
        if media_type == "image/png":
            return prefix == b"\x89PNG\r\n\x1a\n"
        if media_type == "image/jpeg":
            return prefix.startswith(b"\xff\xd8\xff")
        if media_type.startswith("text/"):
            try:
                f.seek(0)
                f.read().decode("utf-8")
                return True
            except UnicodeDecodeError:
                return False
    return True


def main(argv=None):
    FAILURES.clear()
    ap = argparse.ArgumentParser(
        description="Validate source artifact registry and local file closure.")
    ap.add_argument("--repo-root", default=WD)
    ap.add_argument("--manifest", default="artifacts/manifest.json")
    args = ap.parse_args(argv)

    manifest_path = (args.manifest if os.path.isabs(args.manifest)
                     else os.path.join(args.repo_root, args.manifest))
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    with open(os.path.join(WD, "schemas", "source_artifact.schema.json"),
              encoding="utf-8") as f:
        schema = json.load(f)

    errors = list(Draft202012Validator(schema).iter_errors(manifest))
    for error in errors:
        check(False, "schema: %s" % error.message)

    artifacts = manifest.get("artifacts") or []
    ids = set()
    paths = set()
    blobs = set()
    unresolved_sha256 = 0

    for artifact in artifacts:
        artifact_id = artifact.get("artifact_id")
        relpath = artifact.get("path")
        blob = artifact.get("git_blob_sha1")
        check(artifact_id not in ids, "duplicate artifact_id: %s" % artifact_id)
        check(relpath not in paths, "duplicate artifact path: %s" % relpath)
        check(blob not in blobs, "duplicate Git blob identity: %s" % blob)
        ids.add(artifact_id)
        paths.add(relpath)
        blobs.add(blob)
        check(artifact_id == "artifact:git-blob:%s" % blob,
              "artifact ID does not match Git blob: %s" % artifact_id)

        path = os.path.join(args.repo_root, relpath)
        if not check(os.path.isfile(path), "registered artifact is missing: %s" % relpath):
            continue
        check(os.path.getsize(path) == artifact.get("byte_size"),
              "byte size mismatch: %s" % relpath)
        check(git_blob_sha1(path) == blob,
              "Git blob SHA-1 mismatch: %s" % relpath)
        content_sha256 = artifact.get("content_sha256")
        if content_sha256 is None:
            unresolved_sha256 += 1
        else:
            check(sha256_file(path) == content_sha256,
                  "SHA-256 mismatch: %s" % relpath)
        check(media_signature_matches(path, artifact.get("media_type", "")),
              "media signature mismatch: %s" % relpath)

        if artifact.get("training_allowed"):
            check(artifact.get("review_status") == "approved",
                  "training-enabled artifact lacks approval: %s" % relpath)
            check(artifact.get("license_status") in {"owned", "permitted"},
                  "training-enabled artifact lacks license clearance: %s"
                  % relpath)
            check(artifact.get("sensitivity") in {"public", "internal"},
                  "training-enabled artifact has unsafe sensitivity: %s"
                  % relpath)
            check(content_sha256 is not None,
                  "training-enabled artifact lacks SHA-256: %s" % relpath)

    actual_paths = {
        os.path.relpath(path, args.repo_root)
        for path in glob.glob(os.path.join(args.repo_root, "*.dat"))
    }
    check(paths == actual_paths,
          "artifact registry closure mismatch: registered=%s actual=%s"
          % (sorted(paths), sorted(actual_paths)))

    print("Artifacts: %d | training allowed: %d | SHA-256 pending: %d"
          % (len(artifacts),
             sum(bool(a.get("training_allowed")) for a in artifacts),
             unresolved_sha256))
    if FAILURES:
        print("RESULT: ARTIFACT VALIDATION FAILED (%d problems)" % len(FAILURES))
        return 1
    print("RESULT: ARTIFACT REGISTRY VALID")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
