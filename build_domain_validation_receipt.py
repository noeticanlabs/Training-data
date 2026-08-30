#!/usr/bin/env python3
"""Build a non-sensitive, commit-ready receipt for a full domain validation run."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time


RECEIPT_SCHEMA_VERSION = "0.01"


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: str, value: dict) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def is_sha256(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        char in "0123456789abcdef" for char in value
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build a domain-validation evidence receipt.")
    ap.add_argument("--manifest", default="data/domain_manifest.json")
    ap.add_argument("--structural-report", default="data/domain_structural_report.json")
    ap.add_argument("--calibration-report", default="data/domain_calibration_report.json")
    ap.add_argument("--repository-revision", required=True,
                    help="Exact Git commit SHA used for the run.")
    ap.add_argument("--output", required=True,
                    help="Commit-ready JSON receipt path. The private corpus is never copied.")
    args = ap.parse_args(argv)

    manifest = load_json(args.manifest)
    structural = load_json(args.structural_report)
    calibration = load_json(args.calibration_report)
    errors: list[str] = []
    classifier = manifest.get("classifier") or {}

    require(len(args.repository_revision) == 40 and all(
        char in "0123456789abcdef" for char in args.repository_revision
    ), "repository revision must be a complete lowercase Git commit SHA", errors)
    require(structural.get("status") == "PASS", "structural validation did not pass", errors)
    require(calibration.get("status") == "PASS", "calibration validation did not pass", errors)
    require(structural.get("error_count") == 0,
            "structural report contains validation errors", errors)
    require(calibration.get("error_count") == 0,
            "calibration report contains validation errors", errors)
    require(manifest.get("classifier_version") == structural.get("classifier_version"),
            "manifest and structural report classifier identities differ", errors)
    require(manifest.get("rules_version") == structural.get("rules_version"),
            "manifest and structural report rules versions differ", errors)
    require(manifest.get("rules_version") == calibration.get("rules_version"),
            "manifest and calibration report rules versions differ", errors)
    require(classifier.get("identity") == manifest.get("classifier_version"),
            "manifest classifier receipt identity mismatch", errors)
    require(classifier.get("rules_version") == manifest.get("rules_version"),
            "manifest classifier receipt rules version mismatch", errors)
    require(classifier.get("rules_sha256") == manifest.get("inputs", {}).get(
        "rules", {}
    ).get("sha256"), "manifest classifier receipt rules digest mismatch", errors)
    require(manifest.get("governance", {}).get("canonical_content_mutated") is False,
            "manifest does not preserve canonical content", errors)
    require(manifest.get("governance", {}).get("automatic_assignment_is_canonical") is False,
            "manifest incorrectly promotes automatic assignments", errors)
    for name, entry in (manifest.get("inputs") or {}).items():
        require(is_sha256((entry or {}).get("sha256")),
                f"manifest input {name} lacks a valid SHA-256 digest", errors)
    for layer in ("research", "training"):
        require(is_sha256(manifest.get(layer, {}).get("assignments_sha256")),
                f"manifest {layer} assignments lack a valid SHA-256 digest", errors)
        for domain_id, digest in (manifest.get(layer, {}).get("index_sha256") or {}).items():
            require(is_sha256(digest),
                    f"manifest {layer} index {domain_id} lacks a valid SHA-256 digest", errors)
    for layer in ("research", "training"):
        require(
            manifest.get(layer, {}).get("total_assignments")
            == structural.get(layer, {}).get("total_assignments"),
            f"{layer} assignment count differs between manifest and structural report",
            errors,
        )

    receipt = {
        "receipt_schema_version": RECEIPT_SCHEMA_VERSION,
        "status": "PASS" if not errors else "FAIL",
        "generated_at": now_iso(),
        "evidence_scope": "non_sensitive_hash_and_metric_receipt",
        "canonical_content_included": False,
        "repository_revision": args.repository_revision,
        "runtime": {
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
        "classifier": classifier,
        "inputs": {
            name: {"sha256": (entry or {}).get("sha256")}
            for name, entry in (manifest.get("inputs") or {}).items()
        },
        "outputs": {
            "research": {
                "assignments_sha256": manifest.get("research", {}).get("assignments_sha256"),
                "index_sha256": manifest.get("research", {}).get("index_sha256"),
            },
            "training": {
                "assignments_sha256": manifest.get("training", {}).get("assignments_sha256"),
                "index_sha256": manifest.get("training", {}).get("index_sha256"),
            },
        },
        "statistics": {
            "research": {
                key: manifest.get("research", {}).get(key)
                for key in (
                    "total_assignments", "primary_counts", "primary_candidate_counts",
                    "membership_counts", "cross_domain_count", "ambiguous_primary_count",
                )
            },
            "training": {
                key: manifest.get("training", {}).get(key)
                for key in (
                    "total_assignments", "primary_counts", "primary_candidate_counts",
                    "membership_counts", "cross_domain_count", "ambiguous_primary_count",
                )
            },
        },
        "validation": {
            "structural": {
                "report_sha256": sha256_file(args.structural_report),
                "status": structural.get("status"),
                "error_count": structural.get("error_count"),
            },
            "calibration": {
                "report_sha256": sha256_file(args.calibration_report),
                "status": calibration.get("status"),
                "error_count": calibration.get("error_count"),
                "observed": calibration.get("observed"),
                "guardrails": calibration.get("guardrails"),
                "interpretation": calibration.get("interpretation"),
            },
        },
        "errors": errors,
    }
    write_json(args.output, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
