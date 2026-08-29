#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate Domain Registry v0.01 sidecars against the existing dual databases.

Validation is exhaustive, not sampled. It checks schema conformance, subject closure,
domain-registry membership, D00 invariants, source hashes, inherited training evidence,
index closure, uniqueness, and manifest SHA-256 digests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict

try:
    from jsonschema import Draft202012Validator
except ImportError as exc:  # pragma: no cover
    raise SystemExit("jsonschema is required: pip install jsonschema") from exc


def load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def iter_jsonl(path: str):
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on {path} line {line_no}: {exc}") from exc


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def load_research_sources(path: str, errors: list[str]):
    hashes = {}
    for line_no, record in iter_jsonl(path):
        rid = record.get("record_id")
        if not rid:
            errors.append(f"research source line {line_no}: missing record_id")
            continue
        if rid in hashes:
            errors.append(f"research source duplicate record_id: {rid}")
            continue
        hashes[rid] = (record.get("provenance") or {}).get("source_hash")
    return hashes


def load_training_sources(path: str, errors: list[str]):
    hashes = {}
    source_refs = {}
    for line_no, example in iter_jsonl(path):
        tid = example.get("training_example_id")
        if not tid:
            errors.append(f"training source line {line_no}: missing training_example_id")
            continue
        if tid in hashes:
            errors.append(f"training source duplicate training_example_id: {tid}")
            continue
        hashes[tid] = (example.get("transform") or {}).get("content_hash")
        source_refs[tid] = [str(x) for x in example.get("source_record_ids") or []]
    return hashes, source_refs


def validate_assignment_file(path: str, subject_type: str, subject_hashes: dict,
                             registry_ids: set[str], validator: Draft202012Validator,
                             errors: list[str]):
    assignments = {}
    assignment_ids = set()
    expected_indexes: defaultdict[str, dict[str, bool]] = defaultdict(dict)
    primary_counts = Counter()
    membership_counts = Counter()
    cross_domain_count = 0

    for line_no, assignment in iter_jsonl(path):
        schema_errors = sorted(validator.iter_errors(assignment), key=lambda e: list(e.path))
        for err in schema_errors:
            where = ".".join(str(x) for x in err.path) or "<root>"
            errors.append(f"{path} line {line_no} schema {where}: {err.message}")

        aid = assignment.get("assignment_id")
        sid = assignment.get("subject_id")
        require(aid not in assignment_ids, f"duplicate assignment_id {aid}", errors)
        assignment_ids.add(aid)
        require(sid not in assignments, f"duplicate domain assignment for {sid}", errors)
        require(assignment.get("subject_type") == subject_type,
                f"{sid}: subject_type {assignment.get('subject_type')} != {subject_type}", errors)
        require(sid in subject_hashes, f"{sid}: assignment references missing source subject", errors)
        if sid in subject_hashes:
            require(assignment.get("source_hash") == subject_hashes[sid],
                    f"{sid}: source_hash does not match source database", errors)

        domains = assignment.get("domain_ids") or []
        secondary = assignment.get("secondary_domain_ids") or []
        primary = assignment.get("primary_domain_id")
        require(bool(domains), f"{sid}: domain_ids is empty", errors)
        require(all(d in registry_ids for d in domains), f"{sid}: unknown domain id in {domains}", errors)
        require(all(d in registry_ids for d in secondary), f"{sid}: unknown secondary domain", errors)
        require(primary in domains, f"{sid}: primary domain {primary} not in domain_ids", errors)
        require(primary not in secondary, f"{sid}: primary domain repeated as secondary", errors)
        require(set(secondary) == set(domains) - {primary},
                f"{sid}: secondary domains do not equal domain_ids minus primary", errors)
        require(assignment.get("review_status") == "candidate",
                f"{sid}: automatic output must remain candidate", errors)

        if primary == "D00":
            require(domains == ["D00"], f"{sid}: D00 must be the only domain when primary", errors)
            require(not secondary, f"{sid}: D00 assignment cannot have secondary domains", errors)
            require(assignment.get("cross_domain") is False,
                    f"{sid}: D00 assignment cannot be cross-domain", errors)
        else:
            require("D00" not in domains, f"{sid}: D00 cannot mix with substantive domains", errors)
            require(assignment.get("cross_domain") == (len(domains) > 1),
                    f"{sid}: cross_domain flag inconsistent with memberships", errors)

        score_ids = set((assignment.get("scores") or {}).keys())
        require(score_ids.issubset(set(domains)), f"{sid}: score for unselected domain", errors)
        evidence_domains = {e.get("domain_id") for e in assignment.get("evidence") or []}
        require(evidence_domains.issubset(set(domains)), f"{sid}: evidence for unselected domain", errors)

        assignments[sid] = assignment
        primary_counts[primary] += 1
        cross_domain_count += int(bool(assignment.get("cross_domain")))
        for domain_id in domains:
            membership_counts[domain_id] += 1
            if sid in expected_indexes[domain_id]:
                errors.append(f"{sid}: duplicated expected index membership for {domain_id}")
            expected_indexes[domain_id][sid] = domain_id == primary

    require(set(assignments) == set(subject_hashes),
            f"{path}: assignment subjects do not exactly close over source subjects", errors)

    stats = {
        "total_assignments": len(assignments),
        "primary_counts": dict(sorted(primary_counts.items())),
        "membership_counts": dict(sorted(membership_counts.items())),
        "cross_domain_count": cross_domain_count,
    }
    return assignments, expected_indexes, stats


def validate_indexes(index_dir: str, domains: list[dict], expected: dict,
                     errors: list[str]) -> dict[str, str]:
    hashes = {}
    for domain in domains:
        domain_id = domain["domain_id"]
        path = os.path.join(index_dir, f"{domain_id}_{domain['slug']}.jsonl")
        require(os.path.exists(path), f"missing domain index {path}", errors)
        if not os.path.exists(path):
            continue
        actual = {}
        for line_no, entry in iter_jsonl(path):
            sid = entry.get("subject_id")
            require(bool(sid), f"{path} line {line_no}: missing subject_id", errors)
            require(sid not in actual, f"{path}: duplicate subject_id {sid}", errors)
            actual[sid] = bool(entry.get("primary"))
            require(entry.get("review_status") == "candidate",
                    f"{path} {sid}: review_status must remain candidate", errors)
        expected_domain = expected.get(domain_id, {})
        require(actual == expected_domain,
                f"{path}: index contents do not exactly match assignments", errors)
        hashes[domain_id] = sha256_file(path)
    return hashes


def validate_training_inheritance(training_assignments: dict, training_source_refs: dict,
                                  research_assignments: dict, errors: list[str]) -> None:
    for tid, assignment in training_assignments.items():
        source_ids = set(training_source_refs.get(tid, []))
        if assignment.get("primary_domain_id") != "D00":
            require(bool(source_ids), f"{tid}: classified training example has no source records", errors)
        for evidence in assignment.get("evidence") or []:
            if evidence.get("kind") != "inherited_source":
                errors.append(f"{tid}: training assignment has non-inherited evidence")
                continue
            source_id = evidence.get("value")
            domain_id = evidence.get("domain_id")
            require(source_id in source_ids, f"{tid}: inherited evidence source not in candidate lineage", errors)
            source_assignment = research_assignments.get(source_id)
            require(source_assignment is not None, f"{tid}: inherited evidence source has no research assignment", errors)
            if source_assignment is not None:
                require(domain_id in (source_assignment.get("domain_ids") or []),
                        f"{tid}: inherited domain {domain_id} absent from source {source_id}", errors)


def validate_manifest(path: str, registry_path: str, rules_path: str,
                      research_path: str, training_path: str,
                      research_assignment_path: str, training_assignment_path: str,
                      research_index_hashes: dict, training_index_hashes: dict,
                      research_stats: dict, training_stats: dict, errors: list[str]) -> None:
    require(os.path.exists(path), f"missing manifest {path}", errors)
    if not os.path.exists(path):
        return
    manifest = load_json(path)
    require(manifest.get("governance", {}).get("canonical_content_mutated") is False,
            "manifest must declare canonical_content_mutated=false", errors)
    require(manifest.get("governance", {}).get("automatic_assignment_is_canonical") is False,
            "manifest must declare automatic_assignment_is_canonical=false", errors)

    expected_inputs = {
        "research_records": sha256_file(research_path),
        "training_candidates": sha256_file(training_path),
        "registry": sha256_file(registry_path),
        "rules": sha256_file(rules_path),
    }
    for key, digest in expected_inputs.items():
        require(manifest.get("inputs", {}).get(key, {}).get("sha256") == digest,
                f"manifest input digest mismatch: {key}", errors)

    require(manifest.get("research", {}).get("assignments_sha256") == sha256_file(research_assignment_path),
            "manifest research assignments digest mismatch", errors)
    require(manifest.get("training", {}).get("assignments_sha256") == sha256_file(training_assignment_path),
            "manifest training assignments digest mismatch", errors)
    require(manifest.get("research", {}).get("index_sha256") == research_index_hashes,
            "manifest research index digests mismatch", errors)
    require(manifest.get("training", {}).get("index_sha256") == training_index_hashes,
            "manifest training index digests mismatch", errors)

    for key, value in research_stats.items():
        require(manifest.get("research", {}).get(key) == value,
                f"manifest research statistic mismatch: {key}", errors)
    for key, value in training_stats.items():
        require(manifest.get("training", {}).get(key) == value,
                f"manifest training statistic mismatch: {key}", errors)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Validate Domain Registry v0.01 generated sidecars.")
    ap.add_argument("--research", default="data/research_db/records.jsonl")
    ap.add_argument("--training", default="data/training_db/candidates.jsonl")
    ap.add_argument("--research-assignments", default="data/research_db/domains/assignments.jsonl")
    ap.add_argument("--training-assignments", default="data/training_db/domains/assignments.jsonl")
    ap.add_argument("--research-index-dir", default="data/research_db/indexes/domains")
    ap.add_argument("--training-index-dir", default="data/training_db/indexes/domains")
    ap.add_argument("--registry", default="config/domain_registry.json")
    ap.add_argument("--rules", default="config/domain_rules.json")
    ap.add_argument("--schema", default="schemas/domain_assignment.schema.json")
    ap.add_argument("--manifest", default="data/domain_manifest.json")
    args = ap.parse_args(argv)

    errors: list[str] = []
    registry = load_json(args.registry)
    schema = load_json(args.schema)
    validator = Draft202012Validator(schema)
    domains = registry.get("domains") or []
    registry_ids = {d["domain_id"] for d in domains}
    require(len(registry_ids) == len(domains), "domain registry contains duplicate IDs", errors)
    require("D00" in registry_ids, "domain registry must contain D00", errors)

    research_hashes = load_research_sources(args.research, errors)
    training_hashes, training_source_refs = load_training_sources(args.training, errors)

    research_assignments, research_expected_indexes, research_stats = validate_assignment_file(
        args.research_assignments, "research_record", research_hashes, registry_ids, validator, errors
    )
    training_assignments, training_expected_indexes, training_stats = validate_assignment_file(
        args.training_assignments, "training_example", training_hashes, registry_ids, validator, errors
    )

    validate_training_inheritance(
        training_assignments, training_source_refs, research_assignments, errors
    )

    research_index_hashes = validate_indexes(
        args.research_index_dir, domains, research_expected_indexes, errors
    )
    training_index_hashes = validate_indexes(
        args.training_index_dir, domains, training_expected_indexes, errors
    )

    validate_manifest(
        args.manifest, args.registry, args.rules, args.research, args.training,
        args.research_assignments, args.training_assignments,
        research_index_hashes, training_index_hashes,
        research_stats, training_stats, errors,
    )

    report = {
        "status": "PASS" if not errors else "FAIL",
        "registry_version": registry.get("registry_version"),
        "research": research_stats,
        "training": training_stats,
        "errors": errors[:100],
        "error_count": len(errors),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
