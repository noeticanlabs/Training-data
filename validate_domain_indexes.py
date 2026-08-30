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


DOMAIN_SCHEMA_VERSION = "0.02"


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


def classifier_identity(rules: dict, rules_digest: str) -> str:
    rules_version = str(rules.get("rules_version") or "").strip()
    if not rules_version:
        raise ValueError("domain rules must declare a non-empty rules_version")
    return f"domain_rules_v{rules_version}_sha256_{rules_digest[:16]}"


def stable_assignment_id(subject_type: str, subject_id: str, registry_version: str,
                         classifier_version: str) -> str:
    payload = f"{subject_type}|{subject_id}|{registry_version}|{classifier_version}"
    return "domain:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def write_json(path: str, value: dict) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def load_research_sources(path: str, errors: list[str]):
    hashes = {}
    for line_no, record in iter_jsonl(path):
        raw_rid = record.get("record_id")
        if not raw_rid:
            errors.append(f"research source line {line_no}: missing record_id")
            continue
        rid = str(raw_rid)
        if rid in hashes:
            errors.append(f"research source duplicate record_id: {rid}")
            continue
        hashes[rid] = (record.get("provenance") or {}).get("source_hash")
    return hashes


def load_training_sources(path: str, errors: list[str]):
    hashes = {}
    source_refs = {}
    for line_no, example in iter_jsonl(path):
        raw_tid = example.get("training_example_id")
        if not raw_tid:
            errors.append(f"training source line {line_no}: missing training_example_id")
            continue
        tid = str(raw_tid)
        if tid in hashes:
            errors.append(f"training source duplicate training_example_id: {tid}")
            continue
        hashes[tid] = (example.get("transform") or {}).get("content_hash")
        source_refs[tid] = [str(x) for x in example.get("source_record_ids") or []]
    return hashes, source_refs


def validate_assignment_file(path: str, subject_type: str, subject_hashes: dict,
                             registry_ids: set[str], validator: Draft202012Validator,
                             registry_version: str, classifier_version: str,
                             assignment_policy: dict,
                             errors: list[str]):
    assignments = {}
    assignment_ids = set()
    expected_indexes: defaultdict[str, dict[str, dict[str, bool]]] = defaultdict(dict)
    primary_counts = Counter()
    primary_candidate_counts = Counter()
    membership_counts = Counter()
    cross_domain_count = 0
    ambiguous_primary_count = 0

    for line_no, assignment in iter_jsonl(path):
        schema_errors = sorted(validator.iter_errors(assignment), key=lambda e: list(e.path))
        for err in schema_errors:
            where = ".".join(str(x) for x in err.path) or "<root>"
            errors.append(f"{path} line {line_no} schema {where}: {err.message}")

        if not isinstance(assignment, dict):
            errors.append(f"{path} line {line_no}: assignment must be an object")
            continue

        aid = assignment.get("assignment_id")
        sid = assignment.get("subject_id")
        if not isinstance(aid, str) or not aid:
            errors.append(f"{path} line {line_no}: assignment_id must be a non-empty string")
            continue
        if not isinstance(sid, str) or not sid:
            errors.append(f"{path} line {line_no}: subject_id must be a non-empty string")
            continue
        require(aid not in assignment_ids, f"duplicate assignment_id {aid}", errors)
        assignment_ids.add(aid)
        require(sid not in assignments, f"duplicate domain assignment for {sid}", errors)
        require(assignment.get("subject_type") == subject_type,
                f"{sid}: subject_type {assignment.get('subject_type')} != {subject_type}", errors)
        expected_method = "rule_based" if subject_type == "research_record" else "inherited"
        require(assignment.get("assignment_method") == expected_method,
                f"{sid}: assignment_method must be {expected_method}", errors)
        require(assignment.get("classifier_version") == classifier_version,
                f"{sid}: classifier_version does not match loaded rules", errors)
        if sid:
            expected_aid = stable_assignment_id(
                subject_type, sid, registry_version, classifier_version
            )
            require(aid == expected_aid,
                    f"{sid}: assignment_id is not bound to current rule identity", errors)
        require(sid in subject_hashes, f"{sid}: assignment references missing source subject", errors)
        if sid in subject_hashes:
            require(assignment.get("source_hash") == subject_hashes[sid],
                    f"{sid}: source_hash does not match source database", errors)

        raw_domains = assignment.get("domain_ids")
        raw_primary_candidates = assignment.get("primary_domain_candidates")
        raw_secondary = assignment.get("secondary_domain_ids")
        domains = raw_domains if isinstance(raw_domains, list) else []
        primary_candidates = (
            raw_primary_candidates if isinstance(raw_primary_candidates, list) else []
        )
        secondary = raw_secondary if isinstance(raw_secondary, list) else []
        primary = assignment.get("primary_domain_id")
        domain_set = {d for d in domains if isinstance(d, str)}
        primary_candidate_set = {
            d for d in primary_candidates if isinstance(d, str)
        }
        secondary_set = {d for d in secondary if isinstance(d, str)}
        require(bool(domains), f"{sid}: domain_ids is empty", errors)
        require(all(isinstance(d, str) and d in registry_ids for d in domains),
                f"{sid}: unknown domain id in {domains}", errors)
        require(all(isinstance(d, str) and d in registry_ids for d in primary_candidates),
                f"{sid}: unknown primary-domain candidate", errors)
        require(all(isinstance(d, str) and d in registry_ids for d in secondary),
                f"{sid}: unknown secondary domain", errors)
        require(bool(primary_candidates), f"{sid}: primary_domain_candidates is empty", errors)
        require(primary_candidate_set.issubset(domain_set),
                f"{sid}: primary candidate absent from domain_ids", errors)
        require(not primary_candidate_set.intersection(secondary_set),
                f"{sid}: primary candidates overlap secondary domains", errors)
        require(domain_set == primary_candidate_set.union(secondary_set),
                f"{sid}: domains do not equal primary candidates plus secondary domains", errors)
        require(assignment.get("review_status") == "candidate",
                f"{sid}: automatic output must remain candidate", errors)

        if primary == "D00":
            require(domains == ["D00"], f"{sid}: D00 must be the only domain when primary", errors)
            require(primary_candidates == ["D00"],
                    f"{sid}: D00 must be the only primary candidate", errors)
            require(not secondary, f"{sid}: D00 assignment cannot have secondary domains", errors)
            require(assignment.get("cross_domain") is False,
                    f"{sid}: D00 assignment cannot be cross-domain", errors)
        elif primary is None:
            require(len(primary_candidates) >= 2,
                    f"{sid}: null primary requires at least two tied candidates", errors)
            require("D00" not in domains, f"{sid}: ambiguous primary cannot include D00", errors)
            require(assignment.get("cross_domain") is True,
                    f"{sid}: ambiguous primary must be cross-domain", errors)
        else:
            require(primary in domains, f"{sid}: primary domain {primary} not in domain_ids", errors)
            require(primary_candidates == [primary],
                    f"{sid}: resolved primary must be the sole primary candidate", errors)
            require("D00" not in domains, f"{sid}: D00 cannot mix with substantive domains", errors)
            require(assignment.get("cross_domain") == (len(domains) > 1),
                    f"{sid}: cross_domain flag inconsistent with memberships", errors)

        raw_scores = assignment.get("scores")
        scores = raw_scores if isinstance(raw_scores, dict) else {}
        score_ids = set(scores)
        require(score_ids == domain_set,
                f"{sid}: scores must close exactly over selected domains", errors)
        projected = expected_domain_selection(
            {
                domain_id: float(score)
                for domain_id, score in scores.items()
                if domain_id != "D00" and isinstance(score, (int, float))
                and not isinstance(score, bool)
            },
            assignment_policy,
        )
        require(
            (primary, primary_candidates, domains, secondary) == projected,
            f"{sid}: primary candidates and secondary domains are inconsistent with scores",
            errors,
        )
        raw_evidence = assignment.get("evidence")
        evidence = raw_evidence if isinstance(raw_evidence, list) else []
        evidence_domains = {
            item.get("domain_id") for item in evidence
            if isinstance(item, dict) and isinstance(item.get("domain_id"), str)
        }
        require(evidence_domains.issubset(domain_set),
                f"{sid}: evidence for unselected domain", errors)
        if domains == ["D00"]:
            require(scores == {"D00": 0.0},
                    f"{sid}: D00 must carry only a zero score", errors)
            require(not evidence,
                    f"{sid}: D00 must not carry substantive evidence", errors)
        else:
            require(domain_set.issubset(evidence_domains),
                    f"{sid}: selected substantive domain lacks evidence", errors)

        assignments[sid] = assignment
        if primary is None:
            primary_key = "AMBIGUOUS"
        elif isinstance(primary, str):
            primary_key = primary
        else:
            primary_key = "INVALID"
        primary_counts[primary_key] += 1
        ambiguous_primary_count += int(primary is None)
        for domain_id in primary_candidates:
            if isinstance(domain_id, str):
                primary_candidate_counts[domain_id] += 1
        cross_domain_count += int(bool(assignment.get("cross_domain")))
        for domain_id in domains:
            if not isinstance(domain_id, str) or domain_id not in registry_ids:
                continue
            membership_counts[domain_id] += 1
            if sid in expected_indexes[domain_id]:
                errors.append(f"{sid}: duplicated expected index membership for {domain_id}")
            expected_indexes[domain_id][sid] = {
                "primary": domain_id == primary,
                "primary_candidate": domain_id in primary_candidate_set,
            }

    require(set(assignments) == set(subject_hashes),
            f"{path}: assignment subjects do not exactly close over source subjects", errors)

    stats = {
        "total_assignments": len(assignments),
        "primary_counts": dict(sorted(primary_counts.items())),
        "primary_candidate_counts": dict(sorted(primary_candidate_counts.items())),
        "membership_counts": dict(sorted(membership_counts.items())),
        "cross_domain_count": cross_domain_count,
        "ambiguous_primary_count": ambiguous_primary_count,
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
            if not isinstance(entry, dict):
                errors.append(f"{path} line {line_no}: index entry must be an object")
                continue
            sid = entry.get("subject_id")
            require(bool(sid), f"{path} line {line_no}: missing subject_id", errors)
            if not isinstance(sid, str) or not sid:
                continue
            require(sid not in actual, f"{path}: duplicate subject_id {sid}", errors)
            require(set(entry) == {"subject_id", "primary", "primary_candidate", "review_status"},
                    f"{path} line {line_no}: index entry has an invalid field set", errors)
            actual[sid] = {
                "primary": bool(entry.get("primary")),
                "primary_candidate": bool(entry.get("primary_candidate")),
            }
            require(entry.get("review_status") == "candidate",
                    f"{path} {sid}: review_status must remain candidate", errors)
        expected_domain = expected.get(domain_id, {})
        require(actual == expected_domain,
                f"{path}: index contents do not exactly match assignments", errors)
        hashes[domain_id] = sha256_file(path)
    return hashes


def expected_domain_selection(scores: dict[str, float], policy: dict) -> tuple[
        str | None, list[str], list[str], list[str]]:
    primary_min = float(policy.get("primary_min_score", 2.0))
    secondary_min = float(policy.get("secondary_min_score", 1.0))
    max_secondary = int(policy.get("max_secondary_domains", 5))
    ranked = sorted(
        ((domain_id, score) for domain_id, score in scores.items() if domain_id != "D00"),
        key=lambda item: (-item[1], item[0]),
    )
    if not ranked or ranked[0][1] < primary_min:
        return "D00", ["D00"], ["D00"], []

    top_score = ranked[0][1]
    primary_candidates = [
        domain_id for domain_id, score in ranked
        if abs(score - top_score) <= 1e-12
    ]
    primary = primary_candidates[0] if len(primary_candidates) == 1 else None
    remaining = [
        domain_id for domain_id, score in ranked
        if domain_id not in set(primary_candidates) and score >= secondary_min
    ]
    capacity = max(0, (max_secondary + 1) - len(primary_candidates))
    secondary = remaining[:capacity]
    return primary, primary_candidates, primary_candidates + secondary, secondary


def validate_training_inheritance(training_assignments: dict, training_source_refs: dict,
                                  research_assignments: dict, registry: dict,
                                  errors: list[str]) -> None:
    policy = registry.get("assignment_policy") or {}
    for tid, assignment in training_assignments.items():
        source_ids = training_source_refs.get(tid, [])
        scores: defaultdict[str, float] = defaultdict(float)
        best_source: dict[str, tuple[float, str]] = {}

        for source_id in source_ids:
            source_assignment = research_assignments.get(source_id)
            require(source_assignment is not None,
                    f"{tid}: source {source_id} has no research assignment", errors)
            if source_assignment is None:
                continue
            source_scores = source_assignment.get("scores")
            if not isinstance(source_scores, dict):
                continue
            for domain_id, raw_score in source_scores.items():
                if domain_id == "D00":
                    continue
                if not isinstance(raw_score, (int, float)) or isinstance(raw_score, bool):
                    continue
                score = float(raw_score)
                if score > scores[domain_id]:
                    scores[domain_id] = score
                previous = best_source.get(domain_id)
                if previous is None or score > previous[0]:
                    best_source[domain_id] = (score, source_id)

        primary, primary_candidates, domain_ids, secondary = expected_domain_selection(
            dict(scores), policy
        )
        expected_scores = {
            domain_id: round(score, 6)
            for domain_id, score in sorted(scores.items())
            if domain_id in set(domain_ids)
        }
        expected_evidence = []
        for domain_id in domain_ids:
            if domain_id == "D00" or domain_id not in best_source:
                continue
            weight, source_id = best_source[domain_id]
            expected_evidence.append({
                "kind": "inherited_source",
                "value": source_id,
                "domain_id": domain_id,
                "weight": round(weight, 6),
            })
        if domain_ids == ["D00"]:
            expected_scores = {"D00": 0.0}
            expected_evidence = []

        require(assignment.get("primary_domain_id") == primary,
                f"{tid}: inherited primary domain does not match source assignments", errors)
        require(assignment.get("primary_domain_candidates") == primary_candidates,
                f"{tid}: inherited primary candidates do not match source assignments", errors)
        require(assignment.get("domain_ids") == domain_ids,
                f"{tid}: inherited domains do not match source assignments", errors)
        require(assignment.get("secondary_domain_ids") == secondary,
                f"{tid}: inherited secondary domains do not match source assignments", errors)
        require(assignment.get("cross_domain") == (len(domain_ids) > 1),
                f"{tid}: inherited cross-domain flag does not match source assignments", errors)
        require(assignment.get("scores") == expected_scores,
                f"{tid}: inherited scores do not match source assignments", errors)
        require((assignment.get("evidence") or []) == expected_evidence,
                f"{tid}: inherited evidence does not exactly match source lineage", errors)


def validate_manifest(path: str, registry_path: str, rules_path: str,
                      research_path: str, training_path: str,
                      research_assignment_path: str, training_assignment_path: str,
                      research_index_hashes: dict, training_index_hashes: dict,
                      research_stats: dict, training_stats: dict,
                      registry: dict, rules: dict, classifier_version: str,
                      errors: list[str]) -> None:
    require(os.path.exists(path), f"missing manifest {path}", errors)
    if not os.path.exists(path):
        return
    manifest = load_json(path)
    rules_digest = sha256_file(rules_path)
    require(manifest.get("domain_schema_version") == DOMAIN_SCHEMA_VERSION,
            "manifest domain schema version mismatch", errors)
    require(manifest.get("registry_version") == registry.get("registry_version"),
            "manifest registry version mismatch", errors)
    require(manifest.get("rules_version") == rules.get("rules_version"),
            "manifest rules version mismatch", errors)
    require(manifest.get("classifier_version") == classifier_version,
            "manifest classifier identity mismatch", errors)
    require(manifest.get("classifier") == {
                "identity": classifier_version,
                "rules_version": rules.get("rules_version"),
                "rules_sha256": rules_digest,
            }, "manifest classifier receipt mismatch", errors)
    require(manifest.get("governance", {}).get("assignment_status") == "candidate",
            "manifest assignment status must remain candidate", errors)
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
    ap.add_argument("--report", default=None,
                    help="Optional path for the machine-readable structural validation report.")
    args = ap.parse_args(argv)

    errors: list[str] = []
    registry = load_json(args.registry)
    rules = load_json(args.rules)
    expected_classifier_version = classifier_identity(rules, sha256_file(args.rules))
    schema = load_json(args.schema)
    validator = Draft202012Validator(schema)
    domains = registry.get("domains") or []
    registry_ids = {d["domain_id"] for d in domains}
    require(len(registry_ids) == len(domains), "domain registry contains duplicate IDs", errors)
    require("D00" in registry_ids, "domain registry must contain D00", errors)
    assignment_policy = registry.get("assignment_policy") or {}
    require(assignment_policy.get("primary_tie_policy") == "mark_ambiguous",
            "registry must explicitly mark equal-score primary ties as ambiguous", errors)

    research_hashes = load_research_sources(args.research, errors)
    training_hashes, training_source_refs = load_training_sources(args.training, errors)

    research_assignments, research_expected_indexes, research_stats = validate_assignment_file(
        args.research_assignments, "research_record", research_hashes, registry_ids, validator,
        registry["registry_version"], expected_classifier_version, assignment_policy, errors
    )
    training_assignments, training_expected_indexes, training_stats = validate_assignment_file(
        args.training_assignments, "training_example", training_hashes, registry_ids, validator,
        registry["registry_version"], expected_classifier_version, assignment_policy, errors
    )

    validate_training_inheritance(
        training_assignments, training_source_refs, research_assignments, registry, errors
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
        research_stats, training_stats, registry, rules, expected_classifier_version, errors,
    )

    report = {
        "status": "PASS" if not errors else "FAIL",
        "registry_version": registry.get("registry_version"),
        "rules_version": rules.get("rules_version"),
        "classifier_version": expected_classifier_version,
        "research": research_stats,
        "training": training_stats,
        "errors": errors[:100],
        "error_count": len(errors),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.report:
        write_json(args.report, report)
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
