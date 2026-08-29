#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build candidate domain assignments and domain indexes without mutating v0.02 databases.

The canonical research and training JSONL files remain unchanged. This script creates
sidecar assignments keyed by existing IDs and lightweight domain indexes containing
IDs only. Automatic classifications are always review_status="candidate".
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

DOMAIN_SCHEMA_VERSION = "0.01"
CLASSIFIER_VERSION = "domain_rules_v0.01"


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


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
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on {path} line {line_no}: {exc}") from exc


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_assignment_id(subject_type: str, subject_id: str, registry_version: str) -> str:
    payload = f"{subject_type}|{subject_id}|{registry_version}|{CLASSIFIER_VERSION}"
    return "domain:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def compile_rules(rules: dict) -> dict[str, list[tuple[str, re.Pattern[str]]]]:
    compiled: dict[str, list[tuple[str, re.Pattern[str]]]] = {}
    for domain_id, terms in rules.get("keywords", {}).items():
        compiled[domain_id] = []
        for term in terms:
            pattern = re.compile(r"(?<!\w)" + re.escape(term.casefold()) + r"(?!\w)")
            compiled[domain_id].append((term, pattern))
    return compiled


def add_evidence(scores: defaultdict[str, float], evidence: list[dict], kind: str,
                 value: str, domain_id: str, weight: float) -> None:
    scores[domain_id] += weight
    evidence.append({
        "kind": kind,
        "value": value,
        "domain_id": domain_id,
        "weight": weight,
    })


def select_domains(scores: dict[str, float], policy: dict) -> tuple[str | None, list[str], list[str]]:
    primary_min = float(policy.get("primary_min_score", 2.0))
    secondary_min = float(policy.get("secondary_min_score", 1.0))
    max_secondary = int(policy.get("max_secondary_domains", 5))

    ranked = sorted(
        ((domain_id, score) for domain_id, score in scores.items() if domain_id != "D00"),
        key=lambda item: (-item[1], item[0]),
    )
    if not ranked or ranked[0][1] < primary_min:
        return "D00", ["D00"], []

    primary = ranked[0][0]
    secondary = [
        domain_id for domain_id, score in ranked[1:]
        if score >= secondary_min
    ][:max_secondary]
    return primary, [primary] + secondary, secondary


def classify_research_record(record: dict, registry: dict, rules: dict,
                             compiled_rules: dict) -> dict:
    scores: defaultdict[str, float] = defaultdict(float)
    evidence: list[dict] = []
    weights = rules.get("weights", {})
    legacy_weight = float(weights.get("legacy_tag", 2.0))
    title_weight = float(weights.get("title_keyword", 2.0))
    content_weight = float(weights.get("content_keyword", 1.0))
    legacy_map = registry.get("legacy_tag_candidate_map", {})

    raw_tags = list(record.get("project_ids") or []) + list(record.get("tags") or [])
    tags = sorted({str(tag) for tag in raw_tags if tag and not str(tag).startswith("scope:")})
    for tag in tags:
        for domain_id in legacy_map.get(tag, []):
            add_evidence(scores, evidence, "legacy_tag", tag, domain_id, legacy_weight)

    title = str(record.get("title") or "").casefold()
    content = str((record.get("content") or {}).get("text") or "").casefold()

    for domain_id, patterns in compiled_rules.items():
        for term, pattern in patterns:
            if title and pattern.search(title):
                add_evidence(scores, evidence, "keyword", f"title:{term}", domain_id, title_weight)
            if content and content != title and pattern.search(content):
                add_evidence(scores, evidence, "keyword", f"content:{term}", domain_id, content_weight)

    policy = registry.get("assignment_policy", {})
    primary, domain_ids, secondary = select_domains(dict(scores), policy)
    selected = set(domain_ids)
    selected_scores = {
        domain_id: round(score, 6)
        for domain_id, score in sorted(scores.items())
        if domain_id in selected and domain_id != "D00"
    }
    selected_evidence = [item for item in evidence if item["domain_id"] in selected]
    if primary == "D00":
        selected_scores = {"D00": 0.0}
        selected_evidence = []

    subject_id = str(record["record_id"])
    return {
        "schema_version": DOMAIN_SCHEMA_VERSION,
        "assignment_id": stable_assignment_id("research_record", subject_id, registry["registry_version"]),
        "registry_version": registry["registry_version"],
        "subject_type": "research_record",
        "subject_id": subject_id,
        "primary_domain_id": primary,
        "domain_ids": domain_ids,
        "secondary_domain_ids": secondary,
        "cross_domain": len([d for d in domain_ids if d != "D00"]) > 1,
        "assignment_method": "rule_based",
        "review_status": "candidate",
        "classifier_version": CLASSIFIER_VERSION,
        "scores": selected_scores,
        "evidence": selected_evidence,
        "source_hash": (record.get("provenance") or {}).get("source_hash"),
        "created_at": now_iso(),
        "notes": "Automatic candidate assignment; not canonical domain membership.",
    }


def classify_training_example(example: dict, research_assignments: dict[str, dict],
                              registry: dict) -> dict:
    scores: defaultdict[str, float] = defaultdict(float)
    best_source: dict[str, tuple[float, str]] = {}

    for source_id in example.get("source_record_ids") or []:
        assignment = research_assignments.get(str(source_id))
        if not assignment:
            continue
        for domain_id, score in (assignment.get("scores") or {}).items():
            if domain_id == "D00":
                continue
            score = float(score)
            if score > scores[domain_id]:
                scores[domain_id] = score
            previous = best_source.get(domain_id)
            if previous is None or score > previous[0]:
                best_source[domain_id] = (score, str(source_id))

    policy = registry.get("assignment_policy", {})
    primary, domain_ids, secondary = select_domains(dict(scores), policy)
    selected = set(domain_ids)
    selected_scores = {
        domain_id: round(score, 6)
        for domain_id, score in sorted(scores.items())
        if domain_id in selected and domain_id != "D00"
    }
    evidence = []
    for domain_id in domain_ids:
        if domain_id == "D00" or domain_id not in best_source:
            continue
        weight, source_id = best_source[domain_id]
        evidence.append({
            "kind": "inherited_source",
            "value": source_id,
            "domain_id": domain_id,
            "weight": round(weight, 6),
        })
    if primary == "D00":
        selected_scores = {"D00": 0.0}
        evidence = []

    subject_id = str(example["training_example_id"])
    return {
        "schema_version": DOMAIN_SCHEMA_VERSION,
        "assignment_id": stable_assignment_id("training_example", subject_id, registry["registry_version"]),
        "registry_version": registry["registry_version"],
        "subject_type": "training_example",
        "subject_id": subject_id,
        "primary_domain_id": primary,
        "domain_ids": domain_ids,
        "secondary_domain_ids": secondary,
        "cross_domain": len([d for d in domain_ids if d != "D00"]) > 1,
        "assignment_method": "inherited",
        "review_status": "candidate",
        "classifier_version": CLASSIFIER_VERSION,
        "scores": selected_scores,
        "evidence": evidence,
        "source_hash": (example.get("transform") or {}).get("content_hash"),
        "created_at": now_iso(),
        "notes": "Inherited from source research-record candidate assignments; not canonical domain membership.",
    }


def prepare_index_handles(index_dir: str, domains: list[dict]):
    os.makedirs(index_dir, exist_ok=True)
    for stale in Path(index_dir).glob("*.jsonl"):
        stale.unlink()
    handles = {}
    paths = {}
    for domain in domains:
        path = os.path.join(index_dir, f"{domain['domain_id']}_{domain['slug']}.jsonl")
        handles[domain["domain_id"]] = open(path, "w", encoding="utf-8")
        paths[domain["domain_id"]] = path
    return handles, paths


def close_handles(handles: dict) -> None:
    for handle in handles.values():
        handle.close()


def write_index_entry(handles: dict, assignment: dict) -> None:
    primary = assignment.get("primary_domain_id")
    for domain_id in assignment.get("domain_ids") or []:
        handle = handles.get(domain_id)
        if handle is None:
            continue
        entry = {
            "subject_id": assignment["subject_id"],
            "primary": domain_id == primary,
            "review_status": assignment["review_status"],
        }
        handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")


def build_research(research_path: str, output_dir: str, registry: dict, rules: dict,
                   compiled_rules: dict):
    os.makedirs(output_dir, exist_ok=True)
    assignment_path = os.path.join(output_dir, "assignments.jsonl")
    index_dir = os.path.join(os.path.dirname(output_dir), "indexes", "domains")
    handles, index_paths = prepare_index_handles(index_dir, registry["domains"])

    assignment_map: dict[str, dict] = {}
    primary_counts = Counter()
    membership_counts = Counter()
    cross_domain_count = 0
    total = 0

    try:
        with open(assignment_path, "w", encoding="utf-8") as out:
            for record in iter_jsonl(research_path):
                assignment = classify_research_record(record, registry, rules, compiled_rules)
                out.write(json.dumps(assignment, ensure_ascii=False, sort_keys=True) + "\n")
                assignment_map[assignment["subject_id"]] = assignment
                total += 1
                primary_counts[assignment["primary_domain_id"]] += 1
                for domain_id in assignment["domain_ids"]:
                    membership_counts[domain_id] += 1
                cross_domain_count += int(assignment["cross_domain"])
                write_index_entry(handles, assignment)
    finally:
        close_handles(handles)

    return assignment_map, assignment_path, index_paths, {
        "total_assignments": total,
        "primary_counts": dict(sorted(primary_counts.items())),
        "membership_counts": dict(sorted(membership_counts.items())),
        "cross_domain_count": cross_domain_count,
    }


def build_training(training_path: str, output_dir: str, research_assignments: dict[str, dict],
                   registry: dict):
    os.makedirs(output_dir, exist_ok=True)
    assignment_path = os.path.join(output_dir, "assignments.jsonl")
    index_dir = os.path.join(os.path.dirname(output_dir), "indexes", "domains")
    handles, index_paths = prepare_index_handles(index_dir, registry["domains"])

    primary_counts = Counter()
    membership_counts = Counter()
    cross_domain_count = 0
    total = 0

    try:
        with open(assignment_path, "w", encoding="utf-8") as out:
            for example in iter_jsonl(training_path):
                assignment = classify_training_example(example, research_assignments, registry)
                out.write(json.dumps(assignment, ensure_ascii=False, sort_keys=True) + "\n")
                total += 1
                primary_counts[assignment["primary_domain_id"]] += 1
                for domain_id in assignment["domain_ids"]:
                    membership_counts[domain_id] += 1
                cross_domain_count += int(assignment["cross_domain"])
                write_index_entry(handles, assignment)
    finally:
        close_handles(handles)

    return assignment_path, index_paths, {
        "total_assignments": total,
        "primary_counts": dict(sorted(primary_counts.items())),
        "membership_counts": dict(sorted(membership_counts.items())),
        "cross_domain_count": cross_domain_count,
    }


def file_hashes(paths: dict[str, str]) -> dict[str, str]:
    return {domain_id: sha256_file(path) for domain_id, path in sorted(paths.items())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build candidate domain sidecars for research and training databases.")
    ap.add_argument("--research", default="data/research_db/records.jsonl")
    ap.add_argument("--training", default="data/training_db/candidates.jsonl")
    ap.add_argument("--registry", default="config/domain_registry.json")
    ap.add_argument("--rules", default="config/domain_rules.json")
    ap.add_argument("--research-domain-dir", default="data/research_db/domains")
    ap.add_argument("--training-domain-dir", default="data/training_db/domains")
    args = ap.parse_args(argv)

    registry = load_json(args.registry)
    rules = load_json(args.rules)
    compiled_rules = compile_rules(rules)

    research_assignments, research_assignment_path, research_index_paths, research_stats = build_research(
        args.research, args.research_domain_dir, registry, rules, compiled_rules
    )
    training_assignment_path, training_index_paths, training_stats = build_training(
        args.training, args.training_domain_dir, research_assignments, registry
    )

    manifest = {
        "domain_schema_version": DOMAIN_SCHEMA_VERSION,
        "registry_version": registry["registry_version"],
        "rules_version": rules["rules_version"],
        "classifier_version": CLASSIFIER_VERSION,
        "generated_at": now_iso(),
        "governance": {
            "assignment_status": "candidate",
            "canonical_content_mutated": False,
            "automatic_assignment_is_canonical": False,
        },
        "inputs": {
            "research_records": {"path": args.research, "sha256": sha256_file(args.research)},
            "training_candidates": {"path": args.training, "sha256": sha256_file(args.training)},
            "registry": {"path": args.registry, "sha256": sha256_file(args.registry)},
            "rules": {"path": args.rules, "sha256": sha256_file(args.rules)},
        },
        "research": {
            **research_stats,
            "assignments_path": research_assignment_path,
            "assignments_sha256": sha256_file(research_assignment_path),
            "index_sha256": file_hashes(research_index_paths),
        },
        "training": {
            **training_stats,
            "assignments_path": training_assignment_path,
            "assignments_sha256": sha256_file(training_assignment_path),
            "index_sha256": file_hashes(training_index_paths),
        },
    }

    manifest_path = "data/domain_manifest.json"
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")

    print(json.dumps({
        "manifest": manifest_path,
        "research": research_stats,
        "training": training_stats,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
