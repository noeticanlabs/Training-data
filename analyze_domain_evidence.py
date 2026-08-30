#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diagnose evidence strength in generated domain assignments.

This is an observational diagnostic, not a validator and not a truth classifier.
It measures how candidate assignments were produced so calibration decisions can be
based on corpus evidence rather than on aggregate domain counts alone.

In particular it reports:
- resolved substantive primaries that exist only through fallback-primary policy;
- resolved primaries supported only by legacy tags;
- assignments with only one selected evidence item;
- exact-score primary ambiguity;
- primary score margins for resolved substantive primaries;
- per-domain concentration of fallback and legacy-only primaries.

No acceptance thresholds are imposed here. A high or low rate is evidence to inspect,
not by itself proof that an assignment is correct or incorrect.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter


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
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on {path} line {line_no}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path} line {line_no}: assignment must be a JSON object")
            yield value


def safe_rate(count: int, total: int) -> float:
    return (count / total) if total else 0.0


def score_margin(scores: dict, primary: str) -> float | None:
    ranked = sorted(
        (
            (str(domain_id), float(score))
            for domain_id, score in scores.items()
            if domain_id != "D00" and isinstance(score, (int, float)) and not isinstance(score, bool)
        ),
        key=lambda item: (-item[1], item[0]),
    )
    if not ranked or ranked[0][0] != primary:
        return None
    if len(ranked) == 1:
        return None
    return ranked[0][1] - ranked[1][1]


def summarize(path: str, policy: dict) -> dict:
    primary_min = float(policy.get("primary_min_score", 2.0))
    allow_fallback = bool(policy.get("allow_fallback_primary", True))

    total = 0
    d00 = 0
    ambiguous = 0
    resolved_substantive = 0
    cross_domain = 0
    fallback_primary = 0
    legacy_only_primary = 0
    single_evidence_primary = 0
    keyword_only_primary = 0
    mixed_evidence_primary = 0

    primary_counts = Counter()
    fallback_by_domain = Counter()
    legacy_only_by_domain = Counter()
    evidence_kind_counts = Counter()
    margin_buckets = Counter()

    for assignment in iter_jsonl(path):
        total += 1
        primary = assignment.get("primary_domain_id")
        scores = assignment.get("scores") or {}
        evidence = assignment.get("evidence") or []
        cross_domain += int(bool(assignment.get("cross_domain")))

        for item in evidence:
            if isinstance(item, dict) and isinstance(item.get("kind"), str):
                evidence_kind_counts[item["kind"]] += 1

        if primary == "D00":
            d00 += 1
            continue
        if primary is None:
            ambiguous += 1
            continue
        if not isinstance(primary, str):
            continue

        resolved_substantive += 1
        primary_counts[primary] += 1
        primary_score = scores.get(primary)
        primary_evidence = [
            item for item in evidence
            if isinstance(item, dict) and item.get("domain_id") == primary
        ]
        primary_kinds = {
            item.get("kind") for item in primary_evidence
            if isinstance(item.get("kind"), str)
        }

        if isinstance(primary_score, (int, float)) and not isinstance(primary_score, bool):
            if allow_fallback and float(primary_score) < primary_min:
                fallback_primary += 1
                fallback_by_domain[primary] += 1

        if primary_evidence and primary_kinds == {"legacy_tag"}:
            legacy_only_primary += 1
            legacy_only_by_domain[primary] += 1
        elif primary_evidence and primary_kinds == {"keyword"}:
            keyword_only_primary += 1
        elif primary_evidence and len(primary_kinds) > 1:
            mixed_evidence_primary += 1

        if len(primary_evidence) == 1:
            single_evidence_primary += 1

        margin = score_margin(scores, primary)
        if margin is None:
            margin_buckets["single_selected_domain"] += 1
        elif margin < 0.5:
            margin_buckets["margin_lt_0.5"] += 1
        elif margin < 1.0:
            margin_buckets["margin_0.5_to_lt_1"] += 1
        elif margin < 2.0:
            margin_buckets["margin_1_to_lt_2"] += 1
        else:
            margin_buckets["margin_ge_2"] += 1

    return {
        "path": path,
        "total_assignments": total,
        "d00_count": d00,
        "d00_rate": safe_rate(d00, total),
        "ambiguous_primary_count": ambiguous,
        "ambiguous_primary_rate": safe_rate(ambiguous, total),
        "resolved_substantive_primary_count": resolved_substantive,
        "cross_domain_count": cross_domain,
        "cross_domain_rate": safe_rate(cross_domain, total),
        "fallback_primary_policy_enabled": allow_fallback,
        "primary_min_score": primary_min,
        "fallback_primary_count": fallback_primary,
        "fallback_primary_rate_of_all": safe_rate(fallback_primary, total),
        "fallback_primary_rate_of_resolved_substantive": safe_rate(
            fallback_primary, resolved_substantive
        ),
        "legacy_only_primary_count": legacy_only_primary,
        "legacy_only_primary_rate_of_resolved_substantive": safe_rate(
            legacy_only_primary, resolved_substantive
        ),
        "single_evidence_primary_count": single_evidence_primary,
        "single_evidence_primary_rate_of_resolved_substantive": safe_rate(
            single_evidence_primary, resolved_substantive
        ),
        "keyword_only_primary_count": keyword_only_primary,
        "mixed_evidence_primary_count": mixed_evidence_primary,
        "primary_counts": dict(sorted(primary_counts.items())),
        "fallback_primary_by_domain": dict(sorted(fallback_by_domain.items())),
        "legacy_only_primary_by_domain": dict(sorted(legacy_only_by_domain.items())),
        "evidence_kind_counts": dict(sorted(evidence_kind_counts.items())),
        "resolved_primary_score_margin_buckets": dict(sorted(margin_buckets.items())),
        "interpretation": (
            "Diagnostic only. These measurements describe classifier evidence strength; "
            "they do not establish semantic correctness or canonical domain membership."
        ),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Analyze evidence strength in domain assignments.")
    ap.add_argument("--research-assignments", default="data/research_db/domains/assignments.jsonl")
    ap.add_argument("--training-assignments", default="data/training_db/domains/assignments.jsonl")
    ap.add_argument("--registry", default="config/domain_registry.json")
    ap.add_argument("--output", default=None, help="Optional JSON report path")
    args = ap.parse_args(argv)

    registry = load_json(args.registry)
    policy = registry.get("assignment_policy") or {}
    report = {
        "status": "OBSERVATIONAL",
        "registry_version": registry.get("registry_version"),
        "research": summarize(args.research_assignments, policy),
        "training": summarize(args.training_assignments, policy),
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(rendered)
            f.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
