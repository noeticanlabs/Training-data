#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Check Domain Registry candidate assignments for obvious distribution collapse.

This is a calibration gate, not a truth validator. It detects anti-discriminative
failure modes such as near-universal multi-labeling or one domain swallowing most
records. Passing these guardrails does not make candidate assignments canonical.
"""
from __future__ import annotations

import argparse
import json


def load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def metrics(stats: dict) -> dict:
    total = int(stats.get("total_assignments") or 0)
    primary = stats.get("primary_counts") or {}
    memberships = stats.get("membership_counts") or {}
    cross = int(stats.get("cross_domain_count") or 0)

    substantive_primary = {
        domain_id: int(count)
        for domain_id, count in primary.items()
        if domain_id != "D00"
    }
    if substantive_primary:
        dominant_domain, dominant_count = max(
            substantive_primary.items(), key=lambda item: item[1]
        )
    else:
        dominant_domain, dominant_count = None, 0

    return {
        "total_assignments": total,
        "cross_domain_rate": (cross / total) if total else 0.0,
        "avg_domains_per_record": (sum(int(v) for v in memberships.values()) / total) if total else 0.0,
        "dominant_primary_domain": dominant_domain,
        "dominant_primary_share": (dominant_count / total) if total else 0.0,
        "unclassified_rate": (int(primary.get("D00", 0)) / total) if total else 0.0,
    }


def check_layer(name: str, stats: dict, limits: dict, errors: list[str]) -> dict:
    observed = metrics(stats)
    require(
        observed["cross_domain_rate"] <= float(limits["max_cross_domain_rate"]),
        f"{name}: cross-domain rate {observed['cross_domain_rate']:.4f} exceeds "
        f"{float(limits['max_cross_domain_rate']):.4f}",
        errors,
    )
    require(
        observed["avg_domains_per_record"] <= float(limits["max_avg_domains_per_record"]),
        f"{name}: avg domains/record {observed['avg_domains_per_record']:.4f} exceeds "
        f"{float(limits['max_avg_domains_per_record']):.4f}",
        errors,
    )
    require(
        observed["dominant_primary_share"] <= float(limits["max_primary_share_any_substantive_domain"]),
        f"{name}: dominant primary share {observed['dominant_primary_share']:.4f} "
        f"({observed['dominant_primary_domain']}) exceeds "
        f"{float(limits['max_primary_share_any_substantive_domain']):.4f}",
        errors,
    )
    return observed


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Check domain candidate distributions for semantic collapse.")
    ap.add_argument("--manifest", default="data/domain_manifest.json")
    ap.add_argument("--rules", default="config/domain_rules.json")
    args = ap.parse_args(argv)

    manifest = load_json(args.manifest)
    rules = load_json(args.rules)
    guardrails = rules.get("distribution_guardrails") or {}
    errors: list[str] = []

    require(
        guardrails.get("status") == "provisional_anti_degeneracy_checks",
        "rules must declare provisional anti-degeneracy guardrails",
        errors,
    )

    observed = {}
    for layer in ("research", "training"):
        limits = guardrails.get(layer)
        require(isinstance(limits, dict), f"missing {layer} distribution guardrails", errors)
        if isinstance(limits, dict):
            observed[layer] = check_layer(layer, manifest.get(layer) or {}, limits, errors)

    report = {
        "status": "PASS" if not errors else "FAIL",
        "rules_version": rules.get("rules_version"),
        "interpretation": "Anti-degeneracy calibration only; PASS does not establish canonical semantic correctness.",
        "observed": observed,
        "guardrails": {
            "research": guardrails.get("research"),
            "training": guardrails.get("training"),
        },
        "errors": errors,
        "error_count": len(errors),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
