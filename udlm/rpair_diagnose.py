from __future__ import annotations

"""Automatic RPAIR-style diagnosis for failed UDLM developmental stages.

This module does not repair or mutate a model. It compares the frozen parent and
failed candidate, localizes which certified capabilities regressed, measures
internal-state/generation changes, summarizes parameter drift, and emits an
evidence-bounded continuation requirement. Unknown causes remain explicitly
unresolved rather than being guessed.
"""

import json
import math
from collections import defaultdict
from pathlib import Path

import torch

from .model import UDLM, UDLMConfig
from .run_certified import bpb, generation_panel, panel_degeneration, split_lesson, model_hash


def _load_model(path: Path, device: torch.device) -> UDLM:
    payload = torch.load(path, map_location=device)
    model = UDLM(UDLMConfig(**payload["config"])).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def _parameter_family(name: str) -> str:
    n = name.lower()
    if "readout" in n or "head" in n or "out" in n:
        return "readout"
    if "route" in n or ".q" in n or "q_" in n:
        return "routing_Q"
    if "rel" in n or ".r" in n or "r_" in n:
        return "relation_R"
    if "hist" in n or ".h" in n or "h_" in n:
        return "history_H"
    if "input" in n or "embed" in n or "byte" in n:
        return "input_X"
    if "theta" in n:
        return "theta"
    return "shared_core"


def _drift(parent: UDLM, candidate: UDLM) -> dict:
    by_family: dict[str, list[float]] = defaultdict(list)
    top = []
    pstate = parent.state_dict()
    cstate = candidate.state_dict()
    for name in sorted(pstate):
        a = pstate[name].detach().float().cpu()
        b = cstate[name].detach().float().cpu()
        delta = b - a
        rms = float(delta.square().mean().sqrt().item()) if delta.numel() else 0.0
        base = float(a.square().mean().sqrt().item()) if a.numel() else 0.0
        rel = rms / max(base, 1e-8)
        fam = _parameter_family(name)
        by_family[fam].append(rel)
        top.append((rel, name, rms, base, fam))
    top.sort(reverse=True)
    families = {
        fam: {
            "mean_relative_rms_drift": sum(vals) / len(vals),
            "max_relative_rms_drift": max(vals),
            "tensor_count": len(vals),
        }
        for fam, vals in sorted(by_family.items())
    }
    return {
        "families": families,
        "top_tensors": [
            {
                "name": name,
                "family": fam,
                "relative_rms_drift": rel,
                "delta_rms": rms,
                "parent_rms": base,
            }
            for rel, name, rms, base, fam in top[:12]
        ],
    }


def _state_delta(parent_panel: dict, candidate_panel: dict) -> dict:
    out = {}
    for prompt in sorted(parent_panel):
        prow = parent_panel[prompt]
        crow = candidate_panel[prompt]
        out[prompt] = {
            "generation_degeneration_delta": (
                crow["metrics"]["dominant_byte_ratio"]
                + (0.25 if crow["metrics"]["short_cycle"] else 0.0)
                + 0.25 * max(crow["metrics"]["space_ratio"], crow["metrics"]["nul_ratio"])
            ) - (
                prow["metrics"]["dominant_byte_ratio"]
                + (0.25 if prow["metrics"]["short_cycle"] else 0.0)
                + 0.25 * max(prow["metrics"]["space_ratio"], prow["metrics"]["nul_ratio"])
            ),
            "state_rms_delta": {
                field: crow["state"][field]["rms"] - prow["state"][field]["rms"]
                for field in ("x", "q", "r", "h")
            },
            "state_rail_delta": {
                field: crow["state"][field]["rail_095"] - prow["state"][field]["rail_095"]
                for field in ("x", "q", "r", "h")
            },
        }
    return out


def diagnose_stage_failure(
    stage_root: Path,
    failed_stage: str,
    *,
    context: int = 128,
    validation_fraction: float = 0.20,
    max_retention_regression: float = 0.10,
    rail_limit: float = 0.90,
    max_new_bytes: int = 128,
    device_name: str = "cpu",
) -> dict:
    """Diagnose a frozen HOLD/REJECT stage without mutating any checkpoint."""
    device = torch.device(device_name)
    failed_dir = stage_root / failed_stage
    report_path = failed_dir / "report.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"missing failed-stage report: {report_path}")
    stage_report = json.loads(report_path.read_text(encoding="utf-8"))
    if stage_report.get("decision") == "ADMIT":
        raise ValueError(f"{failed_stage} is admitted; no failure diagnosis required")

    parent = _load_model(failed_dir / "parent.pt", device)
    candidate = _load_model(failed_dir / "candidate.pt", device)
    parent_hash = model_hash(parent)
    candidate_hash = model_hash(candidate)

    prior_stage_rows = []
    for prior_dir in sorted(stage_root.glob("C*"), key=lambda p: int(p.name[1:]) if p.name[1:].isdigit() else 999):
        if prior_dir.name == failed_stage:
            break
        prior_report_path = prior_dir / "report.json"
        curriculum_path = prior_dir / "curriculum.txt"
        if not prior_report_path.is_file() or not curriculum_path.is_file():
            continue
        prior_report = json.loads(prior_report_path.read_text(encoding="utf-8"))
        if prior_report.get("decision") != "ADMIT":
            continue
        _, validation = split_lesson(curriculum_path.read_bytes(), validation_fraction)
        p = bpb(parent, validation, context, device)
        c = bpb(candidate, validation, context, device)
        regression = c - p
        prior_stage_rows.append({
            "stage": prior_dir.name,
            "parent_bpb": p,
            "candidate_bpb": c,
            "regression_bpb": regression,
            "exceeds_retention_limit": regression > max_retention_regression,
        })

    parent_panel = generation_panel(parent, max_new_bytes)
    candidate_panel = generation_panel(candidate, max_new_bytes)
    parent_deg = panel_degeneration(parent_panel)
    candidate_deg = panel_degeneration(candidate_panel)
    param_drift = _drift(parent, candidate)
    state_delta = _state_delta(parent_panel, candidate_panel)

    failed_prior = [r for r in prior_stage_rows if r["exceeds_retention_limit"]]
    worst_prior = max(prior_stage_rows, key=lambda r: r["regression_bpb"], default=None)
    failure_class = stage_report.get("failure_class") or "DIAGNOSTIC_COVERAGE_GAP"

    evidence = []
    primary_fault = None
    confidence = "low"
    continuation = "collect additional diagnostics before proposing a repair"

    if failure_class == "RETENTION_FAILURE":
        if failed_prior:
            primary_fault = "CERTIFIED_CAPABILITY_INTERFERENCE"
            confidence = "high"
            affected = ", ".join(r["stage"] for r in failed_prior)
            evidence.append(
                f"candidate exceeds +{max_retention_regression:.3f} BPB retention limit on certified stage(s): {affected}"
            )
            if worst_prior:
                evidence.append(
                    f"worst retained-stage regression is {worst_prior['stage']} at +{worst_prior['regression_bpb']:.3f} BPB"
                )
            continuation = (
                "branch only from the frozen parent; preserve failed-stage acquisition while reducing per-stage "
                "retention interference below the admission limit; reverify every previously admitted stage before promotion"
            )
        else:
            primary_fault = "AGGREGATE_RETENTION_INTERFERENCE_UNLOCALIZED"
            confidence = "medium"
            evidence.append("aggregate retention gate failed but no single certified stage exceeded the per-stage limit")
            continuation = "increase retention-probe resolution before proposing a repair"
    elif failure_class == "DYNAMIC_FAILURE":
        primary_fault = "STATE_DYNAMICS_OUT_OF_BOUNDS"
        confidence = "high"
        evidence.append("candidate violated an empirically governed internal-state rail limit")
        continuation = "localize the offending state family and restore bounded dynamics before retraining"
    elif failure_class == "GENERATION_FAILURE":
        primary_fault = "AUTONOMOUS_ROLLOUT_REGRESSION"
        confidence = "high"
        evidence.append(f"generation degeneration changed by {candidate_deg - parent_deg:+.3f}")
        continuation = "preserve acquisition while restoring autonomous rollout quality to parent-or-better levels"
    elif failure_class in {"ACQUISITION_FAILURE", "ACQUISITION_EVIDENCE_INSUFFICIENT"}:
        primary_fault = "INSUFFICIENT_ACQUISITION"
        confidence = "high"
        evidence.append("candidate did not establish the minimum acquisition gain")
        continuation = "improve learnability of the current stage without changing the certified parent"
    elif failure_class in {"GENERALIZATION_FAILURE", "GENERALIZATION_EVIDENCE_INSUFFICIENT"}:
        primary_fault = "WITHIN_STAGE_GENERALIZATION_FAILURE"
        confidence = "high"
        evidence.append("candidate did not preserve acquisition on disjoint within-stage heldout bytes")
        continuation = "improve within-stage generalization and reverify against the same frozen parent"
    else:
        primary_fault = "DIAGNOSTIC_COVERAGE_GAP"
        evidence.append("available gates identify failure but do not localize a defensible mechanism")

    diagnosis = {
        "kind": "udlm-rpair-diagnosis-v1",
        "failed_stage": failed_stage,
        "decision": stage_report.get("decision"),
        "failure_class": failure_class,
        "primary_fault": primary_fault,
        "diagnostic_confidence": confidence,
        "parent_model_hash": parent_hash,
        "candidate_model_hash": candidate_hash,
        "stage_metrics": {
            "acquisition_gain_bpb": stage_report.get("acquisition_gain_bpb"),
            "heldout_validation_gain_bpb": stage_report.get("heldout_validation_gain_bpb"),
            "aggregate_retention_regression_bpb": stage_report.get("retention_regression_bpb"),
            "generation_regression": stage_report.get("generation_regression"),
            "dynamic_failure_reasons": stage_report.get("dynamic_failure_reasons", []),
        },
        "certified_retention_probes": prior_stage_rows,
        "worst_affected_certified_stage": worst_prior,
        "generation": {
            "parent_degeneration": parent_deg,
            "candidate_degeneration": candidate_deg,
            "delta": candidate_deg - parent_deg,
        },
        "state_delta": state_delta,
        "parameter_drift": param_drift,
        "evidence": evidence,
        "continuation_requirement": continuation,
        "repair_authority": "none; diagnosis is evidence only and may not mutate the frozen canonical parent",
    }
    return diagnosis


def write_stage_diagnosis(stage_root: Path, failed_stage: str, **kwargs) -> Path:
    diagnosis = diagnose_stage_failure(stage_root, failed_stage, **kwargs)
    path = stage_root / failed_stage / "rpair_diagnosis.json"
    path.write_text(json.dumps(diagnosis, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return path
