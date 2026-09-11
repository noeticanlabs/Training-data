from __future__ import annotations

"""Bounded RPAIR repair rehearsal for a failed frozen UDLM stage.

Repairs never mutate the certified parent. For retention interference, the module
branches from the frozen parent, rehearses a small evidence-derived set of
conservative training policies, challenges each against acquisition, heldout,
every admitted prior stage, generation, and state-health gates, then emits a
repair recommendation. Promotion remains separate from repair rehearsal.
"""

import argparse
import copy
import json
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from .model import UDLM, UDLMConfig
from .run_certified import (
    _sample_batch, bpb, dynamic_failure, generation_panel, model_hash,
    panel_degeneration, split_lesson,
)


def load_model(path: Path, device: torch.device) -> UDLM:
    payload = torch.load(path, map_location=device)
    model = UDLM(UDLMConfig(**payload["config"])).to(device)
    model.load_state_dict(payload["state_dict"])
    return model


def admitted_prior(stage_root: Path, failed_stage: str, validation_fraction: float):
    rows = []
    for d in sorted(stage_root.glob("C*"), key=lambda p: int(p.name[1:]) if p.name[1:].isdigit() else 999):
        if d.name == failed_stage:
            break
        rp = d / "report.json"
        cp = d / "curriculum.txt"
        if not rp.is_file() or not cp.is_file():
            continue
        report = json.loads(rp.read_text(encoding="utf-8"))
        if report.get("decision") != "ADMIT":
            continue
        acq, val = split_lesson(cp.read_bytes(), validation_fraction)
        rows.append((d.name, acq, val))
    return rows


def train_policy(model: UDLM, stage_acq: bytes, priors, *, steps: int, batch_size: int,
                 context: int, lr: float, weight_decay: float, grad_clip: float,
                 stage_probability: float, target_stage: str | None, seed: int,
                 device: torch.device) -> float:
    random.seed(seed); torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train(); last = float("nan")
    prior_map = {name: acq for name, acq, _ in priors}
    names = list(prior_map)
    for _ in range(steps):
        if not names or random.random() < stage_probability:
            source = stage_acq
        else:
            # Diagnosis-guided replay: half of prior draws target the worst affected certified stage.
            if target_stage in prior_map and random.random() < 0.5:
                source = prior_map[target_stage]
            else:
                source = prior_map[random.choice(names)]
        x, y = _sample_batch(source, batch_size, context, device)
        logits, _ = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip); optimizer.step()
        last = float(loss.item() / math.log(2.0))
    return last


def evaluate(model: UDLM, parent: UDLM, stage_acq: bytes, stage_val: bytes, priors, *,
             context: int, max_retention: float, max_generation: float, rail_limit: float,
             min_acq: float, min_val: float, max_new_bytes: int, device: torch.device):
    acq_gain = bpb(parent, stage_acq, context, device) - bpb(model, stage_acq, context, device)
    val_gain = bpb(parent, stage_val, context, device) - bpb(model, stage_val, context, device)
    retention = []
    for name, _, val in priors:
        pb = bpb(parent, val, context, device); cb = bpb(model, val, context, device)
        retention.append({"stage": name, "parent_bpb": pb, "candidate_bpb": cb, "regression_bpb": cb-pb})
    worst = max(retention, key=lambda r: r["regression_bpb"], default={"regression_bpb": 0.0, "stage": None})
    pp = generation_panel(parent, max_new_bytes); cp = generation_panel(model, max_new_bytes)
    gen_reg = panel_degeneration(cp) - panel_degeneration(pp)
    dyn_bad, dyn_reasons = dynamic_failure(cp, rail_limit)
    passes = (acq_gain >= min_acq and val_gain >= min_val and worst["regression_bpb"] <= max_retention
              and gen_reg <= max_generation and not dyn_bad)
    return {
        "passes_all_gates": passes,
        "acquisition_gain_bpb": acq_gain,
        "heldout_validation_gain_bpb": val_gain,
        "retention_probes": retention,
        "worst_retention": worst,
        "generation_regression": gen_reg,
        "dynamic_failure_reasons": dyn_reasons,
        "candidate_model_hash": model_hash(model),
        "candidate_generation_panel": cp,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage_root", type=Path)
    ap.add_argument("failed_stage")
    ap.add_argument("--context", type=int, default=128)
    ap.add_argument("--validation-fraction", type=float, default=0.20)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-retention-regression", type=float, default=0.10)
    ap.add_argument("--max-generation-regression", type=float, default=0.12)
    ap.add_argument("--min-acquisition-gain", type=float, default=0.02)
    ap.add_argument("--min-validation-gain", type=float, default=0.01)
    ap.add_argument("--rail-limit", type=float, default=0.90)
    ap.add_argument("--max-new-bytes", type=int, default=128)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    root = args.stage_root; failed = root / args.failed_stage
    diagnosis = json.loads((failed / "rpair_diagnosis.json").read_text(encoding="utf-8"))
    if diagnosis.get("primary_fault") != "CERTIFIED_CAPABILITY_INTERFERENCE":
        raise SystemExit("bounded auto-repair currently requires CERTIFIED_CAPABILITY_INTERFERENCE")
    device = torch.device(args.device)
    parent = load_model(failed / "parent.pt", device); parent.eval()
    lesson = (failed / "curriculum.txt").read_bytes(); stage_acq, stage_val = split_lesson(lesson, args.validation_fraction)
    priors = admitted_prior(root, args.failed_stage, args.validation_fraction)
    worst = (diagnosis.get("worst_affected_certified_stage") or {}).get("stage")

    # Evidence-bounded policy family: same architecture/objective, only optimization pressure changes.
    policies = [
        {"name": "targeted_replay", "lr": 0.0010, "stage_probability": 0.55},
        {"name": "conservative_targeted_replay", "lr": 0.0006, "stage_probability": 0.50},
        {"name": "high_retention_replay", "lr": 0.0008, "stage_probability": 0.40},
    ]
    out = failed / "rpair_repair"; out.mkdir(parents=True, exist_ok=True)
    trials = []
    for i, policy in enumerate(policies, 1):
        model = copy.deepcopy(parent).to(device)
        train_last = train_policy(model, stage_acq, priors, steps=args.steps, batch_size=args.batch_size,
                                  context=args.context, lr=policy["lr"], weight_decay=1e-4, grad_clip=1.0,
                                  stage_probability=policy["stage_probability"], target_stage=worst,
                                  seed=100+i, device=device)
        metrics = evaluate(model, parent, stage_acq, stage_val, priors, context=args.context,
                           max_retention=args.max_retention_regression, max_generation=args.max_generation_regression,
                           rail_limit=args.rail_limit, min_acq=args.min_acquisition_gain,
                           min_val=args.min_validation_gain, max_new_bytes=args.max_new_bytes, device=device)
        trial = {**policy, "train_bpb_last": train_last, **metrics}
        torch.save({"config": model.cfg.__dict__, "state_dict": model.state_dict(), "metadata": trial}, out / f"{i:02d}_{policy['name']}.pt")
        (out / f"{i:02d}_{policy['name']}.json").write_text(json.dumps(trial, indent=2, sort_keys=True), encoding="utf-8")
        trials.append(trial)

    passing = [t for t in trials if t["passes_all_gates"]]
    def score(t):
        return (t["heldout_validation_gain_bpb"] + t["acquisition_gain_bpb"]
                - 4.0 * max(0.0, t["worst_retention"]["regression_bpb"])
                - 2.0 * max(0.0, t["generation_regression"]))
    best = max(passing, key=score) if passing else max(trials, key=score)
    summary = {
        "kind": "udlm-rpair-repair-rehearsal-v1",
        "failed_stage": args.failed_stage,
        "diagnosed_fault": diagnosis["primary_fault"],
        "frozen_parent_hash": model_hash(parent),
        "worst_affected_certified_stage": worst,
        "repair_policy_scope": "optimization/replay pressure only; architecture and language objective frozen",
        "trial_count": len(trials),
        "passing_trial_count": len(passing),
        "best_trial": best,
        "promotion_authority": "none; a passing repair remains a candidate until separately certified",
        "trials": trials,
    }
    (failed / "rpair_repair_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    print(json.dumps({"passing": len(passing), "best": best["name"], "best_passes": best["passes_all_gates"],
                      "best_retention": best["worst_retention"], "best_acq": best["acquisition_gain_bpb"],
                      "best_val": best["heldout_validation_gain_bpb"]}, sort_keys=True))

if __name__ == "__main__":
    main()
