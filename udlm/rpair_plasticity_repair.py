from __future__ import annotations

"""Importance-anchored RPAIR repair for certified-capability interference.

This repair branches from the frozen certified parent. It estimates which learned
parameters the previously admitted stages depend on using squared gradients over
prior-stage acquisition bytes only, then rehearses C3 learning while elastically
anchoring those important parameters to the frozen parent. Held-out validation
bytes remain evaluation-only. No trial has promotion authority.
"""

import argparse
import copy
import json
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from .model import UDLM
from .rpair_repair import admitted_prior, evaluate, load_model
from .run_certified import _sample_batch, model_hash, split_lesson


def estimate_importance(model: UDLM, priors, *, context: int, batch_size: int,
                        batches_per_stage: int, device: torch.device, seed: int) -> dict[str, torch.Tensor]:
    random.seed(seed); torch.manual_seed(seed)
    importance = {n: torch.zeros_like(p, device=device) for n, p in model.named_parameters() if p.requires_grad}
    count = 0
    model.train()
    for _, acquisition, _ in priors:
        if len(acquisition) < 3:
            continue
        for _ in range(batches_per_stage):
            x, y = _sample_batch(acquisition, batch_size, context, device)
            logits, _ = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            model.zero_grad(set_to_none=True)
            loss.backward()
            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    importance[n].add_(p.grad.detach().square())
            count += 1
    if count:
        for n in importance:
            importance[n].div_(count)
    # Normalize globally so lambda has a stable interpretation across runs.
    vals = [v.mean() for v in importance.values() if v.numel()]
    global_mean = torch.stack(vals).mean().clamp_min(1e-12) if vals else torch.tensor(1.0, device=device)
    for n in importance:
        importance[n] = importance[n] / global_mean
    model.zero_grad(set_to_none=True)
    model.eval()
    return importance


def anchored_train(model: UDLM, parent: UDLM, stage_acq: bytes, priors, importance,
                   *, steps: int, batch_size: int, context: int, lr: float,
                   replay_probability: float, anchor_lambda: float, seed: int,
                   device: torch.device) -> float:
    random.seed(seed); torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    parent_params = {n: p.detach().clone() for n, p in parent.named_parameters() if p.requires_grad}
    prior_sources = [acq for _, acq, _ in priors if len(acq) >= 3]
    model.train(); last = float('nan')
    for _ in range(steps):
        source = random.choice(prior_sources) if prior_sources and random.random() < replay_probability else stage_acq
        x, y = _sample_batch(source, batch_size, context, device)
        logits, _ = model(x)
        task_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        penalty_terms = []
        for n, p in model.named_parameters():
            if p.requires_grad:
                penalty_terms.append((importance[n] * (p - parent_params[n]).square()).mean())
        anchor = torch.stack(penalty_terms).mean() if penalty_terms else task_loss.new_zeros(())
        loss = task_loss + anchor_lambda * anchor
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        last = float(task_loss.item() / math.log(2.0))
    return last


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('stage_root', type=Path)
    ap.add_argument('failed_stage')
    ap.add_argument('--context', type=int, default=128)
    ap.add_argument('--validation-fraction', type=float, default=0.20)
    ap.add_argument('--steps', type=int, default=200)
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--importance-batches', type=int, default=8)
    ap.add_argument('--max-retention-regression', type=float, default=0.10)
    ap.add_argument('--max-generation-regression', type=float, default=0.12)
    ap.add_argument('--min-acquisition-gain', type=float, default=0.02)
    ap.add_argument('--min-validation-gain', type=float, default=0.01)
    ap.add_argument('--rail-limit', type=float, default=0.90)
    ap.add_argument('--max-new-bytes', type=int, default=128)
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()

    root = args.stage_root; failed = root / args.failed_stage
    diagnosis = json.loads((failed / 'rpair_diagnosis.json').read_text(encoding='utf-8'))
    if diagnosis.get('primary_fault') != 'CERTIFIED_CAPABILITY_INTERFERENCE':
        raise SystemExit('plasticity repair requires CERTIFIED_CAPABILITY_INTERFERENCE')
    device = torch.device(args.device)
    parent = load_model(failed / 'parent.pt', device); parent.eval()
    lesson = (failed / 'curriculum.txt').read_bytes()
    stage_acq, stage_val = split_lesson(lesson, args.validation_fraction)
    priors = admitted_prior(root, args.failed_stage, args.validation_fraction)

    importance_model = copy.deepcopy(parent).to(device)
    importance = estimate_importance(importance_model, priors, context=args.context,
                                     batch_size=args.batch_size, batches_per_stage=args.importance_batches,
                                     device=device, seed=701)
    policies = [
        {'name': 'importance_anchor_light', 'lr': 0.0008, 'replay_probability': 0.30, 'anchor_lambda': 0.5},
        {'name': 'importance_anchor_medium', 'lr': 0.0008, 'replay_probability': 0.30, 'anchor_lambda': 2.0},
        {'name': 'importance_anchor_strong', 'lr': 0.0006, 'replay_probability': 0.35, 'anchor_lambda': 8.0},
    ]

    out = failed / 'rpair_plasticity_repair'; out.mkdir(parents=True, exist_ok=True)
    trials = []
    for i, policy in enumerate(policies, 1):
        model = copy.deepcopy(parent).to(device)
        train_last = anchored_train(model, parent, stage_acq, priors, importance,
                                    steps=args.steps, batch_size=args.batch_size, context=args.context,
                                    lr=policy['lr'], replay_probability=policy['replay_probability'],
                                    anchor_lambda=policy['anchor_lambda'], seed=800+i, device=device)
        metrics = evaluate(model, parent, stage_acq, stage_val, priors, context=args.context,
                           max_retention=args.max_retention_regression,
                           max_generation=args.max_generation_regression,
                           rail_limit=args.rail_limit, min_acq=args.min_acquisition_gain,
                           min_val=args.min_validation_gain, max_new_bytes=args.max_new_bytes,
                           device=device)
        trial = {**policy, 'train_bpb_last': train_last, **metrics}
        torch.save({'config': model.cfg.__dict__, 'state_dict': model.state_dict(), 'metadata': trial},
                   out / f"{i:02d}_{policy['name']}.pt")
        (out / f"{i:02d}_{policy['name']}.json").write_text(json.dumps(trial, indent=2, sort_keys=True), encoding='utf-8')
        trials.append(trial)

    passing = [t for t in trials if t['passes_all_gates']]
    def score(t):
        return (t['heldout_validation_gain_bpb'] + t['acquisition_gain_bpb']
                - 5.0 * max(0.0, t['worst_retention']['regression_bpb'])
                - 2.0 * max(0.0, t['generation_regression']))
    best = max(passing, key=score) if passing else max(trials, key=score)
    importance_summary = {}
    for n, v in importance.items():
        importance_summary[n] = {'mean': float(v.mean().item()), 'max': float(v.max().item())}
    summary = {
        'kind': 'udlm-rpair-plasticity-repair-v1',
        'failed_stage': args.failed_stage,
        'diagnosed_fault': diagnosis['primary_fault'],
        'frozen_parent_hash': model_hash(parent),
        'mechanism': 'prior-acquisition gradient importance + elastic anchoring to frozen parent',
        'holdout_policy': 'prior validation bytes and failed-stage validation bytes are evaluation-only',
        'trial_count': len(trials),
        'passing_trial_count': len(passing),
        'best_trial': best,
        'importance_summary': importance_summary,
        'promotion_authority': 'none; passing trial remains a candidate until separately certified',
        'trials': trials,
    }
    (failed / 'rpair_plasticity_repair_summary.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True), encoding='utf-8')
    print(json.dumps({'passing': len(passing), 'best': best['name'],
                      'best_passes': best['passes_all_gates'],
                      'best_retention': best['worst_retention'],
                      'best_acq': best['acquisition_gain_bpb'],
                      'best_val': best['heldout_validation_gain_bpb']}, sort_keys=True))


if __name__ == '__main__':
    main()
