from __future__ import annotations

"""Matched C3 localization experiment for UDLM.

Problem definition: Certified Structural Integration Failure.
Leading mechanism hypothesis: localized Q/R interference.

Three conditions branch from the same frozen certified C2 checkpoint:
A) control: unchanged architecture
B) qr_reserve: +1024 initially neutral parameters in Q/R update paths
C) non_qr_reserve: +1024 initially neutral parameters in content/history paths

The experiment verifies function neutrality before training, verifies gradient access to
new parameters, uses identical training exposure/budget for matched conditions, and
reports predictive C3 acquisition, a small held-out structural minimal-pair diagnostic,
per-stage C0-C2 retention, generation, and state health. It does not promote models.
"""

import argparse
import copy
import json
import math
import random
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .model import BOS, UDLM, UDLMConfig, UDLMState
from .run_certified import (
    _append_bank,
    _sample_batch,
    bpb,
    dynamic_failure,
    generation_panel,
    model_hash,
    panel_degeneration,
    split_lesson,
    train_candidate,
)


GRAMMAR_PAIRS = [
    (b"The dogs are here.\n", b"The dogs is here.\n"),
    (b"The dog is here.\n", b"The dog are here.\n"),
    (b"These books are new.\n", b"These books is new.\n"),
    (b"This book is new.\n", b"This book are new.\n"),
    (b"They are ready.\n", b"They is ready.\n"),
    (b"He is ready.\n", b"He are ready.\n"),
    (b"Birds fly away.\n", b"Birds flies away.\n"),
    (b"The bird flies away.\n", b"The bird fly away.\n"),
    (b"They do work.\n", b"They does work.\n"),
    (b"She does work.\n", b"She do work.\n"),
]


class ExpandedUDLM(UDLM):
    """UDLM with one of two matched, zero-output reserve locations."""

    def __init__(self, cfg: UDLMConfig, reserve_kind: str, reserve_hidden: int = 8):
        super().__init__(cfg)
        if reserve_kind not in {"qr_reserve", "non_qr_reserve"}:
            raise ValueError(reserve_kind)
        self.reserve_kind = reserve_kind
        c = cfg
        if reserve_kind == "qr_reserve":
            route_in = c.d_route + c.d_state + c.d_hist + c.d_theta
            rel_in = c.d_rel + 2 * c.d_state + 2 * c.d_route + 2 * c.d_hist
            self.q_reserve = nn.Sequential(
                nn.Linear(route_in, reserve_hidden, bias=False), nn.Tanh(),
                nn.Linear(reserve_hidden, c.d_route, bias=False),
            )
            self.r_reserve = nn.Sequential(
                nn.Linear(rel_in, reserve_hidden, bias=False), nn.Tanh(),
                nn.Linear(reserve_hidden, c.d_rel, bias=False),
            )
            nn.init.zeros_(self.q_reserve[-1].weight)
            nn.init.zeros_(self.r_reserve[-1].weight)
        else:
            uptake_in = c.d_state + c.d_state + c.d_state + c.d_route + c.d_hist + c.d_theta
            hist_in = c.d_hist + c.d_state + c.d_route
            self.content_reserve = nn.Sequential(
                nn.Linear(uptake_in, reserve_hidden, bias=False), nn.Tanh(),
                nn.Linear(reserve_hidden, c.d_state, bias=False),
            )
            self.hist_reserve = nn.Sequential(
                nn.Linear(hist_in, reserve_hidden, bias=False), nn.Tanh(),
                nn.Linear(reserve_hidden, c.d_hist, bias=False),
            )
            nn.init.zeros_(self.content_reserve[-1].weight)
            nn.init.zeros_(self.hist_reserve[-1].weight)

    def reserve_parameters(self):
        for name, p in self.named_parameters():
            if "reserve" in name:
                yield name, p

    def step(self, token: Tensor, state: UDLMState, *, reset_rel: bool = False, reset_hist: bool = False):
        c = self.cfg
        x, r, h, q = state.x, state.r, state.h, state.q
        if reset_rel:
            r = torch.zeros_like(r)
        if reset_hist:
            h = torch.zeros_like(h)

        exc = self.fixed_byte_excitation(token)
        external = c.input_gain * exc.unsqueeze(1) * self.input_pattern.unsqueeze(0)
        theta = self.theta.unsqueeze(0).expand(x.size(0), -1, -1)
        src, dst = self.edge_src, self.edge_dst

        for _ in range(c.micro_steps):
            q_in = torch.cat([q, x, h, theta], dim=-1)
            dq_raw = self.route_update(q_in)
            if self.reserve_kind == "qr_reserve":
                dq_raw = dq_raw + self.q_reserve(q_in)
            dq = torch.tanh(dq_raw)
            q = self._homeostatic_update(
                q, dq, decay=c.q_decay, step=c.q_step,
                target=c.q_rms_target, cap=c.q_rms_cap, strength=c.homeostasis_strength,
            )

            x_s, x_d = x[:, src], x[:, dst]
            q_s, q_d = q[:, src], q[:, dst]
            h_s, h_d = h[:, src], h[:, dst]
            r_in = torch.cat([r, x_s, x_d, q_s, q_d, h_s, h_d], dim=-1)
            dr_raw = self.rel_update(r_in)
            if self.reserve_kind == "qr_reserve":
                dr_raw = dr_raw + self.r_reserve(r_in)
            dr = torch.tanh(dr_raw)
            r = self._homeostatic_update(
                r, dr, decay=c.r_decay, step=c.r_step,
                target=c.r_rms_target, cap=c.r_rms_cap, strength=c.homeostasis_strength,
            )

            th_s, th_d = theta[:, src], theta[:, dst]
            msg_e = self.message(torch.cat([x_s, r, q_s, th_s, th_d], dim=-1))
            incoming = torch.zeros_like(x)
            idx = dst.view(1, -1, 1).expand(x.size(0), -1, c.d_state)
            incoming.scatter_add_(1, idx, msg_e)
            degree = torch.zeros(c.n_nodes, device=x.device, dtype=x.dtype)
            degree.scatter_add_(0, dst, torch.ones_like(dst, dtype=x.dtype))
            incoming = incoming / degree.clamp_min(1.0).view(1, -1, 1)

            local = torch.cat([x, incoming, external, q, h, theta], dim=-1)
            gate = torch.sigmoid(self.uptake(local))
            dx_raw = self.content_delta(local)
            if self.reserve_kind == "non_qr_reserve":
                dx_raw = dx_raw + self.content_reserve(local)
            dx = torch.tanh(dx_raw)
            x = c.x_decay * x + c.x_step * gate * dx + 0.10 * gate * external
            x = self._rms_cap(x, c.state_rms_cap)

            hist_in = torch.cat([h, x, q], dim=-1)
            h_raw = self.hist_update(hist_in)
            if self.reserve_kind == "non_qr_reserve":
                h_raw = h_raw + self.hist_reserve(hist_in)
            h_prop = torch.tanh(h_raw)
            keep = torch.sigmoid(self.hist_gate_logit)
            h = keep * h + (1.0 - keep) * h_prop

        z = torch.cat([x.reshape(x.size(0), -1), q.mean(dim=1), h.mean(dim=1), r.mean(dim=1)], dim=-1)
        logits = self.readout(z)
        return logits, UDLMState(x=x, r=r, h=h, q=q)


def load_parent(path: Path, device: torch.device) -> UDLM:
    payload = torch.load(path, map_location=device)
    model = UDLM(UDLMConfig(**payload["config"])).to(device)
    model.load_state_dict(payload["state_dict"])
    return model


def make_condition(parent: UDLM, condition: str, device: torch.device) -> UDLM:
    if condition == "control":
        return copy.deepcopy(parent).to(device)
    model = ExpandedUDLM(copy.deepcopy(parent.cfg), condition).to(device)
    missing, unexpected = model.load_state_dict(parent.state_dict(), strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected inherited keys: {unexpected}")
    expected_prefix = "q_reserve" if condition == "qr_reserve" else "content_reserve"
    if not any(expected_prefix in k for k in missing):
        raise RuntimeError(f"reserve keys were not recognized as new: {missing}")
    return model


@torch.no_grad()
def neutrality(parent: UDLM, model: UDLM, device: torch.device) -> dict:
    probes = [b"The ", b"Memory ", b"The dogs are here.\n", b"A box is on the table.\n"]
    max_logits = 0.0
    max_state = {"x": 0.0, "q": 0.0, "r": 0.0, "h": 0.0}
    for raw in probes:
        seq = torch.tensor([[BOS, *raw]], dtype=torch.long, device=device)
        lp, sp = parent(seq)
        lm, sm = model(seq)
        max_logits = max(max_logits, float((lp-lm).abs().max().item()))
        for key in max_state:
            a = getattr(sp, key); b = getattr(sm, key)
            max_state[key] = max(max_state[key], float((a-b).abs().max().item()))
    gen_equal = all(parent.generate(p, 64, greedy=True) == model.generate(p, 64, greedy=True) for p in probes[:2])
    passed = max_logits <= 1e-7 and max(max_state.values()) <= 1e-7 and gen_equal
    return {"passed": passed, "max_abs_logit_delta": max_logits, "max_abs_state_delta": max_state,
            "greedy_generation_equal": gen_equal}


def reserve_gradient_access(model: UDLM, stage_acq: bytes, context: int, batch_size: int,
                            device: torch.device, seed: int) -> dict:
    if not isinstance(model, ExpandedUDLM):
        return {"applicable": False}
    random.seed(seed); torch.manual_seed(seed)
    x, y = _sample_batch(stage_acq, batch_size, context, device)
    logits, _ = model(x)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
    model.zero_grad(set_to_none=True); loss.backward()
    reserve_sq = 0.0; total_sq = 0.0; per_param = {}
    reserve_count = 0
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g2 = float(p.grad.detach().square().sum().item())
        total_sq += g2
        if "reserve" in name:
            reserve_sq += g2; reserve_count += p.numel(); per_param[name] = math.sqrt(g2)
    model.zero_grad(set_to_none=True)
    rn = math.sqrt(reserve_sq); tn = math.sqrt(total_sq)
    return {"applicable": True, "reserve_parameter_count": reserve_count,
            "reserve_gradient_norm": rn, "total_gradient_norm": tn,
            "reserve_gradient_fraction": rn / max(tn, 1e-12),
            "per_reserve_parameter_gradient_norm": per_param,
            "passed": rn > 0.0}


@torch.no_grad()
def sequence_nll(model: UDLM, raw: bytes, device: torch.device) -> float:
    seq = [BOS, *raw]
    x = torch.tensor([seq[:-1]], dtype=torch.long, device=device)
    y = torch.tensor([seq[1:]], dtype=torch.long, device=device)
    logits, _ = model(x)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="mean")
    return float(loss.item())


@torch.no_grad()
def grammar_behavior(model: UDLM, device: torch.device) -> dict:
    rows = []
    wins = 0
    margins = []
    for good, bad in GRAMMAR_PAIRS:
        g = sequence_nll(model, good, device); b = sequence_nll(model, bad, device)
        margin = b - g
        wins += int(margin > 0); margins.append(margin)
        rows.append({"good": good.decode(), "bad": bad.decode(), "good_nll": g, "bad_nll": b, "margin": margin,
                     "correct": margin > 0})
    return {"accuracy": wins / len(rows), "mean_margin_nats": sum(margins)/len(margins), "pairs": rows}


def prior_stages(root: Path, validation_fraction: float):
    rows = []
    replay = b""
    for stage in ("C0", "C1", "C2"):
        d = root / stage
        report = json.loads((d/"report.json").read_text())
        if report.get("decision") != "ADMIT":
            raise RuntimeError(f"{stage} is not admitted in source lineage")
        raw = (d/"curriculum.txt").read_bytes()
        acq, val = split_lesson(raw, validation_fraction)
        rows.append((stage, acq, val))
        replay = _append_bank(replay, acq, 262144)
    return rows, replay


def evaluate(model: UDLM, parent: UDLM, stage_acq: bytes, stage_val: bytes, priors,
             parent_behavior: dict, args, device: torch.device) -> dict:
    acq_gain = bpb(parent, stage_acq, args.context, device) - bpb(model, stage_acq, args.context, device)
    val_gain = bpb(parent, stage_val, args.context, device) - bpb(model, stage_val, args.context, device)
    retention = []
    for name, _, val in priors:
        pb = bpb(parent, val, args.context, device); cb = bpb(model, val, args.context, device)
        retention.append({"stage": name, "parent_bpb": pb, "candidate_bpb": cb,
                          "regression_bpb": cb-pb, "passes": cb-pb <= args.max_retention_regression})
    worst = max(retention, key=lambda r: r["regression_bpb"])
    pp = generation_panel(parent, args.max_new_bytes); cp = generation_panel(model, args.max_new_bytes)
    gen_reg = panel_degeneration(cp) - panel_degeneration(pp)
    dyn_bad, dyn_reasons = dynamic_failure(cp, args.rail_limit)
    beh = grammar_behavior(model, device)
    beh_gain = beh["accuracy"] - parent_behavior["accuracy"]
    predictive_pass = acq_gain >= args.min_acquisition_gain and val_gain >= args.min_validation_gain
    retention_pass = all(r["passes"] for r in retention)
    behavior_pass = beh["accuracy"] >= args.min_behavior_accuracy and beh_gain >= args.min_behavior_gain
    core_pass = predictive_pass and retention_pass and gen_reg <= args.max_generation_regression and not dyn_bad
    full_pass = core_pass and behavior_pass
    return {
        "predictive_pass": predictive_pass, "retention_pass": retention_pass,
        "behavior_pass": behavior_pass, "core_integration_pass": core_pass, "full_target_pass": full_pass,
        "acquisition_gain_bpb": acq_gain, "heldout_validation_gain_bpb": val_gain,
        "retention_probes": retention, "worst_retention": worst,
        "generation_regression": gen_reg, "dynamic_failure_reasons": dyn_reasons,
        "grammar_behavior": beh, "grammar_behavior_gain": beh_gain,
        "candidate_model_hash": model_hash(model), "candidate_generation_panel": cp,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage_root", type=Path)
    ap.add_argument("--failed-stage", default="C3")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--context", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.0015)
    ap.add_argument("--replay-ratio", type=float, default=0.35)
    ap.add_argument("--validation-fraction", type=float, default=0.20)
    ap.add_argument("--max-retention-regression", type=float, default=0.10)
    ap.add_argument("--max-generation-regression", type=float, default=0.12)
    ap.add_argument("--min-acquisition-gain", type=float, default=0.02)
    ap.add_argument("--min-validation-gain", type=float, default=0.01)
    ap.add_argument("--min-behavior-accuracy", type=float, default=0.55)
    ap.add_argument("--min-behavior-gain", type=float, default=0.05)
    ap.add_argument("--rail-limit", type=float, default=0.90)
    ap.add_argument("--max-new-bytes", type=int, default=128)
    ap.add_argument("--seeds", default="301,302,303")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    root = args.stage_root; failed = root / args.failed_stage
    parent = load_parent(failed/"parent.pt", device); parent.eval()
    stage_raw = (failed/"curriculum.txt").read_bytes()
    stage_acq, stage_val = split_lesson(stage_raw, args.validation_fraction)
    priors, replay_train = prior_stages(root, args.validation_fraction)
    parent_behavior = grammar_behavior(parent, device)
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    out = failed / "rpair_c3_localization"; out.mkdir(parents=True, exist_ok=True)
    conditions = ["control", "qr_reserve", "non_qr_reserve"]
    all_trials = []

    train_args = SimpleNamespace(
        steps_per_file=args.steps, batch_size=args.batch_size, context=args.context,
        replay_ratio=args.replay_ratio, grad_clip=1.0,
    )

    # Pre-training architecture checks are done once per expansion from the exact same parent.
    architecture_checks = {}
    for condition in conditions:
        random.seed(12345); torch.manual_seed(12345)
        model = make_condition(parent, condition, device)
        check = {"parameter_count": model.parameter_count(),
                 "added_parameter_count": model.parameter_count()-parent.parameter_count()}
        if condition != "control":
            check["neutrality"] = neutrality(parent, model, device)
            check["gradient_access"] = reserve_gradient_access(model, stage_acq, args.context, args.batch_size, device, 991)
        architecture_checks[condition] = check
        if condition != "control" and not check["neutrality"]["passed"]:
            raise RuntimeError(f"{condition} failed neutrality gate: {check['neutrality']}")
        if condition != "control" and not check["gradient_access"]["passed"]:
            raise RuntimeError(f"{condition} failed reserve gradient-access gate: {check['gradient_access']}")

    if architecture_checks["qr_reserve"]["added_parameter_count"] != architecture_checks["non_qr_reserve"]["added_parameter_count"]:
        raise RuntimeError("matched expansion parameter counts differ")

    for seed in seeds:
        for condition in conditions:
            # Reset both initialization and stochastic data selection for a matched seed/condition comparison.
            random.seed(seed); torch.manual_seed(seed)
            model = make_condition(parent, condition, device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
            random.seed(seed); torch.manual_seed(seed)
            train_last = train_candidate(model, optimizer, stage_acq, replay_train, train_args, device)
            metrics = evaluate(model, parent, stage_acq, stage_val, priors, parent_behavior, args, device)
            trial = {"condition": condition, "seed": seed, "train_bpb_last": train_last, **metrics}
            all_trials.append(trial)
            torch.save({"config": model.cfg.__dict__, "state_dict": model.state_dict(),
                        "metadata": trial, "architecture": condition}, out/f"{condition}_seed{seed}.pt")
            (out/f"{condition}_seed{seed}.json").write_text(json.dumps(trial, indent=2, sort_keys=True), encoding="utf-8")

    aggregates = {}
    for condition in conditions:
        rows = [t for t in all_trials if t["condition"] == condition]
        aggregates[condition] = {
            "trial_count": len(rows),
            "core_integration_pass_count": sum(int(t["core_integration_pass"]) for t in rows),
            "full_target_pass_count": sum(int(t["full_target_pass"]) for t in rows),
            "mean_acquisition_gain_bpb": sum(t["acquisition_gain_bpb"] for t in rows)/len(rows),
            "mean_heldout_gain_bpb": sum(t["heldout_validation_gain_bpb"] for t in rows)/len(rows),
            "mean_worst_retention_regression_bpb": sum(t["worst_retention"]["regression_bpb"] for t in rows)/len(rows),
            "max_worst_retention_regression_bpb": max(t["worst_retention"]["regression_bpb"] for t in rows),
            "mean_behavior_accuracy": sum(t["grammar_behavior"]["accuracy"] for t in rows)/len(rows),
            "mean_behavior_gain": sum(t["grammar_behavior_gain"] for t in rows)/len(rows),
        }

    qr = aggregates["qr_reserve"]; non = aggregates["non_qr_reserve"]; ctl = aggregates["control"]
    if qr["core_integration_pass_count"] > 0 and non["core_integration_pass_count"] == 0:
        interpretation = "SUPPORTS_QR_LOCALIZED_PLASTICITY_HYPOTHESIS"
    elif qr["core_integration_pass_count"] > 0 and non["core_integration_pass_count"] > 0:
        interpretation = "SUPPORTS_BROADER_CAPACITY_OR_OPTIMIZATION_FLEXIBILITY"
    elif qr["core_integration_pass_count"] == 0 and non["core_integration_pass_count"] > 0:
        interpretation = "FALSIFIES_QR_EXPANSION_AS_PRIVILEGED_LOCATION_IN_THIS_INTERVENTION"
    else:
        interpretation = "NO_TESTED_EXPANSION_SOLVED_CORE_INTEGRATION"

    summary = {
        "kind": "udlm-c3-matched-localization-v1",
        "problem_definition": "Certified Structural Integration Failure",
        "leading_hypothesis": "localized Q/R interference; insufficient usable Q/R capacity unproven",
        "frozen_parent_hash": model_hash(parent),
        "parent_parameter_count": parent.parameter_count(),
        "architecture_checks": architecture_checks,
        "training": {"steps": args.steps, "batch_size": args.batch_size, "context": args.context,
                     "lr": args.lr, "replay_ratio": args.replay_ratio, "seeds": seeds},
        "gates": {"min_acquisition_gain_bpb": args.min_acquisition_gain,
                  "min_heldout_gain_bpb": args.min_validation_gain,
                  "max_each_prior_stage_regression_bpb": args.max_retention_regression,
                  "max_generation_regression": args.max_generation_regression,
                  "min_behavior_accuracy": args.min_behavior_accuracy,
                  "min_behavior_gain": args.min_behavior_gain},
        "parent_grammar_behavior": parent_behavior,
        "aggregates": aggregates,
        "interpretation": interpretation,
        "nonclaim": "Failure of a tested expansion rejects that intervention; it does not prove Q/R is merely a symptom or that no satisfactory model exists.",
        "promotion_authority": "none",
        "trials": all_trials,
    }
    path = failed/"rpair_c3_localization_summary.json"
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    print(json.dumps({"interpretation": interpretation, "parent_params": parent.parameter_count(),
                      "qr_added": architecture_checks["qr_reserve"]["added_parameter_count"],
                      "non_qr_added": architecture_checks["non_qr_reserve"]["added_parameter_count"],
                      "aggregates": aggregates}, sort_keys=True))


if __name__ == "__main__":
    main()
