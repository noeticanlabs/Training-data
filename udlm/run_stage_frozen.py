from __future__ import annotations

"""Stage-frozen certified developmental training for UDLM.

Each curriculum stage is trained against the last admitted frozen parent. A stage
produces its own immutable evidence bundle. Progression stops on HOLD or REJECT;
no later stage is allowed to train on an uncertified state.
"""

import argparse
import copy
import hashlib
import json
import math
import os
import random
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import torch

from .model import UDLM, UDLMConfig
from .run_certified import (
    _append_bank,
    bpb,
    canonical_json,
    curriculum_files,
    dynamic_failure,
    failure_class,
    generation_panel,
    model_hash,
    panel_degeneration,
    sha256_bytes,
    split_lesson,
    train_candidate,
    validate_receipt_chain,
    write_receipt,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def _save_model(model: UDLM, path: Path, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": asdict(model.cfg), "state_dict": model.state_dict(), "metadata": metadata}, path)


def _file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _stage_name(index: int, path: Path) -> str:
    parent = path.parent.name
    prefix = parent[:2] if len(parent) >= 2 and parent[:2].isdigit() else f"{index-1:02d}"
    return f"C{int(prefix)}"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="UDLM stage-frozen certified developmental trainer")
    p.add_argument("curriculum", type=Path)
    p.add_argument("--out", type=Path, default=Path("artifacts/stage_frozen"))
    p.add_argument("--steps-per-stage", dest="steps_per_file", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--context", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.0015)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--replay-ratio", type=float, default=0.35)
    p.add_argument("--replay-bytes", type=int, default=262144)
    p.add_argument("--retention-bytes", type=int, default=262144)
    p.add_argument("--validation-fraction", type=float, default=0.20)
    p.add_argument("--min-acquisition-gain", type=float, default=0.02)
    p.add_argument("--min-validation-gain", type=float, default=0.01)
    p.add_argument("--max-replay-regression", type=float, default=0.10)
    p.add_argument("--hard-replay-regression", type=float, default=0.25)
    p.add_argument("--max-generation-regression", type=float, default=0.12)
    p.add_argument("--hard-generation-regression", type=float, default=0.20)
    p.add_argument("--rail-limit", type=float, default=0.90)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--max-new-bytes", type=int, default=128)
    p.add_argument("--device", default="cpu")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.steps_per_file < 1 or args.batch_size < 1 or args.context < 1:
        raise ValueError("steps_per_stage, batch_size, and context must be positive")
    if not 0 <= args.replay_ratio <= 1:
        raise ValueError("replay_ratio must be in [0,1]")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    ledger = args.out / "lineage_receipts.jsonl"
    if ledger.exists():
        ledger.unlink()
    validate_receipt_chain(ledger)

    model = UDLM(UDLMConfig()).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    replay_train = b""
    retention_bank = b""
    lessons = curriculum_files(args.curriculum)
    lineage = []
    stopped = False

    print(f"UDLM stage-frozen learner | params={model.parameter_count():,} stages={len(lessons)}")
    print("train stage -> challenge -> freeze evidence -> ADMIT/HOLD/REJECT -> stop unless ADMIT")

    for index, path in enumerate(lessons, 1):
        stage = _stage_name(index, path)
        rel = path.relative_to(args.curriculum).as_posix()
        stage_dir = args.out / stage
        stage_dir.mkdir(parents=True, exist_ok=True)
        lesson = path.read_bytes()
        acquisition, validation = split_lesson(lesson, args.validation_fraction)
        shutil.copy2(path, stage_dir / "curriculum.txt")

        parent_hash = model_hash(model)
        parent_snapshot = copy.deepcopy(model.state_dict())
        optimizer_snapshot = copy.deepcopy(optimizer.state_dict())
        parent_panel = generation_panel(model, args.max_new_bytes)
        acq_before = bpb(model, acquisition, args.context, device)
        val_before = bpb(model, validation, args.context, device)
        ret_before = bpb(model, retention_bank, args.context, device) if len(retention_bank) >= 3 else None
        deg_before = panel_degeneration(parent_panel)

        _save_model(model, stage_dir / "parent.pt", {
            "kind": "udlm-stage-parent-v1", "stage": stage, "stage_index": index,
            "model_hash": parent_hash, "curriculum_sha256": sha256_bytes(lesson),
        })

        train_last = train_candidate(model, optimizer, acquisition, replay_train, args, device)
        candidate_hash = model_hash(model)
        candidate_panel = generation_panel(model, args.max_new_bytes)
        acq_after = bpb(model, acquisition, args.context, device)
        val_after = bpb(model, validation, args.context, device)
        ret_after = bpb(model, retention_bank, args.context, device) if len(retention_bank) >= 3 else None
        deg_after = panel_degeneration(candidate_panel)
        acq_gain = acq_before - acq_after
        val_gain = val_before - val_after
        retention_regression = 0.0 if ret_before is None or ret_after is None else ret_after - ret_before
        generation_regression = deg_after - deg_before
        dynamic_bad, dynamic_reasons = dynamic_failure(candidate_panel, args.rail_limit)
        finite = all(math.isfinite(v) for v in (
            acq_after, val_after, train_last, acq_gain, val_gain,
            retention_regression, generation_regression,
        ))
        catastrophic = (
            not finite
            or retention_regression > args.hard_replay_regression
            or generation_regression > args.hard_generation_regression
            or dynamic_bad
        )
        admissible = (
            finite
            and acq_gain >= args.min_acquisition_gain
            and val_gain >= args.min_validation_gain
            and retention_regression <= args.max_replay_regression
            and generation_regression <= args.max_generation_regression
            and not dynamic_bad
        )

        if admissible:
            decision = "ADMIT"
            classification = None
        elif not catastrophic and acq_gain > 0:
            decision = "HOLD"
            classification = failure_class(
                acq_gain, val_gain, retention_regression, dynamic_bad, generation_regression, args
            )
        else:
            decision = "REJECT"
            classification = failure_class(
                acq_gain, val_gain, retention_regression, dynamic_bad, generation_regression, args
            )

        _save_model(model, stage_dir / "candidate.pt", {
            "kind": "udlm-stage-candidate-v1", "stage": stage, "stage_index": index,
            "decision": decision, "parent_hash": parent_hash, "candidate_hash": candidate_hash,
            "curriculum_sha256": sha256_bytes(lesson),
        })

        if decision == "ADMIT":
            canonical_hash = candidate_hash
            replay_train = _append_bank(replay_train, acquisition, args.replay_bytes)
            retention_bank = _append_bank(retention_bank, validation, args.retention_bytes)
        else:
            model.load_state_dict(parent_snapshot)
            optimizer.load_state_dict(optimizer_snapshot)
            canonical_hash = parent_hash

        canonical_panel = generation_panel(model, args.max_new_bytes)
        receipt_hash = write_receipt(ledger, {
            "kind": "udlm-stage-frozen-receipt-v1",
            "timestamp": time.time(),
            "stage": stage,
            "stage_index": index,
            "lesson_path": rel,
            "lesson_sha256": sha256_bytes(lesson),
            "parent_checkpoint_hash": parent_hash,
            "candidate_checkpoint_hash": candidate_hash,
            "canonical_after_hash": canonical_hash,
            "train_bpb_last": train_last,
            "acquisition_bpb_before": acq_before,
            "acquisition_bpb_after": acq_after,
            "heldout_validation_bpb_before": val_before,
            "heldout_validation_bpb_after": val_after,
            "retention_bpb_before": ret_before,
            "retention_bpb_after": ret_after,
            "generation_degeneration_before": deg_before,
            "generation_degeneration_after": deg_after,
            "dynamic_failure_reasons": dynamic_reasons,
            "decision": decision,
            "failure_class": classification,
        })

        canonical_meta = {
            "kind": "udlm-stage-certified-canonical-v1",
            "stage": stage,
            "stage_index": index,
            "decision": decision,
            "parent_hash": parent_hash,
            "candidate_hash": candidate_hash,
            "canonical_model_hash": canonical_hash,
            "receipt_hash": receipt_hash,
            "curriculum_sha256": sha256_bytes(lesson),
            "progression_allowed": decision == "ADMIT",
        }
        _save_model(model, stage_dir / "canonical.pt", canonical_meta)

        report = {
            **canonical_meta,
            "lesson_path": rel,
            "failure_class": classification,
            "acquisition_gain_bpb": acq_gain,
            "heldout_validation_gain_bpb": val_gain,
            "retention_regression_bpb": retention_regression,
            "generation_regression": generation_regression,
            "dynamic_failure_reasons": dynamic_reasons,
            "parent_generation_panel": parent_panel,
            "candidate_generation_panel": candidate_panel,
            "canonical_generation_panel": canonical_panel,
        }
        _write_json(stage_dir / "report.json", report)
        _write_json(stage_dir / "receipt.json", {"receipt_hash": receipt_hash, "decision": decision})

        manifest_files = {}
        for fp in sorted(stage_dir.iterdir()):
            if fp.is_file() and fp.name != "manifest.json":
                manifest_files[fp.name] = {"bytes": fp.stat().st_size, "sha256": _file_sha(fp)}
        manifest = {
            "kind": "udlm-stage-frozen-artifact-v1",
            "stage": stage,
            "decision": decision,
            "progression_allowed": decision == "ADMIT",
            "parent_model_hash": parent_hash,
            "candidate_model_hash": candidate_hash,
            "canonical_model_hash": canonical_hash,
            "curriculum_sha256": sha256_bytes(lesson),
            "receipt_hash": receipt_hash,
            "files": manifest_files,
        }
        _write_json(stage_dir / "manifest.json", manifest)
        lineage.append(manifest)

        print(
            f"[{stage}] {decision:6s} | acq {acq_gain:+.3f} heldout {val_gain:+.3f} "
            f"retain {retention_regression:+.3f} gen {generation_regression:+.3f} "
            f"canonical={canonical_hash[:12]}"
        )

        if decision != "ADMIT":
            stopped = True
            print(f"progression stopped at {stage}: {decision} ({classification})")
            break

    tail = validate_receipt_chain(ledger)
    summary = {
        "kind": "udlm-stage-frozen-lineage-v1",
        "parameters": model.parameter_count(),
        "stages_available": len(lessons),
        "stages_executed": len(lineage),
        "stopped": stopped,
        "last_stage": lineage[-1]["stage"] if lineage else None,
        "last_decision": lineage[-1]["decision"] if lineage else None,
        "final_canonical_model_hash": model_hash(model),
        "last_receipt_hash": tail,
        "lineage": lineage,
    }
    _write_json(args.out / "lineage_summary.json", summary)
    _save_model(model, args.out / "latest_certified.pt", {
        "kind": "udlm-stage-frozen-latest-v1",
        "canonical_model_hash": summary["final_canonical_model_hash"],
        "last_receipt_hash": tail,
        "last_stage": summary["last_stage"],
        "last_decision": summary["last_decision"],
    })
    print(canonical_json(summary))


if __name__ == "__main__":
    main()
