from __future__ import annotations

"""Fail-closed certified learning runtime for UDLM.

An optimizer update is a candidate, not admitted learning. Candidates are
challenged on acquisition, disjoint within-source holdout, retained evidence,
autonomous generation, and state health before canonical weights can move.
"""

import argparse
import copy
import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F

from .model import BOS, EOS, UDLM, UDLMConfig

EXT = {".txt", ".md", ".json", ".jsonl", ".py", ".ts", ".rs"}
PROMPTS = (b"The ", b"Language ", b"Memory ")


def canonical_json(obj: dict) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def model_hash(model: UDLM) -> str:
    """Deterministic tensor hash; works for scalars and requires no NumPy."""
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        t = value.detach().cpu().contiguous()
        h.update(name.encode("utf-8") + b"\0")
        h.update(str(t.dtype).encode("ascii") + b"\0")
        h.update(",".join(str(x) for x in t.shape).encode("ascii") + b"\0")
        raw = t.reshape(-1).view(torch.uint8)
        h.update(bytes(raw.tolist()) + b"\0")
    return h.hexdigest()


def curriculum_files(root: Path) -> list[Path]:
    if not root.is_dir():
        raise ValueError(f"curriculum directory does not exist: {root}")
    found = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in EXT]
    found.sort(key=lambda p: p.relative_to(root).as_posix().lower())
    if not found:
        raise ValueError(f"no supported curriculum files found under {root}")
    return found


def split_lesson(data: bytes, fraction: float) -> tuple[bytes, bytes]:
    if not 0.05 <= fraction <= 0.50:
        raise ValueError("validation_fraction must be in [0.05, 0.50]")
    if len(data) < 64:
        raise ValueError("lesson must contain at least 64 bytes")
    cut = max(32, min(len(data) - 32, int(len(data) * (1.0 - fraction))))
    return data[:cut], data[cut:]


def _tokens(data: bytes) -> list[int]:
    return [BOS, *data, EOS]


def _sample_batch(data: bytes, batch_size: int, context: int, device: torch.device):
    seq = _tokens(data)
    c = min(context, len(seq) - 1)
    if c < 1:
        raise ValueError("data too short for next-symbol prediction")
    max_start = len(seq) - c - 1
    starts = [random.randrange(max_start + 1) if max_start else 0 for _ in range(batch_size)]
    x = torch.tensor([seq[s:s + c] for s in starts], dtype=torch.long, device=device)
    y = torch.tensor([seq[s + 1:s + c + 1] for s in starts], dtype=torch.long, device=device)
    return x, y


@torch.no_grad()
def bpb(model: UDLM, data: bytes, context: int, device: torch.device, windows: int = 12) -> float:
    """Deterministic fixed-window BPB that does not perturb training RNG."""
    if len(data) < 3:
        return float("nan")
    model.eval()
    seq = _tokens(data)
    c = min(context, len(seq) - 1)
    max_start = len(seq) - c - 1
    if max_start <= 0:
        starts = [0]
    else:
        n = max(2, min(windows, max_start + 1))
        starts = sorted({round(i * max_start / (n - 1)) for i in range(n)})
    total = 0.0
    count = 0
    for start in starts:
        x = torch.tensor([seq[start:start + c]], dtype=torch.long, device=device)
        y = torch.tensor([seq[start + 1:start + c + 1]], dtype=torch.long, device=device)
        logits, _ = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum")
        total += float(loss.item())
        count += y.numel()
    return total / max(count, 1) / math.log(2.0)


@torch.no_grad()
def health(model: UDLM, prompt: bytes) -> dict:
    state = model.initial_state(1, model.theta.device)
    logits = None
    for tok in [BOS] + list(prompt):
        logits, state = model.step(torch.tensor([tok], dtype=torch.long, device=model.theta.device), state)

    def stats(v: torch.Tensor) -> dict:
        a = v.detach().abs()
        return {
            "mean_abs": float(a.mean().item()),
            "rms": float(v.detach().square().mean().sqrt().item()),
            "rail_095": float((a > 0.95).float().mean().item()),
            "max_abs": float(a.max().item()),
        }

    return {
        "x": stats(state.x), "r": stats(state.r), "h": stats(state.h), "q": stats(state.q),
        "top_probability": float(F.softmax(logits, dim=-1).max().item()),
    }


def generation_metrics(out: bytes) -> dict:
    if not out:
        return {"length": 0, "unique_byte_ratio": 0.0, "dominant_byte_ratio": 1.0,
                "space_ratio": 0.0, "nul_ratio": 0.0, "short_cycle": True}
    tail = out[-min(64, len(out)):]
    short_cycle = False
    for period in range(1, min(8, len(tail)) + 1):
        pattern = tail[:period]
        rebuilt = (pattern * ((len(tail) + period - 1) // period))[:len(tail)]
        if sum(a == b for a, b in zip(tail, rebuilt)) / len(tail) >= 0.90:
            short_cycle = True
            break
    counts = [out.count(b) for b in set(out)]
    return {
        "length": len(out), "unique_byte_ratio": len(set(out)) / len(out),
        "dominant_byte_ratio": max(counts) / len(out), "space_ratio": out.count(32) / len(out),
        "nul_ratio": out.count(0) / len(out), "short_cycle": short_cycle,
    }


@torch.no_grad()
def generation_panel(model: UDLM, max_new_bytes: int) -> dict:
    result = {}
    for prompt in PROMPTS:
        out = model.generate(prompt, max_new_bytes=max_new_bytes, greedy=True)
        result[prompt.decode()] = {
            "text": out.decode("utf-8", errors="replace"), "hex": out.hex(),
            "metrics": generation_metrics(out), "state": health(model, prompt),
        }
    return result


def panel_degeneration(panel: dict) -> float:
    scores = []
    for row in panel.values():
        m = row["metrics"]
        scores.append(m["dominant_byte_ratio"] + (0.25 if m["short_cycle"] else 0.0)
                      + 0.25 * max(m["space_ratio"], m["nul_ratio"]))
    return sum(scores) / max(len(scores), 1)


def dynamic_failure(panel: dict, rail_limit: float) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    for prompt, row in panel.items():
        for field in ("x", "r", "h", "q"):
            rail = row["state"][field]["rail_095"]
            if rail > rail_limit:
                reasons.append(f"{field}_rail>{rail_limit:.2f}@{prompt}:{rail:.3f}")
    return bool(reasons), reasons


def validate_receipt_chain(path: Path) -> str:
    if not path.exists():
        return "GENESIS"
    previous = "GENESIS"
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"receipt ledger malformed at line {line_no}") from exc
            recorded_hash = record.pop("receipt_hash", None)
            if not recorded_hash:
                raise RuntimeError(f"receipt ledger missing receipt_hash at line {line_no}")
            if record.get("previous_receipt_hash") != previous:
                raise RuntimeError(f"receipt ledger chain break at line {line_no}")
            expected = hashlib.sha256((previous + "|" + canonical_json(record)).encode()).hexdigest()
            if recorded_hash != expected:
                raise RuntimeError(f"receipt ledger hash mismatch at line {line_no}")
            previous = recorded_hash
    return previous


def write_receipt(path: Path, record: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    previous = validate_receipt_chain(path)
    body = dict(record)
    body["previous_receipt_hash"] = previous
    receipt_hash = hashlib.sha256((previous + "|" + canonical_json(body)).encode()).hexdigest()
    body["receipt_hash"] = receipt_hash
    with path.open("a", encoding="utf-8") as f:
        f.write(canonical_json(body) + "\n")
        f.flush(); os.fsync(f.fileno())
    if validate_receipt_chain(path) != receipt_hash:
        raise RuntimeError("receipt ledger failed post-write validation")
    return receipt_hash


def append_memory(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(canonical_json(record) + "\n")
        f.flush(); os.fsync(f.fileno())


def save_checkpoint(model: UDLM, path: Path, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": asdict(model.cfg), "state_dict": model.state_dict(), "metadata": metadata}, path)


def load_checkpoint(path: Path, device: torch.device) -> UDLM:
    payload = torch.load(path, map_location=device)
    model = UDLM(UDLMConfig(**payload["config"])).to(device)
    model.load_state_dict(payload["state_dict"])
    return model


def _append_bank(bank: bytes, segment: bytes, max_bytes: int) -> bytes:
    if max_bytes <= 0:
        return b""
    merged = bank + b"\n" + segment
    return merged[-max_bytes:] if len(merged) > max_bytes else merged


def train_candidate(model: UDLM, optimizer: torch.optim.Optimizer, acquisition: bytes,
                    replay_train: bytes, args, device: torch.device) -> float:
    model.train()
    last = float("nan")
    for _ in range(args.steps_per_file):
        source = replay_train if replay_train and random.random() < args.replay_ratio else acquisition
        x, y = _sample_batch(source, args.batch_size, args.context, device)
        logits, _ = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        last = float(loss.item() / math.log(2.0))
    return last


def failure_class(acq_gain: float, val_gain: float, retention_regression: float,
                  dynamic_bad: bool, generation_regression: float, args) -> str:
    if not all(math.isfinite(v) for v in (acq_gain, val_gain, retention_regression, generation_regression)):
        return "DIAGNOSTIC_COVERAGE_GAP"
    if dynamic_bad:
        return "DYNAMIC_FAILURE"
    if retention_regression > args.hard_replay_regression:
        return "RETENTION_FAILURE"
    if generation_regression > args.hard_generation_regression:
        return "GENERATION_FAILURE"
    if acq_gain <= 0:
        return "ACQUISITION_FAILURE"
    if val_gain <= 0:
        return "GENERALIZATION_FAILURE"
    return "DIAGNOSTIC_COVERAGE_GAP"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="UDLM certified learning runtime")
    p.add_argument("curriculum", type=Path)
    p.add_argument("--checkpoint", type=Path, default=Path("artifacts/udlm_certified.pt"))
    p.add_argument("--receipts", type=Path, default=Path("artifacts/udlm_certified_receipts.jsonl"))
    p.add_argument("--capabilities", type=Path, default=Path("artifacts/udlm_capability_memory.jsonl"))
    p.add_argument("--failures", type=Path, default=Path("artifacts/udlm_failure_memory.jsonl"))
    p.add_argument("--held-dir", type=Path, default=Path("artifacts/held_candidates"))
    p.add_argument("--report", type=Path, default=Path("artifacts/udlm_certified_report.json"))
    p.add_argument("--steps-per-file", type=int, default=200)
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
        raise ValueError("steps_per_file, batch_size, and context must be positive")
    if not 0 <= args.replay_ratio <= 1:
        raise ValueError("replay_ratio must be in [0,1]")

    random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model = UDLM(UDLMConfig()).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    replay_train = b""
    retention_bank = b""
    counts = {"ADMIT": 0, "HOLD": 0, "REJECT": 0}
    lessons = curriculum_files(args.curriculum)
    validate_receipt_chain(args.receipts)

    print(f"UDLM certified learner | params={model.parameter_count():,} lessons={len(lessons)}")
    print("observe -> candidate -> challenge -> ADMIT/HOLD/REJECT -> diagnose -> receipt")

    for index, path in enumerate(lessons, 1):
        lesson = path.read_bytes()
        rel = path.relative_to(args.curriculum).as_posix()
        acquisition, validation = split_lesson(lesson, args.validation_fraction)

        parent_hash = model_hash(model)
        acq_before = bpb(model, acquisition, args.context, device)
        val_before = bpb(model, validation, args.context, device)
        ret_before = bpb(model, retention_bank, args.context, device) if len(retention_bank) >= 3 else None
        panel_before = generation_panel(model, args.max_new_bytes)
        deg_before = panel_degeneration(panel_before)
        model_snapshot = copy.deepcopy(model.state_dict())
        optimizer_snapshot = copy.deepcopy(optimizer.state_dict())

        train_last = train_candidate(model, optimizer, acquisition, replay_train, args, device)
        candidate_hash = model_hash(model)
        acq_after = bpb(model, acquisition, args.context, device)
        val_after = bpb(model, validation, args.context, device)
        ret_after = bpb(model, retention_bank, args.context, device) if len(retention_bank) >= 3 else None
        panel_after = generation_panel(model, args.max_new_bytes)
        deg_after = panel_degeneration(panel_after)

        acq_gain = acq_before - acq_after
        val_gain = val_before - val_after
        retention_regression = 0.0 if ret_before is None or ret_after is None else ret_after - ret_before
        generation_regression = deg_after - deg_before
        dynamic_bad, dynamic_reasons = dynamic_failure(panel_after, args.rail_limit)
        finite = all(math.isfinite(v) for v in (acq_after, val_after, train_last, acq_gain, val_gain,
                                                 retention_regression, generation_regression))
        catastrophic = (not finite or retention_regression > args.hard_replay_regression
                        or generation_regression > args.hard_generation_regression or dynamic_bad)
        admissible = (finite and acq_gain >= args.min_acquisition_gain
                      and val_gain >= args.min_validation_gain
                      and retention_regression <= args.max_replay_regression
                      and generation_regression <= args.max_generation_regression and not dynamic_bad)

        classification = None
        if admissible:
            decision = "ADMIT"; counts[decision] += 1
            replay_train = _append_bank(replay_train, acquisition, args.replay_bytes)
            retention_bank = _append_bank(retention_bank, validation, args.retention_bytes)
            append_memory(args.capabilities, {
                "kind": "udlm-capability-memory-v2", "timestamp": time.time(), "lesson": rel,
                "candidate_hash": candidate_hash, "acquisition_gain_bpb": acq_gain,
                "heldout_validation_gain_bpb": val_gain, "retention_regression_bpb": retention_regression,
                "generation_regression": generation_regression, "evidence": "within_source_heldout_supported",
            })
        elif not catastrophic and acq_gain > 0:
            decision = "HOLD"; counts[decision] += 1
            classification = failure_class(acq_gain, val_gain, retention_regression, dynamic_bad,
                                           generation_regression, args)
            save_checkpoint(model, args.held_dir / f"candidate_{index:04d}_{candidate_hash[:12]}.pt",
                            {"status": "HELD", "lesson": rel, "parent_hash": parent_hash})
            model.load_state_dict(model_snapshot); optimizer.load_state_dict(optimizer_snapshot)
        else:
            decision = "REJECT"; counts[decision] += 1
            classification = failure_class(acq_gain, val_gain, retention_regression, dynamic_bad,
                                           generation_regression, args)
            model.load_state_dict(model_snapshot); optimizer.load_state_dict(optimizer_snapshot)

        if decision != "ADMIT":
            append_memory(args.failures, {
                "kind": "udlm-failure-memory-v2", "timestamp": time.time(), "lesson": rel,
                "candidate_hash": candidate_hash, "failure_class": classification,
                "dynamic_reasons": dynamic_reasons, "acquisition_gain_bpb": acq_gain,
                "heldout_validation_gain_bpb": val_gain, "retention_regression_bpb": retention_regression,
                "generation_regression": generation_regression,
                "continuation_requirement": ("localize with existing diagnostics before proposing structural repair"
                    if classification == "DIAGNOSTIC_COVERAGE_GAP"
                    else f"restore {classification.lower()} condition and reverify against parent capability"),
            })

        write_receipt(args.receipts, {
            "kind": "udlm-certified-learning-receipt-v2", "timestamp": time.time(),
            "lesson_index": index, "lesson_path": rel, "lesson_sha256": sha256_bytes(lesson),
            "parent_checkpoint_hash": parent_hash, "candidate_checkpoint_hash": candidate_hash,
            "train_bpb_last": train_last, "acquisition_bpb_before": acq_before,
            "acquisition_bpb_after": acq_after, "heldout_validation_bpb_before": val_before,
            "heldout_validation_bpb_after": val_after, "retention_bpb_before": ret_before,
            "retention_bpb_after": ret_after, "generation_degeneration_before": deg_before,
            "generation_degeneration_after": deg_after, "dynamic_failure_reasons": dynamic_reasons,
            "decision": decision, "failure_class": classification, "counts": counts.copy(),
        })
        print(f"[{index:03d}/{len(lessons):03d}] {decision:6s} {rel} | "
              f"acq {acq_gain:+.3f} heldout {val_gain:+.3f} retain {retention_regression:+.3f} "
              f"gen {generation_regression:+.3f}")

    final_hash = model_hash(model)
    tail = validate_receipt_chain(args.receipts)
    metadata = {
        "trainer": "udlm-certified-learning-v2", "counts": counts,
        "canonical_model_hash": final_hash, "last_receipt": tail,
        "validation_semantics": "disjoint within-source heldout; heldout bytes excluded from replay training",
    }
    save_checkpoint(model, args.checkpoint, metadata)
    final_panel = generation_panel(model, args.max_new_bytes)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({**metadata, "generation_panel": final_panel}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved {args.checkpoint} counts={counts}")
    for prompt, row in final_panel.items():
        print(repr(prompt) + " -> " + repr(row["text"]))


if __name__ == "__main__":
    main()
