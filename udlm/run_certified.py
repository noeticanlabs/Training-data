from __future__ import annotations

"""UDLM certified learning runtime.

Principles adapted from CohBit-Copilot and RPAIR:

* an optimizer update is a candidate, not admitted learning;
* failure is captured before it is diagnosed;
* diagnosis does not imply a repair;
* canonical weights move only after evidence gates pass.

The runtime keeps training replay and retention evidence separate so held-out
bytes used for admission are never reused as replay training material.
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
    """Deterministic state-dict hash with no NumPy dependency."""
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        t = value.detach().cpu().contiguous()
        h.update(name.encode("utf-8"))
        h.update(b"\0")
        h.update(str(t.dtype).encode("ascii"))
        h.update(b"\0")
        h.update(",".join(str(x) for x in t.shape).encode("ascii"))
        h.update(b"\0")
        raw = t.view(torch.uint8).reshape(-1)
        h.update(bytes(raw.tolist()))
        h.update(b"\0")
    return h.hexdigest()


def curriculum_files(root: Path) -> list[Path]:
    if not root.is_dir():
        raise ValueError(f"curriculum directory does not exist: {root}")
    found = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in EXT]
    found.sort(key=lambda p: p.relative_to(root).as_posix().lower())
    if not found:
        raise ValueError(f"no supported curriculum files found under {root}")
    return found


def split_lesson(data: bytes, validation_fraction: float = 0.20) -> tuple[bytes, bytes]:
    if not 0.05 <= validation_fraction <= 0.50:
        raise ValueError("validation_fraction must be between 0.05 and 0.50")
    if len(data) < 64:
        raise ValueError("lesson must contain at least 64 bytes for disjoint acquisition/validation")
    cut = int(len(data) * (1.0 - validation_fraction))
    cut = max(32, min(len(data) - 32, cut))
    return data[:cut], data[cut:]


def _token_sequence(data: bytes) -> list[int]:
    # BOS/EOS are trained at true source boundaries; PAD remains reserved.
    return [BOS, *data, EOS]


def _sample_batch(data: bytes, batch_size: int, context: int, device: torch.device):
    seq = _token_sequence(data)
    if len(seq) < 3:
        raise ValueError("data too short for next-symbol prediction")
    c = min(context, len(seq) - 1)
    max_start = len(seq) - c - 1
    starts = [random.randrange(max_start + 1) if max_start > 0 else 0 for _ in range(batch_size)]
    x = torch.tensor([seq[s:s + c] for s in starts], dtype=torch.long, device=device)
    y = torch.tensor([seq[s + 1:s + c + 1] for s in starts], dtype=torch.long, device=device)
    return x, y


def _evaluation_starts(seq_len: int, context: int, windows: int) -> tuple[int, list[int]]:
    c = min(context, seq_len - 1)
    max_start = seq_len - c - 1
    if max_start <= 0:
        return c, [0]
    n = max(2, min(windows, max_start + 1))
    starts = sorted({round(i * max_start / (n - 1)) for i in range(n)})
    return c, starts


@torch.no_grad()
def bpb(model: UDLM, data: bytes, context: int, device: torch.device, windows: int = 12) -> float:
    """Deterministic fixed-window BPB; does not touch training RNG."""
    if len(data) < 3:
        return float("nan")
    model.eval()
    seq = _token_sequence(data)
    c, starts = _evaluation_starts(len(seq), context, windows)
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
        "x": stats(state.x),
        "r": stats(state.r),
        "h": stats(state.h),
        "q": stats(state.q),
        "top_probability": float(F.softmax(logits, dim=-1).max().item()) if logits is not None else None,
    }


def generation_metrics(out: bytes) -> dict:
    if not out:
        return {
            "length": 0,
            "unique_byte_ratio": 0.0,
            "dominant_byte_ratio": 1.0,
            "space_ratio": 0.0,
            "nul_ratio": 0.0,
            "short_cycle": True,
        }
    counts = [out.count(b) for b in set(out)]
    tail = out[-min(64, len(out)):]
    short_cycle = False
    for period in range(1, min(8, len(tail)) + 1):
        pattern = tail[:period]
        rebuilt = (pattern * ((len(tail) + period - 1) // period))[:len(tail)]
        if sum(a == b for a, b in zip(tail, rebuilt)) / len(tail) >= 0.90:
            short_cycle = True
            break
    return {
        "length": len(out),
        "unique_byte_ratio": len(set(out)) / len(out),
        "dominant_byte_ratio": max(counts) / len(out),
        "space_ratio": out.count(32) / len(out),
        "nul_ratio": out.count(0) / len(out),
        "short_cycle": short_cycle,
    }


@torch.no_grad()
def generation_panel(model: UDLM, max_new_bytes: int) -> dict:
    result = {}
    for prompt in PROMPTS:
        out = model.generate(prompt, max_new_bytes=max_new_bytes, greedy=True)
        result[prompt.decode("utf-8")] = {
            "text": out.decode("utf-8", errors="replace"),
            "hex": out.hex(),
            "metrics": generation_metrics(out),
            "state": health(model, prompt),
        }
    return result


def panel_degeneration(panel: dict) -> float:
    scores = []
    for row in panel.values():
        metrics = row["metrics"]
        score = metrics["dominant_byte_ratio"]
        score += 0.25 if metrics["short_cycle"] else 0.0
        score += 0.25 * max(metrics["space_ratio"], metrics["nul_ratio"])
        scores.append(score)
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
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"receipt ledger malformed at line {line_no}") from exc
            recorded = obj.pop("receipt_hash", None)
            if not recorded:
                raise RuntimeError(f"receipt ledger missing receipt_hash at line {line_no}")
            if obj.get("previous_receipt_hash") != previous:
                raise RuntimeError(f"receipt ledger chain break at line {line_no}")
            expected = hashlib.sha256((previous + "|" + canonical_json(obj)).encode("utf-8")).hexdigest()
            if recorded != expected:
                raise RuntimeError(f"receipt ledger hash mismatch at line {line_no}")
            previous = recorded
    return previous


def write_receipt(path: Path, record: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    previous = validate_receipt_chain(path)
    body = dict(record)
    body["previous_receipt_hash"] = previous
    receipt_hash = hashlib.sha256((previous + "|" + canonical_json(body)).encode("utf-8")).hexdigest()
    body["receipt_hash"] = receipt_hash
    with path.open("a", encoding="utf-8") as f:
        f.write(canonical_json(body) + "\n")
        f.flush()
        os.fsync(f.fileno())
    # Fail closed if the append did not produce a valid chain.
    if validate_receipt_chain(path) != receipt_hash:
        raise RuntimeError("receipt ledger failed post-write validation")
    return receipt_hash


def append_memory(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(canonical_json(record) + "\n")
        f.flush()
        os.fsync(f.fileno())


def save_checkpoint(model: UDLM, path: Path, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": asdict(model.cfg), "state_dict": model.state_dict(), "metadata": metadata}, path)


def load_checkpoint(path: Path, device: torch.device) -> UDLM:
    payload = torch.load(path, map_location=device)
    model = UDLM(UDLMConfig(**payload["config"])).to(device)
    model.load_state_dict(payload["state_dict"])
    return model


def append_bank(bank: bytes, segment: bytes, max_bytes: int) -> bytes:
    if max_bytes <= 0:
        return b""
    merged = bank + b"\n" + segment
    return merged[-max_bytes:] if len(merged) > max_bytes else merged


def train_candidate(model: UDLM, optimizer: torch.optim.Optimizer, acquisition: bytes, replay_train: bytes,
                    args, device: torch.device) -> float:
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


def classify_failure(acquisition_gain: float, validation_gain: float, retention_regression: float,
                     dynamic_bad: bool, generation_regression: float, args) -> str:
    if not all(math.isfinite(v) for v in (acquisition_gain, validation_gain, retention_regression, generation_regression)):
        return "DIAGNOSTIC_COVERAGE_GAP"
    if dynamic_bad:
        return "DYNAMIC_FAILURE"
    if retention_regression > args.hard_replay_regression:
        return "RETENTION_FAILURE"
    if generation_regression > args.hard_generation_regression:
        return "GENERATION_FAILURE"
    if acquisition_gain <= 0.0:
        return "ACQUISITION_FAILURE"
    if validation_gain <= 0.0:
        return "GENERALIZATION_FAILURE"
    return "DIAGNOSTIC_COVERAGE_GAP"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="UDLM certified learning runtime")
    ap.add_argument("curriculum", type=Path)
    ap.add_argument("--checkpoint", type=Path, default=Path("artifacts/udlm_certified.pt"))
    ap.add_argument("--receipts", type=Path, default=Path("artifacts/udlm_certified_receipts.jsonl"))
    ap.add_argument("--capabilities", type=Path, default=Path("artifacts/udlm_capability_memory.jsonl"))
    ap.add_argument("--failures", type=Path, default=Path("artifacts/udlm_failure_memory.jsonl"))
    ap.add_argument("--held-dir", type=Path, default=Path("artifacts/held_candidates"))
    ap.add_argument("--report", type=Path, default=Path("artifacts/udlm_certified_report.json"))
    ap.add_argument("--steps-per-file", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--context", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.0015)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--replay-ratio", type=float, default=0.35)
    ap.add_argument("--replay-bytes", type=int, default=262144)
    ap.add_argument("--retention-bytes", type=int, default=262144)
    ap.add_argument("--validation-fraction", type=float, default=0.20)
    ap.add_argument("--min-acquisition-gain", type=float, default=0.02)
    ap.add_argument("--min-validation-gain", type=float, default=0.01)
    ap.add_argument("--max-replay-regression", type=float, default=0.10)
    ap.add_argument("--hard-replay-regression", type=float, default=0.25)
    ap.add_argument("--max-generation-regression", type=float, default=0.12)
    ap.add_argument("--hard-generation-regression", type=float, default=0.20)
    ap.add_argument("--rail-limit", type=float, default=0.90)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-new-bytes", type=int, default=128)
    ap.add_argument("--device", default="cpu")
    return ap


def main() -> None:
    args = build_parser().parse_args()
    if args.steps_per_file < 1 or args.batch_size < 1 or args.context < 1:
        raise ValueError("steps_per_file, batch_size, and context must be positive")
    if not 0.0 <= args.replay_ratio <= 1.0:
        raise ValueError("replay_ratio must be in [0,1]")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model = UDLM(UDLMConfig()).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    replay_train = b""
    retention_bank = b""
    counts = {"ADMIT": 0, "HOLD": 0, "REJECT": 0}
    lesson_files = curriculum_files(args.curriculum)
    validate_receipt_chain(args.receipts)

    print(f"UDLM certified learner | params={model.parameter_count():,} lessons={len(lesson_files)}")
    print("observe -> candidate -> challenge -> ADMIT/HOLD/REJECT -> diagnose -> receipt")

    for index, path in enumerate(lesson_files, 1):
        lesson = path.read_bytes()
        rel = path.relative_to(args.curriculum).as_posix()
        acquisition, validation = split_lesson(lesson, args.validation_fraction)

        parent_hash = model_hash(model)
        acq_before = bpb(model, acquisition, args.context, device)
        val_before = bpb(model, validation, args.context, device)
        retention_before = bpb(model, retention_bank, args.context, device) if len(retention_bank) >= 3 else None
        before_panel = generation_panel(model, args.max_new_bytes)
        degeneration_before = panel_degeneration(before_panel)

        model_snapshot = copy.deepcopy(model.state_dict())
        optimizer_snapshot = copy.deepcopy(optimizer.state_dict())
        train_last = train_candidate(model, optimizer, acquisition, replay_train, args, device)
        candidate_hash = model_hash(model)

        acq_after = bpb(model, acquisition, args.context, device)
        val_after = bpb(model, validation, args.context, device)
        retention_after = bpb(model, retention_bank, args.context, device) if len(retention_bank) >= 3 else None
        after_panel = generation_panel(model, args.max_new_bytes)
        degeneration_after = panel_degeneration(after_panel)

        acquisition_gain = acq_before - acq_after
        validation_gain = val_before - val_after
        retention_regression = 0.0 if retention_before is None or retention_after is None else retention_after - retention_before
        generation_regression = degeneration_after - degeneration_before
        dynamic_bad, dynamic_reasons = dynamic_failure(after_panel, args.rail_limit)

        finite = all(math.isfinite(v) for v in (acq_after, val_after, train_last, acquisition_gain, validation_gain,
                                                 retention_regression, generation_regression))
        catastrophic = (
            not finite
            or retention_regression > args.hard_replay_regression
            or generation_regression > args.hard_generation_regression
            or dynamic_bad
        )
        admissible = (
            finite
            and acquisition_gain >= args.min_acquisition_gain
            and validation_gain >= args.min_validation_gain
            and retention_regression <= args.max_replay_regression
            and generation_regression <= args.max_generation_regression
            and not dynamic_bad
        )

        failure_class = None
        if admissible:
            decision = "ADMIT"
            counts[decision] += 1
            # Only acquisition bytes enter replay. Held-out validation bytes enter
            # a separate retention bank and are never replay-trained.
            replay_train = append_bank(replay_train, acquisition, args.replay_bytes)
            retention_bank = append_bank(retention_bank, validation, args.retention_bytes)
            append_memory(args.capabilities, {
                "kind": "udlm-capability-memory-v2",
                "timestamp": time.time(),
                "lesson": rel,
                "candidate_hash": candidate_hash,
                "acquisition_gain_bpb": acquisition_gain,
                "heldout_validation_gain_bpb": validation_gain,
                "retention_regression_bpb": retention_regression,
                "generation_regression": generation_regression,
                "evidence": "within_source_heldout_supported",
            })
        elif not catastrophic and acquisition_gain > 0.0:
            decision = "HOLD"
            counts[decision] += 1
            failure_class = classify_failure(acquisition_gain, validation_gain, retention_regression,
                                             dynamic_bad, generation_regression, args)
            save_checkpoint(model, args.held_dir / f"candidate_{index:04d}_{candidate_hash[:12]}.pt", {
                "status": "HELD", "lesson": rel, "parent_hash": parent_hash,
            })
            model.load_state_dict(model_snapshot)
            optimizer.load_state_dict(optimizer_snapshot)
        else:
            decision = "REJECT"
            counts[decision] += 1
            failure_class = classify_failure(acquisition_gain, validation_gain, retention_regression,
                                             dynamic_bad, generation_regression, args)
            model.load_state_dict(model_snapshot)
            optimizer.load_state_dict(optimizer_snapshot)

        if decision != "ADMIT":
            append_memory(args.failures, {
                "kind": "udlm-failure-memory-v2",
                "timestamp": time.time(),
                "lesson": rel,
                "candidate_hash": candidate_hash,
                "failure_class": failure_class,
                "dynamic_reasons": dynamic_reasons,
                "acquisition_gain_bpb": acquisition_gain,
                "heldout_validation_gain_bpb": validation_gain,
                "retention_regression_bpb": retention_regression,
                "generation_regression": generation_regression,
                "continuation_requirement": (
                    "localize with existing diagnostics before proposing structural repair"
                    if failure_class == "DIAGNOSTIC_COVERAGE_GAP"
                    else f"restore {failure_class.lower()} condition and reverify against parent capability"
                ),
            })

        write_receipt(args.receipts, {
            "kind": "udlm-certified-learning-receipt-v2",
            "timestamp": time.time(),
            "lesson_index": index,
            "lesson_path": rel,
            "lesson_sha256": sha256_bytes(lesson),
            "parent_checkpoint_hash": parent_hash,
            "candidate_checkpoint_hash": candidate_hash,
            "train_bpb_last": train_last,
            "acquisition_bpb_before": acq_before,
            "acquisition_bpb_after": acq_after,
            "heldout_validation_bpb_before": val_before,
            "heldout_validation_bpb_after": val_after,
            "retention_bpb_before": retention_before,
            "retention_bpb_after": retention_after,
            "generation_degeneration_before": degeneration_before,
            "generation_degeneration_after": degeneration_after,
            "dynamic_failure_reasons": dynamic_reasons,
            "decision": decision,
            "failure_class": failure_class,
            "counts": counts.copy(),
        })
        print(
            f"[{index:03d}/{len(lesson_files):03d}] {decision:6s} {rel} | "
            f"acq {acq_before:.3f}->{acq_after:.3f} ({acquisition_gain:+.3f}) "
            f"heldout {val_before:.3f}->{val_after:.3f} ({validation_gain:+.3f}) "
            f"retain {retention_regression:+.3f} gen {generation_regression:+.3f}"
        )

    final_hash = model_hash(model)
    tail = validate_receipt_chain(args.receipts)
    save_checkpoint(model, args.checkpoint, {
        "trainer": "udlm-certified-learning-v2",
        "counts": counts,
        "canonical_model_hash": final_hash,
        "last_receipt": tail,
        "validation_semantics": "disjoint within-source heldout; heldout bytes excluded from replay training",
    })
    final_panel = generation_panel(model, args.max_new_bytes)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({
        "canonical_model_hash": final_hash,
        "counts": counts,
        "generation_panel": final_panel,
        "last_receipt": tail,
        "validation_semantics": "disjoint within-source heldout; heldout bytes excluded from replay training",
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"saved {args.checkpoint} counts={counts}")
    for prompt, row in final_panel.items():
        print(repr(prompt) + " -> " + repr(row["text"]))


if __name__ == "__main__":
    main()
