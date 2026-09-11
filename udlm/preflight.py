from __future__ import annotations

"""Fast fail-closed preflight for the UDLM certified-learning machinery."""

import copy
import tempfile
from pathlib import Path

import torch

from .model import UDLM, UDLMConfig
from .run_certified import (
    bpb,
    generation_panel,
    model_hash,
    save_checkpoint,
    load_checkpoint,
    train_candidate,
    validate_receipt_chain,
    write_receipt,
)


class Args:
    steps_per_file = 2
    batch_size = 2
    context = 32
    replay_ratio = 0.0
    grad_clip = 1.0


def main() -> None:
    torch.manual_seed(7)
    device = torch.device("cpu")
    model = UDLM(UDLMConfig()).to(device)
    original = model_hash(model)
    if original != model_hash(model):
        raise SystemExit("preflight failed: model hash is not deterministic")

    probe = (b"Language is a structured sequence of symbols and relations. " * 8)
    score = bpb(model, probe, 32, device)
    if not torch.isfinite(torch.tensor(score)):
        raise SystemExit("preflight failed: BPB is non-finite")

    panel = generation_panel(model, 16)
    if set(panel) != {"The ", "Language ", "Memory "}:
        raise SystemExit("preflight failed: generation panel incomplete")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before_state = copy.deepcopy(model.state_dict())
    loss = train_candidate(model, optimizer, probe, b"", Args(), device)
    if not torch.isfinite(torch.tensor(loss)):
        raise SystemExit("preflight failed: training loss is non-finite")
    changed = model_hash(model)
    if changed == original:
        raise SystemExit("preflight failed: training did not change model state")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        ledger = root / "receipts.jsonl"
        first = write_receipt(ledger, {"kind": "preflight", "step": 1})
        second = write_receipt(ledger, {"kind": "preflight", "step": 2})
        if first == second or validate_receipt_chain(ledger) != second:
            raise SystemExit("preflight failed: receipt chain invalid")

        checkpoint = root / "model.pt"
        save_checkpoint(model, checkpoint, {"preflight": True})
        loaded = load_checkpoint(checkpoint, device)
        if model_hash(loaded) != changed:
            raise SystemExit("preflight failed: checkpoint round-trip changed model hash")

        # Corruption must fail closed rather than silently restarting at GENESIS.
        with ledger.open("a", encoding="utf-8") as f:
            f.write('{"broken":true}\n')
        try:
            validate_receipt_chain(ledger)
        except RuntimeError:
            pass
        else:
            raise SystemExit("preflight failed: corrupt ledger was accepted")

    model.load_state_dict(before_state)
    if model_hash(model) != original:
        raise SystemExit("preflight failed: state rollback did not restore parent")

    print("UDLM certified-learning preflight: PASS")
    print(f"parameters={model.parameter_count():,} initial_bpb={score:.4f}")


if __name__ == "__main__":
    main()
