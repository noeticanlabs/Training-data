from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F

from .model import UDLM, UDLMConfig

SUPPORTED_EXTENSIONS = {".txt", ".md", ".json", ".jsonl", ".py", ".ts", ".rs"}


def walk_curriculum(root: Path) -> list[Path]:
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS]
    files.sort(key=lambda p: p.relative_to(root).as_posix().lower())
    if not files:
        raise ValueError(f"no curriculum files found under {root}")
    return files


def sample_batch(data: bytes, batch_size: int, context: int, device: torch.device):
    if len(data) < 3:
        raise ValueError("lesson too short")
    c = min(context, len(data) - 1)
    max_start = len(data) - c - 1
    starts = [random.randrange(0, max_start + 1) if max_start > 0 else 0 for _ in range(batch_size)]
    x = torch.tensor([list(data[s:s+c]) for s in starts], dtype=torch.long, device=device)
    y = torch.tensor([list(data[s+1:s+c+1]) for s in starts], dtype=torch.long, device=device)
    return x, y


@torch.no_grad()
def bpb(model: UDLM, data: bytes, context: int, device: torch.device, batches: int = 2) -> float:
    if len(data) < 3:
        return float("nan")
    model.eval()
    nll = 0.0
    count = 0
    for _ in range(batches):
        x, y = sample_batch(data, 2, context, device)
        logits, _ = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum")
        nll += float(loss.item())
        count += y.numel()
    return nll / max(count, 1) / math.log(2.0)


def train_steps(model: UDLM, opt, lesson: bytes, replay: bytes, args, device: torch.device) -> float:
    model.train()
    last = float("nan")
    for _ in range(args.steps_per_file):
        source = replay if replay and random.random() < args.replay_ratio else lesson
        x, y = sample_batch(source, args.batch_size, args.context, device)
        logits, _ = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        last = float(loss.item() / math.log(2.0))
    return last


def append_replay(replay: bytes, lesson: bytes, max_bytes: int) -> bytes:
    merged = replay + b"\n" + lesson
    return merged[-max_bytes:] if len(merged) > max_bytes else merged


def last_hash(path: Path) -> str:
    if not path.exists():
        return "GENESIS"
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return "GENESIS"
    try:
        return json.loads(lines[-1]).get("receipt_hash", "GENESIS")
    except Exception:
        return "GENESIS"


def write_receipt(path: Path, record: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    prev = last_hash(path)
    body = dict(record)
    body["previous_receipt_hash"] = prev
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    rh = hashlib.sha256((prev + "|" + canonical).encode("utf-8")).hexdigest()
    body["receipt_hash"] = rh
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(body, sort_keys=True, ensure_ascii=False) + "\n")
    return rh


def save_checkpoint(model: UDLM, path: Path, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": asdict(model.cfg), "state_dict": model.state_dict(), "metadata": metadata}, path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Copilot-style UDLM curriculum trainer")
    ap.add_argument("curriculum", type=Path)
    ap.add_argument("--checkpoint", type=Path, default=Path("artifacts/udlm_curriculum_prototype.pt"))
    ap.add_argument("--receipts", type=Path, default=Path("artifacts/udlm_curriculum_receipts.jsonl"))
    ap.add_argument("--steps-per-file", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--context", type=int, default=96)
    ap.add_argument("--lr", type=float, default=0.002)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--replay-ratio", type=float, default=0.25)
    ap.add_argument("--replay-bytes", type=int, default=262144)
    ap.add_argument("--max-replay-regression", type=float, default=0.10)
    ap.add_argument("--lesson-tolerance", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--prompt", default="The system ")
    ap.add_argument("--max-new-bytes", type=int, default=128)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model = UDLM(UDLMConfig()).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    files = walk_curriculum(args.curriculum)
    replay = b""
    accepted = 0
    rolled_back = 0

    print(f"UDLM curriculum trainer | device={device} | params={model.parameter_count():,} | lessons={len(files)}")
    print("workflow: observe -> update proposal -> retention check -> admit/rollback -> receipt")

    for i, path in enumerate(files, 1):
        lesson = path.read_bytes()
        if len(lesson) < 3:
            continue
        rel = path.relative_to(args.curriculum).as_posix()
        before = bpb(model, lesson, args.context, device)
        replay_before = bpb(model, replay, args.context, device) if len(replay) >= 3 else None
        model_snapshot = copy.deepcopy(model.state_dict())
        opt_snapshot = copy.deepcopy(opt.state_dict())
        train_last = train_steps(model, opt, lesson, replay, args, device)
        after = bpb(model, lesson, args.context, device)
        replay_after = bpb(model, replay, args.context, device) if len(replay) >= 3 else None

        lesson_ok = after <= before + args.lesson_tolerance
        retention_ok = replay_before is None or replay_after is None or replay_after <= replay_before + args.max_replay_regression
        if lesson_ok and retention_ok:
            accepted += 1
            replay = append_replay(replay, lesson, args.replay_bytes)
            decision = "accepted"
        else:
            rolled_back += 1
            model.load_state_dict(model_snapshot)
            opt.load_state_dict(opt_snapshot)
            decision = "rolled_back"

        rh = write_receipt(args.receipts, {
            "kind": "udlm-training-receipt-v1",
            "timestamp": time.time(),
            "lesson_index": i,
            "lesson_path": rel,
            "lesson_sha256": hashlib.sha256(lesson).hexdigest(),
            "lesson_bytes": len(lesson),
            "lesson_bpb_before": before,
            "lesson_bpb_after": after,
            "replay_bpb_before": replay_before,
            "replay_bpb_after": replay_after,
            "train_bpb_last": train_last,
            "decision": decision,
            "accepted_total": accepted,
            "rolled_back_total": rolled_back,
        })
        print(f"[{i:03d}/{len(files):03d}] {decision:11s} {rel} | lesson {before:.3f}->{after:.3f} BPB")

    save_checkpoint(model, args.checkpoint, {
        "trainer": "copilot-style-curriculum-v1-local-runtime",
        "accepted": accepted,
        "rolled_back": rolled_back,
        "last_receipt": last_hash(args.receipts),
    })
    generated = model.generate(args.prompt.encode("utf-8"), max_new_bytes=args.max_new_bytes, greedy=True)
    print(f"saved: {args.checkpoint}")
    print(f"accepted={accepted} rolled_back={rolled_back}")
    print("UDLM> " + generated.decode("utf-8", errors="replace"))


if __name__ == "__main__":
    main()
