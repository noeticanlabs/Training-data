#!/usr/bin/env python3
"""Prepare a deterministic human-review queue for CF-LM Corpus Pilot v0.01.

This script never changes admission status. It only extracts a whole-group,
non-personal, pending-review queue with visible input/target material and source
lineage so a reviewer can make explicit admission decisions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

GROUP_DOMAIN = b"CFLM-PILOT-REVIEW-V001-GROUP\x00"
ID_DOMAIN = b"CFLM-PILOT-REVIEW-V001-ID\x00"
VERSION = "CFLM-PILOT-REVIEW-V001"


def load_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on {path} line {line_no}: {exc}") from exc


def is_personal(example: dict) -> bool:
    domains = {str(x) for x in example.get("knowledge_domains") or []}
    metadata = example.get("metadata") or {}
    return "personal" in domains or metadata.get("scope") == "personal"


def stable_key(value: str, domain: bytes) -> str:
    return hashlib.sha256(domain + value.encode("utf-8")).hexdigest()


def has_visible_io(example: dict) -> bool:
    inp = example.get("input") or {}
    target = example.get("target") or {}
    has_input = bool(
        str(inp.get("prompt") or "").strip()
        or inp.get("messages")
        or inp.get("structured") is not None
    )
    has_target = bool(
        str(target.get("text") or "").strip()
        or target.get("messages")
        or target.get("structured") is not None
    )
    return has_input and has_target


def build_review_queue(
    candidates: Path,
    out_dir: Path,
    *,
    target_records: int = 200,
    task: str = "multi_turn_dialogue",
) -> dict:
    if target_records < 1:
        raise ValueError("target_records must be positive")

    groups: dict[str, list[dict]] = {}
    exclusions = {
        "not_pending": 0,
        "personal": 0,
        "wrong_task": 0,
        "preassigned_split": 0,
        "missing_group": 0,
        "empty": 0,
    }

    for example in load_jsonl(candidates):
        if (example.get("admission") or {}).get("status") != "pending_review":
            exclusions["not_pending"] += 1
            continue
        if is_personal(example):
            exclusions["personal"] += 1
            continue
        if str(example.get("task") or "") != task:
            exclusions["wrong_task"] += 1
            continue
        if str(example.get("split") or "unassigned") != "unassigned":
            exclusions["preassigned_split"] += 1
            continue
        split_group = str(example.get("split_group") or "")
        if not split_group:
            exclusions["missing_group"] += 1
            continue
        if not has_visible_io(example):
            exclusions["empty"] += 1
            continue
        groups.setdefault(split_group, []).append(example)

    ordered_groups = sorted(
        groups, key=lambda group: (stable_key(group, GROUP_DOMAIN), group)
    )
    selected: list[dict] = []
    selected_groups: list[str] = []
    for group in ordered_groups:
        if len(selected) >= target_records:
            break
        selected_groups.append(group)
        selected.extend(
            sorted(
                groups[group],
                key=lambda example: (
                    stable_key(str(example["training_example_id"]), ID_DOMAIN),
                    str(example["training_example_id"]),
                ),
            )
        )

    if not selected:
        raise RuntimeError("no eligible pending-review examples for CF-LM pilot review")

    out_dir.mkdir(parents=True, exist_ok=True)
    queue_path = out_dir / "review_queue.jsonl"
    with queue_path.open("w", encoding="utf-8", newline="\n") as handle:
        for example in selected:
            row = {
                "training_example_id": example["training_example_id"],
                "split_group": example["split_group"],
                "task": example["task"],
                "knowledge_domains": example.get("knowledge_domains") or [],
                "source_record_ids": example.get("source_record_ids") or [],
                "input": example.get("input") or {},
                "target": example.get("target") or {},
                "review": {
                    "decision": None,
                    "reviewer": None,
                    "reviewed_at": None,
                    "reason": None,
                },
            }
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    manifest = {
        "version": VERSION,
        "task": task,
        "target_records": target_records,
        "selected_records": len(selected),
        "selected_groups": len(selected_groups),
        "queue_sha256": hashlib.sha256(queue_path.read_bytes()).hexdigest(),
        "exclusions": exclusions,
        "automatic_admission": False,
        "claim_boundary": (
            "Review queue only. No candidate is admitted, rejected, or quarantined "
            "until an explicit review decision is applied."
        ),
    }
    (out_dir / "review_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare CF-LM pilot admission review queue")
    parser.add_argument("--candidates", default="data/training_db/candidates.jsonl")
    parser.add_argument("--out-dir", default="data/reviews/cflm_corpus_pilot_v001")
    parser.add_argument("--target-records", type=int, default=200)
    parser.add_argument("--task", default="multi_turn_dialogue")
    args = parser.parse_args()
    print(
        json.dumps(
            build_review_queue(
                Path(args.candidates),
                Path(args.out_dir),
                target_records=args.target_records,
                task=args.task,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
