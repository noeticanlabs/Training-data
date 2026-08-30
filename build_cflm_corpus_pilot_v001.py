#!/usr/bin/env python3
"""Build the frozen CF-LM Corpus Pilot v0.01 selection from governed candidates.

This recipe never changes canonical admission state. It consumes only examples
already marked `admitted`, excludes personal material, preserves split groups,
and derives a deterministic train/holdout pilot view from otherwise unassigned
examples.

The resulting pilot candidates are a derived experiment view. They are then
passed through `export_cflm_teacher_data_v001.py` to produce the byte-visible
CF-LM packs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

from export_cflm_teacher_data_v001 import (
    build_export as build_teacher_export,
    render_input,
    render_target,
)

VERSION = "CFLM-CORPUS-PILOT-V001"
GROUP_DOMAIN = b"CFLM-CORPUS-PILOT-V001-GROUP\x00"
MANIFEST_DOMAIN = b"CFLM-CORPUS-PILOT-V001-MANIFEST\x00"


def load_jsonl(path: Path) -> Iterable[dict]:
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


def group_key(split_group: str) -> str:
    return hashlib.sha256(GROUP_DOMAIN + split_group.encode("utf-8")).hexdigest()


def _derived_row(example: dict, split: str) -> dict:
    row = json.loads(json.dumps(example))
    row["split"] = split
    transform = dict(row.get("transform") or {})
    transform["pilot_recipe"] = VERSION
    row["transform"] = transform
    metadata = dict(row.get("metadata") or {})
    metadata["pilot_view"] = VERSION
    metadata["pilot_split_derived"] = True
    row["metadata"] = metadata
    return row


def build_pilot(
    candidates: Path,
    out_dir: Path,
    *,
    train_records: int = 128,
    holdout_records: int = 32,
    min_train: int = 32,
    min_holdout: int = 8,
    task: str = "multi_turn_dialogue",
) -> dict:
    if min(train_records, holdout_records, min_train, min_holdout) < 1:
        raise ValueError("record counts must be positive")

    exclusions = {
        "not_admitted": 0,
        "personal": 0,
        "wrong_task": 0,
        "preassigned_split": 0,
        "missing_split_group": 0,
        "empty_input": 0,
        "empty_target": 0,
    }
    groups: dict[str, list[dict]] = {}
    seen_ids: set[str] = set()
    source_rows = 0

    for example in load_jsonl(candidates):
        source_rows += 1
        example_id = str(example.get("training_example_id") or "")
        if not example_id:
            raise RuntimeError("training example missing training_example_id")
        if example_id in seen_ids:
            raise RuntimeError(f"duplicate training_example_id: {example_id}")
        seen_ids.add(example_id)

        if (example.get("admission") or {}).get("status") != "admitted":
            exclusions["not_admitted"] += 1
            continue
        if is_personal(example):
            exclusions["personal"] += 1
            continue
        if str(example.get("task") or "") != task:
            exclusions["wrong_task"] += 1
            continue

        # v0.01 derives its own frozen split only from unassigned examples.
        # Existing train/validation/test/holdout assignments are never repurposed.
        if str(example.get("split") or "unassigned") != "unassigned":
            exclusions["preassigned_split"] += 1
            continue

        split_group = str(example.get("split_group") or "")
        if not split_group:
            exclusions["missing_split_group"] += 1
            continue
        if not render_input(example):
            exclusions["empty_input"] += 1
            continue
        if not render_target(example):
            exclusions["empty_target"] += 1
            continue
        groups.setdefault(split_group, []).append(example)

    if not groups:
        raise RuntimeError("no eligible explicitly admitted unassigned examples for pilot")

    ordered_groups = sorted(groups, key=lambda group: (group_key(group), group))

    # Freeze holdout first so later training results cannot influence its membership.
    holdout_groups: list[str] = []
    holdout_count = 0
    remaining_groups: list[str] = []
    for group in ordered_groups:
        if holdout_count < holdout_records:
            holdout_groups.append(group)
            holdout_count += len(groups[group])
        else:
            remaining_groups.append(group)

    train_groups: list[str] = []
    train_count = 0
    for group in remaining_groups:
        if train_count >= train_records:
            break
        train_groups.append(group)
        train_count += len(groups[group])

    if train_count < min_train or holdout_count < min_holdout:
        raise RuntimeError(
            "insufficient eligible governed data for frozen pilot: "
            f"train={train_count} (min {min_train}), "
            f"holdout={holdout_count} (min {min_holdout})"
        )

    selected: list[dict] = []
    for group in holdout_groups:
        selected.extend(_derived_row(example, "holdout") for example in groups[group])
    for group in train_groups:
        selected.extend(_derived_row(example, "train") for example in groups[group])

    selected.sort(
        key=lambda row: (
            0 if row["split"] == "train" else 1,
            group_key(str(row["split_group"])),
            str(row["training_example_id"]),
        )
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    pilot_candidates = out_dir / "pilot_candidates.jsonl"
    with pilot_candidates.open("w", encoding="utf-8", newline="\n") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    teacher_manifest = build_teacher_export(pilot_candidates, out_dir / "packs")

    train_ids = sorted(
        str(row["training_example_id"]) for row in selected if row["split"] == "train"
    )
    holdout_ids = sorted(
        str(row["training_example_id"]) for row in selected if row["split"] == "holdout"
    )
    selection_payload = json.dumps(
        {"train": train_ids, "holdout": holdout_ids},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    manifest = {
        "pilot_version": VERSION,
        "task": task,
        "source_candidates": str(candidates),
        "source_rows": source_rows,
        "eligible_groups": len(groups),
        "eligible_records": sum(len(value) for value in groups.values()),
        "requested_train_records": train_records,
        "requested_holdout_records": holdout_records,
        "selected_train_records": len(train_ids),
        "selected_holdout_records": len(holdout_ids),
        "selected_train_groups": len(train_groups),
        "selected_holdout_groups": len(holdout_groups),
        "selection_sha256": hashlib.sha256(
            MANIFEST_DOMAIN + selection_payload
        ).hexdigest(),
        "automatic_admission": False,
        "admission_required": "admitted",
        "source_split_required": "unassigned",
        "personal_excluded": True,
        "reasoning_consumed": False,
        "split_group_overlap": 0,
        "exclusion_counts": exclusions,
        "teacher_export": teacher_manifest,
        "claim_boundary": (
            "Deterministic governed pilot selection and byte-pack build only; "
            "no claim of language competence or holdout transfer."
        ),
    }
    (out_dir / "pilot_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build governed CF-LM Corpus Pilot v0.01")
    parser.add_argument("--candidates", default="data/training_db/candidates.jsonl")
    parser.add_argument("--out-dir", default="data/exports/cflm_corpus_pilot_v001")
    parser.add_argument("--train-records", type=int, default=128)
    parser.add_argument("--holdout-records", type=int, default=32)
    parser.add_argument("--min-train", type=int, default=32)
    parser.add_argument("--min-holdout", type=int, default=8)
    parser.add_argument("--task", default="multi_turn_dialogue")
    args = parser.parse_args()
    manifest = build_pilot(
        Path(args.candidates),
        Path(args.out_dir),
        train_records=args.train_records,
        holdout_records=args.holdout_records,
        min_train=args.min_train,
        min_holdout=args.min_holdout,
        task=args.task,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
