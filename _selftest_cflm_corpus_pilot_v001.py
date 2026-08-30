#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import tempfile

from build_cflm_corpus_pilot_v001 import build_pilot
from export_cflm_teacher_data_v001 import decode_pack


def example(
    eid: str,
    group: str,
    *,
    status: str = "admitted",
    scope: str = "work",
    split: str = "unassigned",
    task: str = "multi_turn_dialogue",
) -> dict:
    return {
        "schema_version": "0.02",
        "training_example_id": eid,
        "task": task,
        "knowledge_domains": ["personal"] if scope == "personal" else ["ai_ml"],
        "source_record_ids": [f"research:message:{eid}:u", f"research:message:{eid}:a"],
        "admission": {"status": status},
        "split": split,
        "split_group": group,
        "input": {
            "system": None,
            "prompt": f"prompt {eid}",
            "context": None,
            "messages": [],
            "structured": None,
        },
        "target": {"text": f"target {eid}", "messages": [], "structured": None},
        "metadata": {"scope": scope},
        "transform": {},
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source = root / "candidates.jsonl"
        rows: list[dict] = []
        for group_no in range(8):
            group = f"conversation:g{group_no}"
            rows.extend(example(f"g{group_no}-{index}", group) for index in range(2))
        rows += [
            example("pending", "conversation:pending", status="pending_review"),
            example("personal", "conversation:personal", scope="personal"),
            example("preassigned", "conversation:preassigned", split="test"),
            example("wrong-task", "conversation:wrong-task", task="theorem_proving"),
        ]
        write_jsonl(source, rows)

        out1 = root / "out1"
        out2 = root / "out2"
        first = build_pilot(
            source,
            out1,
            train_records=6,
            holdout_records=4,
            min_train=4,
            min_holdout=2,
        )
        second = build_pilot(
            source,
            out2,
            train_records=6,
            holdout_records=4,
            min_train=4,
            min_holdout=2,
        )

        assert first["selection_sha256"] == second["selection_sha256"]
        assert first["selected_train_records"] >= 6
        assert first["selected_holdout_records"] >= 4
        assert first["split_group_overlap"] == 0
        assert first["automatic_admission"] is False
        assert first["reasoning_consumed"] is False
        assert first["exclusion_counts"]["not_admitted"] == 1
        assert first["exclusion_counts"]["personal"] == 1
        assert first["exclusion_counts"]["preassigned_split"] == 1
        assert first["exclusion_counts"]["wrong_task"] == 1

        pilot_rows = [
            json.loads(line)
            for line in (out1 / "pilot_candidates.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        train_groups = {
            row["split_group"] for row in pilot_rows if row["split"] == "train"
        }
        holdout_groups = {
            row["split_group"] for row in pilot_rows if row["split"] == "holdout"
        }
        assert train_groups.isdisjoint(holdout_groups)

        train_pack = decode_pack((out1 / "packs" / "train.cflm").read_bytes())
        holdout_pack = decode_pack((out1 / "packs" / "holdout.cflm").read_bytes())
        assert len(train_pack) == first["selected_train_records"]
        assert len(holdout_pack) == first["selected_holdout_records"]

        # Source-row order must not alter the frozen pilot selection.
        reversed_source = root / "reversed.jsonl"
        write_jsonl(reversed_source, list(reversed(rows)))
        third = build_pilot(
            reversed_source,
            root / "out3",
            train_records=6,
            holdout_records=4,
            min_train=4,
            min_holdout=2,
        )
        assert third["selection_sha256"] == first["selection_sha256"]

    print("CF-LM Corpus Pilot v0.01 selector self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
