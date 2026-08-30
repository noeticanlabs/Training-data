#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import tempfile

from export_cflm_teacher_data_v001 import build_export, decode_pack, MAGIC


def example(eid, split, group, status="admitted", scope="work", domain="ai_ml", prompt="hello", target="world"):
    return {
        "schema_version": "0.02",
        "training_example_id": eid,
        "task": "multi_turn_dialogue",
        "knowledge_domains": [domain],
        "source_record_ids": [f"research:message:{eid}:u", f"research:message:{eid}:a"],
        "admission": {"status": status},
        "split": split,
        "split_group": group,
        "input": {"system": None, "prompt": prompt, "context": None, "messages": [], "structured": None},
        "target": {"text": target, "messages": [], "structured": None},
        "metadata": {"scope": scope},
    }


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        candidates = root / "candidates.jsonl"
        out = root / "out"
        rows = [
            example("train-a", "train", "g-train", prompt="alpha", target="beta"),
            example("hold-a", "holdout", "g-hold", prompt="gamma", target="delta"),
            example("pending", "train", "g-pending", status="pending_review"),
            example("personal", "train", "g-personal", scope="personal", domain="personal"),
            example("unassigned", "unassigned", "g-unassigned"),
        ]
        write_jsonl(candidates, rows)
        manifest = build_export(candidates, out)

        assert manifest["eligible_records"] == 2
        assert manifest["splits"]["train"]["records"] == 1
        assert manifest["splits"]["holdout"]["records"] == 1
        assert manifest["splits"]["validation"]["records"] == 0
        assert manifest["splits"]["test"]["records"] == 0
        assert manifest["exclusion_counts"]["not_admitted"] == 1
        assert manifest["exclusion_counts"]["personal"] == 1
        assert manifest["exclusion_counts"]["unassigned_split"] == 1
        assert manifest["automatic_admission"] is False
        assert manifest["reasoning_consumed"] is False

        train_blob = (out / "train.cflm").read_bytes()
        assert train_blob.startswith(MAGIC)
        train_records = decode_pack(train_blob)
        assert len(train_records) == 1
        inp, target = train_records[0]
        assert inp.decode("utf-8") == "User: alpha\n\nAssistant: "
        assert target.decode("utf-8") == "beta"

        # Fail closed on split-group leakage.
        leak = root / "leak.jsonl"
        write_jsonl(
            leak,
            [
                example("x1", "train", "same-group"),
                example("x2", "holdout", "same-group"),
            ],
        )
        try:
            build_export(leak, root / "leak-out")
        except RuntimeError as exc:
            assert "split leakage" in str(exc)
        else:
            raise AssertionError("split-group leakage was not rejected")

        # Fail closed when no admitted examples survive.
        none = root / "none.jsonl"
        write_jsonl(none, [example("p", "train", "gp", status="pending_review")])
        try:
            build_export(none, root / "none-out")
        except RuntimeError as exc:
            assert "no explicitly admitted" in str(exc)
        else:
            raise AssertionError("empty governed export was not rejected")

    print("CF-LM teacher-data bridge v0.01 self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
