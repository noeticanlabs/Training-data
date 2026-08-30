#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import tempfile

from apply_cflm_pilot_review_v001 import apply_reviews
from prepare_cflm_pilot_review_v001 import build_review_queue


def example(eid: str, group: str, *, status: str = "pending_review") -> dict:
    return {
        "schema_version": "0.02",
        "training_example_id": eid,
        "task": "multi_turn_dialogue",
        "knowledge_domains": ["ai_ml"],
        "source_record_ids": [f"research:{eid}"],
        "admission": {"status": status},
        "split": "unassigned",
        "split_group": group,
        "input": {"prompt": f"prompt {eid}", "messages": [], "structured": None},
        "target": {"text": f"target {eid}", "messages": [], "structured": None},
        "metadata": {"scope": "work"},
        "transform": {},
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        candidates = root / "candidates.jsonl"
        rows = [
            example("a1", "g1"),
            example("a2", "g1"),
            example("b1", "g2"),
            example("b2", "g2"),
            example("already", "g3", status="admitted"),
        ]
        write_jsonl(candidates, rows)

        q1 = build_review_queue(candidates, root / "q1", target_records=3)
        q2 = build_review_queue(candidates, root / "q2", target_records=3)
        assert q1["queue_sha256"] == q2["queue_sha256"]
        assert q1["automatic_admission"] is False
        assert q1["selected_records"] >= 3

        queue_rows = [
            json.loads(line)
            for line in (root / "q1" / "review_queue.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert all(row["review"]["decision"] is None for row in queue_rows)

        decisions = root / "decisions.jsonl"
        decision_rows = []
        for row in queue_rows[:2]:
            row["review"] = {
                "decision": "admitted",
                "reviewer": "human-reviewer",
                "reviewed_at": "2026-08-30T00:00:00Z",
                "reason": "approved for bounded pilot",
            }
            decision_rows.append(row)
        write_jsonl(decisions, decision_rows)

        manifest = apply_reviews(candidates, decisions, root / "applied")
        assert manifest["counts"]["admitted"] == 2
        assert manifest["source_mutated"] is False
        assert manifest["automatic_admission"] is False

        reviewed = [
            json.loads(line)
            for line in (root / "applied" / "reviewed_candidates.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        admitted_ids = {
            row["training_example_id"]
            for row in reviewed
            if (row.get("admission") or {}).get("status") == "admitted"
        }
        assert "already" in admitted_ids
        assert {row["training_example_id"] for row in decision_rows}.issubset(admitted_ids)

    print("CF-LM pilot review workflow self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
