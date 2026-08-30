#!/usr/bin/env python3
"""Apply explicit CF-LM pilot review decisions to a derived candidate view.

The canonical candidate source is never modified. Only candidates currently in
`pending_review` may be changed, and every decision requires a named reviewer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

VERSION = "CFLM-PILOT-REVIEW-V001"
VALID_DECISIONS = {"admitted", "rejected", "quarantined"}


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


def load_decisions(path: Path) -> dict[str, dict]:
    decisions: dict[str, dict] = {}
    for row in load_jsonl(path):
        example_id = str(row.get("training_example_id") or "")
        review = row.get("review") or row
        decision = review.get("decision")
        reviewer = str(review.get("reviewer") or "").strip()
        if not example_id or decision not in VALID_DECISIONS or not reviewer:
            raise RuntimeError(f"invalid review decision for {example_id or '<missing>'}")
        if decision in {"rejected", "quarantined"} and not str(
            review.get("reason") or ""
        ).strip():
            raise RuntimeError(f"{decision} decision requires reason: {example_id}")
        if example_id in decisions:
            raise RuntimeError(f"duplicate review decision: {example_id}")
        decisions[example_id] = review
    if not decisions:
        raise RuntimeError("review decision file contains no decisions")
    return decisions


def apply_reviews(candidates: Path, decisions_path: Path, out_dir: Path) -> dict:
    decisions = load_decisions(decisions_path)
    counts = {decision: 0 for decision in sorted(VALID_DECISIONS)}
    untouched = 0
    seen_decisions: set[str] = set()

    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "reviewed_candidates.jsonl"
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for example in load_jsonl(candidates):
            example_id = str(example.get("training_example_id") or "")
            review = decisions.get(example_id)
            if review is None:
                untouched += 1
            else:
                seen_decisions.add(example_id)
                if (example.get("admission") or {}).get("status") != "pending_review":
                    raise RuntimeError(
                        f"review decision targets non-pending example: {example_id}"
                    )
                admission = dict(example.get("admission") or {})
                admission.update(
                    {
                        "status": review["decision"],
                        "reviewer": review["reviewer"],
                        "reviewed_at": review.get("reviewed_at"),
                        "reason": review.get("reason"),
                        "recipe": "cflm_pilot_review_v001",
                    }
                )
                example["admission"] = admission
                transform = dict(example.get("transform") or {})
                transform["review_recipe"] = VERSION
                example["transform"] = transform
                counts[review["decision"]] += 1
            handle.write(json.dumps(example, ensure_ascii=False, sort_keys=True) + "\n")

    missing = set(decisions) - seen_decisions
    if missing:
        raise RuntimeError(
            "review decisions not found in candidate source: " + ", ".join(sorted(missing))
        )

    manifest = {
        "version": VERSION,
        "decisions": len(decisions),
        "counts": counts,
        "untouched": untouched,
        "reviewed_candidates_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "source_mutated": False,
        "automatic_admission": False,
    }
    (out_dir / "review_apply_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply explicit CF-LM pilot review decisions")
    parser.add_argument("--candidates", default="data/training_db/candidates.jsonl")
    parser.add_argument("--decisions", required=True)
    parser.add_argument(
        "--out-dir", default="data/reviews/cflm_corpus_pilot_v001/applied"
    )
    args = parser.parse_args()
    print(
        json.dumps(
            apply_reviews(
                Path(args.candidates), Path(args.decisions), Path(args.out_dir)
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
