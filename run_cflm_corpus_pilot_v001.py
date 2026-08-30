#!/usr/bin/env python3
"""Run CF-LM Corpus Pilot v0.01 and emit a reproducibility receipt.

This runner binds the exact governed pilot selection and byte packs to the exact
CF-LM checkout used for evaluation. It verifies pack hashes before invoking the
CF-LM executable and never passes holdout targets into training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

VERSION = "CFLM-CORPUS-PILOT-RUN-V001"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_text(command: list[str], cwd: Path | None = None) -> str:
    completed = subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout.strip()


def verify_pack_hash(pack_path: Path, split_manifest: dict, split: str) -> str:
    expected = str((split_manifest.get(split) or {}).get("sha256") or "")
    if not expected:
        raise RuntimeError(f"missing expected {split} pack hash in manifest")
    observed = sha256_file(pack_path)
    if observed != expected:
        raise RuntimeError(
            f"{split} pack hash mismatch: expected={expected} observed={observed}"
        )
    return observed


def assert_group_separation(pilot_candidates: Path) -> tuple[int, int, int]:
    train_groups: set[str] = set()
    holdout_groups: set[str] = set()
    train_records = 0
    holdout_records = 0
    with pilot_candidates.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            split = str(row.get("split") or "")
            group = str(row.get("split_group") or "")
            if not group:
                raise RuntimeError(f"pilot candidate line {line_no} missing split_group")
            if split == "train":
                train_records += 1
                train_groups.add(group)
            elif split == "holdout":
                holdout_records += 1
                holdout_groups.add(group)
    overlap = train_groups & holdout_groups
    if overlap:
        raise RuntimeError(
            "train/holdout split_group overlap: " + ", ".join(sorted(overlap))
        )
    return train_records, holdout_records, len(overlap)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run and receipt CF-LM Corpus Pilot v0.01")
    parser.add_argument("--pilot-dir", default="data/exports/cflm_corpus_pilot_v001")
    parser.add_argument("--cflm-repo", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--receipt", default=None)
    args = parser.parse_args()

    if args.epochs != 1:
        raise RuntimeError(
            "v0.01 frozen run requires exactly 1 epoch; use a new version for epoch sweeps"
        )

    pilot_dir = Path(args.pilot_dir).resolve()
    cflm_repo = Path(args.cflm_repo).resolve()
    pilot_manifest_path = pilot_dir / "pilot_manifest.json"
    teacher_manifest_path = pilot_dir / "packs" / "manifest.json"
    pilot_candidates_path = pilot_dir / "pilot_candidates.jsonl"
    train_pack = pilot_dir / "packs" / "train.cflm"
    holdout_pack = pilot_dir / "packs" / "holdout.cflm"

    for path in [
        pilot_manifest_path,
        teacher_manifest_path,
        pilot_candidates_path,
        train_pack,
        holdout_pack,
        cflm_repo / "Cargo.toml",
    ]:
        if not path.exists():
            raise RuntimeError(f"required pilot input missing: {path}")

    pilot_manifest = json.loads(pilot_manifest_path.read_text(encoding="utf-8"))
    teacher_manifest = json.loads(teacher_manifest_path.read_text(encoding="utf-8"))
    if pilot_manifest.get("pilot_version") != "CFLM-CORPUS-PILOT-V001":
        raise RuntimeError("unexpected pilot manifest version")
    if teacher_manifest.get("export_version") != "CFLM-TEACHER-DATA-V001":
        raise RuntimeError("unexpected teacher-data export version")
    if pilot_manifest.get("automatic_admission") is not False:
        raise RuntimeError("pilot manifest does not prove automatic_admission=false")
    if teacher_manifest.get("reasoning_consumed") is not False:
        raise RuntimeError("teacher-data manifest does not prove reasoning_consumed=false")

    train_sha = verify_pack_hash(train_pack, teacher_manifest["splits"], "train")
    holdout_sha = verify_pack_hash(holdout_pack, teacher_manifest["splits"], "holdout")
    train_records, holdout_records, group_overlap = assert_group_separation(
        pilot_candidates_path
    )
    if train_records < 1 or holdout_records < 1:
        raise RuntimeError("pilot must contain non-empty train and holdout sets")

    cflm_commit = run_text(["git", "rev-parse", "HEAD"], cflm_repo)
    cflm_branch = run_text(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], cflm_repo
    )
    if cflm_branch != "agent/cf-lm-corpus-pilot-v001":
        raise RuntimeError(
            "CF-LM checkout must be frozen branch agent/cf-lm-corpus-pilot-v001; "
            f"observed {cflm_branch}"
        )
    data_commit = run_text(["git", "rev-parse", "HEAD"], Path.cwd())

    command = [
        "cargo",
        "run",
        "--quiet",
        "--bin",
        "corpus_pilot_v001",
        "--",
        str(train_pack),
        str(holdout_pack),
        "1",
    ]
    metrics_text = run_text(command, cflm_repo)
    try:
        metrics = json.loads(metrics_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "CF-LM pilot runner did not emit valid JSON: " + metrics_text
        ) from exc

    receipt = {
        "receipt_version": VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pilot_version": pilot_manifest["pilot_version"],
        "selection_sha256": pilot_manifest.get("selection_sha256"),
        "pilot_manifest_sha256": sha256_file(pilot_manifest_path),
        "teacher_manifest_sha256": sha256_file(teacher_manifest_path),
        "pilot_candidates_sha256": sha256_file(pilot_candidates_path),
        "train_pack_sha256": train_sha,
        "holdout_pack_sha256": holdout_sha,
        "train_records": train_records,
        "holdout_records": holdout_records,
        "split_group_overlap": group_overlap,
        "epochs": 1,
        "training_data_commit": data_commit,
        "cflm_branch": cflm_branch,
        "cflm_commit": cflm_commit,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "rustc_version": run_text(["rustc", "--version"]),
        "cargo_version": run_text(["cargo", "--version"]),
        "command": command,
        "metrics": metrics,
        "claim_boundary": (
            "Measured CF-LM Corpus Pilot v0.01 receipt only. Metric differences "
            "must be interpreted against frozen shuffled-target, rotated-prompt, "
            "boundary-only, and untrained controls; no semantic claim is implied."
        ),
    }

    receipt_path = (
        Path(args.receipt).resolve()
        if args.receipt
        else pilot_dir / "pilot_run_receipt_v001.json"
    )
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"CF-LM Corpus Pilot v0.01 run FAILED: {exc}", file=sys.stderr)
        raise
