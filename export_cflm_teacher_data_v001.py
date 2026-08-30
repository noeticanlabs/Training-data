#!/usr/bin/env python3
"""Build a fail-closed governed corpus bridge for CF-LM.

The bridge consumes only explicitly admitted Training Database examples and
writes split-separated, length-prefixed UTF-8 experience packs. Research DB
records are never read directly and pending examples are never promoted here.

Pack format (all integers unsigned big-endian):
    MAGIC = b"CFLM-TEACHER-DATA-V001\n"
    repeated records:
        u64 input_len
        input bytes
        u64 target_len
        target bytes

The sidecar records.jsonl preserves provenance and content hashes. CF-LM may
adapt only on the train pack; validation/test/holdout packs are evaluation-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

VERSION = "CFLM-TEACHER-DATA-V001"
MAGIC = b"CFLM-TEACHER-DATA-V001\n"
ORDER_DOMAIN = b"CFLM-TEACHER-DATA-V001-ORDER\x00"
RECORD_DOMAIN = b"CFLM-TEACHER-DATA-V001-RECORD\x00"
VALID_SPLITS = ("train", "validation", "test", "holdout")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


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


def _clean(text) -> str:
    return str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def render_input(example: dict) -> str:
    inp = example.get("input") or {}
    blocks: list[str] = []

    system = _clean(inp.get("system"))
    if system:
        blocks.append(f"System: {system}")

    context = _clean(inp.get("context"))
    if context:
        blocks.append(f"Context: {context}")

    for msg in inp.get("messages") or []:
        role = str(msg.get("role") or "unknown").strip().lower()
        text = _clean(msg.get("text"))
        if not text:
            continue
        label = "User" if role == "user" else "Assistant" if role == "assistant" else role.title()
        blocks.append(f"{label}: {text}")

    prompt = _clean(inp.get("prompt"))
    if prompt:
        blocks.append(f"User: {prompt}")

    if not blocks:
        structured = inp.get("structured")
        if structured is not None:
            blocks.append("Input: " + json.dumps(structured, sort_keys=True, ensure_ascii=False))

    if not blocks:
        return ""

    # Explicit answer boundary. It is visible during both training and evaluation;
    # only the target bytes are withheld at teacher-off evaluation time.
    return "\n\n".join(blocks).strip() + "\n\nAssistant: "


def render_target(example: dict) -> str:
    target = example.get("target") or {}
    text = _clean(target.get("text"))
    if text:
        return text
    messages = target.get("messages") or []
    parts = [_clean(m.get("text")) for m in messages if _clean(m.get("text"))]
    if parts:
        return "\n\n".join(parts)
    structured = target.get("structured")
    if structured is not None:
        return json.dumps(structured, sort_keys=True, ensure_ascii=False)
    return ""


def is_personal(example: dict) -> bool:
    domains = {str(x) for x in example.get("knowledge_domains") or []}
    metadata = example.get("metadata") or {}
    return "personal" in domains or metadata.get("scope") == "personal"


def encode_pack(records: list[dict]) -> bytes:
    out = bytearray(MAGIC)
    for row in records:
        input_bytes = row["input_text"].encode("utf-8")
        target_bytes = row["target_text"].encode("utf-8")
        out += len(input_bytes).to_bytes(8, "big")
        out += input_bytes
        out += len(target_bytes).to_bytes(8, "big")
        out += target_bytes
    return bytes(out)


def decode_pack(blob: bytes) -> list[tuple[bytes, bytes]]:
    if not blob.startswith(MAGIC):
        raise ValueError("bad CF-LM teacher-data magic")
    pos = len(MAGIC)
    out: list[tuple[bytes, bytes]] = []
    while pos < len(blob):
        if pos + 8 > len(blob):
            raise ValueError("truncated input length")
        input_len = int.from_bytes(blob[pos:pos + 8], "big")
        pos += 8
        if pos + input_len > len(blob):
            raise ValueError("truncated input payload")
        inp = blob[pos:pos + input_len]
        pos += input_len
        if pos + 8 > len(blob):
            raise ValueError("truncated target length")
        target_len = int.from_bytes(blob[pos:pos + 8], "big")
        pos += 8
        if pos + target_len > len(blob):
            raise ValueError("truncated target payload")
        target = blob[pos:pos + target_len]
        pos += target_len
        out.append((inp, target))
    return out


def build_export(candidates: Path, out_dir: Path) -> dict:
    rows_by_split: dict[str, list[dict]] = {s: [] for s in VALID_SPLITS}
    seen_ids: set[str] = set()
    split_groups: dict[str, str] = {}
    exclusion_counts = {
        "not_admitted": 0,
        "personal": 0,
        "unassigned_split": 0,
        "empty_input": 0,
        "empty_target": 0,
    }
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
            exclusion_counts["not_admitted"] += 1
            continue
        if is_personal(example):
            exclusion_counts["personal"] += 1
            continue

        split = str(example.get("split") or "unassigned")
        if split not in VALID_SPLITS:
            exclusion_counts["unassigned_split"] += 1
            continue

        split_group = str(example.get("split_group") or "")
        if not split_group:
            raise RuntimeError(f"admitted example {example_id} missing split_group")
        prior = split_groups.setdefault(split_group, split)
        if prior != split:
            raise RuntimeError(
                f"split leakage: split_group {split_group!r} occurs in both {prior!r} and {split!r}"
            )

        input_text = render_input(example)
        target_text = render_target(example)
        if not input_text:
            exclusion_counts["empty_input"] += 1
            continue
        if not target_text:
            exclusion_counts["empty_target"] += 1
            continue

        input_bytes = input_text.encode("utf-8")
        target_bytes = target_text.encode("utf-8")
        record_fingerprint = (
            RECORD_DOMAIN
            + example_id.encode("utf-8")
            + b"\x00"
            + input_bytes
            + b"\x00"
            + target_bytes
        )
        order_key = hashlib.sha256(ORDER_DOMAIN + example_id.encode("utf-8")).hexdigest()
        rows_by_split[split].append(
            {
                "training_example_id": example_id,
                "task": example.get("task"),
                "knowledge_domains": example.get("knowledge_domains") or [],
                "source_record_ids": example.get("source_record_ids") or [],
                "split": split,
                "split_group": split_group,
                "input_text": input_text,
                "target_text": target_text,
                "input_bytes": len(input_bytes),
                "target_bytes": len(target_bytes),
                "record_sha256": sha256_bytes(record_fingerprint),
                "order_key": order_key,
            }
        )

    if not any(rows_by_split.values()):
        raise RuntimeError("no explicitly admitted non-personal examples with assigned splits")

    out_dir.mkdir(parents=True, exist_ok=True)
    sidecar_path = out_dir / "records.jsonl"
    sidecar_rows: list[dict] = []
    split_manifest: dict[str, dict] = {}

    for split in VALID_SPLITS:
        rows = rows_by_split[split]
        rows.sort(key=lambda r: (r["order_key"], r["training_example_id"]))
        pack = encode_pack(rows)
        pack_path = out_dir / f"{split}.cflm"
        pack_path.write_bytes(pack)
        if decode_pack(pack) != [
            (r["input_text"].encode("utf-8"), r["target_text"].encode("utf-8")) for r in rows
        ]:
            raise RuntimeError(f"internal pack round-trip failed for split {split}")

        split_manifest[split] = {
            "records": len(rows),
            "bytes": len(pack),
            "sha256": sha256_bytes(pack),
        }
        for row in rows:
            sidecar_rows.append({k: v for k, v in row.items() if k not in {"input_text", "target_text", "order_key"}})

    with sidecar_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in sidecar_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    manifest = {
        "export_version": VERSION,
        "source_candidates": str(candidates),
        "source_rows": source_rows,
        "eligible_records": len(sidecar_rows),
        "automatic_admission": False,
        "admission_required": "admitted",
        "personal_excluded": True,
        "reasoning_consumed": False,
        "record_boundary_preserved": True,
        "pack_format": "magic + repeated u64be input_len/input + u64be target_len/target",
        "splits": split_manifest,
        "records_sha256": sha256_bytes(sidecar_path.read_bytes()),
        "exclusion_counts": exclusion_counts,
        "claim_boundary": (
            "This export establishes a governed byte-visible curriculum boundary only. "
            "It does not establish CF-LM language competence, semantics, or successful corpus training."
        ),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Export admitted Training DB examples to CF-LM teacher-data packs")
    parser.add_argument("--candidates", default="data/training_db/candidates.jsonl")
    parser.add_argument("--out-dir", default="data/exports/cflm_teacher_data_v001")
    args = parser.parse_args()
    manifest = build_export(Path(args.candidates), Path(args.out_dir))
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
