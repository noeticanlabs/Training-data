#!/usr/bin/env python3
"""Export a benchmark-clean conversational/technical corpus for CohBit-LM LANG-002C.

The exporter is intentionally fail-closed:
- only work-scope conversations are eligible by default;
- any conversation updated on/after the frozen benchmark date is excluded;
- raw reasoning is never exported;
- held-out benchmark entities are redacted;
- conversations containing long frozen benchmark strings are excluded;
- obvious email/phone/IP/SSN-like strings are redacted;
- deterministic hash ordering produces nested prefix curricula;
- the final UTF-8 stream and every admitted source record are content addressed.

The generated files are local experiment artifacts and should remain gitignored.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Iterable

EXPORT_VERSION = "CBLM-LANG-002C-EXPORT-V1"
ORDER_DOMAIN = b"CBLM-LANG-002C-ORDER-V1\x00"
CORPUS_DOMAIN = b"CBLM-LANG-002C-CORPUS-V1\x00"
RECORD_DOMAIN = b"CBLM-LANG-002C-RECORD-V1\x00"
BENCHMARK_HASH_DOMAIN = b"CBLM-LANG-BENCH-001-V1\x00"
EXPECTED_BENCHMARK_SHA256 = "d73212f54d04f6551b61a66132d72b7f2a222019a080235c00e28b25f1140ba2"
DEFAULT_CUTOFF = "2026-08-23T00:00:00+00:00"  # benchmark spec date; exclusive

HELDOUT_ENTITIES = (
    "Nauru", "Yaren", "Tuvalu", "Funafuti", "San Marino", "Monaco",
    "Liechtenstein", "Vaduz", "Andorra la Vella", "Bhutan", "Thimphu",
    "Lesotho", "Maseru", "Iceland", "Reykjavik", "Maldives",
)

EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
PHONE_RE = re.compile(r"(?<!\w)(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}(?!\w)")
IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
ENTITY_PATTERNS = tuple(
    (entity, re.compile(r"(?<!\w)" + re.escape(entity) + r"(?!\w)", re.I))
    for entity in sorted(HELDOUT_ENTITIES, key=len, reverse=True)
)


def _json_sha(prefix: bytes, value) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(prefix + raw).hexdigest()


def _parse_time(value) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _normalize_for_match(text: str) -> str:
    return " ".join(text.casefold().split())


def _iter_strings(value) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_strings(item)


def load_benchmark_strings(path: Path) -> tuple[str, tuple[str, ...]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    stored = data.get("benchmark_sha256")
    payload = dict(data)
    payload.pop("benchmark_sha256", None)
    computed = _json_sha(BENCHMARK_HASH_DOMAIN, payload)
    if stored != computed:
        raise RuntimeError(f"benchmark integrity failure: embedded={stored} computed={computed}")
    if computed != EXPECTED_BENCHMARK_SHA256:
        raise RuntimeError(
            "unexpected frozen benchmark identity: "
            f"expected={EXPECTED_BENCHMARK_SHA256} observed={computed}"
        )
    strings = {
        _normalize_for_match(s)
        for item in data.get("items", [])
        for s in _iter_strings(item)
        if len(_normalize_for_match(s)) >= 30
    }
    return computed, tuple(sorted(strings, key=lambda s: (-len(s), s)))


def redact_text(text: str) -> tuple[str, dict[str, int]]:
    counts = {"email": 0, "phone": 0, "ipv4": 0, "ssn": 0, "heldout_entity": 0}

    def sub_count(pattern: re.Pattern, replacement: str, key: str, value: str) -> str:
        def repl(_match):
            counts[key] += 1
            return replacement
        return pattern.sub(repl, value)

    text = sub_count(EMAIL_RE, "[EMAIL]", "email", text)
    text = sub_count(PHONE_RE, "[PHONE]", "phone", text)
    text = sub_count(IPV4_RE, "[IP]", "ipv4", text)
    text = sub_count(SSN_RE, "[SSN]", "ssn", text)
    for _entity, pattern in ENTITY_PATTERNS:
        text = sub_count(pattern, "[ENTITY]", "heldout_entity", text)
    return text, counts


def render_conversation(rec: dict) -> tuple[str, int]:
    blocks: list[str] = []
    message_count = 0
    for message in rec.get("messages", []):
        role = message.get("role")
        if role not in {"user", "assistant"}:
            continue
        text = str(message.get("text") or "").strip()
        if not text:
            continue
        text = text.replace("[image]", " ").strip()
        if not text:
            continue
        label = "User" if role == "user" else "Assistant"
        blocks.append(f"{label}: {text}")
        message_count += 1
    return "\n\n".join(blocks).strip(), message_count


def _contains_benchmark_string(text: str, frozen_strings: tuple[str, ...]) -> bool:
    normalized = _normalize_for_match(text)
    return any(s in normalized for s in frozen_strings)


def _remaining_entity_hits(text: str) -> list[str]:
    return [entity for entity, pattern in ENTITY_PATTERNS if pattern.search(text)]


def build_export(transcripts: Path, benchmark: Path, out_dir: Path, cutoff_text: str) -> dict:
    cutoff = _parse_time(cutoff_text)
    if cutoff is None:
        raise ValueError("cutoff must resolve to an absolute timestamp")
    benchmark_sha, benchmark_strings = load_benchmark_strings(benchmark)

    candidates: list[dict] = []
    exclusion_counts: dict[str, int] = {
        "scope": 0,
        "personal_tag": 0,
        "do_not_remember": 0,
        "post_cutoff": 0,
        "too_few_messages": 0,
        "benchmark_overlap": 0,
    }
    redaction_counts = {"email": 0, "phone": 0, "ipv4": 0, "ssn": 0, "heldout_entity": 0}
    source_rows = 0

    with transcripts.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rec = json.loads(line)
            source_rows += 1
            if rec.get("scope") != "work":
                exclusion_counts["scope"] += 1
                continue
            if "personal" in set(rec.get("tags", [])):
                exclusion_counts["personal_tag"] += 1
                continue
            if bool(rec.get("is_do_not_remember")):
                exclusion_counts["do_not_remember"] += 1
                continue
            updated = _parse_time(rec.get("update_time")) or _parse_time(rec.get("create_time"))
            if updated is None or updated >= cutoff:
                exclusion_counts["post_cutoff"] += 1
                continue

            text, message_count = render_conversation(rec)
            if message_count < 2 or not text:
                exclusion_counts["too_few_messages"] += 1
                continue
            text, redactions = redact_text(text)
            for key, count in redactions.items():
                redaction_counts[key] += count
            if _contains_benchmark_string(text, benchmark_strings):
                exclusion_counts["benchmark_overlap"] += 1
                continue
            remaining = _remaining_entity_hits(text)
            if remaining:
                raise RuntimeError(f"held-out entity survived redaction in {rec.get('conversation_id')}: {remaining}")

            cid = str(rec.get("conversation_id") or "")
            if not cid:
                raise RuntimeError("eligible transcript record missing conversation_id")
            record_sha = hashlib.sha256(RECORD_DOMAIN + text.encode("utf-8")).hexdigest()
            order_key = hashlib.sha256(ORDER_DOMAIN + cid.encode("utf-8")).hexdigest()
            candidates.append(
                {
                    "conversation_id": cid,
                    "title": rec.get("title"),
                    "create_time": rec.get("create_time"),
                    "update_time": rec.get("update_time"),
                    "tags": rec.get("tags", []),
                    "message_count": message_count,
                    "record_sha256": record_sha,
                    "order_key": order_key,
                    "text": text,
                }
            )

    candidates.sort(key=lambda r: (r["order_key"], r["conversation_id"]))
    if not candidates:
        raise RuntimeError("no eligible conversations after LANG-002C export filters")

    out_dir.mkdir(parents=True, exist_ok=True)
    records_path = out_dir / "records.jsonl"
    corpus_path = out_dir / "corpus.txt"
    manifest_path = out_dir / "manifest.json"

    with records_path.open("w", encoding="utf-8", newline="\n") as out:
        for row in candidates:
            public = {k: v for k, v in row.items() if k != "text"}
            public["text"] = row["text"]
            out.write(json.dumps(public, ensure_ascii=False, sort_keys=True) + "\n")

    corpus_text = "\n\n".join(row["text"] for row in candidates).strip() + "\n"
    corpus_bytes = corpus_text.encode("utf-8")
    corpus_path.write_bytes(corpus_bytes)
    corpus_sha = hashlib.sha256(CORPUS_DOMAIN + corpus_bytes).hexdigest()

    if _contains_benchmark_string(corpus_text, benchmark_strings):
        raise RuntimeError("final corpus contains a frozen benchmark string")
    remaining = _remaining_entity_hits(corpus_text)
    if remaining:
        raise RuntimeError(f"final corpus contains held-out entities: {remaining}")

    manifest = {
        "export_version": EXPORT_VERSION,
        "status": "EXPERIMENTAL_UNREVIEWED_TRANSCRIPT_EXPORT",
        "source_transcripts": str(transcripts),
        "benchmark_path": str(benchmark),
        "benchmark_sha256": benchmark_sha,
        "cutoff_exclusive": cutoff.isoformat(),
        "ordering": "sha256(CBLM-LANG-002C-ORDER-V1\\0 + conversation_id)",
        "format": "UTF-8 dialogue text; User/Assistant labels; reasoning excluded",
        "source_rows": source_rows,
        "admitted_conversations": len(candidates),
        "admitted_messages": sum(r["message_count"] for r in candidates),
        "corpus_bytes": len(corpus_bytes),
        "corpus_sha256": corpus_sha,
        "records_sha256": hashlib.sha256(records_path.read_bytes()).hexdigest(),
        "exclusion_counts": exclusion_counts,
        "redaction_counts": redaction_counts,
        "separation": {
            "benchmark_long_string_matches": 0,
            "heldout_entity_matches": 0,
            "post_benchmark_conversations": 0,
        },
        "claim_boundary": (
            "Private local experimental corpus export. This does not mark source records as "
            "canonically admitted Training Database examples and does not authorize public release."
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the fail-closed CohBit-LM LANG-002C archive corpus")
    ap.add_argument(
        "--transcripts",
        default="data/transcripts/transcripts_work_only.jsonl",
        help="normalized transcript JSONL (work-only recommended)",
    )
    ap.add_argument(
        "--benchmark",
        default="../Cohbit-llm/cohbit-lm/test_vectors/lang_bench/cblm-lang-bench-001-v1.0.json",
        help="path to the sealed LANG benchmark JSON",
    )
    ap.add_argument("--out-dir", default="data/exports/cblm_lang_002c")
    ap.add_argument("--cutoff", default=DEFAULT_CUTOFF)
    args = ap.parse_args()

    manifest = build_export(
        Path(args.transcripts), Path(args.benchmark), Path(args.out_dir), args.cutoff
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
