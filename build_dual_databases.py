#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the two governed databases from normalized transcript JSONL.

Database A: data/research_db/
    Canonical research-history records. Preserves source lineage and reasoning.

Database B: data/training_db/
    Derived training candidates. Excludes reasoning from targets and admits
    nothing automatically. `is_do_not_remember` conversations are rejected.

This builder intentionally does NOT infer epistemic truth/status from prose.
All generated research records start with epistemic_status="unknown" unless a
future reviewed status layer supplies stronger evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter

SCHEMA_VERSION = "0.02"
PIPELINE_VERSION = "0.02"


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_message_id(conversation_id: str, index: int, message: dict) -> str:
    mid = message.get("message_id")
    if mid:
        return str(mid)
    payload = "%s|%d|%s|%s" % (
        conversation_id,
        index,
        message.get("role", ""),
        message.get("text", ""),
    )
    return "generated-" + sha256_text(payload)[:24]


def jsonl_write(path: str, records) -> int:
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def load_jsonl(path: str):
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError("invalid JSON on %s line %d: %s" % (path, line_no, exc))


def default_admission(conversation: dict) -> tuple[str, str | None]:
    if conversation.get("is_do_not_remember"):
        return "rejected", "source conversation is marked do_not_remember"
    return "pending_review", None


def research_records(conversation: dict):
    cid = str(conversation.get("conversation_id") or "unknown")
    tags = list(conversation.get("tags") or [])
    scope = conversation.get("scope") or "misc"
    admission, reason = default_admission(conversation)
    conv_record_id = "research:conversation:%s" % cid

    conv_fingerprint = json.dumps(
        {
            "conversation_id": cid,
            "title": conversation.get("title"),
            "create_time": conversation.get("create_time"),
            "update_time": conversation.get("update_time"),
            "message_ids": [m.get("message_id") for m in conversation.get("messages") or []],
        },
        ensure_ascii=False,
        sort_keys=True,
    )

    yield {
        "schema_version": SCHEMA_VERSION,
        "record_id": conv_record_id,
        "record_type": "conversation",
        "title": conversation.get("title"),
        "project_ids": tags,
        "tags": tags + (["scope:%s" % scope] if scope else []),
        "source": {
            "source_type": conversation.get("source") or "chatgpt",
            "conversation_id": cid,
            "message_ids": [
                str(m.get("message_id"))
                for m in conversation.get("messages") or []
                if m.get("message_id")
            ],
            "artifact_ids": [],
            "source_uri": None,
        },
        "content": {
            "text": conversation.get("title") or "",
            "role": None,
            "content_type": "conversation_header",
            "reasoning": None,
            "images": sum(int(m.get("images") or 0) for m in conversation.get("messages") or []),
        },
        "provenance": {
            "created_at": conversation.get("create_time"),
            "updated_at": conversation.get("update_time"),
            "source_hash": sha256_text(conv_fingerprint),
            "parent_record_ids": [],
            "model_slugs": sorted({
                str(m.get("model_slug"))
                for m in conversation.get("messages") or []
                if m.get("model_slug")
            }),
            "pipeline_version": PIPELINE_VERSION,
        },
        "epistemic_status": "unknown",
        "relations": {},
        "sensitivity": "private" if scope == "personal" else "internal",
        "training_policy": {
            "default_admission": admission,
            "do_not_remember": bool(conversation.get("is_do_not_remember")),
            "contains_reasoning": any(bool(m.get("reasoning")) for m in conversation.get("messages") or []),
            "pii_status": "not_scanned",
            # Source ownership cannot be inferred from transcript presence.
            "license_status": "unknown",
            "notes": reason,
        },
    }

    previous_record_id = conv_record_id
    for index, message in enumerate(conversation.get("messages") or []):
        mid = stable_message_id(cid, index, message)
        record_id = "research:message:%s:%s" % (cid, mid)
        text = message.get("text") or ""
        reasoning = message.get("reasoning")
        source_hash = sha256_text(json.dumps({
            "conversation_id": cid,
            "message_id": mid,
            "role": message.get("role"),
            "text": text,
            "reasoning": reasoning,
            "create_time": message.get("create_time"),
        }, ensure_ascii=False, sort_keys=True))

        yield {
            "schema_version": SCHEMA_VERSION,
            "record_id": record_id,
            "record_type": "message",
            "title": conversation.get("title"),
            "project_ids": tags,
            "tags": tags + (["scope:%s" % scope] if scope else []),
            "source": {
                "source_type": conversation.get("source") or "chatgpt",
                "conversation_id": cid,
                "message_ids": [mid],
                "artifact_ids": [],
                "source_uri": None,
            },
            "content": {
                "text": text,
                "role": message.get("role"),
                "content_type": message.get("content_type") or "text",
                "reasoning": reasoning,
                "images": int(message.get("images") or 0),
            },
            "provenance": {
                "created_at": message.get("create_time"),
                "updated_at": None,
                "source_hash": source_hash,
                "parent_record_ids": [previous_record_id],
                "model_slugs": [str(message.get("model_slug"))] if message.get("model_slug") else [],
                "pipeline_version": PIPELINE_VERSION,
            },
            "epistemic_status": "unknown",
            "relations": {
                "derived_from": [conv_record_id]
            },
            "sensitivity": "private" if scope == "personal" else "internal",
            "training_policy": {
                "default_admission": admission,
                "do_not_remember": bool(conversation.get("is_do_not_remember")),
                "contains_reasoning": bool(reasoning),
                "pii_status": "not_scanned",
                "license_status": "unknown",
                "notes": reason,
            },
        }
        previous_record_id = record_id


def training_candidates(conversation: dict, context_turns: int = 6):
    """Generate generic dialogue candidates only.

    The builder deliberately avoids inferring specialized tasks such as theorem
    proving or code repair from keywords. Those are generated by later reviewed
    task recipes from the canonical research database.
    """
    cid = str(conversation.get("conversation_id") or "unknown")
    tags = list(conversation.get("tags") or [])
    scope = conversation.get("scope") or "misc"
    admission, reason = default_admission(conversation)
    messages = conversation.get("messages") or []

    ids = [stable_message_id(cid, i, m) for i, m in enumerate(messages)]

    for i, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue

        user_index = i - 1
        while user_index >= 0 and messages[user_index].get("role") != "user":
            user_index -= 1
        if user_index < 0:
            continue

        user_message = messages[user_index]
        target_text = message.get("text") or ""
        prompt_text = user_message.get("text") or ""
        if not target_text.strip() or not prompt_text.strip():
            continue

        start = max(0, user_index - context_turns)
        context_messages = [
            {"role": m.get("role") or "unknown", "text": m.get("text") or ""}
            for m in messages[start:user_index]
            if (m.get("text") or "").strip()
        ]

        source_record_ids = [
            "research:message:%s:%s" % (cid, ids[user_index]),
            "research:message:%s:%s" % (cid, ids[i]),
        ]
        for j in range(start, user_index):
            source_record_ids.append("research:message:%s:%s" % (cid, ids[j]))

        source_record_ids = list(dict.fromkeys(source_record_ids))
        fingerprint = json.dumps({
            "cid": cid,
            "assistant_mid": ids[i],
            "task": "multi_turn_dialogue",
            "prompt": prompt_text,
            "target": target_text,
        }, ensure_ascii=False, sort_keys=True)
        example_id = "train:%s:%s" % (cid, sha256_text(fingerprint)[:24])

        yield {
            "schema_version": SCHEMA_VERSION,
            "training_example_id": example_id,
            "task": "multi_turn_dialogue",
            "knowledge_domains": tags or ["misc"],
            "source_record_ids": source_record_ids,
            "source_artifact_ids": [],
            "knowledge_status": "unknown",
            "admission": {
                "status": admission,
                "reviewer": None,
                "reviewed_at": None,
                "reason": reason,
                "recipe": "dialogue_candidate_v0.01",
            },
            "split": "unassigned",
            "split_group": "conversation:%s" % cid,
            "input": {
                "system": None,
                "prompt": prompt_text,
                "context": None,
                "messages": context_messages,
                "structured": None,
            },
            "target": {
                "text": target_text,
                "messages": [],
                "structured": None,
            },
            "evaluation": {},
            "transform": {
                "pipeline_version": PIPELINE_VERSION,
                "recipe_version": "dialogue_candidate_v0.01",
                "content_hash": sha256_text(fingerprint),
            },
            "metadata": {
                "scope": scope,
                "source_title": conversation.get("title"),
                "assistant_model_slug": message.get("model_slug"),
                "reasoning_excluded_from_target": True,
            },
        }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build canonical research DB and derived training candidates.")
    ap.add_argument("--transcripts", default="data/transcripts/transcripts.jsonl")
    ap.add_argument("--research-dir", default="data/research_db")
    ap.add_argument("--training-dir", default="data/training_db")
    ap.add_argument("--context-turns", type=int, default=6)
    args = ap.parse_args(argv)

    os.makedirs(args.research_dir, exist_ok=True)
    os.makedirs(args.training_dir, exist_ok=True)

    research_path = os.path.join(args.research_dir, "records.jsonl")
    candidates_path = os.path.join(args.training_dir, "candidates.jsonl")
    research_temp = research_path + ".tmp"
    candidates_temp = candidates_path + ".tmp"
    source_sha256 = sha256_file(args.transcripts)

    conversation_count = 0

    def all_research_records():
        nonlocal conversation_count
        for conversation in load_jsonl(args.transcripts):
            conversation_count += 1
            yield from research_records(conversation)

    admission_counts = Counter()
    task_counts = Counter()

    def all_training_candidates():
        for conversation in load_jsonl(args.transcripts):
            for record in training_candidates(conversation, args.context_turns):
                admission_counts[record["admission"]["status"]] += 1
                task_counts[record["task"]] += 1
                yield record

    try:
        research_count = jsonl_write(research_temp, all_research_records())
        candidate_count = jsonl_write(
            candidates_temp, all_training_candidates())
        check_sha256 = sha256_file(args.transcripts)
        if check_sha256 != source_sha256:
            raise RuntimeError(
                "source transcripts changed during the two-pass build")
        os.replace(research_temp, research_path)
        os.replace(candidates_temp, candidates_path)
    finally:
        for temp_path in (research_temp, candidates_temp):
            if os.path.exists(temp_path):
                os.remove(temp_path)

    manifest_common = {
        "generated_at": now_iso(),
        "schema_version": SCHEMA_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "source_transcripts": args.transcripts,
        "source_transcripts_sha256": source_sha256,
        "conversation_count": conversation_count,
    }

    research_manifest = dict(manifest_common)
    research_manifest.update({
        "database": "research",
        "record_count": research_count,
        "records_sha256": sha256_file(research_path),
        "epistemic_default": "unknown",
    })
    with open(os.path.join(args.research_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(research_manifest, f, ensure_ascii=False, indent=2)

    training_manifest = dict(manifest_common)
    training_manifest.update({
        "database": "training",
        "candidate_count": candidate_count,
        "candidates_sha256": sha256_file(candidates_path),
        "admission_counts": dict(admission_counts),
        "task_counts": dict(task_counts),
        "automatic_admission": False,
        "reasoning_in_targets": False,
        "split_policy": "unassigned; split_group is conversation-level",
    })
    with open(os.path.join(args.training_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(training_manifest, f, ensure_ascii=False, indent=2)

    print("Research DB: %d records -> %s" % (research_count, research_path))
    print("Training DB: %d candidates -> %s" % (candidate_count, candidates_path))
    print("Admission: %s" % dict(admission_counts))
    print("Tasks: %s" % dict(task_counts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
