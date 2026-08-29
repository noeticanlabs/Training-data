#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Integrity validation for the dual governed databases (v0.02).

Checks
  1. Manifest SHA-256 digests match the files on disk.
  2. Every record conforms to schemas/research_record.schema.json (DB A) or
     schemas/training_example.schema.json (DB B) - full validation by default.
  3. record_id / training_example_id uniqueness.
  4. Lineage: conversation -> message parent chains are intact; every
     relations.derived_from and candidate source_record_ids resolve.
  5. Policy: is_do_not_remember => rejected admission; v0.02 defaults hold
     (epistemic_status == unknown, knowledge_status == unknown, split unassigned).
  6. Content addressing: source_hash recomputation on a sample of records.
  7. Cross-database closure: no candidate references a nonexistent research record.

Exit code 0 = all checks passed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time

from jsonschema import Draft202012Validator

WD = os.path.dirname(os.path.abspath(__file__))
FAILURES = []
STATS = {}


def fail(msg):
    FAILURES.append(msg)
    if len(FAILURES) <= 50:
        print("  FAIL:", msg)


def check(cond, msg):
    if not cond:
        fail(msg)
    return cond


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text):
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def stream_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield n, json.loads(line)
            except json.JSONDecodeError as exc:
                fail("invalid JSON at %s line %d: %s" % (path, n, exc))
                return


def check_manifest(db_dir, records_name, count_key):
    path = os.path.join(db_dir, "manifest.json")
    with open(path, encoding="utf-8") as f:
        manifest = json.load(f)
    records_path = os.path.join(db_dir, records_name)
    digest = sha256_file(records_path)
    key = "%s_sha256" % ("records" if "record_count" in manifest else "candidates")
    check(manifest.get(key) == digest,
          "manifest %s mismatch for %s" % (key, records_name))
    expected = manifest.get(count_key)
    check(isinstance(expected, int) and expected > 0,
          "manifest %s missing/invalid" % count_key)
    return manifest, expected

def validate_research(args, validator, expected_count):
    path = os.path.join(args.research_dir, "records.jsonl")
    rng = random.Random(args.seed)
    record_ids = set()
    schema_checked = hash_checked = 0
    header_count = message_count = rejected_count = 0
    hash_sample_conv = set(rng.sample(range(100000), args.hash_sample))
    hash_sample_msg = set(rng.sample(range(1000000), args.hash_sample * 5))
    prev_map = {}

    for line_no, rec in stream_jsonl(path):
        rid = rec.get("record_id")
        if rid in record_ids:
            fail("duplicate research record_id: %s" % rid)
        record_ids.add(rid)

        rtype = rec.get("record_type")
        if rtype == "conversation":
            header_count += 1
        elif rtype == "message":
            message_count += 1
        else:
            fail("unexpected record_type %r at line %d" % (rtype, line_no))

        # schema validation (sampled or full)
        if args.schema_sample == 0 or schema_checked < args.schema_sample:
            errors = list(validator.iter_errors(rec))
            if errors:
                fail("schema violation at line %d: %s"
                     % (line_no, errors[0].message[:200]))
            schema_checked += 1

        # policy invariants
        policy = rec.get("training_policy") or {}
        if policy.get("do_not_remember"):
            rejected_count += 1
            check(policy.get("default_admission") == "rejected",
                  "do_not_remember record not rejected: %s" % rid)
        check(rec.get("epistemic_status") == "unknown",
              "v0.02 default epistemic_status violated: %s" % rid)

        scope_personal = "scope:personal" in (rec.get("tags") or [])
        expected_sens = "private" if scope_personal else "internal"
        check(rec.get("sensitivity") == expected_sens,
              "sensitivity mismatch for %s" % rid)

        # source_hash spot recompute (deterministic sample)
        if rtype == "conversation":
            if hash_checked < args.hash_sample and \
                    hash(rid) % 1000 < args.hash_sample // 2:
                fp = json.dumps({
                    "conversation_id": rec["source"]["conversation_id"],
                    "title": rec.get("title"),
                    "create_time": rec["provenance"]["created_at"],
                    "update_time": rec["provenance"]["updated_at"],
                    "message_ids": rec["source"]["message_ids"],
                }, ensure_ascii=False, sort_keys=True)
                check(rec["provenance"]["source_hash"] == sha256_text(fp),
                      "conversation source_hash mismatch: %s" % rid)
                hash_checked += 1
            check(not rec["provenance"]["parent_record_ids"],
                  "conversation header has parents: %s" % rid)
        elif rtype == "message":
            mid = (rec["source"]["message_ids"] or [None])[0]
            if hash_checked < args.hash_sample * 5 and \
                    hash(rid) % 1000 < args.hash_sample:
                fp = json.dumps({
                    "conversation_id": rec["source"]["conversation_id"],
                    "message_id": mid,
                    "role": rec["content"].get("role"),
                    "text": rec["content"].get("text"),
                    "reasoning": rec["content"].get("reasoning"),
                    "create_time": rec["provenance"]["created_at"],
                }, ensure_ascii=False, sort_keys=True)
                check(rec["provenance"]["source_hash"] == sha256_text(fp),
                      "message source_hash mismatch: %s" % rid)
                hash_checked += 1

            # lineage: chain integrity per conversation
            parents = rec["provenance"]["parent_record_ids"]
            check(len(parents) == 1, "message %s parent count != 1" % rid)
            conv_id = rec["source"]["conversation_id"]
            if conv_id in prev_map:
                check(prev_map[conv_id] == (parents[0] if parents else None),
                      "broken message chain in conversation %s at %s"
                      % (conv_id, rid))
            prev_map[conv_id] = rid
            for d in (rec.get("relations") or {}).get("derived_from", []):
                if d not in record_ids:
                    # header precedes messages in emission order; if missing
                    # here it would have been a real ordering bug
                    fail("unresolved derived_from in %s -> %s" % (rid, d))

    check(header_count + message_count == expected_count,
          "research record count %d != manifest %s"
          % (header_count + message_count, expected_count))
    STATS.update(research_headers=header_count, research_messages=message_count,
                 research_schema_checked=schema_checked,
                 research_hash_verified=hash_checked, rejected=rejected_count)
    return record_ids

def validate_training(args, validator, expected_count, research_ids):
    path = os.path.join(args.training_dir, "candidates.jsonl")
    seen = set()
    schema_checked = hash_checked = refs_checked = 0
    admission_counts = {}

    for line_no, rec in stream_jsonl(path):
        eid = rec.get("training_example_id")
        if eid in seen:
            fail("duplicate training_example_id: %s" % eid)
        seen.add(eid)

        if args.schema_sample == 0 or schema_checked < args.schema_sample:
            errors = list(validator.iter_errors(rec))
            if errors:
                fail("schema violation at line %d: %s"
                     % (line_no, errors[0].message[:200]))
            schema_checked += 1

        status = (rec.get("admission") or {}).get("status")
        admission_counts[status] = admission_counts.get(status, 0) + 1
        check(rec.get("knowledge_status") == "unknown",
              "v0.02 default knowledge_status violated: %s" % eid)
        check(rec.get("split") == "unassigned",
              "v0.02 default split violated: %s" % eid)
        check((rec.get("metadata") or {}).get("reasoning_excluded_from_target")
              is True, "reasoning flag missing on %s" % eid)

        # target/input sanity
        check(bool((rec.get("target") or {}).get("text", "").strip()),
              "empty target text: %s" % eid)
        check(bool((rec.get("input") or {}).get("prompt", "").strip()),
              "empty input prompt: %s" % eid)

        # lineage resolution into research DB
        src_ids = rec.get("source_record_ids") or []
        cid = None
        for src in src_ids:
            refs_checked += 1
            parts = src.split(":")
            if cid is None:
                cid = parts[2] if len(parts) > 2 else None
            if src not in research_ids:
                fail("candidate %s references missing research record %s"
                     % (eid, src))
                break

        check(rec.get("split_group") == "conversation:%s" % cid,
              "split_group mismatch for %s" % eid)

        # content_hash spot recompute (deterministic ~3% sample)
        if hash_checked < args.hash_sample and \
                hash(eid) % 1000 < args.hash_sample:
            fp = json.dumps({
                "cid": cid,
                "assistant_mid": src_ids[1].rsplit(":", 1)[-1],
                "task": rec["task"],
                "prompt": rec["input"]["prompt"],
                "target": rec["target"]["text"],
            }, ensure_ascii=False, sort_keys=True)
            check(rec["transform"]["content_hash"] == sha256_text(fp),
                  "candidate content_hash mismatch: %s" % eid)
            hash_checked += 1

    check(len(seen) == expected_count,
          "candidate count %d != manifest %s" % (len(seen), expected_count))
    STATS.update(candidates=len(seen), candidate_schema_checked=schema_checked,
                 candidate_hash_verified=hash_checked, refs_checked=refs_checked)
    return admission_counts

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Validate research_db and training_db against schemas, "
                    "lineage, policy, and content-addressing invariants.")
    ap.add_argument("--research-dir", default="data/research_db")
    ap.add_argument("--training-dir", default="data/training_db")
    ap.add_argument("--schema-sample", type=int, default=0,
                    help="validate only N records per DB against the JSON "
                         "schema (0 = full validation)")
    ap.add_argument("--hash-sample", type=int, default=100,
                    help="recompute source_hash/content_hash for ~N records")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    t0 = time.time()
    print("== Dual database validation ==")

    # 1. manifests + file digests
    print("[1/5] manifest digests...")
    research_manifest, research_expected = check_manifest(
        args.research_dir, "records.jsonl", "record_count")
    training_manifest, training_expected = check_manifest(
        args.training_dir, "candidates.jsonl", "candidate_count")

    # 2-6. research DB
    print("[2/5] research DB (schema, policy, lineage, hashing)...")
    with open(os.path.join(WD, "schemas", "research_record.schema.json"),
              encoding="utf-8") as f:
        research_validator = Draft202012Validator(json.load(f))
    research_ids = validate_research(args, research_validator,
                                     research_expected)

    # 7. training DB
    print("[3/5] training DB (schema, policy, admission)...")
    with open(os.path.join(WD, "schemas", "training_example.schema.json"),
              encoding="utf-8") as f:
        training_validator = Draft202012Validator(json.load(f))
    admission_counts = validate_training(args, training_validator,
                                         training_expected, research_ids)

    # summary
    print("[4/5] cross-database closure: %d source references checked"
          % STATS.get("refs_checked", 0))
    print("[5/5] summary")
    print("  research_db: %d headers + %d messages (manifest: %d)"
          % (STATS["research_headers"], STATS["research_messages"],
             research_manifest.get("record_count")))
    print("  schema-validated: research %d, training %d"
          % (STATS["research_schema_checked"],
             STATS["candidate_schema_checked"]))
    print("  hashes recomputed: research %d, training %d"
          % (STATS["research_hash_verified"],
             STATS["candidate_hash_verified"]))
    print("  admission: %s" % admission_counts)
    print("  elapsed: %.1fs" % (time.time() - t0))

    if FAILURES:
        print("\nRESULT: FAILED (%d problems; first 50 shown above)"
              % len(FAILURES))
        return 1
    print("\nRESULT: ALL DATABASE CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



