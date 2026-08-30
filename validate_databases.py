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
import heapq
import hashlib
import json
import os
import time

from jsonschema import Draft202012Validator

WD = os.path.dirname(os.path.abspath(__file__))
FAILURES = []
STATS = {}


SOURCE_ALLOWED_CANDIDATE_STATUSES = {
    "admitted": {
        "admitted", "pending_review", "quarantined", "rejected",
        "superseded_for_training",
    },
    "pending_review": {
        "pending_review", "quarantined", "rejected",
        "superseded_for_training",
    },
    "quarantined": {"quarantined", "rejected"},
    "rejected": {"rejected"},
    "superseded_for_training": {"superseded_for_training", "rejected"},
}


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


class StableHashSampler:
    """Keep the exact N lowest SHA-256-ranked records without full buffering."""

    def __init__(self, limit, seed, namespace):
        self.limit = max(0, int(limit))
        self.seed = str(seed)
        self.namespace = namespace
        self._heap = []

    def add(self, item_id, expected_digest, canonical_payload):
        if self.limit == 0:
            return
        identity = "%s|%s|%s" % (self.seed, self.namespace, item_id)
        rank = int.from_bytes(hashlib.sha256(identity.encode("utf-8")).digest(),
                              "big")
        # The root is the largest selected rank. A lower rank replaces it.
        entry = (-rank, str(item_id), expected_digest, canonical_payload)
        if len(self._heap) < self.limit:
            heapq.heappush(self._heap, entry)
        elif entry > self._heap[0]:
            heapq.heapreplace(self._heap, entry)

    def verify(self, label):
        for _, item_id, expected, payload in sorted(self._heap, reverse=True):
            check(expected == sha256_text(payload),
                  "%s hash mismatch: %s" % (label, item_id))
        return len(self._heap)


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
    record_ids = set()
    research_policies = {}
    schema_checked = 0
    header_count = message_count = rejected_count = 0
    hash_sampler = StableHashSampler(args.hash_sample, args.seed, "research")
    prev_map = {}
    header_ids = {}
    header_policies = {}

    for line_no, rec in stream_jsonl(path):
        rid = rec.get("record_id")
        if rid in record_ids:
            fail("duplicate research record_id: %s" % rid)
        record_ids.add(rid)

        rtype = rec.get("record_type")
        if rtype == "conversation":
            header_count += 1
            conv_id = (rec.get("source") or {}).get("conversation_id")
            header_ids[conv_id] = rid
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
        conv_id = (rec.get("source") or {}).get("conversation_id")
        effective_policy = {
            "default_admission": policy.get("default_admission"),
            "do_not_remember": bool(policy.get("do_not_remember")),
            "pii_status": policy.get("pii_status"),
            "license_status": policy.get("license_status"),
            "sensitivity": rec.get("sensitivity"),
            "record_type": rtype,
            "conversation_id": conv_id,
        }
        research_policies[rid] = effective_policy
        inherited_fields = {
            key: effective_policy[key]
            for key in (
                "default_admission", "do_not_remember", "pii_status",
                "license_status", "sensitivity",
            )
        }
        if rtype == "conversation":
            header_policies[conv_id] = inherited_fields
        elif rtype == "message":
            check(header_policies.get(conv_id) == inherited_fields,
                  "message policy diverges from conversation header: %s" % rid)
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
            fp = json.dumps({
                "conversation_id": rec["source"]["conversation_id"],
                "title": rec.get("title"),
                "create_time": rec["provenance"]["created_at"],
                "update_time": rec["provenance"]["updated_at"],
                "message_ids": rec["source"]["message_ids"],
            }, ensure_ascii=False, sort_keys=True)
            hash_sampler.add(rid, rec["provenance"]["source_hash"], fp)
            check(not rec["provenance"]["parent_record_ids"],
                  "conversation header has parents: %s" % rid)
        elif rtype == "message":
            mid = (rec["source"]["message_ids"] or [None])[0]
            fp = json.dumps({
                "conversation_id": rec["source"]["conversation_id"],
                "message_id": mid,
                "role": rec["content"].get("role"),
                "text": rec["content"].get("text"),
                "reasoning": rec["content"].get("reasoning"),
                "create_time": rec["provenance"]["created_at"],
            }, ensure_ascii=False, sort_keys=True)
            hash_sampler.add(rid, rec["provenance"]["source_hash"], fp)

            # lineage: chain integrity per conversation
            parents = rec["provenance"]["parent_record_ids"]
            check(len(parents) == 1, "message %s parent count != 1" % rid)
            conv_id = rec["source"]["conversation_id"]
            expected_parent = prev_map.get(conv_id, header_ids.get(conv_id))
            check(expected_parent == (parents[0] if parents else None),
                  "broken message chain in conversation %s at %s"
                  % (conv_id, rid))
            prev_map[conv_id] = rid
            derived_from = ((rec.get("relations") or {})
                            .get("derived_from", []))
            check(header_ids.get(conv_id) in derived_from,
                  "message %s is not derived from its conversation header"
                  % rid)
            for d in derived_from:
                if d not in record_ids:
                    # header precedes messages in emission order; if missing
                    # here it would have been a real ordering bug
                    fail("unresolved derived_from in %s -> %s" % (rid, d))

    check(header_count + message_count == expected_count,
          "research record count %d != manifest %s"
          % (header_count + message_count, expected_count))
    hash_checked = hash_sampler.verify("research source")
    STATS.update(research_headers=header_count, research_messages=message_count,
                 research_schema_checked=schema_checked,
                 research_hash_verified=hash_checked, rejected=rejected_count)
    return research_policies

def validate_training(args, validator, expected_count, research_policies):
    path = os.path.join(args.training_dir, "candidates.jsonl")
    seen = set()
    schema_checked = refs_checked = 0
    admission_counts = {}
    hash_sampler = StableHashSampler(args.hash_sample, args.seed, "training")

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
        source_policies = []
        for src in src_ids:
            refs_checked += 1
            parts = src.split(":")
            if cid is None:
                cid = parts[2] if len(parts) > 2 else None
            if src not in research_policies:
                fail("candidate %s references missing research record %s"
                     % (eid, src))
                continue
            source_policy = research_policies[src]
            source_cid = source_policy.get("conversation_id")
            check(source_cid == cid,
                  "candidate %s crosses conversation boundary via %s"
                  % (eid, src))
            source_policies.append((src, source_policy))

        check(rec.get("split_group") == "conversation:%s" % cid,
              "split_group mismatch for %s" % eid)

        # A derived candidate may be more restrictive than its sources, but it
        # may never silently relax a source record's training policy.
        for src, source_policy in source_policies:
            source_status = source_policy.get("default_admission")
            allowed = SOURCE_ALLOWED_CANDIDATE_STATUSES.get(source_status, set())
            check(status in allowed,
                  "candidate %s admission %r relaxes source %s policy %r"
                  % (eid, status, src, source_status))

        if status == "admitted":
            for src, source_policy in source_policies:
                check(not source_policy.get("do_not_remember"),
                      "admitted candidate %s uses do_not_remember source %s"
                      % (eid, src))
                check(source_policy.get("default_admission") == "admitted",
                      "admitted candidate %s uses non-admitted source %s"
                      % (eid, src))
                check(source_policy.get("pii_status") in {"clear", "reviewed"},
                      "admitted candidate %s uses unreviewed PII source %s"
                      % (eid, src))
                check(source_policy.get("license_status") in {"owned", "permitted"},
                      "admitted candidate %s uses unlicensed source %s"
                      % (eid, src))
                check(source_policy.get("sensitivity") in {"public", "internal"},
                      "admitted candidate %s uses sensitive source %s"
                      % (eid, src))

        # content_hash spot recompute using the same stable exact-N sampler.
        check(len(src_ids) >= 2,
              "candidate %s must cite prompt and target source records" % eid)
        for source_index in range(min(2, len(source_policies))):
            src, source_policy = source_policies[source_index]
            check(source_policy.get("record_type") == "message",
                  "candidate %s prompt/target source is not a message: %s"
                  % (eid, src))
        assistant_mid = (src_ids[1].rsplit(":", 1)[-1]
                         if len(src_ids) >= 2 else "")
        fp = json.dumps({
            "cid": cid,
            "assistant_mid": assistant_mid,
            "task": rec["task"],
            "prompt": rec["input"]["prompt"],
            "target": rec["target"]["text"],
        }, ensure_ascii=False, sort_keys=True)
        hash_sampler.add(eid, (rec.get("transform") or {}).get("content_hash"),
                         fp)

    check(len(seen) == expected_count,
          "candidate count %d != manifest %s" % (len(seen), expected_count))
    hash_checked = hash_sampler.verify("candidate content")
    STATS.update(candidates=len(seen), candidate_schema_checked=schema_checked,
                 candidate_hash_verified=hash_checked, refs_checked=refs_checked)
    return admission_counts

def main(argv=None):
    FAILURES.clear()
    STATS.clear()
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
    research_policies = validate_research(args, research_validator,
                                          research_expected)

    # 7. training DB
    print("[3/5] training DB (schema, policy, admission)...")
    with open(os.path.join(WD, "schemas", "training_example.schema.json"),
              encoding="utf-8") as f:
        training_validator = Draft202012Validator(json.load(f))
    admission_counts = validate_training(args, training_validator,
                                         training_expected, research_policies)

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


