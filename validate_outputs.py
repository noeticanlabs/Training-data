#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Post-run integrity validation for the transcripts pipeline."""
from __future__ import annotations
import collections, json, os, sys

WD = os.path.dirname(os.path.abspath(__file__))
TRANSCRIPTS = os.path.join(WD, "data", "transcripts", "transcripts.jsonl")
WORK_ONLY = os.path.join(WD, "data", "transcripts", "transcripts_work_only.jsonl")
STATS = os.path.join(WD, "data", "transcripts", "stats_report.json")
PII = os.path.join(WD, "data", "transcripts", "pii_report.json")
INDEX = os.path.join(WD, "data", "index", "conversations.json")
PROJECTS = os.path.join(WD, "data", "index", "projects.json")


def validate_jsonl(path, label):
    roles = collections.Counter()
    scopes = collections.Counter()
    tags = collections.Counter()
    reasoning = imgs = msgs = empty_text = empty_chain = role_violations = 0
    bad_roles = missing_message_ids = duplicate_conversations = duplicate_messages = 0
    conversation_ids = set()
    message_ids = set()
    total = 0
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            total = i
            r = json.loads(line)
            for k in ("conversation_id", "title", "tags", "scope", "messages",
                      "create_time", "source"):
                assert k in r, "missing %s on line %d" % (k, i)
            cid = r["conversation_id"]
            if cid in conversation_ids:
                duplicate_conversations += 1
            conversation_ids.add(cid)
            scopes[r["scope"]] += 1
            for t in r["tags"]:
                tags[t] += 1
            chain = r["messages"]
            if not chain:
                empty_chain += 1
                continue
            prev_role = None
            for m in chain:
                msgs += 1
                role = m.get("role")
                roles[role] += 1
                if role not in ("user", "assistant"):
                    bad_roles += 1
                mid = m.get("message_id")
                if not mid:
                    missing_message_ids += 1
                elif mid in message_ids:
                    duplicate_messages += 1
                else:
                    message_ids.add(mid)
                assert "text" in m, "no text on line %d" % i
                if not m["text"].strip():
                    empty_text += 1
                if role == "assistant" and m.get("reasoning"):
                    reasoning += 1
                if m.get("images"):
                    imgs += m["images"]
                if prev_role and role == prev_role:
                    role_violations += 1
                prev_role = role
    print("\n=== %s ===" % label)
    print("  records: %d   messages: %d (user=%d assistant=%d)" % (
        total, msgs, roles["user"], roles["assistant"]))
    print("  scopes: %s" % dict(scopes))
    print("  reasoning msgs: %d   image placeholders: %d" % (reasoning, imgs))
    print("  empty-text msgs: %d   empty chains: %d" % (empty_text, empty_chain))
    print("  role violations: %d   invalid roles: %d" %
          (role_violations, bad_roles))
    print("  duplicate conversations: %d   duplicate messages: %d   "
          "missing message IDs: %d" %
          (duplicate_conversations, duplicate_messages, missing_message_ids))
    print("  top tags: %s" % dict(tags.most_common(8)))
    ok = all(value == 0 for value in (
        empty_text, empty_chain, role_violations, bad_roles,
        duplicate_conversations, duplicate_messages, missing_message_ids,
    ))
    print("  OK" if ok else "  FAIL: integrity violations present")
    return ok, {
        "records": total,
        "messages": msgs,
        "conversation_ids": conversation_ids,
        "scopes": scopes,
    }


def main():
    ok = True
    all_ok, all_metrics = validate_jsonl(
        TRANSCRIPTS, "transcripts.jsonl (all)")
    ok &= all_ok
    if os.path.exists(WORK_ONLY):
        work_ok, work_metrics = validate_jsonl(
            WORK_ONLY, "transcripts_work_only.jsonl")
        ok &= work_ok
        is_subset = work_metrics["conversation_ids"].issubset(
            all_metrics["conversation_ids"])
        no_personal = work_metrics["scopes"].get("personal", 0) == 0
        print("  work-only subset closure: %s" %
              ("OK" if is_subset else "FAIL"))
        print("  work-only personal exclusion: %s" %
              ("OK" if no_personal else "FAIL"))
        ok &= is_subset and no_personal

    print("\n=== stats_report.json ===")
    with open(STATS, encoding="utf-8") as f:
        s = json.load(f)
    print("  kept conversations: %d" % s["kept"]["conversations"])
    print("  kept messages: %d" % s["kept"]["messages"]["total"])
    print("  drops by reason: %s" % s["drops"]["conversations_by_reason"])
    print("  models: %s" % s["models"])
    stats_match = (
        s["kept"]["conversations"] == all_metrics["records"] and
        s["kept"]["messages"]["total"] == all_metrics["messages"]
    )
    print("  transcript count reconciliation: %s" %
          ("OK" if stats_match else "FAIL"))
    ok &= stats_match

    if os.path.exists(PII):
        print("\n=== pii_report.json ===")
        with open(PII, encoding="utf-8") as f:
            p = json.load(f)
        print("  totals_by_category: %s" % p["totals_by_category"])
        print("  conversations with hits: %d" % len(p["conversations_with_hits"]))

    print("\n=== index/conversations.json ===")
    with open(INDEX, encoding="utf-8") as f:
        idx = json.load(f)
    if isinstance(idx, dict):
        entries = idx.get("conversations", [])
        print("  header: count=%s generated_at=%s" % (
            idx.get("count"), idx.get("generated_at", "?")))
    else:
        entries = idx
    print("  entries: %d" % len(entries))
    index_count = idx.get("count") if isinstance(idx, dict) else len(entries)
    index_count_ok = index_count == len(entries) == all_metrics["records"]
    entry_ids = {
        entry.get("conversation_id", entry.get("id"))
        for entry in entries
        if isinstance(entry, dict)
    }
    entry_ids.discard(None)
    index_closure_ok = (entry_ids == all_metrics["conversation_ids"])
    print("  count reconciliation: %s" %
          ("OK" if index_count_ok else "FAIL"))
    print("  ID closure with transcripts: %s" %
          ("OK" if index_closure_ok else "FAIL"))
    ok &= index_count_ok and index_closure_ok
    if entries:
        print("  sample keys: %s" % list(entries[0].keys()))

    print("\n=== index/projects.json ===")
    with open(PROJECTS, encoding="utf-8") as f:
        pj = json.load(f)
    if isinstance(pj, dict):
        tags_obj = pj.get("tags", {})
        print("  total conversations: %s" % pj.get("total_conversations"))
        print("  scope: %s" % pj.get("scope"))
        if isinstance(tags_obj, dict) and tags_obj:
            first = next(iter(tags_obj.values()))
            if isinstance(first, dict):
                def n_conv(info):
                    c = info.get("conversations", 0)
                    return len(c) if isinstance(c, list) else c
                ranked = sorted(tags_obj.items(), key=lambda kv: -n_conv(kv[1]))
                for slug, info in ranked[:10]:
                    print("    %-14s %4d convos  %s" % (slug, n_conv(info),
                                                        info.get("label", "")))
            else:
                print("  tags: %s" % dict(sorted(tags_obj.items(),
                                                 key=lambda kv: -kv[1])[:10]))
        project_count_ok = pj.get("total_conversations") == all_metrics["records"]
        print("  count reconciliation: %s" %
              ("OK" if project_count_ok else "FAIL"))
        ok &= project_count_ok
    else:
        for g in pj[:6]:
            print("    %-22s %3d convos  %s" % (g["tag"], len(g["conversations"]),
                                                g.get("label", "")))

    print("\nALL CHECKS PASSED" if ok else "\nVALIDATION FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
