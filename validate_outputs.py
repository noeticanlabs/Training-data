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
    reasoning = imgs = msgs = empty_text = empty_chain = bad = role_violations = 0
    total = 0
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            total = i
            r = json.loads(line)
            for k in ("conversation_id", "title", "tags", "scope", "messages",
                      "create_time", "source"):
                assert k in r, "missing %s on line %d" % (k, i)
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
                roles[m["role"]] += 1
                assert "text" in m, "no text on line %d" % i
                if not m["text"].strip():
                    empty_text += 1
                if m["role"] == "assistant" and m.get("reasoning"):
                    reasoning += 1
                if m.get("images"):
                    imgs += m["images"]
                if prev_role and m["role"] == prev_role:
                    role_violations += 1
                prev_role = m["role"]
    print("\n=== %s ===" % label)
    print("  records: %d   messages: %d (user=%d assistant=%d)" % (
        total, msgs, roles["user"], roles["assistant"]))
    print("  scopes: %s" % dict(scopes))
    print("  reasoning msgs: %d   image placeholders: %d" % (reasoning, imgs))
    print("  empty-text msgs: %d   empty chains: %d" % (empty_text, empty_chain))
    print("  role violations: %d" % role_violations)
    print("  top tags: %s" % dict(tags.most_common(8)))
    ok = empty_text == 0 and empty_chain == 0
    print("  OK" if ok else "  WARN: empty records present")
    return ok


def main():
    ok = True
    ok &= validate_jsonl(TRANSCRIPTS, "transcripts.jsonl (all)")
    if os.path.exists(WORK_ONLY):
        ok &= validate_jsonl(WORK_ONLY, "transcripts_work_only.jsonl")

    print("\n=== stats_report.json ===")
    with open(STATS, encoding="utf-8") as f:
        s = json.load(f)
    print("  kept conversations: %d" % s["kept"]["conversations"])
    print("  kept messages: %d" % s["kept"]["messages"]["total"])
    print("  drops by reason: %s" % s["drops"]["conversations_by_reason"])
    print("  models: %s" % s["models"])

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
    else:
        for g in pj[:6]:
            print("    %-22s %3d convos  %s" % (g["tag"], len(g["conversations"]),
                                                g.get("label", "")))

    print("\nALL CHECKS PASSED" if ok else "\nVALIDATION WARNINGS PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
