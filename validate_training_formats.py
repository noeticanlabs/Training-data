#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Integrity validation for the training-format outputs in data/training/."""
from __future__ import annotations
import json, os, sys

WD = os.path.dirname(os.path.abspath(__file__))
TD = os.path.join(WD, "data", "training")

failures = []


def check(cond, msg):
    if not cond:
        failures.append(msg)
    return cond


def validate_jsonl(path, checker):
    n = 0
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            n = i
            checker(json.loads(line), "%s line %d" % (os.path.basename(path), i))
    return n


def chk_openai(r, where):
    check(set(r.keys()) == {"messages"}, "%s: extra/missing top-level keys" % where)
    msgs = r["messages"]
    check(len(msgs) >= 3, "%s: too few messages" % where)
    check(msgs[0]["role"] == "system", "%s: first message not system" % where)
    for m in msgs:
        check(set(m.keys()) == {"role", "content"}, "%s: bad message keys" % where)
        check(m["role"] in ("system", "user", "assistant"),
              "%s: bad role %s" % (where, m["role"]))
        check(bool(m["content"].strip()), "%s: empty content" % where)
    for a, b in zip(msgs[1:], msgs[2:]):
        check(a["role"] != b["role"], "%s: consecutive same-role turns" % where)
    check(msgs[1]["role"] == "user", "%s: turn 2 not user" % where)


def chk_sharegpt(r, where):
    convs = r["conversations"]
    check(len(convs) >= 2, "%s: too few turns" % where)
    check(convs[0]["from"] == "human", "%s: first turn not human" % where)
    for c in convs:
        check(c["from"] in ("human", "gpt"), "%s: bad 'from'" % where)
        check(bool(c["value"].strip()), "%s: empty value" % where)
    for a, b in zip(convs, convs[1:]):
        check(a["from"] != b["from"], "%s: consecutive same-role" % where)
    check(isinstance(r.get("system"), str) and r["system"],
          "%s: missing system" % where)


def chk_alpaca(r, where):
    for k in ("instruction", "input", "output"):
        check(k in r, "%s: missing %s" % (where, k))
    check(bool(r.get("output", "").strip()), "%s: empty output" % where)
    check(bool(r.get("input", "").strip()), "%s: empty input" % where)


def main():
    # load split manifest for cross-checks
    with open(os.path.join(TD, "split_manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    splits = manifest["splits"]
    counts = {"train": sum(1 for v in splits.values() if v == "train"),
              "val": sum(1 for v in splits.values() if v == "val")}
    print("Manifest: %d conversations (train=%d val=%d) seed=%s"
          % (manifest["n_conversations"], counts["train"], counts["val"],
             manifest["seed"]))
    check(manifest["n_conversations"] == len(splits),
          "manifest count mismatch")

    results = {}
    for fmt, checker in (("openai", chk_openai), ("sharegpt", chk_sharegpt),
                         ("alpaca", chk_alpaca)):
        for split in ("train", "val"):
            path = os.path.join(TD, fmt, "%s.jsonl" % split)
            if not os.path.exists(path):
                failures.append("missing file: %s" % path)
                continue
            n = validate_jsonl(path, checker)
            results["%s/%s" % (fmt, split)] = n
            expect = (counts[split] if fmt != "alpaca" else None)
            if expect is not None:
                check(n == expect, "%s/%s: %d records, expected %d"
                      % (fmt, split, n, expect))
            print("  %-8s %-5s %6d records OK" % (fmt, split, n))

    # parquet
    try:
        import pyarrow.parquet as pq
        for split in ("train", "val"):
            path = os.path.join(TD, "hf", "%s-00000.parquet" % split)
            if not os.path.exists(path):
                failures.append("missing file: %s" % path)
                continue
            t = pq.read_table(path)
            d = t.to_pydict()
            n = t.num_rows
            check(n == counts[split], "hf/%s: %d rows, expected %d"
                  % (split, n, counts[split]))
            check(set(d["split"]) == {split}, "hf/%s: wrong split labels" % split)
            # round-trip content check: messages struct present
            check(len(d["messages"]) == n, "hf/%s: messages column size" % split)
            print("  %-8s %-5s %6d rows OK" % ("hf", split, n))
    except ImportError:
        print("  hf       skipped (pyarrow not installed)")

    print()
    if failures:
        print("FAILURES (%d):" % len(failures))
        for f in failures[:20]:
            print("  -", f)
        return 1
    print("ALL FORMAT CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
