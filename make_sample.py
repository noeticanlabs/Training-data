#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate a seeded random sample of transcripts for human review."""
from __future__ import annotations
import json, os, random

WD = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(WD, "data", "transcripts", "transcripts.jsonl")
OUT = os.path.join(WD, "data", "transcripts", "sample_review.jsonl")

random.seed(42)
records = []
with open(SRC, encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))

sample = random.sample(records, min(10, len(records)))
with open(OUT, "w", encoding="utf-8") as f:
    for r in sample:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print("Wrote %d sample records to %s" % (len(sample), OUT))
