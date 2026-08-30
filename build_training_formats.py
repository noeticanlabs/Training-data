#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build legacy training formats directly from cleaned transcripts.

This route bypasses the governed training-candidate admission layer and is
therefore disabled unless the caller explicitly acknowledges that fact with
``--unsafe-legacy-transcript-export``. Production exports must originate from
reviewed, admitted records in the governed training database.

INPUT   (--transcripts)   data/transcripts/transcripts.jsonl
                          data/transcripts/transcripts_work_only.jsonl (with --scope work)
OUTPUT  (--out-dir)       split_manifest.json   conversation_id -> train|val
                          format_stats.json     per-format x per-split statistics
                          openai/train.jsonl    openai/val.jsonl
                          sharegpt/train.jsonl  sharegpt/val.jsonl
                          alpaca/train.jsonl    alpaca/val.jsonl
                          hf/train-00000.parquet  hf/val-00000.parquet

Splits are assigned per-conversation (never per-message, preventing leakage)
with a fixed seed and shared by every format, so results are directly
comparable across training formats.

Record schemas
--------------
openai  : {"messages": [{"role": "system"|"user"|"assistant", "content": str}, ...]}
          Strict fine-tune API shape - no extra keys.
sharegpt: {"conversations": [{"from": "human"|"gpt", "value": str}, ...],
           "system": str, "id": str, "tags": [...], "scope": str, "title": str}
          Metadata lives at top level (ignored by trainers, kept for analysis).
alpaca  : {"instruction": title, "input": user turn, "output": assistant turn,
           "context": "tags: ..."} - one record per (user -> assistant) pair.
hf      : parquet columns: conversation_id, title, tags, scope,
          messages (list<struct{role, content}>), num_messages, est_tokens, split.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys
import time

CHAR_PER_TOKEN = 3.8  # same heuristic as the transcript converter

# slug -> label, mirrors TAGS in convert_export_to_transcripts.py
TAG_LABELS = {
    "gr-nr": "GR/NR solvers & relativity",
    "coherence": "Coherence / Coh ecosystem",
    "noetica": "Noetica agents & platform",
    "lean-math": "Lean & math formalization",
    "numerics": "Numerical methods & simulation",
    "physics": "Physics / science research",
    "data-ai": "Data & AI/ML",
    "papers": "Manuscripts & reviews",
    "engineering": "Software engineering",
    "documentation": "Docs / manuals / tutorials",
    "creative": "Creative / media",
    "personal": "Personal / life admin",
    "math": "Math help",
    "misc": "general",
}

DEFAULT_SYSTEM = ("You are a research and engineering assistant collaborating "
                  "with the user on their projects. This conversation covers: "
                  "{topics}.")


def estimate_tokens(text):
    return max(0, int(len(text) / CHAR_PER_TOKEN))


def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def write_jsonl(path, records):
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def load_transcripts(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def assign_splits(records, val_ratio, seed):
    """Deterministic per-conversation split; returns {conversation_id: split}."""
    rng = random.Random(seed)
    ids = sorted(r["conversation_id"] for r in records)
    rng.shuffle(ids)
    n_val = max(1, round(len(ids) * val_ratio)) if ids else 0
    val = set(ids[:n_val])
    return {cid: ("val" if cid in val else "train") for cid in ids}


def make_system_prompt(rec, override):
    if override is not None:
        return override
    topics = ", ".join(TAG_LABELS.get(t, t) for t in rec.get("tags", []))
    return DEFAULT_SYSTEM.format(topics=topics)


def messages_for_training(rec, include_reasoning):
    """Copy of messages; drops reasoning unless requested."""
    out = []
    for m in rec["messages"]:
        m2 = (dict(m) if include_reasoning
              else {k: v for k, v in m.items() if k != "reasoning"})
        out.append(m2)
    return out


def message_content(message, include_reasoning):
    content = message["text"]
    if (include_reasoning and message["role"] == "assistant"
            and message.get("reasoning")):
        content = ("<thinking>\n%s\n</thinking>\n\n%s"
                   % (message["reasoning"], content))
    return content


def maybe_truncate(msgs, max_turns):
    if max_turns and len(msgs) > max_turns:
        msgs = msgs[:max_turns]
    return msgs


def trim_dangling(msgs):
    """Drop leading assistant turns and a trailing user turn.

    Training formats expect conversations to start with the user prompt and
    end with an assistant response. Some conversations begin with an assistant
    message (empty leading user turns were dropped upstream) or end mid-turn.
    """
    while msgs and msgs[0]["role"] == "assistant":
        msgs = msgs[1:]
    while msgs and msgs[-1]["role"] == "user":
        msgs = msgs[:-1]
    return msgs

# ---------------------------------------------------------------------------
# Format converters
# ---------------------------------------------------------------------------
def to_openai(rec, system, include_reasoning):
    """Strict OpenAI fine-tune schema: only the `messages` key."""
    out = [{"role": "system", "content": system}]
    for m in rec["messages"]:
        out.append({"role": m["role"],
                    "content": message_content(m, include_reasoning)})
    return {"messages": out}


def to_sharegpt(rec, system, include_reasoning):
    convs = []
    for m in rec["messages"]:
        convs.append({"from": "human" if m["role"] == "user" else "gpt",
                      "value": message_content(m, include_reasoning)})
    return {
        "conversations": convs,
        "system": system,
        "id": rec["conversation_id"],
        "title": rec["title"],
        "tags": rec.get("tags", []),
        "scope": rec.get("scope"),
    }


def to_alpaca_pairs(rec, max_pairs, include_reasoning):
    """One record per adjacent (user -> assistant) turn pair."""
    title = rec["title"]
    ctx = "tags: %s" % ", ".join(rec.get("tags", []))
    msgs = rec["messages"]
    pairs = []
    for i in range(1, len(msgs)):
        u, a = msgs[i - 1], msgs[i]
        if u["role"] == "user" and a["role"] == "assistant":
            pairs.append({
                "instruction": title,
                "input": u["text"],
                "output": message_content(a, include_reasoning),
                "context": ctx,
                "conversation_id": rec["conversation_id"],
            })
            if max_pairs and len(pairs) >= max_pairs:
                break
    return pairs


HF_SCHEMA_FIELDS = ("conversation_id", "title", "tags", "scope",
                    "messages", "num_messages", "est_tokens", "split")


def hf_rows(records, split_name, include_reasoning):
    rows = []
    for rec in records:
        toks = 0
        msgs = []
        for m in rec["messages"]:
            content = message_content(m, include_reasoning)
            toks += estimate_tokens(content)
            msgs.append({"role": m["role"], "content": content})
        rows.append({
            "conversation_id": rec["conversation_id"],
            "title": rec["title"],
            "tags": list(rec.get("tags", [])),
            "scope": rec.get("scope"),
            "messages": msgs,
            "num_messages": len(msgs),
            "est_tokens": toks,
            "split": split_name,
        })
    return rows


def write_parquet(rows, path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    schema = pa.schema([
        ("conversation_id", pa.string()),
        ("title", pa.string()),
        ("tags", pa.list_(pa.string())),
        ("scope", pa.string()),
        ("messages", pa.list_(pa.struct([
            ("role", pa.string()), ("content", pa.string())]))),
        ("num_messages", pa.int32()),
        ("est_tokens", pa.int64()),
        ("split", pa.string()),
    ])
    cols = {f: [r[f] for r in rows] for f in HF_SCHEMA_FIELDS}
    table = pa.table(cols, schema=schema)
    pq.write_table(table, path, compression="snappy")
    return len(rows)


def record_stats(rec, msgs):
    toks = sum(estimate_tokens(m["text"]) for m in msgs if m.get("text"))
    return {"messages": len(msgs), "est_tokens": toks,
            "chars": sum(len(m["text"]) for m in msgs if m.get("text"))}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Build OpenAI/ShareGPT/Alpaca/parquet training formats "
                    "with a shared train/val split.")
    ap.add_argument("--transcripts", default="data/transcripts")
    ap.add_argument("--out-dir", default="data/training")
    ap.add_argument("--scope", choices=["all", "work"], default="all",
                    help="work = use transcripts_work_only.jsonl")
    ap.add_argument("--val-ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-turns", type=int, default=0,
                    help="truncate conversations to first N turns (0 = off)")
    ap.add_argument("--max-pairs-per-conv", type=int, default=20,
                    help="cap on alpaca pairs per conversation")
    ap.add_argument("--system-prompt", default=None,
                    help="override the per-conversation system prompt")
    ap.add_argument("--include-reasoning", action="store_true",
                    help="keep captured reasoning as <thinking> blocks in "
                         "assistant content (default: dropped)")
    ap.add_argument(
        "--unsafe-legacy-transcript-export",
        action="store_true",
        help="acknowledge that direct transcript export bypasses governed "
             "candidate admission (required for this legacy command)",
    )
    args = ap.parse_args(argv)

    if not args.unsafe_legacy_transcript_export:
        print(
            "ERROR: direct transcript export bypasses governed admission. "
            "Use reviewed, admitted training_db candidates for production, "
            "or pass --unsafe-legacy-transcript-export for an explicitly "
            "unsafe legacy experiment.",
            file=sys.stderr,
        )
        return 2

    src = ("transcripts_work_only.jsonl" if args.scope == "work"
           else "transcripts.jsonl")
    src_path = os.path.join(args.transcripts, src)
    if not os.path.exists(src_path):
        print("ERROR: %s not found - run convert_export_to_transcripts.py first"
              % src_path)
        return 1

    records = load_transcripts(src_path)
    for rec in records:
        msgs = messages_for_training(rec, args.include_reasoning)
        msgs = maybe_truncate(msgs, args.max_turns)
        rec["messages"] = trim_dangling(msgs)
    records = [r for r in records if len(r["messages"]) >= 2]

    splits = assign_splits(records, args.val_ratio, args.seed)
    by_split = {"train": [], "val": []}
    for r in records:
        by_split[splits[r["conversation_id"]]].append(r)

    os.makedirs(args.out_dir, exist_ok=True)
    for sub in ("openai", "sharegpt", "alpaca", "hf"):
        os.makedirs(os.path.join(args.out_dir, sub), exist_ok=True)

    with open(os.path.join(args.out_dir, "split_manifest.json"), "w",
              encoding="utf-8") as f:
        json.dump({"generated_at": now_iso(), "source": src_path,
                   "seed": args.seed, "val_ratio": args.val_ratio,
                   "n_conversations": len(records),
                   "splits": splits,
                   "governance": {
                       "source_mode": "legacy_transcript",
                       "unsafe_acknowledged": True,
                       "admission_filter": "none",
                   }}, f, ensure_ascii=False, indent=1)

    stats = {"generated_at": now_iso(), "source": src_path,
             "scope": args.scope, "seed": args.seed,
             "include_reasoning": args.include_reasoning,
             "governance": {
                 "source_mode": "legacy_transcript",
                 "unsafe_acknowledged": True,
                 "admission_filter": "none",
             },
             "max_turns": args.max_turns or None,
             "conversations": {"train": len(by_split["train"]),
                               "val": len(by_split["val"])},
             "formats": {}}

    # ---- openai / sharegpt: one record per conversation ----
    for fmt, builder in (
            ("openai", lambda r: to_openai(
                r, make_system_prompt(r, args.system_prompt),
                args.include_reasoning)),
            ("sharegpt", lambda r: to_sharegpt(
                r, make_system_prompt(r, args.system_prompt),
                args.include_reasoning))):
        fmt_stats = {}
        for split_name, recs in by_split.items():
            out = [builder(r) for r in recs]
            n = write_jsonl(os.path.join(args.out_dir, fmt,
                                         "%s.jsonl" % split_name), out)
            total_msgs = sum(len(r["messages"]) for r in recs)
            total_toks = sum(sum(estimate_tokens(message_content(
                                     m, args.include_reasoning))
                                 for m in r["messages"] if m.get("text"))
                             for r in recs)
            fmt_stats[split_name] = {"records": n, "messages": total_msgs,
                                     "est_tokens": total_toks}
        stats["formats"][fmt] = fmt_stats

    # ---- alpaca: one record per user->assistant pair ----
    alpaca_stats = {}
    for split_name, recs in by_split.items():
        pairs = []
        for r in recs:
            pairs.extend(to_alpaca_pairs(
                r, args.max_pairs_per_conv, args.include_reasoning))
        n = write_jsonl(os.path.join(args.out_dir, "alpaca",
                                     "%s.jsonl" % split_name), pairs)
        alpaca_stats[split_name] = {
            "records": n,
            "est_tokens": sum(estimate_tokens(p["input"]) +
                              estimate_tokens(p["output"]) for p in pairs),
        }
    stats["formats"]["alpaca"] = alpaca_stats

    # ---- hf parquet ----
    hf_stats = {}
    try:
        for split_name, recs in by_split.items():
            rows = hf_rows(recs, split_name, args.include_reasoning)
            path = os.path.join(args.out_dir, "hf",
                                "%s-00000.parquet" % split_name)
            n = write_parquet(rows, path)
            hf_stats[split_name] = {
                "records": n,
                "messages": sum(r["num_messages"] for r in rows),
                "est_tokens": sum(r["est_tokens"] for r in rows),
                "file": os.path.relpath(path),
            }
    except ImportError:
        print("WARNING: pyarrow not installed - skipping parquet output")
        hf_stats = {"error": "pyarrow not installed"}
    stats["formats"]["hf"] = hf_stats

    with open(os.path.join(args.out_dir, "format_stats.json"), "w",
              encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # ---- console summary ----
    print("Source: %s (%s) | %d conversations | splits: train=%d val=%d"
          % (src_path, args.scope, len(records),
             len(by_split["train"]), len(by_split["val"])))
    for fmt, s in stats["formats"].items():
        if "error" in s:
            print("  %-9s SKIPPED (%s)" % (fmt, s["error"]))
            continue
        line = "  %-9s train: %s records" % (fmt, s["train"]["records"])
        if "val" in s:
            line += " | val: %s records" % s["val"]["records"]
        if "est_tokens" in s.get("train", {}):
            line += " | ~%d tok total" % (s["train"]["est_tokens"]
                                          + s["val"]["est_tokens"])
        print(line)
    print("Outputs written to %s" % args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


