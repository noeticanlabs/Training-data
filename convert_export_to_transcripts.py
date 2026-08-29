#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert a ChatGPT/OpenAI account export into clean, tagged training transcripts.

Built for the 'Training-data' repo: ~1,384 conversations spanning a year of R&D work
(Coherence/Coh ecosystem, GR/NR solvers, Noetica/Trinity agent architecture, Lean
formalization, physics research, manuscripts/reviews, data/ML) plus personal chat.

INPUT  (--raw-dir)       conversations-000..013.json  (JSON arrays of conversations)
                         codex.json                   (optional, via --with-codex)
OUTPUT (--out-dir)       transcripts.jsonl               all cleaned/tagged convos
                         transcripts_work_only.jsonl     scope == "work"
                         stats_report.json               pipeline statistics
                         sample_review.jsonl             seeded random sample
                         pii_report.json                 only with --pii-report
       (--index-dir)     conversations.json              header-level index
                         projects.json                   per-topic rollups

RECORD SCHEMA (one JSON object per line)
{
  "conversation_id": "...", "title": "...",
  "create_time": 123.0, "update_time": 456.0,
  "is_archived": false, "is_starred": null, "is_do_not_remember": false,
  "source": "chatgpt",
  "tags": ["gr-nr", "coherence"], "scope": "work",
  "messages": [
    {"role": "user", "content_type": "text", "text": "...", "images": 0,
     "create_time": 123.0, "message_id": "..."},
    {"role": "assistant", "content_type": "text", "text": "...",
     "reasoning": "...", "model_slug": "gpt-5-1-thinking",
     "create_time": 124.0, "message_id": "..."}
  ]
}

Notes
- Canonical turn order = walk `parent` links from `current_node` back to the mapping
  root (verified: all 1,384 conversations in this export resolve cleanly). Stale edit
  branches are automatically dropped.
- Assistant `thoughts` / `reasoning_recap` nodes are captured in the `reasoning` field
  of the assistant text message they precede. Drop that field for strict text.
- Multimodal image parts become an "[image]" placeholder and are counted in `images`.
- Only the output directories are written; raw data stays untouched.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import json
import os
import random
import re
import time

# ---------------------------------------------------------------------------
# Tag vocabulary (grounded in this corpus)
# ---------------------------------------------------------------------------
TAGS = [
    dict(slug="gr-nr", label="GR/NR solvers & relativity", work=True,
         words=["gr/nr", "gr-nr", "gr_nr", "general relativity", "numerical relativity",
                "einstein", "spacetime", "space-time", "gravitational wave", "waveform",
                "gw190521", "ringdown", "black hole", "neutron star", "schwarzschild",
                "kerr", "bssn", "adms", "christoffel", "riemann", "geodesic", "horizon",
                "lapse", "shift vector", "curvature", "relativity", "cosmolog"],
         codes=["nsc"]),
    dict(slug="coherence", label="Coherence / Coh ecosystem", work=True,
         words=["coherence", "coh-mhd", "cohbit", "cohbit-ctrl", "canonical action",
                "magnetohydro", "transition contract", "micro receipt", "reject code",
                "microv", "aeonic", "governance", "governed", "admissible", "audit",
                "nllc", "epoch", "clock"],
         codes=["coh", "ctt", "mhd", "nbc"]),
    dict(slug="noetica", label="Noetica agents & platform", work=True,
         words=["noetica", "noetican", "trinity", "arcadia", "harmonia", "fusioncore",
                "multi-core mind", "dna_v", "phaseloom", "caelus", "gm-os", "master boot",
                "interpreter", "glyph", "agent dna"],
         codes=["acl", "gm-os", "ufe", "gmi", "rfe", "npe"]),
    dict(slug="lean-math", label="Lean & math formalization", work=True,
         words=["mathlib", "theorem", "proof", "formaliz", "type theory", "dependent type",
                "inductive", "axiom", "verif", "nllc"],
         codes=["lean"]),
    dict(slug="numerics", label="Numerical methods & simulation", work=True,
         words=["solver", "simulation", "simulate", "discretiz", "manufactured solution",
                "convergence", "adaptive", "timestep", "dt sweep", "resolution sweep",
                "diagnostic", "stage jumps", "galerkin", "finite element", "finite volume",
                "jacobian", "residual", "truncation error", "cfl", "newton", "stiff",
                "implicit", "explicit", "rhs"],
         codes=["mms"]),
    dict(slug="physics", label="Physics / science research", work=True,
         words=["physics", "quantum", "field theory", "rft", "resonant", "cymatics",
                "mandala", "eden", "atomic", "nuclear", "nist", "optics",
                "electromagnetic", "frequency", "dispersion", "acoustics", "plasmon",
                "photon", "electron", "proton", "neutron", "entropy", "thermodynamics",
                "navier-stokes", "reynolds", "turbulence", "wave", "universe", "fluid"],
         codes=[]),
    dict(slug="data-ai", label="Data & AI/ML", work=True,
         words=["dataset", "training data", "fine-tun", "sft", "chatgpt", "chatbot",
                "embedding", "transformer", "corpus", "tokeniz", "model card",
                "benchmark", "evaluation", "pytorch", "tensorflow", "jax",
                "huggingface", "rag", "retrieval", "vector database", "semantic",
                "neural net", "neural network", "machine learning", "deep learning",
                "model training", "prompt"],
         codes=["gpt", "llm"]),
    dict(slug="papers", label="Manuscripts & reviews", work=True,
         words=["manuscript", "thesis", "referee", "journal", "publication", "publish",
                "preprint", "arxiv", "latex", "citation", "peer review", "submission",
                "author", "coauthor", "scholar", "academic", "review", "feedback",
                "comments"],
         codes=["prd", "prl"]),
    dict(slug="engineering", label="Software engineering", work=True,
         words=["repo", "repository", "github", "branch", "merge", "pull request",
                "commit", "pytest", "unit test", "test coverage", "config", "endpoint",
                "docker", "azure", "aws", "cloud", "deployment", "pipeline", "refactor",
                "lint", "security", "vulnerability", "dependency", "debug",
                "stack trace", "build", "code", "coding", "script", "module",
                "package"],
         codes=["git", "ci", "api", "cli"]),
    dict(slug="documentation", label="Docs / manuals / tutorials", work=True,
         words=["chapter", "rewrite", "editing", "editorial", "template",
                "procedures", "procedure", "policy", "handbook", "guide",
                "tutorial", "document", "documentation", "formatting", "style guide",
                "checklist", "workflow", "training manual", "brochure", "handout",
                "table of contents"],
         codes=["sop", "manual"]),
    dict(slug="creative", label="Creative / media", work=True,
         words=["image", "photo", "art", "music", "song", "story", "fiction",
                "character", "drawing", "illustration", "design", "brand", "logo",
                "video", "animation", "font", "color", "3d", "sketch", "aesthetic",
                "cinematic", "headshot", "portrait"],
         codes=[]),
    dict(slug="personal", label="Personal / life admin", work=False,
         words=["cover letter", "apartment", "landlord", "birthday", "gift",
                "wedding", "workout", "diet", "recipe", "mortgage", "greeting",
                "thanksgiving", "anniversary", "vacation"],
         codes=["cv", "rent", "resume", "crypto", "estate"]),
    dict(slug="math", label="Math help", work=True,
         words=["calculus", "integral", "derivation", "derivative",
                "differential equation", "laplace", "fourier", "linear algebra",
                "eigen", "series", "trigonom", "unit conversion", "algebra",
                "matrix", "limit", "math homework"],
         codes=[]),
]

WORK_SLUGS = {t["slug"] for t in TAGS if t["work"]}

# ---------------------------------------------------------------------------
# PII patterns (scan only; reports counts, never matched text)
# ---------------------------------------------------------------------------
PII_PATTERNS = {
    "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "phone_us": re.compile(r"(?<!\d)(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}(?!\d)"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
    "ipv4": re.compile(r"\b(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
                       r"(?:\.(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3}\b"),
    "eth_hex": re.compile(r"\b0x[a-fA-F0-9]{20,}\b"),
}

CHAR_PER_TOKEN = 3.8  # rough heuristic for mixed prose/code


def clean_text(s):
    """Normalize line endings, strip trailing spaces, collapse blank-line runs."""
    if not s:
        return ""
    t = str(s)
    if "\r" in t:
        t = t.replace("\r\n", "\n").replace("\r", "\n")
    if " \n" in t or "\t\n" in t:
        t = re.sub(r"[ \t]+\n", "\n", t)
    if "\n\n\n" in t:
        t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def parts_to_text(parts):
    """Join content parts into text; non-text (image) parts become [image] markers."""
    chunks = []
    images = 0
    for p in parts or []:
        if isinstance(p, str):
            c = clean_text(p)
            if c:
                chunks.append(c)
        else:
            images += 1
            chunks.append("[image]")
    return "\n\n".join(chunks), images


def estimate_tokens(text):
    return max(0, int(len(text) / CHAR_PER_TOKEN))
# ---------------------------------------------------------------------------
# Conversation reconstruction
# ---------------------------------------------------------------------------
def resolve_chain(mapping, current_node):
    """Ordered list of nodes along the current_node -> root parent chain, or None."""
    ordered = []
    seen = set()
    node_id = current_node
    while node_id is not None:
        if node_id in seen:
            return None  # loop
        seen.add(node_id)
        node = mapping.get(node_id)
        if node is None:
            return None
        ordered.append(node)
        node_id = node.get("parent")
    ordered.reverse()
    return ordered or None


def extract_entry(node):
    """Single mapping node -> normalized message dict, or None."""
    msg = node.get("message")
    if not isinstance(msg, dict):
        return None
    author = msg.get("author") or {}
    role = author.get("role")
    if role not in ("user", "assistant"):
        return None
    content = msg.get("content") or {}
    ctype = content.get("content_type") or "text"
    if ctype == "thoughts":
        # content.thoughts[] = list of {"content": str, "summary": str, ...}
        chunks = [clean_text(t.get("content")) for t in content.get("thoughts") or []
                  if isinstance(t, dict) and t.get("content")]
        text = "\n\n".join(c for c in chunks if c)
        images = 0
    elif ctype == "reasoning_recap":
        # content.content = short recap string
        text = clean_text(content.get("content") or "")
        images = 0
    else:
        parts = content.get("parts") or []
        text, images = parts_to_text(parts)
    md = msg.get("metadata") or {}
    return {
        "role": role,
        "content_type": ctype,
        "text": text,
        "images": images,
        "model_slug": md.get("model_slug") if role == "assistant" else None,
        "create_time": msg.get("create_time"),
        "message_id": msg.get("id") or node.get("id"),
    }


def build_messages(chain, drops):
    """Canonical chain -> (list of user/assistant messages, reasoning-msg counts)."""
    messages = []
    pending_reasoning = []
    reasoning_stats = collections.Counter()
    for node in chain:
        e = extract_entry(node)
        if e is None:
            continue
        role = e["role"]
        ctype = e["content_type"]
        create = e["create_time"] or 0.0
        if ctype in ("thoughts", "reasoning_recap"):
            if e["text"].strip():
                pending_reasoning.append(e["text"])
                reasoning_stats[ctype] += 1
            continue
        # multimodal / text user & assistant messages
        if not e["text"] and e["images"] == 0:
            drops[(role, ctype, "empty")] += 1
            continue
        if role == "user":
            messages.append({
                "role": "user",
                "content_type": ctype,
                "text": e["text"],
                "images": e["images"],
                "create_time": create,
                "message_id": e["message_id"],
            })
        else:  # assistant
            m = {
                "role": "assistant",
                "content_type": ctype,
                "text": e["text"],
                "images": e["images"],
                "model_slug": e["model_slug"],
                "create_time": create,
                "message_id": e["message_id"],
            }
            if pending_reasoning:
                m["reasoning"] = "\n\n".join(r for r in pending_reasoning
                                             if r and r.strip())
                pending_reasoning = []
            messages.append(m)
    if pending_reasoning:
        joined = "\n\n".join(r for r in pending_reasoning if r and r.strip())
        if joined:
            for m in reversed(messages):
                if m["role"] == "assistant":
                    prev = m.get("reasoning", "")
                    m["reasoning"] = (prev + ("\n\n" if prev else "") + joined)
                    break
    return merge_consecutive(messages), reasoning_stats


def merge_consecutive(messages):
    """Merge consecutive same-role messages (assistant->assistant, user->user).

    SFT/chat formats expect alternating user/assistant turns. Concatenate text
    (and reasoning for assistant) with a blank-line separator; keep the first
    message's metadata (earliest create_time, first message_id, model_slug).
    """
    merged = []
    for m in messages:
        if merged and m["role"] == merged[-1]["role"]:
            prev = merged[-1]
            prev["text"] = prev["text"].rstrip() + "\n\n" + m["text"].lstrip()
            prev["images"] = prev.get("images", 0) + m.get("images", 0)
            if m["role"] == "assistant":
                pr = prev.get("reasoning", "")
                mr = m.get("reasoning", "")
                if pr or mr:
                    prev["reasoning"] = (
                        pr.rstrip() + ("\n\n" if pr and mr else "") + mr.lstrip()
                    ) if (pr and mr) else (pr or mr)
        else:
            merged.append(m)
    return merged

# ---------------------------------------------------------------------------
# Tagging
# ---------------------------------------------------------------------------
def compute_tags(title, user_blob):
    """Score tags from the title (x2 evidence) + user message text (x1).

    A tag is assigned when it collects >= 2 evidence points, e.g. one title hit,
    two distinct user-text keyword hits, or a project-code identifier (always
    strong evidence). Scope: personal wins, else work if any work tag matched.
    """
    low_t = title.lower()
    low_u = user_blob.lower()
    matched = []
    has_personal = False
    for t in TAGS:
        hits_t = sum(1 for w in t["words"] if w in low_t)
        hits_u = sum(1 for w in t["words"] if w in low_u)
        code_hits = 0
        if t["codes"]:
            hay = low_t + "\n" + low_u
            code_hits = sum(1 for c in t["codes"]
                            if re.search(r"\b%s\b" % re.escape(c), hay))
        score = 2 * hits_t + hits_u + 2 * code_hits
        if score >= 2:
            matched.append(t["slug"])
            if t["slug"] == "personal":
                has_personal = True
    if not matched:
        matched = ["misc"]
    if has_personal:
        scope = "personal"
    elif any(s in WORK_SLUGS for s in matched):
        scope = "work"
    else:
        scope = "misc"
    return matched, scope


# ---------------------------------------------------------------------------
# PII scan (counts only)
# ---------------------------------------------------------------------------
def scan_pii(text):
    counts = collections.Counter()
    for label, pat in PII_PATTERNS.items():
        n = len(pat.findall(text))
        if n:
            counts[label] += n
    return counts


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def shard_files(raw_dir):
    return sorted(f for f in os.listdir(raw_dir)
                  if re.match(r"conversations-\d{3}\.json$", f))


def _process_shard(args):
    """Worker: load one shard and process its conversations (for ProcessPool)."""
    path, min_messages = args
    with open(path, encoding="utf-8") as f:
        convs = json.load(f)
    out = []
    for conv in convs:
        rec, st, reason, drops = process_conversation(conv, min_messages)
        out.append((rec, st, reason, dict(drops)))
    return out


def load_conversations(raw_dir):
    files = shard_files(raw_dir)
    convs = []
    for fn in files:
        with open(os.path.join(raw_dir, fn), encoding="utf-8") as f:
            convs.extend(json.load(f))
    seen = set()
    uniq = []
    for c in convs:
        cid = c.get("conversation_id") or c.get("id")
        if cid in seen:
            continue
        seen.add(cid)
        uniq.append(c)
    return uniq, len(files)


def process_conversation(conv, min_messages):
    """Return (record, stats, None) or (None, None, drop_reason)."""
    mapping = conv.get("mapping") or {}
    chain = resolve_chain(mapping, conv.get("current_node"))
    if chain is None:  # fallback: chronological order of message nodes
        chain = sorted((n for n in mapping.values()
                        if (n.get("message") or {}).get("content")),
                       key=lambda n: (n.get("message") or {}).get("create_time") or 0.0)
    drops = collections.Counter()
    messages, rstats = build_messages(chain, drops)
    if len(messages) < min_messages:
        return None, None, "too_few_messages", drops
    if not any(m["role"] == "assistant" and m["text"].strip() for m in messages):
        return None, None, "no_assistant_text", drops
    title = (conv.get("title") or "Untitled conversation").strip()
    # Tag signal comes from the title + the user's own words (their intent), not
    # the assistant's long answers (which mention many topics).
    user_blob = "\n".join(m["text"] for m in messages if m["role"] == "user")
    tags, scope = compute_tags(title, user_blob)
    rec = {
        "conversation_id": conv.get("conversation_id") or conv.get("id"),
        "title": title,
        "create_time": conv.get("create_time"),
        "update_time": conv.get("update_time"),
        "is_archived": bool(conv.get("is_archived")),
        "is_starred": conv.get("is_starred"),
        "is_do_not_remember": bool(conv.get("is_do_not_remember")),
        "source": "chatgpt",
        "tags": tags,
        "scope": scope,
        "messages": messages,
    }
    user_msgs = sum(1 for m in messages if m["role"] == "user")
    asst_msgs = sum(1 for m in messages if m["role"] == "assistant")
    total_chars = sum(len(m["text"]) for m in messages if m["text"])
    images = sum(m.get("images", 0) for m in messages)
    models = sorted({m.get("model_slug") for m in messages
                     if m.get("model_slug")})
    times = [m["create_time"] for m in messages if m.get("create_time")]
    st = {
        "conversation_id": rec["conversation_id"],
        "title": rec["title"],
        "num_messages": len(messages),
        "num_user": user_msgs,
        "num_assistant": asst_msgs,
        "num_images": images,
        "num_with_reasoning": sum(1 for m in messages if m.get("reasoning")),
        "text_chars": total_chars,
        "est_tokens": sum(estimate_tokens(m["text"]) for m in messages if m["text"]),
        "model_slugs": models,
        "first_message_time": min(times) if times else None,
        "last_message_time": max(times) if times else None,
        "tags": tags,
        "scope": scope,
        "reasoning_breakdown": dict(rstats),
    }
    return rec, st, None, drops


# ---------------------------------------------------------------------------
# Codex extraction (best-effort; roles in this export are unreliable)
# ---------------------------------------------------------------------------
def extract_codex(path):
    with open(path, encoding="utf-8") as f:
        tasks = json.load(f)
    records = []
    for t in tasks:
        title = t.get("title") or "Untitled task"
        chunks = []
        for turn in t.get("turns") or []:
            role = turn.get("role")
            for item in turn.get("input_items") or []:
                for c in item.get("content") or []:
                    if isinstance(c, dict) and c.get("content_type") == "text" and c.get("text"):
                        txt = clean_text(c["text"])
                        if txt:
                            chunks.append({"role": role or "user", "text": txt})
            for item in turn.get("output_items") or []:
                for c in item.get("content") or []:
                    if isinstance(c, dict) and c.get("content_type") == "text" and c.get("text"):
                        txt = clean_text(c["text"])
                        if txt:
                            chunks.append({"role": role or "assistant", "text": txt})
        if chunks:
            records.append({"task_id": t.get("id"), "title": title,
                            "source": "codex", "chunks": chunks})
    return records


# ---------------------------------------------------------------------------
# Output writers / main
# ---------------------------------------------------------------------------
def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Convert a ChatGPT export into clean, tagged training transcripts.")
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--out-dir", default="data/transcripts")
    ap.add_argument("--index-dir", default="data/index")
    ap.add_argument("--min-messages", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4,
                    help="parallel worker processes for shard parsing (0 = auto)")
    ap.add_argument("--sample-size", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--with-codex", action="store_true",
                    help="also extract codex.json tasks into codex_documents.jsonl")
    ap.add_argument("--pii-report", action="store_true",
                    help="scan transcripts for PII and write pii_report.json (counts only)")
    args = ap.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.index_dir, exist_ok=True)

    files = shard_files(args.raw_dir)
    n_shards = len(files)
    records, stats = [], []
    dropped_reasons = collections.Counter()
    msg_drops = collections.Counter()
    reasoning_total = collections.Counter()
    model_counter = collections.Counter()
    tag_counter = collections.Counter()
    scope_counter = collections.Counter()
    images_total = 0
    msgs_with_reasoning = 0
    n_convs = 0

    workers = max(1, min(args.workers, len(files)))
    tasks = [(os.path.join(args.raw_dir, f), args.min_messages) for f in files]
    if workers == 1:
        batches = [_process_shard(t) for t in tasks]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
            batches = list(ex.map(_process_shard, tasks))
    seen_ids = set()
    for batch in batches:
        for rec, st, reason, drops in batch:
            n_convs += 1
            for k, v in drops.items():
                msg_drops[(k[0], k[2])] += v
            if reason is not None:
                dropped_reasons[reason] += 1
                continue
            cid = rec["conversation_id"]
            if cid in seen_ids:
                continue
            seen_ids.add(cid)
            records.append(rec)
            stats.append(st)
            reasoning_total.update(st["reasoning_breakdown"])
            model_counter.update(st["model_slugs"])
            for t in st["tags"]:
                tag_counter[t] += 1
            scope_counter[st["scope"]] += 1
            images_total += st["num_images"]
            msgs_with_reasoning += st["num_with_reasoning"]

    write_jsonl(os.path.join(args.out_dir, "transcripts.jsonl"), records)
    write_jsonl(os.path.join(args.out_dir, "transcripts_work_only.jsonl"),
                [r for r in records if r["scope"] == "work"])
    sample = random.Random(args.seed).sample(records,
                                             min(args.sample_size, len(records)))
    # ---- index: conversations.json (headers only) ----
    header_keys = ("conversation_id", "title", "num_messages", "num_user",
                   "num_assistant", "num_images", "text_chars", "est_tokens",
                   "model_slugs", "first_message_time", "last_message_time",
                   "tags", "scope")
    header_index = [{k: st[k] for k in header_keys} for st in stats]
    with open(os.path.join(args.index_dir, "conversations.json"), "w",
              encoding="utf-8") as f:
        json.dump({"generated_at": now_iso(), "source_shards": n_shards,
                   "count": len(header_index), "conversations": header_index},
                  f, ensure_ascii=False, indent=2)

    # ---- index: projects.json (per-topic rollups) ----
    label_map = {t["slug"]: t["label"] for t in TAGS}
    label_map["misc"] = "Misc / untagged"
    tag_projects = {}
    for slug in sorted(tag_counter, key=lambda s: -tag_counter[s]):
        ids = [st["conversation_id"] for st in stats if slug in st["tags"]]
        tag_projects[slug] = {
            "label": label_map.get(slug, slug),
            "conversations": len(ids),
            "conversation_ids": ids,
        }
    with open(os.path.join(args.index_dir, "projects.json"), "w",
              encoding="utf-8") as f:
        json.dump({"generated_at": now_iso(), "total_conversations": len(records),
                   "tags": tag_projects, "scope": dict(scope_counter),
                   "tag_labels": label_map}, f, ensure_ascii=False, indent=2)

    # ---- stats report ----
    total_msgs = sum(st["num_messages"] for st in stats)
    total_user = sum(st["num_user"] for st in stats)
    total_asst = sum(st["num_assistant"] for st in stats)
    total_chars = sum(st["text_chars"] for st in stats)
    total_tokens = sum(st["est_tokens"] for st in stats)
    stats_report = {
        "generator": "convert_export_to_transcripts.py",
        "generated_at": now_iso(),
        "source": {
            "shards_parsed": n_shards,
            "conversations_in_export": n_convs,
            "archive_note": "conversations-*.json were parsed from data/raw",
        },
        "filters": {"min_messages": args.min_messages,
                    "require_assistant_text": True,
                    "keep_reasoning_field": True},
        "kept": {
            "conversations": len(records),
            "by_scope": dict(scope_counter),
            "messages": {"total": total_msgs, "user": total_user,
                         "assistant": total_asst},
            "text_chars": total_chars,
            "est_tokens": total_tokens,
            "image_parts": images_total,
            "messages_with_reasoning": msgs_with_reasoning,
        },
        "reasoning_capture": dict(reasoning_total),
        "drops": {
            "conversations_by_reason": dict(dropped_reasons),
            "messages_by_kind": {"%s|%s" % k: v for k, v in sorted(msg_drops.items())},
        },
        "models": dict(model_counter.most_common()),
        "tags": dict(tag_counter.most_common()),
    }
    with open(os.path.join(args.out_dir, "stats_report.json"), "w",
              encoding="utf-8") as f:
        json.dump(stats_report, f, ensure_ascii=False, indent=2)

    # ---- optional PII report ----
    if args.pii_report:
        per_conv = {}
        totals = collections.Counter()
        for rec in records:
            blob = "\n".join(m["text"] for m in rec["messages"])
            for m in rec["messages"]:
                if m.get("reasoning"):
                    blob += "\n" + m["reasoning"]
            counts = scan_pii(blob)
            if counts:
                per_conv[rec["conversation_id"]] = dict(counts)
                totals.update(counts)
        with open(os.path.join(args.out_dir, "pii_report.json"), "w",
                  encoding="utf-8") as f:
            json.dump({"generated_at": now_iso(),
                       "note": "Counts only; no matched text is stored.",
                       "totals_by_category": dict(totals),
                       "conversations_with_hits": per_conv},
                      f, ensure_ascii=False, indent=2)

    # ---- optional codex documents ----
    if args.with_codex:
        codex_path = os.path.join(args.raw_dir, "codex.json")
        if os.path.exists(codex_path):
            codex_recs = extract_codex(codex_path)
            write_jsonl(os.path.join(args.out_dir, "codex_documents.jsonl"),
                        codex_recs)
            stats_report["codex"] = {"tasks_extracted": len(codex_recs)}
            with open(os.path.join(args.out_dir, "stats_report.json"), "w",
                      encoding="utf-8") as f:
                json.dump(stats_report, f, ensure_ascii=False, indent=2)

    # ---- console summary ----
    print("Parsed %d shards; %d conversations in export" % (n_shards, n_convs))
    print("Kept: %d conversations | dropped: %s" % (len(records),
                                                    dict(dropped_reasons)))
    print("Messages: %d (user %d / assistant %d)" % (total_msgs, total_user,
                                                     total_asst))
    print("Scope: %s" % dict(scope_counter))
    print("Tags: %s" % dict(tag_counter.most_common()))
    print("Estimated tokens: %d (%d chars)" % (total_tokens, total_chars))
    print("Outputs written to %s and %s" % (args.out_dir, args.index_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())