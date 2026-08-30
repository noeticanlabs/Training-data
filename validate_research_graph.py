#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Contract check + integrity validator for the Research Graph v0.04.

Modes
  1. Contract self-consistency (always runs):
     - config/research_graph.json vocabularies match both graph schemas;
     - endpoint-class matrix references only declared node types;
     - id_rules cover every node type.
  2. Graph validation (runs when data/research_graph/candidate_*.jsonl exist):
     - schema conformance of every node and relation;
     - ID uniqueness and deterministic recomputation;
     - relation closure (all endpoint node_ids resolve);
     - endpoint-class matrix compliance;
     - lifecycle invariants (supersession symmetry, review evidence);
     - source_record_ids resolution against the Research DB.

Exit code 0 = passed. With no graph files present, mode 1 alone decides.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

from jsonschema import Draft202012Validator

WD = os.path.dirname(os.path.abspath(__file__))
FAILURES = []


def fail(msg):
    FAILURES.append(msg)
    if len(FAILURES) <= 50:
        print("  FAIL:", msg)


def check(cond, msg):
    if not cond:
        fail(msg)
    return cond


def sha256_text(text):
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


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


def node_canonical_key(rec):
    t = rec["node_type"]
    if t == "definition":
        return {"name": rec["name"], "statement": rec["statement"]}
    if t == "claim":
        return {"statement": rec["statement"]}
    if t == "argument":
        return {"conclusion": rec["conclusion"],
                "premises": sorted(rec.get("premises", []))}
    if t == "experiment":
        hyp = rec["hypothesis"]
        if isinstance(hyp, dict):
            hyp = hyp.get("claim_node_id", "")
        return {"hypothesis": hyp, "method": rec["method"]}
    if t == "failure":
        return {"failed_object": rec["failed_object"],
                "failure_mode": rec["failure_mode"]}
    raise ValueError("unknown node_type %r" % t)


def expected_node_id(rec):
    key = json.dumps(node_canonical_key(rec), sort_keys=True,
                     ensure_ascii=False)
    return "graph:%s:%s" % (rec["node_type"], sha256_text(key)[:24])


def expected_relation_id(rec):
    key = json.dumps({"source_node_id": rec["source_node_id"],
                      "relation_type": rec["relation_type"],
                      "target_node_id": rec["target_node_id"]},
                     sort_keys=True, ensure_ascii=False)
    return "relation:" + sha256_text(key)[:24]

def contract_check(config, node_schema, rel_schema):
    print("[1/3] contract self-consistency...")
    check(config["schema_version"]
          == node_schema["properties"]["schema_version"]["const"]
          == rel_schema["properties"]["schema_version"]["const"],
          "graph config and schema versions diverged")
    node_types = set(config["node_types"])
    check(set(node_schema["properties"]["node_type"]["enum"]) == node_types,
          "config node_types != node schema enum")
    check(set(rel_schema["properties"]["relation_type"]["enum"])
          == set(config["relation_types"]),
          "config relation_types != relation schema enum")
    for rt, spec in config["relation_types"].items():
        for side in ("sources", "targets"):
            for cls in spec[side]:
                check(cls in node_types,
                      "relation %s %s class %r not a node type" % (rt, side, cls))
    canonical_rules = config["id_rules"]["node"]["canonical_key"]
    check(set(canonical_rules) == node_types,
          "id_rules do not cover every node type")
    expected_rules = {
        "definition": ["name", "statement"],
        "claim": ["statement"],
        "argument": ["conclusion", "premises:sorted"],
        "experiment": ["hypothesis:stringified", "method"],
        "failure": ["failed_object", "failure_mode"],
    }
    check(canonical_rules == expected_rules,
          "config canonical keys diverge from semantic ID implementation")
    # epistemic_status enum in node schema matches the research record schema
    research_schema = load_json(os.path.join(WD, "schemas",
                                             "research_record.schema.json"))
    research_enum = set(research_schema["properties"]["epistemic_status"]["enum"])
    claim_enum = set(
        node_schema["allOf"][1]["then"]["properties"]["epistemic_status"]["enum"])
    check(claim_enum == research_enum,
          "claim epistemic_status enum diverged from Research DB registry")
    statement_enum = set(node_schema["properties"]["statement_basis"]["enum"])
    check(statement_enum == set(
        config["status_vocabularies"]["statement_basis"]),
        "statement_basis enum diverged from graph registry")
    print("  node types: %d | relation types: %d | enums aligned with Research DB"
          % (len(node_types), len(config["relation_types"])))


def load_research_records(path):
    if not os.path.exists(path):
        return None
    return {
        rec["record_id"]: (rec.get("content") or {}).get("text", "")
        for _, rec in stream_jsonl(path)
        if "record_id" in rec
    }

def validate_graph(args, node_validator, rel_validator, config,
                   research_records):
    print("[2/3] graph validation (candidate files present)...")
    nodes_path = os.path.join(args.graph_dir, "candidate_nodes.jsonl")
    rels_path = os.path.join(args.graph_dir, "candidate_relations.jsonl")
    matrix = config["relation_types"]
    node_types = {}
    nodes = {}

    nodes_exist = os.path.exists(nodes_path)
    rels_exist = os.path.exists(rels_path)
    check(nodes_exist == rels_exist,
          "candidate graph snapshot is incomplete; nodes and relations files "
          "must either both exist or both be absent")

    if nodes_exist:
        for n, rec in stream_jsonl(nodes_path):
            nid = rec.get("node_id")
            if nid in node_types:
                fail("duplicate node_id: %s" % nid)
            errors = list(node_validator.iter_errors(rec))
            if errors:
                fail("node schema violation line %d: %s"
                     % (n, errors[0].message[:200]))
                continue
            node_types[nid] = rec["node_type"]
            nodes[nid] = rec
            check(nid == expected_node_id(rec),
                  "node_id not deterministic: %s (expected %s)"
                  % (nid, expected_node_id(rec)))
            check(rec["lifecycle"] in config["lifecycle_states"],
                  "bad lifecycle on %s" % nid)
            check(not (rec["lifecycle"] == "superseded"
                       and not rec.get("superseded_by")),
                  "superseded node missing superseded_by: %s" % nid)
            review = rec.get("review") or {}
            if rec["lifecycle"] in {"reviewed", "superseded"}:
                check(review.get("status") == "approved",
                      "%s node %s is not approved"
                      % (rec["lifecycle"], nid))
            if rec["lifecycle"] == "rejected":
                check(review.get("status") == "rejected",
                      "rejected node %s lacks rejected review status" % nid)
            if review.get("status") == "approved":
                check(bool(review.get("reviewer")),
                      "approved node %s lacks reviewer" % nid)
                ev = review.get("evidence_record_ids") or []
                check(bool(ev), "approved node %s lacks evidence" % nid)
                if research_records is not None:
                    for e in ev:
                        check(e in research_records,
                              "node %s evidence %s unresolved" % (nid, e))
            if rec["node_type"] == "claim" and \
                    rec.get("epistemic_status") != "unknown":
                check(rec["lifecycle"] == "reviewed"
                      and review.get("status") == "approved"
                      and bool(review.get("evidence_record_ids")),
                      "status-bearing claim %s lacks reviewed evidence" % nid)
            if rec["node_type"] == "argument" and \
                    rec.get("proof_status") != "unknown":
                check(rec["lifecycle"] == "reviewed"
                      and review.get("status") == "approved"
                      and bool(review.get("evidence_record_ids")),
                      "status-bearing argument %s lacks reviewed evidence" % nid)
            if rec.get("statement_basis") == "normalized":
                check(bool(review.get("notes")),
                      "normalized statement %s lacks normalization notes" % nid)
            if research_records is not None:
                for s in rec.get("source_record_ids", []):
                    check(s in research_records,
                          "node %s source %s unresolved" % (nid, s))
                if rec.get("statement_basis") == "verbatim":
                    statement = rec.get("statement", "")
                    source_texts = [research_records.get(s, "")
                                    for s in rec["source_record_ids"]]
                    check(any(statement in text for text in source_texts),
                          "verbatim statement on %s is absent from its sources"
                          % nid)
        print("  nodes validated: %d" % len(node_types))
    else:
        print("  candidate_nodes.jsonl absent")

    supersedes_edges = set()
    if rels_exist:
        seen_rel = set()
        n_rel = 0
        for n, rec in stream_jsonl(rels_path):
            n_rel += 1
            rid = rec.get("relation_id")
            if rid in seen_rel:
                fail("duplicate relation_id: %s" % rid)
            seen_rel.add(rid)
            errors = list(rel_validator.iter_errors(rec))
            if errors:
                fail("relation schema violation line %d: %s"
                     % (n, errors[0].message[:200]))
                continue
            check(rid == expected_relation_id(rec),
                  "relation_id not deterministic: %s" % rid)
            src, tgt, rt = (rec["source_node_id"], rec["target_node_id"],
                            rec["relation_type"])
            check(src in node_types, "relation %s source unresolved" % rid)
            check(tgt in node_types, "relation %s target unresolved" % rid)
            if src in node_types and tgt in node_types:
                spec = matrix[rt]
                check(node_types[src] in spec["sources"],
                      "relation %s: %s cannot source %s"
                      % (rid, rt, node_types[src]))
                check(node_types[tgt] in spec["targets"],
                      "relation %s: %s cannot target %s"
                      % (rid, rt, node_types[tgt]))
            review = rec.get("review") or {}
            if rt == "supersedes" and review.get("status") == "approved":
                supersedes_edges.add((src, tgt))
            if review.get("status") == "approved":
                check(bool(review.get("reviewer")),
                      "approved relation %s lacks reviewer" % rid)
                evidence = review.get("evidence_record_ids") or []
                check(bool(evidence),
                      "approved relation %s lacks evidence" % rid)
                if research_records is not None:
                    for evidence_id in evidence:
                        check(evidence_id in research_records,
                              "relation %s evidence %s unresolved"
                              % (rid, evidence_id))
                if rt == "supersedes" and src in nodes:
                    source_review = (nodes[src].get("review") or {})
                    check(nodes[src].get("lifecycle") == "reviewed"
                          and source_review.get("status") == "approved",
                          "approved supersedes relation %s uses an unreviewed "
                          "replacement" % rid)
            if research_records is not None:
                for source_id in rec.get("source_record_ids", []):
                    check(source_id in research_records,
                          "relation %s source %s unresolved"
                          % (rid, source_id))
        print("  relations validated: %d" % n_rel)
    else:
        print("  candidate_relations.jsonl absent")

    for replacement, replaced in supersedes_edges:
        target = nodes.get(replaced)
        if target is not None:
            check(target.get("lifecycle") == "superseded",
                  "supersedes target %s is not lifecycle=superseded"
                  % replaced)
            check(target.get("superseded_by") == replacement,
                  "supersedes target %s does not point back to %s"
                  % (replaced, replacement))
    for nid, rec in nodes.items():
        if rec.get("lifecycle") == "superseded":
            replacement = rec.get("superseded_by")
            check(replacement in nodes,
                  "superseded node %s replacement is unresolved" % nid)
            check((replacement, nid) in supersedes_edges,
                  "superseded node %s lacks matching approved supersedes "
                  "relation" % nid)

def main(argv=None):
    FAILURES.clear()
    ap = argparse.ArgumentParser(
        description="Research Graph v0.04 contract check and validator.")
    ap.add_argument("--graph-dir", default="data/research_graph")
    ap.add_argument("--research-db",
                    default=os.path.join("data", "research_db",
                                         "records.jsonl"))
    args = ap.parse_args(argv)

    config = load_json(os.path.join(WD, "config", "research_graph.json"))
    node_schema = load_json(os.path.join(WD, "schemas",
                                         "research_graph_node.schema.json"))
    rel_schema = load_json(os.path.join(WD, "schemas",
                                        "research_graph_relation.schema.json"))

    contract_check(config, node_schema, rel_schema)

    if os.path.isdir(args.graph_dir):
        node_validator = Draft202012Validator(node_schema)
        rel_validator = Draft202012Validator(rel_schema)
        research_records = load_research_records(
            os.path.join(WD, args.research_db))
        if research_records is None:
            print("  note: Research DB not found; source resolution skipped")
        validate_graph(args, node_validator, rel_validator, config,
                       research_records)
    else:
        print("[2/3] graph files not present - contract check only "
              "(extraction not started)")

    print("[3/3] summary")
    if FAILURES:
        print("RESULT: FAILED (%d problems; first 50 shown above)"
              % len(FAILURES))
        return 1
    print("RESULT: GRAPH CONTRACT VALID")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
