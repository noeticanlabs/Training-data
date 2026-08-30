#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Functional self-test for validate_research_graph.py using synthetic graphs."""
import json, os, shutil, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import validate_research_graph as vrg

WD = os.path.dirname(os.path.abspath(__file__))
GOOD = os.path.join(WD, "_tmp_graph_good")
BAD = os.path.join(WD, "_tmp_graph_bad")

SRC = "research:message:689033de-fake-src-1"


def claim(statement, cid):
    rec = {
        "schema_version": "0.04",
        "node_type": "claim",
        "node_id": None,
        "source_record_ids": [SRC],
        "lifecycle": "candidate",
        "review": {"status": "unreviewed"},
        "statement": statement,
        "statement_basis": "verbatim",
        "epistemic_status": "unknown",
    }
    rec["node_id"] = vrg.expected_node_id(rec)
    return rec


def write_graph(dirname, nodes, relations):
    os.makedirs(dirname, exist_ok=True)
    with open(os.path.join(dirname, "candidate_nodes.jsonl"), "w",
              encoding="utf-8") as f:
        for r in nodes:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(dirname, "candidate_relations.jsonl"), "w",
              encoding="utf-8") as f:
        for r in relations:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def rel(rt, src, tgt):
    rec = {
        "schema_version": "0.04",
        "relation_id": None,
        "relation_type": rt,
        "source_node_id": src,
        "target_node_id": tgt,
        "source_record_ids": [SRC],
        "basis": "explicit",
        "confidence": None,
        "review": {"status": "pending"},
    }
    rec["relation_id"] = vrg.expected_relation_id(rec)
    return rec


def experiment(supports_id):
    rec = {
        "schema_version": "0.04",
        "node_type": "experiment",
        "node_id": None,
        "source_record_ids": [SRC],
        "lifecycle": "candidate",
        "review": {"status": "unreviewed"},
        "hypothesis": {"claim_node_id": supports_id},
        "method": "numerical simulation of the collapse",
        "parameters": {"grid": "64^3"},
        "result": "constraint violation remained bounded",
        "interpretation": "outcome consistent with the claim",
        "supports": [supports_id],
    }
    rec["node_id"] = vrg.expected_node_id(rec)
    return rec


def failure(failed_id):
    rec = {
        "schema_version": "0.04",
        "node_type": "failure",
        "node_id": None,
        "source_record_ids": [SRC],
        "lifecycle": "candidate",
        "review": {"status": "unreviewed"},
        "failed_object": failed_id,
        "failure_mode": "counterexample found",
        "counterexample": "n=3 configuration violates the bound",
        "violated_assumption": "positivity of the source term",
        "diagnosis": "claim omitted a sign condition",
        "resulting_repair": None,
    }
    rec["node_id"] = vrg.expected_node_id(rec)
    return rec


def main():
    shutil.rmtree(GOOD, ignore_errors=True)
    shutil.rmtree(BAD, ignore_errors=True)

    c1, c2 = claim("Bound holds for all admissible configurations.", "c1"), \
        claim("Bound holds under sign-positive sources.", "c2")
    exp1 = experiment(c1["node_id"])
    fail1 = failure(c2["node_id"])
    good_nodes = [c1, c2, exp1, fail1]
    good_rels = [rel("supports", exp1["node_id"], c1["node_id"]),
                 rel("falsifies", fail1["node_id"], c2["node_id"]),
                 rel("depends_on", c2["node_id"], c1["node_id"])]
    write_graph(GOOD, good_nodes, good_rels)
    rc_good = vrg.main(["--graph-dir", os.path.relpath(GOOD, WD),
                        "--research-db", "_nonexistent_"])
    print(">>> good graph exit code: %d (expect 0)\n" % rc_good)

    # broken graph: claim falsifies claim (matrix violation) + tampered node_id
    bad_rels = [rel("falsifies", c1["node_id"], c2["node_id"])]
    c3 = dict(c1)
    c3["node_id"] = "graph:claim:" + "0" * 24
    bad_nodes = [c1, c2, c3]
    write_graph(BAD, bad_nodes, bad_rels)
    rc_bad = vrg.main(["--graph-dir", os.path.relpath(BAD, WD),
                       "--research-db", "_nonexistent_"])
    print(">>> broken graph exit code: %d (expect 1)" % rc_bad)

    shutil.rmtree(GOOD, ignore_errors=True)
    shutil.rmtree(BAD, ignore_errors=True)
    return 0 if (rc_good == 0 and rc_bad == 1) else 1


if __name__ == "__main__":
    raise SystemExit(main())
