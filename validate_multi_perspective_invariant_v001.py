#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, hashlib
from pathlib import Path
from collections import defaultdict

def sha256_file(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for c in iter(lambda:f.read(1024*1024),b""): h.update(c)
    return h.hexdigest()
def read_jsonl(p):
    rows=[]
    with open(p,encoding="utf-8") as f:
        for i,l in enumerate(f,1):
            if l.strip():
                v=json.loads(l)
                if not isinstance(v,dict): raise ValueError(f"{p}:{i} not object")
                rows.append(v)
    return rows
def req(cond,msg,errs):
    if not cond: errs.append(msg)
def main(argv=None):
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",default="data/imports/multi_perspective_invariant_v009")
    ap.add_argument("--report",default=None)
    args=ap.parse_args(argv); root=Path(args.out); errs=[]
    manifest=json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    paths={
      "perspectives":"research/perspectives.jsonl",
      "facets":"research/facet_invariants.jsonl",
      "assignments":"research/node_invariant_assignments.jsonl",
      "support":"research/invariant_support.jsonl",
      "edges":"graph/relevance_edges.jsonl",
      "training":"training/candidates.jsonl",
      "gi_inputs":"gi/node_inputs.jsonl",
      "gi_targets":"gi/node_targets.jsonl"
    }
    rows={k:read_jsonl(root/v) for k,v in paths.items()}
    req(len(rows["perspectives"])==8,"expected 8 perspectives",errs)
    req(len(rows["facets"])==66,"expected 66 facet invariants",errs)
    req(len(rows["assignments"])==6787,"expected 6787 assignments",errs)
    req(len(rows["edges"])==34644,"expected 34644 relevance edges",errs)
    req(len(rows["training"])==13574,"expected 13574 training candidates",errs)
    req(len(rows["gi_inputs"])==6787 and len(rows["gi_targets"])==6787,"GI closure failed",errs)
    inv_ids={r["invariant_id"] for r in rows["facets"]}
    persp_ids={r["perspective_id"] for r in rows["perspectives"]}
    assignments={r["node_id"]:r for r in rows["assignments"]}
    req(len(assignments)==6787,"duplicate assignment node ids",errs)
    for r in rows["assignments"]:
        req(r.get("canonical_satisfaction_claim") is False,f"{r['node_id']}: satisfaction must be false",errs)
        req(set(r["invariant_ids"]).issubset(inv_ids),f"{r['node_id']}: unknown invariant",errs)
        req(set(r["perspectives"]).issubset(persp_ids),f"{r['node_id']}: unknown perspective",errs)
        req(r["invariant_count"]==len(r["invariant_ids"]),f"{r['node_id']}: invariant count mismatch",errs)
        req(r["perspective_count"]==len(r["perspectives"]),f"{r['node_id']}: perspective count mismatch",errs)
    for s in rows["support"]:
        req(s["node_id"] in assignments,f"support missing node {s['node_id']}",errs)
        if s["node_id"] in assignments:
            req(s["invariant_id"] in assignments[s["node_id"]]["invariant_ids"],
                f"support invariant absent from assignment {s['node_id']}",errs)
        req(s.get("canonical_global_satisfaction_claim") is False,"support cannot assert global satisfaction",errs)
        if s["support_status"]=="SOURCE_DECLARED":
            req(s.get("independent_validation_claim") is False,"source declared cannot be independent validation",errs)
    for e in rows["edges"]:
        req(e["source_id"] in assignments,f"edge source missing {e['source_id']}",errs)
        req(e["target_id"] in inv_ids,f"edge target missing {e['target_id']}",errs)
        req(e.get("canonical_satisfaction_claim") is False,"edge satisfaction must be false",errs)
    input_map={r["node_id"]:r for r in rows["gi_inputs"]}
    target_map={r["node_id"]:r for r in rows["gi_targets"]}
    req(set(input_map)==set(assignments)==set(target_map),"GI ids do not close over assignments",errs)
    forbidden={"perspective_ids","invariant_ids","relation_confidence","support_status","target"}
    for nid,r in input_map.items():
        req(not forbidden.intersection(r),f"{nid}: target leakage into GI input",errs)
        raw=bytes(r["input_bytes"])
        req(hashlib.sha256(raw).hexdigest()==r["input_sha256"],f"{nid}: GI input hash mismatch",errs)
    for t in rows["training"]:
        req(t.get("admission_status")=="pending_review","automatic training admission forbidden",errs)
        req(t.get("split")=="unassigned","source has no benchmark split; split must be unassigned",errs)
        req(t["task"] in {"perspective_membership_prediction","invariant_relevance_prediction"},"unknown task",errs)
    for rel,meta in manifest["output_hashes"].items():
        p=root/rel
        req(p.exists(),f"missing output {rel}",errs)
        if p.exists():
            req(meta["sha256"]==sha256_file(p),f"hash mismatch {rel}",errs)
            req(meta["count"]==len(read_jsonl(p)),f"count mismatch {rel}",errs)
    report={"status":"PASS" if not errs else "FAIL","error_count":len(errs),"errors":errs[:100],
            "observed":{"assignments":len(rows["assignments"]),"support_entries":len(rows["support"]),
                        "relevance_edges":len(rows["edges"]),"training_candidates":len(rows["training"]),
                        "quality_flagged_inputs":sum(bool(x.get("quality_flags")) for x in rows["gi_inputs"])}}
    print(json.dumps(report,indent=2,sort_keys=True))
    if args.report:
        Path(args.report).parent.mkdir(parents=True,exist_ok=True)
        Path(args.report).write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    return 0 if not errs else 1
if __name__=="__main__": raise SystemExit(main())
