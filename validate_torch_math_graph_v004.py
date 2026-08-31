#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

FORBIDDEN_GI_INPUT_FIELDS={"execution_level","operator_ids","executable_mask","exact_operator_mask","operator_multihot"}

def iter_jsonl(path):
    with open(path,encoding="utf-8") as f:
        for n,line in enumerate(f,1):
            if line.strip():
                v=json.loads(line)
                if not isinstance(v,dict): raise ValueError(f"{path}:{n} not object")
                yield n,v
def sha256_file(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for c in iter(lambda:f.read(1<<20),b""):h.update(c)
    return h.hexdigest()
def req(cond,msg,errors):
    if not cond: errors.append(msg)
def main(argv=None):
    ap=argparse.ArgumentParser()
    ap.add_argument("output_dir")
    args=ap.parse_args(argv)
    root=Path(args.output_dir); errors=[]
    receipt=json.loads((root/"receipt.json").read_text(encoding="utf-8"))
    req(receipt.get("status")=="PASS","receipt status not PASS",errors)
    req(receipt.get("security",{}).get("governed_ingestion_uses_weights_only_true") is True,
        "governed loader safety declaration missing",errors)
    for rel,meta in receipt.get("output_hashes",{}).items():
        p=root/rel
        req(p.exists(),f"missing output {rel}",errors)
        if p.exists(): req(sha256_file(p)==meta.get("sha256"),f"hash mismatch {rel}",errors)
    ops=list(iter_jsonl(root/"research/operator_catalog.jsonl"))
    maps=list(iter_jsonl(root/"research/node_execution_mappings.jsonl"))
    edges=list(iter_jsonl(root/"graph/typed_edges_deduplicated.jsonl"))
    train=list(iter_jsonl(root/"training/candidates.jsonl"))
    gi=list(iter_jsonl(root/"gi/node_inputs.jsonl"))
    targets=list(iter_jsonl(root/"gi/node_targets.jsonl"))
    opids={x["operator_id"] for _,x in ops}
    nodeids={x["node_id"] for _,x in maps}
    req(len(opids)==len(ops),"duplicate operator IDs",errors)
    req(len(nodeids)==len(maps),"duplicate mapping node IDs",errors)
    for n,m in maps:
        req(set(m.get("operator_ids",[])).issubset(opids),f"mapping line {n} unknown operator",errors)
        req(m.get("training_admission")=="pending_review",f"mapping line {n} auto-admitted",errors)
        req(m.get("canonical_formula_equivalence") is False,f"mapping line {n} equivalence overclaim",errors)
        req(m.get("operator_count")==len(m.get("operator_ids",[])),f"mapping line {n} operator count",errors)
    edgeids=set()
    for n,e in edges:
        req(e["edge_id"] not in edgeids,f"duplicate edge id line {n}",errors);edgeids.add(e["edge_id"])
        req(e.get("canonical_relation") is False,f"edge line {n} canonical overclaim",errors)
        req(int(e.get("multiplicity",0))>=1,f"edge line {n} invalid multiplicity",errors)
        if e.get("source_relation_type")=="formula_correlation":
            req(e.get("formula_correlation_weight_preserved") is False,
                f"edge line {n} incorrectly claims correlation weight",errors)
    for n,t in train:
        req(t.get("admission")=="pending_review",f"training line {n} auto-admitted",errors)
        req(t.get("split")=="unassigned",f"training line {n} split should remain unassigned",errors)
    gi_ids=set()
    for n,g in gi:
        gi_ids.add(g["node_id"])
        req(not (FORBIDDEN_GI_INPUT_FIELDS & set(g)),f"GI line {n} target leakage",errors)
        req(set(g.get("target_fields_excluded") or [])==FORBIDDEN_GI_INPUT_FIELDS,
            f"GI line {n} excluded-field contract drift",errors)
        b=bytes(g.get("raw_utf8_bytes") or [])
        req(hashlib.sha256(b).hexdigest()==g.get("raw_utf8_sha256"),f"GI line {n} byte hash mismatch",errors)
    target_ids={t["node_id"] for _,t in targets}
    req(gi_ids==target_ids==nodeids,"GI input/target/mapping closure mismatch",errors)
    obs=receipt.get("observed",{})
    req(len(maps)==obs.get("formula_like_nodes"),"formula-like count mismatch",errors)
    req(len(ops)==obs.get("operator_count"),"operator count mismatch",errors)
    req(len(edges)==obs.get("graph_edges_unique_typed"),"edge count mismatch",errors)
    req(len(train)==obs.get("training_candidate_count"),"training candidate count mismatch",errors)
    req(len(gi)==obs.get("gi_input_count"),"GI input count mismatch",errors)
    report={"status":"PASS" if not errors else "FAIL","error_count":len(errors),"errors":errors[:100],
            "observed":{"operators":len(ops),"mappings":len(maps),"edges":len(edges),
                        "training_candidates":len(train),"gi_inputs":len(gi)}}
    print(json.dumps(report,indent=2,sort_keys=True))
    return 0 if not errors else 1
if __name__=="__main__": raise SystemExit(main())
