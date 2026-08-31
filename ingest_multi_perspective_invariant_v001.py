#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, io, json, os, zipfile
from pathlib import Path
from collections import Counter

PIPELINE_VERSION="MULTI-PERSPECTIVE-INVARIANT-GOVERNED-INGESTION-V001"
ROOT="Multi_Perspective_Invariant_Graph_v009/"

def sha256_bytes(b:bytes)->str: return hashlib.sha256(b).hexdigest()
def sha256_file(p:str)->str:
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for c in iter(lambda:f.read(1024*1024),b""): h.update(c)
    return h.hexdigest()
def read_json(z,n): return json.loads(z.read(ROOT+n).decode("utf-8"))
def read_jsonl(z,n):
    return [json.loads(x) for x in z.read(ROOT+n).decode("utf-8").splitlines() if x.strip()]
def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with open(path,"w",encoding="utf-8",newline="\n") as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False,sort_keys=True,separators=(",",":"))+"\n")
def write_json(path,obj):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8",newline="\n")
def quality_flags(label):
    s=str(label or "")
    flags=[]
    if any(((ord(c)<32 and c not in "\n\r\t") or ord(c)==127) for c in s):
        flags.append("contains_control_bytes")
    if len(s)>2000: flags.append("label_gt_2000_chars")
    return flags
def verify_internal(z):
    lines=z.read(ROOT+"00_manifest_schema/SHA256SUMS.txt").decode().splitlines()
    mism=[]; checked=0
    for line in lines:
        if not line.strip(): continue
        expected, rel=line.split(maxsplit=1); rel=rel.strip().lstrip("*")
        got=sha256_bytes(z.read(ROOT+rel)); checked+=1
        if got!=expected: mism.append({"path":rel,"expected":expected,"actual":got})
    return checked,mism
def verify_components(z):
    root_files={Path(n).name:n for n in z.namelist() if n.startswith(ROOT) and not n.endswith("/")}
    checked=0; mism=[]
    for n in z.namelist():
        if not(n.startswith("component_zips/") and n.endswith(".zip")): continue
        c=zipfile.ZipFile(io.BytesIO(z.read(n)))
        for cn in c.namelist():
            base=Path(cn).name
            if base not in root_files:
                mism.append({"component":n,"file":cn,"reason":"no_root_match"}); continue
            checked+=1
            if c.read(cn)!=z.read(root_files[base]):
                mism.append({"component":n,"file":cn,"reason":"byte_mismatch"})
    return checked,mism

def main(argv=None):
    ap=argparse.ArgumentParser()
    ap.add_argument("archive")
    ap.add_argument("--out",default="data/imports/multi_perspective_invariant_v009")
    args=ap.parse_args(argv)
    archive_sha=sha256_file(args.archive)
    z=zipfile.ZipFile(args.archive)
    checked,mism=verify_internal(z); comp_checked,comp_mism=verify_components(z)
    if mism or comp_mism: raise SystemExit("source checksum/component replication failure")

    manifest=read_json(z,"00_manifest_schema/manifest.json")
    perspectives=read_json(z,"01_invariant_ontology/perspectives.json")
    facets=read_json(z,"01_invariant_ontology/facet_invariants.json")
    bundles=read_jsonl(z,"02_node_bundles/node_invariant_bundles.jsonl")
    support_rows=read_jsonl(z,"02_node_bundles/node_invariant_support.jsonl")
    nodes=read_jsonl(z,"03_bundle_edges/augmented_nodes.jsonl")
    mp_edges=read_jsonl(z,"03_bundle_edges/multi_perspective_edges.jsonl")
    base_n=int(manifest["base_nodes"])
    base_nodes=nodes[:base_n]
    node_by_id={n["node_id"]:n for n in base_nodes}

    perspective_rows=[{
        "perspective_id":p["perspective_id"],"label":p["label"],
        "source_archive_sha256":archive_sha,
        "ontology_status":"source_defined",
        "canonical_satisfaction_claim":False
    } for p in perspectives]
    facet_rows=[{
        **f,"source_archive_sha256":archive_sha,
        "ontology_status":"source_defined",
        "canonical_satisfaction_claim":False
    } for f in facets]

    assignment_rows=[]; gi_inputs=[]; gi_targets=[]; candidates=[]
    for b in bundles:
        n=node_by_id[b["node_id"]]
        flags=quality_flags(n.get("label"))
        invs=[x["invariant_id"] for x in b.get("invariant_bundle",[])]
        conf={x["invariant_id"]:x["confidence"] for x in b.get("invariant_bundle",[])}
        assignment_rows.append({
            "node_id":b["node_id"],"node_type":n.get("node_type"),
            "source_bundle_signature":b["bundle_signature"],
            "perspectives":b["perspectives"],"invariant_ids":invs,
            "invariant_count":b["invariant_count"],"perspective_count":b["perspective_count"],
            "invariant_entries":b.get("invariant_bundle",[]),
            "quality_flags":flags,"assignment_semantics":"relevance_or_requirement",
            "canonical_satisfaction_claim":False,"review_status":"candidate",
            "source_archive_sha256":archive_sha
        })
        raw=(str(n.get("label") or "")).encode("utf-8")
        gi_inputs.append({
            "node_id":b["node_id"],"encoding":"utf8_bytes",
            "input_bytes":list(raw),"input_sha256":sha256_bytes(raw),
            "quality_flags":flags
        })
        gi_targets.append({
            "node_id":b["node_id"],"perspective_ids":b["perspectives"],
            "invariant_ids":invs,"relation_confidence":conf,
            "target_semantics":"relevance_or_requirement_not_satisfaction"
        })
        eligible=not flags
        common={"source_node_id":b["node_id"],"input_text":str(n.get("label") or ""),
                "admission_status":"pending_review","split":"unassigned",
                "quality_flags":flags,"training_eligible_candidate":eligible,
                "source_archive_sha256":archive_sha}
        candidates.append({**common,"task":"perspective_membership_prediction",
                           "target":{"perspective_ids":b["perspectives"]}})
        candidates.append({**common,"task":"invariant_relevance_prediction",
                           "target":{"invariant_ids":invs}})

    support_out=[]
    for s in support_rows:
        for item in s.get("support",[]):
            st=item["support_status"]
            support_out.append({
                "node_id":s["node_id"],"invariant_id":item["invariant_id"],
                "support_status":st,"basis":item.get("basis",[]),"scope":item.get("scope"),
                "independent_validation_claim":False if st=="SOURCE_DECLARED" else None,
                "canonical_global_satisfaction_claim":False,
                "source_archive_sha256":archive_sha
            })
    relevance_edges=[{
        "edge_id":e["edge_id"],"source_id":e["source_id"],"target_id":e["target_id"],
        "edge_type":"carries_perspective_invariant",
        "role":e.get("role"),"confidence":e.get("confidence"),"basis":e.get("basis",[]),
        "relation_semantics":"relevance_or_requirement",
        "canonical_satisfaction_claim":False,
        "source_archive_sha256":archive_sha
    } for e in mp_edges]

    out=Path(args.out)
    files={
        "research/perspectives.jsonl":perspective_rows,
        "research/facet_invariants.jsonl":facet_rows,
        "research/node_invariant_assignments.jsonl":assignment_rows,
        "research/invariant_support.jsonl":support_out,
        "graph/relevance_edges.jsonl":relevance_edges,
        "training/candidates.jsonl":candidates,
        "gi/node_inputs.jsonl":gi_inputs,
        "gi/node_targets.jsonl":gi_targets
    }
    for rel,rows in files.items(): write_jsonl(out/rel,rows)
    output_hashes={rel:{"count":len(rows),"sha256":sha256_file(out/rel)} for rel,rows in files.items()}
    receipt={
        "receipt_version":"MULTI-PERSPECTIVE-INVARIANT-GOVERNED-INGESTION-RECEIPT-V001",
        "pipeline_version":PIPELINE_VERSION,"status":"BUILT",
        "source_archive":{"filename":Path(args.archive).name,"sha256":archive_sha,
            "internal_checksums":{"checked_count":checked,"mismatches":mism,"status":"PASS"},
            "component_zip_replication":{"checked_file_copies":comp_checked,"mismatches":comp_mism,"status":"PASS"}},
        "observed":{"base_nodes":len(bundles),"perspectives":len(perspectives),"facet_invariants":len(facets),
            "relevance_edges":len(relevance_edges),"support_entries":len(support_out),
            "training_candidates":len(candidates),"gi_inputs":len(gi_inputs),
            "quality_flagged_inputs":sum(bool(x["quality_flags"]) for x in gi_inputs)},
        "output_hashes":output_hashes,
        "scientific_boundary":"Build preserves invariant relevance/requirements and scoped support. It does not assert invariant satisfaction, proof validity, physical validity, empirical validation, or model/GI learning."
    }
    write_json(out/"manifest.json",receipt)
    print(json.dumps(receipt,indent=2,sort_keys=True))
    return 0
if __name__=="__main__": raise SystemExit(main())
