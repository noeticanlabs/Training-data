#!/usr/bin/env python3
from __future__ import annotations
import argparse, collections, hashlib, io, json, os, re, zipfile
from pathlib import Path
import torch

PIPELINE_VERSION="TORCH-GRAPH-GOVERNED-INGESTION-V001"
ROOT="Torch_Mathematical_Graph_v004/"
GENERIC_DOMAIN="generic"

def sha256_bytes(b: bytes)->str:
    return hashlib.sha256(b).hexdigest()
def sha256_file(path: str)->str:
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for c in iter(lambda:f.read(1<<20),b""): h.update(c)
    return h.hexdigest()
def write_json(path: Path, obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")
def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False,sort_keys=True)+"\n")
def read_json(z, rel): return json.loads(z.read(ROOT+rel).decode("utf-8"))
def read_jsonl(z, rel): return [json.loads(x) for x in z.read(ROOT+rel).decode("utf-8").splitlines() if x.strip()]

def verify_internal_checksums(z)->dict:
    text=z.read(ROOT+"00_manifest/SHA256SUMS.txt").decode("utf-8")
    checked=0; mismatches=[]
    for line in text.splitlines():
        if not line.strip(): continue
        digest, rel=line.split(None,1); rel=rel.strip()
        got=sha256_bytes(z.read(ROOT+rel)); checked+=1
        if got!=digest: mismatches.append({"path":rel,"expected":digest,"actual":got})
    return {"checked_count":checked,"mismatches":mismatches,"status":"PASS" if not mismatches else "FAIL"}

def verify_component_zips(z)->dict:
    mapping={
      "00_manifest":"00_manifest","01_package":"01_package","02_node_registry":"02_node_registry",
      "03_operator_catalog":"03_operator_catalog","04_execution_maps":"04_execution_maps",
      "05_tests":"05_tests","06_metrics":"06_metrics","07_review_queue":"07_review_queue",
      "08_reports":"08_reports"}
    checked=0; mismatches=[]
    for name in sorted(n for n in z.namelist() if n.startswith("component_zips/") and n.endswith(".zip")):
        m=re.search(r"_v004_(\d\d_[^.]*)\.zip$",name)
        if not m or m.group(1) not in mapping:
            mismatches.append({"component":name,"error":"unrecognized component name"}); continue
        sub=mapping[m.group(1)]
        with zipfile.ZipFile(io.BytesIO(z.read(name))) as cz:
            for inner in (x for x in cz.namelist() if not x.endswith("/")):
                root_name=ROOT+sub+"/"+inner
                checked+=1
                if root_name not in z.namelist():
                    mismatches.append({"component":name,"inner":inner,"error":"missing root counterpart"})
                elif cz.read(inner)!=z.read(root_name):
                    mismatches.append({"component":name,"inner":inner,"error":"content mismatch"})
    return {"checked_file_copies":checked,"mismatches":mismatches,"status":"PASS" if not mismatches else "FAIL"}

def content_blob_like(label:str)->bool:
    return (len(label)>500 or len(label.split())>40 or bool(re.search(
        r'(#!/usr/bin|def\s+\w+\(|import\s+\w+|class\s+\w+|```|"content"\s*:)',label)))

def aggregate_typed_edges(data)->list[dict]:
    ids=data["node_ids"]; types=data["edge_types"]
    ei=data["edge_index"].cpu(); et=data["edge_type_index"].cpu()
    counts=collections.Counter((int(ei[0,i]),int(ei[1,i]),int(et[i])) for i in range(ei.shape[1]))
    rows=[]
    for (a,b,t),mult in sorted(counts.items()):
        payload=f"{ids[a]}|{types[t]}|{ids[b]}"
        rows.append({
          "schema_version":"TORCH-GRAPH-EDGE-SIDECAR-V001",
          "edge_id":"torchedge:"+hashlib.sha256(payload.encode()).hexdigest()[:24],
          "source_node_id":ids[a],"target_node_id":ids[b],"source_relation_type":types[t],
          "multiplicity":mult,"canonical_relation":False,
          "formula_correlation_weight_preserved":False if types[t]=="formula_correlation" else None
        })
    return rows

def safe_load_tensor_bytes(b:bytes):
    return torch.load(io.BytesIO(b),weights_only=True,map_location="cpu")

def nse_identity_check(nse_archive:str|None, registry:list[dict])->dict:
    if not nse_archive:
        return {"status":"NOT_RUN","reason":"no NSE archive supplied"}
    nsf={r["node_id"]:r for r in registry if r.get("node_type")=="formula" and r.get("node_id","").startswith("NSF")}
    with zipfile.ZipFile(nse_archive) as z:
        name="Navier_Stokes_Machine_Readable_Formula_Dataset/01_records_jsonl/all.jsonl"
        recs=[json.loads(x) for x in z.read(name).decode("utf-8").splitlines() if x.strip()]
    prev={r["record_id"]:r for r in recs}
    mism=[]
    for nid,r in sorted(nsf.items()):
        p=prev.get(nid)
        if not p: mism.append({"node_id":nid,"error":"missing in NSE archive"}); continue
        if p.get("formula_normalized")!=r.get("label") or p.get("formula_sha256")!=r.get("formula_sha256"):
            mism.append({"node_id":nid,"error":"formula/hash mismatch"})
        if p.get("status")!=r.get("source_status"):
            mism.append({"node_id":nid,"error":"status mismatch"})
    if set(prev)!=set(nsf):
        mism.append({"error":"NSF identity set mismatch","torch_count":len(nsf),"nse_count":len(prev)})
    return {"checked_count":len(nsf),"mismatches":mism,"status":"PASS" if not mism else "FAIL"}

def main(argv=None):
    ap=argparse.ArgumentParser()
    ap.add_argument("archive")
    ap.add_argument("--output-dir",default="data/imports/torch_math_graph_v004")
    ap.add_argument("--nse-archive",default=None)
    args=ap.parse_args(argv)
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    archive_sha=sha256_file(args.archive)
    with zipfile.ZipFile(args.archive) as z:
        checks=verify_internal_checksums(z); comps=verify_component_zips(z)
        manifest=read_json(z,"00_manifest/manifest.json")
        registry=read_json(z,"02_node_registry/executable_node_registry.json")
        catalog=read_json(z,"03_operator_catalog/operator_catalog.json")
        n2o=read_json(z,"04_execution_maps/node_to_operators.json")
        o2n=read_json(z,"04_execution_maps/operator_to_nodes.json")
        pt_bytes=z.read(ROOT+"02_node_registry/torch_graph_dataset.pt")
        data=safe_load_tensor_bytes(pt_bytes)
        source_dataset_loader=z.read(ROOT+"01_package/torch_math_graph/dataset.py").decode("utf-8")
    op_by={o["operator_id"]:o for o in catalog}
    generic={oid for oid,o in op_by.items() if o.get("domain")==GENERIC_DOMAIN}
    mapping_rows=[]
    for r in registry:
        ops=list(r.get("operator_ids") or [])
        only_generic=bool(ops) and set(ops).issubset(generic)
        substantive=[x for x in ops if x not in generic]
        mapping_rows.append({
          "schema_version":"TORCH-GRAPH-EXECUTION-MAPPING-V001",
          "node_id":r["node_id"],"node_type":r["node_type"],"label":r["label"],
          "formula_sha256":r["formula_sha256"],"source_status":r.get("source_status"),
          "source_families":r.get("families") or [],"execution_level":r["execution_level"],
          "operator_ids":ops,"operator_count":len(ops),
          "operator_domains":sorted({op_by[o]["domain"] for o in ops}),
          "only_generic_mapping":only_generic,"substantive_operator_count":len(substantive),
          "content_blob_like":content_blob_like(r["label"]),
          "canonical_formula_equivalence":False,
          "training_admission":"pending_review"
        })
    operator_rows=[{
       "schema_version":"TORCH-GRAPH-OPERATOR-V001",**o,
       "software_primitive":True,"canonical_mathematical_claim":False,"training_admission":"pending_review"
    } for o in catalog]
    edge_rows=aggregate_typed_edges(data)
    gi_inputs=[]; gi_targets=[]; train=[]
    for r,m in zip(registry,mapping_rows):
        raw=list(r["label"].encode("utf-8"))
        gi_inputs.append({
          "schema_version":"TORCH-GRAPH-GI-INPUT-V001","node_id":r["node_id"],
          "node_type":r["node_type"],"raw_utf8_bytes":raw,
          "raw_utf8_sha256":sha256_bytes(bytes(raw)),
          "target_fields_excluded":["execution_level","operator_ids","executable_mask","exact_operator_mask","operator_multihot"]
        })
        gi_targets.append({
          "schema_version":"TORCH-GRAPH-GI-TARGET-V001","node_id":r["node_id"],
          "execution_level":r["execution_level"],"operator_ids":r["operator_ids"]
        })
        base={"schema_version":"TORCH-GRAPH-TRAINING-CANDIDATE-V001","node_id":r["node_id"],
              "input":{"node_type":r["node_type"],"label":r["label"]},
              "split":"unassigned","admission":"pending_review","source_status":r.get("source_status"),
              "content_blob_like":m["content_blob_like"]}
        train.append({**base,"candidate_task":"execution_level_classification",
                      "target":{"execution_level":r["execution_level"]}})
        train.append({**base,"candidate_task":"operator_mapping_admissibility",
                      "target":{"mapping_admissible":bool(r["operator_ids"])}})
        if r["operator_ids"]:
            train.append({**base,"candidate_task":"operator_correspondence_prediction",
                          "target":{"operator_ids":r["operator_ids"],"mapping_level":r["execution_level"]}})
    tensor_meta={
      "schema":data.get("schema"),"num_nodes":len(data["node_ids"]),
      "num_edges_raw":int(data["edge_index"].shape[1]),
      "num_edges_unique_typed":len(edge_rows),
      "num_unique_endpoint_pairs":len(set(zip(data["edge_index"][0].tolist(),data["edge_index"][1].tolist()))),
      "node_types":data["node_types"],"edge_types":data["edge_types"],"operator_ids":data["operator_ids"],
      "shapes":{k:list(v.shape) for k,v in data.items() if isinstance(v,torch.Tensor)}
    }
    write_jsonl(out/"research/operator_catalog.jsonl",operator_rows)
    write_jsonl(out/"research/node_execution_mappings.jsonl",mapping_rows)
    write_jsonl(out/"graph/typed_edges_deduplicated.jsonl",edge_rows)
    write_json(out/"graph/tensor_metadata.json",tensor_meta)
    write_jsonl(out/"training/candidates.jsonl",train)
    write_jsonl(out/"gi/node_inputs.jsonl",gi_inputs)
    write_jsonl(out/"gi/node_targets.jsonl",gi_targets)
    mapped=[m for m in mapping_rows if m["operator_ids"]]
    subst=[m for m in mapped if not m["only_generic_mapping"]]
    risky_status={"fragment_or_low_information","mathematical_expression_requires_manual_review"}
    risky=[m for m in mapping_rows if m["source_status"] in risky_status]
    nsecheck=nse_identity_check(args.nse_archive,registry)
    ei=data["edge_index"]; raw_triples=list(zip(ei[0].tolist(),ei[1].tolist(),data["edge_type_index"].tolist()))
    triple_counts=collections.Counter(raw_triples)
    parent=list(range(len(data["node_ids"])))
    def find(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]]; x=parent[x]
        return x
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b: parent[b]=a
    pairset=set((a,b) for a,b,_ in raw_triples)
    deg=collections.Counter()
    for a,b in pairset: union(a,b); deg[a]+=1;deg[b]+=1
    compsizes=sorted(collections.Counter(find(i) for i in range(len(parent))).values(),reverse=True)
    receipt={
      "receipt_version":"TORCH-GRAPH-GOVERNED-INGESTION-RECEIPT-V001",
      "pipeline_version":PIPELINE_VERSION,"status":"PASS" if checks["status"]==comps["status"]=="PASS" else "FAIL",
      "source_archive":{"filename":Path(args.archive).name,"sha256":archive_sha,
                        "internal_checksums":checks,"component_zip_replication":comps},
      "source_manifest":manifest,
      "security":{"source_dataset_loader_uses_weights_only_false":"weights_only=False" in source_dataset_loader,
                  "governed_ingestion_uses_weights_only_true":True},
      "observed":{
        "graph_nodes":len(data["node_ids"]),"graph_edges_raw":len(raw_triples),
        "graph_edges_unique_typed":len(triple_counts),
        "duplicate_typed_edge_extras":sum(c-1 for c in triple_counts.values()),
        "unique_endpoint_pairs":len(pairset),"graph_component_count":len(compsizes),
        "largest_component_size":compsizes[0] if compsizes else 0,
        "isolated_node_count":sum(1 for i in range(len(parent)) if deg[i]==0),
        "formula_like_nodes":len(mapping_rows),"mapped_nodes":len(mapped),
        "reported_executable_coverage":len(mapped)/len(mapping_rows) if mapping_rows else 0.0,
        "generic_operator_count":len(generic),"generic_only_mapped_nodes":sum(m["only_generic_mapping"] for m in mapped),
        "substantive_mapped_nodes":len(subst),
        "substantive_operator_coverage":len(subst)/len(mapping_rows) if mapping_rows else 0.0,
        "symbolic_context_required":sum(m["execution_level"]=="symbolic_context_required" for m in mapping_rows),
        "exact_or_near_exact_operator_match":sum(m["execution_level"]=="exact_or_near_exact_operator_match" for m in mapping_rows),
        "family_template_executable":sum(m["execution_level"]=="family_template_executable" for m in mapping_rows),
        "content_blob_like_count":sum(m["content_blob_like"] for m in mapping_rows),
        "content_blob_like_exact_count":sum(m["content_blob_like"] and m["execution_level"]=="exact_or_near_exact_operator_match" for m in mapping_rows),
        "risky_source_status_count":len(risky),
        "risky_source_status_exact_count":sum(m["execution_level"]=="exact_or_near_exact_operator_match" for m in risky),
        "operator_count":len(catalog),"training_candidate_count":len(train),"gi_input_count":len(gi_inputs)
      },
      "nse_identity_check":nsecheck,
      "output_hashes":{},
      "scientific_boundary":"PASS establishes archive integrity, safe tensor deserialization, referential consistency, and deterministic sidecar construction. It does not establish mathematical equivalence of operator mappings, proof validity, physical validation, or model/GI learning."
    }
    for rel in ["research/operator_catalog.jsonl","research/node_execution_mappings.jsonl","graph/typed_edges_deduplicated.jsonl","graph/tensor_metadata.json","training/candidates.jsonl","gi/node_inputs.jsonl","gi/node_targets.jsonl"]:
        p=out/rel; receipt["output_hashes"][rel]={"sha256":sha256_file(str(p)),"bytes":p.stat().st_size}
    write_json(out/"receipt.json",receipt)
    print(json.dumps(receipt,indent=2,sort_keys=True))
    return 0 if receipt["status"]=="PASS" and nsecheck["status"] in {"PASS","NOT_RUN"} else 1
if __name__=="__main__": raise SystemExit(main())
