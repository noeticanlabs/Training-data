#!/usr/bin/env python3
"""Governed ingestion for Navier-Stokes machine-readable formula dataset v1.

Preserves source annotations without promoting them to canonical epistemic truth.
Produces a source-specific Research DB extension partition, heuristic-correlation
sidecar, multi-task training candidates, split-isolated GI views, and manifests.
"""
from __future__ import annotations

import argparse, hashlib, itertools, json, os, re, zipfile
from collections import Counter, defaultdict
from pathlib import Path

PIPELINE_VERSION = "NSE-GOVERNED-INGESTION-V001"
BASE = "Navier_Stokes_Machine_Readable_Formula_Dataset/"
RECORDS_PATH = BASE + "01_records_jsonl/all.jsonl"
EDGES_PATH = BASE + "03_edges_jsonl/formula_edges.jsonl"
SOURCE_MANIFEST_PATH = BASE + "00_schema_manifest/manifest.json"
SOURCE_SUMS_PATH = BASE + "00_schema_manifest/SHA256SUMS.txt"


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_hash(obj) -> str:
    return sha256_bytes(json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())


def read_jsonl_bytes(data: bytes):
    for line_no, line in enumerate(data.decode("utf-8").splitlines(), 1):
        if line.strip():
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL line {line_no}: {exc}") from exc


def verify_internal_checksums(z: zipfile.ZipFile) -> dict:
    lines = z.read(SOURCE_SUMS_PATH).decode("utf-8").splitlines()
    checked = []
    mismatches = []
    for line in lines:
        if not line.strip():
            continue
        expected, rel = line.split(None, 1)
        rel = rel.strip()
        actual = sha256_bytes(z.read(BASE + rel))
        checked.append(rel)
        if actual != expected:
            mismatches.append({"path": rel, "expected": expected, "actual": actual})
    return {"checked_count": len(checked), "mismatches": mismatches, "status": "PASS" if not mismatches else "FAIL"}


_RELATION_END_RE = re.compile(r"(?:=|≤|≥|<|>|\\leq?|\\geq?|\\lesssim|\\gtrsim)\s*$")


def quality_flags(formula: str) -> list[str]:
    flags = []
    normalized = formula.strip()
    while normalized.endswith("\\"):
        normalized = normalized[:-1].rstrip()
    if _RELATION_END_RE.search(normalized):
        flags.append("dangling_relation")
    if not formula.strip():
        flags.append("empty_formula")
    return flags


def research_id(source_id: str) -> str:
    return f"research:formula:nse:{source_id}"


def corr_id(edge_id: str) -> str:
    return f"research:formula-correlation:nse:{edge_id}"


def candidate_id(kind: str, payload: dict) -> str:
    return f"train:nse:{kind}:{canonical_hash(payload)[:24]}"


def write_jsonl(path: Path, records) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for obj in records:
            f.write(json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
            count += 1
    return count, sha256_file(str(path))


def formula_research_records(records: list[dict], archive_sha: str, source_manifest: dict, domain_hints: list[str]):
    for r in records:
        flags = quality_flags(r["formula_normalized"])
        yield {
            "schema_version": "nse_formula_research.v0.01",
            "research_formula_id": research_id(r["record_id"]),
            "object_type": "formula",
            "source_record_id": r["record_id"],
            "formula": {
                "raw": r["formula_raw"],
                "normalized": r["formula_normalized"],
                "sha256": r["formula_sha256"],
                "symbol_tokens": r.get("symbol_tokens", []),
                "token_count": r.get("token_count", 0),
                "char_count": r.get("char_count", 0),
                "features": {
                    "contains_equality": bool(r.get("contains_equality")),
                    "contains_inequality": bool(r.get("contains_inequality")),
                    "contains_integral": bool(r.get("contains_integral")),
                    "contains_derivative": bool(r.get("contains_derivative")),
                    "contains_norm": bool(r.get("contains_norm")),
                },
            },
            "source_classification": {
                "families": r.get("families", []),
                "primary_family": r.get("primary_family"),
                "status": r.get("status"),
                "is_proof_critical": bool(r.get("is_proof_critical")),
                "basis": "source_dataset_annotation",
            },
            "source_split": r["split"],
            "provenance": {
                "source_archive_sha256": archive_sha,
                "source_dataset_schema": source_manifest.get("schema"),
                "source_dataset_created": source_manifest.get("created"),
                "occurrence_count": int(r.get("occurrence_count", 1)),
                "source_count": int(r.get("source_count", len(r.get("sources", [])))),
                "sources": r.get("sources", []),
            },
            "domain_hints": [{"domain": d, "basis": "source-specific governed ingestion recipe"} for d in domain_hints],
            "epistemic_status": "unknown",
            "review": {"status": "unreviewed", "notes": None},
            "quality_flags": flags,
            "training_policy": {
                "default_admission": "pending_review",
                "clean_generation_target": not flags,
                "source_status_is_epistemic_status": False,
            },
        }


def correlation_records(edges: list[dict], rec_by_id: dict[str, dict]):
    for e in edges:
        a, b = rec_by_id[e["source_id"]], rec_by_id[e["target_id"]]
        reconstructed = (1.0 if int(e.get("shared_source_count", 0)) > 0 else 0.0) + 0.5 * len(e.get("shared_families", []))
        yield {
            "schema_version": "nse_formula_correlation.v0.01",
            "correlation_id": corr_id(e["edge_id"]),
            "source_edge_id": e["edge_id"],
            "relation_type": "candidate_formula_correlation",
            "source_formula_id": research_id(e["source_id"]),
            "target_formula_id": research_id(e["target_id"]),
            "heuristic": {
                "shared_source_count": int(e.get("shared_source_count", 0)),
                "shared_families": e.get("shared_families", []),
                "source_weight": float(e["weight"]),
                "reconstructed_weight": reconstructed,
                "weight_formula_verified": abs(float(e["weight"]) - reconstructed) < 1e-12,
            },
            "source_split": a["split"],
            "target_split": b["split"],
            "cross_split": a["split"] != b["split"],
            "canonical_dependency": False,
            "review_status": "unreviewed",
            "notes": "Heuristic metadata correlation; not proof, support, or dependency evidence.",
        }


def base_candidate(task: str, subtype: str, r: dict, input_obj: dict, target_obj: dict, source_ids: list[str]):
    payload = {"subtype": subtype, "source": source_ids, "input": input_obj, "target": target_obj}
    return {
        "schema_version": "0.02",
        "training_example_id": candidate_id(subtype, payload),
        "task": task,
        "knowledge_domains": ["mathematics", "physics"],
        "source_record_ids": source_ids,
        "source_artifact_ids": [],
        "knowledge_status": "unknown",
        "admission": {"status": "pending_review", "reviewer": None, "reviewed_at": None, "reason": None, "recipe": PIPELINE_VERSION},
        "split": r["split"],
        "split_group": f"nse-formula:{r['record_id']}",
        "input": input_obj,
        "target": target_obj,
        "evaluation": {"method": "exact-label" if task == "question_answering" else None},
        "transform": {"pipeline_version": PIPELINE_VERSION, "recipe_version": "v0.01", "created_at": None, "content_hash": canonical_hash(payload)},
        "metadata": {"task_subtype": subtype, "source_dataset_record_id": r["record_id"], "source_status_is_epistemic_status": False},
    }


def single_formula_candidates(records: list[dict], completion_fraction: float = 0.60):
    for r in records:
        src = [research_id(r["record_id"])]
        formula_input = {"system": None, "prompt": r["formula_normalized"], "context": None, "messages": [], "structured": {"formula": r["formula_normalized"]}}
       yield base_candidate("question_answering", "formula_primary_family_classification", r, formula_input, {"text": r["primary_family"], "messages": [], "structured": {"primary_family": r["primary_family"]}}, src)
       yield base_candidate("question_answering", "source_formula_status_classification", r, formula_input, {"text": r["status"], "messages": [], "structured": {"source_status": r["status"]}}, src)
       yield base_candidate("question_answering", "source_proof_critical_detection", r, formula_input, {"text": "true" if r.get("is_proof_critical") else "false", "messages": [], "structured": {"is_proof_critical": bool(r.get("is_proof_critical")}}, src)

        toks = list(r.get("symbol_tokens", []))
        if len(toks) >= 4 and not quality_flags(r["formula_normalized"]):
            cut = max(1, min(len(toks)-1, int(len(toks) * completion_fraction)))
            inp, out = toks[:cut], toks[cut:]
            payload = {"prefix_tokens": inp, "suffix_tokens": out, "source": src}
            yield {
                "schema_version": "0.02",
                "training_example_id": candidate_id("formula_symbol_completion", payload),
                "task": "sequence_prediction",
                "knowledge_domains": ["mathematics", "physics"],
                "source_record_ids": src,
                "source_artifact_ids": [],
                "knowledge_status": "unknown",
                "admission": {"status": "pending_review", "reviewer": None, "reviewed_at": None, "reason": None, "recipe": PIPELINE_VERSION},
                "split": r["split"],
                "split_group": f"nse-formula:{r['record_id']}",
                "input": {"system": None, "prompt": None, "context": None, "messages": [], "structured": {"prefix_tokens": inp}},
                "target": {"text": None, "messages": [], "structured": {"suffix_tokens": out}},
                "evaluation": {"method": "exact-sequence"},
                "transform": {"pipeline_version": PIPELINE_VERSION, "recipe_version": "v0.01", "created_at": None, "content_hash": canonical_hash(payload)},
                "metadata": {"task_subtype": "formula_symbol_completion", "source_dataset_record_id": r["record_id"], "quality_flags": []},
            }


def relation_candidates(records: list[dict], edges: list[dict], threshold: float = 1.5):
    rec = {r["record_id"]: r for r in records}
    all_edges = {tuple(sorted((e["source_id"], e["target_id"]))) for e in edges}
    positives_by_split = defaultdict(list)
    for e in edges:
        a, b = rec[e["source_id"]], rec[e["target_id"]]
        if a["split"] == b["split"] and float(e["weight"]) >= threshold:
            positives_by_split[a["split"]].append(e)

    ids_by_split = defaultdict(list)
    for r in records:
        ids_by_split[r["split"]].append(r["record_id"])

    negatives_by_split = {}
    for split, pos in positives_by_split.items():
        candidates = []
        for a, b in itertools.combinations(sorted(ids_by_split[split]), 2):
            pair = (a, b)
            if pair not in all_edges:
                candidates.append((canonical_hash({"split":split,"a":a,"b":b}), a, b))
        candidates.sort()
        negatives_by_split[split] = [(a,b) for _,a,b in candidates[:len(pos)]]

    for split in ("train", "validation", "test"):
        for e in positives_by_split.get(split, []):
            a, b = rec[e["source_id"]], rec[e["target_id"]]
            payload = {"a":a["record_id"],"b":b["record_id"],"label":True,"threshold":threshold}
            yield {
                "schema_version":"0.02","training_example_id":candidate_id("heuristic_formula_correlation",payload),"task":"relation_prediction",
                "knowledge_domains":["mathematics","physics"],"source_record_ids":[research_id(a["record_id"]),research_id(b["record_id"])],"source_artifact_ids":[],"knowledge_status":"unknown",
                "admission":{"status":"pending_review","reviewer":None,"reviewed_at":None,"reason":"Heuristic correlation label; not mathematical dependency.","recipe":PIPELINE_VERSION},
                "split":split,"split_group":f"nse-relation:{split}:{min(a['record_id'],b['record_id'])}:{max(a['record_id'],b['record_id'])}",
                "input":{"system":None,"prompt":None,"context":None,"messages":[],"structured":{"formula_a":a["formula_normalized"],"formula_b":b["formula_normalized"]}},
                "target":{"text":"true","messages":[],"structured":{"heuristic_correlated":True}},
                "evaluation":{"method":"exact-label"},
                "transform":{"pipeline_version":PIPELINE_VERSION,"recipe_version":"v0.01","created_at":None,"content_hash":canonical_hash(payload)},
                "metadata":{"task_subtype":"heuristic_formula_correlation_prediction","correlation_threshold":threshold,"source_edge_id":e["edge_id"],"canonical_dependency":False}
            }
        for a_id, b_id in negatives_by_split.get(split, []):
            a,b=rec[a_id],rec[b_id]
            payload={"a":a_id,"b":b_id,"label":False,"threshold":threshold}
            yield {
                "schema_version":"0.02","training_example_id":candidate_id("heuristic_formula_correlation",payload),"task":"relation_prediction",
                "knowledge_domains":["mathematics","physics"],"source_record_ids":[research_id(a_id),research_id(b_id)],"source_artifact_ids":[],"knowledge_status":"unknown",
                "admission":{"status":"pending_review","reviewer":None,"reviewed_at":None,"reason":"Negative under source heuristic graph; not a claim of mathematical independence.","recipe":PIPELINE_VERSION},
                "split":split,"split_group":f"nse-relation:{split}:{a_id}:{b_id}",
                "input":{"system":None,"prompt":None,"context":None,"messages":[],"structured":{"formula_a":a["formula_normalized"],"formula_b":b["formula_normalized"]}},
                "target":{"text":"false","messages":[],"structured":{"heuristic_correlated":False}},
                "evaluation":{"method":"exact-label"},
                "transform":{"pipeline_version":PIPELINE_VERSION,"recipe_version":"v0.01","created_at":None,"content_hash":canonical_hash(payload)},
                "metadata":{"task_subtype":"heuristic_formula_correlation_prediction","correlation_threshold":threshold,"canonical_dependency":false,"negative_sampling":"same-split absent from full source correlation graph"}
            }


def gi_sequences(records: list[dict]):
    for r in records:
        yield {
            "schema_version":"nse_gi_sequence.v0.01",
            "sequence_id":f"gi:nse:formula:{r['record_id']}",
            "source_formula_id":research_id(r["record_id"]),
            "split":r["split"],
            "input_symbol_tokens":r.get("symbol_tokens",[]),
            "labels":{"primary_family":r.get("primary_family"),"source_status":r.get("status"),"is_proof_critical":bool(r.get("is_proof_critical"))},
            "encoder_contract":{"input_only":"input_symbol_tokens","labels_are_not_input":True,"semantic_solver_in_encoder_forbidden":True},
            "quality_flags":quality_flags(r["formula_normalized"]),
        }


def graph_view(edges: list[dict], rec: dict[str,dict], predicate):
    for e in edges:
        a,b=rec[e["source_id"]],rec[e["target_id"]]
        if a["split"] != b["split"] or not predicate(e):
            continue
        yield {
            "edge_id":e["edge_id"],"source_id":research_id(e["source_id"]),"target_id":research_id(e["target_id"]),"split":a["split"],
            "weight":float(e["weight"]),"shared_source_count":int(e.get("shared_source_count",0),"shared_families":e.get("shared_families",[]),
            "relation_semantics":"heuristic_formula_correlation","canonical_dependency":false,
        }


def provenance_components(records: list[dict]) -> list[list[str]]:
    parent={r["record_id"]:r["record_id"] for r in records}
    def find(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]];x=parent[x]
        return x
    def union(a,b):
        ra,rb=find(a),find(b)
        if ra!=rb: parent[rb]=ra
    by_source=defaultdict(list)
    for r in records:
        for s in r.get("sources",[]): by_source[s.get("sha256")].append(r["record_id"])
    for ids in by_source.values():
        if ids:
            root=ids[0]
            for x in ids[1:]: union(root,x)
    comps=defaultdict(list)
    for rid in parent: comps[find(rid)].append(rid)
    return sorted((sorted(v) for v in comps.values()), key=lambda x:(-len(x),x))


def main(argv=None):
    ap=argparse.ArgumentParser()
    ap.add_argument("--archive",required=True)
    ap.add_argument("--out-dir",default="data/imports/navier_stokes_v001")
    ap.add_argument("--config",default="config/nse_ingestion_v001.json")
    args=ap.parse_args(argv)
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    config=json.loads(Path(args.config).read_text(encoding="utf-8"))
    if config.get("pipeline_version") != PIPELINE_VERSION:
        raise SystemExit("config pipeline_version mismatch")
    relation_threshold=float(config["training"]["relation_positive_threshold"])
    completion_fraction=float(config["training"]["completion_prefix_fraction"])
    domain_hints=list(config.get("domain_hints") or [])
    archive_sha=sha256_file(args.archive)
    with zipfile.ZipFile(args.archive) as z:
        checksum_report=verify_internal_checksums(z)
        if checksum_report["status"]!="PASS": raise SystemExit("source archive checksum verification failed")
        source_manifest=json.loads(z.read(SOURCE_MANIFEST_PATH))
        records=list(read_jsonl_bytes(z.read(RECORDS_PATH)))
        edges=list(read_jsonl_bytes(z.read(EDGES_PATH)))
    rec={r["record_id"]:r for r in records}

    outputs={}
    def write(rel, iterable):
        count,digest=write_jsonl(out/rel,iterable);outputs[rel]={"count":count,"sha256":digest};return count

    write("research/formulas.jsonl",formula_research_records(records,archive_sha,source_manifest,domain_hints))
    write("research/correlations.jsonl",correlation_records(edges,rec))
    single=list(single_formula_candidates(records,completion_fraction))
    rel=list(relation_candidates(records,edges,relation_threshold))
    write("training/candidates.jsonl",iter(single+rel))
    write("gi/formula_sequences.jsonl",gi_sequences(records))
    views={
        "full_same_split":lambda e:True,
        "shared_source":lambda e:int(e.get("shared_source_count",0))>0,
        "weight_ge_1_5":lambda e:float(e["weight"])>=1.5,
        "weight_ge_2_0":lambda e:float(e["weight"])>=2.0,
    }
    for name,pred in views.items():
        for split in ("train","validation","test"):
            write(f"gi/graphs/{name}/{split}.jsonl",(x for x in graph_view(edges,rec,pred) if x["split"]==split))

    comps=provenance_components(records)
    eval_obj={
        "schema_version":"nse_provenance_holdout.v0.01",
        "component_count":len(comps),
        "component_sizes":[len(c) for c in comps],
        "components":[{"component_id":f"source-component-{i:02d}","size":len(c),"source_record_ids":c,"research_formula_ids":[research_id(x) for x in c]} for i,c in enumerate(comps)],
        "auxiliary_holdout_recommendation":({"component_id":"source-component-01","reason":"smaller source-disjoint provenance component; auxiliary evaluation only, not representative by construction"} if len(comps)>1 else None),
    }
    p=out/"evaluation/provenance_components.json";p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(eval_obj,indent=2,sort_keys=True)+"\n",encoding="utf-8");outputs["evaluation/provenance_components.json"]={"count":len(comps),"sha256":sha256_file(str(p))}

    dangling=sum(bool(quality_flags(r["formula_normalized"])) for r in records)
    cross=sum(rec[e["source_id"]]["split"]!=rec[e["target_id"]]["split"] for e in edges)
    source_shas=defaultdict(set)
    for r in records:
        for s in r.get("sources",[]): source_shas[s.get("sha256")].add(r["split"])
    shared_across_splits=sum(len(v)>1 for v in source_shas.values())
    manifest={
        "pipeline_version":PIPELINE_VERSION,"config":{"path":args.config,"sha256":sha256_file(args.config),"relation_positive_threshold":relation_threshold,"completion_prefix_fraction":completion_fraction,"domain_hints":domain_hints},"source_archive":{"filename":Path(args.archive).name,"sha256":archive_sha,"internal_checksums":checksum_report},
        "source_manifest":source_manifest,
        "governance":{"source_status_promoted_to_epistemic":False,"correlation_promoted_to_dependency":false,"automatic_training_admission":false,"cross_split_edges_in_gi_views":false},
        "observed":{"formula_records":len(records),"correlation_edges":len(edges),"cross_split_edge_count":cross,"cross_split_edge_rate":cross/len(edges),"syntactic_quality_flag_count":dangling,"source_sha256_count":len(source_shas),"source_sha256_crossing_splits":shared_across_splits,"provenance_component_sizes":[len(c) for c in comps]},
        "outputs":outputs,
    }
    mp=out/"manifest.json";mp.write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps({"status":"PASS","out_dir":str(out),"observed":manifest["observed"],"outputs":outputs},indent=2,sort_keys=True))
    return 0

if __name__=="__main__": raise SystemExit(main())
