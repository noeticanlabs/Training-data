#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, os, zipfile
from collections import Counter, defaultdict
from pathlib import Path
from jsonschema import Draft202012Validator

BASE="Navier_Stokes_Machine_Readable_Formula_Dataset/"
SOURCE_SUMS_PATH=BASE+"00_schema_manifest/SHA256SUMS.txt"


def sha256_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for c in iter(lambda:f.read(1<<20),b''): h.update(c)
    return h.hexdigest()

def load(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def lines(path):
    for n,l in enumerate(Path(path).read_text(encoding='utf-8').splitlines(),1):
        if l.strip():
            try: yield n,json.loads(l)
            except Exception as e: raise ValueError(f'{path}:{n}: {e}')
def req(cond,msg,errors):
    if not cond: errors.append(msg)

def rid_from_source(source_id): return f"research:formula:nse:{source_id}"

def validate_source_archive(path, errors):
    with zipfile.ZipFile(path) as z:
        for line in z.read(SOURCE_SUMS_PATH).decode().splitlines():
            if not line.strip(): continue
            expected, rel=line.split(None,1); rel=rel.strip()
            actual=hashlib.sha256(z.read(BASE+rel)).hexdigest()
            req(actual==expected,f'source checksum mismatch {rel}',errors)

def validate_schema_file(path,schema,errors):
    v=Draft202012Validator(schema); out=[]
    for n,obj in lines(path):
        for e in v.iter_errors(obj): errors.append(f'{path}:{n} schema {list(e.path)}: {e.message}')
        out.append(obj)
    return out

def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--archive',required=True);ap.add_argument('--out-dir',default='data/imports/navier_stokes_v001')
    ap.add_argument('--formula-schema',default='schemas/nse_formula_research_record.schema.json');ap.add_argument('--correlation-schema',default='schemas/nse_formula_correlation.schema.json');ap.add_argument('--gi-schema',default='schemas/nse_gi_sequence.schema.json');ap.add_argument('--config',default='config/nse_ingestion_v001.json');ap.add_argument('--report',default=None)
    a=ap.parse_args(argv); out=Path(a.out_dir); errors=[]
    config=load(a.config); relation_threshold=float(config['training']['relation_positive_threshold'])
    manifest=load(out/'manifest.json'); validate_source_archive(a.archive,errors)
    req(manifest['source_archive']['sha256']==sha256_file(a.archive),'archive sha mismatch',errors)
    for rel,meta in manifest['outputs'].items():
        p=out/rel; req(p.exists(),f'missing output {rel}',errors)
        if p.exists(): req(meta['sha256']==sha256_file(p),f'output digest mismatch {rel}',errors)

    formulas=validate_schema_file(out/'research/formulas.jsonl',load(a.formula_schema),errors)
    correlations=validate_schema_file(out/'research/correlations.jsonl',load(a.correlation_schema),errors)
    gi=validate_schema_file(out/'gi/formula_sequences.jsonl',load(a.gi_schema),errors)
    req(len(formulas)==263,f'formula count {len(formulas)} !=263',errors); req(len(correlations)==18717,f'correlation count {len(correlations)} !=18717',errors); req(len(gi)==263,f'GI sequence count {len(gi)} !=263',errors)
    f_by={x['research_formula_id']:x for x in formulas}; req(len(f_by)==len(formulas),'duplicate formula research ID',errors)
    source_ids={x['source_record_id'] for x in formulas}; req(len(source_ids)==len(formulas),'duplicate source record ID',errors)
    for f in formulas:
        req(hashlib.sha256(f['formula']['normalized'].encode()).hexdigest()==f['formula']['sha256'],f"{f['research_formula_id']} formula hash mismatch",errors)
        req(f['epistemic_status']=='unknown',f"{f['research_formula_id']} epistemic promotion",errors)
        req(f['training_policy']['default_admission']=='pending_review',f"{f['research_formula_id']} auto-admitted",errors)
        req(f['training_policy']['source_status_is_epistemic_status'] is False,f"{f['research_formula_id']} source status conflated",errors)
        req(f['training_policy']['clean_generation_target']==(not bool(f['quality_flags'])),f"{f['research_formula_id']} quality/target policy mismatch",errors)
    flagged=[f for f in formulas if f['quality_flags']]
    req(manifest['observed']['syntactic_quality_flag_count']==len(flagged),'manifest quality-flag count mismatch',errors)

    edge_pairs=set(); edge_by_source_id={}
    for c in correlations:
        req(c['source_formula_id'] in f_by and c['target_formula_id'] in f_by,f"{c['correlation_id']} dangling endpoint",errors)
        req(c['canonical_dependency'] is False,f"{c['correlation_id']} promoted to dependency",errors)
        req(c['heuristic']['weight_formula_verified'] is True,f"{c['correlation_id']} weight formula mismatch",errors)
        expected=(1.0 if c['heuristic']['shared_source_count']>0 else 0.0)+0.5*len(c['heuristic']['shared_families'])
        req(abs(expected-c['heuristic']['source_weight'])<1e-12,f"{c['correlation_id']} reconstructed weight mismatch",errors)
        req(c['cross_split']==(c['source_split']!=c['target_split']),f"{c['correlation_id']} cross_split mismatch",errors)
        edge_pairs.add(tuple(sorted((c['source_formula_id'],c['target_formula_id'])))); edge_by_source_id[c['source_edge_id']]=c
    req(sum(c['cross_split'] for c in correlations)==7021,'cross-split edge count mismatch',errors)

    gi_by={x['source_formula_id']:x for x in gi}; req(set(gi_by)==set(f_by),'GI sequence closure mismatch',errors)
    for fid,g in gi_by.items():
        req(g['split']==f_by[fid]['source_split'],f'{fid} GI split mismatch',errors)
        req(g['input_symbol_tokens']==f_by[fid]['formula']['symbol_tokens'],f'{fid} GI token mismatch',errors)
        req(g['encoder_contract']['labels_are_not_input'] is True and g['encoder_contract']['semantic_solver_in_encoder_forbidden'] is True,f'{fid} GI encoder contract violated',errors)

    # Training candidate invariants.
    candidates=[obj for _,obj in lines(out/'training/candidates.jsonl')]
    tids=[x['training_example_id'] for x in candidates]; req(len(tids)==len(set(tids)),'duplicate training candidate ID',errors)
    subtype=Counter(x.get('metadata',{}).get('task_subtype') for x in candidates)
    expected_completion=sum(1 for f in formulas if f['formula']['token_count']>=4 and not f['quality_flags'])
    positive_rel=sum(1 for c in correlations if (not c['cross_split']) and c['heuristic']['source_weight']>=relation_threshold)
    expected_subtypes={'formula_primary_family_classification':len(formulas),'source_formula_status_classification':len(formulas),'source_proof_critical_detection':len(formulas),'formula_symbol_completion':expected_completion,'heuristic_formula_correlation_prediction':2*positive_rel}
    expected_total=sum(expected_subtypes.values())
    req(len(candidates)==expected_total,f'training candidate count {len(candidates)} !={expected_total}',errors)
    req(dict(subtype)==expected_subtypes,f'training subtype counts mismatch {dict(subtype)}',errors)
    for x in candidates:
        req(x['admission']['status']=='pending_review',f"{x['training_example_id']} auto-admitted",errors)
        st=x.get('metadata',{}).get('task_subtype')
        src=x['source_record_ids']
        req(all(s in f_by for s in src),f"{x['training_example_id']} missing formula lineage",errors)
        splits={f_by[s]['source_split'] for s in src}; req(splits=={x['split']},f"{x['training_example_id']} split leakage",errors)
        if st=='source_proof_critical_detection':
            structured=x['input']['structured']; req(set(structured)=={'formula'},f"{x['training_example_id']} proof-critical label leakage",errors)
        elif st=='formula_symbol_completion':
            req(not f_by[src[0]]['quality_flags'],f"{x['training_example_id']} completion uses flagged formula",errors)
        elif st=='heuristic_formula_correlation_prediction':
            pair=tuple(sorted(src)); label=x['target']['structured']['heuristic_correlated']
            if label:
                seid=x['metadata'].get('source_edge_id'); req(seid in edge_by_source_id,f"{x['training_example_id']} missing positive edge",errors)
                if seid in edge_by_source_id: req(edge_by_source_id[seid]['heuristic']['source_weight']>=relation_threshold,f"{x['training_example_id']} positive below threshold",errors)
            else:
                req(pair not in edge_pairs,f"{x['training_example_id']} negative exists in source graph",errors)

    # GI graph views: no cross-split edges and view semantics hold.
    predicates={
        'full_same_split':lambda x:True,
        'shared_source':lambda x:x['shared_source_count']>0,
        'weight_ge_1_5':lambda x:x['weight']>=1.5,
        'weight_ge_2_0':lambda x:x['weight']>=2.0,
    }
    expected_counts={'full_same_split':{'train':11206,'validation':124,'test':366},'shared_source':{'train':4680,'validation':59,'test':191},'weight_ge_1_5':{'train':2166,'validation':19,'test':67},'weight_ge_2_0':{'train':320,'validation':2,'test':7}}
    for view,pred in predicates.items():
        for split in ('train','validation','test'):
            rows=[o for _,o in lines(out/f'gi/graphs/{view}/{split}.jsonl')]
            req(len(rows)==expected_counts[view][split],f'{view}/{split} count mismatch',errors)
            for e in rows:
                req(e['split']==split,f'{view}/{split} split mismatch',errors); req(e['canonical_dependency'] is False,f'{view}/{split} dependency promotion',errors); req(pred(e),f'{view}/{split} predicate violation',errors)
                req(f_by[e['source_id']]['source_split']==split and f_by[e['target_id']]['source_split']==split,f'{view}/{split} cross-split endpoint',errors)

    comp=load(out/'evaluation/provenance_components.json'); req(comp['component_sizes']==[250,13],f"provenance component sizes {comp['component_sizes']} != [250,13]",errors)
    comp_ids=[rid for c in comp['components'] for rid in c['research_formula_ids']]; req(set(comp_ids)==set(f_by) and len(comp_ids)==len(f_by),'provenance component closure mismatch',errors)
    report={'status':'PASS' if not errors else 'FAIL','error_count':len(errors),'errors':errors[:100],'observed':{'formula_records':len(formulas),'correlations':len(correlations),'training_candidates':len(candidates),'quality_flagged_formulas':len(flagged),'candidate_subtypes':dict(subtype),'provenance_component_sizes':comp['component_sizes']}}
    text=json.dumps(report,indent=2,sort_keys=True); print(text)
    if a.report: Path(a.report).write_text(text+'\n',encoding='utf-8')
    return 0 if not errors else 1
if __name__=='__main__': raise SystemExit(main())
