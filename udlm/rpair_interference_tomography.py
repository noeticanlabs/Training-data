from __future__ import annotations

"""Gradient interference tomography for failed UDLM developmental stages.

This module measures whether the failed-stage next-byte objective pushes shared
parameters in directions that conflict with gradients supporting already
certified stages. It is diagnostic only: it does not mutate or promote models.
"""

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from .model import UDLM, UDLMConfig
from .run_certified import _sample_batch, split_lesson, model_hash


def load_model(path: Path, device: torch.device) -> UDLM:
    payload = torch.load(path, map_location=device)
    model = UDLM(UDLMConfig(**payload["config"])).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def family(name: str) -> str:
    n = name.lower()
    if "readout" in n: return "readout"
    if "route" in n or ".q" in n or "q_" in n: return "routing_Q"
    if "rel" in n or ".r" in n or "r_" in n: return "relation_R"
    if "hist" in n or ".h" in n or "h_" in n: return "history_H"
    if "uptake" in n or "content_delta" in n or "message" in n: return "content_X_core"
    if "theta" in n: return "theta"
    return "shared_core"


def stage_grad(model: UDLM, raw: bytes, *, batches: int, batch_size: int, context: int,
               seed: int, device: torch.device):
    random.seed(seed); torch.manual_seed(seed)
    accum = {n: torch.zeros_like(p, dtype=torch.float32, device="cpu") for n,p in model.named_parameters() if p.requires_grad}
    used = 0
    model.train()
    for _ in range(batches):
        x,y = _sample_batch(raw, batch_size, context, device)
        logits,_ = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        model.zero_grad(set_to_none=True); loss.backward()
        for n,p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                accum[n] += p.grad.detach().float().cpu()
        used += 1
    for n in accum:
        accum[n] /= max(used,1)
    model.zero_grad(set_to_none=True); model.eval()
    return accum


def dot_stats(a: dict, b: dict):
    by_family = defaultdict(lambda: [0.0,0.0,0.0,0])
    total_dot=total_a2=total_b2=0.0
    tensors=[]
    for n in a:
        av=a[n].reshape(-1); bv=b[n].reshape(-1)
        dot=float(torch.dot(av,bv).item()); a2=float(torch.dot(av,av).item()); b2=float(torch.dot(bv,bv).item())
        denom=math.sqrt(max(a2*b2,1e-30)); cos=dot/denom if denom else 0.0
        fam=family(n); row=by_family[fam]; row[0]+=dot; row[1]+=a2; row[2]+=b2; row[3]+=1
        total_dot+=dot; total_a2+=a2; total_b2+=b2
        tensors.append({"name":n,"family":fam,"cosine":cos,"dot":dot,"a_norm":math.sqrt(a2),"b_norm":math.sqrt(b2),"conflict":dot<0})
    fam_out={}
    for fam,(dot,a2,b2,count) in sorted(by_family.items()):
        denom=math.sqrt(max(a2*b2,1e-30)); fam_out[fam]={"cosine":dot/denom if denom else 0.0,"dot":dot,"a_norm":math.sqrt(a2),"b_norm":math.sqrt(b2),"tensor_count":count,"conflict":dot<0}
    denom=math.sqrt(max(total_a2*total_b2,1e-30)); global_cos=total_dot/denom if denom else 0.0
    tensors.sort(key=lambda r:r["cosine"])
    return {"global_cosine":global_cos,"global_dot":total_dot,"global_conflict":total_dot<0,"families":fam_out,"most_conflicting_tensors":tensors[:12]}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("stage_root",type=Path); ap.add_argument("failed_stage")
    ap.add_argument("--context",type=int,default=128); ap.add_argument("--validation-fraction",type=float,default=0.20)
    ap.add_argument("--batches",type=int,default=8); ap.add_argument("--batch-size",type=int,default=4); ap.add_argument("--device",default="cpu")
    args=ap.parse_args(); device=torch.device(args.device)
    failed=args.stage_root/args.failed_stage; parent=load_model(failed/"parent.pt",device)
    _, failed_val = split_lesson((failed/"curriculum.txt").read_bytes(), args.validation_fraction)
    failed_acq,_ = split_lesson((failed/"curriculum.txt").read_bytes(), args.validation_fraction)
    grads={args.failed_stage: stage_grad(parent,failed_acq,batches=args.batches,batch_size=args.batch_size,context=args.context,seed=900,device=device)}
    priors=[]
    for d in sorted(args.stage_root.glob("C*"), key=lambda p:int(p.name[1:]) if p.name[1:].isdigit() else 999):
        if d.name==args.failed_stage: break
        rp=d/"report.json"; cp=d/"curriculum.txt"
        if not rp.exists() or not cp.exists(): continue
        rep=json.loads(rp.read_text())
        if rep.get("decision")!="ADMIT": continue
        acq,_=split_lesson(cp.read_bytes(),args.validation_fraction)
        grads[d.name]=stage_grad(parent,acq,batches=args.batches,batch_size=args.batch_size,context=args.context,seed=901+len(priors),device=device)
        priors.append(d.name)
    comparisons={p:dot_stats(grads[args.failed_stage],grads[p]) for p in priors}
    worst=min(comparisons.items(), key=lambda kv:kv[1]["global_cosine"], default=(None,None))
    family_conflicts=defaultdict(list)
    for p,c in comparisons.items():
        for fam,row in c["families"].items():
            if row["conflict"]: family_conflicts[fam].append({"stage":p,"cosine":row["cosine"],"dot":row["dot"]})
    diagnosis={
        "kind":"udlm-rpair-gradient-interference-tomography-v1",
        "failed_stage":args.failed_stage,
        "parent_model_hash":model_hash(parent),
        "measurement":"mean acquisition-gradient cosine at frozen certified parent; heldout bytes excluded",
        "comparisons":comparisons,
        "worst_global_conflict_stage":worst[0],
        "worst_global_cosine": None if worst[1] is None else worst[1]["global_cosine"],
        "families_with_conflicts":dict(family_conflicts),
        "interpretation":"negative dot/cosine means a local descent step for the failed stage tends to increase the compared certified-stage loss to first order; nonnegative values do not prove independence",
        "mutation_authority":"none"
    }
    out=failed/"rpair_interference_tomography.json"; out.write_text(json.dumps(diagnosis,indent=2,sort_keys=True),encoding="utf-8")
    print(json.dumps({"worst_stage":diagnosis["worst_global_conflict_stage"],"worst_cosine":diagnosis["worst_global_cosine"],"conflict_families":sorted(family_conflicts)},sort_keys=True))

if __name__=="__main__": main()
