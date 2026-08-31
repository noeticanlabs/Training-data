#!/usr/bin/env python3
from __future__ import annotations
import argparse, io, json, tempfile, zipfile
def main(argv=None):
    ap=argparse.ArgumentParser()
    ap.add_argument("archive")
    args=ap.parse_args(argv)
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch is required for tensor verification") from exc
    root="Multi_Perspective_Invariant_Graph_v009/"
    z=zipfile.ZipFile(args.archive)
    bundles=[json.loads(x) for x in z.read(root+"02_node_bundles/node_invariant_bundles.jsonl").decode().splitlines() if x.strip()]
    support=[json.loads(x) for x in z.read(root+"02_node_bundles/node_invariant_support.jsonl").decode().splitlines() if x.strip()]
    with tempfile.NamedTemporaryFile(suffix=".pt") as f:
        f.write(z.read(root+"08_tensor_layer/multi_perspective_invariant_tensor.pt")); f.flush()
        obj=torch.load(f.name,map_location="cpu",weights_only=True)
    ni={x:i for i,x in enumerate(obj["node_ids"])}
    ii={x:i for i,x in enumerate(obj["facet_invariant_ids"])}
    pi={x:i for i,x in enumerate(obj["perspective_ids"])}
    expected_r=torch.zeros_like(obj["relation_confidence"])
    expected_p=torch.zeros_like(obj["perspective_membership"])
    for b in bundles:
        i=ni[b["node_id"]]
        for inv in b["invariant_bundle"]:
            expected_r[i,ii[inv["invariant_id"]]]=obj["confidence_codes"][inv["confidence"]]
            expected_p[i,pi[inv["perspective"]]]=1
    expected_s=torch.zeros_like(obj["support_state"])
    for s in support:
        i=ni[s["node_id"]]
        for x in s["support"]:
            expected_s[i,ii[x["invariant_id"]]]=obj["support_codes"][x["support_status"]]
    rel_bin=(expected_r>0).to(torch.int64); p_bin=(expected_p>0).to(torch.int64)
    checks={
      "weights_only_load":"PASS",
      "relation_confidence_exact":bool(torch.equal(expected_r,obj["relation_confidence"])),
      "perspective_membership_exact":bool(torch.equal(expected_p,obj["perspective_membership"])),
      "support_state_exact":bool(torch.equal(expected_s,obj["support_state"])),
      "facet_cooccurrence_exact":bool(torch.equal(rel_bin.T@rel_bin,obj["facet_cooccurrence"])),
      "perspective_cooccurrence_exact":bool(torch.equal(p_bin.T@p_bin,obj["perspective_cooccurrence"]))
    }
    ok=all(v=="PASS" or v is True for v in checks.values())
    report={"status":"PASS" if ok else "FAIL","checks":checks,
            "shapes":{"relation_confidence":list(obj["relation_confidence"].shape),
                      "support_state":list(obj["support_state"].shape),
                      "perspective_membership":list(obj["perspective_membership"].shape)},
            "scientific_boundary":"Tensor equality verifies representation consistency only, not invariant satisfaction."}
    print(json.dumps(report,indent=2,sort_keys=True))
    return 0 if ok else 1
if __name__=="__main__": raise SystemExit(main())
