# Multi-Perspective Invariant Governed Ingestion v0.01

## Status

**Implemented and full-source validated.**

Source archive:
`Multi_Perspective_Invariant_Complete_Mathematical_Graph_v009_MultiZip_2026-08-31.zip`

SHA-256:
`d5421ec16b94250c0c2a90dfb15342cc98c285782ad0de4c580d26a6ad7e31e3`

## Purpose

Preserve v0.09 as a governed **multi-perspective invariant sidecar** over stable mathematical-graph node identities.

The source distinction is mandatory:

```text
invariant relevance / requirement
!= invariant support
!= invariant satisfaction
```

No ingestion output promotes a relevance mapping into a proof, physical law, empirical validation, or globally satisfied invariant.

## Source structure independently checked

- 6,787 base nodes
- 6,861 augmented nodes
- 102,739 augmented edges
- 8 perspectives
- 66 facet invariants
- 34,644 node-to-facet relevance mappings
- 1,814 unique bundle signatures
- 1,600 scoped/source support entries
- 22/22 internal checksums PASS
- 23/23 component-ZIP file copies match root files

Support states:

- SOURCE_DECLARED: 1,440
- PARTIAL_CHECK_PASS: 88
- SCOPED_INVARIANT_PASS: 72
- FAIL: 0 in this source

A missing FAIL state in this source is **not evidence that all mapped invariants pass**. Most mapped invariants have no support state at all.

## Tensor verification

`multi_perspective_invariant_tensor.pt` loads under:

```python
torch.load(path, map_location="cpu", weights_only=True)
```

The following were reconstructed independently from the JSON/JSONL source and matched exactly:

- relation-confidence tensor `[6787, 66]`
- perspective-membership tensor `[6787, 8]`
- support-state tensor `[6787, 66]`
- facet co-occurrence matrix
- perspective co-occurrence matrix

This verifies representation consistency only.

## Continuity with Torch executable graph v0.04

All 5,999 node IDs in the prior Torch tensor occur in the v0.09 base graph.

The 788 additional base nodes are evidence/verification infrastructure:

- 615 source-backed binding receipts
- 126 source-backed validator receipts
- 27 invariant validators
- 19 core invariants
- 1 source-specific validator

Thus v0.09 is best treated as an **augmentation over existing mathematical objects**, not as a duplicate corpus.

## Governed outputs

Generated under:

```text
data/imports/multi_perspective_invariant_v009/
```

Research sidecars:
- `research/perspectives.jsonl`
- `research/facet_invariants.jsonl`
- `research/node_invariant_assignments.jsonl`
- `research/invariant_support.jsonl`

Graph:
- `graph/relevance_edges.jsonl`

Training candidates:
- `training/candidates.jsonl`

GI:
- `gi/node_inputs.jsonl`
- `gi/node_targets.jsonl`

All generated training examples remain `pending_review`.
No benchmark split is invented.

## Training views

Two initial task views are emitted:

1. node label -> perspective membership
2. node label -> invariant relevance

These are classification/relation-learning tasks, not satisfaction/proof tasks.

16 input labels contain control-byte artifacts and are retained for research history but flagged as unsuitable for automatic training admission.

## GI boundary

GI input contains only the node label encoded as UTF-8 bytes plus provenance/quality metadata.

Targets are physically separate:

- perspective IDs
- invariant IDs
- relation-confidence labels

Support state is not part of the default GI target.

## Full-source governed build

- 6,787 node assignments
- 34,644 relevance edges
- 1,600 support entries
- 13,574 training candidates
- 6,787 GI inputs
- deterministic rebuild PASS
- governed validator PASS with 0 errors

## Scientific boundary

Validated:
- archive integrity
- component replication
- tensor-safe deserialization
- JSON/tensor equality
- stable node identity continuity
- deterministic sidecar generation
- referential closure
- relevance/support separation
- GI target separation

Not established:
- invariant satisfaction
- theorem/proof correctness
- physical correctness
- empirical validity
- correctness of every heuristic relevance mapping
- model learning
- GI abstraction or understanding
