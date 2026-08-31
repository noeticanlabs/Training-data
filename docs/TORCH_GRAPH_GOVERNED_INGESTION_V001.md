# Torch Executable Mathematical Graph — Governed Ingestion v0.01

**Status:** Implemented and locally source-validated

This layer ingests `Torch_Executable_Complete_Mathematical_Graph_v004_MultiZip_2026-08-31.zip` as an executable sidecar over the existing mathematical/research graph. It does not duplicate canonical mathematical objects and does not promote executable mappings into proofs or formula equivalences.

## Governing separation

```text
canonical mathematical/research objects
              |
              v
source executable graph v0.04
              |
              +--> operator software catalog
              +--> candidate node/operator correspondences
              +--> derived typed-edge sidecar
              +--> pending-review training candidates
              +--> GI raw-byte inputs + separately stored targets
```

Forbidden upgrades:

```text
callable exists => theorem proved
family-template mapping => formula equivalence
numerical implementation => physical validation
operator multi-hot => model input
source graph relation => canonical dependency
```

## Verified source integrity

Archive SHA-256:

`8ea91596a8c1bb1c9337c0372875032bc71a44018f893e3175134eb96d62b45f`

Observed:

- 27/27 primary internal SHA-256 entries match
- all 9 component ZIPs reproduce the root files exactly
- `.pt` loads successfully with `weights_only=True`
- 5,999 graph nodes
- 47,150 raw typed edges
- 47,103 unique typed triples
- 47 duplicate typed-edge extras
- 46,962 unique endpoint pairs
- 94 undirected connected components
- largest component 5,810 nodes
- 17 isolated nodes

## Executable mapping audit

Source report:

- formula-like nodes: 4,337
- nodes with any callable mapping: 3,170
- reported callable coverage: 73.09%
- exact / near-exact mappings: 263
- family-template mappings: 2,907
- symbolic/context-required: 1,167
- operators: 81

Governed refinement:

- generic-domain operators: 8
- generic-only mapped nodes: 664
- nodes with at least one non-generic operator: 2,506
- substantive-operator coverage: **57.78%**

The 73.09% figure is retained only as "has some callable mapping". It must not be quoted as substantive mathematical executability without this qualification.

Every mapping remains:

```text
canonical_formula_equivalence = false
training_admission = pending_review
```

The source also contains 506 formula-like records marked `fragment_or_low_information` or `mathematical_expression_requires_manual_review`; 9 appear in the source exact/near-exact category. There are 353 long/code/prose-like formula-like records, including 31 exact/near-exact mappings. These are review obligations, not clean automatic supervision.

## Navier–Stokes identity closure

Against the governed NSE formula archive:

- 263/263 `NSF` IDs match
- normalized formula strings match
- formula SHA-256 values match
- source status labels match

So the Torch graph links execution metadata to the same stable NSE identities rather than creating a second formula corpus.

## Tensor safety and target separation

The source package's `dataset.py` uses `torch.load(..., weights_only=False)`. The governed ingestion never uses that loader for the uploaded artifact. `safe_torch_math_graph_loader.py` uses:

```python
torch.load(..., weights_only=True, map_location="cpu")
```

GI/model inputs exclude:

- executable mask
- exact-operator mask
- operator multi-hot
- execution level
- operator IDs

These remain separate targets.

## Graph sidecar

Repeated identical typed triples are collapsed to one edge with `multiplicity`. All sidecar relations carry `canonical_relation=false`.

The 18,717 `formula_correlation` tensor edges are binary; their source weights are not present in the `.pt` dataset. Weighted NSE relation experiments must use the separately governed NSE correlation sidecar rather than invented weights.

## Training candidates

The live import produces:

- 4,337 execution-level candidates
- 4,337 operator-mapping admissibility/abstention candidates
- 3,170 operator-correspondence candidates
- **11,844 total**

All are `pending_review` with `split=unassigned`.

No benchmark split is invented because the complete graph archive does not provide a governed split.

## GI view

GI receives raw UTF-8 bytes for the formula-like label while operator/execution labels remain in a separate target file. This supports matched tests of expression recognition, operator-family association, abstention, and graph-assisted relation learning without injecting the answer into the encoder.

## Validation

Local full-source execution:

```text
ingestion: PASS
validator: PASS
deterministic rebuild: PASS
NSE identity closure: PASS
unit fixtures: 3/3 PASS
```

## Scientific boundary

PASS establishes archive integrity, safe tensor deserialization, identity closure, referentially consistent sidecars, deterministic generation, and target separation. It does not establish mathematical equivalence of every mapping, proof validity, physical correctness, numerical convergence of every operator, correctness of all graph relation labels, CohBit learning, or GI understanding.

## Open verification obligations

1. Review the 263 exact/near-exact mappings, prioritizing the 9 risky-status and 31 content-blob-like cases.
2. Build callable-level behavioral tests for all 81 operators; the source's 20 sanity tests do not cover every primitive.
3. Decide whether generic-only mappings belong in executable-mathematics training views or only structural parsing views.
4. Create leakage-resistant graph/task splits before any benchmark claim.
5. Preserve `symbolic_context_required` as explicit abstention/negative supervision rather than treating it as coverage debt to eliminate.
