# Navier-Stokes Governed Ingestion v0.01

**Status:** Implemented and validated against the uploaded 2026-08-31 machine-readable formula archive.

## Purpose

Convert the Navier-Stokes machine-readable formula archive into governed, reproducible views for:

1. the Research Database;
2. the Training Database / CohBit-family consumers;
3. GI symbolic-sequence and relational experiments.

The source archive is preserved as evidence. The ingestion does not silently reinterpret source labels as stronger scientific claims.

## Observed limitation and minimal repair

The core Research DB v0.02 record schema does not contain a neutral `formula` record type. A formula is not necessarily a claim, theorem, proof, or definition.

Therefore v0.01 does **not** coerce formulas into an incorrect canonical type. It creates a source-specific Research DB extension partition with stable IDs:

```text
research:formula:nse:NSF00000
```

A later core-schema migration may promote this neutral formula object into the main Research DB after compatibility and validator updates.

## Source boundary

Source archive:

```text
Navier_Stokes_Machine_Readable_MultiZip_DAT_JSONL_Dataset_2026-08-31.zip
```

Verified archive SHA-256:

```text
b1c8042da505b0e237c5bfcc1003d5486cc1b343240a10e09102062046d7f13c
```

All 24 files listed in the archive's `SHA256SUMS.txt` verified exactly in the live ingestion run.

## Research outputs

```text
data/imports/navier_stokes_v001/
  research/
    formulas.jsonl
    correlations.jsonl
```

### Formula records

All 263 formulas are preserved with:

- raw and normalized formula;
- formula SHA-256;
- symbol tokens;
- source family labels;
- source status label;
- source proof-critical flag;
- exact source path/hash/line provenance;
- original train/validation/test split;
- syntactic quality flags;
- `epistemic_status = unknown`;
- `review.status = unreviewed`;
- `default_admission = pending_review`.

The source field `status` is explicitly stored as `source_classification.status`. It is **not** promoted to the canonical epistemic-status registry.

### Formula correlations

All 18,717 source edges are preserved as:

```text
candidate_formula_correlation
```

For every edge the source weight is independently reconstructed as:

```text
1.0 if the formulas share at least one source, else 0.0
+ 0.5 * number_of_shared_families
```

The relation carries:

```text
canonical_dependency = false
```

because source correlation is not proof, logical dependency, mathematical support, or theorem implication.

## Training candidates

Generated candidates remain `pending_review`.

Live source run:

| Task subtype | Candidates |
|---|---:|
| Formula -> primary family | 263 |
| Formula -> source status | 263 |
| Formula -> source proof-critical flag | 263 |
| Symbol-sequence completion | 222 |
| Heuristic formula-correlation prediction | 4,504 |
| **Total** | **5,515** |

### Leakage controls

For proof-critical classification, source `status` and family labels are deliberately excluded from the model input.

Relation prediction uses only same-split formula pairs. Positive examples use source heuristic weight >= 1.5. Negative pairs are sampled deterministically from same-split pairs absent from the **complete** source correlation graph.

Symbol-completion targets exclude syntactically incomplete formula records.

## Syntactic quality result

The governed ingestion identifies 10 dangling-relation formula variants. Five underlying fragments occur in alternate encoded variants, which explains why an earlier manual inspection identified only five distinct-looking fragments.

These records remain in Research DB history but are not used as clean formula-completion targets.

## GI views

GI receives symbolic source sequences plus labels kept separate from encoder input.

Encoder contract:

```text
input = input_symbol_tokens
labels_are_not_input = true
semantic_solver_in_encoder_forbidden = true
```

Split-isolated relational views:

| GI graph view | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| Full same-split | 11,206 | 124 | 366 | 11,696 |
| Shared-source | 4,680 | 59 | 191 | 4,930 |
| Weight >= 1.5 | 2,166 | 19 | 67 | 2,252 |
| Weight >= 2.0 | 320 | 2 | 7 | 329 |

No cross-split source edge is admitted into these GI graph views.

## Split and provenance boundary

The source formula split contains no repeated formula hashes across splits, but source provenance is not split-disjoint.

Observed:

```text
7,021 / 18,717 source edges cross split boundaries
12 / 15 source SHA-256 groups occur in multiple splits
```

The source-provenance graph contains two connected components:

```text
250 formulas
13 formulas
```

The 13-formula component is exported as an **auxiliary source-disjoint holdout candidate**. It is not assumed to be representative of the full corpus.

## Commands

```powershell
python ingest_nse_formula_dataset_v001.py `
  --archive <path-to-Navier-Stokes-zip> `
  --config config/nse_ingestion_v001.json

python validate_nse_ingestion_v001.py `
  --archive <path-to-Navier-Stokes-zip> `
  --config config/nse_ingestion_v001.json `
  --report data/imports/navier_stokes_v001/validation_report.json

python -m unittest -v test_nse_ingestion_v001.py
```

## Claim status

### Verified

- source archive SHA-256;
- 24 internal source checksums;
- 263 unique governed formula records;
- 18,717 correlation sidecars;
- exact correlation-weight reconstruction;
- source-lineage closure;
- split-isolated GI views;
- deterministic negative relation sampling;
- 5,515 governed training candidates;
- deterministic rebuild output hashes;
- exhaustive validator PASS on the uploaded archive.

### Not established

- mathematical truth of any formula;
- correctness of source family/status labels;
- source correlations as mathematical dependencies;
- model learning from these candidates;
- GI understanding or abstraction;
- representativeness of the 13-record provenance holdout.

## Next highest-value experiment

Use the exact same governed formula source for a matched first comparison:

```text
CohBit/CF-LM: formula -> primary-family classification
GI: symbol sequence -> family-associated state/basin
```

Hold source split and formula identities fixed. Change only the computational substrate and serialization.
