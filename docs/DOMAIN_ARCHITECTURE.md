# Domain Registry v0.01

**Status:** Defined and implemented candidate-classification layer

## Purpose

Break the Research Database and Training Database into scientifically useful domain views without creating separate sources of truth or duplicating canonical content.

The governing relation is:

```text
canonical research record
        |
        +--> candidate domain assignment
        |       +--> domain indexes
        |
        +--> training candidate
                +--> inherited candidate domain assignment
                        +--> training domain indexes
```

A domain view is an index over stable IDs. It is not another database copy.

## Canonical preservation

Domain Registry v0.01 does **not** rewrite:

- `data/research_db/records.jsonl`
- `data/training_db/candidates.jsonl`

The validated v0.02 hashes therefore remain meaningful. Domain assignments are generated as sidecars and can be reviewed, replaced, or regenerated independently.

Automatic classification has status `candidate`. It is never canonical merely because a rule matched.

## Domain registry

`config/domain_registry.json` defines stable domain identities.

| ID | Domain |
|---|---|
| D00 | Unclassified / Pending (system bucket, not a scientific domain) |
| D01 | Mathematics |
| D02 | Physics |
| D03 | Dynamical Systems and Control |
| D04 | Network and Complex Systems |
| D05 | Formal Methods and Theorem Proving |
| D06 | Computer Science and Software Systems |
| D07 | Artificial Intelligence and Machine Learning |
| D08 | Electrical and Power Systems |
| D09 | Fluid and Transport Systems |
| D10 | Infrastructure and Maintenance Engineering |
| D11 | Data and Knowledge Engineering |
| D12 | Research Methodology and Validation |
| D13 | Business and Commercialization |
| D14 | Education and Workforce Development |

Each substantive domain also defines an initial subdomain vocabulary. Subdomains are descriptive registry terms in v0.01; they are not yet independently assigned or promoted.

## Separation of dimensions

Domain is intentionally separate from:

- internal project identity;
- tags;
- research-object type;
- epistemic status;
- training task;
- admission status.

For example, one record may simultaneously be:

```text
project: an internal grid-computation project
record type: experiment
domains: D03 + D04 + D08 + D07
epistemic status: unknown
training admission: pending_review
```

None of those fields implies another.

## Multi-label assignments

Assignments may have one resolved primary domain and multiple secondary domains.

```text
primary_domain_id: D03
primary_domain_candidates: [D03]
secondary_domain_ids: [D04, D08]
domain_ids: [D03, D04, D08]
cross_domain: true
```

Equal top scores are not resolved by domain-identifier order. They remain explicit candidate ambiguity:

```text
primary_domain_id: null
primary_domain_candidates: [D01, D05]
secondary_domain_ids: [D12]
domain_ids: [D01, D05, D12]
cross_domain: true
```

D00 is exclusive. If D00 is primary, it is the only primary candidate and no substantive domain may appear in the same assignment.

## Candidate classifier

`config/domain_rules.json` contains the first auditable rule set.

Evidence sources are deliberately simple:

1. legacy transcript tags;
2. title keyword matches;
3. record-content keyword matches.

Rules produce scores only for candidate organization. They do not establish scientific truth, proof status, or canonical subject identity.

Rule v0.05 counts a substantive legacy domain at most once per record, preserves the contributing tag names in evidence, and permits a provisional primary at the secondary threshold when `allow_fallback_primary` is enabled. The builder and independent validator implement the same fallback contract, and a regression fixture prevents them from drifting apart.

The rule configuration is separated from the registry so vocabulary can change without redefining the domains themselves.

Every assignment records a classifier identity containing the declared rule version and the first 16 hexadecimal characters of the complete rules-file SHA-256 digest. Assignment IDs include that identity. A rule edit therefore produces a new classifier identity and new assignment IDs even when `rules_version` is accidentally left unchanged.

Assignment sidecars exclude volatile build timestamps. Identical canonical inputs, registry bytes, and rule bytes therefore reproduce identical assignment and index bytes. Build time remains receipt metadata in the manifest.

## Research assignments

Run:

```powershell
python build_domain_indexes.py
```

Research outputs:

```text
data/research_db/domains/assignments.jsonl
data/research_db/indexes/domains/D00_unclassified.jsonl
...
data/research_db/indexes/domains/D14_education_workforce_development.jsonl
```

Each index contains IDs and projection metadata only, not duplicated research content.

## Training assignments

Training-example domains are **inherited from source research-record assignments** through `source_record_ids`.

The training classifier does not re-read the target answer and independently decide what domain it belongs to.

Outputs:

```text
data/training_db/domains/assignments.jsonl
data/training_db/indexes/domains/D00_unclassified.jsonl
...
data/training_db/indexes/domains/D14_education_workforce_development.jsonl
```

This preserves the lineage:

```text
training_example_id
    -> source_record_ids
        -> research domain assignments
            -> domain registry
```

## Manifest

The builder writes:

```text
data/domain_manifest.json
```

It records:

- input SHA-256 digests;
- registry/rules digests;
- rule version and digest-bound classifier identity;
- assignment digests;
- every domain-index digest;
- primary-domain counts;
- primary-candidate and ambiguous-primary counts;
- total membership counts;
- cross-domain counts;
- explicit governance declarations that canonical content was not mutated and automatic assignment is not canonical.

The manifest is generated locally and gitignored.

## Validation

After building, run:

```powershell
python validate_domain_indexes.py --report data/domain_structural_report.json
```

Validation is exhaustive and checks:

1. Draft 2020-12 schema conformance for every domain assignment;
2. assignment-ID and subject-ID uniqueness, including binding to the current rule identity;
3. exact closure over every research record and every training candidate;
4. source-hash agreement with the dual databases;
5. domain IDs against the committed registry;
6. D00 exclusivity;
7. resolved, ambiguous, secondary, and cross-domain consistency;
8. automatic `review_status == candidate`;
9. complete recomputation of every inherited training projection, including D00;
10. exact domain-index membership with no missing or extra IDs;
11. manifest SHA-256 and count reconciliation.

The structural validator is followed by the provisional anti-degeneracy calibration gate:

```powershell
python validate_domain_calibration.py --report data/domain_calibration_report.json
```

It rejects excessive multilabel saturation, excessive average memberships, dominant-domain collapse, excessive D00 assignments, excessive unresolved primary ties, and insufficient substantive-domain coverage. A calibration pass is necessary but does not establish canonical semantic correctness.

Deterministic fixture regressions run with:

```powershell
python -m pip install -r requirements.lock
python -m unittest -v test_domain_registry.py
```

The fixture suite includes explicit failures for all-D00 collapse, universal multilabel saturation, broken training inheritance, stale rule identity, and identifier-order tie bias. The same suite runs in CI without requiring the private corpus.

After both full-corpus validators pass, generate a commit-ready receipt with the exact repository commit used for the run:

```powershell
python build_domain_validation_receipt.py `
  --repository-revision <40-character-commit-sha> `
  --output evidence/domain_validation_receipt.json
```

The receipt contains hashes, versions, aggregate statistics, guardrails, runtime metadata, and validator results. It contains no canonical corpus content.

## Review and promotion

v0.01 defines the following review states:

```text
candidate
reviewed
canonical
rejected
```

The builder emits only `candidate`.

A later review workflow may promote or reject assignments, but no automatic rule is permitted to emit `canonical`.

## Training usage

Domain views are orthogonal to task views.

The intended training selection space is:

```text
Domain x Task x Knowledge Status x Admission x Split
```

Examples:

```text
D01 Mathematics x proof_audit
D05 Formal Methods x theorem_proving
D08 Electrical Systems x sequence_prediction
D07 AI/ML x relation_prediction
D03 Dynamical Systems x research_continuation
```

Cross-domain examples remain multi-labelled rather than copied into independent canonical datasets.

## Current boundary

### Completed

- stable domain registry;
- D00 unclassified boundary;
- multi-domain candidate assignments;
- separated classifier rule configuration;
- research-record domain sidecars;
- training domain inheritance;
- lightweight per-domain indexes;
- domain manifest;
- exhaustive validator.
- anti-degeneracy calibration gate;
- digest-bound classifier and assignment identity;
- explicit unresolved-primary representation;
- deterministic regression fixtures and CI.

### Open

- run the builder over the complete local 70,953-record / 34,747-candidate databases;
- inspect domain distribution and D00 rate;
- inspect ambiguous-primary rate and substantive-domain coverage;
- review false-positive / false-negative samples by domain;
- tune candidate rules only if observed failures justify it;
- define human review/promotion workflow;
- commit a non-sensitive full-corpus validation receipt containing hashes, versions, counts, distributions, commands, and pass/fail status;
- integrate reviewed domains into Knowledge/Claim/Dependency Graph v0.03;
- generate admitted Domain x Task training views after training admission begins.

### Verification status

The architecture and implementation are defined. Corpus-scale domain assignments are **not yet validated** until the local builder and validator are run against the generated databases.

### Next highest-value task

Run the domain builder and full validator. Then inspect the resulting domain distribution before changing rules or adding new domain concepts.
