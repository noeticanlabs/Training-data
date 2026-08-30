# Dual Database Architecture v0.02

**Status:** Implemented contract with synthetic mutation coverage. A complete
private-corpus rerun is required after the current integrity repair. The v0.04
Knowledge/Claim/Dependency graph contract is frozen separately; see
[`docs/RESEARCH_GRAPH.md`](RESEARCH_GRAPH.md).

This repository serves two different purposes that must remain structurally separate.

1. **Research Database** — a canonical, provenance-preserving database of the full body of work.
2. **Training Database** — a derived, task-oriented database containing only explicitly admitted training examples.

The governing rule is:

```text
raw sources
    -> canonical research database
        -> reviewed/admitted training views
```

Never build the canonical research database from the training database.
Never modify research history to make training data cleaner.
Never allow a training export to silently become canonical research history.

---

## 1. Database A — Research Database

### Purpose

Preserve the complete research record so the project can reconstruct what was proposed, tested, corrected, falsified, superseded, verified, and left unresolved.

This database is optimized for:

- project memory;
- research search and retrieval;
- theorem/proof dependency reconstruction;
- experiment history;
- source and artifact provenance;
- version history;
- claim-status tracking;
- cross-project connection discovery;
- future reprocessing into new training tasks.

### Canonical principle

The Research Database is **history preserving**, not truth flattening.

A false or superseded claim may remain in the Research Database if it is part of the historical record, but its status and superseding relation must be explicit.

### Core record types

The database should be able to represent:

- conversations;
- messages;
- documents;
- source artifacts;
- code artifacts;
- mathematical definitions;
- claims;
- proofs and proof sketches;
- experiments;
- simulations;
- failures;
- corrections;
- dependencies;
- evidence;
- project/version relations.

### Required provenance

Every canonical record should preserve enough information to reconstruct its origin:

- stable record ID;
- source type;
- source conversation/document/artifact ID;
- source message IDs when applicable;
- timestamps;
- project tags;
- source hash where available;
- parent/preceding record relation;
- model/tool metadata when available;
- sensitivity/privacy status;
- epistemic status;
- supersedes / superseded-by relations.

### Epistemic status registry

At minimum:

```text
defined
derived
proposed
conjectured
proof_sketch
informally_proved
machine_checked
numerically_supported
simulation_supported
empirically_supported
falsified
blocked
unresolved
superseded
historical
unknown
```

Unknown is preferable to silently inferring a stronger status.

### Suggested generated layout

```text
data/
  research_db/
    records.jsonl
    conversations.jsonl
    artifacts.jsonl
    claims.jsonl
    relations.jsonl
    manifest.json
    indexes/
      by_project.json
      by_tag.json
      by_status.json
      by_date.json
```

These generated files remain local/gitignored unless an explicit release decision is made.

---

## 2. Database B — Training Database

### Purpose

Create reproducible, governed training datasets for multiple tasks without corrupting or flattening the research archive.

The Training Database is **derived** from Research Database records through explicit recipes.

### Admission rule

A research record is not training data merely because it exists.

Each candidate training example receives an admission decision:

```text
pending_review
admitted
rejected
quarantined
superseded_for_training
```

Training exports use only `admitted` records unless a recipe explicitly requests another class for a controlled experiment.

### Training dimensions

Each admitted example may carry multiple independent labels.

#### Domain / knowledge

Examples:

```text
mathematics
physics
formal_methods
software_engineering
governed_computation
grid_dynamics
ai_ml
research_methods
documentation
maintenance_engineering
```

#### Task

Examples:

```text
knowledge_retrieval
question_answering
multi_turn_dialogue
mathematical_derivation
theorem_proving
proof_audit
counterexample_search
claim_status_classification
research_continuation
experiment_design
code_generation
code_repair
code_review
technical_writing
summarization
relation_prediction
sequence_prediction
grid_signal_sequence
```

#### Evidence status

Examples:

```text
canonical
verified
provisional
negative_example
superseded_context
```

A single canonical research record may therefore generate several training examples for different tasks.

---

## 3. Separation of knowledge and task behavior

Training examples should distinguish **what the system should know** from **what operation it should learn to perform**.

Example:

```text
knowledge: periodic 3-D NSE enstrophy identity
task: derive identity
```

is different from:

```text
knowledge: periodic 3-D NSE enstrophy identity
task: audit a proposed derivation for an invalid integration-by-parts step
```

and different again from:

```text
knowledge: historical failed NSE route
task: identify why the route was retired
```

This distinction enables multiple training curricula over the same canonical knowledge base.

---

## 4. Training views

Do not maintain one monolithic `training.jsonl` as the conceptual source of truth.

Maintain a common training-example schema and produce named views.

Suggested views:

```text
data/training_db/
  candidates.jsonl
  admitted.jsonl
  manifest.json
  views/
    knowledge_qa.jsonl
    math_reasoning.jsonl
    proof_audit.jsonl
    research_continuation.jsonl
    code_generation.jsonl
    code_repair.jsonl
    technical_writing.jsonl
    grid_sequences.jsonl
  splits/
    train.jsonl
    validation.jsonl
    test.jsonl
```

A view is a projection of admitted examples, not a new source database.

---

## 5. Train / validation / test isolation

Random message-level splitting is not sufficient for this corpus because nearby conversations often repeat the same ideas, proofs, code, or documents.

Default split policy should support:

- chronological holdout;
- complete-conversation holdout;
- project holdout;
- theorem/claim-family holdout;
- artifact-family holdout.

Every exported example carries a `split_group` so related examples cannot silently leak across train and evaluation sets.

---

## 6. Privacy and sensitivity boundary

The Research Database may preserve private history locally.

The Training Database must apply a stricter admission boundary.

Default rules:

- `is_do_not_remember == true` -> archive only / training rejected unless explicitly overridden;
- detected secrets or credentials -> quarantined;
- unresolved PII -> quarantined or reviewed;
- personal conversations -> excluded from work-training views by default;
- raw reasoning fields -> archive-preserved but excluded from default training targets;
- source artifacts without known provenance -> pending review.

---

## 7. Reasoning fields

Reasoning traces are not treated as automatically valid supervision.

They may contain:

- abandoned approaches;
- provisional claims;
- errors corrected in the final answer;
- scaffolding that should not become the learned target.

Default rule:

```text
research_db: preserve when legally/technically available
training_db: exclude from target text unless a named, reviewed recipe admits it
```

Verified derivations, proofs, receipts, test results, and explicit correction chains are preferred reasoning supervision.

---

## 8. Artifact handling

Opaque `.dat` files are source containers, not a semantic file type.

Every source artifact should eventually receive a manifest entry containing:

```text
artifact_id
original_filename
sha256
byte_size
mime_type
encoding
source
project
content_class
sensitivity
license_status
training_allowed
parser
notes
```

Artifact content can then be linked from research records without renaming history or losing provenance.

---

## 9. Data lineage

Every training example must be traceable backward:

```text
training_example_id
    -> source_record_ids
        -> source_message_ids / artifact_ids
            -> raw source hash
```

This makes it possible to answer:

- Where did this example come from?
- What version of the work does it represent?
- Was it later superseded?
- Why was it admitted to training?
- Which evaluation split contains related material?

### Heterogeneous project-archive import

Structured research archives must not be forced through the conversation
builder. `import_project_archive.py` provides a separate fail-closed ingress
path for documents, source artifacts, definitions, source claims, equations,
experiments, results, reconstructed thread summaries, and crosslink notes.

The importer applies these rules:

- verify every file named by `SHA256SUMS.txt` before and after import;
- deduplicate identical source bytes by full SHA-256 while retaining all paths;
- preserve verbatim UTF-8 source text when the archive supplies an exact text
  corpus;
- map historical theorem and axiom labels to unverified research claims;
- keep reconstructed thread summaries as notes, never fabricated dialogue;
- set epistemic status to `unknown` and training admission to `pending_review`;
- create zero training candidates automatically;
- preserve the original archive ZIP hash as the root source identity when the
  ZIP is supplied.

This route makes a project archive reproducible as research evidence without
claiming that it replaces a missing raw transcript export.

---

## 10. Versioning

Version the pipeline, schemas, and training recipes separately.

Example:

```text
research_schema_version: 0.02
training_schema_version: 0.02
pipeline_version: 0.02
training_recipe: math_reasoning_v0.01
```

Changing a training recipe does not alter the canonical Research Database.

---

## 11. Canonical flow

```text
ChatGPT exports / source files / repositories / documents
                         |
                         v
                Source normalization
                         |
                         v
              RESEARCH DATABASE
          complete + provenance preserving
                         |
             review / status / relations
                         |
                         v
             Training candidate builder
                         |
                 admission filters
                         |
                         v
              TRAINING DATABASE
         task-labelled + split-isolated
                         |
             +-----------+-----------+
             |           |           |
             v           v           v
         SFT/chat    reasoning    grid/dynamic
           views       views         views
```

---

## 12. Current status

### Completed

- dual-database boundary defined;
- research-vs-training semantics separated;
- training admission model defined;
- multi-task view model defined;
- privacy and reasoning defaults defined;
- lineage requirement defined;
- canonical research and candidate builders implemented;
- deterministic exact-N content-hash sampling implemented;
- candidate admission constrained by every source record policy;
- semantic graph identity separated from provenance identity;
- opaque artifact registry added with fail-closed training policy;
- governed heterogeneous project-archive importer implemented;
- generic Research Database record validation and zero-candidate manifests
  supported;
- synthetic mutation tests implemented for policy, lineage, hashing, graph
  identity, supersession, and legacy-export governance.

### Open

- rerun the repaired builder and validator on the complete private corpus;
- obtain the original raw thread export if verbatim conversation reconstruction
  is required;
- produce and retain a full-corpus validation receipt;
- complete artifact provenance, sensitivity, and license review;
- deduplication / near-duplicate detection;
- claim and epistemic-status extraction/review workflow;
- deterministic leakage-resistant split generator;
- per-task export recipes;
- grid-signal training export.

### Next highest-value task

Import any available structured project archives into the Research Database,
then separately run the complete private transcript corpus when it becomes
available. Retain both manifests and never promote a candidate until PII,
sensitivity, license, and source-policy review all pass.
