# Research Graph v0.03 — Knowledge / Claim / Dependency Graph

**Status:** Schema frozen. Extraction not started.

This is not a third database. The graph is a **view/index structure over the
Research Database**: every node references canonical `record_id`s and never
replaces them. The v0.02 closure rule is preserved.

```text
Research Database (records.jsonl)
    -> candidate extraction (NOT STARTED)
        -> candidate graph (data/research_graph/candidate_*.jsonl)
            -> review (human-with-evidence only)
                -> canonical graph (reviewed_*.jsonl)
```

Governing rules, inherited from `docs/DATA_ARCHITECTURE.md`:

- Never modify research history to make the graph cleaner.
- Never infer truth from prose. Every status defaults to `unknown`.
- Never promote a node or relation automatically. Review is a
  human-with-evidence act citing explicit `evidence_record_ids`.
- Database integrity does not equal claim truth.

---

## 1. Node classes (frozen)

Five classes, discriminated by `node_type`. Full field requirements live in
`schemas/research_graph_node.schema.json`.

| Class | Meaning | Key fields |
|-------|---------|------------|
| **definition** | A formally introduced object, symbol, term, model, operator, quantity, or rule | `name`, `statement`, `statement_basis`, optional `symbol` |
| **claim** | A statement capable of carrying an epistemic status. `C = (S, sigma)` | `statement`, `statement_basis`, `epistemic_status` (default `unknown`) |
| **argument** | An inferential transition `A: (C_i1..C_ik) -> C_m`. Not every derivation is a proof | `argument_type`, `premises`, `conclusion`, `assumptions`, `proof_status` (default `unknown`), `verification_method` |
| **experiment** | `E = (H, M, P, R, I)`. Experiments `supports` claims; the vocabulary has no `proves` | `hypothesis`, `method`, `parameters`, `result`, `interpretation`, `supports` |
| **failure** | First-class record of `attempt -> failure -> diagnosis -> repair` | `failed_object`, `failure_mode`, `counterexample`, `violated_assumption`, `diagnosis`, `resulting_repair` |

`proof_sketch != informal proof != machine checked proof` is enforced by the
restricted `proof_status` enum. Claim `epistemic_status` reuses the Research
Database's 16-value registry — no parallel ontology.

**Statement policy:** `statement` text is either a verbatim excerpt from the
cited records (`statement_basis: verbatim`) or a normalized paraphrase explicitly
marked (`normalized`, with notes in `review.notes`). Provenance is mandatory
either way; this is the only deliberate, pinned softening of the no-copy rule.


## 2. Relation vocabulary (frozen)

Exactly ten relation types, declared in
`schemas/research_graph_relation.schema.json` with endpoint-class constraints in
`config/research_graph.json`:

```text
depends_on   supports      contradicts   supersedes   refines
tests        falsifies     repairs       defines      uses
```

Every edge carries: `relation_id` (deterministic), `source_node_id`,
`relation_type`, `target_node_id`, `source_record_ids` (>= 1), `basis`
(`explicit` | `inferred`), `confidence` (null unless a recipe defines scoring),
and `review.status` (`pending` by default).

The `basis` field exists because even `C1 -> C2` may itself be an inference.
Inferred edges are recorded as inferred, never silently presented as explicit.

## 3. Identity rules

Node and relation IDs are deterministic content addresses (rules in
`config/research_graph.json`):

```text
node_id     = graph:<type>:sha256(canonical_key)[:24]
relation_id = relation:sha256(source_node_id|relation_type|target_node_id)[:24]
```

Consequences:

- Re-extraction is idempotent: the same claim found in many conversations
  collapses into one node carrying many provenance links.
- A theorem developed across seventeen chats becomes one object.
- A grid principle appearing in forty conversations becomes one definition with
  forty provenance links.
- A failed idea remains connected to the exact result that killed it.

## 4. Lifecycle and promotion

```text
lifecycle: candidate -> reviewed | rejected | superseded
```

- Extraction never promotes. Everything starts `candidate`.
- `candidate -> reviewed` requires `review.status == approved`, a named
  reviewer, and non-empty `review.evidence_record_ids` resolving to Research DB
  records. `automatic_promotion: none`.
- Any `epistemic_status` stronger than `unknown` requires the same evidence bar.
  Automatic truth inference is forbidden in v0.03.
- Supersession is structural: a `supersedes` edge from the replacing node, plus
  the replaced node's `lifecycle: superseded` and `superseded_by` pointer.
  History is never deleted; falsified and superseded nodes remain first-class.

## 5. Why the transition is the valuable unit

With the graph, the archive gains objects normal corpora destroy — the record of
*why knowledge changed*:

```text
conjecture        -> counterexample
failed experiment -> diagnosis
proof gap         -> missing lemma
old definition    -> superseding definition
```

A `transition` is any path `S_t --(evidence/reason)--> S_{t+1}`. Once extraction
exists, these transitions become candidate training material for failure
diagnosis, proof auditing, and research continuation — tasks that dialogue
imitation cannot express.

## 6. Training implications (future, not current)

Once the graph exists, one fragment generates several supervised tasks:
knowledge retrieval (`D -> ?`), dependency prediction, proof continuation, proof
audit (`A -> locate unsupported transition`), status classification
(`C + evidence -> sigma(C)`), research continuation, failure diagnosis. The
task registry (`config/training_tasks.json`) already names these; the graph is
what lets them be generated from structured evidence rather than dialogue shape.

## 7. Verification status

- **v0.02 Research DB:** validated structurally on the complete corpus
  (70,953 records; schemas, hashes, lineage, closure, policy all pass).
- **v0.03 graph:** contract frozen; no graph files exist yet.
  `validate_research_graph.py` performs contract self-consistency checks now
  and full graph validation once files appear.
- **Research claims themselves:** not verified, and not verifiable by this layer.

## 8. Open items

1. candidate extraction recipe (per-class, evidence-pinned);
2. review workflow tooling;
3. canonical status promotion evidence rules (partially specified above);
4. graph integrity validation at extraction time (validator is ready);
5. specialized training-view generator over admitted graph fragments.
