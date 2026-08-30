# Training-data — Research Archive and Governed Training Data

Turned a year of ChatGPT conversations (~1,384 conversations, ~500 MB raw export)
into a clean, tagged, validated corpus. All pipeline processing is local;
raw data stays in `data/raw/` (gitignored).

The repository now separates two distinct data products:

1. **Research Database** — canonical, provenance-preserving history of the work.
2. **Training Database** — derived, task-oriented examples with explicit admission status.

The repository supports two governed ingress routes:

```text
raw chat exports -> normalized transcripts -----+
                                                 +-> canonical Research Database
structured project archives -> archive importer +       -> reviewed Training Database
                                                           -> task/model-specific exports
```

See [`docs/DATA_ARCHITECTURE.md`](docs/DATA_ARCHITECTURE.md) for the full contract.

## Quick start

```powershell
# Install the reproducible core validation environment
python -m pip install --requirement requirements.lock

# Regenerate normalized transcripts from the raw export
python convert_export_to_transcripts.py --raw-dir data/raw --out-dir data/transcripts --index-dir data/index --pii-report --with-codex

# Validate normalized transcript outputs
python validate_outputs.py

# Build the two governed databases
python build_dual_databases.py

# Validate both databases (schemas, lineage, policy, content addressing)
python validate_databases.py

# Import a structured non-conversation project archive without inventing chat
python import_project_archive.py --archive-root <extracted-archive-root> --archive-zip <source.zip> --research-dir <research-output> --training-dir <training-output>

# Validate the v0.04 research graph contract (and graph files once they exist)
python validate_research_graph.py

# Validate opaque source artifact identities and fail-closed reuse policy
python validate_artifact_manifest.py
```

`build_dual_databases.py` creates:

| Database | File | Role |
|----------|------|------|
| Research | `data/research_db/records.jsonl` | Canonical research-history records with lineage, reasoning preservation, epistemic status field, and training policy |
| Research | `data/research_db/manifest.json` | Reproducibility manifest and hashes |
| Training | `data/training_db/candidates.jsonl` | Derived training candidates; nothing is automatically admitted |
| Training | `data/training_db/manifest.json` | Candidate counts, hashes, task/admission summary |

`import_project_archive.py` is the governed path for heterogeneous archives
containing documents, source indexes, claims, equations, experiments, summaries,
and DAT containers. It verifies the archive checksum manifest before and after
the import, deduplicates identical source bytes by SHA-256 while retaining alias
paths, preserves historical status labels only as metadata, and emits no
training candidates. Reconstructed thread summaries remain notes; they are not
promoted into fabricated user/assistant conversations.

Schemas:

- `schemas/research_record.schema.json`
- `schemas/training_example.schema.json`
- `config/training_tasks.json`

Opaque source artifacts are registered in `artifacts/manifest.json` and governed
by `schemas/source_artifact.schema.json`. See
[`docs/LICENSING.md`](docs/LICENSING.md). Format identification and Git identity
do not grant reuse, execution, or training permission.

### Important training boundary

The Research Database is history-preserving. Proposed, failed, superseded, corrected, and verified work may all belong there with explicit status.

The Training Database is selective. A record is not valid training material merely because it exists in project history.

Default training admission states are:

```text
pending_review
admitted
rejected
quarantined
superseded_for_training
```

The first dual-database builder intentionally infers **no theorem/proof truth status** from keywords and admits **nothing automatically**. Specialist task recipes are expected to derive reviewed views from canonical research records.

Source presence is not evidence of licensing. Generated research records therefore
start with `license_status: unknown`; an admitted example requires a reviewed
source policy, acceptable sensitivity, cleared or reviewed PII, and an owned or
permitted license status.

## Normalized transcript outputs

| File | Description |
|------|-------------|
| `data/transcripts/transcripts.jsonl` | All 1,361 clean conversations (69,592 messages, ~54.6 M estimated tokens) |
| `data/transcripts/transcripts_work_only.jsonl` | Work-scope subset (1,120 convos, 54,643 msgs) |
| `data/transcripts/codex_documents.jsonl` | 42 Codex tasks (user-turn text) |
| `data/transcripts/stats_report.json` | Pipeline stats: counts, models, drops, tags |
| `data/transcripts/pii_report.json` | PII scan — counts only, 229 conversations flagged |
| `data/index/conversations.json` | Header-level index: id, title, tags, dates, token est. |
| `data/index/projects.json` | Per-tag rollups (engineering 757, coherence 735, …) |
| `data/raw/` | Original export shards — DO NOT COMMIT |

## Normalized transcript record schema

```json
{
  "conversation_id": "...", "title": "...",
  "create_time": 123.0, "update_time": 456.0,
  "is_archived": false, "source": "chatgpt",
  "tags": ["engineering", "coherence"], "scope": "work",
  "messages": [
    {"role": "user", "content_type": "text", "text": "...", "images": 0,
     "create_time": 123.0, "message_id": "..."},
    {"role": "assistant", "content_type": "text", "text": "...",
     "reasoning": "<thoughts, captured for analysis>",
     "model_slug": "gpt-5-1-thinking", "create_time": 124.0, "message_id": "..."}
  ]
}
```

### Processing rules applied

1. **Canonical chain** — reconstructs the message order by walking `parent` links
   from `current_node` back to root. Stale edit-branches are dropped automatically.
2. **Consecutive-turn merge** — consecutive same-role messages (assistant→assistant,
   user→user) are merged with `\n\n` separators. Result: 0 role violations, near-
   perfect 34,798 user / 34,794 assistant balance. Required for SFT formats that
   expect strict alternation.
3. **Reasoning capture** — `thoughts` / `reasoning_recap` nodes are attached to the
   preceding assistant text message in a separate `reasoning` field (18,755 msgs
   have reasoning). The governed Training Database excludes reasoning from targets by default.
4. **Image parts** — multimodal `image_asset_pointer` parts become an `[image]`
   placeholder and are counted in `images` (1,886 image parts total).
5. **Filtering** — 23 conversations dropped (`too_few_messages`); empty/whitespace
   messages removed; non-user/assistant roles dropped.
6. **Tagging** — domain-vocabulary classifier grounded in this corpus:
   engineering, coherence, papers, noetica, creative, physics, lean-math, data-ai,
   numerics, math, gr-nr, documentation, personal, misc. `scope` = work/personal/misc.

## Multi-task Training Database

The Training Database is designed to support multiple tasks over the same canonical knowledge rather than creating one monolithic corpus.

The task registry currently includes:

- knowledge retrieval and question answering;
- multi-turn dialogue;
- mathematical derivation and theorem proving;
- proof auditing and counterexample search;
- claim-status classification;
- research continuation and experiment design;
- code generation, repair, and review;
- technical writing and summarization;
- relation and sequence prediction;
- grid-signal sequence experiments.

A single Research Database record may support several distinct training examples while all of them retain `source_record_ids` back to the canonical history.

## Legacy/direct training formats

`build_training_formats.py` generates conventional model-training formats
directly from normalized transcripts. This path bypasses governed admission, so
it is fail-closed unless the caller explicitly acknowledges unsafe legacy mode:

```powershell
python build_training_formats.py --unsafe-legacy-transcript-export
python build_training_formats.py --unsafe-legacy-transcript-export --scope work
python build_training_formats.py --unsafe-legacy-transcript-export --include-reasoning
```

Splits are assigned per conversation and shared across formats through `data/training/split_manifest.json`.

| Format | Files | Records (train/val) | ~Tokens |
|--------|-------|--------------------|---------|
| OpenAI fine-tune | `data/training/openai/{train,val}.jsonl` | 1293 / 68 | 54.5 M |
| ShareGPT | `data/training/sharegpt/{train,val}.jsonl` | 1293 / 68 | 54.5 M |
| Alpaca (turn pairs) | `data/training/alpaca/{train,val}.jsonl` | 13,178 / 659 | 25.1 M |
| HF parquet | `data/training/hf/{train,val}-00000.parquet` | 1293 / 68 | 54.5 M |

These direct exports remain useful for compatibility experiments, but they
**bypass the training-admission layer**. Their split manifest records
`source_mode: legacy_transcript`, `unsafe_acknowledged: true`, and
`admission_filter: none`. They are not governed production exports.

Details:
- **OpenAI**: strict `{"messages": [...]}` schema — system prompt first (derived
  per conversation from its tags, or override with `--system-prompt`), strict
  role alternation, no extra keys.
- **ShareGPT**: `conversations` with `from: human|gpt`, plus top-level metadata
  (`id`, `title`, `tags`, `scope`, `system`) that trainers ignore but analysis uses.
- **Alpaca**: one record per (user → assistant) turn pair; `instruction` = title,
  `context` = tags; capped at 20 pairs per conversation (`--max-pairs-per-conv`).
- **HF parquet**: columns `conversation_id, title, tags, scope, messages
  (list<struct{role, content}>), num_messages, est_tokens, split`.
- **Reasoning** is excluded from these formats by default; `--include-reasoning`
  embeds it as a `<thinking>...</thinking>` block inside assistant content.
- Conversations are trimmed to start with the user prompt and end with an
  assistant response before formatting.

Validate the conventional exports with:

```powershell
python validate_training_formats.py
```

The validator accepts only outputs that explicitly declare this legacy bypass.
It does not convert an unsafe export into an admitted one.

## Scientific framework

[`docs/SCIENTIFIC_FRAMEWORK.md`](docs/SCIENTIFIC_FRAMEWORK.md) maps the
implementation to a project-neutral academic framework: observational records,
evidence objects, semantic claims, provenance, derivation identity, admission
control, and reproducibility evidence. It also states what each validation layer
does and does not establish.

## PII note

`pii_report.json` flags 229 conversations with potential PII (phone/email/credit-
card/ipv4/ssn patterns — many are false positives from regex over technical text).
Review before any public release. Nothing here is published by this pipeline; all
outputs stay local.
