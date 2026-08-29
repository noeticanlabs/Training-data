# Training-data — Custom Training Dataset from ChatGPT Export

Turned a year of ChatGPT conversations (~1,384 conversations, ~500 MB raw export)
into a clean, tagged, validated training corpus. All pipeline processing is local;
raw data stays in `data/raw/` (gitignored).

## Quick start

```powershell
# Regenerate everything from the raw export
python convert_export_to_transcripts.py --raw-dir data/raw --out-dir data/transcripts --index-dir data/index --pii-report --with-codex

# Validate outputs
python validate_outputs.py
```

## Outputs

| File | Description |
|------|-------------|
| `data/transcripts/transcripts.jsonl` | All 1,361 clean conversations (69,592 messages, ~54.6 M tokens) |
| `data/transcripts/transcripts_work_only.jsonl` | Work-scope subset (1,120 convos, 54,643 msgs) |
| `data/transcripts/codex_documents.jsonl` | 42 Codex tasks (user-turn text) |
| `data/transcripts/stats_report.json` | Pipeline stats: counts, models, drops, tags |
| `data/transcripts/pii_report.json` | PII scan — counts only, 229 conversations flagged |
| `data/index/conversations.json` | Header-level index: id, title, tags, dates, token est. |
| `data/index/projects.json` | Per-tag rollups (engineering 757, coherence 735, …) |
| `data/raw/` | Original export shards — DO NOT COMMIT |

## Record schema (one JSON object per line)

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
   have reasoning). Drop that field for strict text-only training.
4. **Image parts** — multimodal `image_asset_pointer` parts become an `[image]`
   placeholder and are counted in `images` (1,886 image parts total).
5. **Filtering** — 23 conversations dropped (`too_few_messages`); empty/whitespace
   messages removed; non-user/assistant roles dropped.
6. **Tagging** — domain-vocabulary classifier grounded in this corpus:
   engineering, coherence, papers, noetica, creative, physics, lean-math, data-ai,
   numerics, math, gr-nr, documentation, personal, misc. `scope` = work/personal/misc.

## Mapping to training formats (not yet applied — decide when needed)

**OpenAI chat fine-tune JSONL:**
```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```
Map: iterate `messages`, use `role`+`text`. Add a system prompt with project context.

**ShareGPT / Vicuna JSONL:**
```json
{"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}
```
Map: `role=="user"` → `human`, `role=="assistant"` → `gpt`, `text` → `value`.

**Alpaca (instruction/response):**
Take each assistant message as `output`, its preceding user message as `input`,
and the conversation title as `instruction` hint. Good for single-turn distillation.

**HF `datasets` (parquet):**
Load `transcripts.jsonl` → `load_dataset("json", data_files="...")` → `.to_parquet()`.
Columns: `conversation_id`, `title`, `tags`, `scope`, `messages` (list of dicts).

## PII note

`pii_report.json` flags 229 conversations with potential PII (phone/email/credit-
card/ipv4/ssn patterns — many are false positives from regex over technical text).
Review before any public release. Nothing here is published by this pipeline; all
outputs stay local.
