# CF-LM Teacher-Data Bridge v0.01

**Status:** implementation prototype

Branch: `agent/cflm-teacher-data-bridge-v001`

## Purpose

Create a governed, deterministic boundary from the Noetican Training Database to CF-LM without treating the Research Database itself as training data.

```text
Research DB
  -> reviewed Training DB
    -> CF-LM teacher-data export
      -> CF-LM adaptation on TRAIN only
      -> teacher-off evaluation on validation/test/holdout
```

The exporter never promotes records. It consumes only examples whose `admission.status` is exactly `admitted`.

## Why byte-visible packs

The bridge intentionally exports UTF-8 bytes rather than tokenizer IDs, embeddings, teacher hidden states, logits, or chain-of-thought. UTF-8 remains an external serialization boundary. CF-LM is free to develop its own persistent relational organization over the visible stream.

This first bridge version is about the data boundary and training/evaluation separation. It is not evidence of language competence.

## Eligibility

An example is exportable only when all of the following hold:

- `admission.status == admitted`;
- `split` is one of `train`, `validation`, `test`, or `holdout`;
- its `split_group` does not occur in another split;
- it is not personal-scope data;
- it has non-empty rendered input and target content.

Pending, rejected, quarantined, superseded-for-training, personal, and unassigned examples are not exported.

## Pack format

Each split is written independently as `<split>.cflm`.

```text
magic: CFLM-TEACHER-DATA-V001\n
repeat:
  uint64-be input_length
  input UTF-8 bytes
  uint64-be target_length
  target UTF-8 bytes
```

Record boundaries are therefore first-class. No artificial transition is created between the end of one example and the beginning of the next.

`records.jsonl` stores provenance and content hashes; `manifest.json` stores split hashes, counts, exclusions, and the claim boundary.

## Training rule

Only `train.cflm` may be used to update CF-LM persistent state.

`validation.cflm`, `test.cflm`, and `holdout.cflm` are evaluation-only. Their target bytes are the hidden consequence during teacher-off evaluation and must not be used for adaptation.

## Governance boundary

This bridge preserves the repository's dual-database rule:

```text
research history != training admission
```

It never reads canonical Research DB records directly and never infers epistemic truth or training admission from prose.

## Self-test

Run:

```bash
python _selftest_cflm_teacher_data_v001.py
```

The self-test checks admitted-only export, personal-data exclusion, assigned split enforcement, split-group leakage rejection, deterministic pack decoding, and fail-closed behavior when no admitted example survives.

## Claim ceiling

A PASS establishes only that governed Training Database examples can be transformed deterministically into a provenance-preserving byte-visible CF-LM curriculum with hard train/evaluation separation.

It does not establish:

- successful full-corpus CF-LM training;
- natural-language understanding;
- semantics;
- grammar induction;
- generalization;
- reasoning;
- autonomous representation invention.

The next dependent layer is the CF-LM-side corpus consumer and byte-visible adaptation harness.
