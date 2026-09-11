# UDLM Certified Learning v2 — Pre-Run Audit

This branch exists specifically to audit and harden the certified-learning system before another full training run.

## Corrected faults

- Removed NumPy-dependent model hashing.
- Fixed multidimensional and scalar tensor hashing by flattening before byte reinterpretation and hashing name/dtype/shape.
- Added a fail-closed machinery preflight before curriculum training.
- Switched the CPU runner to pinned CPU PyTorch rather than downloading the CUDA stack.
- Receipt-chain corruption now fails closed instead of silently returning `GENESIS`.
- Split replay-training memory from held-out retention memory so validation bytes are never replay-trained.
- Renamed the evidence claim from “independent” to the accurate `within_source_heldout_supported`.
- Replaced random certification BPB sampling with deterministic fixed evaluation windows.
- Enforced the 259-symbol protocol by training BOS/EOS boundaries and masking BOS/PAD from autonomous surface emission.
- Restricted hard dynamical admission gates to X/R saturation, the state families actually implicated by prior collapse; H/Q remain measured but not presumed causal.
- Fixed failure classification so positive-but-subthreshold evidence is classified as insufficient evidence instead of a diagnostic coverage gap.
- Added `canonical_after_hash` to transition receipts and linked capability/failure memory back to its governing receipt.
- Added always-on failure artifact upload so machinery failures preserve diagnostic evidence.
- Added checkpoint/report/receipt cross-validation after a successful run.

## Evidence boundary

Current validation is disjoint within each source document, not source-disjoint across corpora. The system must not claim source-disjoint generalization from this run.

The legacy Copilot-style trainer remains present for compatibility but is not the certified admission authority. The authoritative live path is `udlm.run_certified` plus the hash-chained v2 receipt ledger.

## Run boundary

A preflight failure is a machinery failure, not a model failure. Only after preflight passes and candidate episodes execute can ADMIT/HOLD/REJECT outcomes be interpreted as learning evidence.
