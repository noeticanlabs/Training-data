# NELS G0–G9: experiential schooling environments and grade gates

**Status:** prototype environment and curriculum, **not** CohField-certified learning.
**Repository:** `noeticanlabs/Training-data`
**Code:** `cohfield_school/grade_worlds.py`
**Machine-readable registry:** `cohfield_school/curriculum/grades_v0_2.json`

## Scope

This repository is the **external school and curriculum source**. CohField remains
the student and owns native learning, native memory, inference and its governed
authority boundaries. The school may only provide **synthetic candidate
experiences** and collect **provisional scores**. It is not a sensor authority,
a paid measurement authorizer, a physical-world verifier, a checkpoint writer,
or a certificate signer.

The 30 names below are **three configurations per grade of one bounded
prototype task family**, not 30 independently validated mechanism families.
They do not yet satisfy source-family-independent generalization.

| Grade | Native educational challenge | First three world configurations | Required genuine graduation evidence |
|---|---|---|---|
| **G0 — Observation** | Distinguish object state, event, and no-op | Workshop Counter, Hallway Beacon, Weather Station | Held-out event recognition with negative no-change controls |
| **G1 — Relationships** | Track stable IDs across relocation/relabeling | Inventory Tracker, Parcel Relay, Moving Tags | Preserve native entity identity despite changed surface attributes |
| **G2 — Causality** | Discover consequences of controlled Boolean interventions | Switchboard, Valve Bench, Relay Grid | Infer mechanism through interventions, not answer copying or correlation alone |
| **G3 — Prediction** | Predict motion/state transitions | Conveyor Cycle, Moving Cart, Signal Timer | Predict novel start states, quantify uncertainty and beat baseline |
| **G4 — Memory** | Encode and recall opaque association | Supply Cache, Artifact Index, Room Map | Recover through **independent complete CohField restart**, with no re-teaching |
| **G5 — Generalization** | Apply acquired Boolean rule with renamed controls | Mechanism Transfer, Control Panel, Renamed Buttons | Transfer across source families; reject dissimilar mechanisms |
| **G6 — Language** | Convert a statement into a typed native relation triple | Instruction Bridge, Spatial Language, Action Phrases | Learn grounded human↔native mapping; avoid unlicensed ambiguity resolution |
| **G7 — Reasoning** | Construct minimal valid key/door plans | Locked Workshop, Tool Sequence, Maintenance Route | Multi-step planning and lawful execution on unseen world states |
| **G8 — Adaptation** | Update behavior after a control reversal | Changing Controls, Faulty Sensor, Reversed Switch | Identify changed regime, revise only applicable knowledge, preserve old competence |
| **G9 — Autonomy** | Choose informative bounded diagnostic measurement | Research Lab, Repair Diagnosis, Budgeted Experiment | Discover own gaps, choose investigations, remain budgeted, stop safely |

**Caution:** Some prototype cases intentionally supply enough information
in the prompt to solve the immediate challenge, and G4's in-session answer
check is **not** a retention test. Passing these exercises does not yet
demonstrate learning, agency, persistence, or autonomy.

## Session protocol

A school session has exactly three phases:

```text
TRAIN
  training_prompt()  => scenario, available probes, bounded budget
  intervene(probe)   => simulated observation + hash-linked audit receipt
           |
           v
EXAM
  begin_exam()       => test challenge; training probes disabled
  submit(answer)     => no answer feedback before close
           |
           v
CLOSED
  provisional_score  => score only, certified=false
```

The session takes `grade`, `world`, a reproducibility seed, and an
intervention budget. The session rejects unknown worlds, duplicate probes,
over-budget probes, training after exam, and repeat submissions. The student's
external adapter must see **only the returned public JSON**, not the
teacher-side generator source, private expectation, or seed.

The training probes are **educational simulations**, not signed physical
measurement or independent CohField evidence. Receipt hashes make basic
mutation visible but are not signatures or independent attestations.

## Execute locally (no GitHub Actions)

From the root of Training-data:

```bash
python -m unittest discover -s cohfield_school/tests -v
python -m cohfield_school.run_grade --grade G2 --world switchboard \
  --seed local-demo --probe set:1:1 --probe set:0:1 --probe set:0:0 \
  --answer-json true
```

`--answer-json` accepts a JSON literal: for example `true`, `7`,
`"state_changed"`, or `["take_key","unlock","open"]`.
The example answer might be wrong for a given deterministic seed. All results
remain unpromoted teacher-side provisional scores.

## What CohField must prove before grade certification

1. **Native acquisition:** Compare frozen pre-schooling CohField against the
   *same* CohField after experiences. Require attributable native state change,
   independently measured performance gain and no hardcoded-answer path.
2. **Independent retention:** Fully terminate the learner process. Use only
   CohField's authorized checkpoint/recovery flow, with receipts and replay
   verified independently; never carry a school answer cache into the restart.
3. **Genuine generalization:** Evaluate in newly authored worlds with different
   mechanism families, changed labels and misleading familiar cues. Test when
   prior rules must be refused. Freeze untouched challenges before training.
4. **No regression:** Measure previously demonstrated competencies, not just the
   new grade; include interference, contradictions and applicability errors.
5. **Authority:** Independent host controls admission; Class A learns/reasons,
   Class B retrieves only permitted evidence, and Class C governs resources.
   Synthetic education must never impersonate paid physical observations.
6. **Independent certification:** Examiner-controlled, secret generator and
   external verifier sign a narrowly scoped, replayable certificate. A school
   cannot certify a learner or its own examination.

The numeric pass/fail thresholds are **not calibrated** yet. They must be
pre-registered against baseline and negative-control distributions rather than
chosen after observing test scores.

## Progression and evidence status

```text
G0 -> G1 -> G2 -> G3 -> G4 -> G5 -> G6 -> G7 -> G8 -> G9
  \_________________ candidate experiences ________________/
                          |
                  native CohField adapter
                          |
           independent examiner + retained-state witness
                          |
                   certificate or HOLD
```

Higher grades do not automatically certify earlier grades. Every issued
credential will have a scope, version, evidence lineage, regression obligations,
requalification schedule, and revocation path.

## Required future integrations

- Explicitly typed school → CohField native adapter, isolated from canonical
  physical observation authority.
- Real episode-by-episode native updates with independent learning receipts.
- Authoritative train/validation/untouched test partition with source-family
  isolation and additional mechanism families per grade.
- Persistent learner-process restart / checkpoint / recovery witness.
- G6 positive language learning cases plus ambiguity and bidirectional tests;
  G7 stateful action execution; G8 longer change detection; G9 actual
  self-chosen bounded experiment loops.
- Independently controlled examination keys and signer.

**No existing grade is certified.** 
