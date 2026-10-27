# CohField Experiential School — NELS-001

**Status:** external teacher-side protocol laboratory, not CohField pretraining,
not a certified learner, and not an independent examiner.

This additive subsystem preserves Training-data's canonical research database,
training-admission system, UDLM curriculum and branch history. It does not
modify the CohField repository.

## Why the school lives here

Training-data owns lessons, environments, source provenance, partitions and
evaluation protocols. CohField owns all real native learning, memory,
representation changes, applicable hypotheses and independent operation.
The school must not write canonical CohField memories, issue instruction
permits, sign evidence as an independent physical source, or mint certificates.

Existing Cohbit-Copilot language fixtures can be translated into experiences,
but copied fixture answers cannot be called acquired native intelligence.
The initial language focus is L6 meaning, L7 ambiguity and L8 context, after
the native learning interface and baseline are qualified.

## Running the zero-dependency environment prototype

From repository root with Python 3.11+:

    python -m unittest discover -s cohfield_school/tests -v

Manual teacher-side exploration:

    from cohfield_school.world import ExperienceWorld
    world = ExperienceWorld("local-demo")
    print(world.public_state())
    print(world.intervene(False, False))
    print(world.intervene(True, False))
    challenges = world.begin_exam()
    for challenge in challenges:
        world.predict(challenge["a"], challenge["b"], False)
    print(world.finalize())

This demonstration uses a deliberately simplistic all-false baseline and
cannot establish learning. The toy world is openly inspectable and shares
the same rule family between teaching and examination; it is therefore
NOT a secure independent competency test.

## CohField adapter v0 — proposed, not integrated

A future thin adapter must exchange typed, bounded events:

- World start: environment ID, features, action schema, budget and curriculum hash.
- Intervention: learner's selected action; school responds with observation
  plus deterministic receipt. Only the active training phase yields outcomes.
- Exam: school assigns unseen inputs, learner predicts, answers are withheld.
- Retention: the *CohField* process is fully terminated and independently
  restored from its own checkpoint; no lessons or answer cache are supplied.
- Transfer: use distinct rule families, entities and mechanisms; include
  intentionally misleading similarities and inapplicable prior knowledge.
- Qualification: an independent examiner controls secret challenges and signs
  a narrowly scoped certificate only after replay, regression and leakage checks.

The laboratory's hash-chained receipts are unsigned audit aids, not external
attestations, physical-truth evidence or proof of persistence.

## Promotion boundary

The prototype's finalize() deliberately returns certified=false,
retention_verified=false and transfer_verified=false. No grader in this
repository can promote CohField capabilities or bypass its Class A/B/C,
H2, witness, recovery or spending protections.

No training or certification may use private conversation exports without
the existing governed-admission and privacy review. Keep all generated
personal data in gitignored local paths.

## Next gates

1. Freeze baseline against CohField's exact canonical runtime commit.
2. Specify a bounded JSON I/O contract and implement a separate adapter.
3. Demonstrate genuine state changes in CohField during training.
4. Restart CohField in a fresh process using only its own persisted state.
5. Independently test held-out transfer, negative transfer and regression.
6. Introduce secret examiner-controlled test families and external signing.
7. Only then consider a scoped grade certification.

Source of schooling approach: noeticanlabs/Cohbit-Copilot,
fixtures/language_gradeschool and src/noetican_grades.
Existing research history remains governed by docs/DATA_ARCHITECTURE.md.
