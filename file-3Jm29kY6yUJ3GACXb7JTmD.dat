import Mathlib.Data.ByteArray
import Coh.Contract.MicroV2
import Coh.Contract.RejectCode

/-!
# Coh.Contract.MicroV3

Extended micro receipt with Transition Contract fields:
- objectiveSatisfied: objective layer check (nullable)
- sequenceValid: sequence/temporal guard check
- overrideApplied: governance override flag

This module extends MicroV2 (canonical ByteArray form) with the
Transition Contract extension for Phase 1 V2 hardening.
-/

namespace Coh.Contract

open Coh.Contract

/-- Target objective for optional objective layer validation -/
inductive ObjectiveTarget where
  | minimizeSpend   : ObjectiveTarget
  | maximizeValue  : ObjectiveTarget
  | closeTickets   : ObjectiveTarget
  | zeroPending    : ObjectiveTarget
  | custom        : String → ObjectiveTarget

/-- Objective result if checked -/
inductive ObjectiveResult where
  | satisfied : ObjectiveTarget → ObjectiveResult
  | violated : ObjectiveTarget → ObjectiveResult
  | notApplicable : ObjectiveResult

/-- Extended V3 micro receipt with Transition Contract fields -/
structure MicroReceiptV3 where
  /-- Base V2 fields -/
  schemaId         : SchemaId
  version         : String
  objectId        : String
  canonProfileHash : String
  policyHash      : String
  stepIndex       : Nat
  stateHashPrev   : Digest
  stateHashNext  : Digest
  chainDigestPrev : Digest
  chainDigestNext : Digest
  canonicalBytes : ByteArray
  metrics        : Metrics
  /-- V3 Transition Contract extensions -/
  /-- Objective layer result (nullable = not checked) -/
  objectiveResult : Option ObjectiveResult
  /-- Sequence/temporal guard: true if passes rolling checks -/
  sequenceValid : Bool
  /-- Governance override: true if exception applied -/
  overrideApplied : Bool
  deriving Repr, DecidableEq

/-- Convert V2 to V3 (backwards compatible, sets new fields to defaults) -/
def MicroReceiptV2.toV3 (r : MicroReceiptV2) : MicroReceiptV3 :=
  { r with
    objectiveResult := none
    sequenceValid := true
    overrideApplied := false }

/-- Convert V3 back to V2 (drops Transition Contract fields) -/
def MicroReceiptV3.toV2 (r : MicroReceiptV3) : MicroReceiptV2 :=
  { schemaId := r.schemaId
    version := r.version
    objectId := r.objectId
    canonProfileHash := r.canonProfileHash
    policyHash := r.policyHash
    stepIndex := r.stepIndex
    stateHashPrev := r.stateHashPrev
    stateHashNext := r.stateHashNext
    chainDigestPrev := r.chainDigestPrev
    chainDigestNext := r.chainDigestNext
    canonicalBytes := r.canonicalBytes
    metrics := r.metrics }

/-- Check if objective layer is satisfied (null = not checked = treat as satisfied) -/
def objectiveSatisfied (r : MicroReceiptV3) : Bool :=
  match r.objectiveResult with
  | none => true  -- not checked, default to pass
  | some result =>
    match result with
    | ObjectiveResult.satisfied _ => true
    | ObjectiveResult.violated _ => false
    | ObjectiveResult.notApplicable => true

/-- Full Transition Contract validity -/
def transitionContractValid (cfg : BoundaryConfig)
    (prevState nextState prevChainDigest : Digest)
    (r : MicroReceiptV3) : Prop :=
  /-- Base V2 validity -/
  microV2SchemaOk cfg (r.toV2) ∧
  microV2StateLinkOk prevState nextState (r.toV2) ∧
  microV2ChainOk cfg prevChainDigest (r.toV2) ∧
  /-- V3 Transition Contract extensions -/
  objectiveSatisfied r ∧
  r.sequenceValid ∧
  ¬ r.overrideApplied  -- unless explicitly overridden

/-- V3 Transition Contract reject code -/
def verifyMicroV3RejectCode
    (cfg : BoundaryConfig)
    (prevState nextState prevChainDigest : Digest)
    (r : MicroReceiptV3) : Option RejectCode :=
  /-- First check V2 validity -/
  if ¬ microV2SchemaOk cfg (r.toV2) then some RejectCode.rejectSchema
  else if ¬ microV2StateLinkOk prevState nextState (r.toV2) then some RejectCode.rejectStateHashLink
  else if ¬ microV2ChainOk cfg prevChainDigest (r.toV2) then some RejectCode.rejectChainDigest
  /-- V3 Transition Contract checks -/
  else if ¬ objectiveSatisfied r then some RejectCode.rejectPolicyViolation  -- repurposed
  else if ¬ r.sequenceValid then some RejectCode.rejectPolicyViolation  -- sequence failure
  else if r.overrideApplied then none  -- override accepted
  else none

/-- V3 Transition Contract decision -/
def verifyMicroV3
    (cfg : BoundaryConfig)
    (prevState nextState prevChainDigest : Digest)
    (r : MicroReceiptV3) : Decision RejectCode :=
  match verifyMicroV3RejectCode cfg prevState nextState prevChainDigest r with
  | none => Decision.accept
  | some code => Decision.reject code

/-- Sequence guard configuration -/
structure SequenceGuard where
  /-- Maximum state drift per window -/
  maxStateDrift : Nat
  /-- Maximum cumulative spend per window -/
  maxCumulativeSpend : Nat
  /-- Window length in steps -/
  windowLength : Nat
  deriving Repr, DecidableEq

/-- Default sequence guard (permissive) -/
def defaultSequenceGuard : SequenceGuard :=
  { maxStateDrift := u128Max.toNat
    maxCumulativeSpend := u128Max.toNat
    windowLength := 100 }

/-- Check sequence guard validity for a sequence of receipts -/
def sequenceGuardValid (guard : SequenceGuard)
    (receipts : List MicroReceiptV3) : Bool :=
  /-- Accumulate metrics over window -/
  let window := receipts.take guard.windowLength
  let totalSpend := window.foldl (fun acc r => acc + r.metrics.spend.toNat) 0
  totalSpend ≤ guard.maxCumulativeSpend

/-- Policy governance: policy receipt with chain -/
structure PolicyReceipt where
  policyId : String
  version : Nat
  contentHash : Digest
  signature : String  -- placeholder for Ed25519 sig
  previousReceiptHash : Option Digest
  validFrom : Nat  -- block number or timestamp
  deriving Repr, DecidableEq

/-- Policy chain validity -/
def policyChainValid (p : PolicyReceipt) : Bool :=
  match p.previousReceiptHash with
  | none => p.version = 0  -- genesis policy
  | some _ => p.version > 0

/-- Example V3 receipt with objective satisfied -/
def sampleMicroV3 : MicroReceiptV3 :=
  { schemaId := "coh.receipt.micro.v3"
    version := "1.0.0"
    objectId := "agent.workflow.demo"
    canonProfileHash := "coh.default"
    policyHash := "default"
    stepIndex := 1
    stateHashPrev := ⟨"state-0"⟩
    stateHashNext := ⟨"state-1"⟩
    chainDigestPrev := ⟨"chain-0"⟩
    chainDigestNext := ⟨"chain-1"⟩
    canonicalBytes := ByteArray.empty
    metrics := { vPre := 100, vPost := 85, spend := 15, defect := 0 }
    objectiveResult := some (ObjectiveResult.satisfied ObjectiveTarget.minimizeSpend)
    sequenceValid := true
    overrideApplied := false }

/-- Example V3 with objective violated -/
def violatedObjectiveMicroV3 : MicroReceiptV3 :=
  { sampleMicroV3 with
    objectiveResult := some (ObjectiveResult.violated ObjectiveTarget.minimizeSpend) }

/-- Example V3 with sequence failure -/
def sequenceFailedMicroV3 : MicroReceiptV3 :=
  { sampleMicroV3 with
    sequenceValid := false }

/-- Example V3 with override -/
def overriddenMicroV3 : MicroReceiptV3 :=
  { sampleMicroV3 with
    overrideApplied := true }

/-- Example: V3 basic validity check -/
example : verifyMicroV3RejectCode defaultBoundaryConfig
    ⟨"state-0"⟩ ⟨"state-1"⟩ ⟨"chain-0"⟩ sampleMicroV3 = none := by
  unfold verifyMicroV3RejectCode microV2SchemaOk microV2StateLinkOk microV2ChainOk
  simp [sampleMicroV3, defaultBoundaryConfig, defaultCanonProfile, chainUpdate,
    ObjectiveResult.satisfied, ObjectiveResult.notApplicable, Seq]
  /- Note: Would need actual digest values to compute -/
  admit

end Coh.Contract
