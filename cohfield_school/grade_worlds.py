"""NELS G0-G9 simulated, bounded teacher worlds (NOT a certified learner).

No native memory, authority, sensor signing, or paid observation is implemented.
Only explicit training probes return outcomes; exam responses are scored at close.
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Any

WORLDS = {
    "G0": ("workshop-counter", "hallway-beacon", "weather-station"),
    "G1": ("inventory-tracker", "parcel-relay", "moving-tags"),
    "G2": ("switchboard", "valve-bench", "relay-grid"),
    "G3": ("conveyor-cycle", "moving-cart", "signal-timer"),
    "G4": ("supply-cache", "artifact-index", "room-map"),
    "G5": ("mechanism-transfer", "control-panel", "renamed-buttons"),
    "G6": ("instruction-bridge", "spatial-language", "action-phrases"),
    "G7": ("locked-workshop", "tool-sequence", "maintenance-route"),
    "G8": ("changing-controls", "faulty-sensor", "reversed-switch"),
    "G9": ("research-lab", "repair-diagnosis", "budgeted-experiment"),
}


def _rng(grade: str, world: str, seed: str, split: str) -> random.Random:
    key = f"nels-v0.2|{grade}|{world}|{seed}|{split}".encode()
    return random.Random(int.from_bytes(hashlib.sha256(key).digest(), "big"))


def _gate(name: str, a: bool, b: bool) -> bool:
    return {"AND": a and b, "OR": a or b, "XOR": a != b, "NAND": not (a and b)}[name]


@dataclass(frozen=True)
class _Case:
    public: dict[str, Any]
    probes: dict[str, dict[str, Any]]
    expected: Any


def _case(grade: str, world: str, seed: str, split: str) -> _Case:
    r = _rng(grade, world, seed, split)
    is_exam = split == "exam"
    if grade == "G0":
        label = r.choice(["lamp", "valve", "motor", "signal"])
        before, after = r.choice([0, 1]), r.choice([0, 1])
        if is_exam:
            before, after = 0, 1
        public = {"objects": [label], "before": {label: before},
                  "after": {label: after}, "question": "Did an observable state change occur?"}
        probes = {"inspect": {"event": "state_changed" if before != after else "no_change"}}
        return _Case(public, probes, "state_changed" if before != after else "no_change")
    if grade == "G1":
        identity = f"entity-{r.randrange(1000, 9000)}"
        old, new = f"bin-{r.randrange(10, 99)}", f"rack-{r.randrange(10, 99)}"
        public = {"before": {"id": identity, "position": old},
                  "after": {"id": identity, "position": new, "label": "renamed"},
                  "question": "What stable ID refers to the moved entity?"}
        return _Case(public, {"follow": {"same_identity": True}}, identity)
    if grade in ("G2", "G5"):
        # Rule is session-stable across train and exam; names vary by split.
        rule = ["AND", "OR", "XOR", "NAND"][_rng(grade, world, seed, "mechanism").randrange(4)]
        names = ("a", "b") if grade == "G2" else (("red", "blue") if not is_exam else ("alpha", "beta"))
        # These three interventions distinguish every candidate gate. Withhold
        # (True, False) to test prediction outside the observed assignments.
        observations = [(True, True), (False, True), (False, False)] if not is_exam else []
        exam_pair = (True, False)
        public = {"controls": list(names), "question": "Predict output for the requested controls",
                  "request": dict(zip(names, exam_pair))}
        probes = {f"set:{int(a)}:{int(b)}": {"controls": dict(zip(names, (a, b))), "output": _gate(rule, a, b)}
                  for a, b in observations}
        # The requested exam pair is withheld during training, despite visible challenge metadata.
        return _Case(public, probes, _gate(rule, *exam_pair))
    if grade == "G3":
        step = _rng(grade, world, seed, "mechanism").choice([-2, -1, 1, 2])
        start = r.randrange(3, 12)
        public = {"position": start, "question": "Predict the position after one tick"}
        probes = {"tick": {"position_before": start, "position_after": start + step}}
        if is_exam:
            return _Case(public, {}, start + step)
        return _Case(public, probes, start + step)
    if grade == "G4":
        # Same opaque association in train and exam. Only external process
        # termination + independently verified checkpoint recovery tests retention.
        code = _rng(grade, world, seed, "memory").randrange(10000, 99999)
        token = f"artifact-{_rng(grade, world, seed, 'token').randrange(100, 999)}"
        public = {"artifact": token, "question": "Recall its recorded access code"}
        probes = {} if is_exam else {"inspect": {"artifact": token, "code": code}}
        return _Case(public, probes, code)
    if grade == "G6":
        left, right = r.choice([("dog", "cat"), ("key", "box"), ("cart", "door")])
        relation = r.choice(["left_of", "inside", "beside"])
        phrasing = {"left_of": "is left of", "inside": "is inside", "beside": "is beside"}[relation]
        sentence = f"{left} {phrasing} {right}"
        public = {"utterance": sentence, "question": "Translate to native triple [subject, relation, object]"}
        return _Case(public, {"inspect": {"entities": [left, right]}} if not is_exam else {}, [left, relation, right])
    if grade == "G7":
        has_key = r.choice((False, True))
        locked = r.choice((False, True))
        public = {"initial": {"has_key": has_key, "door_locked": locked,
                              "door_open": False},
                  "goal": "door_open", "question": "Provide the minimal ordered plan"}
        probes = {"try_open": {"success": not locked, "reason": "locked" if locked else "openable"},
                  "inspect": {"key_available": True},
                  "take_key": {"has_key": True},
                  "unlock": {"door_locked": False},
                  "open": {"door_open": True}}
        plan = (["take_key"] if locked and not has_key else []) + (["unlock"] if locked else []) + ["open"]
        return _Case(public, probes if not is_exam else {}, plan)
    if grade == "G8":
        # Regime-changing mechanism: before and after must not be flattened.
        a = r.choice([False, True])
        public = {"control": a, "regime": "after_reversal",
                  "question": "Predict current signal after the documented reversal"}
        probes = {"before_reversal": {"control": a, "signal": a},
                  "after_reversal": {"control": a, "signal": not a}}
        return _Case(public, probes if not is_exam else {}, not a)
    if grade == "G9":
        cause = _rng(grade, world, seed, "cause").choice(["sensor", "actuator"])
        sensor_cost, actuator_cost = r.choice(((1, 2), (2, 1)))
        public = {"fault": "output_missing", "possible_causes": ["sensor", "actuator"],
                  "probe_costs": {"measure_sensor": sensor_cost, "measure_actuator": actuator_cost,
                                  "inspect_casing": 1},
                  "probe_budget": 2,
                  "question": "Choose the least-cost discriminating probe"}
        probes = {"measure_sensor": {"sensor_ok": cause != "sensor"},
                  "measure_actuator": {"actuator_ok": cause != "actuator"},
                  "inspect_casing": {"casing_ok": True}}
        # This only tests selecting a useful diagnostic action, not actual autonomy.
        expected = "measure_sensor" if sensor_cost < actuator_cost else "measure_actuator"
        return _Case(public, probes if not is_exam else {}, expected)
    raise ValueError(f"Unsupported grade {grade!r}")


class GradeSession:
    """Teacher-side schooling protocol; students see only public JSON messages."""

    def __init__(self, grade: str, world: str, seed: str, budget: int = 3):
        if grade not in WORLDS or world not in WORLDS[grade]:
            raise ValueError("Unknown grade/world pair")
        if not isinstance(seed, str) or not 1 <= len(seed) <= 128:
            raise ValueError("A bounded seed is required")
        if type(budget) is not int or not 1 <= budget <= 5:
            raise ValueError("Budget must be 1..5")
        self.grade, self.world, self.seed = grade, world, seed
        self._train = _case(grade, world, seed, "train")
        self._exam = _case(grade, world, seed, "exam")
        self._budget = budget
        self._used: set[str] = set()
        self._phase = "training"
        self._head = "0" * 64
        self._events: list[dict] = []
        self._answer: Any = None
        self._append("begin", {"grade": grade, "world": world, "budget": budget})

    def _append(self, kind: str, payload: dict) -> None:
        record = {"sequence": len(self._events), "event": kind,
                  "payload": payload, "previous_hash": self._head}
        blob = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
        record["receipt_hash"] = hashlib.sha256(b"nels.grade.v0.2\0" + blob).hexdigest()
        self._head = record["receipt_hash"]
        self._events.append(record)

    def training_prompt(self) -> dict:
        if self._phase != "training":
            raise ValueError("Training is closed")
        return {"grade": self.grade, "world": self.world, "phase": self._phase,
                "scenario": self._train.public.copy(),
                "allowed_probes": sorted(self._train.probes),
                "remaining_budget": self._budget - len(self._used)}

    def intervene(self, probe: str) -> dict:
        if self._phase != "training" or probe not in self._train.probes:
            raise ValueError("Forbidden or unavailable training probe")
        if probe in self._used or len(self._used) >= self._budget:
            raise ValueError("Duplicate or over-budget intervention")
        self._used.add(probe)
        result = self._train.probes[probe].copy()
        self._append("training_observation", {"probe": probe, "result": result})
        return {"observation": result, "remaining_budget": self._budget - len(self._used),
                "receipt_head": self._head, "evidence_class": "simulated_schooling_unverified"}

    def begin_exam(self) -> dict:
        if self._phase != "training":
            raise ValueError("Exam can start only once")
        self._phase = "examination"
        self._append("exam_begin", {"grade": self.grade, "world": self.world})
        return {"grade": self.grade, "world": self.world,
                "phase": self._phase, "challenge": self._exam.public.copy(),
                "training_probe_access": False}

    def submit(self, answer: Any) -> dict:
        if self._phase != "examination":
            raise ValueError("No active exam")
        self._answer = answer
        self._append("exam_submission", {"answer_digest": hashlib.sha256(
            json.dumps(answer, sort_keys=True, default=str).encode()).hexdigest()})
        self._phase = "closed"
        correct = (type(answer) is type(self._exam.expected) and answer == self._exam.expected)
        self._append("exam_closed", {"correct": correct})
        return {"grade": self.grade, "world": self.world,
                "provisional_score": {"correct": int(correct), "total": 1},
                "retention_verified": False, "transfer_verified": False,
                "independent_exam_verified": False, "certified": False,
                "receipt_head": self._head}

    def receipt_export(self) -> tuple[dict, ...]:
        return tuple(json.loads(json.dumps(r)) for r in self._events)
