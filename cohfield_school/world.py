"""NELS-001: deterministic, bounded, teacher-owned experience world.

The student's observations are revealed only through intervene(); examination
outcomes remain hidden until the session is closed. This is an educational lab,
NOT secure independent certification: its rule generator is visible in source.
"""
from __future__ import annotations

from hashlib import sha256
from itertools import product
import json
from typing import Any

RULES = ("and", "or", "xor", "nand")
ZERO_HASH = "0" * 64


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")


def _pair(a: bool, b: bool) -> tuple[bool, bool]:
    if type(a) is not bool or type(b) is not bool:
        raise ValueError("Features must be exact booleans")
    return a, b


class ExperienceWorld:
    """A finite intervention world with auditable, nonauthoritative receipts.

    This deliberately exercises the schooling protocol, not intelligence.
    At most two of the four truth-table states can be taught, leaving unseen
    assignments for a same-world examination.
    """

    def __init__(self, seed: str, *, observation_budget: int = 2):
        if not isinstance(seed, str) or not seed or len(seed) > 128:
            raise ValueError("A bounded nonempty seed is required")
        if type(observation_budget) is not int or not 1 <= observation_budget <= 2:
            raise ValueError("Training budget must be one or two observations")
        digest = sha256(("nels.world.v0\0" + seed).encode("utf-8")).digest()
        self._rule = RULES[digest[0] % len(RULES)]
        self.session_id = sha256(("nels.session.v0\0" + seed).encode()).hexdigest()[:24]
        self.observation_budget = observation_budget
        self.phase = "training"
        self._observed: dict[tuple[bool, bool], bool] = {}
        self._exam_targets: tuple[tuple[bool, bool], ...] = ()
        self._predictions: dict[tuple[bool, bool], bool] = {}
        self._head = ZERO_HASH
        self._receipts: list[dict] = []
        self._append("begin", {"budget": observation_budget})

    def _outcome(self, pair: tuple[bool, bool]) -> bool:
        a, b = pair
        if self._rule == "and":
            return a and b
        if self._rule == "or":
            return a or b
        if self._rule == "xor":
            return a != b
        return not (a and b)

    def _append(self, event: str, payload: dict) -> None:
        body = {
            "session_id": self.session_id,
            "sequence": len(self._receipts),
            "event": event,
            "payload": payload,
            "previous_hash": self._head,
        }
        body["receipt_hash"] = sha256(
            b"nels.experience.receipt.v0\0" + _canonical(body)
        ).hexdigest()
        self._head = body["receipt_hash"]
        self._receipts.append(body)

    def public_state(self) -> dict:
        return {
            "session_id": self.session_id,
            "phase": self.phase,
            "features": ["a", "b"],
            "observations_remaining": (
                self.observation_budget - len(self._observed)
                if self.phase == "training" else 0
            ),
            "receipt_head": self._head,
            "authority": "none",
        }

    def intervene(self, a: bool, b: bool) -> dict:
        if self.phase != "training":
            raise ValueError("Intervention is unavailable outside training")
        pair = _pair(a, b)
        if pair in self._observed:
            raise ValueError("Repeated intervention is not a new experience")
        if len(self._observed) >= self.observation_budget:
            raise ValueError("Intervention budget exhausted")
        outcome = self._outcome(pair)
        self._observed[pair] = outcome
        observation = {"a": a, "b": b, "outcome": outcome}
        self._append("observation", observation)
        return {**observation, "receipt_hash": self._head}

    def begin_exam(self) -> list[dict]:
        if self.phase != "training" or not self._observed:
            raise ValueError("Exam requires a prior training experience")
        self._exam_targets = tuple(
            pair for pair in product((False, True), repeat=2)
            if pair not in self._observed
        )
        self.phase = "examination"
        self._append("exam_begin", {"target_count": len(self._exam_targets)})
        return [{"a": a, "b": b} for a, b in self._exam_targets]

    def predict(self, a: bool, b: bool, predicted: bool) -> dict:
        if self.phase != "examination" or type(predicted) is not bool:
            raise ValueError("A Boolean exam prediction is required")
        pair = _pair(a, b)
        if pair not in self._exam_targets or pair in self._predictions:
            raise ValueError("Prediction not assigned or already submitted")
        self._predictions[pair] = predicted
        self._append("prediction", {"a": a, "b": b, "predicted": predicted})
        # Never expose the expected outcome during the examination.
        return {"accepted": True, "receipt_hash": self._head}

    def finalize(self) -> dict:
        if self.phase != "examination" or len(self._predictions) != len(self._exam_targets):
            raise ValueError("Every held-out assignment must be attempted")
        correct = sum(
            self._predictions[pair] == self._outcome(pair)
            for pair in self._exam_targets
        )
        self.phase = "closed"
        self._append("exam_end", {
            "correct": correct, "total": len(self._exam_targets),
        })
        return {
            "session_id": self.session_id,
            "provisional_score": {
                "correct": correct, "total": len(self._exam_targets),
            },
            "receipt_head": self._head,
            "certified": False,
            "retention_verified": False,
            "transfer_verified": False,
            "independent_exam_verified": False,
            "note": "Same-world toy examination only; no learner or certification authority.",
        }

    def receipts(self) -> tuple[dict, ...]:
        return tuple(dict(event) for event in self._receipts)
