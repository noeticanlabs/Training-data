"""Aggregate a three-world course without claiming learner qualification."""
from __future__ import annotations

from typing import Any
from .grade_worlds import GradeSession, WORLDS


class GradeCourse:
    """School-controlled course; never a self-certifying grade authority.

    This only aggregates one short toy examination per named world. Real
    graduation requires independent held-out testing and CohField retention.
    """

    def __init__(self, grade: str, seed: str, budget_per_world: int = 5):
        if grade not in WORLDS:
            raise ValueError("Unknown grade")
        self.grade = grade
        self.worlds = tuple(WORLDS[grade])
        self._sessions = {world: GradeSession(grade, world, seed, budget_per_world)
                          for world in self.worlds}
        self._results: dict[str, dict] = {}
        self._exams_started: set[str] = set()

    def training(self, world: str) -> dict:
        return self._get(world).training_prompt()

    def intervene(self, world: str, probe: str) -> dict:
        return self._get(world).intervene(probe)

    def exam(self, world: str) -> dict:
        if world in self._exams_started:
            raise ValueError("Exam already started")
        response = self._get(world).begin_exam()
        self._exams_started.add(world)
        return response

    def submit(self, world: str, answer: Any) -> dict:
        if world not in self._exams_started or world in self._results:
            raise ValueError("World examination is unavailable")
        result = self._get(world).submit(answer)
        self._results[world] = result
        return result

    def _get(self, world: str) -> GradeSession:
        if world not in self._sessions:
            raise ValueError("Unknown world")
        return self._sessions[world]

    def report(self) -> dict:
        complete = len(self._results) == len(self.worlds)
        correct = sum(v["provisional_score"]["correct"] for v in self._results.values())
        return {"grade": self.grade, "world_count": len(self.worlds),
                "examined_worlds": len(self._results), "all_worlds_examined": complete,
                "provisional_accuracy": (correct / len(self.worlds) if complete else None),
                "results": {world: self._results[world] for world in self.worlds if world in self._results},
                "native_learning_verified": False,
                "restart_retention_verified": False,
                "independent_transfer_verified": False,
                "regression_verified": False,
                "independent_examination_verified": False,
                "certified": False,
                "grade_status": "UNPROMOTED_SCHOOL_EXERCISE"}
