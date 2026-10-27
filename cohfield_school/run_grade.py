"""Operator-only CLI for inspecting unpromoted school worlds.

No CohField learner is attached and no grade certificate is issued.
"""
from __future__ import annotations
import argparse
import json
from .grade_worlds import GradeSession, WORLDS


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a teacher-side NELS world")
    parser.add_argument("--grade", choices=tuple(WORLDS), required=True)
    parser.add_argument("--world", required=True)
    parser.add_argument("--seed", default="demo")
    parser.add_argument("--probe", action="append", default=[])
    parser.add_argument("--answer-json", help="Submit a single JSON exam response")
    args = parser.parse_args()
    session = GradeSession(args.grade, args.world, args.seed)
    print(json.dumps({"training": session.training_prompt()}, indent=2))
    for probe in args.probe:
        print(json.dumps({"probe": probe, "result": session.intervene(probe)}, indent=2))
    print(json.dumps({"examination": session.begin_exam()}, indent=2))
    if args.answer_json is not None:
        result = session.submit(json.loads(args.answer_json))
        print(json.dumps({"provisional_result": result}, indent=2))
    else:
        print("No exam submission: no score, no certification")


if __name__ == "__main__":
    main()
