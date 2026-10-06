"""Inspectable task scheduling and arm-count selection for two block moves."""
import json
import math
import sys
from pathlib import Path

import numpy as np


# Matches the current scripted controller; deliberately not a learned timing model.
PHASE_SECONDS = {"approach": 1.5, "descend": 1.2, "close": .7, "lift": 1.2,
                 "transport": 1.5, "lower": 1.2, "release": .6, "retreat": 1.2, "settle": .3}
MOVE_SECONDS = sum(PHASE_SECONDS.values())
INITIAL_FINAL_SECONDS = .8


def validate(task):
    objects = task.get("objects", [])
    if len(objects) != 2 or {o["id"] for o in objects} != {"red", "blue"}:
        raise ValueError("This baseline supports exactly one red and one blue object")
    workspace = task["workspace_m"]
    if len(workspace) != 2 or any(not math.isfinite(v) or v <= 0 for v in workspace):
        raise ValueError("Workspace dimensions must be positive and finite")
    for obj in objects:
        for key in ("start_m", "goal_m"):
            point = obj[key]
            if len(point) != 2 or any(not math.isfinite(v) or not 0 <= v <= maximum for v, maximum in zip(point, workspace)):
                raise ValueError(f"{obj['id']} {key} lies outside the workspace")
    overhead = task.get("handoff_overhead_s", 1.5)
    if not math.isfinite(overhead) or overhead < 0:
        raise ValueError("Recruitment overhead must be nonnegative and finite")
    ids = {o["id"] for o in objects}
    for edge in task.get("dependencies", []):
        if len(edge) != 2 or any(n not in ids for n in edge) or edge[0] == edge[1]:
            raise ValueError("Dependencies must be [predecessor, successor] object IDs")


def ordered_objects(task):
    validate(task)
    remaining = {o["id"]: o for o in sorted(task["objects"], key=lambda o: o["start_m"][0])}
    edges = task.get("dependencies", [])
    result = []
    while remaining:
        ready = [n for n in remaining if not any(b == n and a in remaining for a, b in edges)]
        if not ready:
            raise ValueError("Dependency cycle")
        name = ready[0]
        result.append(remaining.pop(name))
    return result


def path_clearance(task):
    """Conservative planar transport-corridor check, not full robot collision planning."""
    lines = [np.linspace(o["start_m"], o["goal_m"], 101) for o in task["objects"]]
    return float(np.linalg.norm(lines[0][:, None, :] - lines[1][None, :, :], axis=2).min())


def plan(task, arms):
    objects = ordered_objects(task)
    if arms not in (1, 2):
        raise ValueError("arms must be 1 or 2")
    clearance = path_clearance(task)
    parallel = arms == 2 and not task.get("dependencies") and clearance >= .12
    assignments = {o["id"]: f"arm_{i+1}" for i, o in enumerate(sorted(objects, key=lambda o: o["start_m"][0]))}
    steps = []
    for index, obj in enumerate(objects):
        start = .4 + (0 if parallel else index * MOVE_SECONDS)
        steps.append({"arm": "arm_1" if arms == 1 else assignments[obj["id"]], "object": obj["id"],
                      "start_s": round(start, 3), "end_s": round(start + MOVE_SECONDS, 3)})
    seconds = INITIAL_FINAL_SECONDS + MOVE_SECONDS * (1 if parallel else 2)
    penalty = task.get("handoff_overhead_s", 1.5) if arms == 2 else 0
    reason = "Independent moves with separated transport corridors" if parallel else (
        "Task dependency requires ordering" if task.get("dependencies") else
        "One arm executes sequentially" if arms == 1 else "Transport corridors overlap; serialize to reduce collision risk")
    return {"arms": arms, "parallel": parallel, "estimated_s": round(seconds, 3),
            "recruitment_penalty_s": penalty, "selection_cost_s": round(seconds+penalty, 3),
            "transport_clearance_m": round(clearance, 4), "reason": reason, "steps": steps}


def choose(task):
    candidates = [plan(task, n) for n in (1, 2)]
    chosen = min(candidates, key=lambda p: (p["selection_cost_s"], p["arms"]))
    return {"candidates": candidates, "chosen_arms": chosen["arms"], "chosen": chosen}


def coordination_text(result):
    steps = result["steps"]
    if result["arms"] == 1:
        return f"Coordinator: One arm will move {steps[0]['object']}, then {steps[1]['object']}."
    first, second = steps
    text = (f"Coordinator: Request a second arm. {first['arm']} takes {first['object']}; "
            f"{second['arm']} takes {second['object']}.\n")
    return text + ("Coordinator: The planned transport corridors are separated; start together." if result["parallel"]
                   else f"Coordinator: {second['arm']} waits until {first['arm']} finishes its move.")


def main(path):
    result = choose(json.loads(Path(path).read_text()))
    print(json.dumps(result, indent=2))
    print(coordination_text(result["chosen"]))


if __name__ == "__main__":
    main(sys.argv[1])
