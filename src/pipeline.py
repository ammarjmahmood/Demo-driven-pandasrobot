"""Run phone/generated video -> task -> resource plans -> physical rollouts."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .extract import extract
from .plan import choose, coordination_text
from .simulate import run


def comparison(out):
    paths = [out / "tracking.mp4", out / "one_arm/rollout.mp4", out / "two_arms/rollout.mp4"]
    captures = [cv2.VideoCapture(str(p)) for p in paths]
    if not all(c.isOpened() for c in captures):
        raise RuntimeError("Cannot read comparison inputs")
    fps = [c.get(cv2.CAP_PROP_FPS) for c in captures]
    counts = [int(c.get(cv2.CAP_PROP_FRAME_COUNT)) for c in captures]
    length = max(n/f for n, f in zip(counts, fps))
    writer = cv2.VideoWriter(str(out / "comparison.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 25, (1440, 400))
    if not writer.isOpened():
        raise RuntimeError("Cannot write comparison")
    previous = [None] * 3
    last_indices = [-1] * 3
    labels = ["VIDEO TO EXTRACTED GOALS", "ONE ARM / SEQUENTIAL", "TWO ARMS / COLLABORATIVE"]
    for i in range(round(length * 25)):
        canvas = np.full((400, 1440, 3), (24, 22, 20), np.uint8)
        for index, (cap, f, n) in enumerate(zip(captures, fps, counts)):
            target = min(round(i/25*f), n-1)
            while last_indices[index] < target:
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(f"Unexpected video end in {paths[index]}")
                previous[index] = frame
                last_indices[index] += 1
            frame = previous[index]
            scale = min(480/frame.shape[1], 320/frame.shape[0])
            resized = cv2.resize(frame, None, fx=scale, fy=scale)
            x = index*480+(480-resized.shape[1])//2
            y = 48+(320-resized.shape[0])//2
            canvas[y:y+resized.shape[0], x:x+resized.shape[1]] = resized
            cv2.putText(canvas, labels[index], (index*480+16, 29), cv2.FONT_HERSHEY_SIMPLEX, .57, (235, 240, 245), 1)
        cv2.putText(canvas, f"Shared clock {i/25:.1f}s | panels hold their final frame when finished | scripted physics baseline", (18, 388), cv2.FONT_HERSHEY_SIMPLEX, .52, (180, 195, 210), 1)
        writer.write(canvas)
    for cap in captures:
        cap.release()
    writer.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--out", type=Path, default=Path("outputs/demo"))
    parser.add_argument("--provenance", choices=["generated", "personal"], required=True)
    parser.add_argument("--width", type=float, default=.5)
    parser.add_argument("--height", type=float, default=.35)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    task = extract(args.video, args.out, args.provenance, args.width, args.height, args.calibration)
    plans = choose(task)
    (args.out / "plans.json").write_text(json.dumps(plans, indent=2)+"\n")
    (args.out / "coordination.txt").write_text(coordination_text(plans["chosen"])+"\n")
    results = [run(task, arms, args.out / folder, not args.no_render)
               for arms, folder in [(1, "one_arm"), (2, "two_arms")]]
    summary = {"input_provenance": args.provenance, "chosen_arms": plans["chosen_arms"],
               "results": [{k: r[k] for k in ["arms", "success", "placement_success", "simulated_seconds", "objects",
                                              "interarm_contact_pairs", "nonbase_robot_table_contact_pairs"]} for r in results],
               "limitations": ["Single scene, scripted controller, no learned policy.",
                               "Controller uses simulator object state.",
                               "Generated footage is development input, not applicant-collected data.",
                               "Planar calibration, fixed 4cm blocks and supplied workspace dimensions.",
                               "Path clearance is a heuristic; actual robot contact is checked during the rollout."]}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    if not args.no_render:
        comparison(args.out)
    print(f"Pipeline complete: {args.out.resolve()}")


if __name__ == "__main__":
    main()
