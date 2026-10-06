"""Extract two colored block moves from an overhead, marked workspace.

Uses the opening frame for block/target separation and reports tracking gaps.
This is geometric perception, not a VLM, VLA, or learned policy.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def color_mask(frame, name):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    if name == "red":
        mask = cv2.inRange(hsv, (0, 140, 55), (10, 255, 255))
        mask |= cv2.inRange(hsv, (170, 140, 55), (179, 255, 255))
    else:
        mask = cv2.inRange(hsv, (98, 140, 45), (135, 255, 255))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))


def centroid(contour):
    m = cv2.moments(contour)
    return np.array([m["m10"] / m["m00"], m["m01"] / m["m00"]])


def marker_corners(frame):
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, 0), (179, 110, 105))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 0.001 * w * h:
            continue
        x, y, cw, ch = cv2.boundingRect(c)
        if not 0.65 < cw / ch < 1.6:
            continue
        candidates.append((area, centroid(c)))
    result = []
    for right, bottom in [(False, False), (True, False), (True, True), (False, True)]:
        matches = [(a, p) for a, p in candidates
                   if (p[0] > 0.72*w if right else p[0] < 0.28*w)
                   and (p[1] > 0.72*h if bottom else p[1] < 0.28*h)]
        if not matches:
            raise ValueError("Four corner markers not found. Supply corners_px in a calibration JSON (TL, TR, BR, BL).")
        result.append(max(matches, key=lambda item: item[0])[1])
    return np.array(result, np.float32)


def initial_scene(frame, workspace_mask):
    scene = {}
    for name in ("red", "blue"):
        mask = color_mask(frame, name) & workspace_mask
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        rings, solids = [], []
        for i, c in enumerate(contours):
            area = cv2.contourArea(c)
            if area < 100 or hierarchy[0, i, 3] != -1:
                continue
            child = hierarchy[0, i, 2]
            hole = cv2.contourArea(contours[child]) if child >= 0 else 0
            if hole / area > 0.3:
                rings.append((area, c, contours[child]))
            else:
                solids.append((area, c))
        if not rings or not solids:
            raise ValueError(f"Could not separate {name} block and target ring in opening frame. Keep hands clear at the start.")
        _, ring, hole = max(rings, key=lambda item: item[0])
        # Ignore the small colored disk at the center of the target.
        goal = centroid(ring)
        solids = [(a, c) for a, c in solids if cv2.pointPolygonTest(ring, tuple(centroid(c)), False) < 0]
        if not solids:
            raise ValueError(f"No {name} block outside its target ring")
        area, block = max(solids, key=lambda item: item[0])
        ring_mask = np.zeros_like(mask)
        cv2.drawContours(ring_mask, [ring], -1, 255, -1)
        cv2.drawContours(ring_mask, [hole], -1, 0, -1)
        ring_mask = cv2.dilate(ring_mask, np.ones((7, 7), np.uint8))
        scene[name] = {"start": centroid(block), "goal": goal, "area": area,
                       "ring_mask": ring_mask, "initial_mask": mask}
    return scene


def extract(video, out, provenance, width=0.50, height=0.35, calibration=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    ok, first = cap.read()
    if not ok:
        raise ValueError(f"Cannot read {video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        raise ValueError("Invalid video FPS")
    config = json.loads(Path(calibration).read_text()) if calibration else {}
    corners = np.array(config["corners_px"], np.float32) if "corners_px" in config else marker_corners(first)
    width, height = config.get("workspace_m", [width, height])
    if width <= 0 or height <= 0 or corners.shape != (4, 2):
        raise ValueError("Invalid workspace dimensions or corner coordinates")
    workspace_mask = np.zeros(first.shape[:2], np.uint8)
    cv2.fillConvexPoly(workspace_mask, corners.astype(np.int32), 255)
    transform = cv2.getPerspectiveTransform(corners, np.float32([[0, 0], [width, 0], [width, height], [0, height]]))
    scene = initial_scene(first, workspace_mask)
    def metric(p):
        return cv2.perspectiveTransform(np.float32([[p]]), transform)[0, 0].round(5).tolist()

    output_video = out / "tracking.mp4"
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError("Cannot create tracking video")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    previous = {n: s["start"].copy() for n, s in scene.items()}
    missing = {n: 0 for n in scene}
    rows, i = [], 0
    final = first
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        final = frame.copy()
        overlay = frame.copy()
        cv2.polylines(overlay, [corners.astype(np.int32)], True, (60, 190, 70), 2)
        for name, info in scene.items():
            mask = color_mask(frame, name) & workspace_mask & ~info["ring_mask"]
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            choices = []
            for c in contours:
                area = cv2.contourArea(c)
                if not 0.15 * info["area"] <= area <= 2.2 * info["area"]:
                    continue
                p = centroid(c)
                dist = np.linalg.norm(p - previous[name])
                if dist > (0.04 + min(missing[name], 20)*0.004) * w:
                    continue
                score = dist / w + 0.07 * abs(np.log(area / info["area"]))
                choices.append((score, p, area))
            p, confidence = None, 0.0
            if choices:
                _, p, area = min(choices, key=lambda item: item[0])
                confidence = float(min(area/info["area"], info["area"]/area))
                previous[name] = p
                missing[name] = 0
                cv2.circle(overlay, tuple(p.astype(int)), 20, (30, 255, 80), 3)
            else:
                missing[name] += 1
            goal = tuple(info["goal"].astype(int))
            cv2.drawMarker(overlay, goal, (255, 230, 80), cv2.MARKER_CROSS, 28, 2)
            cv2.putText(overlay, name + " goal", (goal[0] + 22, goal[1]), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 230, 80), 2)
            rows.append({"frame": i, "time_s": round(i/fps, 4), "object": name,
                         "x_px": "" if p is None else round(float(p[0]), 2),
                         "y_px": "" if p is None else round(float(p[1]), 2),
                         "confidence": round(confidence, 3)})
        cv2.rectangle(overlay, (0, 0), (w, 42), (22, 25, 30), -1)
        cv2.putText(overlay, f"{provenance.upper()} | geometry baseline | gaps are unobserved", (18, 28), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 1)
        writer.write(overlay)
        if i == 0:
            cv2.imwrite(str(out / "extraction.jpg"), overlay)
        i += 1
    cap.release()
    writer.release()
    objects, quality = [], {}
    for name, info in scene.items():
        final_mask = color_mask(final, name)
        roi = np.zeros_like(final_mask)
        radius = int(np.sqrt(info["area"]) * .45)
        cv2.circle(roi, tuple(info["goal"].astype(int)), radius, 255, -1)
        added = cv2.countNonZero(final_mask & roi & ~info["initial_mask"])
        observed = sum(r["object"] == name and r["confidence"] >= .5 for r in rows)
        quality[name] = {"tracking_fraction_confidence_ge_0_5": round(observed/i, 3),
                         "new_color_pixels_at_goal": added,
                         "goal_occupancy_supported": bool(added > .10 * info["area"])}
        objects.append({"id": name, "start_m": metric(info["start"]), "goal_m": metric(info["goal"]),
                        "start_px": info["start"].round(2).tolist(), "goal_px": info["goal"].round(2).tolist()})
    task = {"source": {"kind": provenance, "video": str(Path(video).resolve()),
                       "sha256": hashlib.sha256(Path(video).read_bytes()).hexdigest()},
            "workspace_m": [width, height], "corners_px": corners.round(2).tolist(),
            "objects": objects, "dependencies": [], "arm_speed_m_s": .15,
            "handoff_overhead_s": 1.5,
            "assumptions": ["Planar mapping; no recovered 3D hand trajectory.",
                            "Distinct block moves treated as independent; simultaneity is not proof of independence.",
                            "Workspace dimensions are supplied, not inferred from video.",
                            "Targets are ring centers from the opening frame; final occupancy is a color heuristic."],
            "quality": quality}
    (out / "task.json").write_text(json.dumps(task, indent=2) + "\n")
    with (out / "tracks.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"frames": i, "task": str(out / "task.json"), "quality": quality}, indent=2))
    return task


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--out", type=Path, default=Path("outputs/extracted"))
    parser.add_argument("--provenance", choices=["generated", "personal"], required=True)
    parser.add_argument("--width", type=float, default=.50)
    parser.add_argument("--height", type=float, default=.35)
    parser.add_argument("--calibration", type=Path)
    args = parser.parse_args()
    extract(args.video, args.out, args.provenance, args.width, args.height, args.calibration)


if __name__ == "__main__":
    main()
