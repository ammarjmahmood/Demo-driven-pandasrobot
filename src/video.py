"""Synthetic capture fixture and simple color-based object tracking.

The generated clip is development data only. It is never a substitute for the
applicant's personally recorded footage.
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


COLORS = {
    "red": ((0, 90, 80), (10, 255, 255)),
    "blue": ((100, 90, 80), (130, 255, 255)),
}


def center(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if not (25 <= w <= 110 and 25 <= h <= 110):
            continue
        if cv2.contourArea(contour) < 150:
            continue
        roi = mask[y:y+h, x:x+w]
        fill = cv2.countNonZero(roi) / (w * h)
        if fill < 0.35:
            continue
        candidates.append((fill * cv2.contourArea(contour), contour))
    if not candidates:
        return None
    contour = max(candidates, key=lambda item: item[0])[1]
    m = cv2.moments(contour)
    return (m["m10"] / m["m00"], m["m01"] / m["m00"])


def track(path, output):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        raise RuntimeError("video FPS unavailable")
    rows = []
    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        for name, (lo, hi) in COLORS.items():
            mask = cv2.inRange(hsv, np.array(lo), np.array(hi))
            if name == "red":
                mask |= cv2.inRange(hsv, np.array((170, 90, 80)), np.array((179, 255, 255)))
            xy = center(mask)
            rows.append({"frame": frame_index, "time_s": round(frame_index / fps, 4),
                         "object": name, "x_px": "" if xy is None else round(xy[0], 2),
                         "y_px": "" if xy is None else round(xy[1], 2)})
        frame_index += 1
    cap.release()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame", "time_s", "object", "x_px", "y_px"])
        writer.writeheader()
        writer.writerows(rows)
    return frame_index


def synthetic(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fps, width, height, frames = 20, 640, 480, 100
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("MP4 writer unavailable")
    for i in range(frames):
        frame = np.full((height, width, 3), 225, dtype=np.uint8)
        cv2.rectangle(frame, (40, 40), (600, 440), (30, 30, 30), 2)
        red_t = min(max((i - 15) / 45, 0), 1)
        blue_t = min(max((i - 25) / 50, 0), 1)
        for start, goal, t, color in [((145, 155), (495, 160), red_t, (0, 0, 230)),
                                      ((150, 335), (490, 320), blue_t, (230, 70, 20))]:
            cv2.rectangle(frame, (goal[0] - 25, goal[1] - 25), (goal[0] + 25, goal[1] + 25), (90, 90, 90), 2)
            x = round(start[0] * (1 - t) + goal[0] * t)
            y = round(start[1] * (1 - t) + goal[1] * t)
            cv2.circle(frame, (x, y), 18, color, -1)
        cv2.putText(frame, "SYNTHETIC DEVELOPMENT VIDEO", (110, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
        writer.write(frame)
    writer.release()


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    make = sub.add_parser("synthetic")
    make.add_argument("video", type=Path)
    extract = sub.add_parser("track")
    extract.add_argument("video", type=Path)
    extract.add_argument("csv", type=Path)
    args = parser.parse_args()
    if args.command == "synthetic":
        synthetic(args.video)
    else:
        print(f"tracked {track(args.video, args.csv)} frames -> {args.csv}")


if __name__ == "__main__":
    main()
