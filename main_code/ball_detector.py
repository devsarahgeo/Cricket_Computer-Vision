"""
Ball Detector with Interpolation

Two-pass approach:
  Pass 1 — detect_all_positions(): run ball_detector.pt on every frame,
            record {frame_num: (cx, cy)} for confident detections.
  Interpolation — interpolate_positions(): linearly fill gaps between
            known detections (gaps larger than MAX_GAP are left empty —
            ball is assumed out of view).
  Pass 2 — draw_ball_at(): called per frame during the main player loop;
            draws a solid red arrow for real detections and a dashed
            orange arrow for interpolated positions.
"""


import cv2
import os
import numpy as np
from ultralytics import YOLO

# PATHS
base_dir        = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BALL_MODEL_PATH = os.path.join(base_dir, "models", "ball_detector.pt")

# CONSTANTS
MAX_GAP        = 15     # frames, gaps larger than this mean ball is gone
CONF_THRES     = 0.3

REAL_COLOR     = (0,   0, 255)   # red, actual detection
INTERP_COLOR   = (0, 165, 255)   # orange, interpolated position
ARROW_OFFSET   = 40   # px above ball, height of the triangle

def load_ball_model():
    return YOLO(BALL_MODEL_PATH)


def detect_all_positions(video_path, model, conf_thres=CONF_THRES):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    detections = {}
    frame_num  = 0

    print("  [Ball] Pass 1 — detecting ball positions...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1

        results = model(frame, verbose=False)[0]
        if results.boxes is None or len(results.boxes) == 0:
            continue

        confs    = results.boxes.conf.cpu().numpy()
        best_idx = int(confs.argmax())
        if float(confs[best_idx]) < conf_thres:
            continue

        x1, y1, x2, y2 = map(int, results.boxes.xyxy[best_idx].cpu().numpy())
        detections[frame_num] = ((x1 + x2) // 2, (y1 + y2) // 2)

    cap.release()
    print(f"  [Ball] Detected in {len(detections)}/{total} frames.")
    return detections, total


def interpolate_positions(detections, total_frames, max_gap=MAX_GAP):
    """
    Linearly interpolate ball positions between known detections.

    Parameters
    ----------
    detections   : {frame_num: (cx, cy)}
    total_frames : int
    max_gap      : int, gaps larger than this are not interpolated

    Returns
    -------
    dict  {frame_num: (cx, cy, is_interpolated)}
        is_interpolated=False: real detection
        is_interpolated=True: linearly filled gap
    """
    if not detections:
        return {}

    positions = {}

    # Mark real detections
    for f, (cx, cy) in detections.items():
        positions[f] = (cx, cy, False)

    # Find all detected frame numbers in order
    known = sorted(detections.keys())

    # Fill gaps between consecutive detections
    for i in range(len(known) - 1):
        f1, f2 = known[i], known[i + 1]
        gap = f2 - f1 - 1
        if gap == 0 or gap > max_gap:
            continue  # no gap, or too large to trust interpolation

        x1, y1 = detections[f1]
        x2, y2 = detections[f2]

        for fi in range(f1 + 1, f2):
            t  = (fi - f1) / (f2 - f1)
            cx = int(x1 + t * (x2 - x1))
            cy = int(y1 + t * (y2 - y1))
            positions[fi] = (cx, cy, True)

    interp_count = sum(1 for v in positions.values() if v[2])
    print(f"  [Ball] Interpolated {interp_count} additional frames.")
    return positions


def draw_ball_at(frame, positions, frame_num):
    """
    Draw arrow annotation for the ball on the given frame.

    Solid red arrow   → real detection.
    Dashed orange arrow → interpolated position.

    Parameters
    ----------
    frame      : np.ndarray  (modified in-place)
    positions  : dict returned by interpolate_positions()
    frame_num  : int
    """
    if frame_num not in positions:
        return frame

    cx, cy, is_interp = positions[frame_num]
    color = INTERP_COLOR if is_interp else REAL_COLOR

    # Filled triangle pointing downward — tip touches the ball centre
    tip_y    = cy
    base_y   = max(0, cy - ARROW_OFFSET)
    half_w   = 10   # half-width of the triangle base

    triangle = np.array([
        [cx,          tip_y ],   # bottom tip  (points at ball)
        [cx - half_w, base_y],   # top-left
        [cx + half_w, base_y],   # top-right
    ], dtype=np.int32)

    cv2.fillPoly(frame, [triangle], color)

    cv2.polylines(frame, [triangle], isClosed=True, color=color, thickness=1)

    return frame
