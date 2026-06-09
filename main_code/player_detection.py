"""
This is the main entry pt in script

Cricket Player Tracking — ByteTrack via Ultralytics YOLOv8
==========================================================
Pipeline:
  1. Load fine-tuned YOLO model (player_detector.pt)
  2. Set up video I/O
  3. Detection + ByteTrack tracking loop
  4. Visualisation (bounding box + "Player ID: <id>")
  5. Resource cleanup

"""

import cv2
import os
from ultralytics import YOLO
from ball_detector import (load_ball_model, detect_all_positions,
                           interpolate_positions, draw_ball_at)
from team_player_assignment import (ask_team_colors, load_clip_model,
                                    classify_player_team, get_box_color)
from keypoint_detection import (CricketAnalytics, build_split_frame,
                                OUTPUT_DIR, SPLIT_PANEL_W)

# MODEL LOADING
base_dir        = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
model_path      = os.path.join(base_dir, "models", "player_detector.pt")
video_path      = os.path.join(base_dir, "input_video", "shrt_video.mp4")
output_path     = os.path.join(base_dir, "output",      "player_tracking.mp4")

# Fine-tuned model that detects: batsman, bowler, player, umpire, wicket keeper
model = YOLO(model_path)

# Team assignment — ask for jersey colours and load CLIP once
team_colors  = ask_team_colors()
clip_model, clip_processor = load_clip_model()

# Ball detector — detect + interpolate before main loop
ball_model  = load_ball_model()
ball_dets, _ball_total = detect_all_positions(video_path, ball_model)
ball_positions         = interpolate_positions(ball_dets, _ball_total)

# Cricket analytics (keypoint homography + tactical view)
analytics = CricketAnalytics(fps=0, video_w=0, video_h=0)  # dims set after cap open
analytics.load_model()

# Confidence threshold — detections below this are ignored before tracking
CONF_THRES = 0.1

# Bounding box + text colour (BGR)
BBOX_COLOR  = (0, 255, 0)    # green
TEXT_COLOR  = (255, 255, 255) # white

# VIDEO SETUP
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(output_path,
                      cv2.VideoWriter_fourcc(*"mp4v"),
                      fps, (w, h))

# Split-screen analytics video (original + tactical panel side by side)
analytics_path = os.path.join(base_dir, "output", "analytics_tracking.mp4")
out_analytics  = cv2.VideoWriter(analytics_path,
                                  cv2.VideoWriter_fourcc(*"mp4v"),
                                  fps, (w + SPLIT_PANEL_W + 2, h))

# Patch analytics dims now that we have them
analytics.fps     = max(fps, 1)
analytics.video_w = w
analytics.video_h = h

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video       : {total_frames} frames @ {fps} fps  ({w}x{h})")
print(f"Model       : {model_path}")
print(f"Output      : {output_path}")
print(f"Analytics   : {analytics_path}")
print("-" * 55)

# DETECTION + TRACKING LOOP
frame_num = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_num += 1

    # Progress indicator every 30 frames
    if frame_num % 30 == 0 or frame_num == 1:
        pct = (frame_num / total_frames * 100) if total_frames > 0 else 0
        print(f"  Frame {frame_num}/{total_frames}  ({pct:.1f}%)")

    # Run detection + ByteTrack in one call.
    # persist=True  → Kalman state is carried across frames (stable IDs).
    # tracker="bytetrack.yaml" → use ByteTrack association algorithm.
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=CONF_THRES,
        verbose=False,
    )[0]

    if results.boxes is None:
        tactical = analytics.process_frame(
            frame, frame_num, [], None, team_colors)
        out.write(frame)
        out_analytics.write(build_split_frame(frame, tactical))
        continue

    # VISUALISATION  +  collect tracks for analytics
    player_tracks = []   # (track_id, x1, y1, x2, y2, team_idx)

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        track_id = int(box.id[0]) if box.id is not None else -1
        if track_id == -1:
            continue

        # Crop player region and classify team via CLIP
        crop     = frame[max(0, y1):y2, max(0, x1):x2]
        team_idx = classify_player_team(crop, clip_model, clip_processor, team_colors)

        # Skip players whose jersey doesn't match any entered team colour
        if team_idx == -1:
            continue

        box_color = get_box_color(team_idx, team_colors)

        # Bounding box in team colour
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

        # ID number only, top-left of box
        cv2.putText(frame, str(track_id),
                    (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    box_color, 2)

        player_tracks.append((track_id, x1, y1, x2, y2, team_idx))

    # Pass 2 — draw ball arrow (real or interpolated) for this frame
    draw_ball_at(frame, ball_positions, frame_num)

    # Current ball pixel position for analytics (use real detection if available)
    ball_entry  = ball_positions.get(frame_num)
    ball_pos_px = (ball_entry[0], ball_entry[1]) if ball_entry else None

    # Analytics: keypoint detection, homography update, tactical view
    tactical = analytics.process_frame(
        frame, frame_num, player_tracks, ball_pos_px, team_colors)

    out.write(frame)
    out_analytics.write(build_split_frame(frame, tactical))

# RESOURCE CLEANUP  +  ANALYTICS EXPORT
cap.release()
out.release()
out_analytics.release()
cv2.destroyAllWindows()

# Flush any in-progress delivery and export all analytics
analytics.new_delivery()
analytics.export_all(OUTPUT_DIR)

print(f"\nDone.")
print(f"  Player tracking  → {output_path}")
print(f"  Analytics video  → {analytics_path}")
