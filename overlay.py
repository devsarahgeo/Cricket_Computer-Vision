import cv2
import numpy as np
import os
from ultralytics import YOLO

# ---------------------------
# PATHS
# ---------------------------
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

model_path      = os.path.join(base_dir, "models",      "ball_detector.pt")
video_path      = os.path.join(base_dir, "input_video", "shrt_video.mp4")
homography_path = os.path.join(base_dir, "models",      "homography.npy")
output_path     = os.path.join(base_dir, "output",      "hawkeye_fixed_v3.mp4")

# ---------------------------
# LOAD MODEL & HOMOGRAPHY
# ---------------------------
model = YOLO(model_path)
H     = np.load(homography_path)

def pixel_to_world(x, y):
    pt = np.array([[[x, y]]], dtype=np.float32)
    return cv2.perspectiveTransform(pt, H)[0][0]

# ---------------------------
# VIDEO
# ---------------------------
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
DT  = 1.0 / fps

out = cv2.VideoWriter(output_path,
                      cv2.VideoWriter_fourcc(*"mp4v"),
                      fps, (w, h))

# ---------------------------
# ZONE DEFINITIONS
# Tuned from pixel analysis of this specific video / camera angle.
#
#  Pitch clay surface:   x=[430,985]   y=[245,645]
#  Full detection zone:  x=[380,1100]  y=[80,800]
#
#  BOUNCE ZONE: the horizontal band where the ball hits the pitch.
#  In this camera perspective the pitch mid-section sits at roughly y=430–600.
#  The ball bounces somewhere inside this band.
#  → We only START drawing the trail once the ball has been seen
#    inside the bounce zone (i.e. it has already hit the pitch).
#  → While the ball is still above the bounce zone (y < BOUNCE_Y_TOP)
#    it is in the air → show bounding box only, NO trail yet.
# ---------------------------
TRAJ_X1, TRAJ_X2   = 380, 1100    # left / right detection limits
TRAJ_Y1, TRAJ_Y2   = 80,  800     # top  / bottom detection limits
BOUNCE_Y_TOP        = 430          # above this y → ball still in air (no trail)
BOUNCE_Y_BOTTOM     = 650          # below this y → ball past the bounce, rolling/done

# ---------------------------
# FILTERS
# ---------------------------
MAX_PIXEL_JUMP  = 250   # px/frame — rejects teleporting false positives
MIN_CONSISTENCY = 3     # minimum accepted points before direction lock kicks in

# ---------------------------
# KALMAN FILTER
# ---------------------------
def make_kalman():
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix   = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
    kf.transitionMatrix    = np.array([[1,0,1,0],[0,1,0,1],
                                       [0,0,1,0],[0,0,0,1]], np.float32)
    kf.processNoiseCov     = np.eye(4, dtype=np.float32) * 1e-2
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
    kf.errorCovPost        = np.eye(4, dtype=np.float32)
    return kf

# ---------------------------
# STATE
# ---------------------------
kf            = make_kalman()
kf_active     = False

# Separate lists:
#   air_trail   — points collected before the bounce (NOT drawn as trail)
#   pitch_trail — points collected after the bounce (drawn as trail)
air_trail   = []
pitch_trail = []

bounced       = False   # has the ball passed through the bounce zone this delivery?
prev_pos      = None
prev_world    = None
gap_frames    = 0
MAX_GAP       = 8
TRAIL_MAX     = 80

frame_num     = 0

# ---------------------------
# MAIN LOOP
# ---------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_num += 1

    results      = model(frame, verbose=False)[0]
    ball_accepted = None
    raw_box       = None   # (x1,y1,x2,y2) for drawing bounding box

    # ---- DETECTION --------------------------------------------------------
    if results.boxes is not None and len(results.boxes) > 0:
        boxes   = results.boxes.xyxy.cpu().numpy()
        confs   = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()

        candidates = sorted(
            [(confs[i], boxes[i]) for i in range(len(boxes)) if int(classes[i]) == 1],
            reverse=True
        )

        for conf, box in candidates:
            x1, y1, x2, y2 = map(int, box)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Must be inside detection zone
            if not (TRAJ_X1 < cx < TRAJ_X2 and TRAJ_Y1 < cy < TRAJ_Y2):
                continue

            # Max pixel jump
            if prev_pos is not None:
                if np.hypot(cx - prev_pos[0], cy - prev_pos[1]) > MAX_PIXEL_JUMP:
                    continue

            # Direction consistency (only after enough points)
            all_pts = air_trail + pitch_trail
            if len(all_pts) >= MIN_CONSISTENCY:
                recent_dy = all_pts[-1][1] - all_pts[0][1]
                new_dy    = cy - all_pts[-1][1]
                if recent_dy != 0 and new_dy != 0:
                    if np.sign(recent_dy) != np.sign(new_dy) and abs(new_dy) > 20:
                        continue

            ball_accepted = (cx, cy)
            raw_box       = (x1, y1, x2, y2)
            break

    # ---- UPDATE STATE -----------------------------------------------------
    if ball_accepted:
        cx, cy     = ball_accepted
        gap_frames = 0

        # Feed Kalman
        meas = np.array([[np.float32(cx)], [np.float32(cy)]])
        if not kf_active:
            kf.statePre = np.array([[cx],[cy],[0],[0]], dtype=np.float32)
            kf_active   = True
        kf.correct(meas)
        kf.predict()

        # ----- BOUNDING BOX (always drawn when ball is detected) -----
        x1, y1, x2, y2 = raw_box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

        # Speed
        world_curr = pixel_to_world(cx, cy)
        if prev_world is not None and prev_pos is not None:
            dist  = np.linalg.norm(world_curr - prev_world)
            speed = (dist / DT) * 3.6
            if 10 < speed < 250:
                cv2.putText(frame, f"Speed: {speed:.1f} km/h",
                            (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (255, 255, 255), 2)

        prev_pos   = (cx, cy)
        prev_world = world_curr

        # ----- BOUNCE DETECTION -----
        # Ball enters the bounce zone → mark as bounced from now on.
        if BOUNCE_Y_TOP <= cy <= BOUNCE_Y_BOTTOM:
            bounced = True

        if not bounced:
            # Ball still in air → accumulate but don't draw
            air_trail.append((cx, cy))
        else:
            # Ball has bounced → add to drawable trail
            # On the first post-bounce point, carry over the last air point
            # so the trail starts right at the bounce spot, not mid-air.
            if len(pitch_trail) == 0 and len(air_trail) > 0:
                pitch_trail.append(air_trail[-1])   # anchor at bounce point
            pitch_trail.append((cx, cy))
            if len(pitch_trail) > TRAIL_MAX:
                pitch_trail = pitch_trail[-TRAIL_MAX:]

    else:
        gap_frames += 1

        # Kalman interpolation — only extend the POST-bounce trail
        if kf_active and gap_frames <= MAX_GAP and bounced:
            pred = kf.predict()
            px   = int(pred[0][0])
            py   = int(pred[1][0])

            in_zone    = TRAJ_X1 < px < TRAJ_X2 and TRAJ_Y1 < py < TRAJ_Y2
            consistent = True
            all_pts    = air_trail + pitch_trail
            if len(all_pts) >= MIN_CONSISTENCY:
                recent_dy = all_pts[-1][1] - all_pts[0][1]
                new_dy    = py - all_pts[-1][1]
                if recent_dy != 0 and new_dy != 0:
                    if np.sign(recent_dy) != np.sign(new_dy) and abs(new_dy) > 20:
                        consistent = False

            if in_zone and consistent:
                pitch_trail.append((px, py))
                prev_pos = (px, py)
                if len(pitch_trail) > TRAIL_MAX:
                    pitch_trail = pitch_trail[-TRAIL_MAX:]
        elif gap_frames > MAX_GAP:
            # New delivery → full reset
            air_trail   = []
            pitch_trail = []
            bounced     = False
            prev_pos    = None
            prev_world  = None
            kf_active   = False
            kf          = make_kalman()

    # ---- DRAW FADING TRAIL (pitch_trail only — never air_trail) -----------
    n = len(pitch_trail)
    for i in range(1, n):
        p1 = pitch_trail[i - 1]
        p2 = pitch_trail[i]

        # Clip to safe zone (no lines outside pitch area)
        def in_draw_zone(p):
            return TRAJ_X1 < p[0] < TRAJ_X2 and TRAJ_Y1 < p[1] < TRAJ_Y2

        if not (in_draw_zone(p1) and in_draw_zone(p2)):
            continue

        alpha     = i / n
        thickness = max(1, int(alpha * 5))
        green     = int((1 - alpha) * 180)
        cv2.line(frame, p1, p2, (0, green, 255), thickness)

    out.write(frame)

    cv2.imshow("HawkEye v3", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("Saved →", output_path)