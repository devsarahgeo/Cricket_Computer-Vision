"""
WIP...........

Cricket Analytics — Keypoint Detection & Tactical View

Uses models/keypoint_detection.pt to detect pitch keypoints,
estimate homography dynamically, and generate a full analytics layer.

Pipeline (called from player_detection.py):
  1. Detect pitch keypoints → compute H (image → world metres)
  2. Convert player foot positions & ball centre to world coordinates
  3. Render split-screen tactical bird's-eye view alongside original frame
  4. Accumulate per-delivery ball track, player movement data
  5. Export on completion: player_stats, ball_positions, heatmaps, pitch map

Public API:
    analytics = CricketAnalytics(fps, video_w, video_h)
    analytics.load_model()
    tactical  = analytics.process_frame(
                    frame, frame_num, player_tracks, ball_pos_px, team_colors)
    analytics.new_delivery()     # call when ball tracker resets
    analytics.export_all()

World coordinate convention:
    Origin (0, 0) = batting-end left corner (near, bottom of image)
    x increases right  (0 → 3.05 m)
    y increases far    (0 → 20.12 m toward bowling end)
"""

import cv2
import os
import json
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage import gaussian_filter
from collections import defaultdict
from ultralytics import YOLO

# PITCH DIMENSIONS (metres)
PITCH_LENGTH   = 20.12   # 22 yards
PITCH_WIDTH    = 3.05    # 10 feet

POPPING_CREASE = 1.22    # distance from stumps to popping crease
BOWLING_CREASE = 2.64    # distance from stumps to bowling crease

KEYPOINT_WORLD = {
    0: (0.0,         0.0          ),   # batting-end left corner
    1: (PITCH_WIDTH, 0.0          ),   # batting-end right corner
    2: (0.0,         PITCH_LENGTH ),   # bowling-end left corner
    3: (PITCH_WIDTH, PITCH_LENGTH ),   # bowling-end right corner
    4: (0.0,         POPPING_CREASE),  # batting popping crease left
    5: (PITCH_WIDTH, POPPING_CREASE),  # batting popping crease right
    6: (0.0,         PITCH_LENGTH - POPPING_CREASE),  # bowling popping crease left
    7: (PITCH_WIDTH, PITCH_LENGTH - POPPING_CREASE),  # bowling popping crease right
    8: (PITCH_WIDTH / 2 - 0.11, 0.0               ),  # batting off-stump
    9: (PITCH_WIDTH / 2,        0.0               ),  # batting middle-stump
   10: (PITCH_WIDTH / 2 + 0.11, 0.0               ),  # batting leg-stump
   11: (PITCH_WIDTH / 2 - 0.11, PITCH_LENGTH      ),  # bowling off-stump
   12: (PITCH_WIDTH / 2,        PITCH_LENGTH      ),  # bowling middle-stump
   13: (PITCH_WIDTH / 2 + 0.11, PITCH_LENGTH      ),  # bowling leg-stump
}

CONF_KP = 0.4   # minimum keypoint confidence to accept

# TACTICAL CANVAS

CANVAS_W = 600
CANVAS_H = 600

FIELD_RADIUS_M = 40.0
PX_PER_M       = CANVAS_W / (2 * FIELD_RADIUS_M)   # 7.5 px / m for 600px canvas

# Pitch centre in world coordinates
PITCH_CX_M = PITCH_WIDTH  / 2    # 1.525 m
PITCH_CY_M = PITCH_LENGTH / 2    # 10.06 m

# Canvas pixel corresponding to the pitch centre
CANVAS_CX = CANVAS_W // 2
CANVAS_CY = CANVAS_H // 2

# BGR colours
_COL_FIELD_DARK  = (34,  139,  34)
_COL_FIELD_LIGHT = (100, 190, 139)
_COL_PITCH       = (139, 190, 100)
_COL_CREASE      = (255, 255, 255)
_COL_STUMP       = (180, 100,  50)
_COL_BALL        = (0,   0,   255)
_COL_BALL_EDGE   = (255, 255, 255)
_PLAYER_RADIUS   = 9   # px on canvas

# Split-screen width of the tactical panel (px)
SPLIT_PANEL_W = 480

# Delivery pitch-map zones 
PITCH_ZONES = [
    (0.0,  2.0,  "Yorker",    "#FF5722"),
    (2.0,  6.0,  "Full",      "#FF9800"),
    (6.0, 10.0,  "Good",      "#4CAF50"),
    (10.0, 14.0, "Short",     "#2196F3"),
    (14.0, 20.12,"Bouncer",   "#9C27B0"),
]

# PATHS
_base_dir     = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
KP_MODEL_PATH = os.path.join(_base_dir, "models", "keypoint_detection.pt")
OUTPUT_DIR    = os.path.join(_base_dir, "output")


# CricketAnalytics CLASS

class CricketAnalytics:
    """Keypoint-driven homography + per-frame tactical analytics."""

    def __init__(self, fps=25, video_w=1920, video_h=1080):
        self.fps     = max(fps, 1)
        self.video_w = video_w
        self.video_h = video_h

        self.kp_model = None

        # Homography state
        self.H              = None
        self.H_valid        = False
        self.last_H_frame   = -999

        # Player analytics  {track_id: [(wx, wy, frame_num), ...]}
        self.player_positions = defaultdict(list)
        self.player_colors    = {}   # track_id → BGR tuple

        # Ball analytics
        self.ball_world_track = []         # [(frame_num, wx, wy)]
        self.ball_speed_kmh   = []         # [(frame_num, speed)]
        self._prev_ball_world = None

        # Delivery tracking
        self.deliveries              = []
        self._cur_delivery_ball      = []  # [(wx, wy, frame_num)]

    # MODEL LOADING

    def load_model(self):
        """Load the keypoint detection model. Returns self for chaining."""
        print("  [KeyPoint] Loading model ...")
        self.kp_model = YOLO(KP_MODEL_PATH)
        print("  [KeyPoint] Ready.")
        return self

    # KEYPOINT DETECTION

    def detect_keypoints(self, frame):
        """
        Run keypoint inference.  Handles both YOLO-pose output
        (results.keypoints) and plain detection output (bbox centres).

        Returns list of (keypoint_index, img_x, img_y, confidence).
        Only returns indices that have a matching entry in KEYPOINT_WORLD.
        """
        if self.kp_model is None:
            return []

        results = self.kp_model(frame, verbose=False)[0]
        detected = []

        # YOLO pose format 
        if results.keypoints is not None:
            kp_data = results.keypoints.data  # (n_det, n_kp, 3)
            if len(kp_data) == 0:
                return []
            # Pick the detection with highest mean keypoint confidence
            mean_confs = kp_data[:, :, 2].mean(dim=1)
            best       = int(mean_confs.argmax().item())
            kps        = kp_data[best].cpu().numpy()   # (n_kp, 3)
            for i, (x, y, c) in enumerate(kps):
                if float(c) >= CONF_KP and i in KEYPOINT_WORLD:
                    detected.append((i, float(x), float(y), float(c)))
            return detected

        # Fallback: plain detection format (bbox centres as keypoints) ---
        if results.boxes is None or len(results.boxes) == 0:
            return []
        for i, box in enumerate(results.boxes):
            if i not in KEYPOINT_WORLD:
                continue
            conf = float(box.conf[0])
            if conf < CONF_KP:
                continue
            x1, y1, x2, y2 = map(float, box.xyxy[0])
            detected.append((i, (x1 + x2) / 2, (y1 + y2) / 2, conf))
        return detected

    # HOMOGRAPHY

    def update_homography(self, keypoints):
        """
        Recompute H from at least 4 detected keypoints using RANSAC.
        Returns True if homography was successfully updated.
        """
        if len(keypoints) < 4:
            return False

        img_pts   = np.array([[x, y]                   for _, x, y, _ in keypoints],
                             dtype=np.float32)
        world_pts = np.array([list(KEYPOINT_WORLD[i])  for i, _, _, _ in keypoints],
                             dtype=np.float32)

        H, mask = cv2.findHomography(img_pts, world_pts, cv2.RANSAC, 5.0)
        if H is None:
            return False
        if mask is not None and int(mask.sum()) < 4:
            return False

        self.H       = H
        self.H_valid = True
        return True

    # COORDINATE TRANSFORMS

    def pixel_to_world(self, px, py):
        """
        Map image pixel (px, py) → world (wx, wy) in metres.
        Returns None if homography is not yet available.
        """
        if not self.H_valid:
            return None
        pt = np.array([[[float(px), float(py)]]], dtype=np.float32)
        w  = cv2.perspectiveTransform(pt, self.H)[0][0]
        return float(w[0]), float(w[1])

    def world_to_canvas(self, wx, wy):
        """
        Map world metres → tactical canvas pixel.
        Origin (0,0) = batting end left; y increases toward bowling end.
        On canvas batting end is at the bottom (high canvas_y).
        """
        cx = int(CANVAS_CX + (wx - PITCH_CX_M) * PX_PER_M)
        cy = int(CANVAS_CY - (wy - PITCH_CY_M) * PX_PER_M)   # flip y
        return cx, cy

    # PER-FRAME ENTRY POINT

    def process_frame(self, frame, frame_num, player_tracks, ball_pos_px, team_colors):
        """
        Called once per video frame.

        Parameters
        ----------
        frame         : np.ndarray (BGR) - original frame; keypoints are drawn on it.
        frame_num     : int
        player_tracks : list of (track_id, x1, y1, x2, y2, team_idx)
        ball_pos_px   : (cx, cy) tuple or None
        team_colors   : list[str]  — from ask_team_colors()

        Returns
        -------
        tactical_panel : np.ndarray (BGR, CANVAS_W × CANVAS_H)
        """
        # Keypoints → update homography every 30 frames 
        kps = self.detect_keypoints(frame)
        if kps and (frame_num - self.last_H_frame) >= 30:
            if self.update_homography(kps):
                self.last_H_frame = frame_num

        # Visualise keypoints on original frame
        _draw_keypoints_on_frame(frame, kps)

        # Player world positions 
        for track_id, x1, y1, x2, y2, team_idx in player_tracks:
            foot_x = (x1 + x2) // 2
            world  = self.pixel_to_world(foot_x, y2)
            if world:
                self.player_positions[track_id].append((*world, frame_num))
                if track_id not in self.player_colors:
                    self.player_colors[track_id] = _team_bgr(team_idx, team_colors)

        # Ball world position + speed 
        ball_world = None
        if ball_pos_px is not None:
            ball_world = self.pixel_to_world(*ball_pos_px)
            if ball_world:
                self.ball_world_track.append((frame_num, *ball_world))
                self._cur_delivery_ball.append((*ball_world, frame_num))
                if self._prev_ball_world is not None:
                    dist = np.hypot(ball_world[0] - self._prev_ball_world[0],
                                    ball_world[1] - self._prev_ball_world[1])
                    speed = (dist * self.fps) * 3.6   # km/h
                    if 10 < speed < 250:
                        self.ball_speed_kmh.append((frame_num, speed))
                self._prev_ball_world = ball_world

        #  Build and return tactical canvas 
        return self._draw_tactical(player_tracks, ball_world, team_colors, frame_num)

    # DELIVERY MANAGEMENT

    def new_delivery(self):
        """
        Signal that the ball tracker has reset (new delivery).
        Archives the current delivery data and clears state.
        """
        if self._cur_delivery_ball:
            bounce = _find_bounce(self._cur_delivery_ball)
            self.deliveries.append({
                "ball_track": list(self._cur_delivery_ball),
                "bounce":     bounce,
            })
        self._cur_delivery_ball = []
        self._prev_ball_world   = None

    # TACTICAL CANVAS RENDERING

    def _draw_tactical(self, player_tracks, ball_world, team_colors, frame_num):
        canvas = _make_field_canvas(self)

        if not self.H_valid:
            cv2.putText(canvas, "Awaiting homography ...",
                        (10, CANVAS_H // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            return canvas

        # Players
        for track_id, x1, y1, x2, y2, team_idx in player_tracks:
            foot_x = (x1 + x2) // 2
            world  = self.pixel_to_world(foot_x, y2)
            if world is None:
                continue
            cx, cy = self.world_to_canvas(*world)
            color  = _team_bgr(team_idx, team_colors)
            cv2.circle(canvas, (cx, cy), _PLAYER_RADIUS, color, -1)
            cv2.circle(canvas, (cx, cy), _PLAYER_RADIUS, _COL_CREASE, 1)
            cv2.putText(canvas, str(track_id),
                        (cx + _PLAYER_RADIUS + 2, cy + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, _COL_CREASE, 1)

        # Ball
        if ball_world:
            bx, by = self.world_to_canvas(*ball_world)
            if 0 <= bx < CANVAS_W and 0 <= by < CANVAS_H:
                cv2.circle(canvas, (bx, by), 5, _COL_BALL, -1)
                cv2.circle(canvas, (bx, by), 6, _COL_BALL_EDGE, 1)

        # Speed readout
        if self.ball_speed_kmh:
            spd = self.ball_speed_kmh[-1][1]
            cv2.putText(canvas, f"{spd:.1f} km/h",
                        (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (255, 255, 255), 2)

        # Frame counter
        cv2.putText(canvas, f"#{frame_num}",
                    (8, CANVAS_H - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)

        return canvas

    # EXPORT

    def export_all(self, output_dir=OUTPUT_DIR):
        os.makedirs(output_dir, exist_ok=True)
        self._export_player_stats(output_dir)
        self._export_ball_stats(output_dir)
        self._export_heatmaps(output_dir)
        self._export_pitch_map(output_dir)
        print(f"\n[Analytics] All exports saved → {output_dir}")

    def _export_player_stats(self, output_dir):
        stats = []
        for tid, positions in self.player_positions.items():
            if len(positions) < 2:
                continue
            total_dist, speeds = 0.0, []
            for i in range(1, len(positions)):
                wx1, wy1, f1 = positions[i - 1]
                wx2, wy2, f2 = positions[i]
                dt = (f2 - f1) / self.fps
                if dt <= 0:
                    continue
                d   = np.hypot(wx2 - wx1, wy2 - wy1)
                spd = d / dt * 3.6   # km/h
                total_dist += d
                if 0 < spd < 50:
                    speeds.append(spd)

            stats.append({
                "player_id":     tid,
                "frames_seen":   len(positions),
                "distance_m":    round(total_dist, 2),
                "avg_speed_kmh": round(float(np.mean(speeds)), 2) if speeds else 0.0,
                "max_speed_kmh": round(float(np.max(speeds)),  2) if speeds else 0.0,
            })

        if not stats:
            return

        json_path = os.path.join(output_dir, "player_stats.json")
        with open(json_path, "w") as f:
            json.dump(stats, f, indent=2)

        csv_path = os.path.join(output_dir, "player_stats.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(stats[0].keys()))
            w.writeheader()
            w.writerows(stats)

        print(f"  [Analytics] Player stats → {json_path}")

    def _export_ball_stats(self, output_dir):
        if self.ball_world_track:
            csv_path = os.path.join(output_dir, "ball_positions.csv")
            with open(csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["frame", "world_x_m", "world_y_m"])
                w.writerows(self.ball_world_track)
            print(f"  [Analytics] Ball positions → {csv_path}")

        if self.ball_speed_kmh:
            speeds = [s for _, s in self.ball_speed_kmh]
            report = {
                "avg_speed_kmh":  round(float(np.mean(speeds)), 1),
                "max_speed_kmh":  round(float(np.max(speeds)),  1),
                "num_deliveries": len(self.deliveries),
            }
            rpt_path = os.path.join(output_dir, "ball_speed_report.json")
            with open(rpt_path, "w") as f:
                json.dump(report, f, indent=2)
            print(f"  [Analytics] Speed report  → {rpt_path}")

    def _export_heatmaps(self, output_dir):
        generated = 0
        for tid, positions in self.player_positions.items():
            if len(positions) < 10:
                continue
            xs = [p[0] for p in positions]
            ys = [p[1] for p in positions]

            fig, ax = plt.subplots(figsize=(4, 8))
            ax.set_facecolor("#228B22")
            ax.set_xlim(-5, PITCH_WIDTH + 5)
            ax.set_ylim(-3, PITCH_LENGTH + 3)
            ax.set_aspect("equal")

            # Pitch outline
            pitch_rect = mpatches.Rectangle(
                (0, 0), PITCH_WIDTH, PITCH_LENGTH,
                linewidth=1, edgecolor="white", facecolor="#8BC34A", alpha=0.5)
            ax.add_patch(pitch_rect)

            # Heatmap
            hm, xe, ye = np.histogram2d(
                xs, ys, bins=50,
                range=[[-5, PITCH_WIDTH + 5], [-3, PITCH_LENGTH + 3]])
            hm = gaussian_filter(hm, sigma=2)
            ax.imshow(hm.T,
                      extent=[xe[0], xe[-1], ye[0], ye[-1]],
                      origin="lower", cmap="hot", alpha=0.55, aspect="auto")

            ax.set_title(f"Player {tid} Movement Heatmap", fontsize=11)
            ax.set_xlabel("Width (m)")
            ax.set_ylabel("Length (m)")

            out = os.path.join(output_dir, f"heatmap_player_{tid}.png")
            fig.savefig(out, dpi=100, bbox_inches="tight")
            plt.close(fig)
            generated += 1

        if generated:
            print(f"  [Analytics] {generated} player heatmap(s) saved.")

    def _export_pitch_map(self, output_dir):
        bounces = [(d["bounce"][0], d["bounce"][1])
                   for d in self.deliveries if d.get("bounce")]
        if not bounces:
            return

        fig, ax = plt.subplots(figsize=(3, 7))
        ax.set_facecolor("#228B22")
        ax.set_xlim(-0.5, PITCH_WIDTH + 0.5)
        ax.set_ylim(-1,   PITCH_LENGTH + 1)
        ax.set_aspect("equal")

        # Pitch rectangle
        ax.add_patch(mpatches.Rectangle(
            (0, 0), PITCH_WIDTH, PITCH_LENGTH,
            linewidth=1, edgecolor="white", facecolor="#8BC34A"))

        # Crease lines
        for crease_y in [POPPING_CREASE, PITCH_LENGTH - POPPING_CREASE]:
            ax.hlines(crease_y, 0, PITCH_WIDTH, colors="white", linewidths=0.8)

        # Zone bands (subtle fill)
        for y0, y1, name, hex_color in PITCH_ZONES:
            ax.axhspan(y0, y1, alpha=0.15, color=hex_color)

        # Bounce points (coloured by zone)
        for bx, by in bounces:
            zone_color = "#FFFFFF"
            for y0, y1, _, hc in PITCH_ZONES:
                if y0 <= by < y1:
                    zone_color = hc
                    break
            ax.scatter(bx, by, c=zone_color, s=50, alpha=0.85, zorder=5,
                       edgecolors="white", linewidths=0.5)

        # Legend
        handles = [mpatches.Patch(color=hc, label=nm)
                   for _, _, nm, hc in PITCH_ZONES]
        ax.legend(handles=handles, loc="lower right", fontsize=6, framealpha=0.6)

        ax.set_title(f"Ball Pitch Map  ({len(bounces)} deliveries)", fontsize=10)
        ax.set_xlabel("Width (m)")
        ax.set_ylabel("Distance from batting end (m)")

        out = os.path.join(output_dir, "pitch_map.png")
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  [Analytics] Pitch map     → {out}")


# MODULE-LEVEL HELPERS

def _draw_keypoints_on_frame(frame, keypoints):
    """Annotate detected pitch keypoints on the original frame."""
    for kp_idx, x, y, conf in keypoints:
        ix, iy = int(x), int(y)
        cv2.circle(frame, (ix, iy), 5, (0, 255, 255), -1)
        cv2.putText(frame, f"KP{kp_idx}",
                    (ix + 6, iy - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1)


def _make_field_canvas(analytics):
    """
    Render an empty top-down cricket field canvas.
    Batting end at bottom, bowling end at top.
    """
    canvas = np.full((CANVAS_H, CANVAS_W, 3), _COL_FIELD_DARK, dtype=np.uint8)

    # Outer boundary circle
    cv2.circle(canvas, (CANVAS_CX, CANVAS_CY),
               int(FIELD_RADIUS_M * PX_PER_M), _COL_FIELD_LIGHT, 2)

    # 30-yard (27.4 m) inner circle
    cv2.circle(canvas, (CANVAS_CX, CANVAS_CY),
               int(27.4 * PX_PER_M), (80, 160, 90), 1)

    # Pitch rectangle
    tl = analytics.world_to_canvas(0.0,         PITCH_LENGTH)
    br = analytics.world_to_canvas(PITCH_WIDTH,  0.0         )
    cv2.rectangle(canvas, tl, br, _COL_PITCH, -1)
    cv2.rectangle(canvas, tl, br, (200, 220, 150), 1)

    # Crease lines
    for crease_y in [POPPING_CREASE,
                     PITCH_LENGTH - POPPING_CREASE,
                     BOWLING_CREASE,
                     PITCH_LENGTH - BOWLING_CREASE]:
        p1 = analytics.world_to_canvas(-0.3,              crease_y)
        p2 = analytics.world_to_canvas(PITCH_WIDTH + 0.3, crease_y)
        cv2.line(canvas, p1, p2, _COL_CREASE, 1)

    # Stumps (small dots at each end)
    for stump_y in [0.0, PITCH_LENGTH]:
        for stump_x in [PITCH_WIDTH / 2 - 0.11,
                        PITCH_WIDTH / 2,
                        PITCH_WIDTH / 2 + 0.11]:
            sc = analytics.world_to_canvas(stump_x, stump_y)
            cv2.circle(canvas, sc, 2, _COL_STUMP, -1)

    # Label batting / bowling ends
    bat_y  = analytics.world_to_canvas(PITCH_WIDTH / 2, -0.8)[1]
    bowl_y = analytics.world_to_canvas(PITCH_WIDTH / 2, PITCH_LENGTH + 0.8)[1]
    cv2.putText(canvas, "BAT", (CANVAS_CX - 12, bat_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, _COL_CREASE, 1)
    cv2.putText(canvas, "BOWL", (CANVAS_CX - 16, bowl_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, _COL_CREASE, 1)

    return canvas


def _find_bounce(ball_track):
    """
    - Identify the first bounce point from a delivery ball track.
    - World convention: y increases from batting end (0) to bowling end (20.12 m).
    - The ball is bowled from high y → travels to low y → bounces → rises.
    - Bounce = the frame where y is at its local minimum within pitch bounds.
    """
    if len(ball_track) < 3:
        return None
    in_pitch = [(i, pt[1]) for i, pt in enumerate(ball_track)
                if 0.5 < pt[1] < PITCH_LENGTH - 0.5
                and 0.0 < pt[0] < PITCH_WIDTH]
    if not in_pitch:
        return None
    min_idx = min(in_pitch, key=lambda t: t[1])[0]
    return (ball_track[min_idx][0], ball_track[min_idx][1])


def _team_bgr(team_idx, team_colors):
    """Return BGR bounding-box colour for a team index."""
    try:
        from team_player_assignment import COLOR_MAP_BGR, FALLBACK_COLORS
    except ImportError:
        return (0, 255, 0)
    if team_idx < 0 or team_idx >= len(team_colors):
        return (0, 255, 0)
    name = team_colors[team_idx].lower()
    return COLOR_MAP_BGR.get(name, FALLBACK_COLORS[team_idx % len(FALLBACK_COLORS)])


def build_split_frame(original_frame, tactical_canvas, panel_w=SPLIT_PANEL_W):
    """
    Stack the original broadcast frame and the tactical bird's-eye view
    side by side (left = original, right = tactical).

    Parameters
    ----------
    original_frame  : np.ndarray  (H, W, 3)
    tactical_canvas : np.ndarray  (CANVAS_H, CANVAS_W, 3)
    panel_w         : int  — width of the tactical panel in the split frame

    Returns
    -------
    np.ndarray  (H, W + panel_w, 3)
    """
    h = original_frame.shape[0]
    panel = cv2.resize(tactical_canvas, (panel_w, h))

    # Thin divider line
    divider = np.zeros((h, 2, 3), dtype=np.uint8)
    divider[:] = (80, 80, 80)

    return np.hstack([original_frame, divider, panel])
