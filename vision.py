"""
vision.py  (improved vision model, version 2)
---------------------------------------------
Turns one camera frame into the numbers the RL agent sees:
  [goal_offset, goal_closeness, goal_visible, obst_left, obst_center, obst_right]

The goal marker is RED during the GO leg and GREEN during the RETURN leg.

Improvements over the first version:
  1. Blob detection with connected components: the marker is the largest connected
     group of goal-coloured pixels, measured with its exact pixel statistics.
  2. True bearing angle: the marker's pixel column is converted into a real angle
     using the camera's field of view (perspective-correct, not just pixel position).
  3. Ground-plane distance estimate: the camera height is known (0.2 m), so the row
     where the marker touches the floor tells us how far away it is:
         distance = camera_height / tan(angle below the horizon)
     This gives distance in metres and is not fooled when the marker is partly
     cut off by the image edge (the old "how big does it look" size feature was).
  4. Goal tracking memory: when the marker leaves the view, goal_offset keeps the
     side where it was last seen (-1 = left, +1 = right), so the robot knows which
     way to turn to find it again instead of searching blindly.
"""
import math

import cv2
import numpy as np

# Camera settings (shared with robot_env.py)
CAM_W, CAM_H = 80, 60          # small image = fast training
CAM_FOV = 70                   # vertical field of view in degrees
NEAR, FAR = 0.05, 8.0          # camera clipping planes in metres
CAM_HEIGHT = 0.2               # camera height above the floor in metres

TAN_V = math.tan(math.radians(CAM_FOV / 2))
TAN_H = TAN_V * CAM_W / CAM_H
HALF_HFOV = math.atan(TAN_H)   # half of the horizontal field of view (radians)

SENSE_RANGE = 1.2              # obstacles further than this are ignored (metres)
MIN_GOAL_PIXELS = 3            # fewer pixels than this = marker not visible
MAX_GOAL_DIST = 6.0            # distances are scaled into 0..1 using this range

# HSV colour ranges (OpenCV uses H: 0-179, S: 0-255, V: 0-255).
# Red wraps around the hue circle, so it needs two ranges.
RED_RANGES = [((0, 120, 40), (10, 255, 255)), ((170, 120, 40), (179, 255, 255))]
GREEN_RANGES = [((40, 120, 40), (85, 255, 255))]


def color_mask(hsv, ranges):
    """Binary mask (0/255) of pixels whose HSV value falls inside any of the ranges."""
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in ranges:
        mask |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
    return mask


def find_marker(mask):
    """Largest connected blob in the mask -> dict(cx, pixels, bbox, bottom) or None."""
    n, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    pixels = int(stats[i, cv2.CC_STAT_AREA])
    if pixels < MIN_GOAL_PIXELS:
        return None
    x, y, w, h = (int(v) for v in stats[i, :4])
    return {"cx": float(centroids[i][0]), "pixels": pixels, "bbox": (x, y, w, h), "bottom": y + h}


def bearing(cx, width):
    """Pixel column -> real angle to the marker (radians, positive = to the right)."""
    return math.atan((cx - width / 2.0) / (width / 2.0) * TAN_H)


def ground_distance(bottom_row, height, angle):
    """Distance to where the marker meets the floor, from the camera height."""
    tan_below = (bottom_row - height / 2.0) / (height / 2.0) * TAN_V
    if tan_below <= 1e-3:
        return MAX_GOAL_DIST
    return min(MAX_GOAL_DIST, CAM_HEIGHT / tan_below / math.cos(angle))


def linearize_depth(depth_buffer):
    """PyBullet returns a non-linear depth buffer in [0, 1]; convert it to metres."""
    return FAR * NEAR / (FAR - (FAR - NEAR) * depth_buffer)


class GoalTracker:
    """Remembers where the goal was last seen, for searching when it is out of view."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.last_offset = 0.0

    def update(self, visible, offset):
        if visible:
            self.last_offset = offset
            return offset
        return float(np.sign(self.last_offset))   # -1 lost on the left, +1 on the right, 0 unknown


def process_frame(rgb, depth_buffer, phase, tracker=None):
    """
    rgb          : (H, W, 3) uint8 image
    depth_buffer : (H, W) float depth buffer from PyBullet
    phase        : 0 = GO (look for red), 1 = RETURN (look for green)
    tracker      : GoalTracker that remembers the last-seen side (optional)
    returns      : (features list of 6 floats, debug dict for visualisation)
    """
    h, w = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    red = color_mask(hsv, RED_RANGES)
    green = color_mask(hsv, GREEN_RANGES)
    goal_mask = red if phase == 0 else green

    # ---- 1. Goal marker: colour blob, bearing and ground-plane distance ----
    marker = find_marker(goal_mask)
    dist = None
    if marker is None:
        offset, closeness, visible = 0.0, 0.0, 0.0
    else:
        angle = bearing(marker["cx"], w)
        offset = float(np.clip(angle / HALF_HFOV, -1.0, 1.0))
        dist = ground_distance(marker["bottom"], h, angle)
        closeness = float(np.clip(1.0 - dist / MAX_GOAL_DIST, 0.0, 1.0))
        visible = 1.0
    if tracker is not None:
        offset = tracker.update(visible > 0, offset)

    # ---- 2. Obstacles: depth image ----
    # The camera looks straight ahead, so the floor only appears in the lower half.
    # Anything in the upper half that is close must be a wall or an obstacle.
    depth = linearize_depth(depth_buffer[: h // 2]).copy()
    markers_upper = (red[: h // 2] > 0) | (green[: h // 2] > 0)
    depth[markers_upper] = FAR                     # markers are goals, not obstacles
    sectors = np.array_split(depth, 3, axis=1)     # left, centre, right thirds
    obstacles = [float(np.clip((SENSE_RANGE - s.min()) / SENSE_RANGE, 0.0, 1.0)) for s in sectors]

    features = [offset, closeness, visible] + obstacles
    debug = {"marker": marker, "goal_color": "RED" if phase == 0 else "GREEN",
             "distance_m": dist, "features": features}
    return features, debug


def draw_debug(rgb, debug, scale=6, title=""):
    """Enlarged BGR image with the detected marker boxed, for cv2.imshow during demos."""
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (rgb.shape[1] * scale, rgb.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
    marker = debug["marker"]
    if marker is not None:
        x, y, w, h = [v * scale for v in marker["bbox"]]
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
    off, close, vis, ol, oc, orr = debug["features"]
    dist = f"{debug['distance_m']:.2f} m" if debug["distance_m"] is not None else "-"
    lines = [title,
             f"goal {debug['goal_color']}  visible={int(vis)}  offset={off:+.2f}  distance={dist}",
             f"obstacle  L={ol:.2f}  C={oc:.2f}  R={orr:.2f}"]
    for i, text in enumerate(lines):
        cv2.putText(img, text, (8, 20 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
        cv2.putText(img, text, (8, 20 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return img
