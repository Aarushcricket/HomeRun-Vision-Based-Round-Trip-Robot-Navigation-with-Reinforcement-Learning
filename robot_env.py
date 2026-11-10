"""
robot_env.py
------------
Custom Gymnasium environment: a small two-wheeled (differential-drive) robot in a
6 m x 6 m PyBullet arena with 8 pillar obstacles. One pillar always stands between
the start and the Target, so the robot cannot see its goal at first.

  * GREEN pillar = Base (home)      * RED pillar = Target
  * GO leg     : drive from Base to the red Target
  * RETURN leg : drive back to the green Base

The robot ONLY perceives the world through its front camera (vision.py turns each
frame into numbers). The simulator uses true positions just to give rewards and to
decide when a goal is reached: "the simulator is the teacher, the camera is the
robot's only sense".

Observation (9 numbers):
  [goal_offset, goal_closeness, goal_visible, obst_left, obst_center, obst_right, phase,
   last_turn_left, last_turn_right]
  The last two give the robot one step of memory (which way it just turned), which
  stops it from wobbling left-right-left-right in front of an obstacle.
Actions (3): 0 = forward, 1 = turn left, 2 = turn right

Two ways to reset:
  env.reset()                                   -> TRAINING: random layout, random leg,
                                                   random start pose, episode = one leg
  env.reset(options={"layout_seed": 7})         -> MISSION (evaluation): fixed layout,
                                                   robot at Base, GO then automatically
                                                   switches to RETURN, ends back at Base
"""
import math

import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces
from pybullet_utils import bullet_client

from vision import CAM_FOV, CAM_H, CAM_HEIGHT, CAM_W, FAR, NEAR, GoalTracker, process_frame

# ---------------- World settings ----------------
ARENA_HALF = 3.0                                  # walls at +-3 m
WALL_HALF_THICKNESS = 0.05
BASE_POS = (-2.4, -2.4)                           # green marker (home)
START_POSE = (-1.9, -1.9, math.radians(45))       # robot start beside Base, facing the arena
ROBOT_RADIUS = 0.15
OBSTACLE_RADIUS = 0.2
N_OBSTACLES = 8                                   # was 6 in the first version
MIN_TARGET_DIST = 3.5                            # Target is at least this far from Base (was 3.0)
BLOCKER_RANGE = (0.3, 0.4)                        # blocking pillar sits 40-60 % of the way to the Target
GOAL_RADIUS = 0.45                                # "arrived" when this close to a marker
STEP_SIZE = 0.12                                  # metres per forward action
TURN_ANGLE = math.radians(20)                     # radians per turn action
MAX_STEPS_PER_LEG = 1000
MISSION_START_PROB = 0.4                          # share of training episodes that begin at the real start spots

# ---------------- Rewards ----------------
R_PROGRESS = 3.0     # x (metres moved closer to the goal)
R_GOAL = 10.0        # reaching the goal
R_COLLISION = -2.0   # bumping into a wall or obstacle
R_STEP = -0.02       # small cost per step so the robot doesn't wander

GO, RETURN = 0, 1
ACTION_NAMES = ["forward", "left", "right"]


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


class RoundTripEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, gui=False, seed=None):
        super().__init__()
        self.gui = gui
        self.client = bullet_client.BulletClient(connection_mode=p.GUI if gui else p.DIRECT)
        self.client.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.rng = np.random.default_rng(seed)

        self.action_space = spaces.Discrete(3)
        low = np.array([-1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=np.ones(9, dtype=np.float32), dtype=np.float32)

        self.pose = list(START_POSE)
        self.target_pos = (2.0, 2.0)
        self.obstacle_pos = []
        self.phase = GO
        self.last_action = 0
        self.mission_mode = False
        self.path = []
        self.tracker = GoalTracker()
        self.last_rgb = None
        self.last_debug = None
        self._build_world()

    # ------------------------------------------------------------------ world
    def _visual_body(self, shape, color, pos, **size):
        c = self.client
        vis = c.createVisualShape(shape, rgbaColor=color, **size)
        return c.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1,
                                 baseVisualShapeIndex=vis, basePosition=pos)

    def _build_world(self):
        c = self.client
        if self.gui:
            c.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            c.resetDebugVisualizerCamera(cameraDistance=6.5, cameraYaw=0, cameraPitch=-65,
                                         cameraTargetPosition=[0, 0, 0])
        c.loadURDF("plane.urdf")

        wall_color = [0.75, 0.75, 0.75, 1]
        h = 0.25
        for sign in (-1, 1):
            self._visual_body(p.GEOM_BOX, wall_color, [0, sign * ARENA_HALF, h],
                              halfExtents=[ARENA_HALF, WALL_HALF_THICKNESS, h])
            self._visual_body(p.GEOM_BOX, wall_color, [sign * ARENA_HALF, 0, h],
                              halfExtents=[WALL_HALF_THICKNESS, ARENA_HALF, h])

        # Grey pillars = obstacles (grey has no colour saturation, so it never looks like a marker)
        self.obstacle_ids = [
            self._visual_body(p.GEOM_CYLINDER, [0.45, 0.45, 0.5, 1], [0, 0, -5],
                              radius=OBSTACLE_RADIUS, length=0.5)
            for _ in range(N_OBSTACLES)
        ]
        self.base_marker = self._visual_body(p.GEOM_CYLINDER, [0, 0.8, 0, 1],
                                             [BASE_POS[0], BASE_POS[1], 0.3], radius=0.12, length=0.6)
        self.target_marker = self._visual_body(p.GEOM_CYLINDER, [1, 0, 0, 1], [0, 0, 0.3],
                                               radius=0.12, length=0.6)
        self.robot_id = self._visual_body(p.GEOM_CYLINDER, [0.1, 0.3, 0.9, 1], [0, 0, 0.06],
                                          radius=ROBOT_RADIUS, length=0.12)
        self.nose_id = self._visual_body(p.GEOM_BOX, [1, 1, 0, 1], [0, 0, 0.14],
                                         halfExtents=[0.05, 0.03, 0.02])

    def _sample_layout(self, rng):
        """Random Target position and obstacle positions that leave room to move.
        Pillars are at least 0.8 m apart (centre to centre), so the gap between any
        two pillars is 0.4 m, wider than the 0.3 m robot: a route always exists.

        One pillar (the "blocker") is placed on the straight line between the start
        and the Target, so the robot cannot see the Target at the beginning of the
        trip (or the Base at the start of the return trip). It has to drive around
        the blocker and search for the marker."""
        while True:
            target = tuple(rng.uniform(-2.4, 2.4, size=2))
            if _dist(target, BASE_POS) >= MIN_TARGET_DIST:
                break
        keep_clear = [BASE_POS, target, START_POSE[:2]]

        # Blocking pillar on the start -> Target line (small sideways jitter keeps it natural)
        sx, sy = START_POSE[:2]
        f = rng.uniform(*BLOCKER_RANGE)
        dx, dy = target[0] - sx, target[1] - sy
        length = math.hypot(dx, dy)
        side = rng.uniform(-0.05, 0.05)
        blocker = (sx + f * dx - side * dy / length, sy + f * dy + side * dx / length)
        obstacles = [blocker]
        attempts = 0
        while len(obstacles) < N_OBSTACLES and attempts < 5000:
            attempts += 1
            pos = tuple(rng.uniform(-2.3, 2.3, size=2))
            if all(_dist(pos, k) >= 0.9 for k in keep_clear) and all(_dist(pos, o) >= 0.8 for o in obstacles):
                obstacles.append(pos)
        return target, obstacles

    def _apply_layout(self):
        c = self.client
        c.resetBasePositionAndOrientation(self.target_marker, [*self.target_pos, 0.3], [0, 0, 0, 1])
        for i, oid in enumerate(self.obstacle_ids):
            pos = [*self.obstacle_pos[i], 0.25] if i < len(self.obstacle_pos) else [0, 0, -5]
            c.resetBasePositionAndOrientation(oid, pos, [0, 0, 0, 1])

    def _place_robot(self):
        x, y, yaw = self.pose
        q = self.client.getQuaternionFromEuler([0, 0, yaw])
        self.client.resetBasePositionAndOrientation(self.robot_id, [x, y, 0.06], q)
        self.client.resetBasePositionAndOrientation(
            self.nose_id, [x + 0.09 * math.cos(yaw), y + 0.09 * math.sin(yaw), 0.14], q)

    def _is_free(self, x, y):
        limit = ARENA_HALF - WALL_HALF_THICKNESS - ROBOT_RADIUS
        if abs(x) > limit or abs(y) > limit:
            return False
        return all(_dist((x, y), o) >= ROBOT_RADIUS + OBSTACLE_RADIUS for o in self.obstacle_pos)

    def goal_pos(self):
        return self.target_pos if self.phase == GO else BASE_POS

    @property
    def layout(self):
        return {"base": list(BASE_POS), "target": [float(v) for v in self.target_pos],
                "obstacles": [[float(v) for v in o] for o in self.obstacle_pos],
                "obstacle_radius": OBSTACLE_RADIUS, "arena_half": ARENA_HALF}

    # ---------------------------------------------------------------- camera
    def _observe(self):
        x, y, yaw = self.pose
        c = self.client
        eye = [x + 0.16 * math.cos(yaw), y + 0.16 * math.sin(yaw), CAM_HEIGHT]
        look = [eye[0] + math.cos(yaw), eye[1] + math.sin(yaw), CAM_HEIGHT]
        view = c.computeViewMatrix(eye, look, [0, 0, 1])
        proj = c.computeProjectionMatrixFOV(fov=CAM_FOV, aspect=CAM_W / CAM_H, nearVal=NEAR, farVal=FAR)
        # The CPU "tiny renderer" is used in BOTH training and GUI demos so the robot
        # always sees exactly the same kind of image.
        img = c.getCameraImage(CAM_W, CAM_H, view, proj, renderer=p.ER_TINY_RENDERER,
                               lightDirection=[1, 1, 3], lightAmbientCoeff=0.6,
                               lightDiffuseCoeff=0.4, lightSpecularCoeff=0.0, shadow=0)
        rgb = np.reshape(np.asarray(img[2], dtype=np.uint8), (CAM_H, CAM_W, 4))[:, :, :3]
        depth = np.reshape(np.asarray(img[3], dtype=np.float32), (CAM_H, CAM_W))
        features, debug = process_frame(rgb, depth, self.phase, self.tracker)
        self.last_rgb, self.last_debug = rgb, debug
        memory = [float(self.last_action == 1), float(self.last_action == 2)]
        return np.array(features + [float(self.phase)] + memory, dtype=np.float32)

    # ------------------------------------------------------------- gym API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        options = options or {}

        if "layout_seed" in options:
            # MISSION: fixed layout, start at Base, GO then RETURN
            self.mission_mode = True
            self.target_pos, self.obstacle_pos = self._sample_layout(np.random.default_rng(options["layout_seed"]))
            self.phase = GO
            self.pose = list(START_POSE)
        else:
            # TRAINING: random layout, random leg, random start pose
            self.mission_mode = False
            self.target_pos, self.obstacle_pos = self._sample_layout(self.rng)
            self.phase = int(self.rng.integers(2))
            if self.rng.random() < MISSION_START_PROB:
                # Start where a real mission leg starts (goal hidden behind the blocker),
                # so the robot practises searching for a goal it cannot see yet.
                if self.phase == GO:
                    x, y = START_POSE[:2]
                else:
                    a = self.rng.uniform(-math.pi, math.pi)
                    x, y = self.target_pos[0] + 0.35 * math.cos(a), self.target_pos[1] + 0.35 * math.sin(a)
                    if not self._is_free(x, y):
                        x, y = self.target_pos
            else:
                while True:
                    x, y = self.rng.uniform(-2.6, 2.6, size=2)
                    if self._is_free(x, y) and _dist((x, y), self.goal_pos()) >= 1.0:
                        break
            self.pose = [x, y, float(self.rng.uniform(-math.pi, math.pi))]

        self._apply_layout()
        self._place_robot()
        self.tracker.reset()
        self.leg_steps = 0
        self.last_action = 0
        self.prev_dist = _dist(self.pose[:2], self.goal_pos())
        self.path = [tuple(self.pose[:2])]
        return self._observe(), {}

    def step(self, action):
        action = int(action)
        x, y, yaw = self.pose
        collision, moved = False, 0.0

        if action == 0:
            nx, ny = x + STEP_SIZE * math.cos(yaw), y + STEP_SIZE * math.sin(yaw)
            if self._is_free(nx, ny):
                self.pose[0], self.pose[1], moved = nx, ny, STEP_SIZE
            else:
                collision = True                  # blocked: robot stays where it is
        elif action == 1:
            self.pose[2] = (yaw + TURN_ANGLE + math.pi) % (2 * math.pi) - math.pi
        else:
            self.pose[2] = (yaw - TURN_ANGLE + math.pi) % (2 * math.pi) - math.pi
        self._place_robot()
        self.last_action = action
        self.leg_steps += 1
        self.path.append(tuple(self.pose[:2]))

        d = _dist(self.pose[:2], self.goal_pos())
        reward = R_STEP + R_PROGRESS * (self.prev_dist - d) + (R_COLLISION if collision else 0.0)
        self.prev_dist = d

        event, terminated, truncated = None, False, False
        if d < GOAL_RADIUS:
            reward += R_GOAL
            event = "target_reached" if self.phase == GO else "base_reached"
            if self.mission_mode and self.phase == GO:
                # Mission controller: target reached -> switch to the RETURN leg
                self.phase = RETURN
                self.tracker.reset()              # new goal: forget where the old one was
                self.leg_steps = 0
                self.prev_dist = _dist(self.pose[:2], self.goal_pos())
            else:
                terminated = True
        if not terminated and self.leg_steps >= MAX_STEPS_PER_LEG:
            truncated = True

        info = {"event": event, "collision": collision, "moved": moved, "phase": self.phase}
        return self._observe(), float(reward), terminated, truncated, info

    def close(self):
        try:
            self.client.disconnect()
        except Exception:
            pass
