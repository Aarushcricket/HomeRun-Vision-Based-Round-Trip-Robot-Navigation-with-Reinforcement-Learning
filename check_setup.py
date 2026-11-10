"""
check_setup.py
--------------
Run this FIRST. It opens the 3D world, spins the robot in a full circle, and shows
what the camera sees with the vision features printed on top. If you see the red
Target pillar get a box drawn around it, the CV pipeline is working.

  python check_setup.py
"""
import time

import cv2

from robot_env import RoundTripEnv
from vision import draw_debug

env = RoundTripEnv(gui=True)
obs, _ = env.reset(options={"layout_seed": 1000})
print("State = [goal_offset, goal_closeness, goal_visible, obst_L, obst_C, obst_R, phase, last_left, last_right]")
for step in range(18):                       # 18 x 20 degrees = one full turn
    obs, reward, *_ = env.step(1)            # action 1 = turn left
    dist = env.last_debug["distance_m"]
    print(f"step {step + 1:2d}: {[round(float(v), 2) for v in obs]}"
          + (f"   estimated distance to goal = {dist:.2f} m" if dist is not None else ""))
    cv2.imshow("Robot camera", draw_debug(env.last_rgb, env.last_debug, title="Setup check"))
    cv2.waitKey(1)
    time.sleep(0.3)

print("\nSetup works! Press any key in the camera window to close.")
cv2.waitKey(0)
cv2.destroyAllWindows()
env.close()
