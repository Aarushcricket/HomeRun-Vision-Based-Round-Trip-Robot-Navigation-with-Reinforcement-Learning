# Vision-Based Round-Trip Robot Navigation with Reinforcement Learning

A simulated two-wheeled robot uses its **front camera** to drive from a **Base (green pillar)**
to a **Target (red pillar)** and back again, avoiding **8 grey pillar obstacles**.
Two RL algorithms are trained and compared: **Q-Learning** (tabular, value-based) and
**PPO** (neural network, policy-based).

## How to run (in order, one terminal)

```
python check_setup.py                 # 1. opens the 3D world and the robot camera view
python train.py --algo qlearning      # 2. ONLINE Q-Learning (300,000 steps)
python train.py --algo ppo            # 3. PPO (300,000 steps)
python collect_dataset.py             # 4. trained PPO robot records the dataset (200,000 steps)
python offline_qlearning.py           # 5. OFFLINE Q-Learning, learns only from the dataset
python evaluate.py                    # 6. round trips for all three agents, results table
python plots.py                       # 7. graphs in results/
python evaluate.py --gui --trials 1   # 8. demo: watch each robot + its camera
```

The dataset is created in the `dataset/` folder: `robot_dataset.npz` (used for training),
`robot_dataset.csv` (open in Excel) and `dataset_info.json` (summary).

## Files

| File | What it does |
|---|---|
| `vision.py` | OpenCV vision model: colour blob detection, bearing angle, ground-plane distance, goal tracking, depth-based obstacle sensing |
| `robot_env.py` | PyBullet world + Gymnasium environment: robot, arena, 8 obstacles, rewards, mission controller (GO → RETURN) |
| `qlearning.py` | Tabular Q-Learning written from scratch (TD update rule) |
| `train.py` | Trains online Q-Learning and PPO with the same step budget |
| `collect_dataset.py` | Trained PPO robot drives (20% random actions) and every step is saved as the offline dataset |
| `offline_qlearning.py` | Offline Q-Learning: learns only from the dataset, with a support constraint against extrapolation error |
| `evaluate.py` | Runs all three agents' round trips from Base on the same layouts, prints the table |
| `plots.py` | Learning curves, robot paths, success-rate chart |

## The vision model (vision.py)

Each camera frame becomes six numbers:
`[goal_offset, goal_closeness, goal_visible, obst_left, obst_center, obst_right]`

1. **Colour segmentation:** the frame is converted to HSV and thresholded for red (GO leg)
   or green (RETURN leg).
2. **Blob detection:** connected components find the largest marker-coloured blob.
3. **Bearing:** the blob's column is converted into a real angle using the camera's field of view.
4. **Ground-plane distance:** the camera height is known (0.2 m), so the row where the marker meets
   the floor gives its distance: `distance = camera_height / tan(angle below horizon)`.
5. **Goal tracking:** when the marker leaves the view, the robot remembers which side it was last
   seen on, so it knows which way to turn to find it again.
6. **Obstacles:** the depth image's upper half (above the floor) is split into left / centre / right,
   and the nearest distance in each becomes an obstacle-proximity value.

The state given to the RL agent adds the leg (GO / RETURN) and the last turn (one step of memory).

## How the robot knows where to go

The robot is never given a map or a path. The camera tells it where the goal is right now; RL
training (rewards: + for getting closer, +10 for reaching the goal, −2 for a collision, −0.02 per
step) teaches it what to do with that information. True positions are used only by the simulator
to compute rewards: "the simulator is the teacher, the camera is the robot's only sense".

## Design choices worth mentioning in the report

* **Fair comparison:** same environment, rewards, training steps and test layouts for both algorithms.
* **Harder than the first version:** 8 obstacles instead of 6, the Target is at least 3.5 m from Base
  (was 3.0 m), and one pillar always stands on the line between the start and the Target, so the
  robot cannot see its goal at first and must drive around the pillar to find it. Pillars stay at
  least 0.8 m apart, leaving 0.4 m gaps for the 0.3 m robot, so a route always exists.
* **Practising the hard start:** 40% of training episodes begin at the real start spots (beside Base
  for GO, beside the Target for RETURN), so the robot learns to search for a hidden goal.
* **Difficulty settings** (top of `robot_env.py`): `MIN_TARGET_DIST`, `BLOCKER_RANGE`,
  `N_OBSTACLES`, `MAX_STEPS_PER_LEG`, `MISSION_START_PROB`.
* **Training randomisation:** every training episode has a new layout, random leg and random start pose.
* **Evaluation with ε = 0.05:** a small chance of a random action for both algorithms, as in the
  original DQN paper, to avoid endless loops.

## Offline RL in this project

* **Dataset:** each row is one experience: 9 state values, the action, the reward, the 9 next-state
  values, and whether the episode ended (goal reached) or timed out.
* **Offline Q-Learning** uses the same TD update rule as online Q-Learning, but sweeps over the stored
  experiences for several epochs instead of driving the robot.
* **Support constraint:** an action is only trusted if it appears at least 3 times in that state in the
  dataset. This prevents over-optimism about actions the data never tried (extrapolation error), the
  same idea as BCQ / CQL. `python offline_qlearning.py --plain` turns it off for comparison.
* The simulator is used only to **test** the offline agent, never to train it.