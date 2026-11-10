"""
collect_dataset.py
------------------
Creates the OFFLINE RL DATASET. A trained robot (PPO by default) drives in the
simulator and every step is recorded as one experience:

    (state, action, reward, next_state, terminated, truncated)

To give the dataset some variety, the driver takes a random action EPSILON of the
time (default 20%), like real logged data that is good but not perfect.

  python collect_dataset.py                          # 200,000 steps, PPO driver, 20% random
  python collect_dataset.py --steps 100000           # smaller dataset
  python collect_dataset.py --behaviour qlearning    # use the online Q-Learning robot instead
  python collect_dataset.py --epsilon 0.5            # noisier, lower-quality data

Outputs (folder dataset/):
  robot_dataset.npz   the dataset used by offline_qlearning.py (compressed NumPy arrays)
  robot_dataset.csv   the same data as a spreadsheet you can open, view and submit
  dataset_info.json   summary: size, episodes, success rate, average reward, settings
"""
import argparse
import json
import os
import time

import numpy as np

from robot_env import RoundTripEnv

OBS_NAMES = ["goal_offset", "goal_closeness", "goal_visible", "obst_left", "obst_center",
             "obst_right", "phase", "last_left", "last_right"]


def load_driver(name):
    if name == "ppo":
        from stable_baselines3 import PPO
        model = PPO.load("models/ppo.zip", device="cpu")
        return lambda obs: int(model.predict(obs, deterministic=True)[0])
    if name == "qlearning":
        from qlearning import QLearningAgent
        agent = QLearningAgent.load("models/qlearning.npy")
        return agent.predict
    return None                                           # "random"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--behaviour", choices=["ppo", "qlearning", "random"], default="ppo")
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    model_file = {"ppo": "models/ppo.zip", "qlearning": "models/qlearning.npy"}.get(args.behaviour)
    if model_file and not os.path.exists(model_file):
        print(f"No trained model at {model_file} - run: python train.py --algo {args.behaviour}")
        return
    os.makedirs("dataset", exist_ok=True)
    driver = load_driver(args.behaviour)
    rng = np.random.default_rng(args.seed)
    env = RoundTripEnv(seed=args.seed)       # same random layouts and start spots as training

    n = args.steps
    obs_buf = np.zeros((n, 9), np.float32)
    next_buf = np.zeros((n, 9), np.float32)
    act_buf = np.zeros(n, np.int64)
    rew_buf = np.zeros(n, np.float32)
    term_buf = np.zeros(n, bool)
    trunc_buf = np.zeros(n, bool)
    ep_buf = np.zeros(n, np.int64)

    print(f"Recording {n} steps with the {args.behaviour.upper()} driver "
          f"({int(args.epsilon * 100)}% random actions)...")
    start = time.time()
    obs, _ = env.reset()
    episode, ep_reward, ep_rewards, successes = 0, 0.0, [], 0
    for i in range(n):
        if driver is None or rng.random() < args.epsilon:
            action = int(rng.integers(3))
        else:
            action = driver(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)

        obs_buf[i], act_buf[i], rew_buf[i], next_buf[i] = obs, action, reward, next_obs
        term_buf[i], trunc_buf[i], ep_buf[i] = terminated, truncated, episode
        ep_reward += reward

        if terminated or truncated:
            ep_rewards.append(ep_reward)
            successes += int(terminated)              # terminated = goal reached
            episode += 1
            ep_reward = 0.0
            obs, _ = env.reset()
        else:
            obs = next_obs
        if (i + 1) % 20_000 == 0:
            print(f"  {i + 1:>8d}/{n} steps recorded | {episode} episodes finished")
    env.close()

    np.savez_compressed("dataset/robot_dataset.npz", observations=obs_buf, actions=act_buf,
                        rewards=rew_buf, next_observations=next_buf, terminals=term_buf,
                        timeouts=trunc_buf, episode=ep_buf)

    header = ",".join(["episode"] + OBS_NAMES + ["action", "reward"]
                      + ["next_" + c for c in OBS_NAMES] + ["terminated", "truncated"])
    table = np.column_stack([ep_buf, obs_buf, act_buf, rew_buf, next_buf, term_buf, trunc_buf])
    fmt = ["%d"] + ["%.4f"] * 9 + ["%d", "%.4f"] + ["%.4f"] * 9 + ["%d", "%d"]
    np.savetxt("dataset/robot_dataset.csv", table, delimiter=",", header=header, comments="", fmt=fmt)

    info = {
        "transitions": n,
        "finished_episodes": episode,
        "goal_reached_episodes": successes,
        "success_rate_percent": round(100 * successes / max(episode, 1), 1),
        "mean_episode_reward": round(float(np.mean(ep_rewards)), 2) if ep_rewards else None,
        "behaviour_policy": args.behaviour,
        "random_action_probability": args.epsilon,
        "observation_columns": OBS_NAMES,
        "actions": {"0": "forward", "1": "turn left", "2": "turn right"},
        "seconds_to_collect": round(time.time() - start, 1),
    }
    json.dump(info, open("dataset/dataset_info.json", "w"), indent=2)
    print("\nDataset saved:")
    print("  dataset/robot_dataset.npz   (used by offline_qlearning.py)")
    print("  dataset/robot_dataset.csv   (open in Excel to view)")
    print("  dataset/dataset_info.json   (summary)")
    print(f"  {n} transitions, {episode} episodes, goal reached in {info['success_rate_percent']}% of episodes")
    print("Next: python offline_qlearning.py")


if __name__ == "__main__":
    main()