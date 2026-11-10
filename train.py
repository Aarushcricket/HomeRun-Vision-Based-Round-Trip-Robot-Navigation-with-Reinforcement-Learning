"""
train.py
--------
Trains both agents (Q-Learning and PPO) with the SAME number of environment steps,
so the comparison is fair.

  python train.py                       # train qlearning and ppo (300,000 steps each)
  python train.py --algo ppo            # train only one algorithm
  python train.py --steps 50000         # quick test run

Outputs:
  models/qlearning.npy, models/ppo.zip                    trained agents
  logs/<algo>.monitor.csv                                 episode rewards (for plots.py)
  models/train_times.json                                 training time per algorithm
"""
import argparse
import json
import os
import time

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from qlearning import QLearningAgent
from robot_env import RoundTripEnv

ALGOS = ["qlearning", "ppo"]


class ProgressPrinter(BaseCallback):
    """Prints the average episode reward every `every` steps."""

    def __init__(self, name, monitor, total, every=10_000):
        super().__init__()
        self.name, self.monitor, self.total, self.every = name, monitor, total, every

    def _on_step(self):
        if self.num_timesteps % self.every == 0:
            rewards = self.monitor.get_episode_rewards()[-50:]
            mean = np.mean(rewards) if rewards else float("nan")
            print(f"  [{self.name}] {self.num_timesteps:>8d}/{self.total} steps | "
                  f"mean reward (last 50 ep) = {mean:7.2f}")
        return True


def train(name, steps, seed):
    print(f"\n=== Training {name.upper()} for {steps} steps ===")
    env = RoundTripEnv(gui=False, seed=seed)
    start = time.time()

    if name == "qlearning":
        agent = QLearningAgent(seed=seed)
        agent.learn(env, steps, log_path="logs/qlearning.monitor.csv")
        agent.save("models/qlearning.npy")
    else:
        monitored = Monitor(env, filename=f"logs/{name}")
        model = PPO("MlpPolicy", monitored, learning_rate=3e-4, n_steps=2048, batch_size=64,
                    n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
                    policy_kwargs=dict(net_arch=[64, 64]), seed=seed, device="cpu", verbose=0)
        model.learn(total_timesteps=steps, callback=ProgressPrinter(name, monitored, steps))
        model.save(f"models/{name}")

    minutes = (time.time() - start) / 60
    env.close()
    print(f"=== {name.upper()} done in {minutes:.1f} min ===")

    times_path = "models/train_times.json"
    times = json.load(open(times_path)) if os.path.exists(times_path) else {}
    times[name] = round(minutes, 2)
    json.dump(times, open(times_path, "w"), indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=ALGOS + ["all"], default="all")
    parser.add_argument("--steps", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    print("Checking the environment follows the Gymnasium API...")
    env = RoundTripEnv(gui=False, seed=args.seed)
    check_env(env, warn=True)
    env.close()
    print("Environment OK.")

    for name in (ALGOS if args.algo == "all" else [args.algo]):
        train(name, args.steps, args.seed)
    print("\nAll training finished. Next: python evaluate.py")


if __name__ == "__main__":
    main()
