"""
qlearning.py
------------
Tabular Q-Learning (a Temporal-Difference method), written from scratch so every
line of the algorithm is visible for the report and viva.

The camera features are continuous numbers, but a Q-table needs a finite number of
states, so each feature is cut into a few bins ("discretization"):

  goal direction : far-left / left / centre / right / far-right  (when visible)
                   lost-left / lost-right / unknown              (when not visible)  (8)
  goal distance  : far (> 2.5 m) / medium (1 - 2.5 m) / near (< 1 m)                 (3)
  obstacle L,C,R : clear / near / very near                                          (3 each)
  phase          : GO / RETURN                                                       (2)
  last action    : forward / left / right                                            (3)
  -> 8 * 3 * 3*3*3 * 2 * 3 = 3888 states, 3 actions

Update rule:  Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_a' Q(s',a') - Q(s,a) ]
"""
import json
import time

import numpy as np

OFFSET_EDGES = [-0.6, -0.2, 0.2, 0.6]
CLOSENESS_EDGES = [1 - 2.5 / 6.0, 1 - 1.0 / 6.0]     # 2.5 m and 1.0 m (closeness = 1 - d / 6)
OBST_EDGES = [0.15, 0.5]
STATE_SHAPE = (8, 3, 3, 3, 3, 2, 3)


def discretize(obs):
    offset, closeness, visible, ol, oc, orr, phase, last_left, last_right = obs
    if visible < 0.5:
        g = 5 if offset < -0.5 else (6 if offset > 0.5 else 7)   # lost left / lost right / unknown
        s = 0
    else:
        g = int(np.digitize(offset, OFFSET_EDGES))
        s = int(np.digitize(closeness, CLOSENESS_EDGES))
    obst = [int(np.digitize(v, OBST_EDGES)) for v in (ol, oc, orr)]
    last = 1 if last_left > 0.5 else (2 if last_right > 0.5 else 0)
    return int(np.ravel_multi_index((g, s, *obst, int(round(phase)), last), STATE_SHAPE))


class QLearningAgent:
    def __init__(self, n_actions=3, alpha=0.1, gamma=0.99,
                 eps_start=1.0, eps_end=0.05, eps_decay_fraction=0.5, seed=0):
        self.n_actions = n_actions
        self.alpha, self.gamma = alpha, gamma
        self.eps_start, self.eps_end, self.eps_decay_fraction = eps_start, eps_end, eps_decay_fraction
        self.q = np.zeros((int(np.prod(STATE_SHAPE)), n_actions), dtype=np.float32)
        self.rng = np.random.default_rng(seed)

    def epsilon(self, step, total):
        """Exploration rate: decays linearly from eps_start to eps_end."""
        frac = min(1.0, step / (self.eps_decay_fraction * total))
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def predict(self, obs):
        """Greedy action (used after training)."""
        return int(np.argmax(self.q[discretize(obs)]))

    def learn(self, env, total_timesteps, log_path, print_every=10_000):
        """Train for a fixed number of environment steps (same budget as PPO)."""
        t_start = time.time()
        log = open(log_path, "w")
        log.write("#" + json.dumps({"t_start": t_start, "env_id": "RoundTripEnv"}) + "\n")
        log.write("r,l,t\n")                      # same format as Stable-Baselines3 Monitor logs

        obs, _ = env.reset()
        s = discretize(obs)
        ep_reward, ep_len, recent = 0.0, 0, []
        for step in range(1, total_timesteps + 1):
            # epsilon-greedy action selection
            if self.rng.random() < self.epsilon(step, total_timesteps):
                a = int(self.rng.integers(self.n_actions))
            else:
                a = int(np.argmax(self.q[s]))

            obs, r, terminated, truncated, _ = env.step(a)
            s_next = discretize(obs)

            # TD target: no future value if the goal was reached (true terminal state)
            target = r if terminated else r + self.gamma * np.max(self.q[s_next])
            self.q[s, a] += self.alpha * (target - self.q[s, a])

            s = s_next
            ep_reward += r
            ep_len += 1
            if terminated or truncated:
                log.write(f"{ep_reward:.4f},{ep_len},{time.time() - t_start:.2f}\n")
                recent = (recent + [ep_reward])[-50:]
                obs, _ = env.reset()
                s = discretize(obs)
                ep_reward, ep_len = 0.0, 0

            if step % print_every == 0:
                mean = np.mean(recent) if recent else float("nan")
                print(f"  [qlearning] {step:>8d}/{total_timesteps} steps | "
                      f"mean reward (last 50 ep) = {mean:7.2f} | eps = {self.epsilon(step, total_timesteps):.2f}")
        log.close()

    def save(self, path):
        np.save(path, self.q)

    @classmethod
    def load(cls, path):
        agent = cls()
        agent.q = np.load(path)
        return agent
