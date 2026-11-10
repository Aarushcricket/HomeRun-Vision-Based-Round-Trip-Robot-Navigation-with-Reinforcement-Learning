"""
offline_qlearning.py
--------------------
OFFLINE Q-Learning: learns ONLY from the recorded dataset (dataset/robot_dataset.npz).
It never runs the simulator while learning.

It uses exactly the same TD update rule as online Q-Learning:
    Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_a' Q(s',a') - Q(s,a) ]
but instead of new experience from the robot, it sweeps over the stored experiences
many times (epochs).

The offline problem: the agent cannot try actions that are missing from the data, so
plain Q-Learning can be over-optimistic about them ("extrapolation error").
Fix used here (a simple, tabular form of the idea behind BCQ / CQL):
    an action only counts if the dataset contains it at least MIN_COUNT times in
    that state. The max in the TD target and the final greedy choice both ignore
    unsupported actions.  Use --plain to switch this off and compare.

  python offline_qlearning.py               # 20 epochs, with the support constraint
  python offline_qlearning.py --plain       # plain offline Q-Learning (no constraint)

Outputs: models/offline_qlearning.npy, logs/offline_qlearning_td.csv
"""
import argparse
import json
import os
import time

import numpy as np

from qlearning import STATE_SHAPE, discretize

N_STATES = int(np.prod(STATE_SHAPE))
UNSUPPORTED = -1e6          # value written for actions the dataset does not support


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--min-count", type=int, default=3)
    parser.add_argument("--plain", action="store_true", help="no support constraint")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not os.path.exists("dataset/robot_dataset.npz"):
        print("No dataset found - run: python collect_dataset.py")
        return
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    start = time.time()

    data = np.load("dataset/robot_dataset.npz")
    print(f"Loaded {len(data['actions'])} transitions from dataset/robot_dataset.npz")
    S = np.array([discretize(o) for o in data["observations"]])
    S2 = np.array([discretize(o) for o in data["next_observations"]])
    A, R, DONE = data["actions"], data["rewards"], data["terminals"]

    # How often each (state, action) appears in the data
    counts = np.zeros((N_STATES, 3), np.int64)
    np.add.at(counts, (S, A), 1)
    supported = np.ones_like(counts, bool) if args.plain else counts >= args.min_count
    print(f"States seen in the dataset: {int((counts.sum(1) > 0).sum())} of {N_STATES}")

    q = np.zeros((N_STATES, 3), np.float64)
    rng = np.random.default_rng(args.seed)
    td_log = open("logs/offline_qlearning_td.csv", "w")
    td_log.write("epoch,mean_abs_td_error\n")
    for epoch in range(1, args.epochs + 1):
        total = 0.0
        for i in rng.permutation(len(A)):
            s, a, s2 = S[i], A[i], S2[i]
            if DONE[i]:
                target = R[i]                                  # goal reached: no future value
            else:
                allowed = q[s2][supported[s2]]
                target = R[i] + args.gamma * (allowed.max() if allowed.size else 0.0)
            delta = target - q[s, a]
            q[s, a] += args.alpha * delta
            total += abs(delta)
        mean_td = total / len(A)
        td_log.write(f"{epoch},{mean_td:.5f}\n")
        print(f"  epoch {epoch:2d}/{args.epochs} | mean |TD error| = {mean_td:.4f}")
    td_log.close()

    # Final policy: greedy over supported actions only. In a state with no supported
    # action, fall back to the action the dataset used most there.
    policy_q = np.where(supported, q, UNSUPPORTED)
    no_support = ~supported.any(1) & (counts.sum(1) > 0)
    policy_q[no_support] = UNSUPPORTED
    policy_q[no_support, counts[no_support].argmax(1)] = 0.0
    policy_q[counts.sum(1) == 0] = 0.0                          # never-seen states
    np.save("models/offline_qlearning.npy", policy_q.astype(np.float32))

    minutes = (time.time() - start) / 60
    times_path = "models/train_times.json"
    times = json.load(open(times_path)) if os.path.exists(times_path) else {}
    times["offline_qlearning"] = round(minutes, 2)
    json.dump(times, open(times_path, "w"), indent=2)
    mode = "plain" if args.plain else f"support constraint (min {args.min_count} samples)"
    print(f"\nOffline Q-Learning done in {minutes:.1f} min ({mode}).")
    print("Saved models/offline_qlearning.npy. Next: python evaluate.py")


if __name__ == "__main__":
    main()