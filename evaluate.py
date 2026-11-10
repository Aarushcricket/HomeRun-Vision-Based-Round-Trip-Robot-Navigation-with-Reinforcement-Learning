"""
evaluate.py
-----------
Runs the ROUND-TRIP mission (Base -> Target -> Base) for all three trained agents
(online Q-Learning, PPO, offline Q-Learning), one after another, always starting from
the Base. All agents are tested on the same set of layouts, so the comparison is fair.
Prints a results table at the end.

  python evaluate.py                  # 10 trials per algorithm, fast (no window)
  python evaluate.py --gui            # watch it: 3D window + robot camera view
  python evaluate.py --gui --trials 1 # quick demo, one round trip per algorithm

Outputs: results/results.csv, results/paths.json (used by plots.py)
"""
import argparse
import csv
import json
import os
import time

import numpy as np

from qlearning import QLearningAgent
from robot_env import GO, RoundTripEnv

ALGOS = ["qlearning", "ppo", "offline_qlearning"]
LABELS = {"qlearning": "Q-Learning", "ppo": "PPO", "offline_qlearning": "Offline Q-Learning"}
EVAL_EPSILON = 0.05            # same small chance of a random action for every agent


def load_agent(name):
    if name in ("qlearning", "offline_qlearning"):
        path = f"models/{name}.npy"
        return QLearningAgent.load(path) if os.path.exists(path) else None
    path = "models/ppo.zip"
    if not os.path.exists(path):
        return None
    from stable_baselines3 import PPO
    return PPO.load(path, device="cpu")


def choose_action(name, agent, obs, rng):
    # A small random action now and then stops a robot from getting stuck in an
    # endless left-right loop. This is the standard evaluation trick from the
    # original DQN paper (epsilon = 0.05), applied equally to every agent.
    if rng.random() < EVAL_EPSILON:
        return int(rng.integers(3))
    if name != "ppo":
        return agent.predict(obs)
    action, _ = agent.predict(obs, deterministic=True)
    return int(action)


def round_trip(env, name, agent, layout_seed, show=False):
    """One full mission. Returns a dict of statistics."""
    obs, _ = env.reset(options={"layout_seed": layout_seed})
    # The random-action generator is seeded by the layout, so every agent gets the same
    # random numbers on the same trial, and a trial gives the same result with or
    # without the GUI (the demo replays exactly what the evaluation measured).
    rng = np.random.default_rng(layout_seed)
    stats = {"reached_target": False, "back_at_base": False, "steps_go": 0,
             "steps_return": 0, "collisions": 0, "path_m": 0.0}
    start = time.time()
    while True:
        leg = env.phase                                     # leg BEFORE this action
        obs, _, terminated, truncated, info = env.step(choose_action(name, agent, obs, rng))
        stats["steps_go" if leg == GO else "steps_return"] += 1
        stats["collisions"] += int(info["collision"])
        stats["path_m"] += info["moved"]
        if info["event"] == "target_reached":
            stats["reached_target"] = True
        if info["event"] == "base_reached":
            stats["back_at_base"] = True

        if show:
            import cv2
            from vision import draw_debug
            leg_name = "GO -> Target" if env.phase == GO else "RETURN -> Base"
            cv2.imshow("Robot camera", draw_debug(env.last_rgb, env.last_debug,
                                                  title=f"{LABELS[name]} | {leg_name}"))
            cv2.waitKey(1)
            time.sleep(0.03)
        if terminated or truncated:
            break
    stats["time_s"] = time.time() - start
    stats["path_xy"] = [list(pt) for pt in env.path]
    return stats


def avg(values):
    return f"{np.mean(values):.1f}" if values else "-"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()
    os.makedirs("results", exist_ok=True)

    times = json.load(open("models/train_times.json")) if os.path.exists("models/train_times.json") else {}
    layout_seeds = [1000 + i for i in range(args.trials)]     # same layouts for every agent
    env = RoundTripEnv(gui=args.gui)
    rows, paths = [], {}

    for name in ALGOS:
        agent = load_agent(name)
        if agent is None:
            hint = "python offline_qlearning.py" if name == "offline_qlearning" else f"python train.py --algo {name}"
            print(f"[skip] no trained model for {name} - run: {hint}")
            continue
        print(f"\n--- {LABELS[name]}: {args.trials} round trips, starting from Base ---")
        trials = []
        for i, seed in enumerate(layout_seeds):
            s = round_trip(env, name, agent, seed, show=args.gui)
            trials.append(s)
            status = "FULL ROUND TRIP" if s["back_at_base"] else ("target only" if s["reached_target"] else "failed")
            print(f"  trial {i + 1:2d}: {status:15s} | GO {s['steps_go']:3d} steps | "
                  f"RETURN {s['steps_return']:3d} steps | collisions {s['collisions']}")
            if i == 0:
                paths[name] = {"layout": env.layout, "path": s["path_xy"]}

        full = [t for t in trials if t["back_at_base"]]
        rows.append({
            "Algorithm": LABELS[name],
            "Round trip %": f"{100 * len(full) / len(trials):.0f}%",
            "Target %": f"{100 * sum(t['reached_target'] for t in trials) / len(trials):.0f}%",
            "Steps GO": avg([t["steps_go"] for t in trials if t["reached_target"]]),
            "Steps RETURN": avg([t["steps_return"] for t in full]),
            "Collisions": f"{np.mean([t['collisions'] for t in trials]):.1f}",
            "Path (m)": avg([t["path_m"] for t in full]),
            "Time (s)": avg([t["time_s"] for t in full]),
            "Train (min)": str(times.get(name, "-")),
        })

    env.close()
    if args.gui:
        import cv2
        cv2.destroyAllWindows()
    if not rows:
        return

    cols = list(rows[0].keys())
    widths = {c: max(len(c), *(len(r[c]) for r in rows)) for c in cols}
    print(f"\n=== Round-Trip Navigation Results ({args.trials} trials each) ===")
    print(" | ".join(c.ljust(widths[c]) for c in cols))
    print("-+-".join("-" * widths[c] for c in cols))
    for r in rows:
        print(" | ".join(r[c].ljust(widths[c]) for c in cols))
    print("(Steps, path and time are averaged over successful trips only.)")

    with open("results/results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)
    json.dump(paths, open("results/paths.json", "w"))
    print("\nSaved results/results.csv and results/paths.json. Next: python plots.py")


if __name__ == "__main__":
    main()