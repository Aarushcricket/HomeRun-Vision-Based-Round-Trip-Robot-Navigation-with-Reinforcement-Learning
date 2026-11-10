"""
plots.py
--------
Makes the figures for the report (saved in results/):
  learning_curves.png  training reward of the two online algorithms on one graph
  offline_training.png offline Q-Learning: TD error per epoch over the dataset
  paths.png            top-down view of each robot's round trip (first test layout)
  success_rates.png    bar chart of round-trip success rate

  python plots.py
"""
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")          # save images only, no window (avoids Tcl/Tk errors on Windows)
import matplotlib.pyplot as plt
import numpy as np

ALGOS = ["qlearning", "ppo", "offline_qlearning"]
LABELS = {"qlearning": "Q-Learning", "ppo": "PPO", "offline_qlearning": "Offline Q-Learning"}
COLORS = {"qlearning": "tab:orange", "ppo": "tab:purple", "offline_qlearning": "tab:green"}


def read_monitor(path):
    rewards, lengths = [], []
    with open(path) as f:
        next(f)                                    # skip the JSON header line
        for row in csv.DictReader(f):
            rewards.append(float(row["r"]))
            lengths.append(int(float(row["l"])))
    return np.array(rewards), np.array(lengths)


def moving_average(x, window):
    window = max(1, min(window, len(x)))
    return np.convolve(x, np.ones(window) / window, mode="valid")


def learning_curves():
    plt.figure(figsize=(8, 5))
    found = False
    for name in ALGOS:
        path = f"logs/{name}.monitor.csv"
        if not os.path.exists(path):
            continue
        rewards, lengths = read_monitor(path)
        if len(rewards) == 0:
            continue
        found = True
        steps = np.cumsum(lengths)
        smooth = moving_average(rewards, 100)
        plt.plot(steps[len(steps) - len(smooth):], smooth, label=LABELS[name], color=COLORS[name])
    if not found:
        print("No training logs found - run train.py first.")
        plt.close()
        return
    plt.xlabel("Training steps")
    plt.ylabel("Episode reward (moving average of 100)")
    plt.title("Learning curves")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/learning_curves.png", dpi=150)
    plt.close()
    print("Saved results/learning_curves.png")


def offline_curve():
    path = "logs/offline_qlearning_td.csv"
    if not os.path.exists(path):
        return
    data = np.genfromtxt(path, delimiter=",", names=True)
    plt.figure(figsize=(7, 4))
    plt.plot(np.atleast_1d(data["epoch"]), np.atleast_1d(data["mean_abs_td_error"]),
             marker="o", color=COLORS["offline_qlearning"])
    plt.xlabel("Epoch (one pass over the dataset)")
    plt.ylabel("Mean |TD error|")
    plt.title("Offline Q-Learning training")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/offline_training.png", dpi=150)
    plt.close()
    print("Saved results/offline_training.png")


def path_plots():
    if not os.path.exists("results/paths.json"):
        print("No results/paths.json - run evaluate.py first.")
        return
    data = json.load(open("results/paths.json"))
    names = [n for n in ALGOS if n in data]
    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 5), squeeze=False)
    for ax, name in zip(axes[0], names):
        layout, path = data[name]["layout"], np.array(data[name]["path"])
        h = layout["arena_half"]
        ax.plot([-h, h, h, -h, -h], [-h, -h, h, h, -h], color="gray")
        for ox, oy in layout["obstacles"]:
            ax.add_patch(plt.Circle((ox, oy), layout["obstacle_radius"], color="dimgray"))
        ax.add_patch(plt.Circle(layout["base"], 0.12, color="green"))
        ax.add_patch(plt.Circle(layout["target"], 0.12, color="red"))
        ax.plot(path[:, 0], path[:, 1], color=COLORS[name], linewidth=1.5)
        ax.plot(path[0, 0], path[0, 1], "ko", markersize=5)
        ax.set_title(LABELS[name])
        ax.set_aspect("equal")
        ax.set_xlim(-h - 0.2, h + 0.2)
        ax.set_ylim(-h - 0.2, h + 0.2)
    fig.suptitle("Round-trip paths (green = Base, red = Target, black dot = start)")
    fig.tight_layout()
    fig.savefig("results/paths.png", dpi=150)
    plt.close(fig)
    print("Saved results/paths.png")


def success_chart():
    if not os.path.exists("results/results.csv"):
        return
    with open("results/results.csv") as f:
        rows = list(csv.DictReader(f))
    names = [r["Algorithm"] for r in rows]
    values = [float(r["Round trip %"].rstrip("%")) for r in rows]
    by_label = {v: COLORS[k] for k, v in LABELS.items()}
    colors = [by_label.get(n, "gray") for n in names]
    plt.figure(figsize=(7, 4))
    bars = plt.bar(names, values, color=colors)
    for bar, v in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, v + 1, f"{v:.0f}%", ha="center")
    plt.ylim(0, 110)
    plt.ylabel("Round-trip success rate (%)")
    plt.title("Success rate by algorithm")
    plt.tight_layout()
    plt.savefig("results/success_rates.png", dpi=150)
    plt.close()
    print("Saved results/success_rates.png")


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    learning_curves()
    offline_curve()
    path_plots()
    success_chart()