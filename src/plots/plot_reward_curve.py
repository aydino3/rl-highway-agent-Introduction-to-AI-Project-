from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_from_monitor(monitor_csv: str, out_path: str, window: int = 50) -> None:
    df = pd.read_csv(monitor_csv, comment="#")

    if "r" not in df.columns:
        raise KeyError(f"'r' column not found in {monitor_csv}. Columns: {list(df.columns)}")

    df["episode"] = range(1, len(df) + 1)
    df["reward_ma"] = df["r"].rolling(window=window, min_periods=1).mean()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(df["episode"], df["r"])
    plt.plot(df["episode"], df["reward_ma"])
    plt.xlabel("Episodes")
    plt.ylabel("Episode Reward")
    plt.title("Reward vs Episodes")
    plt.legend(["Reward", f"Moving Avg (window={window})"])
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Saved plot to: {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--monitor", type=str, default="outputs/logs/monitor.csv")
    p.add_argument("--out", type=str, default="outputs/plots/reward_vs_episodes.png")
    p.add_argument("--window", type=int, default=50)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    plot_from_monitor(args.monitor, args.out, window=args.window)


if __name__ == "__main__":
    main()
