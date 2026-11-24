import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
LABEL = {"baseline": "Baseline (no mitigation)", "ramp_20MWps": "Ramp limit 20 MW/s",
         "ramp_10MWps": "Ramp limit 10 MW/s", "floor90_30s": "Power floor 90 %, 30 s",
         "ft_droop": "Freq.-responsive fine-tuning"}
COLOR = {"baseline": "#8a8984", "ramp_20MWps": "#2a78d6", "ramp_10MWps": "#1baf7a",
         "floor90_30s": "#eb6834", "ft_droop": "#8e5bd0"}
EVENTS = {"trip": ("Training trip (−345 MW)", 2390), "restart": ("Training restart (+340 MW)", 2630),
          "ckpt": ("Checkpoint dip (−300 MW, 25 s)", 620)}
METRICS = {"f_dev_max_mHz": "Max |Δf| (mHz)", "rocof_max_mHz_s": "Max RoCoF, 0.5 s (mHz/s)"}

def main():
    r = pd.read_csv("results_part1.csv")
    order = list(LABEL)
    y = np.arange(len(order))[::-1]

    fig, axes = plt.subplots(2, 3, figsize=(12, 5.2), sharey=True, constrained_layout=True)
    for j, (ev, (title, _)) in enumerate(EVENTS.items()):
        d = r[r.event == ev].set_index("scenario").loc[order]
        for i, (m, mlab) in enumerate(METRICS.items()):
            ax, v = axes[i, j], d[m].values
            ax.barh(y, v, height=0.62, color=[COLOR[s] for s in order])
            for yy, vv in zip(y, v):
                ax.text(vv + v.max() * 0.02, yy, f"{vv:.1f}", va="center", fontsize=8)
            ax.set_xlim(0, v.max() * 1.2)
            ax.grid(axis="x", alpha=0.3)
            if i == 0: ax.set_title(title, loc="left")
            if j == 0: ax.set_ylabel(mlab)
    axes[0, 0].set_yticks(y, [LABEL[s] for s in order])
    fig.suptitle("Mitigation options, ~700 MW AI campus at bus 16, IEEE 39-bus (ANDES)", x=0.01, ha="left")
    fig.savefig("fig_part1_mitigation.png", dpi=300)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    for ax, (ev, (title, t0)) in zip(axes, EVENTS.items()):
        for s in order:
            d = pd.read_csv(f"runs_part1/{s}_{ev}.csv")
            ax.plot(d.t + t0, d.f_coi_Hz, color=COLOR[s], lw=1.3, label=LABEL[s])
        ax.set_title(title, loc="left"); ax.set_xlabel("Profile time (s)"); ax.grid(alpha=0.3)
    axes[0].set_ylabel("COI frequency (Hz)"); axes[0].legend(frameon=False, fontsize=8)
    fig.savefig("fig_part1_traces.png", dpi=300)


if __name__ == "__main__":
    main()
