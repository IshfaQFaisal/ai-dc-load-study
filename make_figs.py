import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = "."
from analyze_rare import load, fit, g_factory, subset_sim, p_at, OUT
from rare_lib import physical, DIM, RHO

plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": 0.6, "lines.linewidth": 1.0,
                     "legend.fontsize": 7, "legend.frameon": False, "savefig.bbox": "tight",
                     "savefig.pad_inches": 0.02})
BLUE, ORANGE, GREEN, PURPLE, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8e5bd0", "#8a8984"
W1, W2 = 3.5, 7.16


def fig_profile():
    d = pd.read_csv(f"{SRC}/ai_load_profile.csv")
    t = d.t_s / 60
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.1), constrained_layout=True)
    for col, c, lab in (("P_train_MW", BLUE, "Training"), ("P_inference_MW", GREEN, "Inference"),
                        ("P_finetune_MW", ORANGE, "Fine-tuning")):
        ax[0].plot(t, d[col], color=c, lw=0.6, label=lab)
    ax[0].set(xlabel="Time (min)", ylabel="IT power (MW)", xlim=(0, 60))
    ax[0].legend(loc="center right")
    ax[1].plot(t, d.P_facility_MW, color="k", lw=0.6)
    ax[1].set(xlabel="Time (min)", ylabel="Facility power (MW)", xlim=(0, 60))
    for a, s in zip(ax, "ab"):
        a.grid(alpha=0.25, lw=0.4); a.set_title(f"({s})", loc="left", fontsize=8)
    fig.savefig("fig_profile.pdf")


def fig_part1():
    label = {"baseline": "Baseline", "ramp_20MWps": "Ramp limit 20 MW/s", "ramp_10MWps": "Ramp limit 10 MW/s",
             "floor90_30s": "Power floor", "ft_droop": "Freq.-responsive fine-tuning"}
    color = {"baseline": GRAY, "ramp_20MWps": BLUE, "ramp_10MWps": GREEN, "floor90_30s": ORANGE, "ft_droop": PURPLE}
    events = {"trip": ("(a) Training trip", 2390), "restart": ("(b) Training restart", 2630),
              "ckpt": ("(c) Checkpoint dip", 620)}
    fig, axes = plt.subplots(1, 3, figsize=(W2, 2.2), constrained_layout=True)
    for ax, (ev, (title, t0)) in zip(axes, events.items()):
        for s in label:
            d = pd.read_csv(f"{SRC}/runs_part1/{s}_{ev}.csv")
            ax.plot(d.t + t0, d.f_coi_Hz, color=color[s], lw=0.9, label=label[s])
        ax.set_title(title, loc="left", fontsize=8); ax.set_xlabel("Time (s)"); ax.grid(alpha=0.25, lw=0.4)
    axes[0].set_ylabel("COI frequency (Hz)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=5)
    fig.savefig("fig_part1.pdf")


def fig_part2():
    rng = np.random.default_rng(3)
    Xtr, tr = load("train")
    gp = fit(Xtr, tr[OUT].values)
    grid = np.linspace(0, 600, 241)
    curves, fail = {}, None
    for rho in (RHO, 0.0):
        runs = []
        for rep in range(10):
            lv, u_f, g_f = subset_sim(g_factory(gp, rho), 550.0, rng)
            runs.append([p_at(lv, y) for y in grid])
            if rho == RHO and rep == 0:
                fail = u_f[g_f > 500]
        P = np.array(runs); P[P <= 0] = np.nan
        curves[rho] = P
    gC = g_factory(gp, RHO)
    vals = np.concatenate([gC(rng.standard_normal((200_000, DIM))) for _ in range(10)])
    pm = np.array([np.mean(vals > y) for y in grid])

    fig, ax = plt.subplots(figsize=(W1, 2.5), constrained_layout=True)
    for rho, c, lab in ((RHO, BLUE, rf"Correlated trips ($\rho$ = {RHO})"), (0.0, ORANGE, r"Independent trips ($\rho$ = 0)")):
        P = curves[rho]
        ax.fill_between(grid, np.nanmin(P, 0), np.nanmax(P, 0), color=c, alpha=0.2, lw=0)
        ax.semilogy(grid, np.nanmedian(P, 0), color=c, lw=1.1, label=lab)
    ax.semilogy(grid[pm > 0], pm[pm > 0], color="k", ls="--", lw=0.8, label=r"Crude MC on surrogate ($2\times10^6$)")
    ax.axvline(500, color=GRAY, ls=":", lw=0.8)
    ax.set(xlabel="Maximum RoCoF over 0.5 s (mHz/s)", ylabel="Exceedance probability", ylim=(1e-9, 1.5), xlim=(0, 600))
    ax.grid(alpha=0.25, lw=0.4, which="major"); ax.legend(loc="lower left")
    fig.savefig("fig_tail.pdf")

    tf, hf, _ = physical(fail, RHO)
    t0, h0, _ = physical(rng.standard_normal((200_000, DIM)), RHO)
    fig, ax = plt.subplots(1, 2, figsize=(W1, 1.8), constrained_layout=True)
    k = np.arange(6)
    ax[0].bar(k - 0.2, [np.mean(t0.sum(1) == i) for i in k], 0.4, color=GRAY, label="All")
    ax[0].bar(k + 0.2, [np.mean(tf.sum(1) == i) for i in k], 0.4, color=BLUE, label="> 0.5 Hz/s")
    ax[0].set(xlabel="Sites tripped", ylabel="Share", xticks=k); ax[0].legend(loc="upper center")
    bins = np.linspace(0.4, 1.0, 25)
    ax[1].hist(h0, bins, density=True, color=GRAY, alpha=0.7)
    ax[1].hist(hf, bins, density=True, color=BLUE, alpha=0.8)
    ax[1].set(xlabel="Inertia multiplier", ylabel="Density")
    for a, s in zip(ax, "ab"):
        a.set_title(f"({s})", loc="left", fontsize=8); a.grid(alpha=0.25, lw=0.4)
    fig.savefig("fig_anatomy.pdf")


def main():
    fig_profile()
    fig_part1()
    fig_part2()
    print("figures saved")


if __name__ == "__main__":
    main()
