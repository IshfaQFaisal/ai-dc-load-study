import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from multiprocessing import Pool
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
from rare_lib import physical, features, simulate, DIM, RHO, RATED_MW

Y_STAR = 500.0
LEVELS = [100, 200, 300, 400, 500]
OUT = "rocof_mHz_s"
SCALE = np.array([200.0] * 5 + [1.0, 1.0])


def load(name):
    d = pd.read_csv(f"rare_design_{name}.csv")
    X = d[[f"trip{i}" for i in range(1, 6)]].values * 0.8 * RATED_MW * d.L.values[:, None]
    return np.column_stack([X, d.h, d.L]) / SCALE, d


def fit(X, y):
    k = C(1.0) * Matern(length_scale=[1.0] * 7, nu=2.5) + WhiteKernel(1e-3)
    return GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=5, random_state=0).fit(X, y)


def g_factory(gp, rho):
    def g(u):
        t, h, L = physical(u, rho)
        return gp.predict(features(t, h, L) / SCALE)
    return g


def subset_sim(g, y_max, rng, N=5000, p0=0.1, max_levels=12):
    nc, ns = int(p0 * N), int(1 / p0)
    u = rng.standard_normal((N, DIM)); gu = g(u)
    levels, P = [(1.0, gu)], 1.0
    for _ in range(max_levels):
        order = np.argsort(gu)[::-1]
        b = gu[order[nc - 1]]
        if b >= y_max:
            break
        P *= p0
        cu, cg = u[order[:nc]].copy(), gu[order[:nc]].copy()
        chains_u, chains_g = [cu.copy()], [cg.copy()]
        for _ in range(ns - 1):
            cand = cu + rng.standard_normal(cu.shape)
            keep = rng.random(cu.shape) < np.minimum(1, np.exp(-0.5 * (cand**2 - cu**2)))
            cand = np.where(keep, cand, cu)
            gc = g(cand)
            acc = gc > b
            cu = np.where(acc[:, None], cand, cu); cg = np.where(acc, gc, cg)
            chains_u.append(cu.copy()); chains_g.append(cg.copy())
        u, gu = np.vstack(chains_u), np.concatenate(chains_g)
        levels.append((P, gu))
    return levels, u, gu


def p_at(levels, y):
    for P, gl in reversed(levels):
        if gl.min() <= y:
            return P * np.mean(gl > y)
    return levels[-1][0] * np.mean(levels[-1][1] > y)


def main():
    rng = np.random.default_rng(3)
    Xtr, tr = load("train"); Xck, ck = load("check")
    gp = fit(Xtr, tr[OUT].values)
    pred = gp.predict(Xck)
    r2 = 1 - np.sum((pred - ck[OUT]) ** 2) / np.sum((ck[OUT] - ck[OUT].mean()) ** 2)
    rmse = np.sqrt(np.mean((pred - ck[OUT]) ** 2))
    print(f"surrogate check: R2 = {r2:.4f}, RMSE = {rmse:.2f} mHz/s")

    rows, curves, tail = [], {}, None
    for rho in (RHO, 0.0):
        g = g_factory(gp, rho)
        ps, cs = [], []
        for rep in range(10):
            lv, u_f, g_f = subset_sim(g, Y_STAR * 1.1, rng)
            ps.append([p_at(lv, y) for y in LEVELS]); cs.append(lv)
            if rho == RHO and rep == 0:
                tail = (u_f, g_f)
        ps = np.array(ps)
        curves[rho] = cs
        for j, y in enumerate(LEVELS):
            m, s = ps[:, j].mean(), ps[:, j].std(ddof=1)
            rows.append(dict(rho=rho, threshold=y, p_sus=m, cov_sus=s / m if m > 0 else np.nan))

    gC = g_factory(gp, RHO)
    vals = np.concatenate([gC(rng.standard_normal((200_000, DIM))) for _ in range(10)])
    for r in rows:
        if r["rho"] == RHO:
            r["p_mc_surrogate_2e6"] = np.mean(vals > r["threshold"])

    u_f, g_f = tail
    fail_idx = np.where(g_f > Y_STAR)[0]
    near_idx = np.where((g_f > 0.85 * Y_STAR) & (g_f <= Y_STAR))[0]
    pick = lambda idx, n: rng.choice(idx, size=min(n, len(idx)), replace=False)
    sel = np.concatenate([pick(fail_idx, 30), pick(near_idx, 20)])
    t, h, L = physical(u_f[sel], RHO)
    X = [list(t[i]) + [h[i], L[i]] for i in range(len(sel))]
    with Pool(2) as pool:
        true = pd.DataFrame(pool.map(simulate, X))
    true["surrogate"] = g_f[sel]
    true["group"] = ["fail"] * min(30, len(fail_idx)) + ["near"] * min(20, len(near_idx))
    true.to_csv("rare_tail_check.csv", index=False)
    a = np.mean(true.loc[true.group == "fail", OUT] > Y_STAR)
    b = np.mean(true.loc[true.group == "near", OUT] > Y_STAR) if (true.group == "near").any() else 0.0
    p_fail = np.mean([p_at(lv, Y_STAR) for lv in curves[RHO]])
    p_near = np.mean([p_at(lv, 0.85 * Y_STAR) - p_at(lv, Y_STAR) for lv in curves[RHO]])
    p_corr = p_fail * a + p_near * b
    tail_rmse = np.sqrt(np.mean((true.surrogate - true[OUT]) ** 2))
    print(f"tail check: a = {a:.2f}, b = {b:.2f}, tail RMSE = {tail_rmse:.1f} mHz/s; "
          f"P(RoCoF > {Y_STAR:.0f}) surrogate = {p_fail:.2e}, corrected = {p_corr:.2e}")

    res = pd.DataFrame(rows)
    res.loc[(res.rho == RHO) & (res.threshold == Y_STAR), "p_corrected"] = p_corr
    res["andes_runs_for_10pct_cov"] = (1 - res.p_sus) / (res.p_sus * 0.01)
    res.to_csv("rare_results.csv", index=False)
    pd.set_option("display.width", 200)
    print(res.to_string(index=False, float_format=lambda v: f"{v:.3g}"))

    tf, hf, Lf = physical(u_f[fail_idx], RHO)
    t0, h0, _ = physical(rng.standard_normal((200_000, DIM)), RHO)
    anat = dict(k_fail=tf.sum(1), h_fail=hf, k_all=t0.sum(1), h_all=h0)
    print("among failures: mean sites tripped", tf.sum(1).mean().round(2),
          "| share with 5 sites", np.mean(tf.sum(1) == 5).round(2),
          "| median inertia", np.median(hf).round(2))

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#8a8984"

    fig, ax = plt.subplots(figsize=(7.5, 4.2), constrained_layout=True)
    for rho, col, lab in ((RHO, BLUE, f"Correlated trips (ρ = {RHO})"), (0.0, ORANGE, "Independent trips (ρ = 0)")):
        grid = np.linspace(0, 600, 241)
        P = np.array([[p_at(lv, y) for y in grid] for lv in curves[rho]])
        P = np.where(P > 0, P, np.nan)
        ax.fill_between(grid, np.nanmin(P, 0), np.nanmax(P, 0), color=col, alpha=0.2, lw=0)
        ax.semilogy(grid, np.nanmedian(P, 0), color=col, lw=1.6, label=lab + " — subset simulation (median, range of 10)")
    yy = np.linspace(0, vals.max(), 200)
    pm = np.array([np.mean(vals > y) for y in yy])
    ax.semilogy(yy[pm > 0], pm[pm > 0], color="k", lw=1.2, ls="--", label="Brute-force MC on surrogate (2×10⁶)")
    ax.errorbar([Y_STAR], [p_corr], fmt="o", color="#8e5bd0", ms=6, label="Bias-corrected with ANDES tail runs")
    ax.axvline(Y_STAR, color=GRAY, lw=1, ls=":")
    ax.text(Y_STAR + 5, 3e-1, "0.5 Hz/s\nRoCoF setting", fontsize=8, color=GRAY)
    ax.set(xlabel="Max RoCoF, 0.5 s window (mHz/s)", ylabel="P(RoCoF > y | disturbance)", ylim=(1e-8, 1.5))
    ax.grid(alpha=0.3, which="both"); ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.set_title("Risk of large RoCoF from data-centre trips (5 sites × 250 MW)", loc="left")
    fig.savefig("fig_part2_tail.png", dpi=300)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), constrained_layout=True)
    k = np.arange(6)
    axes[0].bar(k - 0.2, [np.mean(anat["k_all"] == i) for i in k], 0.4, color=GRAY, label="All disturbances")
    axes[0].bar(k + 0.2, [np.mean(anat["k_fail"] == i) for i in k], 0.4, color=BLUE, label=f"RoCoF > {Y_STAR:.0f} mHz/s")
    axes[0].set(xlabel="Sites tripped", ylabel="Share", title="How many sites trip"); axes[0].legend(frameon=False, fontsize=8)
    bins = np.linspace(0.4, 1.0, 25)
    axes[1].hist(anat["h_all"], bins, density=True, color=GRAY, alpha=0.7, label="All disturbances")
    axes[1].hist(anat["h_fail"], bins, density=True, color=BLUE, alpha=0.7, label=f"RoCoF > {Y_STAR:.0f} mHz/s")
    axes[1].set(xlabel="Inertia multiplier", ylabel="Density", title="System inertia at the event"); axes[1].legend(frameon=False, fontsize=8)
    for a_ in axes: a_.grid(alpha=0.3)
    fig.suptitle("What a rare severe event looks like", x=0.01, ha="left")
    fig.savefig("fig_part2_anatomy.png", dpi=300)

    fig, ax = plt.subplots(figsize=(4.8, 4.2), constrained_layout=True)
    lim = [0, max(ck[OUT].max(), true[OUT].max()) * 1.05]
    ax.plot(lim, lim, color=GRAY, lw=1)
    ax.scatter(ck[OUT], pred, s=16, color=BLUE, label=f"Check runs (R² = {r2:.3f})")
    ax.scatter(true[OUT], true.surrogate, s=16, color=ORANGE, label="Tail verification runs")
    ax.axvline(Y_STAR, color=GRAY, ls=":", lw=1); ax.axhline(Y_STAR, color=GRAY, ls=":", lw=1)
    ax.set(xlabel="ANDES RoCoF (mHz/s)", ylabel="Surrogate RoCoF (mHz/s)", xlim=lim, ylim=lim)
    ax.legend(frameon=False, fontsize=8); ax.grid(alpha=0.3)
    fig.savefig("fig_part2_validation.png", dpi=300)
    print("saved figures")


if __name__ == "__main__":
    main()
