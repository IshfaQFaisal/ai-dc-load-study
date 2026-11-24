import numpy as np
import pandas as pd
from scipy.stats import norm
from analyze_rare import subset_sim, p_at, fit, SCALE
from rare_lib import features, RATED_MW

BASE = dict(rho=0.7, p_trip=0.10, h_mean=0.8, h_sd=0.12, h_min=0.4)
CASES = [("base", {})] + \
        [(f"rho={v}", dict(rho=v)) for v in (0.0, 0.3, 0.5, 0.9)] + \
        [(f"p_trip={v}", dict(p_trip=v)) for v in (0.05, 0.20)] + \
        [(f"h_sd={v}", dict(h_sd=v)) for v in (0.08, 0.16)] + \
        [(f"h_min={v}", dict(h_min=v)) for v in (0.3, 0.5)]
LEVELS = (300.0, 500.0)
REPEATS = 5


def physical(u, rho, p_trip, h_mean, h_sd, h_min):
    z, e, uh, uL = u[:, 0], u[:, 1:6], u[:, 6], u[:, 7]
    trips = (rho * z[:, None] + np.sqrt(1 - rho**2) * e) > norm.ppf(1 - p_trip)
    h = np.clip(h_mean + h_sd * uh, h_min, 1.0)
    L = 0.6 + 0.4 * norm.cdf(uL)
    return trips.astype(float), h, L


def to_X(d):
    lost = d[[f"trip{i}" for i in range(1, 6)]].values * 0.8 * RATED_MW * d.L.values[:, None]
    return np.column_stack([lost, d.h, d.L]) / SCALE


def main():
    rng = np.random.default_rng(5)
    tr = pd.read_csv("rare_design_train.csv")
    ck = pd.read_csv("rare_design_check.csv")
    lo = pd.read_csv("rare_design_lowH.csv")
    lo_fit, lo_val = lo.iloc[8:], lo.iloc[:8]
    fitset = pd.concat([tr, lo_fit])
    gp = fit(to_X(fitset), fitset.rocof_mHz_s.values)
    for name, d in (("check (h >= 0.4)", ck), ("low-inertia hold-out", lo_val)):
        err = gp.predict(to_X(d)) - d.rocof_mHz_s.values
        print(f"surrogate on {name}: RMSE = {np.sqrt(np.mean(err**2)):.2f} mHz/s, max |err| = {np.abs(err).max():.2f}")

    rows = []
    for name, change in CASES:
        prm = {**BASE, **change}

        def g(u, prm=prm):
            t, h, L = physical(u, **prm)
            return gp.predict(features(t, h, L) / SCALE)

        ps = []
        for _ in range(REPEATS):
            levels, _, _ = subset_sim(g, 550.0, rng)
            ps.append([p_at(levels, y) for y in LEVELS])
        ps = np.array(ps)
        row = dict(case=name, **prm)
        for j, y in enumerate(LEVELS):
            m = ps[:, j].mean()
            row[f"P_gt_{int(y)}"] = m
            row[f"cov_{int(y)}"] = ps[:, j].std(ddof=1) / m if m > 0 else np.nan
        rows.append(row)
        print(f"{name:>12}: P(>300) = {row['P_gt_300']:.2e}   P(>500) = {row['P_gt_500']:.2e} (cov {row['cov_500']:.2f})", flush=True)
    pd.DataFrame(rows).to_csv("sensitivity.csv", index=False)


if __name__ == "__main__":
    main()
