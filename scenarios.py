import os
import numpy as np
import pandas as pd
from grid_lib import run, metrics, ramp_limit, power_floor, DT, F0

SCALE, BUS = 10.0, 16
WIN = dict(trip=(2390, 2590), restart=(2630, 2730), ckpt=(620, 700))
K, DB = 8.0, 0.010


def window(prof, col, ev):
    tq = np.arange(*WIN[ev], DT)
    return np.interp(tq, prof.t_s, prof[col]) * SCALE


def replay(P):
    def P_fn(k, f):
        return P[min(k, len(P) - 1)]
    return P_fn


def droop(P, P_ft):
    def P_fn(k, f):
        k = min(k, len(P) - 1)
        d = f - F0
        d = 0.0 if abs(d) < DB else d - np.sign(d) * DB
        alpha = np.clip(1 + K * d, 0.2, 1.1)
        return P[k] + P_ft[k] * (alpha - 1)
    return P_fn


def record(name, ev, P_req, df):
    m = metrics(df)
    diff = df.P_MW.values - P_req[:len(df)]
    m.update(scenario=name, event=ev,
             extra_MWh=diff.clip(min=0).sum() * DT / 3600,
             deferred_MWh=(-diff).clip(min=0).sum() * DT / 3600)
    df.to_csv(f"runs_part1/{name}_{ev}.csv", index=False)
    print(f"{name:>12} {ev:>8}: dev {m['f_dev_max_mHz']:6.1f} mHz  RoCoF {m['rocof_max_mHz_s']:6.1f} mHz/s  "
          f"Vswing {m['V_swing_mpu']:5.1f} mpu  extra {m['extra_MWh']:.2f} MWh  "
          f"deferred {m['deferred_MWh']:.2f} MWh", flush=True)
    return m


def main():
    os.makedirs("runs_part1", exist_ok=True)
    prof = pd.read_csv("ai_load_profile.csv")
    res = []
    for ev in WIN:
        P = window(prof, "P_facility_MW", ev)
        cases = {"baseline": replay(P),
                 "ramp_20MWps": replay(ramp_limit(P, 20)),
                 "ramp_10MWps": replay(ramp_limit(P, 10)),
                 "floor90_30s": replay(power_floor(P)),
                 "ft_droop": droop(P, window(prof, "P_finetune_MW", ev))}
        for name, P_fn in cases.items():
            res.append(record(name, ev, P, run(P_fn, BUS, len(P))))
    pd.DataFrame(res).to_csv("results_part1.csv", index=False)


if __name__ == "__main__":
    main()
