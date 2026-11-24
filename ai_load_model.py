import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CFG = dict(
    T=3600.0, dt=0.1, seed=1,
    P_peak=700.0, P_idle=80.0, tau_gpu=0.3,
    k_aux=0.35, P_base_per_gpu=60.0,
    N_train=40000, N_ft=10000, n_ft_jobs=20, N_inf=20000, n_inf_groups=200,
    train_start=60.0, U_max=0.97, ripple=0.04, T_iter=1.2,
    ckpt_every=600.0, ckpt_len=25.0, U_ckpt=0.10,
    failures=[(2400.0, 240.0)],
    U_ft=0.90, beta=0.55, mean_high=120.0, mean_low=30.0,
    eval_every=200.0, eval_len=15.0, U_eval=0.05,
    lam0=0.8, lam_amp=0.4, lam_period=1800.0,
    u_q=0.18, q_len=3.0,
    PUE=1.25, tau_cool=300.0,
)


def lag(x, dt, tau):
    y = np.zeros_like(x)
    y[..., 0] = x[..., 0]
    a = dt / (tau + dt)
    for k in range(1, x.shape[-1]):
        y[..., k] = y[..., k - 1] + a * (x[..., k] - y[..., k - 1])
    return y


def gpu_power(U, c):
    return c["P_idle"] + (c["P_peak"] - c["P_idle"]) * np.clip(U, 0, 1)


def training_U(t, c, rng):
    U = np.where(t >= c["train_start"], c["U_max"], 0.0)
    U = U - c["ripple"] * (np.sin(2 * np.pi * t / c["T_iter"]) > 0) * (U > 0)
    tt = t - c["train_start"]
    ck = (tt > 0) & ((tt % c["ckpt_every"]) > c["ckpt_every"] - c["ckpt_len"])
    U = np.where(ck, c["U_ckpt"], U)
    for tf, dur in c["failures"]:
        U = np.where((t >= tf) & (t < tf + dur), 0.0, U)
    return U + rng.normal(0, 0.01, t.size) * (U > 0)


def finetune_U(t, c, rng, n_jobs):
    dt = t[1] - t[0]
    U = np.zeros((n_jobs, t.size))
    for j in range(n_jobs):
        start = rng.uniform(0, 600)
        alpha = np.ones(t.size)
        k, high = 0, True
        while k < t.size:
            dwell = rng.exponential(c["mean_high"] if high else c["mean_low"])
            n = max(1, int(dwell / dt))
            alpha[k:k + n] = 1.0 if high else c["beta"]
            k += n
            high = not high
        u = c["U_ft"] * alpha
        phase = rng.uniform(0, c["eval_every"])
        ev = ((t + phase) % c["eval_every"]) < c["eval_len"]
        u = np.where(ev, c["U_eval"], u)
        U[j] = np.where(t >= start, u, 0.0)
    return U


def inference_U(t, c, rng, n_groups):
    dt = t[1] - t[0]
    lam = c["lam0"] * (1 + c["lam_amp"] * np.sin(2 * np.pi * t / c["lam_period"]))
    arrivals = rng.poisson(lam * dt, size=(n_groups, t.size))
    L = int(c["q_len"] / dt)
    csum = np.cumsum(arrivals, axis=1)
    active = csum - np.concatenate([np.zeros((n_groups, L)), csum[:, :-L]], axis=1)
    return np.minimum(1.0, c["u_q"] * active)


def build(c=CFG):
    rng = np.random.default_rng(c["seed"])
    t = np.arange(0, c["T"], c["dt"])

    P_tr = c["N_train"] * gpu_power(training_U(t, c, rng), c)
    U_ft = finetune_U(t, c, rng, c["n_ft_jobs"])
    P_ft = (c["N_ft"] / c["n_ft_jobs"]) * gpu_power(U_ft, c).sum(0)
    U_in = inference_U(t, c, rng, c["n_inf_groups"])
    P_in = (c["N_inf"] / c["n_inf_groups"]) * gpu_power(U_in, c).sum(0)

    P_gpu = lag(np.vstack([P_tr, P_ft, P_in]), c["dt"], c["tau_gpu"])
    N = c["N_train"] + c["N_ft"] + c["N_inf"]
    P_IT = P_gpu.sum(0) * (1 + c["k_aux"]) + N * c["P_base_per_gpu"]
    P_fac = P_IT + (c["PUE"] - 1) * lag(P_IT, c["dt"], c["tau_cool"])

    df = pd.DataFrame({
        "t_s": t,
        "P_train_MW": P_gpu[0] * (1 + c["k_aux"]) / 1e6,
        "P_finetune_MW": P_gpu[1] * (1 + c["k_aux"]) / 1e6,
        "P_inference_MW": P_gpu[2] * (1 + c["k_aux"]) / 1e6,
        "P_IT_MW": P_IT / 1e6,
        "P_facility_MW": P_fac / 1e6,
    })
    return df


def metrics(p, dt, P_fac_idle):
    dpdt = np.diff(p) / dt
    return {
        "peak_MW": p.max(), "avg_MW": p.mean(), "idle_MW": P_fac_idle,
        "peak/avg": p.max() / p.mean(), "peak/idle": p.max() / P_fac_idle,
        "max_ramp_MW/s": dpdt.max(), "max_decline_MW/s": dpdt.min(),
        "|dP/dt| p99_MW/s": np.percentile(np.abs(dpdt), 99),
        "max_1s_change_MW": np.max(np.abs(p[int(1/dt):] - p[:-int(1/dt)])),
    }


def main(c=CFG):
    df = build(c)
    N = c["N_train"] + c["N_ft"] + c["N_inf"]
    P_fac_idle = N * (c["P_idle"] * (1 + c["k_aux"]) + c["P_base_per_gpu"]) * c["PUE"] / 1e6
    m = metrics(df["P_facility_MW"].values, c["dt"], P_fac_idle)
    for k, v in m.items():
        print(f"{k:>20}: {v:8.2f}")
    df.to_csv("ai_load_profile.csv", index=False)

    fig, ax = plt.subplots(3, 1, figsize=(10, 9), constrained_layout=True)
    for col, lab in [("P_train_MW", "Training"), ("P_finetune_MW", "Fine-tuning"),
                     ("P_inference_MW", "Inference")]:
        ax[0].plot(df.t_s, df[col], lw=0.7, label=lab)
    ax[0].set(title="IT power by phase", ylabel="MW"); ax[0].legend()
    ax[1].plot(df.t_s, df.P_facility_MW, lw=0.7, color="k")
    ax[1].set(title="Facility power (input to ANDES)", ylabel="MW")
    dpdt = np.diff(df.P_facility_MW) / c["dt"]
    ax[2].hist(dpdt, bins=300, log=True)
    ax[2].set(title="dP/dt distribution", xlabel="MW/s", ylabel="count (log)")
    fig.savefig("ai_load_profile.png", dpi=300)
    return df


if __name__ == "__main__":
    main()
