import numpy as np
import pandas as pd
import andes

andes.config_logger(stream_level=50)

SBASE, F0, DT, T_HOLD = 100.0, 60.0, 0.125, 2.0
TANPHI = np.tan(np.arccos(0.98))


def run(P_fn, buses, n_steps, tau_meas=0.5, h_scale=1.0):
    buses = [buses] if np.isscalar(buses) else list(buses)
    ids = [f"DC{j}" for j in range(len(buses))]
    p0 = np.atleast_1d(P_fn(0, F0)) * np.ones(len(buses)) / SBASE

    ss = andes.load(andes.get_case("ieee39/ieee39_full.xlsx"), setup=False, no_output=True)
    for idx, b, p in zip(ids, buses, p0):
        ss.add("PQ", dict(idx=idx, bus=b, p0=p, q0=p * TANPHI))
    ss.PQ.config.p2p, ss.PQ.config.p2z = 1.0, 0.0
    ss.PQ.config.q2q, ss.PQ.config.q2z = 1.0, 0.0
    ss.setup()
    for i in ss.Toggle.idx.v:
        ss.Toggle.set_status(i, 0)
    if h_scale != 1.0:
        for i, m in zip(ss.GENROU.idx.v, list(ss.GENROU.M.v)):
            ss.GENROU.set(src="M", idx=i, attr="v", value=m * h_scale)
    ss.PFlow.run()
    ss.TDS.config.no_tqdm = 1
    ss.TDS.config.tstep, ss.TDS.config.fixt, ss.TDS.config.shrinkt = 1 / 32, 1, 0

    M = np.array(ss.GENROU.M.v)
    uid = [ss.Bus.idx2uid(b) for b in buses]
    n_hold = int(T_HOLD / DT)
    a = DT / (tau_meas + DT)
    f_meas, rows = F0, []
    for k in range(n_hold + n_steps):
        p = np.atleast_1d(P_fn(max(0, k - n_hold), f_meas)) * np.ones(len(buses)) / SBASE
        for idx, pj in zip(ids, p):
            ss.PQ.set(src="Ppf", idx=idx, attr="v", value=pj)
            ss.PQ.set(src="Qpf", idx=idx, attr="v", value=pj * TANPHI)
        ss.TDS.config.tf = (k + 1) * DT
        ss.TDS.run()
        f = F0 * (M @ np.array(ss.GENROU.omega.v)) / M.sum()
        f_meas += a * (f - f_meas)
        V = ss.Bus.v.v
        rows.append(dict(t=(k + 1 - n_hold) * DT, P_MW=p.sum() * SBASE, f_coi_Hz=f,
                         V_dc_pu=min(V[u] for u in uid), V_min_pu=min(V)))
    df = pd.DataFrame(rows)
    return df[df.t > 0].reset_index(drop=True)


def metrics(df):
    f = df.f_coi_Hz.values
    w = int(0.5 / DT)
    rocof = (f[w:] - f[:-w]) / (w * DT)
    return dict(f_dev_max_mHz=1000 * np.abs(f - F0).max(),
                rocof_max_mHz_s=1000 * np.abs(rocof).max(),
                V_swing_mpu=1000 * (df.V_dc_pu.max() - df.V_dc_pu.min()),
                V_min_all_pu=df.V_min_pu.min())


def ramp_limit(P, R):
    out = np.zeros_like(P); out[0] = P[0]
    for k in range(1, len(P)):
        out[k] = out[k-1] + np.clip(P[k] - out[k-1], -R * DT, R * DT)
    return out


def power_floor(P, frac=0.9, hold=30.0, R=5.0):
    out = np.zeros_like(P); out[0] = P[0]
    ref, t_low, taper = P[0], 0.0, False
    for k in range(1, len(P)):
        if taper:
            out[k] = max(P[k], out[k-1] - R * DT)
            if out[k] <= P[k] + 1e-9:
                taper, ref, t_low = False, P[k], 0.0
        elif P[k] < frac * ref:
            t_low += DT
            if t_low <= hold:
                out[k] = frac * ref
            else:
                taper, out[k] = True, max(P[k], out[k-1] - R * DT)
        else:
            out[k], ref, t_low = P[k], P[k], 0.0
    return out
