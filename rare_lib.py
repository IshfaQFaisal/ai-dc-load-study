import numpy as np
from scipy.stats import norm
from grid_lib import run, metrics, DT

SITES = [3, 8, 16, 21, 27]
RATED_MW, IT_SHARE = 250.0, 0.8
P_TRIP, RHO = 0.10, 0.7
H_MEAN, H_SD, H_MIN = 0.8, 0.12, 0.4
L_MIN = 0.6
DIM = 8
T_PRE, T_POST = 1.0, 9.0
TRIP_Z = norm.ppf(1 - P_TRIP)


def physical(u, rho=RHO):
    u = np.atleast_2d(u)
    z, e, uh, uL = u[:, 0], u[:, 1:6], u[:, 6], u[:, 7]
    trips = (rho * z[:, None] + np.sqrt(1 - rho**2) * e) > TRIP_Z
    h = np.clip(H_MEAN + H_SD * uh, H_MIN, 1.0)
    L = L_MIN + (1 - L_MIN) * norm.cdf(uL)
    return trips.astype(float), h, L


def features(trips, h, L):
    lost = trips * IT_SHARE * RATED_MW * L[:, None]
    return np.column_stack([lost, h, L])


def simulate(x):
    trips, h, L = np.asarray(x[:5], float), float(x[5]), float(x[6])
    p_pre = np.full(5, RATED_MW * L)
    p_post = p_pre * (1 - IT_SHARE * trips)
    k_event = int(T_PRE / DT)

    def P_fn(k, f):
        return p_pre if k < k_event else p_post

    m = metrics(run(P_fn, SITES, int((T_PRE + T_POST) / DT), h_scale=h))
    return dict(**{f"trip{j+1}": trips[j] for j in range(5)}, h=h, L=L,
                lost_MW=float(np.sum(p_pre - p_post)),
                rocof_mHz_s=m["rocof_max_mHz_s"], f_dev_mHz=m["f_dev_max_mHz"])
