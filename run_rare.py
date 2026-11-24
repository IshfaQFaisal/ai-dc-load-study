import numpy as np
import pandas as pd
from multiprocessing import Pool
from scipy.stats import qmc
from rare_lib import simulate, H_MIN, L_MIN


def design(n_per_k, n_zero, seed):
    rng = np.random.default_rng(seed)
    ks = [0] * n_zero + [k for k in range(1, 6) for _ in range(n_per_k)]
    hl = qmc.LatinHypercube(d=2, seed=seed).random(len(ks))
    X = []
    for k, (a, b) in zip(ks, hl):
        trips = np.zeros(5)
        trips[rng.choice(5, size=k, replace=False)] = 1
        X.append(list(trips) + [H_MIN + a * (1 - H_MIN), L_MIN + b * (1 - L_MIN)])
    return X


def main():
    sets = {"train": design(28, 0, 11) + design(0, 6, 12), "check": design(8, 0, 21)}
    with Pool(2) as pool:
        for name, X in sets.items():
            pd.DataFrame(pool.map(simulate, X)).to_csv(f"rare_design_{name}.csv", index=False)
            print(f"saved rare_design_{name}.csv ({len(X)} runs)", flush=True)


if __name__ == "__main__":
    main()
