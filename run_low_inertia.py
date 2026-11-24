import numpy as np
import pandas as pd
from multiprocessing import Pool
from scipy.stats import qmc
from rare_lib import simulate, L_MIN


def main():
    rng = np.random.default_rng(31)
    ks = [3] * 8 + [4] * 12 + [5] * 20
    hl = qmc.LatinHypercube(d=2, seed=31).random(len(ks))
    X = []
    for k, (a, b) in zip(ks, hl):
        trips = np.zeros(5)
        trips[rng.choice(5, size=k, replace=False)] = 1
        X.append(list(trips) + [0.30 + a * 0.15, L_MIN + b * (1 - L_MIN)])
    with Pool(2) as pool:
        pd.DataFrame(pool.map(simulate, X)).to_csv("rare_design_lowH.csv", index=False)
    print("saved rare_design_lowH.csv", flush=True)


if __name__ == "__main__":
    main()
