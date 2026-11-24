"""
Phase response curves across autapse parameters, injected currents and perturbation shapes.
Each condition (e, f, tau, I_h, perturbation) is one joblib job; output is one long dataframe.

    python PRC_sweep.py           # run the sweep -> OUT_CSV
    python PRC_sweep.py --check   # self-check: spike mode (SAVE_ALL=False) == full mode
"""
import sys
import itertools
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from phase_response import phase_response, make_perturbation


# ----------------------------- PARAMETERS -----------------------------
RS = {'name': 'RS', 'C': 100, 'k': 0.7, 'v_r': -60, 'v_t': -40, 'v_peak': 35,
      'a': 0.03, 'b': -2, 'c': -50, 'd': 100, 'e': 0.0, 'f': 0.0, 'tau': 0.0}
#FS = {'name': 'FS', 'C': 20, 'k': 1, 'v_r': -55, 'v_t': -40, 'v_peak': 25,
#      'a': 0.2, 'b': -2, 'c': -45, 'd': 0, 'e': 0.0, 'f': 0.0, 'tau': 0.0}

BASE = RS                       # base neuron; e, f, tau are overwritten by the sweep
E_VALUES = np.array([0.2])               # autapse decay rate (1/ms)
F_VALUES = np.arange(0, 401, 10)     # autapse strength (pA jump per spike)
TAU_VALUES = np.array([2.0])              # autapse delay (ms)
I_VALUES = np.array([100, 150, 200, 250]) # step current I_0 (pA)

PERTURBATIONS = [               # kwargs for make_perturbation (H in pA, times in ms)
    dict(shape="square", H=100, T=0.01)
]
'''
    dict(shape="exp", H=20, tau_d=2.0),
    dict(shape="biexp", H=20, tau_r=0.5, tau_d=2.0),
]'''

DT = 0.001                      # ms
N_JOBS = 6
OUT_CSV = "PRC_sweep.csv"
# ----------------------------------------------------------------------


def conditions():
    """All (e, f, tau, I_h, pert) combos; f == 0 means no autapse, so e/tau collapse to one value."""
    seen = set()
    for e, f, tau, I_h, k in itertools.product(E_VALUES, F_VALUES, TAU_VALUES, I_VALUES, range(len(PERTURBATIONS))):
        if f == 0:
            e, tau = 0.0, 0.0
        if (e, f, tau, I_h, k) not in seen:
            seen.add((e, f, tau, I_h, k))
            yield e, f, tau, I_h, PERTURBATIONS[k]


def run_condition(e, f, tau, I_h, pert, SAVE_ALL=False):
    params = {**BASE, 'e': e, 'f': f, 'tau': tau}
    df = phase_response(params, I_h, None, None, perturbation=make_perturbation(dt=DT, **pert),
                        SAVE_ALL=SAVE_ALL, dt=DT)
    cond = {'name': BASE['name'], 'e': e, 'f': f, 'tau': tau,
            'shape': pert['shape'], 'H': pert['H'], 'T': pert.get('T', np.nan),
            'tau_r': pert.get('tau_r', np.nan), 'tau_d': pert.get('tau_d', np.nan)}
    return df.assign(**cond)[list(cond) + ["I_h", "Hz", "Perturbation time", "PRC"]]


def check():
    pert = dict(shape="exp", H=20, tau_d=2.0)
    full = run_condition(0.03, 8.0, 0.5, 250, pert, SAVE_ALL=True)
    spike = run_condition(0.03, 8.0, 0.5, 250, pert, SAVE_ALL=False)
    assert len(full) > 0
    assert np.allclose(full.PRC, spike.PRC, equal_nan=True), np.nanmax(np.abs(full.PRC - spike.PRC))
    print("check passed: spike mode matches full mode")


def main():
    conds = list(conditions())
    print(f"{len(conds)} conditions on {N_JOBS} workers")
    results = Parallel(n_jobs=N_JOBS, verbose=5)(delayed(run_condition)(*c) for c in conds)
    df = pd.concat(results, ignore_index=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"saved {len(df)} rows to {OUT_CSV}")


if __name__ == "__main__":
    check() if "--check" in sys.argv else main()
