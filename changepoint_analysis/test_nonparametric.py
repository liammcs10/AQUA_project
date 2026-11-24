'''Self-checks for the reference-free ISI detectors in CP_functions. Run: python test_nonparametric.py'''
import numpy as np
from CP_functions import edivisive_split, library_cp, isi_time_series, edivisive_isi_cp, npid, npid_cp

rng = np.random.default_rng(0)

# fast e-divisive split == ruptures Dynp under CostEnergy (library_cp), CP and score
for _ in range(5):
    x = np.r_[rng.normal(0, 1, 30), rng.normal(rng.uniform(0, 2), 1, 40)]
    t = np.arange(len(x)) + 1.
    bkp, drop = edivisive_split(x)
    CP, score = library_cp(x, t, 1, 'edivisive')
    assert t[bkp - 1] == CP and np.isclose(drop[bkp], score), (bkp, CP)

# time grid: every grid point lies inside the ISI whose value it takes
isi = rng.gamma(4, 50, 30)
t_isi = 100 + np.cumsum(isi)
x, tg = isi_time_series(isi, t_isi, 10.)
i = np.searchsorted(t_isi, tg)
assert np.all((t_isi[i] - isi[i] < tg) & (tg <= t_isi[i])) and np.allclose(x, isi[i])

# planted step found, both modes; NPID finds it and nothing on noise
isi = np.r_[rng.gamma(4, 50, 40), rng.gamma(4, 20, 60)]
t_isi = np.cumsum(isi)
t_cp = t_isi[39]
for mode in ('isi', 'time'):
    assert abs(edivisive_isi_cp(isi, t_isi, mode)[0] - t_cp) < 1000   # ~5 pre-change ISIs
assert abs(npid_cp(isi, t_isi)[0] - t_cp) < 1000   # ~5 pre-change ISIs

x = np.r_[rng.normal(0, 1, 100), rng.normal(2, 1, 100)]
cps = npid(x)[0]
assert len(cps) == 1 and abs(cps[0] - 99) <= 3, cps
assert np.mean([len(npid(rng.normal(0, 1, 200))[0]) > 0 for _ in range(100)]) < 0.1   # ~5% false alarms at 0.9
assert np.isnan(npid_cp(np.array([1., 2.]), np.array([1., 3.]))[0])
print('ok')
