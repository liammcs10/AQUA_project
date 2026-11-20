"""
Plotting functions for phase response curves (PRCs).
Works on the long dataframe from PRC_sweep.py (columns incl. f, I_h, Hz, Perturbation time, PRC)
and on the notebook's per-condition dfs (I_h, Hz, Perturbation time, PRC).

    python PRC_plots.py           # plot everything from PRC_sweep.csv
    python PRC_plots.py --check   # self-check of the metrics and plot functions
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

X, Y = "Perturbation time", "PRC"
XLABEL, YLABEL = "Normalised perturbation time", "Phase advance (fraction of period)"


def select(df, **cond):
    """Rows matching every column == value, e.g. select(df, f=0, I_h=150)."""
    return df[np.logical_and.reduce([df[k] == v for k, v in cond.items()])]


def _ax(ax):
    return plt.subplots(figsize=(5, 5))[1] if ax is None else ax


def _colorbar(ax, cmap, vmin, vmax, label):
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin, vmax))
    ax.figure.colorbar(sm, ax=ax, label=label)


# ----------------------------- PRC curves -----------------------------
def plot_prc(df, ax=None, hue="I_h", palette="icefire"):
    """PRC against normalised perturbation time, one line per `hue` value."""
    ax = _ax(ax)
    sns.lineplot(data=df, x=X, y=Y, hue=hue, palette=palette, ax=ax)
    ax.set(xlabel=XLABEL, ylabel=YLABEL)
    return ax


def prc_difference(df_a, df_b, by="I_h"):
    """PRC of b minus a, per `by` value. b is interpolated onto a's perturbation times (they differ slightly)."""
    rows = []
    for v in sorted(set(df_a[by]) & set(df_b[by])):
        a, b = df_a[df_a[by] == v].sort_values(X), df_b[df_b[by] == v].sort_values(X)
        xa = a[X].to_numpy(float)
        rows.append(pd.DataFrame({by: v, X: xa,
                                  "PRC difference": np.interp(xa, b[X].to_numpy(float), b[Y].to_numpy(float)) - a[Y].to_numpy(float)}))
    return pd.concat(rows, ignore_index=True)


def plot_prc_compare(df_a, df_b, labels=("no autapse", "autapse"), hue="I_h", palette="icefire"):
    """1x3: PRC of a, PRC of b, and b - a."""
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    plot_prc(df_a, ax[0], hue, palette)
    plot_prc(df_b, ax[1], hue, palette)
    lims = np.array([ax[0].get_ylim(), ax[1].get_ylim()])
    ax[0].set_ylim(lims[:, 0].min(), lims[:, 1].max())
    ax[1].set_ylim(ax[0].get_ylim())
    ax[0].set_title(labels[0])
    ax[1].set_title(labels[1])
    sns.lineplot(data=prc_difference(df_a, df_b, hue), x=X, y="PRC difference", hue=hue, palette=palette, ax=ax[2])
    ax[2].axhline(0, color="k", lw=0.5)
    ax[2].set(xlabel=XLABEL, title=f"{labels[1]} - {labels[0]}")
    fig.tight_layout()
    return ax


def plot_prc_waterfall(df, I_h, by="f", offset=None, ax=None, cmap="viridis"):
    """PRCs for one I_0, one curve per `by` value, each shifted up by `offset`."""
    ax = _ax(ax)
    d = select(df, I_h=I_h)
    vals = np.sort(d[by].unique())
    if offset is None:
        offset = 1.1 * d.groupby(by)[Y].agg(lambda p: p.max() - p.min()).max()
    colors = plt.get_cmap(cmap)(np.linspace(0, 1, len(vals)))
    for k, (v, c) in enumerate(zip(vals, colors)):
        curve = d[d[by] == v].sort_values(X)
        ax.plot(curve[X], curve[Y] + k * offset, color=c)
    _colorbar(ax, cmap, vals.min(), vals.max(), by)
    ax.set(xlabel=XLABEL, ylabel=f"{YLABEL} (offset)", title=f"I_0 = {I_h} pA")
    return ax


# ----------------------------- scalar metrics -----------------------------
def prc_area(d):
    """Area under one PRC (trapezoid). Not np.sum: N_pert varies with firing period, so sums aren't comparable."""
    d = d.sort_values(X)
    return np.trapezoid(d[Y], d[X])


def prc_mean_time(d):
    """PRC-weighted mean perturbation time. Unstable when the PRC sums to ~0 (e.g. mostly negative PRCs)."""
    return np.average(d[X], weights=d[Y])


XLABELS = {"I_h": "I_0 [pA]", "Hz": "Firing rate [Hz]"}


def plot_metric_vs(df, metric, ylabel, x="I_h", by="f", ax=None, cmap="viridis"):
    """metric(one PRC) against column `x` (I_h or Hz), one line per `by` value. One PRC = one (by, I_h) pair."""
    ax = _ax(ax)
    m = (df.groupby([by, "I_h"])[[X, Y, x]]
         .apply(lambda d: pd.Series({"x": d[x].iloc[0], "m": metric(d)}))
         .reset_index().sort_values("x"))
    vals = np.sort(m[by].unique())
    colors = plt.get_cmap(cmap)(np.linspace(0, 1, len(vals)))
    for v, c in zip(vals, colors):
        line = m[m[by] == v]
        ax.plot(line["x"], line["m"], "o-", color=c, ms=3)
    _colorbar(ax, cmap, vals.min(), vals.max(), by)
    ax.set(xlabel=XLABELS.get(x, x), ylabel=ylabel)
    return ax


AREA, MEAN_T = "Area under PRC", "Mean normalised perturbation time"


def plot_area_vs_I(df, by="f", ax=None):
    return plot_metric_vs(df, prc_area, AREA, "I_h", by, ax)


def plot_mean_time_vs_I(df, by="f", ax=None):
    return plot_metric_vs(df, prc_mean_time, MEAN_T, "I_h", by, ax)


def plot_area_vs_Hz(df, by="f", ax=None):
    return plot_metric_vs(df, prc_area, AREA, "Hz", by, ax)


def plot_mean_time_vs_Hz(df, by="f", ax=None):
    return plot_metric_vs(df, prc_mean_time, MEAN_T, "Hz", by, ax)


# ----------------------------- script -----------------------------
def check():
    t = np.linspace(0, 1, 101)
    flat = pd.DataFrame({X: t, Y: np.ones_like(t)})
    bump = pd.DataFrame({X: t, Y: np.sin(np.pi * t)})
    assert np.isclose(prc_area(flat), 1)
    assert np.isclose(prc_mean_time(bump), 0.5)
    # fake sweep: 3 f values x 2 currents
    df = pd.concat([bump.assign(f=f, I_h=I, Hz=10.0, PRC=bump[Y] * (1 + f / 100) * I / 100)
                    for f in (0, 50, 100) for I in (100, 200)], ignore_index=True)
    diff = prc_difference(select(df, f=0), select(df, f=100))
    assert np.allclose(diff["PRC difference"], np.tile(bump[Y], 2) * np.repeat([1, 2], 101))
    plot_prc(select(df, f=0)); plot_prc_compare(select(df, f=0), select(df, f=100))
    plot_prc_waterfall(df, 100); plot_area_vs_I(df); plot_mean_time_vs_I(df)
    plot_area_vs_Hz(df); plot_mean_time_vs_Hz(df)
    plt.close("all")
    print("check passed")


def main(path="PRC_sweep.csv"):
    df = pd.read_csv(path)
    I0, f_max = sorted(df["I_h"].unique())[0], df["f"].max()
    plot_prc(select(df, f=0)).set_title("no autapse")
    plot_prc_compare(select(df, f=0), select(df, f=f_max), labels=("f = 0", f"f = {f_max}"))
    plot_prc_waterfall(df, I0)
    plot_area_vs_I(df)
    plot_mean_time_vs_I(df)
    plot_area_vs_Hz(df)
    plot_mean_time_vs_Hz(df)
    plt.show()


if __name__ == "__main__":
    check() if "--check" in sys.argv else main()
