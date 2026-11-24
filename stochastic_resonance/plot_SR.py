"""Waterfall plot and heatmap of mean firing rate relative to f = 0 vs noise variance, from a saved SR csv.

Usage:  python plot_SR.py [data/SR_white.csv ...]   (default: every data/SR_*.csv)
Figures go to ./plots/<csv name>.png and ./plots/<csv name>_heatmap.png
"""
import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

PLOT_DIR = './plots/'
OFFSET_FRAC = 0.5    # vertical offset between lines, as a fraction of the largest mean rate


def plot_sr(csv_path, out_path=None):
    df = pd.read_csv(csv_path)
    summ = df.groupby(['f', 'noise_var']).rate_Hz.agg(['mean', 'sem']).fillna(0).reset_index()
    fs = np.sort(summ.f.unique())
    assert fs[0] == 0, f'{csv_path}: no f = 0 baseline to subtract'

    # rate relative to the f = 0 neuron at the same noise variance; SEMs combined in quadrature
    base = summ[summ.f == 0].set_index('noise_var')
    summ['mean'] -= summ.noise_var.map(base['mean'])
    summ['sem'] = np.where(summ.f == 0, 0., np.hypot(summ['sem'], summ.noise_var.map(base['sem'])))
    offset = OFFSET_FRAC * max(summ['mean'].abs().max(), 1e-9)
    norm = plt.Normalize(fs.min(), fs.max())

    fig, ax = plt.subplots(figsize=(7, 10))
    for i, (f, g) in enumerate(summ.groupby('f')):
        c = plt.cm.viridis(norm(f))
        y = g['mean'] + i * offset
        ax.axhline(i * offset, color=c, lw=0.5, ls=':', alpha=0.6)    # zero line: same rate as f = 0
        ax.fill_between(g.noise_var, y - g['sem'], y + g['sem'], color=c, alpha=0.2, lw=0)
        ax.plot(g.noise_var, y, color=c, marker='o', ms=2, lw=1.2)
        if i % 5 == 0:
            ax.text(g.noise_var.max() * 1.01, i * offset, f'{f:g}', color=c, va='center', fontsize=8)

    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='viridis'), ax=ax, label='f (pA)', pad=0.08)
    ax.set_xlabel(r'noise variance (pA$^2$)')
    ax.set_ylabel(f'mean rate - rate(f = 0) (Hz), offset {offset:.2g} Hz per f step')
    ax.set_yticks([])
    r = df.iloc[0]
    title = (f"{r.noise_type} noise, I = threshold - {r.threshold - r.I_dc:g} pA, "
             f"e = {r.e}, tau = {r.tau} ms")
    ax.set_title(title)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    fig.tight_layout()

    if out_path is None:
        os.makedirs(PLOT_DIR, exist_ok=True)
        out_path = os.path.join(PLOT_DIR, os.path.splitext(os.path.basename(csv_path))[0] + '.png')
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    # heatmap of the same relative rates: f vs noise variance, diverging colours centred on 0
    grid = summ.pivot(index='f', columns='noise_var', values='mean')
    fig, ax = plt.subplots(figsize=(7, 6))
    m = ax.pcolormesh(grid.columns, grid.index, grid.values, shading='nearest',
                      cmap='RdBu_r', norm=mcolors.CenteredNorm(0))
    fig.colorbar(m, ax=ax, label='mean rate - rate(f = 0) (Hz)')
    ax.set_xlabel(r'noise variance (pA$^2$)')
    ax.set_ylabel('f (pA)')
    ax.set_title(title)
    fig.tight_layout()
    heat_path = os.path.splitext(out_path)[0] + '_heatmap.png'
    fig.savefig(heat_path, dpi=200)
    plt.close(fig)
    return out_path, heat_path


if __name__ == '__main__':
    for p in sys.argv[1:] or sorted(glob.glob('./data/SR_*.csv')):
        print('saved', *plot_sr(p))
