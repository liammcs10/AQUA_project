"""
Phase portraits of the four bifurcations of a 2D rest state that lead to
oscillations:

    1. saddle-node (off invariant circle)
    2. saddle-node on invariant circle (SNIC)
    3. subcritical Andronov-Hopf
    4. supercritical Andronov-Hopf

Each column shows the phase plane just before (top) and just after (bottom)
the bifurcation, with equilibria, saddles, limit cycles, flow arrows and the
basin of attraction of the rest state (shaded, boundary dashed).

The saddle-node columns use the persistent sodium plus potassium model
(I_Na,p + I_K, Izhikevich 2007), the Andronov-Hopf columns use the Cartesian
normal forms.
"""

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


# ---------------------------------------------------------------------------
# I_Na,p + I_K MODEL PARAMETERS
# tau = 0.152 -> saddle-node off invariant circle (bistable)
# tau = 1.0   -> saddle-node on invariant circle
# ---------------------------------------------------------------------------
C = 1.0
G_L, E_L = 8.0, -80.0
G_NA, E_NA = 20.0, 60.0
G_K, E_K = 10.0, -90.0
M_HALF, M_K = -20.0, 15.0
N_HALF, N_K = -25.0, 5.0

# ---------------------------------------------------------------------------
# PLOTTING PARAMETERS
# ---------------------------------------------------------------------------
N_QUIVER = 11
N_BASIN = 120
SAVE_PATH = "phase_plots/phase_diagrams.svg"

COL_BASIN = "#cfe3f5"
COL_BOUNDARY = "#1f5a99"
COL_CYCLE = "#c0392b"
COL_FLOW = "0.55"
COL_STREAM = "0.85"
COL_UNSTABLE_MAN = "#7d3c98"


def m_inf(V):
    return 1.0 / (1.0 + np.exp((M_HALF - V) / M_K))


def n_inf(V):
    return 1.0 / (1.0 + np.exp((N_HALF - V) / N_K))


def inapk_rhs(I, tau):
    """Vector field of the I_Na,p + I_K model, state = (V, n)."""

    def rhs(V, n):
        dV = (I - G_L * (V - E_L) - G_NA * m_inf(V) * (V - E_NA)
              - G_K * n * (V - E_K)) / C
        dn = (n_inf(V) - n) / tau
        return dV, dn

    return rhs


def inapk_equilibria(I, V_lo=-89.0, V_hi=40.0):
    """Intersections of the V- and n-nullclines."""

    def g(V):
        n_null = (I - G_L * (V - E_L) - G_NA * m_inf(V) * (V - E_NA)) / (G_K * (V - E_K))
        return n_null - n_inf(V)

    Vs = np.linspace(V_lo, V_hi, 5000)
    roots = [brentq(g, a, b) for a, b in zip(Vs[:-1], Vs[1:]) if g(a) * g(b) < 0]
    return [np.array([V, n_inf(V)]) for V in roots]


def hopf_rhs(mu, sub, omega=1.0):
    """Andronov-Hopf normal form, subcritical includes a stabilising r^5 term."""

    def rhs(x, y):
        r2 = x**2 + y**2
        a = mu + r2 - r2**2 if sub else mu - r2
        return a * x - omega * y, omega * x + a * y

    return rhs


def jacobian(rhs, p, h=1e-6):
    J = np.zeros((2, 2))
    for j in range(2):
        dp = np.zeros(2)
        dp[j] = h
        fp = np.array(rhs(*(p + dp)))
        fm = np.array(rhs(*(p - dp)))
        J[:, j] = (fp - fm) / (2 * h)
    return J


def classify(rhs, p):
    """Return 'stable', 'unstable' or 'saddle' and the Jacobian eigen-decomposition."""
    w, v = np.linalg.eig(jacobian(rhs, p))
    re = w.real
    if re[0] * re[1] < 0:
        kind = "saddle"
    elif np.all(re < 0):
        kind = "stable"
    else:
        kind = "unstable"
    return kind, w, v


def saddle_manifold(rhs, saddle, w, v, bounds, scale, T=60.0, stable=True):
    """Both branches of a saddle's stable (backward integration) or unstable
    (forward integration) manifold."""
    vs = v[:, np.argmin(w.real) if stable else np.argmax(w.real)].real
    direction = -1.0 if stable else 1.0
    branches = []
    for sign in (1, -1):
        p0 = saddle + sign * 1e-4 * vs * scale

        def back(t, s):
            return [direction * d for d in rhs(*s)]

        def leave(t, s):
            (x0, x1), (y0, y1) = bounds
            mx, my = 0.3 * (x1 - x0), 0.3 * (y1 - y0)
            return min(s[0] - (x0 - mx), (x1 + mx) - s[0], s[1] - (y0 - my), (y1 + my) - s[1])

        leave.terminal = True
        sol = solve_ivp(back, [0, T], p0, events=leave, max_step=T / 2000, rtol=1e-8, atol=1e-10)
        branches.append(sol.y)
    return branches


def attractor_cycle(rhs, p0, t_transient, t_keep, max_step):
    """Stable limit cycle obtained by forward integration from p0."""
    sol = solve_ivp(lambda t, s: rhs(*s), [0, t_transient + t_keep], p0,
                    max_step=max_step, rtol=1e-8, atol=1e-10)
    keep = sol.t > t_transient
    return sol.y[:, keep]


def basin_mask(rhs, node, bounds, scale, T, dt):
    """Grid points whose forward trajectory converges to the stable node/focus."""
    (x0, x1), (y0, y1) = bounds
    X, Y = np.meshgrid(np.linspace(x0, x1, N_BASIN), np.linspace(y0, y1, N_BASIN))
    x, y = X.copy(), Y.copy()
    for _ in range(int(T / dt)):
        # RK4 step on the whole grid at once
        k1 = rhs(x, y)
        k2 = rhs(x + 0.5 * dt * k1[0], y + 0.5 * dt * k1[1])
        k3 = rhs(x + 0.5 * dt * k2[0], y + 0.5 * dt * k2[1])
        k4 = rhs(x + dt * k3[0], y + dt * k3[1])
        x = x + dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        y = y + dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        x = np.clip(x, x0 - 10 * (x1 - x0), x1 + 10 * (x1 - x0))
        y = np.clip(y, y0 - 10 * (y1 - y0), y1 + 10 * (y1 - y0))
    dist = np.hypot((x - node[0]) / scale[0], (y - node[1]) / scale[1])
    return X, Y, dist < 1e-2


def draw_flow(ax, rhs, bounds, scale, n_quiver=N_QUIVER):
    (x0, x1), (y0, y1) = bounds
    gx, gy = np.meshgrid(np.linspace(x0, x1, 200), np.linspace(y0, y1, 200))
    u, v = rhs(gx, gy)
    ax.streamplot(gx, gy, u / scale[0], v / scale[1], color=COL_STREAM, density=0.9,
                  linewidth=0.5, arrowstyle="-", zorder=1)

    # sparse, unit-length arrows in axis-scaled units
    qx, qy = np.meshgrid(np.linspace(x0, x1, n_quiver + 2)[1:-1],
                         np.linspace(y0, y1, n_quiver + 2)[1:-1])
    u, v = rhs(qx, qy)
    u, v = u / scale[0], v / scale[1]
    norm = np.hypot(u, v) + 1e-12
    ax.quiver(qx, qy, u / norm * scale[0], v / norm * scale[1], color=COL_FLOW,
              angles="xy", scale_units="xy", scale=n_quiver * 1.3 / (x1 - x0) * scale[0],
              width=0.006, headwidth=4, headlength=4.5, zorder=2)


def draw_equilibrium(ax, p, kind):
    style = dict(markersize=8, markeredgecolor="k", markeredgewidth=1.2, zorder=6)
    if kind == "stable":
        ax.plot(*p, "o", markerfacecolor="k", **style)
    elif kind == "unstable":
        ax.plot(*p, "o", markerfacecolor="w", **style)
    else:
        ax.plot(*p, "o", markerfacecolor="k", fillstyle="left",
                markerfacecoloralt="w", **style)


def draw_panel(ax, spec):
    rhs, bounds = spec["rhs"], spec["bounds"]
    scale = np.array([bounds[0][1] - bounds[0][0], bounds[1][1] - bounds[1][0]])

    eqs = [(p, *classify(rhs, p)) for p in spec["equilibria"]]
    rest = [p for p, kind, _, _ in eqs if kind == "stable"]

    # basin of attraction of the rest state
    if rest:
        X, Y, mask = basin_mask(rhs, rest[0], bounds, scale, spec["basin_T"], spec["basin_dt"])
        ax.contourf(X, Y, mask.astype(float), levels=[0.5, 1.5], colors=[COL_BASIN], zorder=0)

    draw_flow(ax, rhs, bounds, scale, spec.get("quiver", N_QUIVER))

    for p, kind, w, v in eqs:
        if kind == "saddle":
            for br in saddle_manifold(rhs, p, w, v, bounds, scale, T=spec["manifold_T"]):
                ax.plot(br[0], br[1], "--", color=COL_BOUNDARY, lw=2, zorder=4)
            for br in saddle_manifold(rhs, p, w, v, bounds, scale, T=spec["manifold_T"],
                                      stable=False):
                ax.plot(br[0], br[1], "-", color=COL_UNSTABLE_MAN, lw=1.4, zorder=4)

    # an unstable cycle is itself the basin boundary of the rest state
    for cyc, stable in spec.get("cycles", []):
        ax.plot(cyc[0], cyc[1], "-" if stable else "--", color=COL_CYCLE, lw=2.2,
                zorder=5 if stable else 4)

    for p, kind, _, _ in eqs:
        draw_equilibrium(ax, p, kind)

    ax.set_xlim(*bounds[0])
    ax.set_ylim(*bounds[1])
    ax.set_xticks([])
    ax.set_yticks([])
    if "xlabel" in spec:
        ax.set_xlabel(spec["xlabel"], fontsize=10)
        ax.set_ylabel(spec["ylabel"], fontsize=10)

    # magnified view around the node and saddle
    if "inset" in spec:
        loc, zoom_bounds = spec["inset"]
        axin = ax.inset_axes(loc)
        zoom = {k: val for k, val in spec.items() if k not in ("inset", "xlabel", "ylabel")}
        zoom.update(bounds=zoom_bounds, quiver=7)
        draw_panel(axin, zoom)
        for side in axin.spines.values():
            side.set_edgecolor("0.3")
        ax.indicate_inset_zoom(axin, edgecolor="0.3", alpha=0.8)


def circle(r, n=400):
    t = np.linspace(0, 2 * np.pi, n)
    return np.vstack([r * np.cos(t), r * np.sin(t)])


def build_specs():
    """Panel definitions: columns = bifurcations, (before, after)."""
    inapk_kw = dict(xlabel="membrane potential, V", ylabel="K$^+$ activation, n",
                    basin_T=60.0, manifold_T=40.0)
    hopf_bounds = ((-1.4, 1.4), (-1.4, 1.4))
    hopf_kw = dict(xlabel="x", ylabel="y", bounds=hopf_bounds, basin_T=40.0,
                   basin_dt=0.02, manifold_T=20.0)

    columns = []

    # 1. saddle-node off invariant circle: fast K^+ (tau=0.152), bistable
    tau = 0.152
    sn_bounds = ((-72.0, -5.0), (-0.04, 0.72))
    col = []
    for I in (4.0, 5.0):
        rhs = inapk_rhs(I, tau)
        cyc = attractor_cycle(rhs, [-20.0, 0.25], 150.0, 30.0, 0.01)
        spec = dict(rhs=rhs, bounds=sn_bounds, equilibria=inapk_equilibria(I),
                    cycles=[(cyc, True)], basin_dt=0.004, **inapk_kw)
        if I < 4.5:
            spec["inset"] = ([0.04, 0.5, 0.42, 0.46], ((-65.0, -57.0), (-0.0015, 0.0035)))
        col.append(spec)
    columns.append(("Saddle-node", col))

    # 2. saddle-node on invariant circle: slow K^+ (tau=1)
    tau = 1.0
    snic_bounds = ((-82.0, 15.0), (-0.05, 0.75))
    col = []
    for I, has_cycle in ((3.0, False), (5.0, True)):
        rhs = inapk_rhs(I, tau)
        cycles = []
        if has_cycle:
            cycles.append((attractor_cycle(rhs, [-60.0, 0.1], 150.0, 30.0, 0.02), True))
        # before the bifurcation the invariant circle is the saddle's unstable manifold
        spec = dict(rhs=rhs, bounds=snic_bounds, equilibria=inapk_equilibria(I),
                    cycles=cycles, basin_dt=0.01, **inapk_kw)
        if not has_cycle:
            spec["inset"] = ([0.04, 0.5, 0.42, 0.46], ((-67.0, -55.0), (-0.0015, 0.0035)))
        col.append(spec)
    columns.append(("Saddle-node on invariant circle", col))

    # 3. subcritical Andronov-Hopf
    col = []
    for mu in (-0.15, 0.15):
        disc = np.sqrt(1 + 4 * mu)
        cycles = [(circle(np.sqrt((1 + disc) / 2)), True)]
        if mu < 0:
            cycles.append((circle(np.sqrt((1 - disc) / 2)), False))
        col.append(dict(rhs=hopf_rhs(mu, sub=True), equilibria=[np.zeros(2)],
                        cycles=cycles, **hopf_kw))
    columns.append(("Subcritical Andronov-Hopf", col))

    # 4. supercritical Andronov-Hopf
    col = []
    for mu in (-0.3, 0.3):
        cycles = [(circle(np.sqrt(mu)), True)] if mu > 0 else []
        col.append(dict(rhs=hopf_rhs(mu, sub=False), equilibria=[np.zeros(2)],
                        cycles=cycles, **hopf_kw))
    columns.append(("Supercritical Andronov-Hopf", col))

    return columns


def main():
    columns = build_specs()
    fig, axes = plt.subplots(2, 4, figsize=(16, 8.4), constrained_layout=True)

    for j, (title, col) in enumerate(columns):
        for i, spec in enumerate(col):
            draw_panel(axes[i, j], spec)
        axes[0, j].set_title(title, fontsize=12, fontweight="bold")

    for i, label in enumerate(("Before bifurcation", "After bifurcation")):
        axes[i, 0].annotate(label, xy=(-0.22, 0.5), xycoords="axes fraction",
                            rotation=90, va="center", ha="center",
                            fontsize=12, fontweight="bold")

    handles = [
        Line2D([], [], marker="o", ls="", mfc="k", mec="k", ms=8, label="stable equilibrium"),
        Line2D([], [], marker="o", ls="", mfc="w", mec="k", ms=8, label="unstable equilibrium"),
        Line2D([], [], marker="o", ls="", mfc="k", mfcalt="w", fillstyle="left", mec="k",
               ms=8, label="saddle"),
        Line2D([], [], color=COL_CYCLE, lw=2.2, label="stable limit cycle"),
        Line2D([], [], color=COL_CYCLE, lw=2.2, ls="--", label="unstable limit cycle"),
        Line2D([], [], color=COL_BOUNDARY, lw=2, ls="--", label="saddle stable manifold"),
        Line2D([], [], color=COL_UNSTABLE_MAN, lw=1.4, label="saddle unstable manifold"),
        Patch(facecolor=COL_BASIN, label="basin of rest state"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False, fontsize=10)

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    fig.savefig(SAVE_PATH)
    print(f"Saved {SAVE_PATH}")


if __name__ == "__main__":
    main()
