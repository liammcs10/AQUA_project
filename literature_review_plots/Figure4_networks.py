"""
Figure 4: schematic drawings of randomly connected (liquid-state-machine-like)
networks, one all-excitatory and one all-inhibitory, with synapses between
neurons and autapses (self-connections) on a random subset of neurons.

Run from this folder:  python Figure4_networks.py
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, ArrowStyle
from matplotlib.path import Path

custom_palette = {'Excitatory': '#2678fc', 'Inhibitory': '#f20c4d'}

SAVE_DIR = './plots/Figure 4/'

# --- network parameters ---
N_NEURONS = 12
P_SYNAPSE = 0.18        # connection probability for each ordered pair i -> j
P_AUTAPSE = 0.4         # fraction of neurons carrying an autapse
SEEDS = {'Excitatory': 8, 'Inhibitory': 9}

# --- drawing parameters (data units, network fills roughly a unit disc) ---
NODE_RADIUS = 0.085
MIN_CLEARANCE = 0.16    # minimum distance from a synapse to an unrelated neuron
MAX_OVERLAP = 0.3       # largest fraction of a synapse allowed to run along another
CURVATURES = (0.18, -0.18, 0.32, -0.32)
LOOP_LENGTH = 0.25      # how far the autapse loop extends beyond the soma
LOOP_OPENING = 0.6      # half-angle (rad) between the two ends of the loop
MIN_LOOP_GAP = 0.1      # free space a neuron needs around its autapse loop
LINEWIDTH = 1.3
AXIS_LIMIT = 1.45
FIGSIZE = (3.2, 3.2)


def scatter_positions(n, rng, n_candidates=40):
    """
    Random but evenly spaced positions in the unit disc: each new neuron is the
    candidate lying farthest from the neurons already placed.
    """
    def random_points(k):
        r, theta = np.sqrt(rng.uniform(size=k)), rng.uniform(0, 2 * np.pi, size=k)
        return np.column_stack([r * np.cos(theta), r * np.sin(theta)])

    positions = random_points(1)
    while len(positions) < n:
        candidates = random_points(n_candidates)
        gaps = np.min(np.linalg.norm(candidates[:, None, :] - positions[None, :, :], axis=2), axis=1)
        positions = np.vstack([positions, candidates[np.argmax(gaps)]])
    return positions


def synapse_curve(p_pre, p_post, rad, n_samples=40):
    """Sample the quadratic Bezier that matplotlib's 'arc3' style draws."""
    dx, dy = p_post - p_pre
    control = (p_pre + p_post) / 2 + rad * np.array([dy, -dx])
    t = np.linspace(0, 1, n_samples)[:, None]
    return (1 - t) ** 2 * p_pre + 2 * t * (1 - t) * control + t ** 2 * p_post


def clearance(curve, positions, exclude):
    """Smallest distance from a curve to any neuron not in exclude."""
    others = np.delete(positions, list(exclude), axis=0)
    return np.min(np.linalg.norm(curve[:, None, :] - others[None, :, :], axis=2))


def overlap(curve, other, tolerance=0.07):
    """Fraction of a curve that runs alongside another curve."""
    gaps = np.min(np.linalg.norm(curve[:, None, :] - other[None, :, :], axis=2), axis=1)
    return np.mean(gaps < tolerance)


def random_network(n, p_synapse, p_autapse, rng):
    """
    Returns neuron positions, synapses {(pre, post): curvature} and the list of
    autaptic neurons. Synapses are drawn at random, but a connection is skipped
    if it cannot be routed without passing over an unrelated neuron or running
    on top of another synapse.
    """
    positions = scatter_positions(n, rng)
    synapses = {}

    def curve(pre, post, rad):
        return synapse_curve(positions[pre], positions[post], rad)

    def try_connect(pre, post):
        if (post, pre) in synapses:
            # same curvature bows the reciprocal synapse to the opposite side
            rads = (synapses[(post, pre)],)
        else:
            rads = CURVATURES
        existing = [curve(*pair, rad) for pair, rad in synapses.items()]
        best_rad, best_gap = None, MIN_CLEARANCE
        for rad in rads:
            new = curve(pre, post, rad)
            gap = clearance(new, positions, (pre, post))
            if gap >= best_gap and all(overlap(new, old) < MAX_OVERLAP for old in existing):
                best_rad, best_gap = rad, gap
        if best_rad is None:
            return False
        synapses[(pre, post)] = best_rad
        return True

    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    for k in rng.permutation(len(pairs)):
        if rng.uniform() < p_synapse:
            try_connect(*pairs[k])

    # no neuron is left out of the network
    for i in range(n):
        if not any(i in pair for pair in synapses):
            for j in np.argsort(np.linalg.norm(positions - positions[i], axis=1))[1:]:
                if try_connect(i, int(j)) or try_connect(int(j), i):
                    break

    # autapses go on a random subset of the neurons that have room for a loop
    angles, gaps = zip(*[autapse_angle(i, positions, synapses) for i in range(n)])
    roomy = [i for i in np.argsort(gaps)[::-1] if gaps[i] >= MIN_LOOP_GAP]
    n_autapses = min(int(round(p_autapse * n)), len(roomy))
    chosen = sorted(rng.choice(roomy, size=n_autapses, replace=False).tolist())
    autapses = {i: angles[i] for i in chosen}
    return positions, synapses, autapses


def loop_point(centre, radius, phi):
    return centre + radius * np.array([np.cos(phi), np.sin(phi)])


def autapse_angle(i, positions, synapses):
    """
    Direction for the autapse loop of neuron i with the most free space, and
    the gap between that loop and the nearest synapse or neuron.
    """
    curves = [synapse_curve(positions[pre], positions[post], rad)
              for (pre, post), rad in synapses.items()]
    obstacles = np.vstack(curves + [np.delete(positions, i, axis=0)])

    angles = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    reach = NODE_RADIUS + LOOP_LENGTH
    gaps = []
    for angle in angles:
        # tip and both shoulders of the loop
        probes = np.array([loop_point(positions[i], reach, angle),
                           loop_point(positions[i], 0.8 * reach, angle - LOOP_OPENING),
                           loop_point(positions[i], 0.8 * reach, angle + LOOP_OPENING)])
        gaps.append(np.min(np.linalg.norm(probes[:, None, :] - obstacles[None, :, :], axis=2)))
    best = int(np.argmax(gaps))
    return angles[best], gaps[best]


def autapse_path(centre, angle, end_gap):
    """Teardrop loop leaving the soma and returning to it head-on."""
    control_radius = NODE_RADIUS + 1.6 * LOOP_LENGTH
    vertices = [loop_point(centre, NODE_RADIUS, angle - LOOP_OPENING),
                loop_point(centre, control_radius, angle - LOOP_OPENING),
                loop_point(centre, control_radius, angle + LOOP_OPENING),
                loop_point(centre, NODE_RADIUS + end_gap, angle + LOOP_OPENING)]
    return Path(vertices, [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4])


def terminal_style(neuron_type):
    """Arrowheads for excitatory terminals, flat bars for inhibitory ones."""
    if neuron_type == 'Inhibitory':
        return ArrowStyle('-[', widthB=2.6, lengthB=0)
    return ArrowStyle('-|>', head_length=4.5, head_width=2.2)


def draw_network(neuron_type, ax=None):
    rng = np.random.default_rng(SEEDS[neuron_type])
    colour = custom_palette[neuron_type]
    positions, synapses, autapses = random_network(N_NEURONS, P_SYNAPSE, P_AUTAPSE, rng)

    if ax is None:
        _, ax = plt.subplots(figsize=FIGSIZE)

    somata = [Circle(p, NODE_RADIUS, facecolor=colour, edgecolor='none', zorder=3)
              for p in positions]
    for soma in somata:
        ax.add_patch(soma)

    arrow_kwargs = dict(arrowstyle=terminal_style(neuron_type), color=colour,
                        linewidth=LINEWIDTH, mutation_scale=1, zorder=2,
                        capstyle='round', joinstyle='round')

    for (pre, post), rad in synapses.items():
        ax.add_patch(FancyArrowPatch(positions[pre], positions[post],
                                     connectionstyle=f'arc3,rad={rad}',
                                     patchA=somata[pre], patchB=somata[post],
                                     shrinkA=0, shrinkB=2.5, **arrow_kwargs))

    end_gap = 0.035
    for i, angle in autapses.items():
        path = autapse_path(positions[i], angle, end_gap)
        ax.add_patch(FancyArrowPatch(path=path, **arrow_kwargs))

    ax.set_xlim(-AXIS_LIMIT, AXIS_LIMIT)
    ax.set_ylim(-AXIS_LIMIT, AXIS_LIMIT)
    ax.set_aspect('equal')
    ax.axis('off')
    return ax


if __name__ == '__main__':
    os.makedirs(SAVE_DIR, exist_ok=True)
    for neuron_type in custom_palette:
        ax = draw_network(neuron_type)
        ax.figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
        filename = os.path.join(SAVE_DIR, f'network_{neuron_type.lower()}.svg')
        ax.figure.savefig(filename, transparent=True)
        plt.close(ax.figure)
        print(f'Saved {filename}')
