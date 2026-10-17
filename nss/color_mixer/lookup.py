"""
Lookup of the Lum / Sat tables.  A table is sampled on the probe lattice of synthetic_v5.tif, in the pixel's own linear-ProPhoto HSV coordinates:

    V   max channel, 33 nodes (VALS: 28 log-spaced 0.003 .. 0.9, then 0.93 0.96 0.98 0.99 0.995)
    S   (max - min) / max, nodes S_NODES (a virtual node at 0)
    m   position of the middle channel, folded: m = 2 * min(f, 1 - f), f = (mid - min) / (max - min)  (0 at a primary or secondary colour, 1 half way); nodes M_NODES

and read with trilinear interpolation, flat outside the lattice.  `Cell` computes the eight corner indices and weights of a set of pixels once; `Cell.sample` then reads any table of
shape (n_m, n_v, n_s) and `Cell.sample_levels` any stack of shape (n_levels, n_m, n_v, n_s) at a per-pixel slider level.
"""
import numpy as np

VALS = np.concatenate([np.geomspace(0.003, 0.9, 28), [0.93, 0.96, 0.98, 0.99, 0.995]])
S_NODES = np.array([0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
M_NODES = np.array([0.0, 0.083, 0.167, 0.25, 0.333, 0.417, 0.5, 0.583, 0.667, 0.833, 1.0])
LOG_VALS = np.log(VALS)


def fold(f):
    """folded position of the middle channel from f = (mid - min) / (max - min)."""
    return 2.0 * np.minimum(f, 1.0 - f)


def _axis(nodes, x):
    x = np.clip(x, nodes[0], nodes[-1])
    j = np.clip(np.searchsorted(nodes, x, side="right") - 1, 0, len(nodes) - 2)
    return j, (x - nodes[j]) / (nodes[j + 1] - nodes[j])


class Cell:
    def __init__(self, v, s, m):
        jv, av = _axis(LOG_VALS, np.log(np.maximum(v, 1e-9)))
        js, as_ = _axis(S_NODES, s)
        jm, am = _axis(M_NODES, m)
        self.idx, self.w = [], []
        ns, nv = len(S_NODES), len(VALS)
        for dm in (0, 1):
            for dv in (0, 1):
                for ds in (0, 1):
                    self.idx.append(((jm + dm) * nv + (jv + dv)) * ns + (js + ds))
                    self.w.append((am if dm else 1 - am) * (av if dv else 1 - av) * (as_ if ds else 1 - as_))
        self.idx, self.w = np.array(self.idx), np.array(self.w)

    def take(self, mask):
        """the same cell for the subset `mask` of the pixels."""
        c = object.__new__(Cell)
        c.idx, c.w = self.idx[:, mask], self.w[:, mask]
        return c

    def sample(self, table):
        flat = np.asarray(table).reshape(-1)
        return (flat[self.idx] * self.w).sum(axis=0)

    def sample_levels(self, stack, levels, level):
        """stack (n, M, V, S) at ascending slider `levels`, |slider| `level` per pixel: linear between the two neighbouring levels, and between 0 and the first one
        (the effect is 0 at slider 0)."""
        stack = np.asarray(stack)
        n = len(levels)
        levels = np.asarray(levels, np.float64)
        j = np.clip(np.searchsorted(levels, level, side="right") - 1, -1, n - 2)
        lo = levels[np.maximum(j, 0)]
        hi = levels[j + 1]
        base = np.where(j >= 0, lo, 0.0)
        w = np.clip((level - base) / (hi - base), 0.0, 1.0)
        flat = stack.reshape(n, -1)
        top = (flat[j + 1, self.idx] * self.w).sum(axis=0)
        bot = np.where(j >= 0, (flat[np.maximum(j, 0), self.idx] * self.w).sum(axis=0), 0.0)
        return bot + (top - bot) * w
    