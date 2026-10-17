"""
Lum / Sat+ / Sat- response tables of the Color Mixer, read from tables.npz (written by tests/test_data/make_mixer_tables.py from the Lightroom measurements; do not edit the .npz).

Axes: [level,] m, V, S of lookup.py (levels 25, 50, 75, 100; the effect at level 0 is zero).
  LUM_POS_V / LUM_NEG_V   sqrt(V_out / V_in) - 1          LUM_POS_S / LUM_NEG_S   ln(S_out / S_in)
  SAT_POS_V               V_out / V_in - 1                SAT_POS_S               ln(S_out / S_in)
  SAT_NEG_V               V_out / V_in - 1   (S_out follows from the range gain 1 - a)
"""
import os
import numpy as np

_data = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tables.npz"))

LUM_POS_V = _data["LUM_POS_V"].astype(np.float64)
LUM_POS_S = _data["LUM_POS_S"].astype(np.float64)
LUM_NEG_V = _data["LUM_NEG_V"].astype(np.float64)
LUM_NEG_S = _data["LUM_NEG_S"].astype(np.float64)
SAT_POS_V = _data["SAT_POS_V"].astype(np.float64)
SAT_POS_S = _data["SAT_POS_S"].astype(np.float64)
SAT_NEG_V = _data["SAT_NEG_V"].astype(np.float64)

# Lightroom leaves grays alone (checked on the neutral ramp of every export); the least-squares fit leaves ~0.1-0.2 % there, so the S = 0 column is forced to no change.
for _t in (LUM_POS_V, LUM_POS_S, LUM_NEG_V, LUM_NEG_S, SAT_POS_V, SAT_POS_S, SAT_NEG_V):
    _t[..., 0] = 0.0