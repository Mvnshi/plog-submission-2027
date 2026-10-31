"""Exact three-state Viterbi with soft transition costs and deterministic tie breaks."""

import math
from numbers import Real

import numpy as np


def decode(log_emissions, team_switch_cost=7.0, dead_switch_cost=5.0, boundary_relief=0.0):
    """Maximize sum emissions minus transition costs; no hard minimum duration.

    A short possession survives when its evidence exceeds two transition costs.
    The experimental boundary relief reduces costs at changes in the final H/A emission
    contrast, including changes supplied by event evidence. Its default is zero; it is not
    an independently learned boundary probability. Head/tail states have no extra penalty.
    Emissions must be finite (missing evidence is a neutral log score, not NaN).
    """
    emissions = np.asarray(log_emissions, dtype=np.float64)
    if emissions.ndim != 2 or emissions.shape[1] != 3 or not np.isfinite(emissions).all():
        raise ValueError("expected finite (n,3) log emissions in H,A,D order")
    parameters = {"team_switch_cost": team_switch_cost, "dead_switch_cost": dead_switch_cost,
                  "boundary_relief": boundary_relief}
    for name, value in parameters.items():
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise ValueError(f"{name} must be a finite numeric scalar")
        try:
            value = float(value)
        except (OverflowError, TypeError, ValueError):
            raise ValueError(f"{name} must be a finite numeric scalar") from None
        if not math.isfinite(value):
            raise ValueError(f"{name} must be a finite numeric scalar")
        parameters[name] = value
    team_switch_cost = parameters["team_switch_cost"]
    dead_switch_cost = parameters["dead_switch_cost"]
    boundary_relief = parameters["boundary_relief"]
    if min(team_switch_cost, dead_switch_cost) < 0 or not 0 <= boundary_relief <= 1:
        raise ValueError("transition costs must be nonnegative and boundary_relief in [0,1]")
    n = len(emissions)
    if not n:
        return np.empty(0, dtype=np.int8)
    costs = np.array([[0, team_switch_cost, dead_switch_cost],
                      [team_switch_cost, 0, dead_switch_cost],
                      [dead_switch_cost, dead_switch_cost, 0]], dtype=np.float64)
    parents = np.zeros((n, 3), dtype=np.int8)
    scores = emissions[0].copy()
    contrast = np.tanh((emissions[:, 0] - emissions[:, 1]) / 2.0)
    for t in range(1, n):
        relief = 1.0 - boundary_relief * min(abs(contrast[t] - contrast[t - 1]) / 2.0, 1.0)
        candidates = scores[:, None] - costs * relief
        parents[t] = np.argmax(candidates, axis=0)
        scores = candidates[parents[t], np.arange(3)] + emissions[t]
        scores -= scores.max()  # stable even on a multi-hour sequence
    states = np.empty(n, dtype=np.int8)
    states[-1] = np.argmax(scores)
    for t in range(n - 1, 0, -1):
        states[t - 1] = parents[t, states[t]]
    return states
