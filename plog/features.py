"""Team aggregate geometry; track slots are never treated as player identities."""

import numpy as np


def centered_mean(values, width):
    """Finite-observation mean without filling gaps or inventing observations."""
    values = np.asarray(values, dtype=np.float64)
    width = max(0, int(width))
    idx = np.arange(len(values))
    lo, hi = np.maximum(0, idx - width), np.minimum(len(values), idx + width + 1)
    valid = np.isfinite(values)
    total = np.r_[0.0, np.cumsum(np.where(valid, values, 0.0))]
    count = np.r_[0, np.cumsum(valid)]
    return np.divide(total[hi] - total[lo], count[hi] - count[lo],
                     out=np.full(len(values), np.nan), where=(count[hi] - count[lo]) > 0)


def extract(tracking):
    """Return frame-indexed evidence in metres, m/s, or explicitly named masks.

    Velocity is only computed between adjacent complete observations. Player movement uses
    supplied speeds and team aggregates, avoiding finite differences across identity switches.
    Partial coordinate observations do not locate an object. NaNs remain missing.
    """
    scale = np.asarray(tracking.pitch, dtype=np.float64)
    ball = tracking.ball.astype(np.float64) * scale
    players = tracking.players.astype(np.float64) * scale
    visible = np.isfinite(ball).all(axis=1)
    player_visible = np.isfinite(players).all(axis=2)
    n = len(ball)
    speed = np.full(n, np.nan)
    if n > 1:
        adjacent = visible[1:] & visible[:-1]
        speed[1:] = np.where(adjacent, np.linalg.norm(np.diff(ball, axis=0), axis=1) * tracking.fps, np.nan)
    acceleration = np.full(n, np.nan)
    if n > 1:
        acceleration[1:] = np.diff(speed) * tracking.fps
    distances = np.linalg.norm(players - ball[:, None, :], axis=2)
    distances[~player_visible | ~visible[:, None]] = np.inf
    out = {"ball_visible": visible, "ball_speed": speed, "ball_acceleration": acceleration,
           "ball_outside": visible & ((ball[:, 0] < 0) | (ball[:, 0] > scale[0]) |
                                      (ball[:, 1] < 0) | (ball[:, 1] > scale[1]))}
    for team, name in ((0, "home"), (1, "away")):
        mask = tracking.teams == team
        valid = player_visible[:, mask]
        points = players[:, mask]
        count = valid.sum(axis=1)
        out[name + "_count"] = count
        if mask.any():
            out["distance_" + name] = np.min(distances[:, mask], axis=1)
            sums = np.where(valid[..., None], points, 0).sum(axis=1)
            center = np.divide(sums, count[:, None], out=np.full((n, 2), np.nan), where=count[:, None] > 0)
            supplied = tracking.player_speed[:, mask]
            ok = valid & np.isfinite(supplied) & (supplied >= 0) & (supplied <= 12.0)
            moving_count = ok.sum(axis=1)
            out[name + "_speed"] = np.divide(np.where(ok, supplied, 0).sum(axis=1), moving_count,
                                               out=np.full(n, np.nan), where=moving_count > 0)
        else:
            out["distance_" + name] = np.full(n, np.inf)
            out[name + "_speed"] = np.full(n, np.nan)
            center = np.full((n, 2), np.nan)
        out[name + "_center_x"], out[name + "_center_y"] = center.T
    both = np.stack([out["home_speed"], out["away_speed"]], axis=1)
    count = np.isfinite(both).sum(axis=1)
    out["player_speed"] = np.divide(np.nansum(both, axis=1), count,
                                     out=np.full(n, np.nan), where=count > 0)
    return out
