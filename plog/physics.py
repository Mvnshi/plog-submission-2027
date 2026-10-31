"""Conservative short-gap ball reconstruction for geometry experiments only.

This is a constant-velocity hypothesis with an uncalibrated confidence cap, not an observed
trajectory or a learned flight model. It cannot model aerial motion, touches or deflections.
Never use reconstructed samples as observations for dead-ball detection. Retain the original
ball array for that detector. Call independently on each half-open temporal block: this
function has no metadata with which to recognize halftime or other timeline boundaries.
"""

import math
from numbers import Real

import numpy as np


def _finite_scalar(value, name, *, allow_zero=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite numeric scalar")
    try:
        value = float(value)
    except (OverflowError, TypeError, ValueError):
        raise ValueError(f"{name} must be a finite numeric scalar") from None
    if not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
        kind = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {kind}")
    return value


def reconstruct_short_gaps(ball, fps, pitch, max_gap_s=.4, max_speed_mps=45):
    """Return ``(ball_copy, confidence)`` with bounded internal linear interpolation.

    Positions have shape (frames, 2) in normalized pitch coordinates; pitch is (length,
    width) in metres. Complete original points retain confidence 1, and unfilled incomplete
    points retain confidence 0. A filled point has confidence .5, a trust cap rather than a
    calibrated probability. The input array and all originally complete rows stay unchanged.

    Any partially missing coordinate pair is an incomplete whole point, matching Tracking's
    visibility definition. When such a row is filled, both coordinates in the returned copy
    follow the line; its original finite component is not treated as a reliable constraint.
    The supplied original array still retains that partial observation unchanged.

    Only consecutive incomplete runs with two adjacent complete endpoints and duration
    ``missing_frames / fps <= max_gap_s`` are eligible. Endpoint displacement divided by
    elapsed endpoint time must be no greater than max_speed_mps. Leading/trailing gaps and
    long or implausible gaps are never extrapolated or bridged. max_gap_s=0 disables filling.
    Off-pitch observations are permitted and never clipped.
    """
    fps = _finite_scalar(fps, "fps")
    max_gap_s = _finite_scalar(max_gap_s, "max_gap_s", allow_zero=True)
    max_speed_mps = _finite_scalar(max_speed_mps, "max_speed_mps")
    pitch = np.asarray(pitch, dtype=np.float64)
    if pitch.shape != (2,) or not np.isfinite(pitch).all() or (pitch <= 0).any():
        raise ValueError("pitch must contain positive finite length and width")
    ball = np.asarray(ball, dtype=np.float64)
    if ball.ndim != 2 or ball.shape[1] != 2 or np.isinf(ball).any():
        raise ValueError("ball must have shape (frames,2), with NaN for missing coordinates")
    reconstructed = ball.copy()
    complete = np.isfinite(ball).all(axis=1)
    confidence = complete.astype(np.float64)
    if len(ball) < 3 or max_gap_s == 0 or complete.all():
        return reconstructed, confidence

    edges = np.diff(np.r_[False, ~complete, False].astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    for start, stop in zip(starts, stops):
        gap_length = int(stop - start)
        if start == 0 or stop == len(ball) or gap_length / fps > max_gap_s:
            continue
        # Runs were extracted from original visibility, so these adjacent endpoints are
        # complete originals. No earlier imputation can become a new endpoint.
        left, right = ball[start - 1], ball[stop]
        elapsed_s = (gap_length + 1) / fps
        displacement_m = (right - left) * pitch
        endpoint_speed = np.linalg.norm(displacement_m) / elapsed_s
        if not np.isfinite(endpoint_speed) or endpoint_speed > max_speed_mps:
            continue
        fraction = np.arange(1, gap_length + 1, dtype=np.float64) / (gap_length + 1)
        reconstructed[start:stop] = left + fraction[:, None] * (right - left)
        confidence[start:stop] = .5
    return reconstructed, confidence
