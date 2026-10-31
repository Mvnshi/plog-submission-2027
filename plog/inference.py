"""Local inference from observations; does not import ground truth or evaluation paths."""

import numpy as np

from .config import Config
from .decoder import decode
from .events import event_evidence
from .features import centered_mean, extract


def emissions(tracking, config=None, events=(), model=None):
    """Return uncalibrated H/A/D log scores and an evidence-availability mask.

    Invisible ball gives zero geometry/dead evidence. Independent observed stationary-ball
    and slow-player cues, or observed out-of-pitch positions, may support D. No-ball is
    never itself a stoppage cue. Motion spikes reduce geometry trust.
    """
    cfg = config or Config()
    f = extract(tracking)
    n = tracking.n_frames
    scores = np.zeros((n, 3), dtype=np.float64)
    scores[:, 2] = cfg.dead_bias if cfg.use_dead_ball else -30.0
    distance_home, distance_away = f["distance_home"].copy(), f["distance_away"].copy()
    position_confidence = f["ball_visible"].astype(float)
    if cfg.use_reconstruction:
        from .physics import reconstruct_short_gaps
        filled, position_confidence = reconstruct_short_gaps(
            tracking.ball, tracking.fps, tracking.pitch,
            max_gap_s=cfg.reconstruction_max_gap_s, max_speed_mps=cfg.max_ball_speed_mps)
        imputed = ~f["ball_visible"] & (position_confidence > 0)
        if imputed.any():
            distance = np.linalg.norm((tracking.players[imputed] - filled[imputed, None, :]) *
                                      np.asarray(tracking.pitch), axis=2)
            distance[~np.isfinite(distance)] = np.inf
            for team, destination in ((0, distance_home), (1, distance_away)):
                if np.any(tracking.teams == team):
                    destination[imputed] = np.min(distance[:, tracking.teams == team], axis=1)
    near = np.minimum(distance_home, distance_away)
    reliability = np.exp(-near / cfg.control_radius_m)
    reliability *= position_confidence
    speed = f["ball_speed"]
    impossible = np.isfinite(speed) & (speed > cfg.max_ball_speed_mps)
    # A jump casts doubt on its two endpoints; no interpolation across it.
    if n > 1:
        impossible[:-1] |= impossible[1:].copy()
    reliability[impossible] = 0
    home_d = np.minimum(distance_home, 1000.0)
    away_d = np.minimum(distance_away, 1000.0)
    contrast = np.tanh((away_d - home_d) / cfg.distance_scale_m)
    available = np.zeros(n, dtype=bool)
    if cfg.use_geometry:
        signal = cfg.geometry_weight * reliability * contrast
        scores[:, 0] += signal
        scores[:, 1] -= signal
        available |= reliability > 0
    phase, restart = event_evidence(events, n)
    if cfg.use_events:
        scores[:, 0] += cfg.event_weight * phase
        scores[:, 1] -= cfg.event_weight * phase
        available |= phase != 0
    # Factor team ownership conditional on live play: confidence in which team owns a
    # stationary ball must not, by itself, suppress independently supported stoppages.
    team_max = scores[:, :2].max(axis=1, keepdims=True)
    team_norm = team_max + np.log(np.exp(scores[:, :2] - team_max).sum(axis=1, keepdims=True))
    scores[:, :2] -= team_norm
    if cfg.use_dead_ball:
        # Require current ball and player observations, not merely historical stillness.
        still = np.where(np.isfinite(speed), (speed < cfg.still_ball_mps).astype(float), np.nan)
        slow = np.clip(1 - f["player_speed"] / cfg.slow_players_mps, 0, 1)
        width = int(round(cfg.still_window_s * tracking.fps / 2))
        persistence = centered_mean(still, width)
        stationary = np.nan_to_num(persistence * slow, nan=0.0)
        observed = f["ball_visible"] & np.isfinite(speed) & ~impossible
        stationary *= observed
        scores[:, 2] += cfg.dead_still_weight * stationary
        # Restarts only reinforce an independently observed stationary scene.
        if cfg.use_events:
            scores[:, 2] += cfg.event_weight * restart * stationary
        outside = f["ball_outside"] & ~impossible
        scores[:, 2] += cfg.outside_pitch_weight * outside
        available |= (stationary > 0) | outside
    if model is not None:
        from .learning import ball_free_features
        matrix, names = ball_free_features(tracking)
        if tuple(model.feature_names) != tuple(names):
            raise ValueError("ball-free model feature schema does not match inference")
        probabilities = np.asarray(model.predict_proba(matrix), dtype=np.float64)
        if probabilities.shape != (n, 3) or not np.isfinite(probabilities).all() or (probabilities < 0).any():
            raise ValueError("ball-free model must produce finite (n,3) probabilities")
        if not np.allclose(probabilities.sum(axis=1), 1.0):
            raise ValueError("ball-free probabilities must sum to one")
        # Prefer observed geometry; use the contextual model mainly during uncertainty.
        gate = cfg.ball_free_weight * (1.0 - reliability if cfg.use_geometry else np.ones(n))
        if cfg.factorized_model_fusion:
            alive = probabilities[:, :2].sum(axis=1, keepdims=True)
            team = np.divide(probabilities[:, :2], alive, out=np.full((n, 2), 0.5), where=alive > 0)
            scores[:, :2] += gate[:, None] * np.log(np.clip(team, 1e-6, 1.0))
            normalizer = np.logaddexp(scores[:, 0], scores[:, 1])
            scores[:, :2] -= normalizer[:, None]
            # Who is near the ball informs ownership, not whether a restart has happened.
            # Mix independent liveness energies without gating away a D prediction merely
            # because a player stands next to the ball. These logits remain uncalibrated.
            dead = np.clip(probabilities[:, 2], 1e-6, 1 - 1e-6)
            model_logit = np.log(dead) - np.log1p(-dead)
            scores[:, 2] = (1 - cfg.model_dead_weight) * scores[:, 2] + cfg.model_dead_weight * model_logit
        else:
            scores += gate[:, None] * np.log(np.clip(probabilities, 1e-6, 1.0))
        available |= (f["home_count"] + f["away_count"]) > 0
    if not cfg.use_dead_ball:
        # Ablation disables BOTH heuristic and learned stoppage evidence.
        scores[:, 2] = -30.0
    # A no-evidence frame must favor neither live nor dead. Otherwise a constant live
    # prior would turn long occlusions during a stoppage into invented restarts.
    scores[~available] = 0.0
    return scores, available


def infer(tracking, config=None, events=(), model=None, blocks=None):
    """Return H/A/D/X labels, resetting all temporal evidence at each half-open block.

    No information anywhere in a block => all X. Outside blocks => X. Compute features
    separately per block, so finite differences, smoothing and state cannot span halftime.
    Competition callers supply published half boundaries; external matches default to full input.
    """
    from .data import Tracking
    cfg = config or Config()
    n = tracking.n_frames
    blocks = [(0, n)] if blocks is None else list(blocks)
    labels = np.full(n, "X", dtype="<U1")
    last = 0
    for start, stop in blocks:
        if not (isinstance(start, (int, np.integer)) and isinstance(stop, (int, np.integer))
                and last <= start < stop <= n):
            raise ValueError("blocks must be ordered, nonoverlapping, half-open and in range")
        last = stop
        part = Tracking(players=tracking.players[start:stop], ball=tracking.ball[start:stop],
                        teams=tracking.teams, player_speed=tracking.player_speed[start:stop],
                        fps=tracking.fps, pitch=tracking.pitch)
        local_events = []
        for event in events:
            a, b = int(event["start_frame"]), int(event["end_frame"])
            if a < 0 or b < a or b > n:
                raise ValueError("event interval outside tactical timeline")
            if a < stop and b > start:
                local_events.append({**event, "start_frame": max(a, start) - start,
                                     "end_frame": min(b, stop) - start})
        score, available = emissions(part, cfg, local_events, model)
        if not available.any():
            continue
        states = decode(score, cfg.team_switch_cost, cfg.dead_switch_cost, cfg.boundary_relief) \
            if cfg.use_decoder else np.argmax(score, axis=1)
        labels[start:stop] = np.array(["H", "A", "D"])[states]
    return labels
