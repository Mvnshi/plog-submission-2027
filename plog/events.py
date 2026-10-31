"""Bounded weak evidence from automatic tactical phases, never event-to-label copying."""

import numpy as np

TEAM_ALIASES = {"barcelona": 0, "atletic club": 1, "athletic club": 1,
                "home": 0, "away": 1}
ATTACK = {"build up", "progression", "creating chances", "attacking transition", "long ball",
          "ball in final third", "ball in the box", "players in the box"}
DEFENSE = {"defensive transition", "defending in defensive third", "defending in middle third",
           "defending in attacking third"}


def event_evidence(events, n_frames, aliases=None):
    """(signed team support, restart-window mask), both indexed on tactical frames.

    Team support is normalized to [-1,1], so overlapping correlated tags cannot accumulate
    unbounded confidence. Set pieces locate an uncertain window, never the exact restart.
    N/A and unknown team names carry no team support. Intervals are half-open.
    """
    aliases = TEAM_ALIASES if aliases is None else {k.strip().casefold(): v for k, v in aliases.items()}
    support = np.zeros(n_frames, dtype=np.float64)
    count = np.zeros(n_frames, dtype=np.float64)
    restarts = np.zeros(n_frames, dtype=np.float64)
    for event in events:
        start, stop = int(event["start_frame"]), int(event["end_frame"])
        if start < 0 or stop < start or stop > n_frames:
            raise ValueError("event interval outside tactical timeline")
        code = (event.get("code") or "").strip().casefold()
        tags = {(k or "").strip().casefold(): (v or "").strip().casefold()
                for k, v in event.get("tags", {}).items()}
        team = aliases.get(tags.get("team", ""))
        if code == "set pieces":
            restarts[start:stop] = 1.0
        if team in (0, 1) and code in ATTACK | DEFENSE:
            sign = (1.0 if team == 0 else -1.0) * (-1.0 if code in DEFENSE else 1.0)
            support[start:stop] += sign
            count[start:stop] += 1
    return np.divide(support, count, out=np.zeros(n_frames), where=count > 0), restarts
