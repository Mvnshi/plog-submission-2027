"""Tracking schemas and strict loaders. Never print or transmit tracking records.

The tactical ATD is already aligned: its record id is the zero-based video
frame plus one. Ground-truth counter offsets in ``games.py`` do not apply.
Slots identify tracklets, not people. Coordinates remain normalized to the
pitch; speeds supplied by the exporter are in metres per second.
"""

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from games import GAME, GAMES, require


def _positive_count(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


@dataclass
class Tracking:
    """Collective tracking on one timeline, with no assumption of stable identity.

    Array rows are exactly video frames 0..n-1. NaNs mark missing observations,
    including partially missing coordinate pairs. Finite off-pitch observations
    are permitted because an out-of-play ball may be outside the pitch.
    """

    players: np.ndarray
    ball: np.ndarray
    teams: np.ndarray
    player_speed: np.ndarray
    fps: float = 25.0
    pitch: tuple = (105.3, 68.0)

    def __post_init__(self):
        self.players = np.asarray(self.players, dtype=np.float32)
        self.ball = np.asarray(self.ball, dtype=np.float32)
        self.player_speed = np.asarray(self.player_speed, dtype=np.float32)
        raw_teams = np.asarray(self.teams)
        if self.players.ndim != 3 or self.players.shape[-1] != 2:
            raise ValueError("players must have shape (frames, slots, 2)")
        n_frames, n_slots, _ = self.players.shape
        if n_frames < 1 or n_slots < 1:
            raise ValueError("tracking must have at least one frame and one slot")
        if self.ball.shape != (n_frames, 2):
            raise ValueError("ball must have shape (frames, 2)")
        if self.player_speed.shape != (n_frames, n_slots):
            raise ValueError("player_speed must have shape (frames, slots)")
        if raw_teams.shape != (n_slots,):
            raise ValueError("teams must have shape (slots,)")
        if not np.isin(raw_teams, [0, 1]).all():
            raise ValueError("team membership must be 0 (home) or 1 (away)")
        self.teams = raw_teams.astype(np.int8)
        for name in ("players", "ball", "player_speed"):
            if np.isinf(getattr(self, name)).any():
                raise ValueError(f"{name} contains infinity; use NaN for missing observations")
        self.fps = float(self.fps)
        if not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("fps must be finite and positive")
        pitch = np.asarray(self.pitch, dtype=np.float64)
        if pitch.shape != (2,) or not np.isfinite(pitch).all() or (pitch <= 0).any():
            raise ValueError("pitch must contain positive finite length and width")
        self.pitch = tuple(float(value) for value in pitch)

    @property
    def n_frames(self):
        return self.players.shape[0]


def _metadata_teams(meta_path):
    root = ET.parse(meta_path).getroot()
    teams = [element.get("id") for element in root.iter("Team")]
    if len(teams) != 2 or None in teams or len(set(teams)) != 2:
        raise ValueError("ATD metadata must define two distinct team ids")
    slots = [element.get("teamId") for element in root.iter("Player")]
    if not slots:
        raise ValueError("ATD metadata contains no Player track slots")
    if any(team not in teams for team in slots):
        raise ValueError("ATD metadata contains a track slot with unknown team membership")
    # FIFA/EPTS convention, independently documented in DATA.md: first is home.
    return np.asarray([teams.index(team) for team in slots], dtype=np.int8)


def _triple(group, line_number):
    fields = group.strip().split(",")
    if len(fields) != 3 or any(not field.strip() for field in fields):
        raise ValueError(f"ATD line {line_number}: expected x,y,speed triple")
    try:
        values = tuple(float(field) for field in fields)
    except ValueError:
        raise ValueError(f"ATD line {line_number}: invalid numeric field") from None
    if any(np.isinf(value) for value in values):
        raise ValueError(f"ATD line {line_number}: infinity is not a missing observation")
    return values


def load_atd(data_dir=None, expected_frames=199319):
    """Read tactical tracking only; fail on every alignment or schema mismatch.

    This function performs local file I/O only. Run it in the permitted local
    environment when working with restricted competition assets. Errors identify
    the record number without echoing restricted record contents.
    """
    expected_frames = _positive_count(expected_frames, "expected_frames")
    meta_path, raw_path = require(["atd_meta", "atd_raw"], GAME, data_dir, "tactical")
    teams = _metadata_teams(meta_path)
    n_slots = len(teams)
    players = np.empty((expected_frames, n_slots, 2), dtype=np.float32)
    speeds = np.empty((expected_frames, n_slots), dtype=np.float32)
    ball = np.empty((expected_frames, 2), dtype=np.float32)
    count = 0
    with open(raw_path, "r", encoding="utf-8-sig") as handle:
        for count, line in enumerate(handle, start=1):
            if count > expected_frames:
                raise ValueError(f"ATD has more than {expected_frames} records")
            parts = line.strip().split(":")
            if len(parts) != 3:
                raise ValueError(f"ATD line {count}: expected frame:players:ball")
            try:
                frame_id = int(parts[0])
            except ValueError:
                raise ValueError(f"ATD line {count}: frame id is not an integer") from None
            if frame_id != count:
                raise ValueError(f"ATD line {count}: frame id must equal record number (1-based)")
            groups = parts[1].split(";")
            if len(groups) != n_slots:
                raise ValueError(f"ATD line {count}: slot count disagrees with metadata")
            for slot, group in enumerate(groups):
                x, y, speed = _triple(group, count)
                players[count - 1, slot] = (x, y)
                speeds[count - 1, slot] = speed
            x, y, _ = _triple(parts[2], count)
            ball[count - 1] = (x, y)
    if count != expected_frames:
        raise ValueError(f"ATD has {count} records; expected {expected_frames}")
    game = GAMES[GAME]
    return Tracking(players, ball, teams, speeds, game["fps"], game["pitch"])


_NPZ_KEYS = ("players", "ball", "teams", "player_speed", "fps", "pitch")


def load_npz(path):
    """Load the public/external/synthetic interchange format without pickle.

    Labels, if an external corpus stores them in the same archive, are not read
    here. Dataset licensing and match exclusion belong to the experiment manifest.
    """
    with np.load(path, allow_pickle=False) as archive:
        missing = set(_NPZ_KEYS) - set(archive.files)
        if missing:
            raise ValueError("tracking NPZ missing keys: " + ", ".join(sorted(missing)))
        values = {key: archive[key] for key in _NPZ_KEYS}
    if values["fps"].shape != ():
        raise ValueError("tracking NPZ fps must be a scalar")
    return Tracking(**values)


def save_npz(path, tracking):
    """Save normalized tracking locally, with no telemetry or implicit upload."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.savez_compressed(handle, **{key: getattr(tracking, key) for key in _NPZ_KEYS})
    return path
