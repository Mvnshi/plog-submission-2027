"""Frozen engineering defaults, not tuned against competition labels."""

from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class Config:
    control_radius_m: float = 3.0
    distance_scale_m: float = 1.5
    max_ball_speed_mps: float = 45.0
    still_ball_mps: float = 0.4
    slow_players_mps: float = 0.8
    still_window_s: float = 1.0
    geometry_weight: float = 3.0
    event_weight: float = 0.65
    dead_bias: float = -2.0
    dead_still_weight: float = 4.0
    outside_pitch_weight: float = 4.0
    team_switch_cost: float = 7.0
    dead_switch_cost: float = 5.0
    boundary_relief: float = 0.0
    ball_free_weight: float = 0.65
    model_dead_weight: float = 0.65
    factorized_model_fusion: bool = False
    reconstruction_max_gap_s: float = 0.4
    use_reconstruction: bool = False
    use_geometry: bool = True
    use_dead_ball: bool = True
    use_events: bool = True
    use_decoder: bool = True

    def __post_init__(self):
        flags = {"use_geometry", "use_dead_ball", "use_events", "use_decoder", "use_reconstruction",
                 "factorized_model_fusion"}
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name in flags:
                if type(value) is not bool:
                    raise ValueError(f"{field.name} must be boolean")
            elif isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                raise ValueError(f"{field.name} must be finite numeric")
        for name in ("control_radius_m", "distance_scale_m", "max_ball_speed_mps",
                     "still_ball_mps", "slow_players_mps", "still_window_s", "reconstruction_max_gap_s"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("geometry_weight", "event_weight", "dead_still_weight", "outside_pitch_weight",
                     "team_switch_cost", "dead_switch_cost", "ball_free_weight"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        if not 0 <= self.boundary_relief <= 1:
            raise ValueError("boundary_relief must be in [0,1]")
        if not 0 <= self.model_dead_weight <= 1:
            raise ValueError("model_dead_weight must be in [0,1]")


def load_config(path=None):
    if path is None:
        return Config()
    return Config(**json.loads(Path(path).read_text(encoding="utf-8")))


def config_dict(config):
    return asdict(config)
