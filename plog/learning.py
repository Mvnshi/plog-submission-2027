"""Small, inspectable ball-free model; no competition labels are consumed here."""

from dataclasses import dataclass
import json
from pathlib import Path
import warnings

import numpy as np


CLASSES = ("H", "A", "D")


def _finite_quantiles(values, quantiles):
    """Vectorised linear nanquantile over slots (axis1), avoiding a Python loop per frame."""
    finite = np.isfinite(values)
    count = finite.sum(axis=1)
    ordered = np.sort(np.where(finite, values, np.inf), axis=1)
    result = []
    for quantile in quantiles:
        position = np.maximum(count - 1, 0) * quantile
        low = np.floor(position).astype(np.int64)
        high = np.ceil(position).astype(np.int64)
        low_value = np.take_along_axis(ordered, np.expand_dims(low, 1), axis=1).squeeze(axis=1)
        high_value = np.take_along_axis(ordered, np.expand_dims(high, 1), axis=1).squeeze(axis=1)
        low_value = np.where(count > 0, low_value, 0)
        high_value = np.where(count > 0, high_value, 0)
        value = low_value + (position - low) * (high_value - low_value)
        result.append(np.where(count > 0, value, np.nan))
    return result


def ball_free_features(tracking):
    """Team-level shape/motion features, independent of ball observations and slot identity.

    Coordinates remain normalised. Speed is in m/s. Missing populations have a zero feature
    vector with explicit coverage flags; this is distinct from claiming stationary players.
    Supplied scalar speeds avoid differentiating fragmented identity slots.
    """
    columns, names = [], []
    for team, side in enumerate(("home", "away")):
        xy = np.asarray(tracking.players[:, tracking.teams == team], dtype=np.float64)
        speed = np.asarray(tracking.player_speed[:, tracking.teams == team], dtype=np.float64)
        valid = np.isfinite(xy).all(axis=2)
        xy = np.where(valid[..., None], xy, np.nan)
        speed = np.where(valid & np.isfinite(speed) & (speed >= 0), speed, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            if xy.shape[1]:
                mean = np.nanmean(xy, axis=1)
                std = np.nanstd(xy, axis=1)
                low, high = _finite_quantiles(xy, (0.1, 0.9))
                speeds = np.column_stack((np.nanmean(speed, axis=1),
                                          _finite_quantiles(speed, (0.9,))[0]))
            else:
                mean = std = low = high = np.full((tracking.n_frames, 2), np.nan)
                speeds = np.full((tracking.n_frames, 2), np.nan)
        # Centroid displacement over one second avoids raw per-slot velocity/identity switches.
        lag = max(1, round(float(tracking.fps)))
        delta = np.full_like(mean, np.nan)
        if tracking.n_frames > lag:
            delta[lag:] = (mean[lag:] - mean[:-lag]) * float(tracking.fps) / lag
        population = valid.sum(axis=1) / 11.0
        values = np.column_stack((population, mean, std, high - low, speeds, delta,
                                  np.isfinite(speed).sum(axis=1) / 11.0))
        columns.append(np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0))
        names.extend(side + "_" + name for name in
                     ("coverage", "mean_x", "mean_y", "std_x", "std_y", "span_x", "span_y",
                      "speed_mean", "speed_q90", "centroid_dx", "centroid_dy", "speed_coverage"))
    return np.concatenate(columns, axis=1), tuple(names)


def _softmax(logits):
    logits = logits - np.max(logits, axis=1, keepdims=True)
    values = np.exp(logits)
    return values / values.sum(axis=1, keepdims=True)


@dataclass
class SoftmaxModel:
    """Regularised multinomial logistic baseline, fixed output order H/A/D.

    Standardisation is fitted ONLY to the training matrix. The model has no hidden state,
    random seed, pretrained checkpoint, or dependency beyond numpy. Full-batch optimisation
    is deterministic on a pinned runtime. Frame labels X are deliberately excluded by caller.
    """
    weights: np.ndarray
    bias: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    feature_names: tuple
    temperature: float = 1.0

    @classmethod
    def fit(cls, matrix, labels, feature_names=None, *, epochs=300, learning_rate=0.1,
            l2=0.001, class_balance=True):
        matrix = np.asarray(matrix, dtype=np.float64)
        labels = np.asarray(labels)
        if matrix.ndim != 2 or not len(matrix) or not np.isfinite(matrix).all():
            raise ValueError("training matrix must be nonempty, finite and two dimensional")
        if labels.shape != (len(matrix),) or not np.isin(labels, CLASSES).all():
            raise ValueError("training labels must align with matrix and contain only H/A/D")
        if epochs < 1 or learning_rate <= 0 or l2 < 0:
            raise ValueError("invalid optimisation parameters")
        if feature_names is None:
            feature_names = tuple("feature_{}".format(i) for i in range(matrix.shape[1]))
        if len(feature_names) != matrix.shape[1] or len(set(feature_names)) != len(feature_names):
            raise ValueError("feature names must uniquely describe matrix columns")
        mean = matrix.mean(axis=0)
        scale = matrix.std(axis=0)
        scale[scale < 1e-8] = 1.0
        x = (matrix - mean) / scale
        y = np.array([CLASSES.index(str(label)) for label in labels], dtype=np.int64)
        count = np.bincount(y, minlength=3)
        sample_weight = np.ones(len(y), dtype=np.float64)
        if class_balance:
            sample_weight = len(y) / (max(1, np.count_nonzero(count)) * np.maximum(count[y], 1))
        sample_weight /= sample_weight.sum()
        weights = np.zeros((x.shape[1], 3), dtype=np.float64)
        bias = np.zeros(3, dtype=np.float64)
        target = np.eye(3)[y]
        for _ in range(epochs):
            residual = (_softmax(x @ weights + bias) - target) * sample_weight[:, None]
            weights -= learning_rate * (x.T @ residual + l2 * weights)
            bias -= learning_rate * residual.sum(axis=0)
        return cls(weights, bias, mean, scale, tuple(feature_names))

    def predict_proba(self, matrix):
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.mean) or not np.isfinite(matrix).all():
            raise ValueError("prediction matrix must be finite and match fitted features")
        if not np.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be positive")
        return _softmax((((matrix - self.mean) / self.scale) @ self.weights + self.bias)
                        / self.temperature)

    def save(self, path):
        """Plain JSON weights: inspectable and byte-stable; no pickle execution."""
        record = {"format": "plog-softmax-v1", "classes": list(CLASSES),
                  "feature_names": list(self.feature_names), "temperature": self.temperature}
        for name in ("weights", "bias", "mean", "scale"):
            record[name] = getattr(self, name).tolist()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, sort_keys=True, indent=2, allow_nan=False) + "\n",
                        encoding="utf-8", newline="\n")
        return path

    @classmethod
    def load(cls, path):
        record = json.loads(Path(path).read_text(encoding="utf-8"))
        if record.get("format") != "plog-softmax-v1" or record.get("classes") != list(CLASSES):
            raise ValueError("unsupported model format or class order")
        arrays = [np.asarray(record[name], dtype=np.float64)
                  for name in ("weights", "bias", "mean", "scale")]
        weights, bias, mean, scale = arrays
        names = tuple(record["feature_names"])
        if (weights.shape != (len(names), 3) or bias.shape != (3,)
                or mean.shape != (len(names),) or scale.shape != (len(names),)
                or any(not np.isfinite(a).all() for a in arrays) or (scale <= 0).any()
                or len(set(names)) != len(names)):
            raise ValueError("invalid model shapes or numeric values")
        temperature = float(record["temperature"])
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be positive")
        return cls(weights, bias, mean, scale, names, temperature)
