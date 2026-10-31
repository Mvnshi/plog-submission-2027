import json
from pathlib import Path
import tempfile
import unittest
import warnings

import numpy as np

from plog.data import Tracking
from plog.learning import SoftmaxModel, ball_free_features, _finite_quantiles


class LearningTests(unittest.TestCase):
    def tracking(self):
        rng = np.random.default_rng(73)
        return Tracking(rng.random((40, 4, 2)), rng.random((40, 2)), np.array([0, 0, 1, 1]),
                        rng.random((40, 4)))

    def test_ball_free_really_ignores_ball_and_slot_order(self):
        data = self.tracking()
        before, names = ball_free_features(data)
        data.ball[:] = np.nan
        order = [1, 0, 3, 2]
        data.players = data.players[:, order]
        data.player_speed = data.player_speed[:, order]
        after, after_names = ball_free_features(data)
        np.testing.assert_allclose(before, after)
        self.assertEqual(names, after_names)
        self.assertEqual(len(names), before.shape[1])

    def test_missing_population_has_finite_features(self):
        data = self.tracking()
        data.players[:] = np.nan
        data.player_speed[:] = np.nan
        features, _ = ball_free_features(data)
        self.assertTrue(np.isfinite(features).all())
        self.assertEqual(float(abs(features).sum()), 0.0)

    def test_vectorized_quantiles_preserve_linear_nanquantile_values(self):
        rng = np.random.default_rng(10)
        values = rng.normal(size=(11, 7, 2))
        values[rng.random(values.shape) < 0.2] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for quantile, actual in zip((0.1, 0.9), _finite_quantiles(values, (0.1, 0.9))):
                np.testing.assert_allclose(actual, np.nanquantile(values, quantile, axis=1), atol=1e-15)

    def test_fit_learns_separable_classes_and_roundtrips_bytes(self):
        rng = np.random.default_rng(4)
        centers = np.array([[-3, 0], [3, 0], [0, 4]])
        x = np.concatenate([center + rng.normal(0, 0.2, (50, 2)) for center in centers])
        labels = np.repeat(["H", "A", "D"], 50)
        model = SoftmaxModel.fit(x, labels, ("x", "y"), epochs=200)
        probability = model.predict_proba(x)
        np.testing.assert_allclose(probability.sum(axis=1), 1)
        self.assertEqual(float(np.mean(np.array(["H", "A", "D"])[probability.argmax(1)] == labels)), 1.0)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "weights.json"
            model.save(path)
            original = path.read_bytes()
            restored = SoftmaxModel.load(path)
            np.testing.assert_array_equal(probability, restored.predict_proba(x))
            restored.save(path)
            self.assertEqual(original, path.read_bytes())

    def test_fails_closed_on_invalid_training_and_model_values(self):
        with self.assertRaises(ValueError):
            SoftmaxModel.fit(np.zeros((2, 1)), ["H", "X"])
        with self.assertRaises(ValueError):
            SoftmaxModel.fit(np.array([[np.nan]]), ["H"])
        model = SoftmaxModel.fit(np.zeros((2, 1)), ["H", "A"])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "weights.json"
            model.save(path)
            record = json.loads(path.read_text())
            record["scale"] = [0]
            path.write_text(json.dumps(record))
            with self.assertRaises(ValueError):
                SoftmaxModel.load(path)


if __name__ == "__main__":
    unittest.main()
