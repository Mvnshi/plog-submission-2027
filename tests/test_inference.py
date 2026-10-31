import itertools
import unittest
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from plog.config import Config, load_config
from plog.data import Tracking
from plog.decoder import decode
from plog.events import event_evidence
from plog.features import extract
from plog.inference import emissions, infer
from plog.synthetic import fixture


class InferenceTests(unittest.TestCase):
    def test_default_config_matches_file(self):
        self.assertEqual(Config(), load_config("configs/default.json"))

    def test_invalid_config(self):
        for kwargs in ({"event_weight": float("nan")}, {"team_switch_cost": -1},
                       {"use_decoder": "false"}, {"control_radius_m": 0}):
            with self.assertRaises(ValueError):
                Config(**kwargs)

    def test_decoder_matches_exhaustive_objective(self):
        rng = np.random.default_rng(10)
        logits = rng.normal(size=(6, 3))
        got = decode(logits, 1.1, .7, 0)
        cost = np.array([[0, 1.1, .7], [1.1, 0, .7], [.7, .7, 0]])
        def score(path):
            return sum(logits[t, s] for t, s in enumerate(path)) - sum(cost[a, b] for a, b in zip(path, path[1:]))
        expected = max(itertools.product(range(3), repeat=6), key=score)
        self.assertAlmostEqual(score(got), score(expected), places=10)

    def test_short_strong_possession_survives(self):
        logits = np.tile([4., -4., -5.], (101, 1))
        logits[48:53] = [-4, 4, -5]
        got = decode(logits)
        np.testing.assert_array_equal(got[48:53], 1)
        self.assertEqual(np.count_nonzero(got == 1), 5)

    def test_isolated_weak_deflection_is_suppressed(self):
        logits = np.tile([1., -1., -5.], (101, 1))
        logits[50] = [-1, 1, -5]
        np.testing.assert_array_equal(decode(logits), 0)

    def test_all_missing_is_unknown_not_dead(self):
        tracking = Tracking(np.full((100, 2, 2), np.nan), np.full((100, 2), np.nan),
                            [0, 1], np.full((100, 2), np.nan))
        scores, available = emissions(tracking)
        self.assertTrue(np.isfinite(scores).all())
        self.assertFalse(available.any())
        np.testing.assert_array_equal(infer(tracking), "X")

    def test_occlusion_does_not_create_dead_segment(self):
        tracking, truth = fixture(corrupt=False)
        tracking.ball[80:260] = np.nan
        pred = infer(tracking)
        np.testing.assert_array_equal(pred[80:260], "H")

    def test_reconstruction_is_optional_geometry_only(self):
        tracking, _ = fixture(corrupt=False)
        tracking.ball[100:105] = np.nan
        ordinary, available = emissions(tracking)
        self.assertFalse(available[100:105].any())
        rebuilt, available = emissions(tracking, replace(Config(), use_reconstruction=True))
        self.assertTrue(available[100:105].all())
        self.assertTrue((rebuilt[100:105, 0] > rebuilt[100:105, 1]).all())
        np.testing.assert_allclose(rebuilt[100:105, 2], Config().dead_bias)
        self.assertTrue(np.isnan(tracking.ball[100:105]).all())

    def test_factorized_model_dead_evidence_independent_of_nearest_team(self):
        from plog.learning import ball_free_features
        tracking, _ = fixture(corrupt=False)
        names = ball_free_features(tracking)[1]
        model = SimpleNamespace(feature_names=names,
                                predict_proba=lambda matrix: np.tile([.08, .02, .90], (len(matrix), 1)))
        ordinary, _ = emissions(tracking, model=model)
        separate, _ = emissions(tracking, replace(Config(), factorized_model_fusion=True), model=model)
        # Strong geometry can locate an owner during preparation without proving live play.
        self.assertLess(ordinary[100, 2], ordinary[100, 0])
        self.assertGreater(separate[100, 2], separate[100, 0])

    def test_model_feature_names_must_match_not_just_width(self):
        from plog.learning import ball_free_features
        tracking, _ = fixture(corrupt=False)
        names = ball_free_features(tracking)[1]
        model = SimpleNamespace(feature_names=names[::-1], predict_proba=lambda matrix: None)
        with self.assertRaisesRegex(ValueError, "schema"):
            emissions(tracking, model=model)

    def test_dead_ablation_disables_learned_as_well_as_heuristic_cues(self):
        from plog.learning import ball_free_features
        tracking, _ = fixture(corrupt=False)
        model = SimpleNamespace(feature_names=ball_free_features(tracking)[1],
                                predict_proba=lambda matrix: np.tile([.001, .001, .998], (len(matrix), 1)))
        config = replace(Config(), use_dead_ball=False, factorized_model_fusion=True)
        scores, available = emissions(tracking, config, model=model)
        np.testing.assert_allclose(scores[available, 2], -30.0)
        self.assertFalse(np.any(infer(tracking, config, model=model) == "D"))

    def test_stationary_scene_supports_dead_ball(self):
        tracking = Tracking(np.tile([[[.501, .5], [.7, .7]]], (100, 1, 1)),
                            np.tile([[.5, .5]], (100, 1)), [0, 1], np.full((100, 2), .01))
        self.assertGreater(np.mean(infer(tracking) == "D"), .95)

    def test_occluded_dead_ball_does_not_invent_restart(self):
        tracking = Tracking(np.tile([[[.501, .5], [.7, .7]]], (200, 1, 1)),
                            np.tile([[.5, .5]], (200, 1)), [0, 1], np.full((200, 2), .01))
        tracking.ball[50:150] = np.nan
        scores, available = emissions(tracking)
        self.assertFalse(available[50:150].any())
        np.testing.assert_array_equal(scores[50:150], 0.0)
        np.testing.assert_array_equal(infer(tracking)[50:150], "D")

    def test_team_slot_permutation_invariant(self):
        tracking, _ = fixture()
        permutation = [3, 0, 2, 1]
        shuffled = Tracking(tracking.players[:, permutation], tracking.ball, tracking.teams[permutation],
                            tracking.player_speed[:, permutation])
        np.testing.assert_array_equal(infer(tracking), infer(shuffled))

    def test_partial_nan_is_not_ball_observation(self):
        tracking, _ = fixture(corrupt=False)
        tracking.ball[5] = [np.nan, .3]
        f = extract(tracking)
        self.assertFalse(f["ball_visible"][5])
        self.assertTrue(np.isinf(f["distance_home"][5]))
        self.assertTrue(np.isnan(f["ball_speed"][6]))

    def test_block_reset_cannot_carry_team_or_motion(self):
        tracking, _ = fixture(corrupt=False)
        tracking.ball[800:] = np.nan
        pred = infer(tracking, blocks=[(0, 300), (800, tracking.n_frames)])
        np.testing.assert_array_equal(pred[300:], "X")

    def test_events_are_bounded_and_na_is_not_home(self):
        event = {"start_frame": 0, "end_frame": 10, "code": "BUILD UP", "tags": {"Team": "Barcelona"}}
        support, restart = event_evidence([event] * 200, 10)
        np.testing.assert_array_equal(support, 1)
        event["tags"]["Team"] = "N/A"
        support, _ = event_evidence([event], 10)
        np.testing.assert_array_equal(support, 0)
        event["tags"]["Team"] = "Barcelona"
        event["code"] = "DEFENSIVE TRANSITION"
        support, _ = event_evidence([event], 10)
        np.testing.assert_array_equal(support, -1)

    def test_invalid_intervals_and_scores_rejected(self):
        tracking, _ = fixture()
        for blocks in ([(10, 5)], [(0, 50), (40, 80)], [(0, 2000)]):
            with self.assertRaises(ValueError):
                infer(tracking, blocks=blocks)
        with self.assertRaises(ValueError):
            decode([[0, np.nan, 1]])
        with self.assertRaises(ValueError):
            event_evidence([{"start_frame": 0, "end_frame": 11}], 10)


if __name__ == "__main__":
    unittest.main()
