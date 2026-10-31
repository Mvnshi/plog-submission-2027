"""Physics candidate behavior on constructed arrays; no football data or labels."""

import unittest

import numpy as np

from plog.physics import reconstruct_short_gaps


class PhysicsTests(unittest.TestCase):
    def test_short_internal_gap_is_linear_with_uncertain_confidence(self):
        ball = np.array([[.1, .2], [np.nan, np.nan], [np.nan, np.nan], [.103, .206]])
        original = ball.copy()
        reconstructed, confidence = reconstruct_short_gaps(ball, 25, (100, 50))
        np.testing.assert_allclose(reconstructed, [[.1, .2], [.101, .202], [.102, .204], [.103, .206]])
        np.testing.assert_array_equal(confidence, [1, .5, .5, 1])
        np.testing.assert_equal(ball, original)
        self.assertFalse(np.shares_memory(reconstructed, ball))

    def test_partial_point_is_missing_whole_pair_in_copy(self):
        ball = np.array([[.1, .2], [.9, np.nan], [.102, .204]])
        original = ball.copy()
        reconstructed, confidence = reconstruct_short_gaps(ball, 25, (100, 50))
        np.testing.assert_allclose(reconstructed[1], [.101, .202])
        np.testing.assert_equal(ball, original)
        self.assertEqual(confidence[1], .5)

    def test_gap_duration_cutoff_and_long_gap_untouched(self):
        for gap_length, filled in ((10, True), (11, False)):
            with self.subTest(gap_length=gap_length):
                ball = np.full((gap_length + 2, 2), np.nan)
                ball[0], ball[-1] = [.2, .3], [.21, .31]
                reconstructed, confidence = reconstruct_short_gaps(ball, 25, (100, 50), max_gap_s=.4)
                self.assertEqual(bool(np.isfinite(reconstructed[1:-1]).all()), filled)
                np.testing.assert_array_equal(confidence[1:-1], .5 if filled else 0)

    def test_endpoint_jump_is_rejected_in_physical_units(self):
        ball = np.array([[0, .5], [np.nan, np.nan], [1, .5]])
        reconstructed, confidence = reconstruct_short_gaps(ball, 25, (105.3, 68))
        np.testing.assert_equal(reconstructed, ball)
        np.testing.assert_array_equal(confidence, [1, 0, 1])

    def test_endpoint_time_includes_both_adjacent_intervals(self):
        # 2m between observations two intervals apart => 25m/s, not 50m/s.
        ball = np.array([[0, .5], [np.nan, np.nan], [.02, .5]])
        reconstructed, confidence = reconstruct_short_gaps(ball, 25, (100, 50), max_speed_mps=30)
        np.testing.assert_allclose(reconstructed[1], [.01, .5])
        self.assertEqual(confidence[1], .5)

    def test_leading_and_trailing_gaps_are_never_extrapolated(self):
        ball = np.array([[np.nan, .1], [np.nan, np.nan], [.3, .4], [.301, .4], [.3, np.nan]])
        reconstructed, confidence = reconstruct_short_gaps(ball, 25, (100, 50))
        np.testing.assert_equal(reconstructed, ball)
        np.testing.assert_array_equal(confidence, [0, 0, 1, 1, 0])

    def test_blocks_must_be_sliced_by_caller(self):
        ball = np.array([[.2, .3], [np.nan, np.nan], [np.nan, np.nan], [.202, .304]])
        # The API cannot infer a half boundary between rows1/2. Slicing at that boundary
        # gives a trailing gap and a leading gap, so neither half invents a bridge.
        first, first_confidence = reconstruct_short_gaps(ball[:2], 25, (100, 50))
        second, second_confidence = reconstruct_short_gaps(ball[2:], 25, (100, 50))
        np.testing.assert_equal(np.concatenate((first, second)), ball)
        np.testing.assert_array_equal(np.r_[first_confidence, second_confidence], [1, 0, 0, 1])

    def test_complete_originals_preserved_including_off_pitch(self):
        ball = np.array([[-.01, 1.02], [.1, .2], [.3, .4]])
        reconstructed, confidence = reconstruct_short_gaps(ball, 25, (105.3, 68))
        np.testing.assert_array_equal(reconstructed, ball)
        np.testing.assert_array_equal(confidence, 1)

    def test_zero_gap_limit_disables_reconstruction(self):
        ball = np.array([[.1, .2], [np.nan, np.nan], [.101, .202]])
        reconstructed, confidence = reconstruct_short_gaps(ball, 25, (100, 50), max_gap_s=0)
        np.testing.assert_equal(reconstructed, ball)
        np.testing.assert_array_equal(confidence, [1, 0, 1])

    def test_empty_and_all_missing_inputs(self):
        reconstructed, confidence = reconstruct_short_gaps(np.empty((0, 2)), 25, (100, 50))
        self.assertEqual(reconstructed.shape, (0, 2))
        self.assertEqual(confidence.shape, (0,))
        ball = np.full((20, 2), np.nan)
        reconstructed, confidence = reconstruct_short_gaps(ball, 25, (100, 50))
        np.testing.assert_equal(reconstructed, ball)
        np.testing.assert_array_equal(confidence, 0)

    def test_invalid_shapes_infinity_pitch_and_scalar_parameters(self):
        for ball in ([], [1, 2], np.zeros((2, 3)), [[0, float("inf")]]):
            with self.subTest(ball=ball):
                with self.assertRaises(ValueError):
                    reconstruct_short_gaps(ball, 25, (100, 50))
        ball = np.zeros((3, 2))
        for pitch in ((0, 50), (100,), (100, float("nan")), (100, float("inf"))):
            with self.subTest(pitch=pitch):
                with self.assertRaises(ValueError):
                    reconstruct_short_gaps(ball, 25, pitch)
        for kwargs in ({"fps": 0}, {"fps": True}, {"fps": float("nan")},
                       {"max_gap_s": -1}, {"max_gap_s": "1"},
                       {"max_speed_mps": 0}, {"max_speed_mps": float("inf")}):
            with self.subTest(kwargs=kwargs):
                parameters = {"fps": 25, "pitch": (100, 50)} | kwargs
                with self.assertRaises(ValueError):
                    reconstruct_short_gaps(ball, **parameters)


if __name__ == "__main__":
    unittest.main()
