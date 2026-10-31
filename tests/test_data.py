"""Only synthetic tracking is constructed or read in these tests."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from plog.data import Tracking, load_atd, load_npz, save_npz


META = '<root><Team id="h"/><Team id="a"/><Player teamId="h"/><Player teamId="a"/></root>'
RAW = '1:0.2,0.3,1.0;0.8,0.4,2.0:0.25,0.3,3.0\n2:NaN,NaN,NaN;0.75,0.4,2.0:NaN,0.5,NaN\n'


class TrackingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        tactical = self.directory / 'tactical'
        tactical.mkdir()
        self.meta = tactical / 'BAR-ATH_tactical_atd.xml'
        self.raw = tactical / 'BAR-ATH_tactical_atd.txt'
        self.meta.write_text(META, encoding='utf-8')
        self.raw.write_text(RAW, encoding='utf-8')

    def load(self):
        return load_atd(self.directory, expected_frames=2)

    def test_exact_frame_alignment_and_partial_nan(self):
        tracking = self.load()
        self.assertEqual(tracking.n_frames, 2)
        np.testing.assert_array_equal(tracking.teams, [0, 1])
        np.testing.assert_allclose(tracking.ball[0], [.25, .3])
        self.assertTrue(np.isnan(tracking.ball[1, 0]))
        self.assertAlmostEqual(float(tracking.ball[1, 1]), .5)

    def test_rejects_missing_duplicate_reordered_and_extra_frames(self):
        cases = [RAW.splitlines()[0] + '\n', RAW.replace('2:', '1:'), RAW.replace('1:', '2:', 1), RAW + RAW.splitlines()[1] + '\n']
        for text in cases:
            with self.subTest(text=text):
                self.raw.write_text(text, encoding='utf-8')
                with self.assertRaises(ValueError):
                    self.load()

    def test_rejects_malformed_records_without_skipping(self):
        for text in [RAW.replace('0.2,0.3,1.0;', ''), RAW.replace('0.2,0.3,1.0', '0.2,0.3'), RAW.replace('0.2', 'bad', 1), RAW.replace('0.2', 'inf', 1), RAW.replace('1:', '\n1:', 1), RAW.replace('1:', '1.0:', 1)]:
            with self.subTest(text=text):
                self.raw.write_text(text, encoding='utf-8')
                with self.assertRaises(ValueError):
                    self.load()

    def test_unknown_team_does_not_become_away(self):
        self.meta.write_text(META.replace('Player teamId="a"', 'Player teamId="unknown"'), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'unknown team'):
            self.load()

    def test_duplicate_or_missing_team_metadata_rejected(self):
        for text in [META.replace('Team id="a"', 'Team id="h"'), META.replace('<Team id="a"/>', ''), '<root><Team id="h"/><Team id="a"/></root>']:
            self.meta.write_text(text, encoding='utf-8')
            with self.assertRaises(ValueError):
                self.load()

    def test_npz_roundtrip_without_pickle(self):
        tracking = self.load()
        path = save_npz(self.directory / 'tracking.npz', tracking)
        copy = load_npz(path)
        for key in ('players', 'ball', 'teams', 'player_speed'):
            np.testing.assert_equal(getattr(copy, key), getattr(tracking, key))
        self.assertEqual(copy.fps, tracking.fps)
        self.assertEqual(copy.pitch, tracking.pitch)

    def test_missing_npz_key_rejected(self):
        path = self.directory / 'bad.npz'
        np.savez(path, players=np.zeros((2, 2, 2)))
        with self.assertRaisesRegex(ValueError, 'missing keys'):
            load_npz(path)

    def test_dataclass_rejects_shapes_infinities_and_invalid_teams(self):
        base = dict(players=np.zeros((2, 2, 2)), ball=np.zeros((2, 2)), teams=[0, 1], player_speed=np.zeros((2, 2)))
        for changes in [dict(ball=np.zeros((1, 2))), dict(player_speed=np.zeros((2, 1))), dict(players=np.zeros((2, 2))), dict(teams=[0, .5]), dict(teams=[0, 2]), dict(fps=0), dict(pitch=[105.3, float('inf')]), dict(ball=np.full((2, 2), float('inf')))]:
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    Tracking(**(base | changes))

    def test_finite_outside_pitch_is_allowed(self):
        tracking = self.load()
        tracking.ball[0] = [-.02, 1.01]
        Tracking(tracking.players, tracking.ball, tracking.teams, tracking.player_speed)

    def test_invalid_expected_frame_count(self):
        for count in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                load_atd(self.directory, expected_frames=count)


if __name__ == '__main__':
    unittest.main()
