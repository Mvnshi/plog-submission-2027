"""Competition input checks use generated synthetic files, never licensed assets."""

from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
import warnings

import numpy as np

from games import GAME, half_video_blocks
from plog.cli import main
from plog.data import load_atd


META = (
    '<Root><Team id="z-home"/><Team id="a-away"/>'
    '<Player id="fragment-9" teamId="a-away"/>'
    '<Player id="fragment-2" teamId="z-home"/>'
    '<Player id="fragment-1" teamId="a-away"/></Root>'
)
ROW = '1:0.1,0.2,1;0.3,0.4,2;0.5,0.6,3:0.7,0.8,4\n'


class ReadinessInputTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="plog-synthetic-input-")
        self.addCleanup(self.temporary.cleanup)
        # Spaces, square brackets and non-ASCII text are ordinary legal Windows paths.
        self.directory = Path(self.temporary.name) / "synthetic inputs [é]"
        tactical = self.directory / "tactical"
        tactical.mkdir(parents=True)
        self.meta = tactical / "BAR-ATH_tactical_atd.xml"
        self.raw = tactical / "BAR-ATH_tactical_atd.txt"
        self.meta.write_text(META, encoding="utf-8")
        self.raw.write_text(ROW, encoding="utf-8")

    def test_team_order_and_track_slot_order_are_independent(self):
        tracking = load_atd(self.directory, expected_frames=1)
        # Home is the first metadata team, not the lexically first ID or slot.
        np.testing.assert_array_equal(tracking.teams, [1, 0, 1])
        np.testing.assert_allclose(tracking.players[0], [[.1, .2], [.3, .4], [.5, .6]])
        np.testing.assert_allclose(tracking.player_speed[0], [1, 2, 3])

    def test_utf8_bom_and_windows_newlines_preserve_frame_zero(self):
        self.raw.write_bytes(b"\xef\xbb\xbf" + ROW.replace("\n", "\r\n").encode("utf-8"))
        tracking = load_atd(self.directory, expected_frames=1)
        np.testing.assert_allclose(tracking.ball[0], [.7, .8])
        self.assertEqual(tracking.n_frames, 1)

    def test_corrected_tracking_counter_is_rejected_as_an_atd_id(self):
        self.raw.write_text(ROW.replace("1:", "1702057:", 1), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "frame id must equal record number"):
            load_atd(self.directory, expected_frames=1)

    def test_empty_input_and_trailing_blank_record_are_not_silently_skipped(self):
        for text, expected in (("", "0 records"), (ROW + "\n", "more than 1 records")):
            with self.subTest(kind=expected):
                self.raw.write_text(text, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, expected):
                    load_atd(self.directory, expected_frames=1)

    def test_infinity_in_every_observation_component_is_rejected(self):
        # Ball speed is unused by inference but still belongs to the input triple.
        for old in ("0.1", "0.2", ",1;", "0.7", "0.8", ",4\n"):
            with self.subTest(component=old):
                replacement = ",inf;" if old == ",1;" else ",inf\n" if old == ",4\n" else "inf"
                self.raw.write_text(ROW.replace(old, replacement, 1), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "infinity"):
                    load_atd(self.directory, expected_frames=1)

    def test_finite_number_overflow_cannot_reach_inference(self):
        self.raw.write_text(ROW.replace("0.1", "1e100", 1), encoding="utf-8")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with self.assertRaisesRegex(ValueError, "contains infinity"):
                load_atd(self.directory, expected_frames=1)

    def test_cli_numeric_failure_never_echoes_record_or_replaces_existing_output(self):
        marker = "SYNTHETIC_RECORD_CONTENT_MUST_STAY_LOCAL"
        self.raw.write_text(ROW.replace("0.1", marker, 1), encoding="utf-8")
        output = self.directory / "existing.csv"
        original = b"synthetic previously preserved bytes\n"
        output.write_bytes(original)
        stderr, stdout = io.StringIO(), io.StringIO()
        with redirect_stderr(stderr), redirect_stdout(stdout):
            code = main(["offline", "--data-dir", str(self.directory), "--out", str(output)])
        self.assertEqual(code, 1)
        self.assertIn("ATD line 1: invalid numeric field", stderr.getvalue())
        self.assertNotIn(marker, stderr.getvalue() + stdout.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())
        self.assertEqual(output.read_bytes(), original)
        self.assertFalse(output.with_suffix(".csv.manifest.json").exists())

    def test_cli_rejects_invalid_encoding_without_creating_output(self):
        self.raw.write_bytes(b"\xffSYNTHETIC_INVALID_ENCODING\n")
        output = self.directory / "never-created.csv"
        stderr = io.StringIO()
        with redirect_stderr(stderr), redirect_stdout(io.StringIO()):
            code = main(["offline", "--data-dir", str(self.directory), "--out", str(output)])
        self.assertEqual(code, 1)
        self.assertNotIn("Traceback", stderr.getvalue())
        self.assertNotIn("SYNTHETIC_INVALID_ENCODING", stderr.getvalue())
        self.assertFalse(output.exists())

    def test_published_half_ranges_include_last_frame_and_exclude_intermission(self):
        actual = [(first, last + 1) for _, first, last in half_video_blocks(GAME)]
        self.assertEqual(actual, [(11185, 90652), (110808, 190198)])
        mask = np.zeros(199319, dtype=bool)
        for first, stop in actual:
            mask[first:stop] = True
        self.assertEqual(int(mask.sum()), 158857)
        for frame in (11185, 90651, 110808, 190197):
            self.assertTrue(mask[frame])
        for frame in (0, 11184, 90652, 110807, 190198, 199318):
            self.assertFalse(mask[frame])


if __name__ == "__main__":
    unittest.main()
