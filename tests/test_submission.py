from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from plog.submission import audit_reproduction, sha256_file, write_submission


class SubmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def test_exact_lf_bytes_and_round_trip_audit(self):
        left = write_submission(self.directory / 'left.csv', [' h ', 'a', 'D', 'X'], expected_rows=4)
        right = write_submission(self.directory / 'right.csv', ['H', 'A', 'D', 'X'], expected_rows=4)
        self.assertEqual(left.read_bytes(), b'frame,label\n0,H\n1,A\n2,D\n3,X\n')
        self.assertTrue(audit_reproduction(left, right)['equal'])
        self.assertEqual(audit_reproduction(left, right)['sha256'], sha256_file(left))

    def test_mismatched_bytes_fail_even_if_labels_equivalent(self):
        left = write_submission(self.directory / 'left.csv', ['H'], expected_rows=1)
        right = self.directory / 'right.csv'
        right.write_bytes(left.read_bytes().replace(b'\n', b'\r\n'))
        with self.assertRaisesRegex(ValueError, 'byte-for-byte'):
            audit_reproduction(left, right)

    def test_bad_labels_and_wrong_lengths_never_replace_existing_file(self):
        path = self.directory / 'existing.csv'
        path.write_bytes(b'preserve')
        for labels in ([None], [''], ['Q'], [np.nan], ['H', 'A']):
            with self.subTest(labels=labels):
                with self.assertRaises(ValueError):
                    write_submission(path, labels, expected_rows=1)
                self.assertEqual(path.read_bytes(), b'preserve')

    def test_full_frame_contract_with_official_validator(self):
        # Synthetic labels validate format only; this is never a competition prediction.
        path = write_submission(self.directory / 'synthetic.csv', np.full(199319, 'X'))
        result = subprocess.run([sys.executable, 'evaluation/validate_submission.py', str(path)],
                                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('199319 rows, frame 0..199318', result.stdout)

    def test_rejects_nonpositive_or_noninteger_expected_rows(self):
        for count in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                write_submission(self.directory / 'bad.csv', ['X'], count)


if __name__ == '__main__':
    unittest.main()
