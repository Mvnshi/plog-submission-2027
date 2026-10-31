import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from plog.data import save_npz
from plog.synthetic import fixture


class CLITests(unittest.TestCase):
    def test_npz_inference_validates_and_reproduces(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            tracking, _ = fixture()
            source = save_npz(root / "synthetic.npz", tracking)
            first, second = root / "first.csv", root / "second.csv"
            base = [sys.executable, "-m", "plog", "infer", "--tracking", str(source)]
            run = subprocess.run(base + ["--out", str(first)], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            run = subprocess.run(base + ["--out", str(second), "--audit-reference", str(first)],
                                 capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            record = json.loads(first.with_suffix(".csv.manifest.json").read_text())
            self.assertTrue(record["official_validation"])
            self.assertEqual(record["rows"], tracking.n_frames)
            run = subprocess.run(base + ["--out", str(first), "--audit-reference", str(first)],
                                 capture_output=True, text=True)
            self.assertEqual(run.returncode, 1)
            self.assertIn("different paths", run.stderr)

    def test_cli_audit_fails_on_mismatch(self):
        with tempfile.TemporaryDirectory() as root:
            a, b = Path(root) / "a.csv", Path(root) / "b.csv"
            a.write_text("frame,label\n0,H\n")
            b.write_text("frame,label\n0,A\n")
            run = subprocess.run([sys.executable, "-m", "plog", "audit", str(a), str(b)],
                                 capture_output=True, text=True)
            self.assertEqual(run.returncode, 1)
            self.assertIn("differ byte-for-byte", run.stderr)


if __name__ == "__main__":
    unittest.main()
