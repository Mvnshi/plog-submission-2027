"""Operator runner tests use public files and generated synthetic ATD only."""

import os
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import sys
import unittest

from evaluation import possession_metrics as pm


@unittest.skipUnless(os.name == "nt", "PowerShell competition runner is Windows-specific")
class CompetitionRunnerTests(unittest.TestCase):
    repository = Path(__file__).resolve().parents[1]

    def invoke(self, *arguments, script=None, cwd=None, timeout=120):
        return subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
             str(script or self.repository / "run_competition.ps1"), *arguments],
            cwd=cwd or self.repository, capture_output=True, text=True, timeout=timeout)

    def copied_public_project(self, root):
        """Public files only, no .git or licensed fixture needed; reuse the installed interpreter."""
        manifest = json.loads((self.repository / "release/frozen-runtime.json").read_text())
        sources = set(manifest["source_sha256"]) | set(manifest["artifacts_sha256"]) | {"release/frozen-runtime.json"}
        for source in sources:
            target = root / source
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.repository / source, target)
        runner = (self.repository / "run_competition.ps1").read_text()
        executable = sys.executable.replace("'", "''")
        runner = runner.replace("$pythonPath = Join-Path $projectDirectory '.venv/Scripts/python.exe'",
                                "$pythonPath = '" + executable + "'")
        (root / "run_competition.ps1").write_text(runner)
        return manifest

    def junction(self, link, target):
        result = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)],
                                capture_output=True, text=True)
        if result.returncode:
            self.skipTest("Windows junction creation unavailable on this host")

    def test_preflight_needs_no_competition_directory_and_works_from_other_cwd(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = self.invoke("-PreflightOnly", "-DataDir", "missing inputs [synthetic]",
                                 "-OutputDirectory", "missing outputs [synthetic]", cwd=temporary)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("No competition path was opened", result.stdout)
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_relative_paths_use_caller_directory_and_preserve_existing_output(self):
        with tempfile.TemporaryDirectory(prefix="plog caller synthetic ") as temporary:
            root = Path(temporary)
            (root / "input folder [synthetic]").mkdir()
            output = root / "output folder [synthetic]"
            output.mkdir()
            (output / "plog.csv").write_bytes(b"preserve these synthetic bytes")
            result = self.invoke("-DataDir", "input folder [synthetic]", "-OutputDirectory",
                                 "output folder [synthetic]", cwd=root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Output files already exist", result.stderr)
            self.assertEqual((output / "plog.csv").read_bytes(), b"preserve these synthetic bytes")

    def test_output_directory_cannot_be_an_existing_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "synthetic inputs").mkdir()
            (root / "synthetic output").write_bytes(b"keep this file")
            result = self.invoke("-DataDir", "synthetic inputs", "-OutputDirectory",
                                 "synthetic output", cwd=root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("OutputDirectory names an existing file", result.stderr)
            self.assertEqual((root / "synthetic output").read_bytes(), b"keep this file")

    def test_malformed_metadata_has_fixed_error_and_no_output_or_traceback(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tactical = root / "synthetic inputs" / "tactical"
            tactical.mkdir(parents=True)
            (tactical / "BAR-ATH_tactical_atd.xml").write_text(
                '<Root><Team>SYNTHETIC_CONTENT_MARKER</Root>', encoding="utf-8")
            (tactical / "BAR-ATH_tactical_atd.txt").write_text(
                'SYNTHETIC_RAW_MARKER', encoding="utf-8")
            output = root / "synthetic output"
            result = self.invoke("-DataDir", str(tactical.parent), "-OutputDirectory", str(output))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Tactical ATD metadata is missing or invalid", result.stderr)
            self.assertIn("No predictions were generated", result.stderr)
            combined = result.stdout + result.stderr
            for unexpected in ("Traceback", "ParseError", "SYNTHETIC_CONTENT_MARKER", "SYNTHETIC_RAW_MARKER"):
                self.assertNotIn(unexpected, combined)
            self.assertFalse(output.exists())

    def test_unknown_xml_encoding_has_fixed_error_without_traceback(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tactical = root / "synthetic inputs" / "tactical"
            tactical.mkdir(parents=True)
            (tactical / "BAR-ATH_tactical_atd.xml").write_bytes(
                b'<?xml version="1.0" encoding="SYNTHETIC_UNKNOWN_ENCODING"?><Root/>')
            output = root / "synthetic output"
            result = self.invoke("-DataDir", str(tactical.parent), "-OutputDirectory", str(output))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Tactical ATD metadata is missing or invalid", result.stderr)
            for unexpected in ("Traceback", "LookupError", "SYNTHETIC_UNKNOWN_ENCODING"):
                self.assertNotIn(unexpected, result.stdout + result.stderr)
            self.assertFalse(output.exists())

    def test_unvalidated_challenger_is_rejected_before_execution(self):
        result = self.invoke("-Candidate", "challenger", "-PreflightOnly")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("No separately validated challenger", result.stderr)

    def test_public_checkout_is_not_accepted_as_licensed_data_folder(self):
        result = self.invoke("-DataDir", str(self.repository))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Keep licensed data outside", result.stderr)

    def test_changed_model_or_configuration_fails_before_python_or_data_io(self):
        for tamper in ("configs/selected.json", "models/ball_free_sportec_v1.json"):
            with self.subTest(tamper=tamper), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                shutil.copyfile(self.repository / "run_competition.ps1", root / "run_competition.ps1")
                shutil.copyfile(self.repository / "run_offline.ps1", root / "run_offline.ps1")
                for source in ("configs/selected.json", "models/ball_free_sportec_v1.json"):
                    target = root / source
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(self.repository / source, target)
                with (root / tamper).open("ab") as handle:
                    handle.write(b"\n")
                result = self.invoke("-PreflightOnly", script=root / "run_competition.ps1")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("hash changed", result.stderr)
                self.assertFalse((root / ".venv").exists())

    def test_critical_runtime_source_tamper_fails_without_git_or_licensed_input(self):
        for source in ("plog/inference.py", "run_offline.ps1"):
            with self.subTest(source=source), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                self.copied_public_project(root)
                with (root / source).open("ab") as handle:
                    handle.write(b"\n# Synthetic source drift\n")
                result = self.invoke("-PreflightOnly", script=root / "run_competition.ps1")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("Frozen runtime source hash changed", result.stderr)
                self.assertFalse((root / ".git").exists())

    def test_canonical_lf_source_pins_accept_crlf_checkout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self.copied_public_project(root)
            for source in manifest["source_sha256"]:
                path = root / source
                path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
            result = self.invoke("-PreflightOnly", script=root / "run_competition.ps1")
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_junction_into_public_checkout_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = root / "public-copy"
            project.mkdir()
            self.copied_public_project(project)
            link = root / "private-looking-junction"
            self.junction(link, project)
            try:
                result = self.invoke("-DataDir", str(link), script=project / "run_competition.ps1")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("Keep licensed data outside", result.stderr)
                self.assertIn("junction", result.stderr)
            finally:
                os.rmdir(link)  # Remove only the junction, never recursively traverse its target.

    def test_nonexistent_output_below_junction_into_checkout_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = root / "public-copy"
            project.mkdir()
            self.copied_public_project(project)
            private = root / "synthetic-private-folder"
            private.mkdir()
            link = root / "private-looking-junction"
            self.junction(link, project)
            try:
                output = link / "not-created" / "submission"
                result = self.invoke("-DataDir", str(private), "-OutputDirectory", str(output),
                                     script=project / "run_competition.ps1")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("Keep generated competition files outside", result.stderr)
                self.assertFalse((project / "not-created").exists())
            finally:
                os.rmdir(link)

    def test_full_selected_release_on_synthetic_atd_validates_and_reproduces(self):
        with tempfile.TemporaryDirectory(prefix="plog-competition-synthetic-") as temporary:
            root = Path(temporary)
            tactical = root / "licensed schema [synthetic]" / "tactical"
            tactical.mkdir(parents=True)
            (tactical / "BAR-ATH_tactical_atd.xml").write_text(
                '<Root><Team id="synthetic-home"/><Team id="synthetic-away"/>'
                '<Player teamId="synthetic-home"/><Player teamId="synthetic-away"/></Root>',
                encoding="utf-8")
            with (tactical / "BAR-ATH_tactical_atd.txt").open("w", encoding="utf-8") as handle:
                handle.writelines(f"{frame}:0.501,0.5,2;0.8,0.5,2:0.5,0.5,0\n"
                                  for frame in range(1, 199320))
            output = root / "results [synthetic]"
            result = self.invoke("-DataDir", tactical.parent.name, "-OutputDirectory", output.name,
                                 cwd=root, timeout=150)
            self.assertEqual(result.returncode, 0, result.stderr)
            submitted = output / "plog.csv"
            regenerated = output / "plog.csv.regenerated.csv"
            self.assertEqual(submitted.read_bytes(), regenerated.read_bytes())
            self.assertIsNone(pm.validate_submission(pm.read_label_csv(submitted), 199319))
            self.assertTrue((output / "plog.csv.manifest.json").exists())
            first_bytes = submitted.read_bytes()
            repeated = self.invoke("-DataDir", str(tactical.parent), "-OutputDirectory", str(output))
            self.assertNotEqual(repeated.returncode, 0)
            self.assertIn("Output files already exist", repeated.stderr)
            self.assertEqual(first_bytes, submitted.read_bytes())


if __name__ == "__main__":
    unittest.main()
