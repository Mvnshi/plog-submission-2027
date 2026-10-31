"""All downloads, ZIPs, observations, videos and submissions here are generated synthetic fixtures."""

from contextlib import contextmanager, redirect_stderr, redirect_stdout
import errno
import io
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile
import unittest
from unittest import mock
import warnings
import zipfile

from plog import bundle


def metadata(slots=2):
    return ('<Root><Team id="synthetic-home"/><Team id="synthetic-away"/>'
            + ''.join(f'<Player teamId="synthetic-{"home" if i < slots // 2 else "away"}"/>'
                      for i in range(slots)) + '</Root>').encode()


def raw(rows=3, slots=2):
    players = ';'.join('0.501,0.5,2' if i < slots // 2 else '0.8,0.5,2' for i in range(slots))
    return ''.join(f'{i}:{players}:0.5,0.5,0\n' for i in range(1, rows + 1)).encode()


def archive_bytes(members):
    buffer = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_STORED) as archive:
            for name, data in members:
                archive.writestr(name, data)
    return buffer.getvalue()


class BundleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='plog synthetic bundle ')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / 'downloads [é]'
        self.destination = self.root / 'prepared [é]'
        self.source.mkdir()

    def zip(self, name='drive-download-001.zip', members=None):
        path = self.source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(archive_bytes(members if members is not None else [
            ('Drive folder/tactical/' + bundle.REQUIRED[0], metadata()),
            ('Drive folder/tactical/' + bundle.REQUIRED[1], raw()),
        ]))
        return path

    def prepare(self):
        with redirect_stdout(io.StringIO()):
            return bundle.prepare_bundle(self.source, self.destination, expected_frames=3, expected_slots=2)

    def test_multiple_archives_nested_folders_duplicates_and_idempotence(self):
        first = self.zip(members=[('outer/tactical/' + bundle.REQUIRED[0], metadata())])
        second = self.zip('Drive [two]/Initial Challenge - 002.zip', [
            ('wrapped/tactical/' + bundle.REQUIRED[1], raw()),
            ('duplicate/tactical/' + bundle.REQUIRED[0], metadata()),
            ('broadcast/BAR-ATH_broadcast_atd.xml', b'<synthetic-optional/>'),
        ])
        originals = {p: p.read_bytes() for p in (first, second)}
        self.prepare()
        targets = [self.destination / bundle.ASSETS[name] for name in bundle.REQUIRED]
        timestamps = [p.stat().st_mtime_ns for p in targets]
        self.prepare()
        self.assertEqual([p.stat().st_mtime_ns for p in targets], timestamps)
        self.assertEqual(targets[0].read_bytes(), metadata())
        self.assertEqual(targets[1].read_bytes(), raw())
        self.assertEqual({p: p.read_bytes() for p in originals}, originals)
        self.assertTrue((self.destination / 'broadcast/BAR-ATH_broadcast_atd.xml').exists())

    def test_already_extracted_loose_files_are_supported(self):
        directory = self.source / 'nested/tactical'
        directory.mkdir(parents=True)
        (directory / bundle.REQUIRED[0]).write_bytes(metadata())
        (directory / bundle.REQUIRED[1]).write_bytes(raw())
        self.prepare()
        self.assertEqual((self.destination / bundle.ASSETS[bundle.REQUIRED[1]]).read_bytes(), raw())

    def test_conflicting_files_across_archives_stop_before_publication(self):
        self.zip()
        self.zip('other.zip', [('tactical/' + bundle.REQUIRED[1], b'SYNTHETIC_CONFLICT_MARKER')])
        with self.assertRaisesRegex(bundle.BundleError, 'Conflicting copies'):
            self.prepare()
        self.assertFalse((self.destination / 'tactical').exists())

    def test_conflicting_existing_destination_is_preserved_without_partial_publish(self):
        self.zip()
        target = self.destination / 'tactical' / bundle.REQUIRED[0]
        target.parent.mkdir(parents=True)
        target.write_bytes(b'preserve synthetic existing bytes')
        with self.assertRaisesRegex(bundle.BundleError, 'Prepared files conflict'):
            self.prepare()
        self.assertEqual(target.read_bytes(), b'preserve synthetic existing bytes')
        self.assertFalse((target.parent / bundle.REQUIRED[1]).exists())

    def test_duplicate_archive_members_require_exact_equality(self):
        path = 'tactical/' + bundle.REQUIRED[0]
        self.zip(members=[(path, metadata()), (path, metadata()), ('tactical/' + bundle.REQUIRED[1], raw())])
        self.prepare()
        self.assertEqual((self.destination / path).read_bytes(), metadata())

    def test_conflicting_duplicate_archive_members_are_rejected(self):
        self.zip(members=[('same.bin', b'one'), ('SAME.bin', b'two')])
        with self.assertRaisesRegex(bundle.BundleError, 'Conflicting duplicate'):
            self.prepare()
        self.assertFalse((self.destination / 'tactical').exists())

    def test_crc_failure_in_even_an_unused_member_prevents_publication(self):
        marker = b'UNIQUE_SYNTHETIC_CRC_PAYLOAD'
        path = self.zip(members=[('tactical/' + bundle.REQUIRED[0], metadata()),
                                 ('tactical/' + bundle.REQUIRED[1], raw()), ('unused.bin', marker)])
        data = bytearray(path.read_bytes())
        data[data.index(marker)] ^= 1
        path.write_bytes(data)
        with self.assertRaisesRegex(bundle.BundleError, 'ZIP is corrupt'):
            self.prepare()
        self.assertFalse((self.destination / 'tactical').exists())
        self.assertFalse((self.destination / '.plog-preparation/staged/tactical').exists())

    def test_truncated_zip_is_rejected(self):
        path = self.zip()
        path.write_bytes(path.read_bytes()[:-30])
        with self.assertRaisesRegex(bundle.BundleError, 'ZIP is corrupt'):
            self.prepare()

    def test_nested_zip_is_verified_and_organized(self):
        inner = archive_bytes([('tactical/' + bundle.REQUIRED[0], metadata()),
                               ('tactical/' + bundle.REQUIRED[1], raw())])
        self.zip(members=[('Drive folder/Initial Challenge - Data.zip', inner)])
        self.prepare()
        self.assertEqual((self.destination / bundle.ASSETS[bundle.REQUIRED[1]]).read_bytes(), raw())
        self.assertEqual(list((self.destination / '.plog-preparation/nested').iterdir()), [])

    def test_corrupt_nested_archive_is_not_treated_as_an_independent_valid_piece(self):
        self.zip(members=[('nested.zip', b'not a synthetic zip')])
        with self.assertRaisesRegex(bundle.BundleError, 'ZIP is corrupt'):
            self.prepare()

    def test_archive_path_traversal_and_windows_aliases_are_rejected(self):
        cases = ('../outside.bin', '/absolute.bin', r'C:\outside.bin', r'..\outside.bin',
                 'folder/file:stream', 'folder/NUL.txt', 'folder/COM1', 'folder/trailing. /file',
                 'folder/./file', 'folder//file', 'folder/unsafe?.txt', 'folder/COM\u00b9.txt')
        for name in cases:
            with self.subTest(name=name):
                self.zip(members=[(name, b'synthetic')])
                with self.assertRaisesRegex(bundle.BundleError, 'unsafe'):
                    self.prepare()
        self.assertFalse((self.root / 'outside.bin').exists())

    def test_archive_symlinks_and_file_directory_collisions_are_rejected(self):
        link = zipfile.ZipInfo('link')
        link.create_system = 3
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        self.zip(members=[(link, b'../outside')])
        with self.assertRaisesRegex(bundle.BundleError, 'unsafe'):
            self.prepare()
        for members in ([('a', b'file'), ('a/b', b'child')], [('a/b', b'child'), ('A', b'file')]):
            self.zip(members=members)
            with self.assertRaisesRegex(bundle.BundleError, 'collision'):
                self.prepare()

    def test_missing_tactical_metadata_and_broadcast_substitution_fail(self):
        self.zip(members=[('broadcast/BAR-ATH_broadcast_atd.xml', metadata()),
                          ('broadcast/BAR-ATH_broadcast_atd.txt', raw()),
                          ('tactical/' + bundle.REQUIRED[1], raw())])
        with self.assertRaisesRegex(bundle.BundleError, 'Missing required tactical ATD metadata'):
            self.prepare()

    def test_filename_and_source_folder_must_agree(self):
        self.zip(members=[('broadcast/' + bundle.REQUIRED[0], metadata())])
        with self.assertRaisesRegex(bundle.BundleError, 'source folder'):
            self.prepare()

    def test_incomplete_and_complete_loose_videos_are_never_opened_or_renamed(self):
        self.zip()
        names = ('BAR-ATH_tactical-001.mp4', 'BAR-ATH_tactical-002.mp4', 'BAR-ATH_tactical.mp4')
        for name in names:
            (self.source / name).write_bytes(b'synthetic incomplete video, deliberately not MP4')
        original_open = Path.open
        def guarded_open(path, *args, **kwargs):
            if path.suffix.lower() == '.mp4':
                self.fail('Organizer opened a loose video')
            return original_open(path, *args, **kwargs)
        with mock.patch.object(Path, 'open', guarded_open):
            self.prepare()
        self.assertTrue(all((self.source / name).exists() for name in names))
        self.assertFalse((self.destination / 'tactical/BAR-ATH_tactical.mp4').exists())

    def test_insufficient_disk_space_fails_before_extraction(self):
        self.zip()
        with mock.patch.object(bundle.shutil, 'disk_usage', return_value=shutil._ntuple_diskusage(100, 100, 0)):
            with self.assertRaisesRegex(bundle.BundleError, 'Insufficient free disk'):
                self.prepare()
        self.assertFalse(self.destination.exists())

    def test_interrupted_publication_resumes_and_preserves_completed_files(self):
        original = self.zip()
        first_target = self.destination / bundle.ASSETS[bundle.REQUIRED[1]]
        installer = bundle._install
        def interrupt(opener, target, parts, size):
            installer(opener, target, parts, size)
            if target == first_target:
                raise KeyboardInterrupt
        with mock.patch.object(bundle, '_install', side_effect=interrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.prepare()
        prior_time = first_target.stat().st_mtime_ns
        stale = self.destination / '.plog-preparation/parts' / ('a' * 32 + '.part')
        stale.write_bytes(b'generated synthetic partial copy')
        self.prepare()
        self.assertFalse(stale.exists())
        self.assertEqual(first_target.stat().st_mtime_ns, prior_time)
        self.assertTrue(original.exists())
        self.assertEqual((self.destination / bundle.ASSETS[bundle.REQUIRED[0]]).read_bytes(), metadata())

    def test_disk_full_during_write_cleans_partial_and_rerun_recovers(self):
        self.zip()
        original_open = Path.open
        @contextmanager
        def failing_write(path, *args, **kwargs):
            with original_open(path, *args, **kwargs) as handle:
                broken = mock.Mock(wraps=handle)
                broken.write.side_effect = OSError(errno.ENOSPC, 'SYNTHETIC_PRIVATE_DISK_MESSAGE')
                yield broken
        def patched_open(path, *args, **kwargs):
            if path.suffix == '.part' and args and args[0] == 'xb':
                return failing_write(path, *args, **kwargs)
            return original_open(path, *args, **kwargs)
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(Path, 'open', patched_open), redirect_stdout(out), redirect_stderr(err):
            code = bundle.main(['--source', str(self.source), '--destination', str(self.destination)])
        self.assertEqual(code, 1)
        self.assertNotIn('SYNTHETIC_PRIVATE_DISK_MESSAGE', err.getvalue())
        self.assertEqual(list((self.destination / '.plog-preparation/parts').iterdir()), [])
        self.assertFalse((self.destination / 'tactical').exists())
        self.prepare()

    def test_source_and_destination_directory_links_are_rejected(self):
        target = self.root / 'ordinary target'
        target.mkdir()
        link = self.source / 'linked folder'
        if os.name == 'nt':
            result = subprocess.run(['cmd', '/c', 'mklink', '/J', str(link), str(target)], capture_output=True)
            if result.returncode:
                self.skipTest('Junction creation unavailable')
        else:
            link.symlink_to(target, target_is_directory=True)
        try:
            with self.assertRaisesRegex(bundle.BundleError, 'linked or reparse'):
                self.prepare()
            with self.assertRaisesRegex(bundle.BundleError, 'linked or reparse'):
                bundle.prepare_bundle(target, link / 'new prepared folder')
        finally:
            if os.name == 'nt':
                os.rmdir(link)  # Remove only this generated junction, never its target.
            else:
                link.unlink()

    def test_invalid_frame_coverage_and_slot_count_do_not_publish(self):
        self.zip(members=[('tactical/' + bundle.REQUIRED[0], metadata()),
                          ('tactical/' + bundle.REQUIRED[1], raw(rows=2))])
        with self.assertRaisesRegex(bundle.BundleError, 'frame coverage'):
            self.prepare()
        self.assertFalse((self.destination / 'tactical').exists())
        with self.assertRaisesRegex(bundle.BundleError, 'structure differs'):
            bundle.validate_bundle(self.destination / '.plog-preparation/staged', expected_frames=2, expected_slots=46)

    def test_repository_destination_and_overlapping_paths_are_rejected(self):
        with self.assertRaisesRegex(bundle.BundleError, 'outside the public repository'):
            bundle.prepare_bundle(self.source, bundle.PROJECT / 'synthetic-never-created')
        with self.assertRaisesRegex(bundle.BundleError, 'non-overlapping'):
            bundle.prepare_bundle(self.source, self.source / 'prepared')
        (self.destination / '.git').mkdir(parents=True)
        with self.assertRaisesRegex(bundle.BundleError, 'outside every Git'):
            self.prepare()

    def test_lock_prevents_concurrent_runs_and_recovers_after_release(self):
        self.zip()
        with bundle._lock(self.destination):
            with self.assertRaisesRegex(bundle.BundleError, 'Another preparation'):
                self.prepare()
        self.prepare()

    def test_declared_archive_limits_and_nesting_are_bounded(self):
        self.zip()
        with mock.patch.object(bundle, 'MAX_BYTES', 1):
            with self.assertRaisesRegex(bundle.BundleError, 'expansion limit'):
                self.prepare()
        self.zip(members=[('nested.zip', archive_bytes([('anything', b'public synthetic')]))])
        with mock.patch.object(bundle, 'MAX_DEPTH', 0):
            with self.assertRaisesRegex(bundle.BundleError, 'Nested archive limit'):
                self.prepare()

    def test_cli_errors_are_content_free_and_no_tracebacks(self):
        path = self.zip()
        path.write_bytes(b'SYNTHETIC_SECRET_MARKER')
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = bundle.main(['--source', str(self.source), '--destination', str(self.destination)])
        self.assertEqual(code, 1)
        self.assertIn('PLOG FIRST CONTACT FAILED', err.getvalue())
        for forbidden in ('SYNTHETIC_SECRET_MARKER', path.name, 'Traceback'):
            self.assertNotIn(forbidden, out.getvalue() + err.getvalue())

    def test_existing_outputs_get_new_version_without_modification(self):
        output = self.destination / 'plog-output'
        output.mkdir(parents=True)
        previous = output / 'plog.csv'
        previous.write_bytes(b'preserve synthetic checked submission')
        self.assertEqual(bundle._output_directory(self.destination), output / 'run-0002')
        self.assertEqual(bundle._output_directory(self.destination), output / 'run-0003')
        self.assertEqual(previous.read_bytes(), b'preserve synthetic checked submission')

    @unittest.skipUnless(os.name == 'nt', 'PowerShell wrapper is Windows-specific')
    def test_preparation_wrapper_uses_public_preflight_and_strict_full_match_contract(self):
        self.zip()
        result = subprocess.run(['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
                                 str(bundle.PROJECT / 'scripts/prepare_competition_bundle.ps1'),
                                 '-SourceDirectory', str(self.source), '-DestinationDirectory', str(self.destination)],
                                capture_output=True, text=True, timeout=40)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('frame coverage failed', result.stderr)
        self.assertNotIn('Traceback', result.stderr)
        self.assertFalse((self.destination / 'plog-output').exists())


@unittest.skipUnless(os.name == 'nt', 'Frozen PowerShell pipeline is Windows-specific')
class FirstContactIntegrationTests(unittest.TestCase):
    def test_full_synthetic_zip_to_frozen_submission_with_version_preservation(self):
        from evaluation import possession_metrics as pm
        from plog.submission import write_submission
        with tempfile.TemporaryDirectory(prefix='plog-first-contact-synthetic-') as temporary:
            root = Path(temporary)
            source = root / 'downloads [synthetic]'
            source.mkdir()
            destination = root / 'licensed [synthetic]'
            # A previous format-valid, byte-reproduced synthetic submission must remain unchanged.
            previous = destination / 'plog-output/plog.csv'
            write_submission(previous, ['X'] * 199319)
            previous_bytes = previous.read_bytes()
            self.assertIsNone(pm.validate_submission(pm.read_label_csv(previous), 199319))
            with zipfile.ZipFile(source / 'drive-download [synthetic].zip', 'w', zipfile.ZIP_DEFLATED) as archive:
                archive.writestr('Drive/tactical/' + bundle.REQUIRED[0], metadata(46))
                players = ';'.join('0.501,0.5,2' if i < 23 else '0.8,0.5,2' for i in range(46))
                with archive.open('Drive/tactical/' + bundle.REQUIRED[1], 'w') as stream:
                    for frame in range(1, 199320):
                        stream.write(f'{frame}:{players}:0.5,0.5,0\n'.encode())
            result = subprocess.run(['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
                                     str(bundle.PROJECT / 'run_first_contact.ps1'), '-SourceDirectory', str(source),
                                     '-DestinationDirectory', str(destination)],
                                    capture_output=True, text=True, timeout=600)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(bundle.SUCCESS, result.stdout)
            output = destination / 'plog-output/run-0002/plog.csv'
            self.assertEqual(output.read_bytes(), output.with_suffix('.csv.regenerated.csv').read_bytes())
            self.assertIsNone(pm.validate_submission(pm.read_label_csv(output), 199319))
            self.assertEqual(previous.read_bytes(), previous_bytes)
            self.assertTrue(output.with_suffix('.csv.manifest.json').exists())
            self.assertNotIn('SHA256', result.stdout)


if __name__ == '__main__':
    unittest.main()
