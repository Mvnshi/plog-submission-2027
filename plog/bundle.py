"""Offline-only bundle preparation. Develop/test with synthetic inputs; never send assets to AI.

No network APIs or general-purpose archive extraction are used. Only exact, public non-video
asset names are materialized. Persistent staging and atomic no-replace writes permit recovery.
"""

import argparse
from contextlib import contextmanager
import ctypes
import hashlib
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import uuid
import zipfile

from games import asset_paths


PROJECT = Path(__file__).resolve().parents[1]
CHUNK = 1024 * 1024
RESERVE = 256 * CHUNK
MAX_BYTES = 200 * 1024**3
MAX_MEMBER = 64 * 1024**3
MAX_ENTRIES = 100000
MAX_DEPTH = 4
ASSETS = {
    Path(value).name: Path(source) / Path(value).name
    for source in ("tactical", "broadcast")
    for kind, value in asset_paths(data_dir="unused-public-schema", source=source).items()
    if kind != "video"
}
REQUIRED = ("BAR-ATH_tactical_atd.xml", "BAR-ATH_tactical_atd.txt")
SUCCESS = "PLOG FIRST CONTACT PASSED: official validation and byte-for-byte regeneration passed."


class BundleError(Exception):
    """Only fixed, non-sensitive operator messages may be stored in this exception."""


def _safe_components(path):
    """Reject links/reparse points rather than following them into unexpected locations."""
    for component in (path, *path.parents):
        try:
            info = component.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise BundleError("A linked or reparse-point path was found. Use ordinary local folders.")


def _local_path(value):
    path = Path(os.path.abspath(os.path.expanduser(value)))
    _safe_components(path)
    if os.name == "nt":
        if str(path).startswith("\\\\") or ctypes.windll.kernel32.GetDriveTypeW(str(path.anchor)) == 4:
            raise BundleError("Network paths are not supported. Use a local disk.")
    return path.resolve()


def _paths(source, destination):
    source, destination = _local_path(source), _local_path(destination)
    if not source.is_dir():
        raise BundleError("Source directory is missing or unreadable. Check SourceDirectory locally.")
    if source == PROJECT or source in PROJECT.parents:
        raise BundleError("Choose the download folder itself, not a repository root or its parent.")
    if destination == PROJECT or PROJECT in destination.parents:
        raise BundleError("DestinationDirectory must be outside the public repository.")
    if source == destination or source in destination.parents or destination in source.parents:
        raise BundleError("SourceDirectory and DestinationDirectory must be separate, non-overlapping folders.")
    if any((parent / ".git").exists() for parent in (destination, *destination.parents)):
        raise BundleError("DestinationDirectory must be outside every Git checkout.")
    if destination.exists() and not destination.is_dir():
        raise BundleError("DestinationDirectory names a file. Choose a separate local folder.")
    return source, destination


def _space(path, needed):
    existing = path
    while not existing.exists():
        existing = existing.parent
    if shutil.disk_usage(existing).free < needed + RESERVE:
        raise BundleError("Insufficient free disk space. Free local space or choose another destination.")


def _mkdir(path):
    _safe_components(path)
    if any((parent / ".git").exists() for parent in (path, *path.parents)):
        raise BundleError("A preparation directory is inside a Git checkout. Choose a separate destination.")
    path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir():
        raise BundleError("A preparation directory collides with an existing file.")


def _regular(path):
    _safe_components(path)
    if not stat.S_ISREG(path.stat().st_mode):
        raise BundleError("An input or destination is not a regular local file.")


@contextmanager
def _lock(destination):
    _space(destination, 0)
    work = destination / ".plog-preparation"
    _mkdir(work)
    lock_path = work / "prepare.lock"
    _safe_components(lock_path)
    with lock_path.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise BundleError("Another preparation or inference run is using this destination. Wait for it to finish.") from None
        try:
            yield work
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _same(left, right):
    while True:
        a, b = left.read(CHUNK), right.read(CHUNK)
        if a != b:
            return False
        if not a:
            return True


def _install(opener, target, parts, size):
    """Copy atomically; existing files must compare byte-for-byte and are never replaced."""
    _safe_components(target)
    if target.exists():
        _regular(target)
        with opener() as incoming, target.open("rb") as existing:
            if not _same(incoming, existing):
                raise BundleError("Conflicting copies of an official file were found. Resolve downloads locally or use a new destination.")
        return
    _space(target.parent, size)
    _mkdir(target.parent)
    part = parts / (uuid.uuid4().hex + ".part")
    try:
        count = 0
        with opener() as incoming, part.open("xb") as outgoing:
            while block := incoming.read(CHUNK):
                count += len(block)
                if count > size:
                    raise BundleError("An input changed during preparation. Finish downloading and rerun locally.")
                outgoing.write(block)
            if count != size:
                raise BundleError("An input is incomplete or changed during preparation. Finish downloading and rerun locally.")
            outgoing.flush()
            os.fsync(outgoing.fileno())
        _safe_components(target)
        # Windows rename fails if a target exists. POSIX uses an atomic no-replace link.
        if os.name == "nt":
            os.rename(part, target)
        else:
            os.link(part, target)
            part.unlink()
    finally:
        part.unlink(missing_ok=True)  # Only this run's generated temporary file, never a download.


def _member(info):
    original = info.orig_filename
    name = original.replace("\\", "/")
    directory = name.endswith("/")
    parts = name.rstrip("/").split("/")
    mode = stat.S_IFMT(info.external_attr >> 16)
    if (not name or name.startswith("/") or len(name) > 4096 or "\0" in original
            or any(not p or p in (".", "..") or p[-1:] in (" ", ".") or ":" in p
                   or any(ord(c) < 32 for c in p) or any(c in p for c in '<>"|?*')
                   or re.fullmatch(r"(?i)(con|prn|aux|nul|com[0-9\u00b9\u00b2\u00b3]|lpt[0-9\u00b9\u00b2\u00b3])(?:\..*)?", p)
                   for p in parts)
            or mode not in (0, stat.S_IFREG, stat.S_IFDIR)
            or info.external_attr & 0x400 or info.flag_bits & 1
            or (mode == stat.S_IFDIR and not directory)
            or info.file_size > MAX_MEMBER):
        raise BundleError("An unsafe, encrypted or unsupported archive member was found. Obtain an intact official ZIP locally.")
    return "/".join(parts), directory


def _category(name):
    parts = name.replace("\\", "/").split("/")
    relative = ASSETS.get(parts[-1])
    if relative is not None:
        source_folders = [p for p in parts[:-1] if p.lower() in ("tactical", "broadcast")]
        if source_folders and source_folders[-1].lower() != relative.parts[0]:
            raise BundleError("An official filename disagrees with its tactical/broadcast source folder. Resolve the bundle locally.")
    return relative


class _Inventory:
    def __init__(self, stage, parts, work, destination):
        self.stage, self.parts, self.work = stage, parts, work
        self.destination = destination
        self.found = set()
        self.bytes = 0
        self.entries = 0
        self.archives = set()

    def accept(self, name, opener, size):
        relative = _category(name)
        if relative is not None:
            _install(opener, self.stage / relative, self.parts, size)
            self.found.add(Path(name).name)

    def archive(self, path, depth=0):
        if depth > MAX_DEPTH:
            raise BundleError("Nested archive limit reached. Obtain a standalone official tracking ZIP locally.")
        try:
            with zipfile.ZipFile(path) as archive:
                entries, names, hashes, parent_names = [], {}, {}, set()
                for info in archive.infolist():
                    name, directory = _member(info)
                    self.bytes += info.file_size
                    self.entries += 1
                    if self.bytes > MAX_BYTES or self.entries > MAX_ENTRIES:
                        raise BundleError("Archive expansion limit reached. Use a smaller official download set locally.")
                    key = name.casefold()
                    if key in names and names[key][1] != directory:
                        raise BundleError("Archive file/directory collision. Obtain an intact official ZIP locally.")
                    for parent in Path(key).parents:
                        prior = names.get(parent.as_posix())
                        if prior is not None and not prior[1]:
                            raise BundleError("Archive file/directory collision. Obtain an intact official ZIP locally.")
                    if not directory and key in parent_names:
                        raise BundleError("Archive file/directory collision. Obtain an intact official ZIP locally.")
                    if directory and info.file_size:
                        raise BundleError("Invalid archive directory record. Obtain an intact official ZIP locally.")
                    digest = hashlib.sha256()
                    count = 0
                    with archive.open(info) as stream:
                        while block := stream.read(CHUNK):
                            count += len(block)
                            if count > info.file_size:
                                raise BundleError("An archive member exceeded its declared size. Redownload locally.")
                            digest.update(block)
                    if count != info.file_size:
                        raise BundleError("An archive member is incomplete. Redownload locally.")
                    if key in names and not directory:
                        old = names[key][0]
                        if hashes[key] != digest.hexdigest():
                            raise BundleError("Conflicting duplicate archive members. Obtain an unambiguous official ZIP locally.")
                        with archive.open(old) as left, archive.open(info) as right:
                            if not _same(left, right):
                                raise BundleError("Conflicting duplicate archive members. Obtain an unambiguous official ZIP locally.")
                    else:
                        entries.append((info, name, directory, digest.hexdigest()))
                        names[key] = (info, directory)
                        hashes[key] = digest.hexdigest()
                    parent_names.update(parent.as_posix() for parent in Path(key).parents if parent.as_posix() != ".")
                # Every member (including skipped video) has passed CRC/length checks before writes.
                needed = 0
                for info, name, directory, digest in entries:
                    if directory:
                        continue
                    relative = _category(name)
                    if relative is not None:
                        needed += info.file_size * (int(not (self.stage / relative).exists())
                                                    + int(not (self.destination / relative).exists()))
                    elif name.lower().endswith(".zip") and digest not in self.archives:
                        needed += info.file_size * int(not (self.work / "nested" / (digest + ".zip")).exists())
                _space(self.work, needed)
                for info, name, directory, digest in entries:
                    if directory:
                        continue
                    opener = lambda info=info: archive.open(info)
                    if name.lower().endswith(".zip"):
                        if digest in self.archives:
                            continue
                        self.archives.add(digest)
                        nested = self.work / "nested" / (digest + ".zip")
                        _install(opener, nested, self.parts, info.file_size)
                        self.archive(nested, depth + 1)
                        nested.unlink()  # Generated cache only; original download remains unchanged.
                    else:
                        self.accept(name, opener, info.file_size)
        except (zipfile.BadZipFile, NotImplementedError, RuntimeError, EOFError):
            raise BundleError("A ZIP is corrupt, incomplete, encrypted or split. Redownload that archive locally; do not concatenate files.") from None


def _walk(source):
    pending, count = [source], 0
    while pending:
        directory = pending.pop()
        _safe_components(directory)
        with os.scandir(directory) as entries:
            children = sorted(entries, key=lambda item: item.name.casefold())
        for entry in children:
            count += 1
            if count > MAX_ENTRIES:
                raise BundleError("Download-folder entry limit reached. Choose a smaller source folder locally.")
            path = Path(entry.path)
            _safe_components(path)
            if entry.is_dir(follow_symlinks=False):
                pending.append(path)
            elif entry.is_file(follow_symlinks=False):
                # Leave every video, including -001/-002 parts, completely unopened.
                if path.suffix.lower() == ".zip" or path.name in ASSETS:
                    yield path
                elif re.search(r"(?i)\.(z[0-9]{2}|zip\.[0-9]{3})$", path.name):
                    raise BundleError("Split archive segments found. Obtain complete standalone ZIPs locally; do not join them blindly.")
            else:
                raise BundleError("A non-regular download entry was found. Use ordinary local files.")


def validate_bundle(directory, *, expected_frames=199319, expected_slots=46):
    """Format-only validation via the unchanged frozen loader. No GT, correction or fitting."""
    from .data import load_atd
    try:
        tracking = load_atd(directory, expected_frames=expected_frames)
        if len(tracking.teams) != expected_slots or set(tracking.teams) != {0, 1} or tracking.fps != 25:
            raise BundleError("Tactical tracking structure differs from the public schema. Restore the original tactical bundle locally.")
    except BundleError:
        raise
    except Exception:
        raise BundleError("Tactical tracking format or frame coverage failed. Restore the original tactical bundle locally; do not edit observations.") from None


def _prepare(source, destination, work, *, expected_frames=199319, expected_slots=46):
    parts, stage = work / "parts", work / "staged"
    _mkdir(parts)
    _mkdir(stage)
    # Recover only generated, recognizably named incomplete temporary files under the lock.
    for part in parts.iterdir():
        _regular(part)
        if not re.fullmatch(r"[0-9a-f]{32}\.part", part.name):
            raise BundleError("Unexpected preparation-cache entry. Choose a new destination locally.")
        part.unlink()
    inventory = _Inventory(stage, parts, work, destination)
    for path in _walk(source):
        _regular(path)
        if path.suffix.lower() == ".zip":
            inventory.archive(path)
        else:
            size = path.stat().st_size
            inventory.bytes += size
            if size > MAX_MEMBER:
                raise BundleError("An official input exceeds the supported size. Check the download locally.")
            if inventory.bytes > MAX_BYTES:
                raise BundleError("Input size limit reached. Choose a smaller official download set locally.")
            inventory.accept(path.relative_to(source).as_posix(), lambda path=path: path.open("rb"), size)
    for name, category in zip(REQUIRED, ("metadata", "raw tracking")):
        if name not in inventory.found:
            raise BundleError("Missing required tactical ATD " + category + ". Download the original tactical tracking bundle locally and rerun.")
    print(f"Checking tactical tracking structure and all {expected_frames} frame records locally.", flush=True)
    validate_bundle(stage, expected_frames=expected_frames, expected_slots=expected_slots)
    # Check every existing target before publishing any new target.
    for name in sorted(inventory.found):
        target, staged = destination / ASSETS[name], stage / ASSETS[name]
        _safe_components(target)
        if target.exists():
            _regular(target)
            with target.open("rb") as left, staged.open("rb") as right:
                if not _same(left, right):
                    raise BundleError("Prepared files conflict with the downloads. Preserve this destination and choose a new one.")
    _space(destination, sum((stage / ASSETS[name]).stat().st_size for name in inventory.found
                            if not (destination / ASSETS[name]).exists()))
    for name in sorted(inventory.found):
        staged = stage / ASSETS[name]
        _install(lambda staged=staged: staged.open("rb"), destination / ASSETS[name], parts, staged.stat().st_size)
    # These are the already validated bytes: atomic copies or full byte comparisons above.
    # The authoritative runner will read/validate them again on each independent inference run.


def prepare_bundle(source, destination, *, expected_frames=199319, expected_slots=46):
    """Offline library entry; non-default schema sizes exist solely for small synthetic tests."""
    source, destination = _paths(source, destination)
    with _lock(destination) as work:
        _prepare(source, destination, work, expected_frames=expected_frames, expected_slots=expected_slots)
    return destination


def _output_directory(destination):
    root = destination / "plog-output"
    _mkdir(root)
    artifacts = ("plog.csv", "plog.csv.manifest.json", "plog.csv.regenerated.csv", "plog.csv.regenerated.csv.manifest.json")
    for name in artifacts:
        _safe_components(root / name)
    if not any((root / name).exists() for name in artifacts):
        return root
    for number in range(2, 10000):
        candidate = root / f"run-{number:04d}"
        _safe_components(candidate)
        if not candidate.exists():
            candidate.mkdir()
            return candidate
    raise BundleError("Output version limit reached. Choose a new destination locally.")


def first_contact(source, destination):
    source, destination = _paths(source, destination)
    with _lock(destination) as work:
        _prepare(source, destination, work)
        output = _output_directory(destination)
        logs = work / "logs"
        _mkdir(logs)
        log = logs / (uuid.uuid4().hex + ".log")
        print("Running the frozen release twice with official validation. Keep this terminal open.", flush=True)
        with log.open("xb") as local_log:
            result = subprocess.run(
                ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                 str(PROJECT / "run_competition.ps1"), "-DataDir", str(destination),
                 "-OutputDirectory", str(output)], cwd=PROJECT, stdout=local_log, stderr=subprocess.STDOUT,
            )
        if result.returncode or not all((output / name).is_file() for name in
                ("plog.csv", "plog.csv.manifest.json", "plog.csv.regenerated.csv", "plog.csv.regenerated.csv.manifest.json")):
            raise BundleError("Frozen inference or byte validation failed. No submission is ready. Inspect local logs outside hosted tools.")
        return output / "plog.csv"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Offline-only official bundle organizer; run outside hosted AI sessions")
    parser.add_argument("--source", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    try:
        print("Checking downloads offline; large ZIP integrity checks can take several minutes. Videos will not be assembled.", flush=True)
        if args.run:
            output = first_contact(args.source, args.destination)
            print(SUCCESS)
            print("CSV: " + str(output))
        else:
            prepare_bundle(args.source, args.destination)
            print("PLOG BUNDLE PREPARED: tactical tracking format and frame coverage passed. No predictions generated.")
        return 0
    except BundleError as error:
        print("PLOG FIRST CONTACT FAILED: " + str(error), file=sys.stderr)
    except KeyboardInterrupt:
        print("PLOG FIRST CONTACT STOPPED: rerun the same command locally to resume safely.", file=sys.stderr)
    except Exception:
        print("PLOG FIRST CONTACT FAILED: local file operation failed. Check permissions, disk space and completed downloads locally; rerun to resume.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
