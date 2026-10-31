"""Deterministic CSV writing and byte-for-byte reproduction checks.

No tracking, labels or audit artifacts are uploaded or logged by these helpers.
All validation uses the unmodified executable official metric definition.
"""

import hashlib
from pathlib import Path
import os
import tempfile

import numpy as np
import pandas as pd

from evaluation import possession_metrics as pm


def write_submission(path, labels, expected_rows=199319):
    """Atomically write exact ``frame,label`` UTF-8 CSV bytes with LF newlines.

    Validation happens before touching an existing output. All callers must name
    an explicit expected row count when writing a shorter external fixture.
    """
    if isinstance(expected_rows, (bool, np.bool_)) or not isinstance(expected_rows, (int, np.integer)) or expected_rows <= 0:
        raise ValueError("expected_rows must be a positive integer")
    labels = np.asarray(labels, dtype=object)
    if labels.ndim != 1:
        raise ValueError("labels must be a one-dimensional array")
    table = pd.DataFrame({"frame": np.arange(len(labels), dtype=np.int64), "label": labels})
    reason = pm.validate_submission(table, expected_rows)
    if reason:
        raise ValueError(reason)
    # Normalize exactly as the official scorer does, after checking blank cells.
    labels = np.array(pm.LABELS)[pm.label_codes(table)]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write("frame,label\n")
            handle.writelines(f"{frame},{label}\n" for frame, label in enumerate(labels))
        reason = pm.validate_submission(pm.read_label_csv(temporary), expected_rows)
        if reason:
            raise ValueError("written CSV failed official validation: " + reason)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return path


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_reproduction(reference, candidate):
    """Compare actual bytes and hashes; raise on any mismatch.

    Return metadata only. The comparison never prints or returns label contents.
    Equal hashes are useful provenance, but bytes are independently compared to
    fulfill the reproduction requirement literally.
    """
    reference, candidate = Path(reference), Path(candidate)
    reference_hash, candidate_hash = sha256_file(reference), sha256_file(candidate)
    equal = reference.stat().st_size == candidate.stat().st_size
    if equal:
        with reference.open("rb") as left, candidate.open("rb") as right:
            while True:
                a, b = left.read(1024 * 1024), right.read(1024 * 1024)
                if a != b:
                    equal = False
                    break
                if not a:
                    break
    if not equal or reference_hash != candidate_hash:
        raise ValueError("reproduction failed: files differ byte-for-byte")
    return {"equal": True, "sha256": reference_hash, "size_bytes": reference.stat().st_size}
