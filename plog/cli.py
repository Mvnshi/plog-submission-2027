"""Small command line entry points. All model inference and file access are local."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from games import GAME, GAMES, half_video_blocks
from .config import config_dict, load_config
from .data import load_atd, load_npz
from .inference import infer
from .submission import audit_reproduction, sha256_file, write_submission


ROOT = Path(__file__).resolve().parents[1]


def official_validate(path, rows=199319):
    """Invoke the unmodified official validator, returning no label contents."""
    result = subprocess.run([sys.executable, str(ROOT / "evaluation" / "validate_submission.py"),
                             str(path), "--rows", str(rows)], capture_output=True, text=True)
    if result.returncode:
        raise ValueError("official validator rejected output; inspect locally outside hosted tooling")


def main(argv=None):
    parser = argparse.ArgumentParser(description="PLOG local possession research and offline submission")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (("infer", "Infer public/external NPZ tracking"),
                            ("offline", "Restricted competition execution: run outside hosted-agent session")):
        cmd = commands.add_parser(name, help=help_text)
        cmd.add_argument("--config", default=str(ROOT / "configs" / "default.json"))
        cmd.add_argument("--model", help="Optional externally trained plog-softmax-v1 JSON")
        cmd.add_argument("--out", required=True)
        cmd.add_argument("--audit-reference", help="Previously submitted CSV; must differ from --out")
        if name == "infer":
            cmd.add_argument("--tracking", required=True)
            cmd.add_argument("--events", help="Public/external JSON list with tactical-frame intervals")
        else:
            cmd.add_argument("--data-dir", required=True)
            cmd.add_argument("--events", action="store_true", help="Use automatic Smart Tagging asset")
    validate = commands.add_parser("validate", help="Run official format checks")
    validate.add_argument("csv")
    validate.add_argument("--rows", type=int, default=199319)
    audit = commands.add_parser("audit", help="Compare submission and regenerated CSV byte-for-byte")
    audit.add_argument("reference")
    audit.add_argument("candidate")
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            official_validate(args.csv, args.rows)
            print("Official format validation passed.")
            return 0
        if args.command == "audit":
            result = audit_reproduction(args.reference, args.candidate)
            print(json.dumps(result, sort_keys=True))
            return 0
        if args.audit_reference and Path(args.audit_reference).resolve() == Path(args.out).resolve():
            raise ValueError("audit reference and generated output must be different paths")
        config = load_config(args.config)
        model = None
        if args.model:
            from .learning import SoftmaxModel
            model = SoftmaxModel.load(args.model)
        start = time.perf_counter()
        events, blocks = (), None
        if args.command == "offline":
            tracking = load_atd(args.data_dir)
            blocks = [(first, last + 1) for _, first, last in half_video_blocks(GAME)]
            if args.events:
                from starter.load_smart_tagging import load
                events = load(data_dir=args.data_dir)
            rows = GAMES[GAME]["n_video_frames"]
        else:
            tracking = load_npz(args.tracking)
            if args.events:
                events = json.loads(Path(args.events).read_text(encoding="utf-8"))
            rows = tracking.n_frames
        labels = infer(tracking, config=config, events=events, model=model, blocks=blocks)
        output = write_submission(args.out, labels, expected_rows=rows)
        official_validate(output, rows)
        if args.audit_reference:
            audit_reproduction(args.audit_reference, output)
        # Keep provenance beside output, in an ignored destination; never include arrays.
        record = {"format": "plog-run-v1", "config": config_dict(config),
                  "config_sha256": hashlib.sha256(json.dumps(config_dict(config), sort_keys=True).encode()).hexdigest(),
                  "model_sha256": sha256_file(args.model) if args.model else None,
                  "output_sha256": sha256_file(output), "rows": rows,
                  "seconds": time.perf_counter() - start, "official_validation": True,
                  "source": "tactical" if args.command == "offline" else "external_npz"}
        output.with_suffix(output.suffix + ".manifest.json").write_text(
            json.dumps(record, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")
        print(f"Wrote and validated {output}; SHA256 {record['output_sha256']}")
        return 0
    except (ValueError, OSError, KeyError) as error:
        print(f"PLOG: {error}", file=sys.stderr)
        return 1
