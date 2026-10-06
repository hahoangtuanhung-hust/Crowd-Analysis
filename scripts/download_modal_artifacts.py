"""Download one completed Modal Volume run without starting a GPU job."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

VOLUME_NAME = "crowd-analysis-data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--volume", default=VOLUME_NAME)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", args.run_id):
        raise ValueError("run-id must contain only letters, numbers, underscores and hyphens")
    output = args.output_dir or Path("outputs/common_path") / args.run_id
    if output.exists() and any(output.iterdir()) and not args.force:
        raise FileExistsError(f"Output directory is not empty: {output}; use --force")
    output.mkdir(parents=True, exist_ok=True)
    remote = f"common_path/runs/{args.run_id}"
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    command = [sys.executable, "-m", "modal", "volume", "get", args.volume, remote, str(output)]
    if args.force:
        command.append("--force")
    result = subprocess.run(
        command,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(result.returncode)
    print(f"Downloaded Modal run {args.run_id} to {output.resolve()}")


if __name__ == "__main__":
    main()
