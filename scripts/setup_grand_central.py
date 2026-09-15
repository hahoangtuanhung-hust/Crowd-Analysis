from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.datasets import GrandCentralDataset, GrandCentralPaths

DATASET_URL = (
    "https://www.dropbox.com/s/7y90xsxq0l0yv8d/"
    "cvpr2015_pedestrianWalkingPathDataset.rar?dl=1"
)
LICENSE_WARNING = (
    "The Grand Central walking-path dataset has no clear redistribution license. "
    "Use it only for research, education, internal demos, and benchmarks."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Set up and validate Grand Central data")
    parser.add_argument("--root", type=Path, help="Override GC_DATASET_ROOT")
    parser.add_argument("--video", type=Path, help="Existing converted video")
    parser.add_argument("--archive", type=Path, help="Existing annotation .rar or .zip")
    parser.add_argument(
        "--opentraj-dir",
        type=Path,
        help="Existing OpenTraj datasets/GC directory containing Annotation and H.json",
    )
    parser.add_argument("--download", action="store_true", help="Download the trusted OpenTraj archive")
    parser.add_argument("--yes", action="store_true", help="Confirm the license warning non-interactively")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def install_file(source: Path, destination: Path) -> str:
    source = source.expanduser().resolve()
    if not source.is_file() or source.stat().st_size == 0:
        raise ValueError(f"Source file is missing or empty: {source}")
    if destination.exists():
        if (
            destination.stat().st_size == source.stat().st_size
            and sha256(destination) == sha256(source)
        ):
            return "existing"
        raise FileExistsError(
            f"Destination exists with different content: {destination}. "
            "Remove it explicitly first."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        return "hard_link"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def copy_opentraj(source: Path, paths: GrandCentralPaths) -> dict[str, int]:
    source = source.expanduser().resolve()
    annotation_source = source / "Annotation"
    if not annotation_source.is_dir():
        candidates = list(source.rglob("Annotation"))
        annotation_source = candidates[0] if len(candidates) == 1 else annotation_source
    files = sorted(annotation_source.glob("*.txt")) if annotation_source.is_dir() else []
    if not files:
        raise FileNotFoundError(f"No annotation .txt files found under {source}")
    copied = 0
    for annotation in files:
        destination = paths.annotations / annotation.name
        if not destination.exists():
            shutil.copy2(annotation, destination)
            copied += 1
    dataset_dir = annotation_source.parent
    homography_source = next(
        (
            candidate
            for candidate in (dataset_dir / "H.json", source / "H.json", source.parent / "H.json")
            if candidate.is_file()
        ),
        None,
    )
    homography_copied = 0
    if homography_source is not None:
        destination = paths.homography / "H.json"
        if not destination.exists():
            shutil.copy2(homography_source, destination)
            homography_copied = 1
    for name in ("ReadMe.md", "README.md", "reference.jpg"):
        candidate = dataset_dir / name
        if candidate.is_file():
            destination = paths.homography / name
            if not destination.exists():
                shutil.copy2(candidate, destination)
    return {"annotation_files_copied": copied, "homography_files_copied": homography_copied}


def extract_archive(archive: Path, destination: Path) -> Path:
    archive = archive.expanduser().resolve()
    if not archive.is_file() or archive.stat().st_size == 0:
        raise ValueError(f"Archive is missing or empty: {archive}")
    extract_root = destination / "_extracted"
    extract_root.mkdir(parents=True, exist_ok=True)
    if archive.suffix.lower() == ".zip":
        shutil.unpack_archive(str(archive), str(extract_root))
        return extract_root
    executable = shutil.which("7z") or shutil.which("unrar")
    if executable is None:
        raise RuntimeError(
            "Cannot extract .rar: install 7-Zip (7z) or unrar and put it on PATH, "
            "then rerun this command."
        )
    command = [executable, "x", str(archive), f"-o{extract_root}", "-y"]
    if Path(executable).stem.lower() == "unrar":
        command = [executable, "x", "-o+", str(archive), str(extract_root)]
    subprocess.run(command, check=True)
    return extract_root


def download_archive(paths: GrandCentralPaths, *, confirmed: bool) -> Path:
    destination = paths.root / "downloads" / "cvpr2015_pedestrianWalkingPathDataset.rar"
    print(f"Source URL: {DATASET_URL}")
    print(f"Destination: {destination.resolve()}")
    print("Estimated size: unknown (verify free space before continuing)")
    print(f"License warning: {LICENSE_WARNING}")
    if not confirmed:
        answer = input("Continue download? [y/N] ").strip().lower()
        if answer not in {"y", "yes"}:
            raise SystemExit("Download cancelled")
    if destination.is_file() and destination.stat().st_size > 0:
        print("Archive already exists; skipping download")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(DATASET_URL, timeout=60) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output)
    if temporary.stat().st_size == 0:
        temporary.unlink(missing_ok=True)
        raise ValueError("Downloaded archive is empty")
    temporary.replace(destination)
    return destination


def write_metadata(
    dataset: GrandCentralDataset,
    *,
    actions: dict[str, Any],
) -> dict[str, Any]:
    validation = dataset.validate()
    destination = dataset.paths.processed / "metadata.json"
    existing: dict[str, Any] = {}
    if destination.is_file():
        existing = json.loads(destination.read_text(encoding="utf-8"))
    metadata = {
        "dataset": "Grand Central Station walking paths",
        "camera_id": dataset.camera_id,
        "source_page": "https://www.ee.cuhk.edu.hk/~xgwang/grandcentral.html",
        "integration_source": "https://github.com/crowdbotp/OpenTraj/tree/master/datasets/GC",
        "license": "not clearly specified by dataset publisher",
        "allowed_project_use": ["research", "education", "internal_demo", "benchmark"],
        "redistribution": "not authorized by this project",
        "actions": actions,
        "video_sha256": sha256(dataset.paths.video),
        "validation": validation,
    }
    if "normalization" in existing:
        metadata["normalization"] = existing["normalization"]
    destination.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    report = dataset.paths.processed / "validation_report.json"
    report.write_text(json.dumps(validation, indent=2), encoding="utf-8")
    return metadata


def main() -> int:
    args = parse_args()
    paths = GrandCentralPaths.from_env(args.root)
    paths.create()
    actions: dict[str, Any] = {}

    if args.video is not None:
        actions["video"] = install_file(args.video, paths.video)
    elif not paths.video.is_file():
        raise FileNotFoundError(
            f"No video at {paths.video}. Pass --video data/videos/data.mp4."
        )

    archive = args.archive
    if args.download:
        archive = download_archive(paths, confirmed=args.yes)
    if archive is not None:
        extracted = extract_archive(archive, paths.root)
        actions["archive"] = {"path": str(archive.resolve()), "sha256": sha256(archive)}
        actions.update(copy_opentraj(extracted, paths))
    elif args.opentraj_dir is not None:
        actions.update(copy_opentraj(args.opentraj_dir, paths))
    elif not any(paths.annotations.glob("*.txt")):
        raise FileNotFoundError(
            "No annotations installed. Pass --archive or --opentraj-dir."
        )

    dataset = GrandCentralDataset(paths)
    metadata = write_metadata(dataset, actions=actions)
    validation = metadata["validation"]
    print(json.dumps({
        "status": "ready",
        "root": str(paths.root.resolve()),
        "video": validation["video"],
        "annotation_files": validation["annotation_files"],
        "annotation_points": validation["annotation_points"],
        "annotated_frames": validation["annotated_frames"],
        "homography": validation["homography"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
