from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import cv2
from fastapi import UploadFile

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


@dataclass(frozen=True, slots=True)
class UploadedVideo:
    video_id: str
    source_token: str
    original_name: str
    size_bytes: int


class UploadStore:
    def __init__(self, directory: str | Path, max_upload_mb: int) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_upload_mb * 1024 * 1024

    async def save(self, upload: UploadFile) -> UploadedVideo:
        original_name = upload.filename or "video.mp4"
        suffix = Path(original_name).suffix.lower()
        if suffix not in ALLOWED_VIDEO_EXTENSIONS:
            raise ValueError(f"Unsupported video extension: {suffix or '(none)'}")

        video_id = uuid4().hex
        filename = f"{video_id}{suffix}"
        destination = self.directory / filename
        size = 0
        try:
            with destination.open("wb") as stream:
                while chunk := await upload.read(1024 * 1024):
                    size += len(chunk)
                    if size > self.max_bytes:
                        raise OverflowError("Video exceeds configured upload limit")
                    stream.write(chunk)
            capture = cv2.VideoCapture(str(destination))
            valid = capture.isOpened() and int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) > 0
            capture.release()
            if not valid:
                raise ValueError("Uploaded file is not a readable video")
        except Exception:
            destination.unlink(missing_ok=True)
            raise
        finally:
            await upload.close()

        return UploadedVideo(
            video_id=video_id,
            source_token=f"upload:{filename}",
            original_name=original_name,
            size_bytes=size,
        )

    def resolve(self, source: str) -> tuple[str, str, bool]:
        if source.startswith("upload:"):
            filename = source.removeprefix("upload:")
            if not filename or Path(filename).name != filename:
                raise ValueError("Invalid upload source token")
            candidate = (self.directory / filename).resolve()
            if candidate.parent != self.directory.resolve() or not candidate.is_file():
                raise ValueError("Uploaded video was not found")
            return str(candidate), "upload", False
        if source.lower().startswith(("rtsp://", "rtsps://")):
            return source, "rtsp", True
        candidate = Path(source)
        if candidate.is_file():
            return str(candidate.resolve()), "file", False
        raise ValueError("Source must be an upload token, an existing file, or an RTSP URL")
