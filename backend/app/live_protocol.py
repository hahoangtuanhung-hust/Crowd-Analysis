from __future__ import annotations

import json
import struct
from typing import Any

PACKET_VERSION = 1
HEADER_LENGTH_BYTES = 4
MAX_HEADER_BYTES = 64 * 1024
MAX_JPEG_BYTES = 1_800_000


def pack_frame_packet(metadata: dict[str, Any], jpeg: bytes) -> bytes:
    if metadata.get("version") != PACKET_VERSION:
        raise ValueError(f"Frame packet version must be {PACKET_VERSION}")
    if not jpeg or len(jpeg) > MAX_JPEG_BYTES:
        raise ValueError(f"JPEG payload must be within 1..{MAX_JPEG_BYTES} bytes")
    header = json.dumps(metadata, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(header) > MAX_HEADER_BYTES:
        raise ValueError(f"Frame header exceeds {MAX_HEADER_BYTES} bytes")
    return struct.pack("!I", len(header)) + header + jpeg


def unpack_frame_packet(packet: bytes) -> tuple[dict[str, Any], bytes]:
    if len(packet) < HEADER_LENGTH_BYTES:
        raise ValueError("Frame packet is missing its header length")
    (header_length,) = struct.unpack("!I", packet[:HEADER_LENGTH_BYTES])
    if not 0 < header_length <= MAX_HEADER_BYTES:
        raise ValueError("Frame packet header length is invalid")
    jpeg_start = HEADER_LENGTH_BYTES + header_length
    if jpeg_start > len(packet):
        raise ValueError("Frame packet header is truncated")
    if jpeg_start == len(packet):
        raise ValueError("Frame packet has no JPEG payload")
    try:
        metadata = json.loads(packet[HEADER_LENGTH_BYTES:jpeg_start])
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Frame packet header is not valid UTF-8 JSON") from exc
    jpeg = packet[jpeg_start:]
    if metadata.get("version") != PACKET_VERSION:
        raise ValueError("Unsupported frame packet version")
    if len(jpeg) > MAX_JPEG_BYTES:
        raise ValueError("Frame packet JPEG exceeds the configured limit")
    return metadata, jpeg
