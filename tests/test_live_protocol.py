import pytest

from backend.app.live_protocol import MAX_JPEG_BYTES, pack_frame_packet, unpack_frame_packet


def test_live_frame_packet_round_trip() -> None:
    metadata = {"version": 1, "session_id": "session-1", "frame_id": 42}
    jpeg = b"\xff\xd8payload\xff\xd9"

    decoded_metadata, decoded_jpeg = unpack_frame_packet(pack_frame_packet(metadata, jpeg))

    assert decoded_metadata == metadata
    assert decoded_jpeg == jpeg


@pytest.mark.parametrize(
    "payload",
    [b"", b"x" * (MAX_JPEG_BYTES + 1)],
    ids=["empty", "oversize"],
)
def test_live_frame_packet_rejects_invalid_jpeg_size(payload: bytes) -> None:
    with pytest.raises(ValueError, match="JPEG payload"):
        pack_frame_packet({"version": 1}, payload)


def test_live_frame_packet_rejects_truncated_header() -> None:
    with pytest.raises(ValueError, match="header"):
        unpack_frame_packet(b"\x00\x00\x00\x10{}")
