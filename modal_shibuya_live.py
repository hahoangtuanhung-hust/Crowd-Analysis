"""Ephemeral authenticated Shibuya live-inference WebSocket on Modal GPU."""

from __future__ import annotations

import os

import modal


APP_NAME = "crowd-shibuya-live"
VOLUME_NAME = "crowd-analysis-data"
LIVE_DURATION_SECONDS = 200.0
SOURCE_HASH = "47e92a36bd8fb8cf98989500fd25ebf8356f6495c8f20038ad03ad8e4e30a502"
MODEL_HASH = "9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libglib2.0-0", "ffmpeg")
    .pip_install(
        "lap>=0.5.12",
        "numpy>=2,<3",
        "opencv-python-headless>=4.10,<6",
        "psutil>=6,<8",
        "pydantic>=2.10,<3",
        "pyyaml>=6,<7",
        "ultralytics>=8.4.116,<9",
    )
    .add_local_dir("backend", remote_path="/root/backend")
    .add_local_file("scripts/live_common_path.py", remote_path="/root/scripts/live_common_path.py")
    .add_local_file("scripts/__init__.py", remote_path="/root/scripts/__init__.py")
    .add_local_file("configs/shibuya.yaml", remote_path="/root/configs/shibuya.yaml")
    .add_local_file("yolo26n.pt", remote_path="/root/model/yolo26n.pt")
)


@app.function(
    image=image,
    gpu="T4",
    volumes={"/root/data": volume},
    max_containers=1,
    timeout=600,
    scaledown_window=30,
    env={"LIVE_SESSION_TOKEN": os.environ.get("LIVE_SESSION_TOKEN", "")},
)
@modal.concurrent(max_inputs=10)
@modal.asgi_app()
def live_api():
    import asyncio
    import hashlib
    import json
    import queue
    import re
    import sys
    import threading
    import uuid
    from pathlib import Path
    from urllib.parse import parse_qs

    import torch

    sys.path.insert(0, "/root")
    from backend.app.live_protocol import pack_frame_packet
    from scripts.live_common_path import LiveCommonPathProcessor, digest

    token = os.environ.get("LIVE_SESSION_TOKEN", "")
    token_fingerprint = hashlib.sha256(token.encode()).hexdigest()[:12] if token else None
    source = Path(f"/root/data/common_path/inputs/{SOURCE_HASH}.mp4")
    model = Path("/root/model/yolo26n.pt")
    config = Path("/root/configs/shibuya.yaml")
    pipeline_lock = asyncio.Lock()

    async def send_json(send, payload: dict[str, object]) -> None:
        await send({
            "type": "websocket.send",
            "text": json.dumps(payload, separators=(",", ":"), allow_nan=False),
        })

    async def receive_json(receive) -> dict[str, object]:
        message = await receive()
        if message["type"] == "websocket.disconnect":
            raise ConnectionError("browser disconnected")
        if message["type"] != "websocket.receive" or message.get("text") is None:
            raise ValueError("Control messages must be JSON text")
        value = json.loads(message["text"])
        if not isinstance(value, dict):
            raise ValueError("Control message must be a JSON object")
        return value

    async def health(send) -> None:
        payload = json.dumps({
            "status": "ready" if token and torch.cuda.is_available() else "unavailable",
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "authenticated": bool(token),
            "token_fingerprint": token_fingerprint,
        }).encode()
        await send({
            "type": "http.response.start",
            "status": 200,
            "headers": [(b"content-type", b"application/json"),
                        (b"content-length", str(len(payload)).encode())],
        })
        await send({"type": "http.response.body", "body": payload, "more_body": False})

    async def stream(scope, receive, send) -> None:
        first = await receive()
        if first["type"] != "websocket.connect":
            return
        await send({"type": "websocket.accept", "subprotocol": "", "headers": []})
        query = parse_qs(scope.get("query_string", b"").decode())
        received_token = query.get("token", [""])[0]
        if not received_token:
            try:
                authentication = await asyncio.wait_for(receive_json(receive), timeout=10)
            except (asyncio.TimeoutError, ConnectionError, ValueError):
                authentication = {}
            if authentication.get("action") == "authenticate":
                received_token = str(authentication.get("token", ""))
        if not token or received_token != token:
            await send_json(send, {
                "type": "event",
                "event": "error",
                "event_seq": 0,
                "message": "invalid session token",
                "expected_token_fingerprint": token_fingerprint,
                "received_token_fingerprint": hashlib.sha256(received_token.encode()).hexdigest()[:12],
            })
            await send({"type": "websocket.close", "code": 1008, "reason": ""})
            return
        if pipeline_lock.locked():
            await send_json(send, {
                "type": "event",
                "event": "error",
                "event_seq": 0,
                "message": "A live processing session is already running",
            })
            await send({"type": "websocket.close", "code": 1013, "reason": ""})
            return

        async with pipeline_lock:
            sequence = 0
            audit_events: list[dict[str, object]] = []
            stop_event = threading.Event()
            processor: LiveCommonPathProcessor | None = None
            iterator = None
            first_frame_sent_seq: int | None = None
            first_client_ack_seq: int | None = None
            producer: threading.Thread | None = None

            def next_sequence(event: str, **details: object) -> int:
                nonlocal sequence
                sequence += 1
                audit_events.append({"seq": sequence, "event": event, **details})
                return sequence

            async def send_event(event: str, **details: object) -> int:
                seq = next_sequence(event, **details)
                await send_json(send, {
                    "type": "event", "event": event, "event_seq": seq, **details
                })
                return seq

            await send_event("ready", source="data-shibuya.mp4")
            receiver = None
            try:
                command = await asyncio.wait_for(receive_json(receive), timeout=60)
                if command.get("action") != "start":
                    raise ValueError("First command must be start")
                duration = float(command.get("duration_seconds", LIVE_DURATION_SECONDS))
                preview_fps = float(command.get("preview_fps", 10))
                max_paths = command.get("max_paths", 3)
                if isinstance(max_paths, bool) or not isinstance(max_paths, int) or not 1 <= max_paths <= 5:
                    raise ValueError("max_paths must be an integer in [1, 5]")
                run_id = str(command.get("run_id", ""))
                client_kind = str(command.get("client_kind", "unknown"))
                if not 1 <= duration <= LIVE_DURATION_SECONDS:
                    raise ValueError(
                        f"duration_seconds must be within [1, {LIVE_DURATION_SECONDS:g}]"
                    )
                if not 1 <= preview_fps <= 15:
                    raise ValueError("preview_fps must be within [1, 15]")
                if command.get("processing_mode", "realtime_pts") != "realtime_pts":
                    raise ValueError("Only realtime_pts is permitted by the live endpoint")
                if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", run_id):
                    raise ValueError("Invalid run_id")
                if not torch.cuda.is_available():
                    raise RuntimeError("Modal GPU unavailable; local/CPU inference is refused")
                if not source.is_file() or digest(source) != SOURCE_HASH:
                    raise RuntimeError("Shibuya source is missing or its content hash changed")
                if not model.is_file() or digest(model) != MODEL_HASH:
                    raise RuntimeError("YOLO model is missing or its content hash changed")

                session_id = uuid.uuid4().hex
                stream_epoch = uuid.uuid4().hex
                output = Path("/root/data/common_path/runs") / run_id
                await send_event(
                    "loading_model",
                    session_id=session_id,
                    stream_epoch=stream_epoch,
                    processing_mode="realtime_pts",
                    realtime_1x=True,
                )
                processor = LiveCommonPathProcessor(
                    source=source,
                    config_path=config,
                    model_path=model,
                    output=output,
                    run_id=run_id,
                    session_id=session_id,
                    stream_epoch=stream_epoch,
                    duration_seconds=duration,
                    preview_fps=preview_fps,
                    source_name="data-shibuya.mp4",
                    device_name=torch.cuda.get_device_name(0),
                    stop_event=stop_event,
                    processing_mode="realtime_pts",
                    max_paths=max_paths,
                )

                async def receive_controls() -> None:
                    nonlocal first_client_ack_seq
                    while not stop_event.is_set():
                        message = await receive_json(receive)
                        action = message.get("action")
                        if action == "stop":
                            next_sequence("browser_stop")
                            stop_event.set()
                        elif action == "set_max_paths":
                            try:
                                value = processor.set_max_paths(message.get("max_paths"))
                                await send_event("max_paths_applied", applied_max_paths=value)
                            except ValueError as exc:
                                await send_event("control_error", message=str(exc))
                        elif action == "frame_ack" and first_client_ack_seq is None:
                            first_client_ack_seq = next_sequence(
                                "first_client_frame_ack",
                                frame_id=int(message.get("frame_id", -1)),
                                client_kind=client_kind,
                            )

                receiver = asyncio.create_task(receive_controls())
                iterator = processor.frames()
                latest_frame: queue.Queue[object] = queue.Queue(maxsize=1)
                producer_error: list[BaseException] = []
                producer_done = object()
                transport_drops = 0

                def produce_frames() -> None:
                    nonlocal transport_drops
                    try:
                        for produced in iterator:
                            while True:
                                try:
                                    latest_frame.put_nowait(produced)
                                    break
                                except queue.Full:
                                    latest_frame.get_nowait()
                                    transport_drops += 1
                    except BaseException as exc:
                        producer_error.append(exc)
                    finally:
                        while True:
                            try:
                                latest_frame.put(producer_done, timeout=0.1)
                                break
                            except queue.Full:
                                if stop_event.is_set():
                                    latest_frame.get_nowait()

                producer = threading.Thread(
                    target=produce_frames, name=f"live-producer-{session_id[:8]}", daemon=True
                )
                producer.start()
                while not stop_event.is_set():
                    item = await asyncio.to_thread(latest_frame.get)
                    if item is producer_done:
                        break
                    metadata, jpeg = item
                    metadata["transport_dropped_frames"] = transport_drops
                    frame_seq = next_sequence(
                        "frame_sent",
                        frame_id=metadata["frame_id"],
                        media_time_s=metadata["media_time_s"],
                    )
                    metadata["event_seq"] = frame_seq
                    await send({"type": "websocket.send", "bytes": pack_frame_packet(metadata, jpeg)})
                    if first_frame_sent_seq is None:
                        first_frame_sent_seq = frame_seq
                if stop_event.is_set():
                    producer.join(timeout=30)
                else:
                    await asyncio.to_thread(producer.join)
                if producer_error:
                    raise producer_error[0]
                manifest = processor.final_manifest or {}
                manifest["transport_dropped_frames"] = transport_drops
                completed_seq = next_sequence(
                    "processing_completed",
                    frames_processed=manifest.get("frames_processed", 0),
                    status=manifest.get("status", "unknown"),
                )
                verification = {
                    "client_kind": client_kind,
                    "first_frame_sent_seq": first_frame_sent_seq,
                    "first_client_ack_seq": first_client_ack_seq,
                    "processing_completed_seq": completed_seq,
                    "client_received_before_completion": bool(
                        first_frame_sent_seq is not None
                        and first_client_ack_seq is not None
                        and first_frame_sent_seq < first_client_ack_seq < completed_seq
                    ),
                    "ui_received_before_completion": bool(
                        client_kind == "browser-ui"
                        and first_frame_sent_seq is not None
                        and first_client_ack_seq is not None
                        and first_frame_sent_seq < first_client_ack_seq < completed_seq
                    ),
                    "events": audit_events,
                }
                (output / "live_verification.json").write_text(
                    json.dumps(verification, indent=2), encoding="utf-8"
                )
                manifest["live_verification"] = verification
                manifest["artifacts"] = sorted(path.name for path in output.iterdir() if path.is_file())
                (output / "manifest.json").write_text(
                    json.dumps(manifest, indent=2), encoding="utf-8"
                )
                await asyncio.to_thread(volume.commit)
                await send_json(send, {
                    "type": "event",
                    "event": "ended",
                    "event_seq": completed_seq,
                    "manifest": manifest,
                    "verification": verification,
                })
                await send({"type": "websocket.close", "code": 1000, "reason": ""})
            except ConnectionError:
                stop_event.set()
            except Exception as exc:
                stop_event.set()
                await send_event("error", message=str(exc))
                await send({"type": "websocket.close", "code": 1011, "reason": ""})
            finally:
                stop_event.set()
                if producer is not None and producer.is_alive():
                    await asyncio.to_thread(producer.join, 30)
                if (
                    iterator is not None
                    and (producer is None or not producer.is_alive())
                    and processor is not None
                    and processor.final_manifest is None
                ):
                    await asyncio.to_thread(iterator.close)
                if receiver is not None:
                    receiver.cancel()
                    try:
                        await receiver
                    except (asyncio.CancelledError, ConnectionError):
                        pass

    async def asgi(scope, receive, send):
        if scope["type"] == "lifespan":
            while True:
                message = await receive()
                if message["type"] == "lifespan.startup":
                    await send({"type": "lifespan.startup.complete"})
                elif message["type"] == "lifespan.shutdown":
                    await send({"type": "lifespan.shutdown.complete"})
                    return
        elif scope["type"] == "http":
            await receive()
            await health(send)
        elif scope["type"] == "websocket" and scope.get("path") == "/ws/live":
            await stream(scope, receive, send)
        elif scope["type"] == "websocket":
            first = await receive()
            if first["type"] == "websocket.connect":
                await send({"type": "websocket.close", "code": 1008, "reason": ""})

    return asgi
