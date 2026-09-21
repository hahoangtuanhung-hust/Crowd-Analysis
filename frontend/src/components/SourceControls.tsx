import { useRef, useState } from "react";
import { FileVideo, Play, Radio, Square, Upload } from "lucide-react";

import { startStream, stopStream, uploadVideo } from "../api";
import type { SessionStatus } from "../types";

interface SourceControlsProps {
  status: SessionStatus | undefined;
  replayMode?: boolean;
  modalLive?: {
    connected: boolean;
    sourceName: string;
    onStart: () => void;
    onStop: () => void;
  };
}

export function SourceControls({ status, replayMode = false, modalLive }: SourceControlsProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [mode, setMode] = useState<"video" | "rtsp">("video");
  const [file, setFile] = useState<File | null>(null);
  const [rtsp, setRtsp] = useState("");
  const [busy, setBusy] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState("");
  const running = status === "starting" || status === "running";

  if (modalLive) {
    return (
      <section className="source-bar modal-source-bar" aria-label="Modal live video source">
        <div className="modal-source-kind"><Radio size={17} aria-hidden="true" /> Modal GPU Live</div>
        <div className="modal-source-name">
          <strong>{modalLive.sourceName}</strong>
          <span>realtime_pts · dominant live flow</span>
        </div>
        <div className="source-actions">
          {running ? (
            <button className="button danger" type="button" onClick={modalLive.onStop}>
              <Square size={16} fill="currentColor" aria-hidden="true" /> Stop
            </button>
          ) : (
            <button className="button primary" type="button" onClick={modalLive.onStart} disabled={!modalLive.connected || status === "completed" || status === "stopped"}>
              <Play size={17} fill="currentColor" aria-hidden="true" /> Start GPU inference
            </button>
          )}
        </div>
      </section>
    );
  }

  async function handleStart() {
    setError("");
    setBusy(true);
    try {
      let source = "";
      if (mode === "video") {
        if (!file) throw new Error("Select a video file");
        setUploading(true);
        const uploaded = await uploadVideo(file);
        setUploading(false);
        source = uploaded.source_token;
      } else {
        source = rtsp.trim();
        if (!source) throw new Error("Enter an RTSP URL");
      }
      await startStream(source, "camera-01", replayMode ? true : undefined);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Unable to start stream");
    } finally {
      setUploading(false);
      setBusy(false);
    }
  }

  async function handleStop() {
    setError("");
    setBusy(true);
    try {
      await stopStream();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Unable to stop stream");
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="source-bar" aria-label="Video source">
      <div className="segmented" aria-label="Source type">
        <button className={mode === "video" ? "selected" : ""} onClick={() => setMode("video")} type="button">
          <FileVideo size={17} aria-hidden="true" /> Video
        </button>
        <button className={mode === "rtsp" ? "selected" : ""} onClick={() => setMode("rtsp")} type="button">
          <Radio size={17} aria-hidden="true" /> RTSP
        </button>
      </div>

      {mode === "video" ? (
        <div className="file-control">
          <input
            ref={inputRef}
            className="visually-hidden"
            type="file"
            accept="video/mp4,video/quicktime,video/x-msvideo,video/x-matroska"
            onChange={(event) => setFile(event.target.files?.[0] ?? null)}
          />
          <button className="button secondary" type="button" onClick={() => inputRef.current?.click()} disabled={running || busy}>
            <Upload size={17} aria-hidden="true" /> Choose file
          </button>
          <span className="file-name" title={file?.name}>{file?.name ?? "No file selected"}</span>
        </div>
      ) : (
        <label className="rtsp-control">
          <span className="visually-hidden">RTSP URL</span>
          <input
            value={rtsp}
            onChange={(event) => setRtsp(event.target.value)}
            placeholder="rtsp://camera/stream"
            disabled={running || busy}
          />
        </label>
      )}

      <div className="source-actions">
        {running ? (
          <button className="button danger" type="button" onClick={handleStop} disabled={busy}>
            <Square size={16} fill="currentColor" aria-hidden="true" /> Stop
          </button>
        ) : (
          <button className="button primary" type="button" onClick={handleStart} disabled={busy}>
            <Play size={17} fill="currentColor" aria-hidden="true" /> {uploading ? "Uploading..." : busy ? "Starting..." : "Start"}
          </button>
        )}
      </div>
      {error && <p className="source-error" role="alert">{error}</p>}
    </section>
  );
}
