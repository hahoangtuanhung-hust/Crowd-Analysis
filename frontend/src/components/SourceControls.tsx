import { useRef, useState } from "react";
import { FileVideo, Play, Radio, Square, Upload } from "lucide-react";

import { startStream, stopStream, uploadVideo } from "../api";
import type { SessionStatus } from "../types";

interface SourceControlsProps {
  status: SessionStatus | undefined;
}

export function SourceControls({ status }: SourceControlsProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [mode, setMode] = useState<"video" | "rtsp">("video");
  const [file, setFile] = useState<File | null>(null);
  const [rtsp, setRtsp] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const running = status === "starting" || status === "running";

  async function handleStart() {
    setError("");
    setBusy(true);
    try {
      const source = mode === "video"
        ? file ? (await uploadVideo(file)).source_token : ""
        : rtsp.trim();
      if (!source) throw new Error(mode === "video" ? "Select a video file" : "Enter an RTSP URL");
      await startStream(source, "camera-01");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Unable to start stream");
    } finally {
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
            <Play size={17} fill="currentColor" aria-hidden="true" /> {busy ? "Starting" : "Start"}
          </button>
        )}
      </div>
      {error && <p className="source-error" role="alert">{error}</p>}
    </section>
  );
}
