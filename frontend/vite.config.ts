import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, ".", "");
  const configuredTarget = mode === "replay"
    ? env.VITE_REPLAY_BACKEND_URL?.trim() || "http://127.0.0.1:8000"
    : env.VITE_BACKEND_URL?.trim() || "http://127.0.0.1:8000";
  const target = configuredTarget.replace(/\/+$/, "");
  const wsTarget = target.replace(/^https:/, "wss:").replace(/^http:/, "ws:");

  return {
    plugins: [react()],
    server: {
      port: 5173,
      proxy: {
        "/api": {
          target,
          changeOrigin: true,
          secure: false,
        },
        "/health": {
          target,
          changeOrigin: true,
          secure: false,
        },
        "/metrics": {
          target,
          changeOrigin: true,
          secure: false,
        },
        "/ws": {
          target: wsTarget,
          ws: true,
          changeOrigin: true,
        },
      },
    },
  };
});
