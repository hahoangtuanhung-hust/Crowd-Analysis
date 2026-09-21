import { Component, type ErrorInfo, type ReactNode } from "react";
import { createRoot } from "react-dom/client";

import App from "./App";
import "./styles.css";

class RootErrorBoundary extends Component<{ children: ReactNode }, { error: string | null }> {
  state = { error: null as string | null };

  static getDerivedStateFromError(error: unknown) {
    return { error: error instanceof Error ? error.message : "Unexpected UI error" };
  }

  componentDidCatch(error: unknown, info: ErrorInfo) {
    console.error("Crowd Analysis UI failed to render", error, info.componentStack);
  }

  render() {
    if (this.state.error) {
      return (
        <main className="fatal-error" role="alert">
          <h1>Crowd Analysis</h1>
          <strong>Không thể khởi tạo giao diện</strong>
          <p>{this.state.error}</p>
        </main>
      );
    }
    return this.props.children;
  }
}

const root = document.getElementById("root");
if (!root) throw new Error("Missing #root mount element");

createRoot(root).render(
  <RootErrorBoundary>
    <App />
  </RootErrorBoundary>
);
