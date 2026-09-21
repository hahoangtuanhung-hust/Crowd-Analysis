import { createHash } from "node:crypto";
import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";

import { chromium } from "playwright";

const url = process.argv[2] ?? "http://localhost:5173";
const videoPath = path.resolve(process.argv[3] ?? "../data/videos/data-test.mp4");
const outputRoot = path.resolve(
  process.argv[4] ?? "../outputs/common_path/diagnostics-20260920-c",
);
const screenshotDir = path.join(outputRoot, "ui_screenshots");
await mkdir(screenshotDir, { recursive: true });

const browser = await chromium.launch({
  headless: true,
  executablePath: "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
});
const browserVersion = browser.version();
const page = await browser.newPage({ viewport: { width: 1600, height: 1100 } });
const consoleErrors = [];
const pageErrors = [];
const failedRequests = [];
const badResponses = [];

page.on("console", (message) => {
  if (message.type() === "error") consoleErrors.push(message.text());
});
page.on("pageerror", (error) => pageErrors.push(error.message));
page.on("requestfailed", (request) => {
  failedRequests.push({
    url: request.url(),
    method: request.method(),
    error: request.failure()?.errorText ?? "unknown",
  });
});
page.on("response", (response) => {
  if (response.status() >= 400) {
    badResponses.push({ url: response.url(), status: response.status() });
  }
});

const api = (route) => page.evaluate(async (value) => {
  const response = await fetch(value, { cache: "no-store" });
  if (!response.ok) throw new Error(`${value}: ${response.status}`);
  return response.json();
}, route);

async function waitForRunningFrame() {
  await page.waitForFunction(() => {
    const text = document.querySelector("[data-testid='replay-frame-meta']")?.textContent ?? "";
    return text.includes("frame") && !text.includes("frame -1");
  }, { timeout: 30_000 });
}

async function captureEvidence(index, elapsedSeconds) {
  const stage = page.locator(".video-stage");
  const screenshotPath = path.join(
    screenshotDir,
    `replay-${String(index).padStart(2, "0")}-${elapsedSeconds}s.png`,
  );
  const screenshot = await stage.screenshot({ path: screenshotPath });
  const frameResponse = await page.request.get(`${url}/api/stream/frame.jpg`, {
    timeout: 10_000,
  });
  if (!frameResponse.ok()) {
    throw new Error(`frame endpoint returned ${frameResponse.status()}`);
  }
  const frame = await frameResponse.body();
  const health = await api("/health");
  const commonPaths = await api("/api/analytics/common-paths");
  return {
    elapsed_seconds: elapsedSeconds,
    screenshot: path.relative(outputRoot, screenshotPath).replaceAll("\\", "/"),
    screenshot_sha256: createHash("sha256").update(screenshot).digest("hex"),
    rendered_frame_sha256: createHash("sha256").update(frame).digest("hex"),
    overlay_text: await page.locator("[data-testid='replay-frame-meta']").innerText(),
    session: health.session,
    active_or_cooling_paths: commonPaths.paths.filter((item) =>
      item.state === "active" || item.state === "cooling"
    ).length,
    image: await page.locator(".video-stage img").evaluate((image) => ({
      natural_width: image.naturalWidth,
      natural_height: image.naturalHeight,
      current_src: image.currentSrc,
    })),
  };
}

let report;
try {
  await page.goto(url, { waitUntil: "networkidle", timeout: 30_000 });
  await page.waitForFunction(() =>
    document.body.innerText.includes("Replay Backend Connected"),
  );
  await page.waitForFunction(() =>
    document.body.innerText.includes("N/A (cache)"),
  );
  const initialRuntime = await api("/api/runtime");
  const visualization = await api("/api/config/visualization");
  if (initialRuntime.mode !== "replay" || initialRuntime.detector_calls !== 0) {
    throw new Error(`runtime guard failed: ${JSON.stringify(initialRuntime)}`);
  }

  await page.locator("input[type='file']").setInputFiles(videoPath);
  await page.getByRole("button", { name: "Start", exact: true }).click();
  await waitForRunningFrame();
  const observationStarted = Date.now();
  const evidence = [];
  await page.waitForTimeout(2_000);
  evidence.push(await captureEvidence(1, 2));
  const replayBadgeVisible = await page.getByLabel("Cache replay").isVisible();
  const dashboardScreenshotPath = path.join(screenshotDir, "replay-dashboard.png");
  await page.screenshot({ path: dashboardScreenshotPath });
  await page.waitForTimeout(5_000);
  evidence.push(await captureEvidence(2, 7));
  await page.waitForTimeout(5_000);
  evidence.push(await captureEvidence(3, 12));

  const firstEpoch = evidence[0].session.stream_epoch;
  const runtimeBeforeStop = await api("/api/runtime");
  await page.getByRole("button", { name: "Stop", exact: true }).click();
  await page.waitForFunction(() => document.body.innerText.includes("STOPPED"));

  await page.getByRole("button", { name: "Start", exact: true }).click();
  await waitForRunningFrame();
  await page.waitForFunction((previousEpoch) => {
    const text = document.querySelector("[data-testid='replay-frame-meta']")?.textContent ?? "";
    return text.includes("epoch ")
      && !text.includes(previousEpoch.slice(0, 8))
      && /frame \d+/.test(text);
  }, firstEpoch, { timeout: 30_000 });
  await page.waitForTimeout(1_000);
  const restartedHealth = await api("/health");
  const restartedRuntime = await api("/api/runtime");
  const restartScreenshotPath = path.join(screenshotDir, "replay-after-restart.png");
  await page.locator(".video-stage").screenshot({ path: restartScreenshotPath });
  const epochReset = restartedHealth.session.stream_epoch !== firstEpoch;
  const cacheRewound = restartedHealth.session.frame_id >= 0
    && restartedHealth.session.frame_id < 100;
  const noInference = restartedRuntime.detector_calls === 0
    && restartedRuntime.detector_workers === 0
    && restartedRuntime.cache_reader_instances === 1
    && restartedRuntime.cache_mismatch_count === 0;

  await page.getByRole("button", { name: "Stop", exact: true }).click();
  await page.waitForFunction(() => document.body.innerText.includes("STOPPED"));
  const finalRuntime = await api("/api/runtime");
  const expectedStreamAborts = failedRequests.filter((item) =>
    item.url.includes("/api/stream.mjpg") && item.error.includes("ABORTED")
  );
  const unexpectedFailedRequests = failedRequests.filter((item) =>
    !expectedStreamAborts.includes(item)
  );
  const distinctFrameHashes = new Set(
    evidence.map((item) => item.rendered_frame_sha256),
  ).size;
  const distinctScreenshotHashes = new Set(
    evidence.map((item) => item.screenshot_sha256),
  ).size;
  const passed = distinctFrameHashes === evidence.length
    && distinctScreenshotHashes === evidence.length
    && replayBadgeVisible
    && evidence.every((item) => item.image.natural_width > 0)
    && epochReset
    && cacheRewound
    && noInference
    && finalRuntime.detector_calls === 0
    && finalRuntime.cache_mismatch_count === 0
    && consoleErrors.length === 0
    && pageErrors.length === 0
    && unexpectedFailedRequests.length === 0
    && badResponses.length === 0;

  report = {
    status: passed ? "verified" : "failed",
    url,
    source_mode: "modal_detection_cache_replay",
    topology: "local Edge browser -> Vite proxy -> local replay API; Modal output was downloaded before UI verification",
    browser: `Microsoft Edge ${browserVersion}`,
    headless: true,
    observed_seconds: (Date.now() - observationStarted) / 1000,
    initial_runtime: initialRuntime,
    visualization,
    dashboard_screenshot: path.relative(outputRoot, dashboardScreenshotPath).replaceAll("\\", "/"),
    evidence,
    stop_start: {
      first_epoch: firstEpoch,
      restarted_epoch: restartedHealth.session.stream_epoch,
      restarted_frame_id: restartedHealth.session.frame_id,
      epoch_reset: epochReset,
      cache_rewound: cacheRewound,
      runtime_before_stop: runtimeBeforeStop,
      runtime_after_restart: restartedRuntime,
      screenshot: path.relative(outputRoot, restartScreenshotPath).replaceAll("\\", "/"),
    },
    checks: {
      replay_label_visible: replayBadgeVisible,
      inference_timing_shown_as_not_applicable: true,
      decoded_image_visible: evidence.every((item) => item.image.natural_width > 0),
      distinct_rendered_frame_hashes: distinctFrameHashes,
      distinct_video_screenshot_hashes: distinctScreenshotHashes,
      video_continued_without_active_path: evidence.every(
        (item) => item.active_or_cooling_paths === 0,
      ),
      point_only_contract: visualization.show_tracking_points
        && !visualization.show_bounding_boxes
        && !visualization.show_individual_trajectories,
      detector_calls_zero: finalRuntime.detector_calls === 0,
      detector_workers_zero: finalRuntime.detector_workers === 0,
      one_cache_reader_across_restart: finalRuntime.cache_reader_instances === 1,
      cache_frame_alignment_errors: finalRuntime.cache_mismatch_count,
      skipped_cache_rows_for_dropped_frames: finalRuntime.cache_skipped_rows,
    },
    final_runtime: finalRuntime,
    console_errors: consoleErrors,
    page_errors: pageErrors,
    network: {
      bad_responses: badResponses,
      unexpected_failed_requests: unexpectedFailedRequests,
      expected_stream_aborts_on_stop: expectedStreamAborts,
    },
    limitations: [
      "UI replay used the local copy of Modal-produced cache/output; this does not verify a direct Modal-to-browser stream.",
      "No Active Path was supported by this 60-second cache, so the UI verifies stable playback with no Active rather than drawing an unsupported route.",
    ],
  };
} catch (error) {
  report = {
    status: "failed",
    url,
    browser: `Microsoft Edge ${browserVersion}`,
    error: error instanceof Error ? error.stack : String(error),
    console_errors: consoleErrors,
    page_errors: pageErrors,
    failed_requests: failedRequests,
    bad_responses: badResponses,
  };
} finally {
  if (report?.status === "failed") {
    await page.request.post(`${url}/api/stream/stop`, { timeout: 5_000 }).catch(() => null);
  }
  await writeFile(
    path.join(outputRoot, "ui_verification.json"),
    `${JSON.stringify(report, null, 2)}\n`,
    "utf8",
  );
  await browser.close();
}

if (report.status !== "verified") {
  console.error(JSON.stringify(report, null, 2));
  process.exitCode = 1;
} else {
  console.log(JSON.stringify({
    status: report.status,
    observed_seconds: report.observed_seconds,
    evidence: report.evidence.map((item) => ({
      overlay_text: item.overlay_text,
      frame_id: item.session.frame_id,
      screenshot: item.screenshot,
    })),
    stop_start: report.stop_start,
    final_runtime: report.final_runtime,
  }, null, 2));
}
