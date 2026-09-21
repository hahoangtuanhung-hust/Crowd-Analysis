const crypto = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require(path.resolve(__dirname, '../frontend/node_modules/playwright'));

const url = process.env.TEST_UI_URL;
const output = path.resolve(process.env.TEST_OUTPUT_DIR);
const stopAtMediaSeconds = Number(process.env.TEST_STOP_AT_MEDIA_SECONDS || '0');
const captureMediaSeconds = (process.env.TEST_CAPTURE_MEDIA_SECONDS || '')
  .split(',').map(value => Number(value.trim())).filter(value => Number.isFinite(value) && value > 0);
if (!url || !process.env.TEST_OUTPUT_DIR) throw new Error('TEST_UI_URL and TEST_OUTPUT_DIR are required');
fs.mkdirSync(output, { recursive: true });

(async () => {
  const browser = await chromium.launch({
    executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe',
    headless: true,
  });
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
  const record = {
    url,
    topology: 'Chrome -> Vite UI -> authenticated WSS -> Modal T4 ASGI -> YOLO/ByteTrack/dominant_live_flow/render',
    browser: `Chrome ${browser.version()}`,
    frames: [],
    console_errors: [],
    failed_requests: [],
    http_errors: [],
  };
  page.on('console', message => {
    if (message.type() === 'error') record.console_errors.push(message.text());
  });
  page.on('requestfailed', request => {
    record.failed_requests.push(`${request.method()} ${request.url()}: ${request.failure()?.errorText}`);
  });
  page.on('response', response => {
    if (response.status() >= 400) record.http_errors.push({ status: response.status(), url: response.url() });
  });
  try {
    await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 30000 });
    await page.getByText('Modal GPU Connected').waitFor({ timeout: 180000 });
    await page.getByRole('button', { name: 'Start GPU inference' }).click();
    await page.getByTestId('modal-frame-meta').waitFor({ timeout: 180000 });
    await page.waitForFunction(() => document.querySelector('.video-stage img')?.naturalWidth > 0, null, { timeout: 30000 });

    for (let index = 1; index <= 3; index++) {
      if (index > 1) await page.waitForTimeout(3000);
      const image = await page.locator('.video-stage img').screenshot();
      const filename = `browser-live-${index}.png`;
      await page.locator('.video-stage').screenshot({ path: path.join(output, filename) });
      record.frames.push({
        filename,
        metadata: await page.getByTestId('modal-frame-meta').textContent(),
        dominant_flow_status: await page.getByTestId('dominant-flow-status').textContent(),
        image_sha256: crypto.createHash('sha256').update(image).digest('hex'),
        dimensions: await page.locator('.video-stage img').evaluate(img => [img.naturalWidth, img.naturalHeight]),
      });
    }

    for (const mediaSeconds of captureMediaSeconds) {
      await page.waitForFunction(seconds => {
        const text = document.querySelector('[data-testid="modal-frame-meta"]')?.textContent || '';
        const match = text.match(/t=([0-9.]+)s/);
        return match && Number(match[1]) >= seconds;
      }, mediaSeconds, { timeout: 590000 });
      const filename = `browser-media-${mediaSeconds}s.png`;
      const metadata = await page.getByTestId('modal-frame-meta').textContent();
      const image = await page.locator('.video-stage img').screenshot();
      await page.locator('.video-stage').screenshot({ path: path.join(output, filename) });
      record.frames.push({
        filename,
        metadata,
        dominant_flow_status: await page.getByTestId('dominant-flow-status').textContent(),
        image_sha256: crypto.createHash('sha256').update(image).digest('hex'),
        dimensions: await page.locator('.video-stage img').evaluate(img => [img.naturalWidth, img.naturalHeight]),
      });
    }

    if (stopAtMediaSeconds > 0) {
      await page.waitForFunction(seconds => {
        const text = document.querySelector('[data-testid="modal-frame-meta"]')?.textContent || '';
        const match = text.match(/t=([0-9.]+)s/);
        return match && Number(match[1]) >= seconds;
      }, stopAtMediaSeconds, { timeout: 300000 });
      record.stop_metadata = await page.getByTestId('modal-frame-meta').textContent();
      await page.getByRole('button', { name: 'Stop' }).click();
      await page.getByText('stopped', { exact: true }).waitFor({ timeout: 60000 });
    } else {
      await page.locator('.completion-banner').waitFor({ timeout: 590000 });
    }

    record.final_status = (await page.locator('.session-pill').textContent())?.trim();
    record.final_metadata = await page.getByTestId('modal-frame-meta').textContent();
    record.alerts = await page.locator('[role=alert]').allTextContents();
    record.overlay_controls = await page.locator('.overlay-controls').innerText();
    record.disabled_overlay_count = await page.locator('.overlay-controls input:disabled').count();
    await page.screenshot({ path: path.join(output, 'browser-final-full.png'), fullPage: true });
    record.passed = record.frames.length >= 3
      && new Set(record.frames.map(item => item.image_sha256)).size === 3
      && record.frames.every(item => item.dimensions[0] === 1280 && item.dimensions[1] === 720)
      && record.alerts.length === 0
      && record.console_errors.length === 0
      && record.failed_requests.length === 0
      && record.http_errors.length === 0
      && record.disabled_overlay_count >= 10
      && (stopAtMediaSeconds > 0 ? record.final_status === 'stopped' : record.final_status === 'completed');
  } catch (error) {
    record.passed = false;
    record.error = String(error);
    record.alerts = await page.locator('[role=alert]').allTextContents().catch(() => []);
    await page.screenshot({ path: path.join(output, 'browser-error.png'), fullPage: true }).catch(() => {});
  } finally {
    fs.writeFileSync(path.join(output, 'ui_verification.json'), JSON.stringify(record, null, 2));
    await browser.close();
    console.log(JSON.stringify(record, null, 2));
  }
  if (!record.passed) process.exitCode = 1;
})().catch(error => { console.error(error); process.exitCode = 1; });
