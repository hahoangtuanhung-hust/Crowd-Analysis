const crypto = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');

const input = path.resolve(process.argv[2]);
const source = JSON.parse(fs.readFileSync(input, 'utf8'));
const directory = path.dirname(input);
const screenshotFiles = source.frames.map(frame => path.join(directory, frame.filename));
const screenshotHashes = screenshotFiles.map(file =>
  crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex'));
const metadataComplete = source.frames.every(frame =>
  frame.metadata.includes('data-shibuya-test.mp4')
  && frame.dominant_flow_status.includes('Dòng di chuyển đông nhất')
  && /t=[0-9.]+s/.test(frame.metadata));
const passed = source.frames.length >= 3
  && new Set(source.frames.map(frame => frame.image_sha256)).size === source.frames.length
  && new Set(screenshotHashes).size === screenshotHashes.length
  && source.frames.every(frame => frame.dimensions[0] === 1280 && frame.dimensions[1] === 720)
  && screenshotFiles.every(file => fs.statSync(file).size > 0)
  && metadataComplete
  && source.alerts.length === 0
  && source.console_errors.length === 0
  && source.failed_requests.length === 0
  && source.http_errors.length === 0
  && source.disabled_overlay_count >= 10
  && ['completed', 'stopped'].includes(source.final_status);

const audit = {
  passed,
  predicate_version: 3,
  original_result: source.passed,
  original_false_reason: source.frames.length === 3
    ? null
    : 'The v1 harness required exactly three frames before milestone captures were added.',
  frame_count: source.frames.length,
  unique_rendered_image_hashes: new Set(source.frames.map(frame => frame.image_sha256)).size,
  unique_screenshot_hashes: new Set(screenshotHashes).size,
  metadata_complete: metadataComplete,
  final_status: source.final_status,
  console_errors: source.console_errors,
  failed_requests: source.failed_requests,
  http_errors: source.http_errors,
  alerts: source.alerts,
  disabled_overlay_count: source.disabled_overlay_count,
  screenshot_sha256: Object.fromEntries(source.frames.map((frame, index) => [
    frame.filename, screenshotHashes[index]
  ])),
};
const destination = path.join(directory, 'ui_verification_audit.json');
fs.writeFileSync(destination, JSON.stringify(audit, null, 2));
console.log(JSON.stringify(audit, null, 2));
if (!passed) process.exitCode = 1;
