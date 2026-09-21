# Directional Grid Common Path implementation report

## Implementation

The application now supports `legacy`, `directional_grid`, and `shadow` through
`analytics.common_path.engine`. The new engine is independent of detector, FastAPI, and UI
code. It consumes immutable event-time points keyed by camera, epoch, Track ID, and segment.

The implementation includes bounded causal track history, an eight-bin 32x18 directional
histogram, unique-track directed edges over 30/180-second windows, bounded beam candidate
search, ordered real-tracklet validation, and Candidate/Active/Cooling/Retired hysteresis.
Predicted ByteTrack grace points are marked `observed=false` and cannot create evidence.
Timestamp regressions fail fast; gaps, spatial jumps, calibration, zones, and new sessions
start new state/segments.

## Test evidence

- Full local suite: 89 tests passed.
- Frontend production build: passed.
- Modal smoke: `dg-smoke-20260919-b`, Tesla T4, one completed 25-second/750-frame GPU job.
- Modal integration: `dg-integration-20260919-final`, Tesla T4, one completed
  60-second/1,800-frame GPU job using the final directional engine.
- CPU replay: ByteTrack was rebuilt from the same content-addressed Modal detection
  cache and emitted the `tracklets-observed-v2` cache; no model was loaded locally.
- UI transport replay: frame versions and JPEG hashes changed over time after file pacing.

The 25-second directional replay reported 646 directed edges; the 60-second GPU
integration reported 1,267. Both produced zero switches, no validated complete route,
and `insufficient_data`. Legacy found one route with 56 IDs on the 25-second cache.
This difference is intentional: legacy combines edge-level support, while the directional
engine requires complete ordered tracklet evidence. Visual review showed fragmented,
multi-directional local flow rather than one defensible entrance-to-exit route.

The final GPU integration measured analytics p50/p95 at 8.404/13.773 ms,
inference p50/p95 at 21.236/23.396 ms, and offline processing at 11.614 FPS.
Sampled RAM peak was 5,371.21 MB; CUDA allocated/reserved peaks were 95.68/118.0 MB.
Remote wall time was 155.58 seconds. These are offline component timings, not
source-to-browser end-to-end latency.

## Limitations

The integration clip is only 60 seconds and cannot validate the 180-second window or long-run
memory behavior. It uses an uncalibrated pixel grid. Tracker fragmentation is material, with
2,054 rejected jumps and bounded-state evictions, so complete-route support may be undercounted.
The browser integration was unavailable; the backend MJPEG transport and rendered frames were
reviewed, but DOM-level UI verification remains `UI_NOT_VERIFIED`.
