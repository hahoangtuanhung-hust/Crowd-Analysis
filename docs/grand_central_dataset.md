# Grand Central Dataset Integration

## Scope and licensing

This integration uses the CVPR 2015 pedestrian walking-path annotations distributed through the OpenTraj Grand Central integration. The publisher and OpenTraj do not state a clear dataset license. Use the data only for research, education, internal demonstrations, and benchmarking. Do not redistribute it, include it in a container image, commit it to Git, or claim commercial-use permission.

The [original CUHK page](https://www.ee.cuhk.edu.hk/~xgwang/grandcentral.html) also links KLT keypoint tracks (`trajectories.rar` and `trajectoriesNew.rar`). Those are not the pedestrian walking-path annotations parsed here.

## Verified local data

The setup and validation scripts measured the installed files rather than copying reference numbers:

| Property | Verified value |
| --- | ---: |
| Annotation files / pedestrian IDs | 12,684 |
| Trajectory points | 456,410 |
| Distinct annotated source frames | 5,741 |
| Source frame range | 0 to 120,000 |
| Average points per annotated frame | 79.500 |
| Maximum points in one annotated frame | 289 |
| Local MP4 | 1920 x 1080, 30 fps, 6,003 frames, 200.1 s |
| Annotated frames overlapping the MP4 | 251 (source frame 0 to 5,000) |

The local MP4 SHA-256 and full validation payload are recorded in `data/grand-central/processed/metadata.json` and `validation_report.json`.

## Raw annotation format

Each numeric filename is one publisher pedestrian/track ID. Its content is whitespace-separated repeated triples:

```text
x y source_frame_id x y source_frame_id ...
```

- `x` is the first token and increases from left to right.
- `y` is the second token and increases from top to bottom.
- Coordinates are pixel-space trajectory points in the 1920 x 1080 source view.
- There are no bounding boxes, detection confidence values, or explicit lost-track flags.
- Source frame IDs use 25 fps: `timestamp = source_frame_id / 25`.
- The normal annotation step is 20 source frames, or 0.8 seconds (1.25 Hz).
- A gap larger than 20 frames starts a new normalized trajectory segment. No missing points are interpolated.

The OpenTraj `loader_gcs.py` names the first two raw tokens `py` and `px`, then emits them in swapped order. Direct overlays on the supplied 1920 x 1080 frames proved that first-token-as-x is correct; the swapped interpretation is spatially invalid. This adapter deliberately differs from that loader. OpenTraj also interpolates gaps and multiplies transformed coordinates by 0.8; this project does neither because those operations are not part of the raw annotation contract.

## Video timebase

The walking-path annotation IDs remain on the 25 fps source timeline, while the supplied MP4 was resampled to 30 fps. Alignment uses timestamps, not identical frame IDs:

```text
annotation_timestamp = annotation_frame_id / 25
video_frame_id = round(annotation_timestamp * 30)
```

This was visually checked at the start and at 120 seconds (`annotation frame 3000` to `MP4 frame 3600`). Only annotations whose timestamp is at most 200.1 seconds can be compared with the supplied clip.

## Normalized schema

`scripts/prepare_grand_central.py` writes the following ordered schema to Parquet and optionally CSV:

```text
camera_id, source, frame_id, timestamp, track_id, x, y,
x1, y1, x2, y2, foot_x, foot_y, world_x, world_y,
confidence, zone_id
```

For ground truth, `source=ground_truth`, `x/y` are the raw trajectory point, and `foot_x=x`, `foot_y=y`. Bounding-box and confidence fields remain null. For model output, `source=prediction`, bbox/confidence are populated and the foot point is the bbox bottom-center. Ground-truth and prediction artifacts are never merged.

## Homography and zones

The adapter accepts the OpenTraj `H.json` key `homog` and the normalized `homography.json` key `matrix`. It rejects non-3x3, non-finite, or singular matrices. Homography is applied exactly once from the pixel foot point.

OpenTraj describes this matrix as manually calculated from concourse dimensions; it is not official camera calibration. Therefore this project labels values `unverified_ground_unit`, never metres. Pixel analytics remains available without homography.

`configs/grand_central_zones.json` contains six manually defined polygons for the fixed view. They are application configuration, not publisher labels. Ground-plane analytics transforms each polygon once with the same matrix used for points. Zone transitions use the existing two-observation debounce.

## Artifacts

```text
data/grand-central/
  raw/video/grand_central.mp4
  raw/annotations/*.txt
  raw/homography/H.json
  processed/trajectories.parquet
  processed/trajectories.csv
  processed/metadata.json
  processed/validation_report.json
  processed/normalization_report.json
  processed/zones.json
  processed/homography.json
  outputs/tracking/
  outputs/heatmaps/
  outputs/popular_paths/
  outputs/zone_flows/
  outputs/benchmarks/
```

All three data directories and archive/video extensions are ignored by Git. `data/grand-central` and `data/videos` are excluded from the Docker build context.

## Reproducible commands

Install from the provided MP4 and an existing OpenTraj checkout:

```powershell
python scripts/setup_grand_central.py --video data/videos/data.mp4 --opentraj-dir path/to/OpenTraj/datasets/GC
```

Or use the manually downloaded archive:

```powershell
python scripts/setup_grand_central.py --video data/videos/data.mp4 --archive path/to/cvpr2015_pedestrianWalkingPathDataset.rar
```

For RAR files, `7z` or `unrar` must be on `PATH`. `--download` shows the trusted OpenTraj URL, destination, size status, and license warning before confirmation. Existing valid files are not downloaded or copied again.

Prepare the full normalized corpus with a debug CSV:

```powershell
python scripts/prepare_grand_central.py --coordinate-mode ground_plane --duration all --csv
```

Run independent ground-truth analytics over the supplied clip:

```powershell
python scripts/analyze_grand_central.py --source ground_truth --coordinate-mode pixel_space --duration all
```

Run the one-minute inference baselines and evaluation:

```powershell
python scripts/run_grand_central_inference.py --duration 1m --inference-interval 1
python scripts/run_grand_central_inference.py --duration 1m --inference-interval 2
python scripts/evaluate_grand_central.py --duration 1m --inference-interval 1 --threshold-pixels 60
python scripts/benchmark_grand_central.py
```

The evaluation reruns YOLO only at annotated timestamps for count and detection-point metrics. It uses ByteTrack output separately for trajectory coverage. Because the ground truth has no boxes and is sampled at 1.25 Hz, precision/recall are explicitly distance-based proxies; IDF1 and HOTA are null.

## Known limitations

- The installed video is only the first 200.1 seconds of a longer annotation corpus.
- Reference duration/frame counts published in different sources are inconsistent; local validation values are authoritative for this installation.
- YOLO26n at 640 px misses many small, distant people in this view; the recorded benchmark must not be interpreted as production accuracy.
- The homography unit and manual zones require independent site calibration before operational use.
- The dataset has no clear commercial-use or redistribution license.
