from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from backend.app.core.config import AnalyticsConfig
from backend.app.schemas import PopularPath, TrackPoint, Trajectory


@dataclass(slots=True)
class TrackletFeature:
    track_id: int
    points: list[tuple[float, float]]
    start_point: tuple[float, float]
    end_point: tuple[float, float]
    mean_point: tuple[float, float]
    displacement: float
    vector: tuple[float, float]
    direction_rad: float
    dwell_seconds: float
    is_stationary: bool


class DDCRPClustering:
    """Distance-Dependent Chinese Restaurant Process (DD-CRP) for trajectory clustering.
    
    Implements non-parametric Bayesian clustering of pedestrian trajectories following
    Zhou et al. (Semantic Analysis of Crowded Scenes Based on Non-Parametric Tracklet Clustering).
    Discovers natural pathways and clusters stationary pedestrians (e.g. buying tickets, queues).
    """

    def __init__(
        self,
        config: AnalyticsConfig,
        alpha: float = 0.4,
        spatial_scale: float = 75.0,
        direction_weight: float = 0.6,
        stationary_threshold: float = 18.0,
    ) -> None:
        self.config = config
        self.alpha = alpha
        self.spatial_scale = spatial_scale
        self.direction_weight = direction_weight
        self.stationary_threshold = stationary_threshold
        self._tracklets: dict[int, TrackletFeature] = {}
        self._assignments: dict[int, int] = {}  # track_id -> cluster_id
        self._clusters: dict[int, list[int]] = defaultdict(list)
        self._next_cluster_id = 1

    def observe(self, trajectory: Trajectory) -> None:
        if not trajectory.confirmed or len(trajectory.points) < self.config.min_confirmed_points:
            return

        feature = self._extract_feature(trajectory)
        self._tracklets[trajectory.track_id] = feature
        self._cluster_tracklet(feature)

    def observe_all(self, trajectories: Sequence[Trajectory]) -> None:
        for trajectory in trajectories:
            self.observe(trajectory)

    def _extract_feature(self, trajectory: Trajectory) -> TrackletFeature:
        coords = [(p.x, p.y) for p in trajectory.points]
        start = coords[0]
        end = coords[-1]
        mean_x = sum(c[0] for c in coords) / len(coords)
        mean_y = sum(c[1] for c in coords) / len(coords)
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        disp = math.hypot(dx, dy)
        dwell = trajectory.points[-1].timestamp - trajectory.points[0].timestamp
        direction = math.atan2(dy, dx)
        is_stat = disp < self.stationary_threshold

        return TrackletFeature(
            track_id=trajectory.track_id,
            points=coords,
            start_point=start,
            end_point=end,
            mean_point=(mean_x, mean_y),
            displacement=disp,
            vector=(dx, dy),
            direction_rad=direction,
            dwell_seconds=max(0.1, dwell),
            is_stationary=is_stat,
        )

    def _distance(self, f1: TrackletFeature, f2: TrackletFeature) -> float:
        d_mean = math.hypot(f1.mean_point[0] - f2.mean_point[0], f1.mean_point[1] - f2.mean_point[1])
        d_start = math.hypot(f1.start_point[0] - f2.start_point[0], f1.start_point[1] - f2.start_point[1])
        d_end = math.hypot(f1.end_point[0] - f2.end_point[0], f1.end_point[1] - f2.end_point[1])
        spatial_dist = 0.5 * d_mean + 0.25 * (d_start + d_end)

        # Angular/Direction distance
        if f1.is_stationary and f2.is_stationary:
            # Both are stationary: only spatial distance matters
            angular_dist = 0.0
        elif f1.is_stationary != f2.is_stationary:
            # One moving, one stationary: penalty
            angular_dist = 2.0
        else:
            diff = abs(f1.direction_rad - f2.direction_rad)
            diff = min(diff, 2.0 * math.pi - diff)
            angular_dist = 1.0 - math.cos(diff)

        return spatial_dist + self.direction_weight * self.spatial_scale * angular_dist

    def _cluster_tracklet(self, target: TrackletFeature) -> None:
        if target.track_id in self._assignments:
            old_cluster = self._assignments[target.track_id]
            if target.track_id in self._clusters[old_cluster]:
                self._clusters[old_cluster].remove(target.track_id)

        # DD-CRP linkage: calculate decay f(d_ij) = exp(-d_ij / sigma)
        best_link = None
        best_prob = self.alpha  # Self-link probability threshold

        for track_id, other in self._tracklets.items():
            if track_id == target.track_id:
                continue
            dist = self._distance(target, other)
            prob = math.exp(-dist / self.spatial_scale)
            if prob > best_prob:
                best_prob = prob
                best_link = track_id

        if best_link is not None and best_link in self._assignments:
            cluster_id = self._assignments[best_link]
        else:
            cluster_id = self._next_cluster_id
            self._next_cluster_id += 1

        self._assignments[target.track_id] = cluster_id
        self._clusters[cluster_id].append(target.track_id)

    def top_paths(self, limit: int = 5) -> tuple[PopularPath, ...]:
        active_clusters = [
            (cid, members)
            for cid, members in self._clusters.items()
            if len(members) > 0
        ]
        if not active_clusters:
            return ()

        total_tracks = sum(len(m) for _, m in active_clusters)
        sorted_clusters = sorted(active_clusters, key=lambda x: -len(x[1]))[:limit]

        paths: list[PopularPath] = []
        for rank, (cluster_id, members) in enumerate(sorted_clusters, start=1):
            feats = [self._tracklets[tid] for tid in members if tid in self._tracklets]
            if not feats:
                continue

            count = len(feats)
            pct = round(count / total_tracks * 100.0, 2) if total_tracks else 0.0

            # Check if predominantly stationary (e.g. Ticket Booth Area)
            stationary_count = sum(1 for f in feats if f.is_stationary)
            is_predominantly_stationary = stationary_count / count >= 0.5

            if is_predominantly_stationary:
                avg_x = sum(f.mean_point[0] for f in feats) / count
                avg_y = sum(f.mean_point[1] for f in feats) / count
                label = f"Quầy vé / Xếp hàng ({int(avg_x)}, {int(avg_y)}) - {count} người"
                regions = (f"TicketQueue-({int(avg_x)},{int(avg_y)})",)
                kind = "ddcrp_stationary"
            else:
                avg_start_x = sum(f.start_point[0] for f in feats) / count
                avg_start_y = sum(f.start_point[1] for f in feats) / count
                avg_end_x = sum(f.end_point[0] for f in feats) / count
                avg_end_y = sum(f.end_point[1] for f in feats) / count
                label = f"Pathway #{rank}: ({int(avg_start_x)}, {int(avg_start_y)}) → ({int(avg_end_x)}, {int(avg_end_y)})"
                regions = (f"Origin-({int(avg_start_x)},{int(avg_start_y)})", f"Dest-({int(avg_end_x)},{int(avg_end_y)})")
                kind = "ddcrp"

            paths.append(
                PopularPath(
                    path_id=f"ddcrp-{cluster_id}",
                    label=label,
                    count=count,
                    percentage=pct,
                    regions=regions,
                    kind=kind,
                )
            )

        return tuple(paths)
