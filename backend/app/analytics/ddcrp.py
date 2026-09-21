from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Sequence

from backend.app.core.config import AnalyticsConfig
from backend.app.schemas import PopularPath, Trajectory, CommonPath, CommonPathSnapshot


@dataclass(slots=True)
class Tracklet:
    track_id: int
    points: list[tuple[float, float]]

    @property
    def start(self) -> tuple[float, float]:
        return self.points[0]

    @property
    def end(self) -> tuple[float, float]:
        return self.points[-1]


@dataclass(slots=True)
class RepresentativeLine:
    cluster_id: int
    start: tuple[float, float]
    end: tuple[float, float]
    track_count: int


class DDCRPClustering:
    """Distance-Dependent Chinese Restaurant Process (DD-CRP) for trajectory clustering.

    Implements non-parametric Bayesian clustering of pedestrian trajectories following
    Hassanein et al. (Semantic Analysis of Crowded Scenes Based on Non-Parametric Tracklet Clustering).
    """

    def __init__(
        self,
        config: AnalyticsConfig,
        alpha: float | None = None,
        spatial_scale: float | None = None,
        direction_weight: float | None = None,
        stationary_threshold: float | None = None,
    ) -> None:
        updates = {}
        if alpha is not None:
            updates["ddcrp_alpha"] = alpha
        if spatial_scale is not None:
            updates["ddcrp_delta_max"] = spatial_scale
        if stationary_threshold is not None:
            updates["ddcrp_stationary_threshold"] = stationary_threshold
        self.config = config.model_copy(update=updates)
        self.direction_weight = direction_weight
        self._active_tracklets: dict[int, Tracklet] = {}
        self._next_cluster_id = 1
        self._latest_snapshot_paths: tuple[PopularPath, ...] = ()
        self._latest_common_snapshot = CommonPathSnapshot(timestamp=0.0, paths=())

    def observe_all(
        self, trajectories: Sequence[Trajectory], timestamp: float = 0.0
    ) -> None:
        # Extract tracklets of duration `ddcrp_tracklet_seconds`
        self._active_tracklets.clear()
        for t in trajectories:
            if not t.confirmed or len(t.points) < 2:
                continue

            # Take the latest segment of `ddcrp_tracklet_seconds`
            latest_time = t.points[-1].timestamp
            cutoff = latest_time - self.config.ddcrp_tracklet_seconds

            # Find points within the time window
            valid_points = [p for p in t.points if p.timestamp >= cutoff]
            if len(valid_points) < 2:
                # If too few points in the window, take the last 2 points at minimum
                valid_points = t.points[-2:]

            tracklet = Tracklet(
                track_id=t.track_id,
                points=[(p.x, p.y) for p in valid_points]
            )
            self._active_tracklets[t.track_id] = tracklet

        if not self._active_tracklets:
            self._latest_snapshot_paths = ()
            self._latest_common_snapshot = CommonPathSnapshot(timestamp=timestamp, paths=())
            return

        tracklets = list(self._active_tracklets.values())
        stationary = [item for item in tracklets if self._distance(item.start, item.end)
                      <= self.config.ddcrp_stationary_threshold]
        moving = [item for item in tracklets if item not in stationary]

        # Level 1: Cluster parallel tracklets
        l1_clusters = self._run_ddcrp_level_1(moving)

        # Extract representative lines for Level 1 clusters
        rep_lines = self._extract_representative_lines(l1_clusters)

        # Level 2: Cluster continuing representative lines
        l2_clusters = self._run_ddcrp_level_2(rep_lines)

        # Format into PopularPaths
        self._format_paths(l2_clusters, timestamp)
        moving_paths = list(self._latest_snapshot_paths)
        moving_common = list(self._latest_common_snapshot.paths)
        stationary_paths, stationary_common = self._format_stationary(stationary, timestamp)
        all_paths = moving_paths + stationary_paths
        total = sum(path.count for path in all_paths)
        self._latest_snapshot_paths = tuple(
            replace(path, percentage=round(path.count / total * 100.0, 2))
            for path in all_paths
        ) if total else ()
        self._latest_common_snapshot = CommonPathSnapshot(
            timestamp=timestamp, paths=tuple(moving_common + stationary_common)
        )

    def _format_stationary(
        self, tracklets: list[Tracklet], timestamp: float
    ) -> tuple[list[PopularPath], list[CommonPath]]:
        if not tracklets:
            return [], []
        groups: list[list[Tracklet]] = []
        for tracklet in tracklets:
            group = next((items for items in groups if any(
                self._distance(tracklet.start, other.start) <= self.config.ddcrp_delta_max
                for other in items)), None)
            if group is None:
                groups.append([tracklet])
            else:
                group.append(tracklet)
        paths, common = [], []
        for index, group in enumerate(groups, start=1):
            count = len(group)
            x = sum(item.start[0] for item in group) / count
            y = sum(item.start[1] for item in group) / count
            path_id = f"ddcrp-stationary-{index}"
            paths.append(PopularPath(path_id, f"Xếp hàng ({int(x)}, {int(y)})",
                                     count, 0.0, (f"Area-({int(x)},{int(y)})",),
                                     "ddcrp_stationary"))
            common.append(CommonPath(path_id, "scene", "scene", "active", count, count,
                                     1.0, 1.0, "stationary", ((x, y), (x, y)), timestamp))
        return paths, common

    def top_paths(self, limit: int = 5) -> tuple[PopularPath, ...]:
        return tuple(sorted(self._latest_snapshot_paths, key=lambda p: -p.count)[:limit])

    # ---------------------------------------------------------
    # Similarity Metrics (Equations 1-6)
    # ---------------------------------------------------------

    def _distance(self, p1: tuple[float, float], p2: tuple[float, float]) -> float:
        return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

    def _hausdorff_distance(self, t1: Tracklet, t2: Tracklet) -> float:
        # Equation 1 & 2
        # Optimization: A tracklet of 1.5s is roughly linear. We sample 3 points (start, mid, end)
        pts1 = (t1.points[0], t1.points[len(t1.points)//2], t1.points[-1])
        pts2 = (t2.points[0], t2.points[len(t2.points)//2], t2.points[-1])

        d1 = max(min(self._distance(p1, p2) for p2 in pts2) for p1 in pts1)
        d2 = max(min(self._distance(p2, p1) for p1 in pts1) for p2 in pts2)
        return max(d1, d2)

    def _min_point_distance(self, t1: RepresentativeLine, t2: RepresentativeLine) -> float:
        # Equation 3
        # Representative lines only have start and end points for this calculation
        pts1 = [t1.start, t1.end]
        pts2 = [t2.start, t2.end]
        return min(self._distance(p1, p2) for p1 in pts1 for p2 in pts2)

    def _overlap_ratio(self, start1: tuple[float, float], end1: tuple[float, float],
                             start2: tuple[float, float], end2: tuple[float, float]) -> float:
        m_start = ((start1[0] + start2[0])/2, (start1[1] + start2[1])/2)
        m_end = ((end1[0] + end2[0])/2, (end1[1] + end2[1])/2)

        v_line = (m_end[0] - m_start[0], m_end[1] - m_start[1])
        v_len = math.hypot(v_line[0], v_line[1])
        if v_len == 0: return 1.0

        v_unit = (v_line[0]/v_len, v_line[1]/v_len)

        def project(p: tuple[float, float]) -> float:
            return (p[0] - m_start[0])*v_unit[0] + (p[1] - m_start[1])*v_unit[1]

        proj_t1 = (project(start1), project(end1))
        proj_t2 = (project(start2), project(end2))

        min_t1, max_t1 = min(proj_t1), max(proj_t1)
        min_t2, max_t2 = min(proj_t2), max(proj_t2)

        overlap_start = max(min_t1, min_t2)
        overlap_end = min(max_t1, max_t2)
        overlap_len = max(0.0, overlap_end - overlap_start)

        union_len = max(max_t1, max_t2) - min(min_t1, min_t2)
        if union_len == 0: return 1.0

        return overlap_len / union_len

    def _angle_between(self, start1: tuple[float, float], end1: tuple[float, float],
                             start2: tuple[float, float], end2: tuple[float, float]) -> float:
        v1 = (end1[0] - start1[0], end1[1] - start1[1])
        v2 = (end2[0] - start2[0], end2[1] - start2[1])
        a1 = math.atan2(v1[1], v1[0])
        a2 = math.atan2(v2[1], v2[0])
        diff = abs(a1 - a2)
        return min(diff, 2*math.pi - diff)

    def _similarity(self,
                    start1: tuple[float, float], end1: tuple[float, float],
                    start2: tuple[float, float], end2: tuple[float, float],
                    delta: float) -> float:
        # Equation 4, 5, 6
        o_ij = self._overlap_ratio(start1, end1, start2, end2)

        theta_min = self.config.ddcrp_theta_min
        theta_max = self.config.ddcrp_theta_max
        delta_min = self.config.ddcrp_delta_min
        delta_max = self.config.ddcrp_delta_max

        sigma_theta = theta_max + o_ij * (theta_min - theta_max)
        sigma_delta = delta_min + o_ij * (delta_max - delta_min)

        theta = self._angle_between(start1, end1, start2, end2)

        sim_theta = math.exp(- (theta / max(1e-5, sigma_theta))**2)
        sim_delta = math.exp(- (delta / max(1e-5, sigma_delta))**2)

        return sim_theta * sim_delta

    # ---------------------------------------------------------
    # Level 1: Cluster parallel tracklets
    # ---------------------------------------------------------

    def _run_ddcrp_level_1(self, tracklets: list[Tracklet]) -> dict[int, list[Tracklet]]:
        assignments: dict[int, int] = {}
        delta_max = self.config.ddcrp_delta_max

        # Spatial Hashing O(N)
        grid_size = delta_max * 2.0
        grid: dict[tuple[int, int], list[int]] = defaultdict(list)
        for i, t in enumerate(tracklets):
            cx = int(t.start[0] // grid_size)
            cy = int(t.start[1] // grid_size)
            grid[(cx, cy)].append(i)

        for i, t_i in enumerate(tracklets):
            best_link = i
            best_prob = self.config.ddcrp_alpha

            cx = int(t_i.start[0] // grid_size)
            cy = int(t_i.start[1] // grid_size)

            candidates = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    candidates.extend(grid.get((cx+dx, cy+dy), []))

            for j in candidates:
                if i == j:
                    continue

                t_j = tracklets[j]

                # Spatial gating: if tracklets are too far apart, skip immediately
                if abs(t_i.start[0] - t_j.start[0]) > delta_max * 2.0:
                    continue
                if abs(t_i.start[1] - t_j.start[1]) > delta_max * 2.0:
                    continue

                delta = self._hausdorff_distance(t_i, t_j)

                # If Hausdorff distance is beyond limit, similarity will be near 0
                if delta > delta_max:
                    continue

                sim = self._similarity(t_i.start, t_i.end, t_j.start, t_j.end, delta)

                # Likelihood is proportional to similarity
                if sim > best_prob:
                    best_prob = sim
                    best_link = j

            assignments[i] = best_link

        return self._build_clusters_from_links(assignments, tracklets)

    # ---------------------------------------------------------
    # Extract Representative Lines
    # ---------------------------------------------------------

    def _extract_representative_lines(self, clusters: dict[int, list[Tracklet]]) -> list[RepresentativeLine]:
        lines: list[RepresentativeLine] = []
        for cid, group in clusters.items():
            if not group:
                continue

            # Average orientation and center of mass
            avg_dx = sum((t.end[0] - t.start[0]) for t in group) / len(group)
            avg_dy = sum((t.end[1] - t.start[1]) for t in group) / len(group)

            com_x = sum(p[0] for t in group for p in t.points) / sum(len(t.points) for t in group)
            com_y = sum(p[1] for t in group for p in t.points) / sum(len(t.points) for t in group)

            # Find terminal points by projecting all points onto the average line passing through COM
            v_len = math.hypot(avg_dx, avg_dy)
            if v_len == 0:
                continue

            v_unit = (avg_dx/v_len, avg_dy/v_len)

            def project(p: tuple[float, float]) -> float:
                return (p[0] - com_x)*v_unit[0] + (p[1] - com_y)*v_unit[1]

            projections = []
            for t in group:
                for p in (t.start, t.end):
                    projections.append((project(p), p))

            if not projections:
                continue

            projections.sort(key=lambda x: x[0])

            start_proj = projections[0][0]
            end_proj = projections[-1][0]

            start_pt = (com_x + start_proj * v_unit[0], com_y + start_proj * v_unit[1])
            end_pt = (com_x + end_proj * v_unit[0], com_y + end_proj * v_unit[1])

            lines.append(RepresentativeLine(
                cluster_id=cid,
                start=start_pt,
                end=end_pt,
                track_count=len(group)
            ))

        return lines

    # ---------------------------------------------------------
    # Level 2: Cluster continuous lines
    # ---------------------------------------------------------

    def _run_ddcrp_level_2(self, lines: list[RepresentativeLine]) -> dict[int, list[RepresentativeLine]]:
        assignments: dict[int, int] = {}
        delta_max = self.config.ddcrp_delta_max

        # Spatial Hashing O(N)
        grid_size = delta_max * 2.0
        grid: dict[tuple[int, int], list[int]] = defaultdict(list)
        for i, l in enumerate(lines):
            # Use midpoint of the representative line for hashing
            cx = int(((l.start[0] + l.end[0]) / 2.0) // grid_size)
            cy = int(((l.start[1] + l.end[1]) / 2.0) // grid_size)
            grid[(cx, cy)].append(i)

        for i, l_i in enumerate(lines):
            best_link = i
            best_prob = self.config.ddcrp_alpha

            cx = int(((l_i.start[0] + l_i.end[0]) / 2.0) // grid_size)
            cy = int(((l_i.start[1] + l_i.end[1]) / 2.0) // grid_size)

            candidates = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    candidates.extend(grid.get((cx+dx, cy+dy), []))

            for j in candidates:
                if i == j:
                    continue

                l_j = lines[j]

                # Spatial gating using bounding boxes of the lines
                min_x_i, max_x_i = min(l_i.start[0], l_i.end[0]), max(l_i.start[0], l_i.end[0])
                min_y_i, max_y_i = min(l_i.start[1], l_i.end[1]), max(l_i.start[1], l_i.end[1])
                min_x_j, max_x_j = min(l_j.start[0], l_j.end[0]), max(l_j.start[0], l_j.end[0])
                min_y_j, max_y_j = min(l_j.start[1], l_j.end[1]), max(l_j.start[1], l_j.end[1])

                # If they are strictly non-overlapping with a margin of delta_max, skip
                if max_x_i < min_x_j - delta_max or min_x_i > max_x_j + delta_max:
                    continue
                if max_y_i < min_y_j - delta_max or min_y_i > max_y_j + delta_max:
                    continue

                delta = self._min_point_distance(l_i, l_j)
                if delta > delta_max:
                    continue

                sim = self._similarity(l_i.start, l_i.end, l_j.start, l_j.end, delta)

                if sim > best_prob:
                    best_prob = sim
                    best_link = j

            assignments[i] = best_link

        return self._build_clusters_from_links(assignments, lines)

    # ---------------------------------------------------------
    # Utility
    # ---------------------------------------------------------

    def _build_clusters_from_links(self, assignments: dict[int, int], items: list) -> dict[int, list]:
        # assignments maps item index -> linked item index
        # We need to find connected components (clusters)

        parent = {i: i for i in assignments}
        def find(i: int) -> int:
            if parent[i] != i:
                parent[i] = find(parent[i])
            return parent[i]

        def union(i: int, j: int) -> None:
            root_i = find(i)
            root_j = find(j)
            if root_i != root_j:
                parent[root_i] = root_j

        for i, link in assignments.items():
            union(i, link)

        clusters: dict[int, list] = defaultdict(list)
        for i, item in enumerate(items):
            root = find(i)
            clusters[root].append(item)

        return dict(clusters)

    def _format_paths(self, l2_clusters: dict[int, list[RepresentativeLine]], timestamp: float = 0.0) -> tuple[PopularPath, ...]:
        paths: list[PopularPath] = []
        common_paths: list[CommonPath] = []
        total_tracks = sum(sum(l.track_count for l in group) for group in l2_clusters.values())
        if total_tracks == 0:
            self._latest_snapshot_paths = ()
            self._latest_common_snapshot = CommonPathSnapshot(timestamp=timestamp, paths=())
            return ()

        for cid, group in l2_clusters.items():
            if not group:
                continue

            count = sum(l.track_count for l in group)
            pct = round(count / total_tracks * 100.0, 2)

            avg_dx = sum((l.end[0] - l.start[0]) for l in group) / len(group)
            avg_dy = sum((l.end[1] - l.start[1]) for l in group) / len(group)
            v_len = math.hypot(avg_dx, avg_dy)
            if v_len == 0:
                continue
            v_unit = (avg_dx/v_len, avg_dy/v_len)

            def project(p: tuple[float, float]) -> float:
                return p[0]*v_unit[0] + p[1]*v_unit[1]

            pts = []
            for l in group:
                pts.append((project(l.start), l.start))
                pts.append((project(l.end), l.end))

            pts.sort(key=lambda x: x[0])
            source = pts[0][1]
            sink = pts[-1][1]

            label = f"Pathway: ({int(source[0])}, {int(source[1])}) → ({int(sink[0])}, {int(sink[1])})"

            paths.append(
                PopularPath(
                    path_id=f"ddcrp-path-{cid}",
                    label=label,
                    count=count,
                    percentage=pct,
                    regions=(f"Source-({int(source[0])},{int(source[1])})", f"Sink-({int(sink[0])},{int(sink[1])})"),
                    kind="ddcrp",
                )
            )

            polyline = tuple(p[1] for p in pts)

            common_paths.append(
                CommonPath(
                    path_id=f"ddcrp-path-{cid}",
                    origin_zone="scene",
                    destination_zone="scene",
                    state="active",
                    unique_tracks_short=count,
                    unique_tracks_long=count,
                    score=1.0,
                    confidence=1.0,
                    direction="ddcrp",
                    polyline=polyline,
                    updated_at=timestamp,
                )
            )

        self._latest_snapshot_paths = tuple(paths)
        self._latest_common_snapshot = CommonPathSnapshot(timestamp=timestamp, paths=tuple(common_paths))
        return self._latest_snapshot_paths

    def snapshot(self) -> CommonPathSnapshot:
        if not hasattr(self, "_latest_common_snapshot"):
            return CommonPathSnapshot(timestamp=0.0, paths=())
        return self._latest_common_snapshot
