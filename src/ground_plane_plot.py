from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Optional, Sequence

from geoconverter import GeoConverter


def _safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def frange(start: float, stop: float, step: float) -> list[float]:
    """Inclusive float range with basic safety against infinite loops if step sign is wrong."""
    vals: list[float] = []
    if step == 0:
        return vals

    if (stop - start) * step < 0:
        step = -step

    x = float(start)
    stop = float(stop)
    step = float(step)

    if step > 0:
        while x <= stop:
            vals.append(x)
            x += step
    else:
        while x >= stop:
            vals.append(x)
            x += step
    return vals


def plot_ground_plane(
    *,
    ref_lat: float,
    ref_lon: float,
    drone_positions: Sequence[dict] | None = None,
    targets_d1: Sequence[tuple[float, float]] | None = None,
    targets_d2: Sequence[tuple[float, float]] | None = None,
    targets_fused_average: Sequence[tuple[float, float]] | None = None,
    targets_fused_bearing_intersection: Sequence[tuple[float, float]] | None = None,
    measurements_d1: Sequence[dict] | None = None,
    measurements_d2: Sequence[dict] | None = None,
    # --- plotting switches ---
    draw_bearing_rays: bool = True,
    draw_distance_circles: bool = False,
    ray_length_m: float | None = None,
    ticks: float | None = None,  # interpreted as half-range (meters)
    # --- alignment (rotation) ---
    align_drone1_to_north: bool = False,
    align_mode: str = "target_to_drone1",  # "target_to_drone1" or "drone1_to_target"
    # --- “paper mode” styling ---
    title: str | None = None,
    show_legend: bool = False,
    show_drone_labels: bool = False,
    show_monocular_points: bool = True,
    out_path: Optional[str | Path] = None,
    show: bool = False,
    dpi: int = 300,
    figsize: tuple[float, float] = (3.35, 3.0),  # ~single-column IEEE width
):
    """
    Publication-oriented ground-plane plot.

    Coordinate conventions:
      - Local XY uses: +Y = North, +X = East (via GeoConverter.geo_to_xy).
      - If align_drone1_to_north=True, the entire scene is rotated so that:
          * align_mode="target_to_drone1": vector (origin -> drone1) points to +Y
          * align_mode="drone1_to_target": vector (drone1 -> origin) points to +Y

    ticks semantics (as you requested):
      - ticks=T sets a fixed plot range:
          X in [-T, +T]
          Y in [-0.2*T, +T]
        with 1-meter tick spacing on each axis.
    """

    # Lazy import so non-plot pipeline works without matplotlib.
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "lines.linewidth": 1.0,
        }
    )

    ref_x, ref_y = GeoConverter.geo_to_xy(ref_lat, ref_lon)
    if ref_x is None or ref_y is None:
        raise ValueError("Invalid reference lat/lon for plotting")

    def to_local_xy(lat: float, lon: float) -> Optional[tuple[float, float]]:
        abs_x, abs_y = GeoConverter.geo_to_xy(lat, lon)
        if abs_x is None or abs_y is None:
            return None
        return (float(abs_x) - float(ref_x), float(abs_y) - float(ref_y))

    def bearing_to_unit_vec(bearing_deg: float) -> tuple[float, float]:
        # Convention: 0° north (+y), 90° east (+x)
        b = math.radians(float(bearing_deg))
        return (math.sin(b), math.cos(b))

    def rotate_xy(x: float, y: float, theta_rad: float) -> tuple[float, float]:
        c = math.cos(theta_rad)
        s = math.sin(theta_rad)
        return (x * c - y * s, x * s + y * c)

    # ---- Collect drone positions (unrotated first) ----
    drone_xy_raw: list[tuple[float, float, str]] = []
    for d in (drone_positions or []):
        if not isinstance(d, dict):
            continue
        lat = _safe_float(d.get("lat"))
        lon = _safe_float(d.get("lon"))
        if lat is None or lon is None:
            continue
        xy = to_local_xy(lat, lon)
        if xy is None:
            continue
        drone_xy_raw.append((xy[0], xy[1], str(d.get("label", "UAV"))))

    # ---- Compute rotation so drone1 becomes "0°" (north/up) ----
    theta = 0.0
    if align_drone1_to_north and drone_xy_raw:
        d1 = next((p for p in drone_xy_raw if "1" in p[2]), drone_xy_raw[0])
        d1x, d1y, _ = d1

        # Note: for north-based coords (+Y), angle from +Y is atan2(x, y)
        if align_mode == "target_to_drone1":
            # rotate so (origin -> drone1) points to +Y
            phi = math.atan2(d1x, d1y)
            theta = -phi
        elif align_mode == "drone1_to_target":
            # rotate so (drone1 -> origin) points to +Y => (-d1x, -d1y) to +Y
            phi = math.atan2(-d1x, -d1y)
            theta = -phi

    # Apply rotation to drones
    if theta != 0.0:
        drone_xy: list[tuple[float, float, str]] = []
        for x, y, lbl in drone_xy_raw:
            rx, ry = rotate_xy(x, y, theta)
            drone_xy.append((rx, ry, lbl))
    else:
        drone_xy = drone_xy_raw

    # Convert series to local XY (rotated)
    def series_to_xy(series: Sequence[tuple[float, float]] | None) -> tuple[list[float], list[float]]:
        xs: list[float] = []
        ys: list[float] = []
        for lat, lon in (series or []):
            xy = to_local_xy(float(lat), float(lon))
            if xy is None:
                continue
            x, y = xy
            if theta != 0.0:
                x, y = rotate_xy(x, y, theta)
            xs.append(x)
            ys.append(y)
        return xs, ys

    d1_xs, d1_ys = series_to_xy(targets_d1)
    d2_xs, d2_ys = series_to_xy(targets_d2)
    avg_xs, avg_ys = series_to_xy(targets_fused_average)
    bi_xs, bi_ys = series_to_xy(targets_fused_bearing_intersection)

    fig, ax = plt.subplots(figsize=figsize, dpi=int(dpi))
    if title:
        ax.set_title(title)

    # --- color scheme ---
    drone1_color = "#0072B2"
    drone2_color = "#D55E00"

    # Track all drawn coordinates to set axis limits safely
    all_x: list[float] = []
    all_y: list[float] = []

    # drones: colored triangles
    for (x, y, lbl) in (drone_xy or []):
        color = drone1_color if "1" in lbl else drone2_color
        ax.scatter([x], [y], s=30, marker="^", c=color, zorder=4)
        all_x.append(x)
        all_y.append(y)
        if show_drone_labels:
            ax.annotate(lbl, (x, y), textcoords="offset points", xytext=(4, 4))

    if show_monocular_points:
        if d1_xs:
            ax.scatter(d1_xs, d1_ys, s=14, marker="o", c=drone1_color, zorder=2)
            all_x.extend(d1_xs)
            all_y.extend(d1_ys)
        if d2_xs:
            ax.scatter(d2_xs, d2_ys, s=14, marker="o", c=drone2_color, zorder=2)
            all_x.extend(d2_xs)
            all_y.extend(d2_ys)

    if avg_xs:
        ax.scatter(avg_xs, avg_ys, s=25, marker="*", c="k", zorder=5)
        all_x.extend(avg_xs)
        all_y.extend(avg_ys)

    if bi_xs:
        ax.scatter(bi_xs, bi_ys, s=25, marker="X", c="k", zorder=5)
        all_x.extend(bi_xs)
        all_y.extend(bi_ys)

    # Bearing rays / range circles
    def draw_measurements(measurements: Sequence[dict] | None, *, color: str):
        if not measurements:
            return
        for m in measurements:
            if not isinstance(m, dict):
                continue
            o_lat = _safe_float(m.get("origin_lat"))
            o_lon = _safe_float(m.get("origin_lon"))
            bearing = _safe_float(m.get("bearing_deg"))
            dist = _safe_float(m.get("distance_m"))
            if o_lat is None or o_lon is None:
                continue

            origin_xy = to_local_xy(o_lat, o_lon)
            if origin_xy is None:
                continue
            ox, oy = origin_xy
            if theta != 0.0:
                ox, oy = rotate_xy(ox, oy, theta)

            all_x.append(ox)
            all_y.append(oy)

            if draw_distance_circles and dist is not None and dist > 0:
                circ = mpatches.Circle(
                    (ox, oy),
                    radius=float(dist),
                    fill=False,
                    edgecolor=color,
                    linestyle="-",
                    linewidth=0.9,
                    zorder=1,
                )
                ax.add_patch(circ)
                all_x.extend([ox - dist, ox + dist])
                all_y.extend([oy - dist, oy + dist])

            if draw_bearing_rays and bearing is not None:
                ux, uy = bearing_to_unit_vec(bearing)
                # rotate the direction vector too (so ray matches rotated plot)
                if theta != 0.0:
                    ux, uy = rotate_xy(ux, uy, theta)

                L = (
                    float(ray_length_m)
                    if ray_length_m is not None
                    else (float(dist) if (dist is not None and dist > 0) else 30.0)
                )
                L = max(1.0, L)

                x2, y2 = ox + ux * L, oy + uy * L
                ax.plot([ox, x2], [oy, y2], c=color, linestyle="-", linewidth=1.0, zorder=1)
                all_x.append(x2)
                all_y.append(y2)

    draw_measurements(measurements_d1, color=drone1_color)
    draw_measurements(measurements_d2, color=drone2_color)

    # --- Annotate estimated distances and bearings (original bearings, not rotated) ---
    d1_dist = d1_bearing = d2_dist = d2_bearing = None
    if measurements_d1 and len(measurements_d1) > 0:
        d1_dist = _safe_float(measurements_d1[0].get("distance_m"))
        d1_bearing = _safe_float(measurements_d1[0].get("bearing_deg"))
    if measurements_d2 and len(measurements_d2) > 0:
        d2_dist = _safe_float(measurements_d2[0].get("distance_m"))
        d2_bearing = _safe_float(measurements_d2[0].get("bearing_deg"))

    if (d1_dist is not None and d1_bearing is not None and d2_dist is not None and d2_bearing is not None):
        annotation_text = (
            f"Drone 1: {d1_dist:.2f}m\n"
            f"Drone 2: {d2_dist:.2f}m"
        )
        ax.annotate(
            annotation_text,
            xy=(0.98, 0.02),
            xycoords="axes fraction",
            fontsize=7,
            color="k",
            ha="right",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.5", alpha=0.8),
        )

    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.grid(True, linewidth=0.4, alpha=0.5)
    ax.set_aspect("equal", adjustable="box")

    # --- Axis limits (auto, then overridden by ticks if provided) ---
    if all_x and all_y:
        max_abs = max(max(map(abs, all_x)), max(map(abs, all_y)))
        axis_limit = max(15.0, max_abs * 1.15)
    else:
        axis_limit = 15.0

    ax.set_xlim(-axis_limit, axis_limit)
    ax.set_ylim(-axis_limit, axis_limit)

    # --- Fixed axis limits & ticks (your requested behavior) ---
    if ticks is not None:
        try:
            tick_abs = abs(float(ticks))
            if tick_abs < 1e-9:
                tick_abs = 1.0

            # Fixed axis range: X symmetric, Y asymmetric
            ax.set_xlim(-tick_abs, tick_abs)
            ax.set_ylim(-0.2 * tick_abs, 1.5*tick_abs)

            # 1-meter ticks matching each axis bounds
            x_ticks = list(range(-int(math.floor(tick_abs)), int(math.ceil(tick_abs)) + 1))
            y_min = int(math.floor(-0.2 * tick_abs))
            y_max = int(math.ceil(1.5 * tick_abs))
            y_ticks = list(range(y_min, y_max + 1))

            ax.set_xticks(x_ticks)
            ax.set_yticks(y_ticks)

            axis_limit = tick_abs
        except Exception:
            pass

    # Legend (optional)
    if show_legend:
        from matplotlib.lines import Line2D

        handles = []
        if drone_xy:
            handles.append(Line2D([0], [0], marker="^", linestyle="None", markersize=5, color="k", label="Drones"))
        if show_monocular_points:
            handles.append(Line2D([0], [0], marker="o", linestyle="None", markersize=5, color="drone1_color", label="Drone1 Estimation"))
            handles.append(Line2D([0], [0], marker="o", linestyle="None", markersize=5, color="drone2_color", label="Drone2 Estimation"))
        if avg_xs:
            handles.append(Line2D([0], [0], marker="*", linestyle="None", markersize=5, color="k", label="Fused Average"))
        if bi_xs:
            handles.append(Line2D([0], [0], marker="X", linestyle="None", markersize=5, color="k", label="Fused Bearing intersection"))
        if handles:
            ax.legend(handles=handles, loc="upper right", frameon=True)

    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)

    if show:
        plt.show()

    plt.close(fig)


def build_series_from_frame(
    *,
    drone1_lat: float,
    drone1_lon: float,
    drone2_lat: float,
    drone2_lon: float,
    detections1: Iterable[dict] | None = None,
    detections2: Iterable[dict] | None = None,
    fused_detections: Iterable[dict] | None = None,
):
    drone_positions = [
        {"label": "drone1", "lat": float(drone1_lat), "lon": float(drone1_lon)},
        {"label": "drone2", "lat": float(drone2_lat), "lon": float(drone2_lon)},
    ]

    targets_d1: list[tuple[float, float]] = []
    targets_d2: list[tuple[float, float]] = []
    targets_fused_average: list[tuple[float, float]] = []
    targets_fused_bearing_intersection: list[tuple[float, float]] = []
    measurements_d1: list[dict] = []
    measurements_d2: list[dict] = []

    for det in detections1 or []:
        geo = det.get("person_geoposition") if isinstance(det, dict) else None
        if not (geo and isinstance(geo, dict)):
            continue
        lat = _safe_float(geo.get("latitude"))
        lon = _safe_float(geo.get("longitude"))
        if lat is None or lon is None:
            continue
        targets_d1.append((lat, lon))

        bearing = _safe_float(det.get("bearing_deg", det.get("bearing")))
        dist = _safe_float(det.get("distance_m"))
        if bearing is not None or dist is not None:
            measurements_d1.append(
                {
                    "origin_lat": float(drone1_lat),
                    "origin_lon": float(drone1_lon),
                    "bearing_deg": bearing,
                    "distance_m": dist,
                }
            )

    for det in detections2 or []:
        geo = det.get("person_geoposition") if isinstance(det, dict) else None
        if not (geo and isinstance(geo, dict)):
            continue
        lat = _safe_float(geo.get("latitude"))
        lon = _safe_float(geo.get("longitude"))
        if lat is None or lon is None:
            continue
        targets_d2.append((lat, lon))

        bearing = _safe_float(det.get("bearing_deg", det.get("bearing")))
        dist = _safe_float(det.get("distance_m"))
        if bearing is not None or dist is not None:
            measurements_d2.append(
                {
                    "origin_lat": float(drone2_lat),
                    "origin_lon": float(drone2_lon),
                    "bearing_deg": bearing,
                    "distance_m": dist,
                }
            )

    for det in fused_detections or []:
        if not isinstance(det, dict):
            continue
        geo_avg = det.get("fused_geoposition_average")
        geo_bi = det.get("fused_geoposition_bearing_intersection")
        if geo_avg and isinstance(geo_avg, dict):
            lat = _safe_float(geo_avg.get("latitude"))
            lon = _safe_float(geo_avg.get("longitude"))
            if lat is not None and lon is not None:
                targets_fused_average.append((lat, lon))
        if geo_bi and isinstance(geo_bi, dict):
            lat = _safe_float(geo_bi.get("latitude"))
            lon = _safe_float(geo_bi.get("longitude"))
            if lat is not None and lon is not None:
                targets_fused_bearing_intersection.append((lat, lon))

    series_gps = {
        "drone_positions": drone_positions,
        "targets_d1": targets_d1,
        "targets_d2": targets_d2,
        "targets_fused_average": targets_fused_average,
        "targets_fused_bearing_intersection": targets_fused_bearing_intersection,
        "measurements_d1": measurements_d1,
        "measurements_d2": measurements_d2,
    }
    return {"gps": series_gps}