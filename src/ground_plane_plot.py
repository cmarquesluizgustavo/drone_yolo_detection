from __future__ import annotations
from pathlib import Path
import math
from typing import Iterable, Optional, Sequence

from geoconverter import GeoConverter


def _safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def plot_ground_plane(
    *,
    ref_lat: float,
    ref_lon: float,
    drone_positions: Sequence[dict] | None = None,
    targets_d1: Sequence[tuple[float, float]] | None = None,
    targets_d2: Sequence[tuple[float, float]] | None = None,
    targets_fused: Sequence[tuple[float, float]] | None = None,
    measurements_d1: Sequence[dict] | None = None,
    measurements_d2: Sequence[dict] | None = None,
    draw_bearing_rays: bool = True,
    draw_distance_circles: bool = True,
    ray_length_m: float | None = None,
    title: str = "Ground plane (local XY meters)",
    out_path: Optional[str | Path] = None,
    show: bool = False,
    dpi: int = 140,
):
    """Plot a 2D ground-plane visualization (local X/Y meters).

    Inputs are plain lat/lon lists:
    - drone_positions: list of {'label': str, 'lat': float, 'lon': float}
    - targets_d1/targets_d2/targets_fused: list of (lat, lon)

    Local XY is computed as UTM(XY(lat,lon)) - UTM(XY(ref_lat,ref_lon)).
    """

    # Import lazily so the rest of the pipeline works without plotting.
    import matplotlib

    # Ensure headless-safe backend when running in Docker/servers.
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    ref_x, ref_y = GeoConverter.geo_to_xy(ref_lat, ref_lon)
    if ref_x is None or ref_y is None:
        raise ValueError("Invalid reference lat/lon for plotting")

    def to_local_xy(lat: float, lon: float):
        abs_x, abs_y = GeoConverter.geo_to_xy(lat, lon)
        if abs_x is None or abs_y is None:
            return None
        return (float(abs_x) - float(ref_x), float(abs_y) - float(ref_y))

    def bearing_to_unit_vec(bearing_deg: float):
        # Must match GeoConverter.polar_to_xy convention: 0°=north(+y), 90°=east(+x)
        b = math.radians(float(bearing_deg))
        return (math.sin(b), math.cos(b))

    drone_xs: list[float] = []
    drone_ys: list[float] = []
    drone_labels: list[str] = []

    d1_xs: list[float] = []
    d1_ys: list[float] = []
    d2_xs: list[float] = []
    d2_ys: list[float] = []
    fused_xs: list[float] = []
    fused_ys: list[float] = []

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
        x, y = xy
        drone_xs.append(x)
        drone_ys.append(y)
        drone_labels.append(str(d.get("label", "drone")))

    for lat, lon in (targets_d1 or []):
        xy = to_local_xy(float(lat), float(lon))
        if xy is None:
            continue
        d1_xs.append(xy[0])
        d1_ys.append(xy[1])

    for lat, lon in (targets_d2 or []):
        xy = to_local_xy(float(lat), float(lon))
        if xy is None:
            continue
        d2_xs.append(xy[0])
        d2_ys.append(xy[1])

    for lat, lon in (targets_fused or []):
        xy = to_local_xy(float(lat), float(lon))
        if xy is None:
            continue
        fused_xs.append(xy[0])
        fused_ys.append(xy[1])

    import re
    fig, ax = plt.subplots(figsize=(7.5, 7.5), dpi=int(dpi))
    # Clean up the title: remove (GPS) and extra info, keep only main part
    clean_title = title
    # Remove (GPS) and similar
    clean_title = re.sub(r"\s*\(.*?\)", "", clean_title)
    ax.set_title(clean_title.strip())

    if d1_xs:
        ax.scatter(d1_xs, d1_ys, s=14, alpha=0.8, marker="o", label="targets (drone1)")
    if d2_xs:
        ax.scatter(d2_xs, d2_ys, s=14, alpha=0.8, marker="s", label="targets (drone2)")
    if fused_xs:
        ax.scatter(fused_xs, fused_ys, s=30, alpha=0.9, marker="*", label="targets (fused)")

    if drone_xs:
        ax.scatter(drone_xs, drone_ys, s=70, marker="^", label="drones")
        for x, y, lbl in zip(drone_xs, drone_ys, drone_labels):
            ax.annotate(lbl, (x, y), textcoords="offset points", xytext=(6, 6), fontsize=9)

    # Bearing rays + distance circles (from per-drone measurements)
    def draw_measurements(measurements: Sequence[dict] | None, *, color: str, name: str):
        if not measurements:
            return

        ray_labeled = False
        circle_labeled = False

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

            if draw_distance_circles and dist is not None and dist > 0:
                lbl = f"range ({name})" if not circle_labeled else None
                circle_labeled = True
                circ = mpatches.Circle(
                    (ox, oy),
                    radius=float(dist),
                    fill=False,
                    linewidth=1.2,
                    alpha=0.55,
                    edgecolor=color,
                    label=lbl,
                )
                ax.add_patch(circ)

            if draw_bearing_rays and bearing is not None:
                dx, dy = bearing_to_unit_vec(bearing)
                L = float(ray_length_m) if ray_length_m is not None else (float(dist) if (dist is not None and dist > 0) else 30.0)
                L = max(1.0, L)
                lbl = f"bearing ({name})" if not ray_labeled else None
                ray_labeled = True
                ax.plot([ox, ox + dx * L], [oy, oy + dy * L], color=color, linewidth=1.6, alpha=0.7, label=lbl)

    draw_measurements(measurements_d1, color="#1f77b4", name="drone1")
    draw_measurements(measurements_d2, color="#ff7f0e", name="drone2")

    ax.set_xlabel("X (m, east +)")
    ax.set_ylabel("Y (m, north +)")
    ax.grid(True, linewidth=0.6, alpha=0.5)
    ax.set_aspect("equal", adjustable="box")

    series_count = len(drone_xs) + len(d1_xs) + len(d2_xs) + len(fused_xs)
    if series_count > 1:
        # Place legend outside the plot area on the right
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0., fontsize=10)

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax


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
    """Build plot series using GPS lat/lon drone positions.

    Target points come from detection-provided geopositions (person_geoposition / fused_geoposition).
    """

    drone_positions = [
        {"label": "drone1", "lat": float(drone1_lat), "lon": float(drone1_lon)},
        {"label": "drone2", "lat": float(drone2_lat), "lon": float(drone2_lon)},
    ]

    targets_d1: list[tuple[float, float]] = []
    targets_d2: list[tuple[float, float]] = []
    targets_fused: list[tuple[float, float]] = []
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
            measurements_d1.append({
                "origin_lat": float(drone1_lat),
                "origin_lon": float(drone1_lon),
                "bearing_deg": bearing,
                "distance_m": dist,
            })

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
            measurements_d2.append({
                "origin_lat": float(drone2_lat),
                "origin_lon": float(drone2_lon),
                "bearing_deg": bearing,
                "distance_m": dist,
            })

    for det in fused_detections or []:
        geo = det.get("fused_geoposition") if isinstance(det, dict) else None
        if not (geo and isinstance(geo, dict)):
            continue
        lat = _safe_float(geo.get("latitude"))
        lon = _safe_float(geo.get("longitude"))
        if lat is None or lon is None:
            continue
        targets_fused.append((lat, lon))

    series_gps = {
        "drone_positions": drone_positions,
        "targets_d1": targets_d1,
        "targets_d2": targets_d2,
        "targets_fused": targets_fused,
        "measurements_d1": measurements_d1,
        "measurements_d2": measurements_d2,
    }
    return {
        "gps": series_gps,
    }
