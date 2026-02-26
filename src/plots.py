from __future__ import annotations

import math
from pathlib import Path
from geoconverter import GeoConverter
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def ray_distance_to_target(ox, oy, ux, uy, targets):
    best_target = None

    for tx, ty in targets:
        vx = tx - ox
        vy = ty - oy
        t = vx * ux + vy * uy # projecao de v em t

        if best_target is None or t < best_target:
            best_target = t

    return best_target


def plot_ground_plane(
    ref_lat,
    ref_lon,
    drone_positions,
    targets_1,
    targets_2,
    targets_fused_avg,
    targets_fused_bi,
    targets_fused_wavg,
    #targets_fused_wbi,
    measurements_1,
    measurements_2,
    distance_info,
    ticks_lim,
    out_path,
    draw_bearing_rays: bool = True,
    draw_distance_circles: bool = False,
):
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

    def to_local_xy(lat, lon):
        abs_x, abs_y = GeoConverter.geo_to_xy(lat, lon)
        return (abs_x - ref_x, abs_y - ref_y)

    def bearing_to_unit_vec(bearing_deg):
        # 0° NORTE (+y), 90° LESTE (+x)
        b = math.radians(float(bearing_deg))
        return (math.sin(b), math.cos(b))

    drone_xy_raw: list[tuple[float, float, str]] = []
    for d in (drone_positions or []):
        if not isinstance(d, dict):
            continue
        lat = d.get("lat")
        lon = d.get("lon")
        if lat is None or lon is None:
            continue
        xy = to_local_xy(lat, lon)
        if xy is None:
            continue
        drone_xy_raw.append((xy[0], xy[1], str(d.get("label", "UAV"))))
    else:
        drone_xy = drone_xy_raw

    def series_to_xy(series):
        xs: list[float] = []
        ys: list[float] = []
        for lat, lon in (series or []):
            xy = to_local_xy(float(lat), float(lon))
            if xy is None:
                continue
            x, y = xy
            xs.append(x)
            ys.append(y)
        return xs, ys

    d1_xs, d1_ys = series_to_xy(targets_1)
    d2_xs, d2_ys = series_to_xy(targets_2)
    avg_xs, avg_ys = series_to_xy(targets_fused_avg)
    bi_xs, bi_ys = series_to_xy(targets_fused_bi)
    bi_targets_xy = list(zip(bi_xs, bi_ys))

    wavg_xs, wavg_ys = series_to_xy(targets_fused_wavg)
    #wbi_xs, wbi_ys = series_to_xy(targets_fused_wbi)


    fig, ax = plt.subplots(figsize=(3.35, 3.0), dpi=300)

    drone1_color = "#0072B2"
    drone2_color = "#D55E00"

    # Track all drawn coordinates to set axis limits safely
    all_x = []
    all_y = []

    # drones: colored triangles
    for (x, y, lbl) in (drone_xy or []):
        color = drone1_color if "1" in lbl else drone2_color
        ax.scatter([x], [y], s=30, marker="^", c=color, zorder=4)
        all_x.append(x)
        all_y.append(y)

        ax.scatter(d1_xs, d1_ys, s=14, marker="o", c=drone1_color, zorder=2)
        all_x.extend(d1_xs)
        all_y.extend(d1_ys)

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

    if wavg_xs:
        ax.scatter(wavg_xs, wavg_ys, s=25, marker="*", c="m", zorder=5)
        all_x.extend(wavg_xs)
        all_y.extend(wavg_ys)

    # if wbi_xs:
    #     ax.scatter(wbi_xs, wbi_ys, s=25, marker="X", c="m", zorder=5)
    #     all_x.extend(wbi_xs)
    #     all_y.extend(wbi_ys)

    # Bearing rays / range circles
    def draw_measurements(measurements, color, intersection_targets):
        if not measurements:
            return
        for m in measurements:
            if not isinstance(m, dict):
                continue
            o_lat = m.get("origin_lat")
            o_lon = m.get("origin_lon")
            bearing = m.get("bearing_deg")
            dist = m.get("distance_m")

            origin_xy = to_local_xy(o_lat, o_lon)
            ox, oy = origin_xy

            all_x.append(ox)
            all_y.append(oy)

            if draw_distance_circles:
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

            if draw_bearing_rays:
                ux, uy = bearing_to_unit_vec(bearing)

                x1, y1 = ox + ux * dist, oy + uy * dist
                ax.plot([ox, x1], [oy, y1], c=color, linewidth=1.0, zorder=2)

                all_x.append(x1)
                all_y.append(y1)

                # ---- dashed continuation: to bearing intersection ----
                dashed = ray_distance_to_target(ox, oy, ux, uy, intersection_targets)

                if dashed is not None:
                    x2, y2 = ox + ux * dashed, oy + uy * dashed
                    ax.plot(
                        [x1, x2],
                        [y1, y2],
                        c=color,
                        linewidth=0.9,
                        linestyle="--",
                        zorder=1,
                    )

                    all_x.append(x2)
                    all_y.append(y2)



    draw_measurements(
        measurements_1,
        color=drone1_color,
        intersection_targets=bi_targets_xy,
    )

    draw_measurements(
        measurements_2,
        color=drone2_color,
        intersection_targets=bi_targets_xy,
    )

    # --- Annotate estimated distances  ---
    d1_pitch = distance_info.get("d1_pitch")
    d1_pinhole = distance_info.get("d1_pinhole")
    d1_fused = distance_info.get("d1_fused")

    d2_pitch = distance_info.get("d2_pitch")
    d2_pinhole = distance_info.get("d2_pinhole")
    d2_fused = distance_info.get("d2_fused")

    fused_avg_d1 = distance_info.get("fused_avg_d1")
    fused_bi_d1  = distance_info.get("fused_bi_d1")

    fused_avg_d2 = distance_info.get("fused_avg_d2")
    fused_bi_d2  = distance_info.get("fused_bi_d2")

    fused_wavg_d1 = distance_info.get("fused_weighted_avg_d1")
    fused_wbi_d1  = distance_info.get("fused_weighted_bearing_intersection_d1")

    fused_wavg_d2 = distance_info.get("fused_weighted_avg_d2")
    fused_wbi_d2  = distance_info.get("fused_weighted_bearing_intersection_d2")

    def fmt(v):
        return f"{v:.2f} m" if v is not None else "—"

    x_center = 0.5
    y_base = -0.05      # closer to plot
    row = 0.03           # tight vertical spacing
    gap = 0.010           # gap between drones
    fs_text  = 7

    fig.text(
        x_center, y_base + 3*row,
        f"UAV1 pitch dist.: {fmt(d1_pitch)} | UAV1 pinhole dist.: {fmt(d1_pinhole)} | UAV1 fused dist.: {fmt(d1_fused)}",
        fontsize=fs_text,
        ha="center",
        va="top",
        color=drone1_color,
    )

    fig.text(
        x_center, y_base + 2*row,
        f"Fused avg: {fmt(fused_avg_d1)}  |  Fused int: {fmt(fused_bi_d1)} | Fused w-avg: {fmt(fused_wavg_d1)}",#  |  Fused w-int: {fmt(fused_wbi_d1)}",
        fontsize=fs_text,
        ha="center",
        va="top",
        color=drone1_color,
    )

    # -------- Drone 2 --------
    y2 = y_base + row - gap

    fig.text(
        x_center, y2 - row,
        f"UAV2 pitch dist.: {fmt(d2_pitch)} | UAV2 pinhole dist.: {fmt(d2_pinhole)} | UAV2 fused dist.: {fmt(d2_fused)}",
        fontsize=fs_text,
        ha="center",
        va="top",
        color=drone2_color,
    )

    fig.text(
        x_center, y2 - 2*row,
        f"Fused avg: {fmt(fused_avg_d2)}  |  Fused int: {fmt(fused_bi_d2)} | Fused w-avg: {fmt(fused_wavg_d2)}",#  |  Fused w-int: {fmt(fused_wbi_d2)}",
        fontsize=fs_text,
        ha="center",
        va="top",
        color=drone2_color,
    )

    fig.subplots_adjust(bottom=0.2)


    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.grid(True, linewidth=0.4, alpha=0.5)
    ax.set_aspect("equal", adjustable="box")

    from matplotlib.ticker import MultipleLocator, FixedLocator

    # Axis limits
    ax.set_xlim(-ticks_lim, ticks_lim)
    ax.set_ylim(int(-0.5*ticks_lim), int(1.5*ticks_lim))

    # ---- Major ticks: only -tick, 0, +tick ----
    ax.xaxis.set_major_locator(FixedLocator([-ticks_lim, 0.0, ticks_lim]))
    ax.yaxis.set_major_locator(FixedLocator([-ticks_lim, 0.0, ticks_lim, int(1.5*ticks_lim)]))

    # ---- Minor ticks: 1 m grid ----
    ax.xaxis.set_minor_locator(MultipleLocator(1.0))
    ax.yaxis.set_minor_locator(MultipleLocator(1.0))

    # ---- Grid ----
    ax.grid(which="major", linewidth=0.6, alpha=0.8)
    ax.grid(which="minor", linewidth=0.3, alpha=0.4)

    # Hide minor tick marks (keep grid only)
    ax.tick_params(which="minor", length=0)

    # Legend (optional)
    # from matplotlib.lines import Line2D

    # handles = []
    # handles.append(Line2D([0], [0], marker="o", linestyle="None", markersize=5, color=drone1_color, label="Drone1 Estimation"))
    # handles.append(Line2D([0], [0], marker="o", linestyle="None", markersize=5, color=drone2_color, label="Drone2 Estimation"))
    # if drone_xy:
    #     handles.append(Line2D([0], [0], marker="^", linestyle="None", markersize=5, color="k", label="Drones"))
    # if avg_xs:
    #     handles.append(Line2D([0], [0], marker="*", linestyle="None", markersize=5, color="k", label="Fused Average"))
    # if bi_xs:
    #     handles.append(Line2D([0], [0], marker="X", linestyle="None", markersize=5, color="k", label="Fused Bearing intersection"))
    # if handles:
    #     ax.legend(handles=handles, loc="upper right", frameon=True)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)

    plt.close(fig)


def build_series_from_frame(drone1_lat,drone1_lon,drone2_lat,drone2_lon,detections1,detections2,fused_detections):
    drone_positions = [
        {"label": "drone1", "lat": drone1_lat, "lon": drone1_lon},
        {"label": "drone2", "lat": drone2_lat, "lon": drone2_lon},
    ]

    targets_d1 = [] # now is fused
    # targets_d1_pitch = []
    # targets_d1_pinhole = []
    # targets_d1_fused = []
    
    targets_d2 = [] # now is fused
    # targets_d2_pitch = []
    # targets_d2_pinhole = []
    # targets_d2_fused = []

    targets_fused_average = []
    targets_fused_bearing_intersection = []
    targets_fused_weighted_average = []
    #targets_fused_weighted_bearing_intersection = []
    measurements_d1 = []
    measurements_d2 = []

    for det in detections1 or []:
        geo = det.get("person_geoposition") if isinstance(det, dict) else None
        if not (geo and isinstance(geo, dict)):
            continue
        lat = geo.get("latitude")
        lon = geo.get("longitude")
        if lat is None or lon is None:
            continue
        targets_d1.append((lat, lon))

        bearing = det.get("bearing_deg", det.get("bearing"))
        dist = det.get("distance_m")
        if bearing is not None or dist is not None:
            measurements_d1.append(
                {
                    "origin_lat": drone1_lat,
                    "origin_lon": drone1_lon,
                    "bearing_deg": bearing,
                    "distance_m": dist,
                }
            )

    for det in detections2 or []:
        geo = det.get("person_geoposition") if isinstance(det, dict) else None
        if not (geo and isinstance(geo, dict)):
            continue
        lat = geo.get("latitude")
        lon = geo.get("longitude")
        if lat is None or lon is None:
            continue
        targets_d2.append((lat, lon))

        bearing = det.get("bearing_deg", det.get("bearing"))
        dist = det.get("distance_m")
        if bearing is not None or dist is not None:
            measurements_d2.append(
                {
                    "origin_lat": drone2_lat,
                    "origin_lon": drone2_lon,
                    "bearing_deg": bearing,
                    "distance_m": dist,
                }
            )

    for det in fused_detections:
        geo_avg = det.get("fused_geoposition_average")
        geo_bi = det.get("fused_geoposition_bearing_intersection")
        geo_wavg = det.get("fused_geoposition_weighted_average")
        #geo_wbi = det.get("fused_geoposition_weighted_bearing_intersection")
        if geo_avg and isinstance(geo_avg, dict):
            lat = geo_avg.get("latitude")
            lon = geo_avg.get("longitude")
            if lat is not None and lon is not None:
                targets_fused_average.append((lat, lon))
        if geo_bi and isinstance(geo_bi, dict):
            lat = geo_bi.get("latitude")
            lon = geo_bi.get("longitude")
            if lat is not None and lon is not None:
                targets_fused_bearing_intersection.append((lat, lon))

        if geo_wavg and isinstance(geo_wavg, dict):
            lat = geo_wavg.get("latitude")
            lon = geo_wavg.get("longitude")
            if lat is not None and lon is not None:
                targets_fused_weighted_average.append((lat, lon))
        # if geo_wbi and isinstance(geo_wbi, dict):
        #     lat = geo_wbi.get("latitude")
        #     lon = geo_wbi.get("longitude")
        #     if lat is not None and lon is not None:
        #         targets_fused_weighted_bearing_intersection.append((lat, lon))

    series_gps = {
        "drone_positions": drone_positions,
        "targets_d1": targets_d1,
        "targets_d2": targets_d2,
        "targets_fused_average": targets_fused_average,
        "targets_fused_bearing_intersection": targets_fused_bearing_intersection,
        "targets_fused_weighted_average": targets_fused_weighted_average,
        #"targets_fused_weighted_bearing_intersection": targets_fused_weighted_bearing_intersection,
        "measurements_d1": measurements_d1,
        "measurements_d2": measurements_d2,
    }
    return {"gps": series_gps}