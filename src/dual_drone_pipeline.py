import cv2
import os
import numpy as np
from pathlib import Path

from detection_pipeline import DetectionPipeline
from detection_fusion import DualDroneFusion, Detection
from camera import Camera
from detection_statistics import DetectionStatistics
from position_estimation import (
    estimate_distance_pitch, estimate_bearing, estimate_distance,
    target_geoposition, distance_from_geoposition
)
from geoconverter import GeoConverter
import viewer
from ground_plane_plot import plot_ground_plane, build_series_from_frame


class DualDroneDetectionPipeline:
    """Pipeline for processing synchronized video streams from two drones."""

    def __init__(
        self,
        model_path: str,
        person_confidence_threshold: float,
        enable_weapon_detection: bool = True,
        weapon_confidence_threshold: float = 0.5,
        sample_majority_threshold: int = 1,
        association_threshold: float = 100.0,
        device=None,
    ):  # distance threshold for cross-drone detection matching (meters)

        # Create two independent detection pipelines (one per drone)
        self.pipeline_drone1 = DetectionPipeline(
            model_path,
            person_confidence_threshold=person_confidence_threshold,
            enable_weapon_detection=enable_weapon_detection,
            weapon_confidence_threshold=weapon_confidence_threshold,
            sample_majority_threshold=sample_majority_threshold,
            device=device,
        )

        self.pipeline_drone2 = DetectionPipeline(
            model_path,
            person_confidence_threshold=person_confidence_threshold,
            enable_weapon_detection=enable_weapon_detection,
            weapon_confidence_threshold=weapon_confidence_threshold,
            sample_majority_threshold=sample_majority_threshold,
            device=device,
        )

        self.fusion = DualDroneFusion(association_threshold_m=association_threshold)

        self.camera_drone1 = Camera()
        self.camera_drone2 = Camera()

        # Keep separate stats so we can compare single-drone vs fusion quality.
        self.stats_drone1 = self.pipeline_drone1.stats
        self.stats_drone2 = self.pipeline_drone2.stats
        self.stats_fused = DetectionStatistics(sample_majority_threshold=sample_majority_threshold)

        # Backward-compatible alias: treat "stats" as the fused stats.
        self.stats = self.stats_fused

        self.save_crops = True
        self.enable_weapon_detection = enable_weapon_detection
        self.person_confidence_threshold = person_confidence_threshold
        self.verbose = True  # set True to print per-frame debug output
        self.show_weapon_confidence = False

        # Optional: write a 2D ground-plane (local XY) plot per sample.
        self.enable_ground_plane_plot = False

        self.ground_plane_plot_filename = "ground_plane.png"

    def print_console_output(
        self,
        best_det1,
        best_det2,
        fused_detections,
        w1_conf,
        w2_conf,
    ):
        """Print console output with detection information.

        Parameters should use the *same* best-detection selection that
        DetectionStatistics uses (closest person by distance) so that
        analyze_log.py can faithfully reproduce the pipeline metrics.
        """
        best_fused_avg = next((d for d in fused_detections if d.get("source") == "fused_average"), None)
        best_fused_bi = next((d for d in fused_detections if d.get("source") == "fused_bearing_intersection"), None)

        d1_h = best_det1.get("distance_pinhole_m") if best_det1 else None
        d1_t = best_det1.get("distance_pitch_m") if best_det1 else None
        d2_h = best_det2.get("distance_pinhole_m") if best_det2 else None
        d2_t = best_det2.get("distance_pitch_m") if best_det2 else None

        geo1 = best_det1.get("person_geoposition") if best_det1 else None
        geo2 = best_det2.get("person_geoposition") if best_det2 else None

        # Average fusion method
        geo_fused_avg = best_fused_avg.get("fused_geoposition_average") if best_fused_avg else None
        d1_fused_avg = best_fused_avg.get("distance_drone1_average_m") if best_fused_avg else None
        d2_fused_avg = best_fused_avg.get("distance_drone2_average_m") if best_fused_avg else None

        # Bearing intersection fusion method
        geo_fused_bi = best_fused_bi.get("fused_geoposition_bearing_intersection") if best_fused_bi else None
        d1_fused_bi = best_fused_bi.get("distance_drone1_bearing_intersection_m") if best_fused_bi else None
        d2_fused_bi = best_fused_bi.get("distance_drone2_bearing_intersection_m") if best_fused_bi else None

        def fmt(val):
            return f"{val:.2f}" if val is not None else "None"

        def fmt_geo(geo):
            if geo and isinstance(geo, dict):
                return f"({geo.get('latitude', 0):.6f},{geo.get('longitude', 0):.6f})"
            return "None"

        print(
            "DISTANCE: "
            f"d1_height-based={fmt(d1_h)}, "
            f"d1_pitch-based={fmt(d1_t)}, "
            f"d2_height-based={fmt(d2_h)}, "
            f"d2_pitch-based={fmt(d2_t)}, "
            f"geo1={fmt_geo(geo1)}, "
            f"geo2={fmt_geo(geo2)}"
        )
        print(
            "FUSED_AVERAGE: "
            f"d1_fused={fmt(d1_fused_avg)}, "
            f"d2_fused={fmt(d2_fused_avg)}, "
            f"geo_fused={fmt_geo(geo_fused_avg)}"
        )
        print(
            "FUSED_BEARING_INTERSECTION: "
            f"d1_fused={fmt(d1_fused_bi)}, "
            f"d2_fused={fmt(d2_fused_bi)}, "
            f"geo_fused={fmt_geo(geo_fused_bi)}"
        )

    def add_fused_geopositions(self, fused_detections, detections1, detections2):
        """Legacy method - geopositions are now added by direct triangulation in fuse_frame_detections."""
        pass

    def process_dual_drone_samples(self, input_dir_drone1, input_dir_drone2, output_base_dir, only_angle=None):
        # Check if directories contain angle subdirectories
        angle_dirs_drone1 = [
            d for d in os.listdir(input_dir_drone1)
            if os.path.isdir(os.path.join(input_dir_drone1, d)) and d.isdigit()
        ]
        angle_dirs_drone2 = [
            d for d in os.listdir(input_dir_drone2)
            if os.path.isdir(os.path.join(input_dir_drone2, d)) and d.isdigit()
        ]

        # Find common angle directories
        common_angles = sorted(set(angle_dirs_drone1) & set(angle_dirs_drone2))

        if common_angles:
            # Optional filter: process only a single requested angle.
            if only_angle is not None:
                chosen = None
                only_angle_str = str(only_angle).strip()
                if only_angle_str in common_angles:
                    chosen = only_angle_str
                else:
                    try:
                        only_angle_int = int(only_angle_str)
                    except Exception:
                        only_angle_int = None

                    if only_angle_int is not None:
                        for a in common_angles:
                            try:
                                if int(a) == only_angle_int:
                                    chosen = a
                                    break
                            except Exception:
                                continue

                if chosen is None:
                    print(
                        f"Requested --angle={only_angle} not found. "
                        f"Common angles available: {', '.join(common_angles)}"
                    )
                    return

                common_angles = [chosen]

            # Process each angle subdirectory separately
            print(f"\n{'='*60}")
            print(f"Found angle subdirectories: {', '.join(common_angles)}")
            print("Processing each angle separately...")
            print(f"{'='*60}\n")

            for angle in common_angles:
                print(f"\n{'='*60}")
                print(f"PROCESSING ANGLE: {angle}°")
                print(f"{'='*60}\n")

                angle_dir_drone1 = os.path.join(input_dir_drone1, angle)
                angle_dir_drone2 = os.path.join(input_dir_drone2, angle)
                angle_output_dir = os.path.join(output_base_dir, f"angle_{angle}")

                # Reset statistics for this angle
                for s in (self.stats_drone1, self.stats_drone2, self.stats_fused):
                    s.reset()
                    s._in_batch_mode = True

                # Process this angle
                self._process_angle_directory(angle_dir_drone1, angle_dir_drone2, angle_output_dir, angle)

                # Print statistics for this angle
                print(f"\n{'='*60}")
                print(f"STATISTICS FOR ANGLE {angle}°")
                print(f"{'='*60}")
                print("\n--- Drone 1 ---")
                self.stats_drone1.print_summary()
                print("\n--- Drone 2 ---")
                self.stats_drone2.print_summary()
                print("\n--- Fused ---")
                self.stats_fused.print_summary()

                # Fusion localization gain (range RMSE improvement) vs single-drone baselines.
                try:
                    base_rmse_d1 = self.stats_drone1.compute_rmse(self.stats_drone1.distance_pairs)
                    base_rmse_d2 = self.stats_drone2.compute_rmse(self.stats_drone2.distance_pairs)
                    fused_rmse_d1 = self.stats_fused.compute_rmse(self.stats_fused.distance_pairs_fused_d1)
                    fused_rmse_d2 = self.stats_fused.compute_rmse(self.stats_fused.distance_pairs_fused_d2)

                    print("\nFUSION LOCALIZATION GAIN:")

                    def _print_gain(label: str, base_rmse: float | None, fused_rmse: float | None):
                        if base_rmse is None or fused_rmse is None or base_rmse <= 0:
                            print(f"   VS {label}: N/A")
                            return
                        gain_abs = base_rmse - fused_rmse
                        gain_pct = (1.0 - (fused_rmse / base_rmse)) * 100.0
                        print(
                            f"   VS {label}: base_rmse={base_rmse:.3f}m, "
                            f"fused_rmse={fused_rmse:.3f}m, gain={gain_abs:.3f}m, gain_pct={gain_pct:.1f}%"
                        )

                    _print_gain("D1", base_rmse_d1, fused_rmse_d1)
                    _print_gain("D2", base_rmse_d2, fused_rmse_d2)

                    # By (distance, height) buckets.
                    combos = sorted(
                        set(getattr(self.stats_drone1, "distance_pairs_by_dist_height", {}).keys())
                        | set(getattr(self.stats_drone2, "distance_pairs_by_dist_height", {}).keys())
                        | set(getattr(self.stats_fused, "distance_pairs_fused_d1_by_dist_height", {}).keys())
                        | set(getattr(self.stats_fused, "distance_pairs_fused_d2_by_dist_height", {}).keys())
                    )

                    if combos:
                        print("\nFUSION GAIN BY (DISTANCE, HEIGHT):")
                        for (dist, height) in combos:
                            print(f"   Distance: {dist}m, Height: {height}m")

                            base_pairs_d1 = getattr(self.stats_drone1, "distance_pairs_by_dist_height", {}).get((dist, height))
                            fused_pairs_d1 = getattr(self.stats_fused, "distance_pairs_fused_d1_by_dist_height", {}).get((dist, height))
                            base_pairs_d2 = getattr(self.stats_drone2, "distance_pairs_by_dist_height", {}).get((dist, height))
                            fused_pairs_d2 = getattr(self.stats_fused, "distance_pairs_fused_d2_by_dist_height", {}).get((dist, height))

                            rmse_base_d1 = self.stats_drone1.compute_rmse(base_pairs_d1) if base_pairs_d1 else None
                            rmse_fused_d1_b = self.stats_fused.compute_rmse(fused_pairs_d1) if fused_pairs_d1 else None
                            rmse_base_d2 = self.stats_drone2.compute_rmse(base_pairs_d2) if base_pairs_d2 else None
                            rmse_fused_d2_b = self.stats_fused.compute_rmse(fused_pairs_d2) if fused_pairs_d2 else None

                            _print_gain("D1", rmse_base_d1, rmse_fused_d1_b)
                            _print_gain("D2", rmse_base_d2, rmse_fused_d2_b)
                except Exception:
                    pass
        else:
            # No angle subdirectories, process directly
            self._process_angle_directory(input_dir_drone1, input_dir_drone2, output_base_dir, None)
            for s in (self.stats_drone1, self.stats_drone2, self.stats_fused):
                s.finalize()

    def _process_angle_directory(self, input_dir_drone1, input_dir_drone2, output_base_dir, angle=None):
        """Process a single angle directory or direct sample directory."""
        # Get sample directories from both drones
        samples_drone1 = [
            d for d in os.listdir(input_dir_drone1)
            if os.path.isdir(os.path.join(input_dir_drone1, d))
        ]
        samples_drone2 = [
            d for d in os.listdir(input_dir_drone2)
            if os.path.isdir(os.path.join(input_dir_drone2, d))
        ]

        # Find matching sample pairs (same sample name in both directories)
        sample_pairs = [s for s in samples_drone1 if s in samples_drone2]

        if len(sample_pairs) == 0:
            print(f"No matching sample pairs found in {input_dir_drone1} and {input_dir_drone2}")
            return

        print(f"Found {len(sample_pairs)} matching sample pairs")

        # Create output structure
        detections_base_dir = os.path.join(output_base_dir, "detections_dual_drone")
        crops_base_dir = os.path.join(output_base_dir, "crops_dual_drone")
        fused_base_dir = os.path.join(output_base_dir, "fused_detections")

        # Process each synchronized pair
        for sample_idx, sample_name in enumerate(sample_pairs, 1):
            print(f"\n{'='*60}")
            print(f"Processing sample {sample_idx}/{len(sample_pairs)}: {sample_name}")
            print(f"{'='*60}")

            sample_path_drone1 = os.path.join(input_dir_drone1, sample_name)
            sample_path_drone2 = os.path.join(input_dir_drone2, sample_name)

            # Determine ground truth from filename convention when possible.
            try:
                sample_meta = self.pipeline_drone1.detector.extract_filename_metadata(sample_name)
            except Exception:
                sample_meta = {}
            if (sample_meta or {}).get("sample_class") in ("real", "falso"):
                sample_ground_truth = (sample_meta.get("sample_class") == "real")
                sample_class = sample_meta.get("sample_class")
            else:
                sample_ground_truth = sample_name.lower().startswith("real")
                sample_class = "real" if sample_ground_truth else "falso"

            print(f"  Sample type: {sample_class.upper()} (has_weapons={sample_ground_truth})")

            # Mark start of new sample
            for s in (self.stats_drone1, self.stats_drone2, self.stats_fused):
                s.start_new_sample(sample_ground_truth, sample_class)

            # Process synchronized sample pair
            self.process_synchronized_sample_pair(
                sample_path_drone1,
                sample_path_drone2,
                output_base_dir,
                sample_name,
                detections_base_dir,
                crops_base_dir,
                fused_base_dir,
            )

        # Finalize statistics for this angle
        for s in (self.stats_drone1, self.stats_drone2, self.stats_fused):
            s.finalize()

    def process_synchronized_sample_pair(
        self,
        sample_dir_drone1,
        sample_dir_drone2,
        output_base,
        sample_name,
        detections_base,
        crops_base,
        fused_base,
    ):
        # Supported image formats
        SUPPORTED_FORMATS = [".jpg", ".jpeg", ".png", ".bmp"]

        def get_image_files(directory):
            exts = {e.lower() for e in SUPPORTED_FORMATS}
            return sorted(
                [
                    os.path.join(directory, f)
                    for f in os.listdir(directory)
                    if os.path.splitext(f)[1].lower() in exts
                ]
            )

        images_drone1 = get_image_files(sample_dir_drone1)
        images_drone2 = get_image_files(sample_dir_drone2)

        print(f"  Drone 1: {len(images_drone1)} frames")
        print(f"  Drone 2: {len(images_drone2)} frames")

        synchronized_pairs = self.synchronize_by_frame_index(images_drone1, images_drone2)

        print(f"  Synchronized: {len(synchronized_pairs)} frame pairs")

        sample_det_d1 = os.path.join(detections_base, f"{sample_name}_drone1")
        sample_det_d2 = os.path.join(detections_base, f"{sample_name}_drone2")
        sample_det_fused = os.path.join(fused_base, sample_name)
        sample_crops_d1 = os.path.join(crops_base, f"{sample_name}_drone1")
        sample_crops_d2 = os.path.join(crops_base, f"{sample_name}_drone2")

        for dir_path in [sample_det_d1, sample_det_d2, sample_det_fused, sample_crops_d1, sample_crops_d2]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

        # Process each synchronized frame pair
        for frame_idx, (frame1_path, frame2_path) in enumerate(synchronized_pairs):
            if frame_idx % 10 == 0:
                print(f"    Processing frame {frame_idx + 1}/{len(synchronized_pairs)}")

            frame_result = self.process_frame_pair(
                frame1_path,
                frame2_path,
                frame_idx,
                sample_det_d1,
                sample_det_d2,
                sample_det_fused,
                sample_crops_d1,
                sample_crops_d2,
                sample_name,
            )

            if self.enable_ground_plane_plot and isinstance(frame_result, dict):
                try:
                    series_bundle = frame_result.get("series", None)
                    if series_bundle and isinstance(series_bundle, dict):
                        gps = series_bundle.get("gps")
                        if gps:
                            frame_meta = self.pipeline_drone1.detector.extract_filename_metadata(frame1_path)
                            real_distance = frame_meta.get("distance_m")

                            out_path = os.path.join(
                                sample_det_fused,
                                f"{sample_name}_frame_{frame_idx:04d}_ground_gps.png",
                            )

                            # ---- Choose fused center for this frame (origin = (0,0)) ----
                            fused_lat = None
                            fused_lon = None

                            # Prefer bearing-intersection fused position if present
                            bi_list = gps.get("targets_fused_bearing_intersection") or []
                            if len(bi_list) > 0:
                                try:
                                    fused_lat = float(bi_list[-1][0])
                                    fused_lon = float(bi_list[-1][1])
                                except Exception:
                                    fused_lat, fused_lon = None, None

                            # Otherwise fall back to fused average
                            if fused_lat is None or fused_lon is None:
                                avg_list = gps.get("targets_fused_average") or []
                                if len(avg_list) > 0:
                                    try:
                                        fused_lat = float(avg_list[-1][0])
                                        fused_lon = float(avg_list[-1][1])
                                    except Exception:
                                        fused_lat, fused_lon = None, None

                            # Final fallback: drone1 position for this frame
                            if fused_lat is None or fused_lon is None:
                                try:
                                    fused_lat = float(frame_result.get("drone1_lat"))
                                    fused_lon = float(frame_result.get("drone1_lon"))
                                except Exception:
                                    fused_lat, fused_lon = None, None

                            if fused_lat is None or fused_lon is None:
                                raise ValueError("No valid center lat/lon available for plotting")

                            tick_half_range = int(real_distance) if real_distance is not None else 10

                            plot_ground_plane(
                                ref_lat=fused_lat,  # origin is fused (or fallback drone1)
                                ref_lon=fused_lon,
                                drone_positions=gps.get("drone_positions"),
                                targets_d1=gps.get("targets_d1"),
                                targets_d2=gps.get("targets_d2"),
                                targets_fused_average=gps.get("targets_fused_average"),
                                targets_fused_bearing_intersection=gps.get("targets_fused_bearing_intersection"),
                                measurements_d1=gps.get("measurements_d1"),
                                measurements_d2=gps.get("measurements_d2"),
                                title=None,
                                ticks=tick_half_range,  # axis [-tick, +tick], 1m tick spacing (per your plot_ground_plane)
                                show_legend=True,
                                show_drone_labels=False,
                                show_monocular_points=True,
                                draw_bearing_rays=True,
                                draw_distance_circles=False,
                                ray_length_m=float(2*real_distance),
                                dpi=300,
                                figsize=(3.35, 3.0),
                                out_path=out_path,
                                show=False,
                            )

                            print(f"[plot_ground_plane] Saved plot to {out_path}")
                except Exception as e:
                    if getattr(self, "verbose", False):
                        print(f"    ⚠ Ground-plane plot failed on frame {frame_idx:04d}: {e}")

        print(f"  ✓ Completed all {len(synchronized_pairs)} frames")

    @staticmethod
    def synchronize_by_frame_index(frames1, frames2):
        """Synchronize frames from two drones by extracting frame indices."""
        def extract_frame_num(path):
            import re
            basename = path.split("/")[-1].split("\\")[-1]
            matches = re.findall(r"_(\d{4})", basename)
            if matches:
                return int(matches[-1])
            matches = re.findall(r"(\d{4})", basename)
            if matches:
                return int(matches[-1])
            return -1

        frames1_dict = {extract_frame_num(f): f for f in frames1}
        frames2_dict = {extract_frame_num(f): f for f in frames2}

        common_indices = sorted(set(frames1_dict.keys()) & set(frames2_dict.keys()))
        return [(frames1_dict[idx], frames2_dict[idx]) for idx in common_indices]

    def process_frame_pair(
        self,
        frame1_path,
        frame2_path,
        frame_idx,
        output_det_d1,
        output_det_d2,
        output_fused,
        output_crops_d1,
        output_crops_d2,
        sample_name,
    ):
        # Load images
        img1 = cv2.imread(frame1_path)
        img2 = cv2.imread(frame2_path)

        if img1 is None or img2 is None:
            print(f"    ⚠ Warning: Failed to load frame {frame_idx}")
            return None

        # Load telemetry data for both cameras from txt files
        self.camera_drone1.load_telemetry_from_video_path(frame1_path)
        self.camera_drone2.load_telemetry_from_video_path(frame2_path)

        # Debug: Print drone positions on first frame
        if frame_idx == 0:
            print("  DEBUG Drone positions:")
            print(
                f"    Drone 1: lat={self.camera_drone1.lat:.6f}, "
                f"lon={self.camera_drone1.lon:.6f}, yaw={self.camera_drone1.yaw_deg:.1f}°"
            )
            print(
                f"    Drone 2: lat={self.camera_drone2.lat:.6f}, "
                f"lon={self.camera_drone2.lon:.6f}, yaw={self.camera_drone2.yaw_deg:.1f}°"
            )

        # Extract height from filename metadata
        frame_meta = self.pipeline_drone1.detector.extract_filename_metadata(frame1_path)
        if frame_meta.get("height_m"):
            self.camera_drone1.height_m = frame_meta["height_m"]
            self.camera_drone2.height_m = frame_meta["height_m"]

        # Get filenames
        filename1 = os.path.basename(frame1_path)
        filename2 = os.path.basename(frame2_path)
        name1, ext1 = os.path.splitext(filename1)
        name2, ext2 = os.path.splitext(filename2)

        # Detect people using current estimation methods
        detections1 = self.detect_people_with_estimation(img1, drone_id=1)
        detections2 = self.detect_people_with_estimation(img2, drone_id=2)

        # Optional: filter known background people before crop/weapon detection and fusion.
        detections1 = self.pipeline_drone1.filter_people_detections(detections1, img1.shape)
        detections2 = self.pipeline_drone2.filter_people_detections(detections2, img2.shape)

        # Verbose output for detections
        if self.verbose or (frame_idx % 10 == 0):
            print(f"      Frame {frame_idx}: D1={len(detections1)} people, D2={len(detections2)} people")

        # Testing-time condition metadata (used for RMSE comparisons only)
        real_distance = frame_meta.get("distance_m")
        cam_height_m = frame_meta.get("height_m")

        # Process crops and weapon detection for both drones
        weapon_results1 = []
        weapon_results2 = []

        if detections1:
            crops1 = self.pipeline_drone1.extract_person_crops(img1, detections1)
            if self.save_crops:
                self.pipeline_drone1.save_bounding_box_crops(img1, detections1, output_crops_d1, name1)
            if self.enable_weapon_detection:
                weapon_results1 = self.pipeline_drone1.detect_weapons_in_crops(crops1)

        if detections2:
            crops2 = self.pipeline_drone2.extract_person_crops(img2, detections2)
            if self.save_crops:
                self.pipeline_drone2.save_bounding_box_crops(img2, detections2, output_crops_d2, name2)
            if self.enable_weapon_detection:
                weapon_results2 = self.pipeline_drone2.detect_weapons_in_crops(crops2)

        # --- FUSION: Compute both methods for each detection pair ---
        fused_detections = []
        from position_estimation import (
            fuse_average_target_positions_from_distance_bearing,
            triangulate_target_by_bearing_intersection,
        )

        if detections1 and detections2:
            det1 = min(detections1, key=lambda d: d.get("distance_m", float("inf")))
            det2 = min(detections2, key=lambda d: d.get("distance_m", float("inf")))

            avg_latlon = fuse_average_target_positions_from_distance_bearing(
                self.camera_drone1, det1["distance_m"], det1["bearing_deg"],
                self.camera_drone2, det2["distance_m"], det2["bearing_deg"],
            )

            tri_latlon = triangulate_target_by_bearing_intersection(
                self.camera_drone1, det1["bearing_deg"],
                self.camera_drone2, det2["bearing_deg"],
            )

            fused_conf = self.fusion.fuse_confidence(det1["person_confidence"], det2["person_confidence"])

            # Average method result
            avg_dist1 = distance_from_geoposition(self.camera_drone1, avg_latlon[0], avg_latlon[1])
            avg_dist2 = distance_from_geoposition(self.camera_drone2, avg_latlon[0], avg_latlon[1])
            fused_detections.append(
                {
                    "bbox": None,
                    "person_confidence": fused_conf,
                    "has_weapon": det1.get("has_weapon", False) or det2.get("has_weapon", False),
                    "weapon_confidence": max(det1.get("weapon_confidence", 0.0), det2.get("weapon_confidence", 0.0)),
                    "fused_geoposition_average": {"latitude": avg_latlon[0], "longitude": avg_latlon[1]},
                    "distance_drone1_average_m": avg_dist1,
                    "distance_drone2_average_m": avg_dist2,
                    "source": "fused_average",
                }
            )

            # Bearing intersection result (if valid)
            if tri_latlon is not None:
                tri_dist1 = distance_from_geoposition(self.camera_drone1, tri_latlon[0], tri_latlon[1])
                tri_dist2 = distance_from_geoposition(self.camera_drone2, tri_latlon[0], tri_latlon[1])
                fused_detections.append(
                    {
                        "bbox": None,
                        "person_confidence": fused_conf,
                        "has_weapon": det1.get("has_weapon", False) or det2.get("has_weapon", False),
                        "weapon_confidence": max(det1.get("weapon_confidence", 0.0), det2.get("weapon_confidence", 0.0)),
                        "fused_geoposition_bearing_intersection": {"latitude": tri_latlon[0], "longitude": tri_latlon[1]},
                        "distance_drone1_bearing_intersection_m": tri_dist1,
                        "distance_drone2_bearing_intersection_m": tri_dist2,
                        "source": "fused_bearing_intersection",
                    }
                )

        # Ground-plane points for this frame
        try:
            frame_series = build_series_from_frame(
                drone1_lat=float(self.camera_drone1.lat),
                drone1_lon=float(self.camera_drone1.lon),
                drone2_lat=float(self.camera_drone2.lat),
                drone2_lon=float(self.camera_drone2.lon),
                detections1=detections1,
                detections2=detections2,
                fused_detections=[
                    d for d in (fused_detections or [])
                    if isinstance(d, dict) and d.get("source") in ("fused_average", "fused_bearing_intersection")
                ],
            )
        except Exception:
            frame_series = None

        # Verbose output for fusion (fixed source labels)
        if self.verbose or (frame_idx % 10 == 0):
            fused_count = len(
                [d for d in fused_detections if d.get("source") in ("fused_average", "fused_bearing_intersection")]
            )
            print(f"      Frame {frame_idx}: Fused {fused_count} detections from both drones, total {len(fused_detections)} detections")

        # Draw detections on individual frames (with fused stats in each bbox)
        img1_annotated = viewer.draw_boxes_fusion(
            img1, detections1, weapon_results1, "Drone 1",
            fused_detections, 1, self.show_weapon_confidence
        )
        img2_annotated = viewer.draw_boxes_fusion(
            img2, detections2, weapon_results2, "Drone 2",
            fused_detections, 2, self.show_weapon_confidence
        )

        # Save individual drone detections
        cv2.imwrite(os.path.join(output_det_d1, f"{name1}_detected{ext1}"), img1_annotated)
        cv2.imwrite(os.path.join(output_det_d2, f"{name2}_detected{ext2}"), img2_annotated)

        # Create visualization of fused detections (side-by-side view)
        fused_vis = viewer.fused_visualization(img1_annotated, img2_annotated)

        fused_filename = f"{sample_name}_frame_{frame_idx:04d}_fused.jpg"
        fused_path = os.path.join(output_fused, fused_filename)
        if fused_vis is not None:
            cv2.imwrite(fused_path, fused_vis)

        # Update statistics
        try:
            sample_meta = self.pipeline_drone1.detector.extract_filename_metadata(sample_name)
        except Exception:
            sample_meta = {}
        if (sample_meta or {}).get("sample_class") in ("real", "falso"):
            has_weapons_gt = (sample_meta.get("sample_class") == "real")
        else:
            has_weapons_gt = sample_name.lower().startswith("real")

        weapons_fused = sum(1 for d in fused_detections if d.get("has_weapon", False))

        sample_class = (
            sample_meta.get("sample_class")
            if (sample_meta or {}).get("sample_class") in ("real", "falso")
            else ("real" if has_weapons_gt else "falso")
        )

        # Select CLOSEST detection from each drone (smallest distance)
        best_det1 = None
        if detections1:
            det1_with_dist = [d for d in detections1 if d.get("distance_m") is not None]
            if det1_with_dist:
                best_det1 = sorted(det1_with_dist, key=lambda d: d.get("distance_m", float("inf")))[0]
            else:
                best_det1 = max(detections1, key=lambda d: d.get("person_confidence", d.get("confidence", 0)))

        best_det2 = None
        if detections2:
            det2_with_dist = [d for d in detections2 if d.get("distance_m") is not None]
            if det2_with_dist:
                best_det2 = sorted(det2_with_dist, key=lambda d: d.get("distance_m", float("inf")))[0]
            else:
                best_det2 = max(detections2, key=lambda d: d.get("person_confidence", d.get("confidence", 0)))

        # Distance evaluation pairs (method-specific) - ONLY BEST DETECTION
        distances1 = []
        distances2 = []
        if best_det1 and best_det1.get("distance_m") is not None:
            distances1.append(best_det1["distance_m"])
        if best_det2 and best_det2.get("distance_m") is not None:
            distances2.append(best_det2["distance_m"])

        pairs1_p = []
        pairs1_pitch = []
        pairs2_p = []
        pairs2_pitch = []
        pairs1_primary = []
        pairs2_primary = []

        if real_distance is not None:
            if best_det1:
                if best_det1.get("distance_m") is not None:
                    pairs1_primary.append((best_det1["distance_m"], real_distance))
                if best_det1.get("distance_pinhole_m") is not None:
                    pairs1_p.append((best_det1["distance_pinhole_m"], real_distance))
                if best_det1.get("distance_pitch_m") is not None:
                    pairs1_pitch.append((best_det1["distance_pitch_m"], real_distance))
            if best_det2:
                if best_det2.get("distance_m") is not None:
                    pairs2_primary.append((best_det2["distance_m"], real_distance))
                if best_det2.get("distance_pinhole_m") is not None:
                    pairs2_p.append((best_det2["distance_pinhole_m"], real_distance))
                if best_det2.get("distance_pitch_m") is not None:
                    pairs2_pitch.append((best_det2["distance_pitch_m"], real_distance))

        # Weapon detection for best detection only
        weapons_detected_d1_best = 0
        people_with_weapons_d1 = 0
        w1_conf = 0.0
        if best_det1 and weapon_results1:
            best_idx1 = detections1.index(best_det1)
            if best_idx1 < len(weapon_results1):
                wr1 = weapon_results1[best_idx1]
                if wr1.get("has_weapons", False):
                    weapons_detected_d1_best = len(wr1.get("weapon_detections", []))
                    people_with_weapons_d1 = 1
                wdets1 = wr1.get("weapon_detections", [])
                if wdets1:
                    w1_conf = max(w.get("weapon_confidence", w.get("confidence", 0.0)) for w in wdets1)

        weapons_detected_d2_best = 0
        people_with_weapons_d2 = 0
        w2_conf = 0.0
        if best_det2 and weapon_results2:
            best_idx2 = detections2.index(best_det2)
            if best_idx2 < len(weapon_results2):
                wr2 = weapon_results2[best_idx2]
                if wr2.get("has_weapons", False):
                    weapons_detected_d2_best = len(wr2.get("weapon_detections", []))
                    people_with_weapons_d2 = 1
                wdets2 = wr2.get("weapon_detections", [])
                if wdets2:
                    w2_conf = max(w.get("weapon_confidence", w.get("confidence", 0.0)) for w in wdets2)

        if getattr(self, "verbose", False):
            self.print_console_output(best_det1, best_det2, fused_detections, w1_conf, w2_conf)

        # Individual drone stats (using only best detection)
        self.stats_drone1.add_image_results(
            1 if best_det1 else 0,
            weapons_detected_d1_best,
            people_with_weapons_d1,
            has_weapons_gt,
            distances=distances1,
            distance_pairs=pairs1_primary,
            real_distance=real_distance,
            cam_height_m=cam_height_m,
            sample_class=sample_class,
            distance_pairs_pinhole=pairs1_p,
            distance_pairs_pitch=pairs1_pitch,
        )

        self.stats_drone2.add_image_results(
            1 if best_det2 else 0,
            weapons_detected_d2_best,
            people_with_weapons_d2,
            has_weapons_gt,
            distances=distances2,
            distance_pairs=pairs2_primary,
            real_distance=real_distance,
            cam_height_m=cam_height_m,
            sample_class=sample_class,
            distance_pairs_pinhole=pairs2_p,
            distance_pairs_pitch=pairs2_pitch,
        )

        # Fused-distance RMSE pairs: per-drone distance to fused geoposition (fixed source labels + keys)
        fused_pairs = []
        fused_pairs_d1 = []
        fused_pairs_d2 = []

        if real_distance is not None:
            for d in fused_detections:
                if d.get("source") == "fused_average":
                    geo = d.get("fused_geoposition_average")
                elif d.get("source") == "fused_bearing_intersection":
                    geo = d.get("fused_geoposition_bearing_intersection")
                else:
                    continue

                if not (geo and isinstance(geo, dict)):
                    continue
                lat = geo.get("latitude")
                lon = geo.get("longitude")
                if lat is None or lon is None:
                    continue

                try:
                    dist1_f = distance_from_geoposition(self.camera_drone1, float(lat), float(lon))
                    dist2_f = distance_from_geoposition(self.camera_drone2, float(lat), float(lon))
                except Exception:
                    continue

                fused_pairs.append((dist1_f, real_distance))
                fused_pairs.append((dist2_f, real_distance))
                fused_pairs_d1.append((dist1_f, real_distance))
                fused_pairs_d2.append((dist2_f, real_distance))

        # Fused detection stats (weapon metrics + fusion-quality RMSE)
        self.stats_fused.add_image_results(
            len(fused_detections),
            weapons_fused,
            weapons_fused,
            has_weapons_gt,
            distances=None,
            distance_pairs=None,
            real_distance=real_distance,
            cam_height_m=cam_height_m,
            sample_class=sample_class,
            distance_pairs_pinhole=None,
            distance_pairs_pitch=None,
            distance_pairs_fused=fused_pairs,
            distance_pairs_fused_d1=fused_pairs_d1,
            distance_pairs_fused_d2=fused_pairs_d2,
        )

        return {
            "drone1_lat": self.camera_drone1.lat,
            "drone1_lon": self.camera_drone1.lon,
            "drone2_lat": self.camera_drone2.lat,
            "drone2_lon": self.camera_drone2.lon,
            "series": frame_series or {},
        }

    def detect_people_with_estimation(self, image, drone_id):
        """Detect people and estimate their distance/bearing/geoposition for fusion."""
        detector = self.pipeline_drone1.detector if drone_id == 1 else self.pipeline_drone2.detector
        camera = self.camera_drone1 if drone_id == 1 else self.camera_drone2

        infer_kwargs = dict(
            imgsz=640,
            iou=0.6,
            conf=self.person_confidence_threshold,
            classes=[0],
            verbose=False,
        )
        if getattr(detector, "device", None) is not None:
            infer_kwargs["device"] = getattr(detector, "device")

        results = detector.model(image, **infer_kwargs)

        detections_info = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())

                if class_id != 0 or confidence < self.person_confidence_threshold:
                    continue

                person_height_px = float(y2 - y1)
                y_bottom = float(y2)
                x_center = float((x1 + x2) / 2)

                distance_pinhole_m = estimate_distance(camera, person_height_px) if person_height_px > 0 else None

                distance_pitch_m = None
                if camera.height_m > 0:
                    try:
                        distance_pitch_m = estimate_distance_pitch(camera, y_bottom)
                    except (ZeroDivisionError, ValueError, TypeError):
                        distance_pitch_m = None

                bearing_deg = estimate_bearing(camera, x_center)

                distance_m = None
                if distance_pitch_m is not None:
                    distance_m = float(distance_pitch_m)
                elif distance_pinhole_m is not None:
                    distance_m = float(distance_pinhole_m)

                person_lat_pinhole, person_lon_pinhole = None, None
                if distance_pinhole_m is not None and bearing_deg is not None and camera.lat != 0 and camera.lon != 0:
                    person_lat_pinhole, person_lon_pinhole = target_geoposition(
                        camera, float(distance_pinhole_m), float(bearing_deg)
                    )

                person_lat_pitch, person_lon_pitch = None, None
                if distance_pitch_m is not None and bearing_deg is not None and camera.lat != 0 and camera.lon != 0:
                    person_lat_pitch, person_lon_pitch = target_geoposition(
                        camera, float(distance_pitch_m), float(bearing_deg)
                    )

                person_geoposition = None
                if person_lat_pitch is not None and person_lon_pitch is not None:
                    person_geoposition = {"latitude": person_lat_pitch, "longitude": person_lon_pitch}
                elif person_lat_pinhole is not None and person_lon_pinhole is not None:
                    person_geoposition = {"latitude": person_lat_pinhole, "longitude": person_lon_pinhole}

                detections_info.append(
                    {
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "confidence": confidence,
                        "person_confidence": confidence,
                        "distance_m": distance_m,
                        "distance_pinhole_m": distance_pinhole_m,
                        "distance_pitch_m": distance_pitch_m,
                        "bearing_deg": bearing_deg,
                        "cam_height_m": camera.height_m,
                        "cam_pitch_deg": camera.pitch_deg,
                        "cam_yaw_deg": camera.yaw_deg,
                        "cam_roll_deg": camera.roll_deg,
                        "cam_lat": camera.lat,
                        "cam_lon": camera.lon,
                        "person_geoposition": person_geoposition,
                        "geo_position_pinhole": (
                            {"latitude": person_lat_pinhole, "longitude": person_lon_pinhole}
                            if person_lat_pinhole is not None
                            else None
                        ),
                        "geo_position_pitch": (
                            {"latitude": person_lat_pitch, "longitude": person_lon_pitch}
                            if person_lat_pitch is not None
                            else None
                        ),
                    }
                )

        return detections_info

    def find_geoposition_by_bbox(self, detections, bbox):
        """Find geoposition information for a detection by its bounding box."""
        if not detections or not bbox:
            return None

        for det in detections:
            if det.get("bbox") == list(bbox) or det.get("bbox") == bbox:
                return det.get("geo_position_pinhole")

        return None